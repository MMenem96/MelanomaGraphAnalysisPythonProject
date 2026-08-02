"""
Multi-λ feature extraction with the split-FIRST, augment-AFTER discipline.

Single pass over the 1,613 canonical images. For each (image, augmentation)
pair we:
  1. preprocess once (hair removal + Telea inpainting + Gaussian blur),
  2. derive a lesion mask once,
  3. extract geometric + colour + texture features once,
  4. extract MKT features four times (one per λ configuration).

We then write four train pickles and four test pickles, one pair per λ,
each carrying:
    source_id, aug_tag, label, <feature columns…>

QA visualisations (hair removal before/after, h/v flip examples) are saved
to paper_pipeline/output/qa/ for the first 5 training samples of each class.

Outputs:
    paper_pipeline/output/features/features_train_<lambda>.pkl
    paper_pipeline/output/features/features_test_<lambda>.pkl
    paper_pipeline/output/manifests/feature_extraction_<timestamp>.json
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_pipeline.pipeline import qa
from paper_pipeline.pipeline.augmentation_transform import (
    TAG_HFLIP,
    TAG_ORIGINAL,
    TAG_VFLIP,
    TEST_TAGS,
    TRAIN_TAGS,
    variants,
)
from paper_pipeline.pipeline.dataset import LABEL_BCC, LABEL_BKL, Sample, load_canonical_samples
from paper_pipeline.pipeline.mkt_lambda import LAMBDA_CONFIGS, all_lambda_names
from paper_pipeline.pipeline.preprocessing import get_processor

from src.conventional_features import ConventionalFeatureExtractor

LOG = logging.getLogger("feature_extraction")

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
MASK_THRESHOLD = 10
N_MKT = 64
P_MKT = 0.5

# Save QA samples for the first N images of each class.
QA_SAMPLES_PER_CLASS = 5


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 3 and bgr.shape[2] == 3:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr


def _generate_lesion_mask(image_rgb: np.ndarray, threshold: int = MASK_THRESHOLD) -> np.ndarray:
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
        return image_rgb[:, :, 3] > threshold
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb
    mask = gray < (255 - threshold)
    from skimage.morphology import closing, disk, opening, remove_small_objects
    mask = opening(mask, disk(2))
    mask = closing(mask, disk(3))
    mask = remove_small_objects(mask, min_size=100)
    return mask.astype(bool)


def _flatten_features(features: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in features.items():
        if isinstance(value, (list, np.ndarray)):
            arr = np.asarray(value).ravel()
            for i, v in enumerate(arr):
                flat[f"{key}_{i}"] = float(v)
        elif isinstance(value, (int, float, np.floating, np.integer)):
            flat[key] = float(value)
    return flat


class _LambdaScopedExtractor(ConventionalFeatureExtractor):
    """A ConventionalFeatureExtractor whose `_compute_lambda` is overridden
    by an externally supplied function."""

    def __init__(self, lambda_fn):
        super().__init__()
        self._lambda_fn = lambda_fn

    def _compute_lambda(self, N):  # type: ignore[override]
        return self._lambda_fn(N)


def _preprocess_rgb(image_rgb: np.ndarray, save_qa_id: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (preprocessed_rgb, hair_mask). Optionally save QA images."""
    proc = get_processor()
    grayscale = proc.convert_to_grayscale(image_rgb)
    combined_hair_mask, _bh, _th = proc.apply_combined_hair_detection(grayscale)
    inpainted = proc.apply_inpainting(image_rgb, combined_hair_mask)
    smoothed = proc.apply_gaussian_blur(inpainted)
    if save_qa_id is not None:
        out_dir = PROJECT_ROOT / "paper_pipeline" / "output" / "qa" / "hair_removal"
        out_dir.mkdir(parents=True, exist_ok=True)
        qa._save_rgb(out_dir / f"{save_qa_id}_01_raw.png", image_rgb)
        qa._save_rgb(out_dir / f"{save_qa_id}_02_hair_mask.png", combined_hair_mask)
        qa._save_rgb(out_dir / f"{save_qa_id}_03_inpainted.png", inpainted)
    return smoothed, combined_hair_mask


def _extract_multi_lambda(
    image_rgb: np.ndarray,
    base_extractor: ConventionalFeatureExtractor,
    lambda_extractors: dict[str, _LambdaScopedExtractor],
) -> dict[str, dict[str, float]]:
    """For one image variant, return {lambda_name: feature_dict}.

    Geometric / colour / texture features are computed once and reused for
    every λ; only the MKT features differ between λ configurations.
    """
    mask = _generate_lesion_mask(image_rgb)
    if mask.sum() < 100:
        return {name: {} for name in lambda_extractors}

    try:
        geometric = base_extractor.extract_geometric_features(mask)
        texture = base_extractor.extract_texture_features(image_rgb, mask)
        color = base_extractor.extract_color_features(image_rgb, mask)
    except Exception as e:
        LOG.warning("Non-MKT feature failure (%s); skipping image.", e)
        return {name: {} for name in lambda_extractors}

    base = {}
    for d in (geometric, texture, color):
        base.update(d)

    out: dict[str, dict[str, float]] = {}
    for name, extractor in lambda_extractors.items():
        try:
            mkt = extractor.extract_mdfkt_features(image_rgb, mask, N=N_MKT, p=P_MKT)
        except Exception as e:
            LOG.warning("MKT extraction failed for λ=%s: %s", name, e)
            mkt = {}
        combined = {**base, **mkt}
        out[name] = _flatten_features(combined)
    return out


def _process_sample_for_all_lambdas(
    sample: Sample,
    aug_tags: tuple[str, ...],
    base_extractor: ConventionalFeatureExtractor,
    lambda_extractors: dict[str, _LambdaScopedExtractor],
    save_qa: bool,
) -> dict[str, list[dict]]:
    """Return {lambda_name: [row, row, ...]} for one source image and its
    augmentations.
    """
    bgr = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
    if bgr is None:
        LOG.warning("cv2.imread returned None for %s; skipping.", sample.path)
        return {name: [] for name in lambda_extractors}
    image_rgb = _bgr_to_rgb(bgr)

    aug_qa_dir = PROJECT_ROOT / "paper_pipeline" / "output" / "qa" / "augmentation"

    rows_per_lambda: dict[str, list[dict]] = {name: [] for name in lambda_extractors}
    aug_images = dict(variants(image_rgb, aug_tags))

    if save_qa and set(aug_images) >= {TAG_ORIGINAL, TAG_HFLIP, TAG_VFLIP}:
        aug_qa_dir.mkdir(parents=True, exist_ok=True)
        qa._save_rgb(aug_qa_dir / f"{sample.image_id}_orig.png", aug_images[TAG_ORIGINAL])
        qa._save_rgb(aug_qa_dir / f"{sample.image_id}_h_flip.png", aug_images[TAG_HFLIP])
        qa._save_rgb(aug_qa_dir / f"{sample.image_id}_v_flip.png", aug_images[TAG_VFLIP])

    for tag, variant in aug_images.items():
        # Save hair-removal QA only for the original of the first 5 of each class.
        qa_id = f"{sample.image_id}_{tag}" if (save_qa and tag == TAG_ORIGINAL) else None
        processed, _ = _preprocess_rgb(variant, qa_id)
        feats_by_lambda = _extract_multi_lambda(processed, base_extractor, lambda_extractors)
        for name, feats in feats_by_lambda.items():
            if not feats:
                continue
            rows_per_lambda[name].append({
                "source_id": sample.image_id,
                "aug_tag": tag,
                "label": sample.label,
                **feats,
            })
    return rows_per_lambda


def stratified_split_by_id(samples: list[Sample], test_size: float, random_state: int):
    ids = [s.image_id for s in samples]
    labels = [s.label for s in samples]
    train_ids, test_ids, _, _ = train_test_split(
        ids, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    id_to_sample = {s.image_id: s for s in samples}
    train = [id_to_sample[i] for i in train_ids]
    test = [id_to_sample[i] for i in test_ids]
    return train, test


def verify_no_leakage(train_rows: list[dict], test_rows: list[dict], lambda_name: str) -> None:
    train_ids = {r["source_id"] for r in train_rows}
    test_ids = {r["source_id"] for r in test_rows}
    overlap = train_ids & test_ids
    if overlap:
        raise RuntimeError(
            f"LEAKAGE in λ={lambda_name}: {len(overlap)} source_ids in both splits. "
            f"First 5: {sorted(overlap)[:5]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "paper_pipeline" / "output" / "features")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--balance-strategy",
        default="bcc_only",
        choices=["both", "bcc_only", "bcc_only_balanced", "rot_both"],
        help=(
            "How to apply augmentation: "
            "'both' = 3x augment both classes (preserves imbalance), "
            "'bcc_only' = 3x augment BCC only, BKL kept as-is (slight BCC majority), "
            "'bcc_only_balanced' = 3x augment BCC then deterministic downsample to BKL count = perfect balance, "
            "'rot_both' = current bcc_only_balanced PLUS a single rotation (+15deg) applied to both classes."
        ),
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="First N samples per class (debug).")
    parser.add_argument("--lambdas", type=str, default=",".join(all_lambda_names()),
                        help="Comma-separated lambda configs to extract.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    chosen_lambdas = [s.strip() for s in args.lambdas.split(",") if s.strip()]
    for name in chosen_lambdas:
        if name not in LAMBDA_CONFIGS:
            LOG.error("Unknown lambda %r. Known: %s", name, list(LAMBDA_CONFIGS))
            return 2

    LOG.info("Loading canonical samples …")
    samples = load_canonical_samples()
    LOG.info("  total = %d (BCC=%d, BKL=%d)",
             len(samples),
             sum(1 for s in samples if s.label == LABEL_BCC),
             sum(1 for s in samples if s.label == LABEL_BKL))

    if args.limit:
        bcc = [s for s in samples if s.label == LABEL_BCC][: args.limit]
        bkl = [s for s in samples if s.label == LABEL_BKL][: args.limit]
        samples = bcc + bkl
        LOG.info("  --limit %d  → using %d samples", args.limit, len(samples))

    train_samples, test_samples = stratified_split_by_id(
        samples, args.test_size, args.random_state
    )
    LOG.info("Split: train=%d, test=%d  (test_size=%.2f, random_state=%d)",
             len(train_samples), len(test_samples), args.test_size, args.random_state)
    LOG.info("Train balance: %s", dict(Counter(s.label for s in train_samples)))
    LOG.info("Test balance : %s", dict(Counter(s.label for s in test_samples)))

    LOG.info("Building per-λ feature extractors for: %s", chosen_lambdas)
    base_extractor = ConventionalFeatureExtractor()
    lambda_extractors = {
        name: _LambdaScopedExtractor(LAMBDA_CONFIGS[name]) for name in chosen_lambdas
    }

    # Per-label augmentation policy.
    from paper_pipeline.pipeline.augmentation_transform import TAG_ROT_P15
    full_aug = TRAIN_TAGS if not args.no_augment else (TAG_ORIGINAL,)
    if args.balance_strategy == "both":
        bcc_tags = full_aug
        bkl_tags = full_aug
    elif args.balance_strategy in ("bcc_only", "bcc_only_balanced"):
        bcc_tags = full_aug
        bkl_tags = (TAG_ORIGINAL,)
    elif args.balance_strategy == "rot_both":
        # BCC gets {orig, h_flip, v_flip, rot+15} = 4 variants.
        # BKL gets {orig, rot+15} = 2 variants.
        # Resulting ratio ~411*4 : 879*2 = 1644 : 1758 ≈ 1 : 1.07 (near-balanced naturally).
        bcc_tags = tuple(list(full_aug) + [TAG_ROT_P15])
        bkl_tags = (TAG_ORIGINAL, TAG_ROT_P15)
    else:
        raise ValueError(f"Unknown --balance-strategy {args.balance_strategy}")
    LOG.info("Augmentation policy: BCC=%s, BKL=%s, balance=%s",
             list(bcc_tags), list(bkl_tags), args.balance_strategy)

    # === Process training set ===
    n_bcc_train = sum(1 for s in train_samples if s.label == LABEL_BCC)
    n_bkl_train = sum(1 for s in train_samples if s.label == LABEL_BKL)
    LOG.info("--- TRAIN PASS (BCC×%d augs, BKL×%d augs) ---",
             len(bcc_tags), len(bkl_tags))
    LOG.info("    BCC train samples = %d  → expected pre-balance rows = %d",
             n_bcc_train, n_bcc_train * len(bcc_tags))
    LOG.info("    BKL train samples = %d  → expected pre-balance rows = %d",
             n_bkl_train, n_bkl_train * len(bkl_tags))
    bcc_qa_saved = 0
    bkl_qa_saved = 0
    train_rows_per_lambda: dict[str, list[dict]] = {n: [] for n in chosen_lambdas}
    started = time.time()
    for i, sample in enumerate(train_samples):
        tags_here = bcc_tags if sample.label == LABEL_BCC else bkl_tags
        save_qa = (
            (sample.label == LABEL_BCC and bcc_qa_saved < QA_SAMPLES_PER_CLASS)
            or (sample.label == LABEL_BKL and bkl_qa_saved < QA_SAMPLES_PER_CLASS)
        )
        rows_per_lambda = _process_sample_for_all_lambdas(
            sample, tags_here, base_extractor, lambda_extractors, save_qa=save_qa
        )
        for name in chosen_lambdas:
            train_rows_per_lambda[name].extend(rows_per_lambda[name])
        if save_qa:
            if sample.label == LABEL_BCC:
                bcc_qa_saved += 1
            else:
                bkl_qa_saved += 1
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - started)
            LOG.info("  TRAIN  %d/%d  (%.2f img/s)", i + 1, len(train_samples), rate)

    LOG.info("Train pass done in %.1fs", time.time() - started)
    for n, rows in train_rows_per_lambda.items():
        from collections import Counter as _C
        cnt = _C(r["label"] for r in rows)
        LOG.info("  λ=%s  train rows = %d  (BCC=%d, BKL=%d)",
                 n, len(rows), cnt.get(LABEL_BCC, 0), cnt.get(LABEL_BKL, 0))

    # Balance BCC to BKL count if requested (random_state-controlled, preserve originals)
    if args.balance_strategy == "bcc_only_balanced":
        import random as _rnd
        for n in chosen_lambdas:
            rows = train_rows_per_lambda[n]
            bcc_rows = [r for r in rows if r["label"] == LABEL_BCC]
            bkl_rows = [r for r in rows if r["label"] == LABEL_BKL]
            target_bcc = len(bkl_rows)
            if len(bcc_rows) > target_bcc:
                originals = [r for r in bcc_rows if r["aug_tag"] == TAG_ORIGINAL]
                flipped = [r for r in bcc_rows if r["aug_tag"] != TAG_ORIGINAL]
                need_from_flipped = max(0, target_bcc - len(originals))
                rng = _rnd.Random(args.random_state)
                rng.shuffle(flipped)
                kept_flipped = flipped[:need_from_flipped]
                new_bcc = originals + kept_flipped
                LOG.info("  λ=%s  balanced BCC: kept %d originals + %d flipped = %d (target %d)",
                         n, len(originals), len(kept_flipped), len(new_bcc), target_bcc)
                train_rows_per_lambda[n] = bkl_rows + new_bcc
            else:
                LOG.info("  λ=%s  BCC rows (%d) <= BKL rows (%d); no downsample needed",
                         n, len(bcc_rows), len(bkl_rows))

    # === Process test set ===
    LOG.info("--- TEST PASS (%d images × 1 tag) ---", len(test_samples))
    test_rows_per_lambda: dict[str, list[dict]] = {n: [] for n in chosen_lambdas}
    started = time.time()
    for i, sample in enumerate(test_samples):
        rows_per_lambda = _process_sample_for_all_lambdas(
            sample, TEST_TAGS, base_extractor, lambda_extractors, save_qa=False
        )
        for name in chosen_lambdas:
            test_rows_per_lambda[name].extend(rows_per_lambda[name])
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - started)
            LOG.info("  TEST   %d/%d  (%.2f img/s)", i + 1, len(test_samples), rate)
    LOG.info("Test pass done in %.1fs", time.time() - started)
    for n, rows in test_rows_per_lambda.items():
        LOG.info("  λ=%s  test rows = %d", n, len(rows))

    # === Verify and save ===
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "test_size": args.test_size,
        "random_state": args.random_state,
        "augment": not args.no_augment,
        "balance_strategy": args.balance_strategy,
        "bcc_train_tags": list(bcc_tags),
        "bkl_train_tags": list(bkl_tags),
        "test_tags": list(TEST_TAGS),
        "n_train_samples": len(train_samples),
        "n_test_samples": len(test_samples),
        "lambdas": chosen_lambdas,
        "outputs": {},
    }

    for name in chosen_lambdas:
        verify_no_leakage(train_rows_per_lambda[name], test_rows_per_lambda[name], name)
        train_df = pd.DataFrame(train_rows_per_lambda[name])
        test_df = pd.DataFrame(test_rows_per_lambda[name])
        train_path = args.out_dir / f"features_train_{name}.pkl"
        test_path = args.out_dir / f"features_test_{name}.pkl"
        train_df.to_pickle(train_path)
        test_df.to_pickle(test_path)
        n_cols = train_df.shape[1]
        LOG.info("Saved  %s  (%d rows × %d cols)", train_path.name, len(train_df), n_cols)
        LOG.info("Saved  %s  (%d rows × %d cols)", test_path.name, len(test_df), n_cols)
        manifest["outputs"][name] = {
            "train_pickle": str(train_path),
            "test_pickle": str(test_path),
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
            "n_columns": n_cols,
        }

    manifest_path = qa.write_manifest(manifest, "feature_extraction")
    LOG.info("Wrote manifest: %s", manifest_path)
    LOG.info("Next step: python paper_pipeline/pipeline/train_eval.py --lambdas %s",
             ",".join(chosen_lambdas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
