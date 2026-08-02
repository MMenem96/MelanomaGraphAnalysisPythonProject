"""
7-class (Track B) feature extraction — lesion-grouped split, class-equalising
augmentation.

Sibling of `feature_extraction.py`, which stays untouched so the published
binary protocol keeps reproducing identical numbers. All heavy lifting
(preprocessing, mask derivation, per-λ feature extraction) is imported from
that module; what differs here is only:

  1. **Split unit.** HAM10000 has 10,015 images but 7,470 unique lesions, so the
     split groups on `lesion_id` (`StratifiedGroupKFold`), not `image_id`.
     `--split-unit image` reproduces the leaky image-level split for comparison —
     the gap between the two arms is itself a reportable result.
  2. **Per-class augmentation.** `--balance equalize` raises every class to the
     largest class using the composite flip×rotation tag pool;
     `--balance capped4` caps at the 4 flip variants; `--balance none` extracts
     each training image once.

Augmentation is train-only and post-split in every mode. Test images are never
augmented.

Outputs:
    paper_pipeline/output/features7/features_train_<lambda>_<suffix>.pkl
    paper_pipeline/output/features7/features_test_<lambda>_<suffix>.pkl
    paper_pipeline/output/manifests/feature_extraction7_<timestamp>.json
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_pipeline.pipeline import qa
from paper_pipeline.pipeline.augmentation_transform import (
    TAG_ORIGINAL,
    TEST_TAGS,
    build_tag_pool,
)
from paper_pipeline.pipeline.dataset import (
    CLASSES7,
    Sample,
    load_multiclass7_samples,
)
from paper_pipeline.pipeline.feature_extraction import (
    _LambdaScopedExtractor,
    _process_sample_for_all_lambdas,
)
from paper_pipeline.pipeline.mkt_lambda import LAMBDA_CONFIGS

from src.conventional_features import ConventionalFeatureExtractor

LOG = logging.getLogger("feature_extraction7")

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_LAMBDA = "odd_harmonic"   # the configuration the binary paper reports as best
CAPPED_TARGET = 3000


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def grouped_split(
    samples: list[Sample], test_size: float, random_state: int
) -> tuple[list[Sample], list[Sample]]:
    """Stratified split that keeps every `lesion_id` wholly on one side.

    `StratifiedGroupKFold` with n_splits = round(1/test_size) gives a stratified,
    group-respecting partition; we take the first fold as the test set.
    """
    y = np.array([s.label for s in samples])
    groups = np.array([s.lesion_id for s in samples])
    n_splits = max(2, round(1.0 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(np.zeros(len(samples)), y, groups))
    return [samples[i] for i in train_idx], [samples[i] for i in test_idx]


def image_level_split(
    samples: list[Sample], test_size: float, random_state: int
) -> tuple[list[Sample], list[Sample]]:
    """Leaky arm: stratified on label, ignores `lesion_id` (matches the binary paper)."""
    idx = np.arange(len(samples))
    y = [s.label for s in samples]
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=random_state
    )
    return [samples[i] for i in train_idx], [samples[i] for i in test_idx]


def report_split(train: list[Sample], test: list[Sample]) -> dict:
    tr_g = {s.lesion_id for s in train}
    te_g = {s.lesion_id for s in test}
    shared = tr_g & te_g
    LOG.info("Split: train=%d imgs / %d lesions   test=%d imgs / %d lesions",
             len(train), len(tr_g), len(test), len(te_g))
    LOG.info("Lesions on BOTH sides: %d  %s",
             len(shared), "(clean)" if not shared else "(LEAKY — image-level arm)")
    tr_c, te_c = Counter(s.dx for s in train), Counter(s.dx for s in test)
    LOG.info("%-7s %8s %8s", "class", "train", "test")
    for dx in CLASSES7:
        LOG.info("%-7s %8d %8d", dx, tr_c[dx], te_c[dx])
    return {
        "train_images": len(train), "test_images": len(test),
        "train_lesions": len(tr_g), "test_lesions": len(te_g),
        "lesions_in_both": len(shared),
        "train_per_class": dict(tr_c), "test_per_class": dict(te_c),
    }


# ---------------------------------------------------------------------------
# Per-class augmentation policy
# ---------------------------------------------------------------------------

def build_augmentation_policy(
    train: list[Sample], mode: str, rotation_step: int
) -> dict[str, tuple[str, ...]]:
    """Return {dx: (tag, ...)} — how many variants each class contributes.

    equalize : every class is raised to the largest class's image count, using
               flips first and then flip×rotation composites.
    capped4  : min(4, ceil(CAPPED_TARGET / n_c)) — flips only, no rotations.
    none     : one row per training image.
    """
    per_class = Counter(s.dx for s in train)
    pool = build_tag_pool(rotation_step)

    if mode == "none":
        return {dx: (TAG_ORIGINAL,) for dx in per_class}

    if mode == "capped4":
        policy = {}
        for dx, n in per_class.items():
            m = min(4, max(1, math.ceil(CAPPED_TARGET / n)))
            policy[dx] = tuple(pool[:m])
        return policy

    if mode == "equalize":
        target = max(per_class.values())
        policy = {}
        for dx, n in per_class.items():
            m = max(1, math.ceil(target / n))
            if m > len(pool):
                LOG.warning(
                    "%s needs %d variants but the pool holds %d; capping "
                    "(class will stay under target).", dx, m, len(pool)
                )
                m = len(pool)
            policy[dx] = tuple(pool[:m])
        return policy

    raise ValueError(f"Unknown --balance {mode!r}")


def log_policy(train: list[Sample], policy: dict[str, tuple[str, ...]]) -> dict:
    per_class = Counter(s.dx for s in train)
    LOG.info("--- augmentation policy ---")
    LOG.info("%-7s %8s %6s %10s", "class", "images", "mult", "rows")
    out = {}
    for dx in CLASSES7:
        if dx not in per_class:
            continue
        m = len(policy[dx])
        LOG.info("%-7s %8d %6d %10d", dx, per_class[dx], m, per_class[dx] * m)
        out[dx] = {"images": per_class[dx], "multiplier": m, "rows": per_class[dx] * m}
    LOG.info("total train rows ≈ %d", sum(v["rows"] for v in out.values()))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "paper_pipeline" / "output" / "features7")
    ap.add_argument("--lambdas", type=str, default=DEFAULT_LAMBDA,
                    help=f"Comma-separated λ configs. Known: {list(LAMBDA_CONFIGS)}")
    ap.add_argument("--split-unit", choices=["lesion", "image"], default="lesion",
                    help="'lesion' = grouped (correct); 'image' = leaky comparison arm.")
    ap.add_argument("--balance", choices=["equalize", "capped4", "none"], default="equalize")
    ap.add_argument("--rotation-step", type=int, default=15)
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    ap.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    ap.add_argument("--limit-per-class", type=int, default=None,
                    help="First N images per class (smoke test).")
    ap.add_argument("--suffix", default=None,
                    help="Output filename suffix (default: <split-unit>_<balance>).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")

    chosen = [s.strip() for s in args.lambdas.split(",") if s.strip()]
    for name in chosen:
        if name not in LAMBDA_CONFIGS:
            LOG.error("Unknown λ %r. Known: %s", name, list(LAMBDA_CONFIGS))
            return 2
    suffix = args.suffix or f"{args.split_unit}_{args.balance}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    LOG.info("Loading 7-class samples …")
    samples = load_multiclass7_samples()
    LOG.info("  total = %d images, %d lesions, classes = %s",
             len(samples), len({s.lesion_id for s in samples}), dict(Counter(s.dx for s in samples)))

    if args.limit_per_class:
        by_dx: dict[str, list[Sample]] = defaultdict(list)
        for s in samples:
            by_dx[s.dx].append(s)
        samples = [s for dx in CLASSES7 for s in by_dx[dx][: args.limit_per_class]]
        LOG.info("  --limit-per-class %d → %d samples", args.limit_per_class, len(samples))

    splitter = grouped_split if args.split_unit == "lesion" else image_level_split
    train_samples, test_samples = splitter(samples, args.test_size, args.random_state)
    split_info = report_split(train_samples, test_samples)

    policy = build_augmentation_policy(train_samples, args.balance, args.rotation_step)
    policy_info = log_policy(train_samples, policy)

    LOG.info("Building per-λ extractors for: %s", chosen)
    base_extractor = ConventionalFeatureExtractor()
    lambda_extractors = {n: _LambdaScopedExtractor(LAMBDA_CONFIGS[n]) for n in chosen}

    def run_pass(pass_samples: list[Sample], tags_for, label: str) -> dict[str, list[dict]]:
        rows_per_lambda: dict[str, list[dict]] = {n: [] for n in chosen}
        started = time.time()
        for i, sample in enumerate(pass_samples):
            got = _process_sample_for_all_lambdas(
                sample, tags_for(sample), base_extractor, lambda_extractors, save_qa=False
            )
            for n in chosen:
                for row in got[n]:
                    row["lesion_id"] = sample.lesion_id
                    row["dx"] = sample.dx
                rows_per_lambda[n].extend(got[n])
            if (i + 1) % 100 == 0:
                elapsed = time.time() - started
                rate = (i + 1) / elapsed
                eta = (len(pass_samples) - i - 1) / rate / 60.0
                LOG.info("  %s  %d/%d  (%.2f img/s, ETA %.1f min)",
                         label, i + 1, len(pass_samples), rate, eta)
        LOG.info("%s pass done in %.1f min", label, (time.time() - started) / 60.0)
        return rows_per_lambda

    LOG.info("--- TRAIN PASS ---")
    train_rows = run_pass(train_samples, lambda s: policy[s.dx], "TRAIN")
    LOG.info("--- TEST PASS (1 tag per image, never augmented) ---")
    test_rows = run_pass(test_samples, lambda s: TEST_TAGS, "TEST")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "timestamp": timestamp,
        "split_unit": args.split_unit,
        "balance": args.balance,
        "rotation_step": args.rotation_step,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "lambdas": chosen,
        "classes": CLASSES7,
        "split": split_info,
        "augmentation_policy": policy_info,
        "files": {},
    }

    for n in chosen:
        tr = pd.DataFrame(train_rows[n])
        te = pd.DataFrame(test_rows[n])
        if tr.empty or te.empty:
            LOG.error("λ=%s produced empty frames (train=%d, test=%d)", n, len(tr), len(te))
            return 1

        # Leakage re-check, at BOTH levels.
        img_overlap = set(tr["source_id"]) & set(te["source_id"])
        les_overlap = set(tr["lesion_id"]) & set(te["lesion_id"])
        if img_overlap:
            raise RuntimeError(f"IMAGE LEAKAGE in λ={n}: {len(img_overlap)} shared source_ids")
        if les_overlap and args.split_unit == "lesion":
            raise RuntimeError(f"LESION LEAKAGE in λ={n}: {len(les_overlap)} shared lesion_ids")
        LOG.info("λ=%s  train %s  test %s  | lesions shared: %d",
                 n, tr.shape, te.shape, len(les_overlap))
        LOG.info("     train rows/class: %s", dict(Counter(tr["dx"])))
        LOG.info("     test  rows/class: %s", dict(Counter(te["dx"])))

        tr_path = args.out_dir / f"features_train_{n}_{suffix}.pkl"
        te_path = args.out_dir / f"features_test_{n}_{suffix}.pkl"
        tr.to_pickle(tr_path)
        te.to_pickle(te_path)
        manifest["files"][n] = {"train": str(tr_path), "test": str(te_path),
                                "train_rows": len(tr), "test_rows": len(te),
                                "n_columns": int(tr.shape[1])}
        LOG.info("Wrote %s and %s", tr_path.name, te_path.name)

    qa.write_manifest(manifest, "feature_extraction7")
    LOG.info("Done. Next: python paper_pipeline/pipeline/train_eval7.py --suffix %s", suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
