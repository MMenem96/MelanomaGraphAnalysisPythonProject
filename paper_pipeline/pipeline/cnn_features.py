"""
Extract pretrained-CNN features for every canonical image (and its training
augmentations) and save to a single pickle.

CNN features do NOT depend on the MKT λ configuration, so we extract them
ONCE and merge them into each of the four (handcrafted+MKT) pickles
afterwards via `merge_cnn_features.py`.

Backbone: ResNet-50, ImageNet weights, frozen, global-average-pooled →
2048-dim feature vector per image. (Other options EfficientNet-B0 / 1280-dim
and DenseNet-121 / 1024-dim are available — see `--backbone`.)

Pipeline:
    1. Load canonical samples (514 BCC + 1099 BKL = 1613).
    2. Stratified 80/20 split on image IDs (random_state=42).
    3. For each TRAIN BCC sample: extract CNN features for orig, h_flip,
       v_flip (3×). Optionally downsample to BKL count.
    4. For each TRAIN BKL sample: extract CNN features for orig only (1×).
    5. For each TEST sample: extract CNN features for orig only.
    6. Save one pickle with all rows.

Output columns: source_id, aug_tag, label, cnn_0, cnn_1, …, cnn_2047.
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

# Quiet TF startup noise
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from paper_pipeline.pipeline.augmentation_transform import (
    TAG_HFLIP, TAG_ORIGINAL, TAG_VFLIP, TEST_TAGS, TRAIN_TAGS, variants,
)
from paper_pipeline.pipeline.dataset import (
    LABEL_BCC, LABEL_BKL, Sample, load_canonical_samples,
)

LOG = logging.getLogger("cnn_features")

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
INPUT_SIZE = 224
BATCH_SIZE = 16


def get_backbone(name: str):
    """Returns (model, preprocess_fn, feature_dim)."""
    name = name.lower()
    if name == "resnet50":
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        m = ResNet50(weights="imagenet", include_top=False, pooling="avg",
                     input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
        return m, preprocess_input, 2048
    if name == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
        m = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg",
                           input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
        return m, preprocess_input, 1280
    if name == "densenet121":
        from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
        m = DenseNet121(weights="imagenet", include_top=False, pooling="avg",
                        input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
        return m, preprocess_input, 1024
    raise ValueError(f"Unknown backbone {name}")


def _load_and_resize(path: Path) -> np.ndarray | None:
    """Load a segmented PNG, convert BGR→RGB, resize to 224×224."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != INPUT_SIZE or rgb.shape[1] != INPUT_SIZE:
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return rgb


def stratified_split_by_id(samples: list[Sample], test_size: float, random_state: int):
    ids = [s.image_id for s in samples]
    labels = [s.label for s in samples]
    train_ids, test_ids, _, _ = train_test_split(
        ids, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    id_to_sample = {s.image_id: s for s in samples}
    return [id_to_sample[i] for i in train_ids], [id_to_sample[i] for i in test_ids]


def build_image_task_list(
    train_samples: list[Sample],
    test_samples: list[Sample],
    balance_strategy: str,
    no_augment: bool,
) -> list[tuple[str, int, str, np.ndarray]]:
    """Return list of (source_id, label, aug_tag, image_rgb).
    Per-label augmentation policy."""
    from paper_pipeline.pipeline.augmentation_transform import TAG_ROT_P15
    full_aug = (TAG_ORIGINAL,) if no_augment else TRAIN_TAGS
    if balance_strategy == "both":
        bcc_tags = full_aug; bkl_tags = full_aug
    elif balance_strategy == "rot_both":
        bcc_tags = tuple(list(full_aug) + [TAG_ROT_P15])
        bkl_tags = (TAG_ORIGINAL, TAG_ROT_P15)
    else:  # bcc_only or bcc_only_balanced
        bcc_tags = full_aug; bkl_tags = (TAG_ORIGINAL,)

    tasks = []
    skipped = 0
    for sample in train_samples:
        tags = bcc_tags if sample.label == LABEL_BCC else bkl_tags
        img = _load_and_resize(sample.path)
        if img is None:
            skipped += 1; continue
        for tag, variant in variants(img, tags):
            tasks.append((sample.image_id, sample.label, tag, variant))
    for sample in test_samples:
        img = _load_and_resize(sample.path)
        if img is None:
            skipped += 1; continue
        for tag, variant in variants(img, TEST_TAGS):
            tasks.append((sample.image_id, sample.label, tag, variant))
    if skipped:
        LOG.warning("Skipped %d unreadable images", skipped)
    return tasks


def extract_in_batches(
    tasks: list[tuple[str, int, str, np.ndarray]],
    model,
    preprocess_fn,
    batch_size: int,
) -> np.ndarray:
    """Return (n_tasks, feature_dim) array of CNN features."""
    n = len(tasks)
    feats: list[np.ndarray] = []
    started = time.time()
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        imgs = np.stack([tasks[i][3] for i in range(batch_start, batch_end)], axis=0).astype(np.float32)
        preprocessed = preprocess_fn(imgs.copy())
        batch_feats = model.predict(preprocessed, verbose=0)
        feats.append(batch_feats)
        if batch_end % (batch_size * 8) == 0 or batch_end == n:
            elapsed = time.time() - started
            rate = batch_end / elapsed
            eta = (n - batch_end) / rate if rate > 0 else 0
            LOG.info("  %d / %d   (%.1f img/s)   ETA %.1fs", batch_end, n, rate, eta)
    return np.vstack(feats)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone", default="resnet50",
        choices=["resnet50", "efficientnetb0", "densenet121"],
    )
    parser.add_argument(
        "--balance-strategy", default="bcc_only_balanced",
        choices=["both", "bcc_only", "bcc_only_balanced", "rot_both"],
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Debug")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "paper_pipeline" / "output" / "features",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    LOG.info("Loading canonical samples…")
    samples = load_canonical_samples()
    if args.limit:
        bcc = [s for s in samples if s.label == LABEL_BCC][:args.limit]
        bkl = [s for s in samples if s.label == LABEL_BKL][:args.limit]
        samples = bcc + bkl
        LOG.info("  --limit %d  → %d samples", args.limit, len(samples))

    train_samples, test_samples = stratified_split_by_id(
        samples, args.test_size, args.random_state
    )
    LOG.info("Split: train=%d, test=%d", len(train_samples), len(test_samples))
    LOG.info("Train labels: %s", dict(Counter(s.label for s in train_samples)))
    LOG.info("Test  labels: %s", dict(Counter(s.label for s in test_samples)))

    LOG.info("Building task list (loading images + applying augmentations)…")
    tasks = build_image_task_list(
        train_samples, test_samples, args.balance_strategy, args.no_augment
    )
    LOG.info("Total tasks: %d", len(tasks))
    LOG.info("Task tag counts: %s", dict(Counter(t[2] for t in tasks)))

    LOG.info("Loading backbone %s…", args.backbone)
    model, preprocess_fn, dim = get_backbone(args.backbone)
    LOG.info("  feature dim = %d", dim)

    LOG.info("Extracting CNN features in batches of %d…", args.batch_size)
    feats = extract_in_batches(tasks, model, preprocess_fn, args.batch_size)
    LOG.info("Features shape: %s", feats.shape)

    # Build DataFrame
    LOG.info("Building DataFrame…")
    base = pd.DataFrame({
        "source_id": [t[0] for t in tasks],
        "label": [t[1] for t in tasks],
        "aug_tag": [t[2] for t in tasks],
    })
    cnn_cols = pd.DataFrame(feats, columns=[f"cnn_{i}" for i in range(feats.shape[1])])
    df = pd.concat([base, cnn_cols], axis=1)
    LOG.info("DataFrame shape: %s", df.shape)

    # We deliberately DO NOT apply balanced-downsampling here.
    # The balancing already happened in the handcrafted-feature pipeline
    # (`feature_extraction.py`), which is the authoritative source of which
    # (source_id, aug_tag) rows are in the training set. The `merge_cnn_features`
    # step performs an inner join with the handcrafted pickle, which selects
    # exactly the right CNN rows. Doing the random sample twice (once here,
    # once there) would produce different orderings and cause merge mismatches.
    LOG.info("Not applying balancing here — merge step will inner-join with "
             "the authoritative handcrafted set.")

    # Save
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"cnn_features_{args.backbone}.pkl"
    df.to_pickle(out)
    LOG.info("Saved %s  (%d rows × %d cols)", out, len(df), df.shape[1])
    LOG.info("File size: %.1f MB", out.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
