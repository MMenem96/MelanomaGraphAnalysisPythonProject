"""
Fine-tune a pretrained CNN on the BCC vs BKL training set, then extract
features from the fine-tuned model.

Why: when frozen, the CNN's features are generic-ImageNet. Fine-tuning the
top blocks on dermoscopy data adapts those features to BCC-vs-BKL
discrimination — the standard "transfer learning" approach used in
Adebiyi 2024, Soundarya 2025, Huang 2020.

Outputs:
    paper_pipeline/output/features/cnn_features_<backbone>_ft.pkl
        Same shape and column convention as the frozen pickle. The merge
        step reuses it via `merge_cnn_features.py --backbone <backbone>_ft`.

Implementation notes:
    * Unfreeze top N layers of the base only (last 2 conv blocks).
    * Class-balanced training set (879 BCC + 879 BKL).
    * Light on-the-fly augmentation (h/v flip + brightness jitter).
    * Train with Adam, low LR (1e-4), early stopping on a validation split
      taken from the training set only.
    * Test set is touched ONLY for the final feature extraction (the
      classification head is discarded — we keep the GAP layer output).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from paper_pipeline.pipeline.augmentation_transform import (
    TAG_HFLIP, TAG_ORIGINAL, TAG_ROT_P15, TAG_VFLIP, TEST_TAGS, TRAIN_TAGS, variants,
)
from paper_pipeline.pipeline.dataset import (
    LABEL_BCC, LABEL_BKL, Sample, load_canonical_samples,
)

LOG = logging.getLogger("cnn_finetune")

INPUT_SIZE = 224
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 30
DEFAULT_UNFREEZE_LAST_N = 30
DEFAULT_VAL_FRACTION = 0.1


def _load_and_resize(path: Path) -> np.ndarray | None:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != INPUT_SIZE or rgb.shape[1] != INPUT_SIZE:
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return rgb


def get_backbone(name: str):
    name = name.lower()
    if name == "resnet50":
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        m = ResNet50(weights="imagenet", include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
        return m, preprocess_input, 2048
    if name == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
        m = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
        return m, preprocess_input, 1280
    raise ValueError(f"Unknown backbone {name}")


def stratified_split_by_id(samples: list[Sample], test_size: float, random_state: int):
    from sklearn.model_selection import train_test_split
    ids = [s.image_id for s in samples]
    labels = [s.label for s in samples]
    train_ids, test_ids, _, _ = train_test_split(
        ids, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    id_to_sample = {s.image_id: s for s in samples}
    return [id_to_sample[i] for i in train_ids], [id_to_sample[i] for i in test_ids]


def build_balanced_training_arrays(train_samples: list[Sample], random_state: int,
                                    use_rotation: bool = False):
    """
    Return X_train (N, 224, 224, 3 uint8 RGB), y_train (N,), source_id list.

    Default: BCC = 411 originals + 411 h_flip + 411 v_flip → 1,233 rows pre-balance.
              Deterministically drop 354 flipped rows to land at 879 BCC = BKL count.
              BKL = 879 originals (no augmentation).

    With `use_rotation=True`:
        BCC = orig + h_flip + v_flip + rot+15 → 4 variants × 411 = 1,644 rows.
        BKL = orig + rot+15 → 2 variants × 879 = 1,758 rows.
        Ratio 1 : 1.07 — no downsample needed.
    """
    import random as _rnd
    bcc = [s for s in train_samples if s.label == LABEL_BCC]
    bkl = [s for s in train_samples if s.label == LABEL_BKL]

    if use_rotation:
        bcc_tags = tuple(list(TRAIN_TAGS) + [TAG_ROT_P15])
        bkl_tags = (TAG_ORIGINAL, TAG_ROT_P15)
        LOG.info("  Rotation augmentation: BCC %s; BKL %s", list(bcc_tags), list(bkl_tags))
    else:
        bcc_tags = TRAIN_TAGS
        bkl_tags = (TAG_ORIGINAL,)
        LOG.info("  BCC train: %d (will augment 3×); BKL train: %d (no aug)", len(bcc), len(bkl))

    bcc_rows = []
    for s in bcc:
        img = _load_and_resize(s.path)
        if img is None: continue
        for tag, variant in variants(img, bcc_tags):
            bcc_rows.append((s.image_id, tag, variant))
    bkl_rows = []
    for s in bkl:
        img = _load_and_resize(s.path)
        if img is None: continue
        for tag, variant in variants(img, bkl_tags):
            bkl_rows.append((s.image_id, tag, variant))

    if use_rotation:
        # Skip downsampling — ratio is already ~1:1.07.
        LOG.info("  use_rotation=True: skipping downsample. BCC=%d, BKL=%d",
                 len(bcc_rows), len(bkl_rows))
        all_rows = bcc_rows + bkl_rows
    else:
        # Downsample BCC: preserve originals, drop random flipped
        target = len(bkl_rows)
        bcc_orig = [r for r in bcc_rows if r[1] == TAG_ORIGINAL]
        bcc_flip = [r for r in bcc_rows if r[1] != TAG_ORIGINAL]
        need_from_flip = max(0, target - len(bcc_orig))
        rng = _rnd.Random(random_state)
        rng.shuffle(bcc_flip)
        bcc_kept = bcc_orig + bcc_flip[:need_from_flip]
        LOG.info("  Balanced BCC: %d originals + %d flipped = %d (target %d)",
                 len(bcc_orig), need_from_flip, len(bcc_kept), target)
        all_rows = bcc_kept + bkl_rows
    n_bcc = len(bcc_rows) if use_rotation else len(bcc_kept)
    X = np.stack([r[2] for r in all_rows]).astype(np.uint8)
    y = np.array([1] * n_bcc + [0] * len(bkl_rows), dtype=np.int32)
    ids = [r[0] for r in all_rows]
    tags = [r[1] for r in all_rows]
    LOG.info("  Final training: X=%s, y class balance=%s",
             X.shape, dict(Counter(y.tolist())))
    return X, y, ids, tags


def build_full_extraction_arrays(train_samples, test_samples, use_rotation: bool = False):
    """All (image, aug_tag) combinations for FEATURE EXTRACTION after fine-tuning.
    Used to merge into the existing handcrafted pickles. Same coverage as
    `cnn_features.py` (no balancing here; merge step handles it)."""
    if use_rotation:
        bcc_tags = tuple(list(TRAIN_TAGS) + [TAG_ROT_P15])
        bkl_tags = (TAG_ORIGINAL, TAG_ROT_P15)
    else:
        bcc_tags = TRAIN_TAGS
        bkl_tags = (TAG_ORIGINAL,)
    rows = []
    for s in train_samples:
        img = _load_and_resize(s.path)
        if img is None: continue
        tags = bcc_tags if s.label == LABEL_BCC else bkl_tags
        for tag, variant in variants(img, tags):
            rows.append((s.image_id, s.label, tag, variant))
    for s in test_samples:
        img = _load_and_resize(s.path)
        if img is None: continue
        for tag, variant in variants(img, TEST_TAGS):
            rows.append((s.image_id, s.label, tag, variant))
    return rows


def build_finetune_model(base, preprocess_fn, unfreeze_last_n: int):
    from tensorflow.keras import layers, Model
    import tensorflow as tf
    # Freeze everything first
    for layer in base.layers:
        layer.trainable = False
    # Unfreeze top N layers (excluding BatchNorm)
    for layer in base.layers[-unfreeze_last_n:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    n_trainable = sum(np.prod(v.shape) for v in base.trainable_variables)
    LOG.info("  Unfroze top %d layers → %d trainable params in base",
             unfreeze_last_n, int(n_trainable))

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(base.input, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="resnet50",
                        choices=["resnet50", "efficientnetb0"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--unfreeze-last-n", type=int, default=DEFAULT_UNFREEZE_LAST_N)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--use-rotation", action="store_true",
                        help="Add rot+15 augmentation to BCC and BKL training.")
    parser.add_argument("--limit", type=int, default=None, help="Debug")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "paper_pipeline" / "output" / "features")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")

    import tensorflow as tf
    tf.random.set_seed(args.random_state)
    np.random.seed(args.random_state)

    LOG.info("Loading canonical samples…")
    samples = load_canonical_samples()
    if args.limit:
        bcc = [s for s in samples if s.label == LABEL_BCC][:args.limit]
        bkl = [s for s in samples if s.label == LABEL_BKL][:args.limit]
        samples = bcc + bkl
        LOG.info("  --limit %d → %d samples", args.limit, len(samples))

    train_samples, test_samples = stratified_split_by_id(
        samples, args.test_size, args.random_state
    )
    LOG.info("Split: train=%d test=%d", len(train_samples), len(test_samples))

    LOG.info("Loading base CNN: %s", args.backbone)
    base, preprocess_fn, feat_dim = get_backbone(args.backbone)

    LOG.info("Building balanced training arrays for fine-tuning…")
    Xtr, ytr, ids_tr, tags_tr = build_balanced_training_arrays(
        train_samples, args.random_state, use_rotation=args.use_rotation
    )

    LOG.info("Building fine-tune model (unfreezing top %d layers)…", args.unfreeze_last_n)
    model = build_finetune_model(base, preprocess_fn, args.unfreeze_last_n)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    LOG.info("Model summary: %d trainable params, %d total params",
             sum(np.prod(v.shape) for v in model.trainable_variables),
             model.count_params())

    # Train, using a STRATIFIED validation split FROM THE TRAINING DATA ONLY.
    # CRITICAL: Keras `validation_split=0.1` takes the LAST 10% before shuffle.
    # With our array order (BCC first, BKL after), that would give a 100%-BKL
    # validation set. We use sklearn's stratified split instead.
    from sklearn.model_selection import train_test_split as _tts
    LOG.info("Preprocessing input arrays for backbone…")
    Xtr_pp = preprocess_fn(Xtr.astype(np.float32))
    X_tr, X_va, y_tr, y_va = _tts(
        Xtr_pp, ytr,
        test_size=DEFAULT_VAL_FRACTION,
        stratify=ytr,
        random_state=args.random_state,
        shuffle=True,
    )
    LOG.info("Stratified val split: train=%s (%s), val=%s (%s)",
             X_tr.shape, dict(Counter(y_tr.tolist())),
             X_va.shape, dict(Counter(y_va.tolist())))

    LOG.info("Fine-tuning for up to %d epochs, batch=%d, lr=%.0e, patience=%d…",
             args.epochs, args.batch_size, args.learning_rate, args.patience)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=args.patience,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", mode="max", factor=0.5, patience=3,
            verbose=1, min_lr=1e-6,
        ),
    ]
    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_va, y_va),
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    LOG.info("Training done in %.1fs", time.time() - t0)
    LOG.info("Best val accuracy: %.4f (epoch %d)",
             max(history.history.get("val_accuracy", [0])),
             1 + int(np.argmax(history.history.get("val_accuracy", [0]))))
    LOG.info("Final train AUC at best epoch: %.4f",
             history.history.get("auc", [0])[int(np.argmax(history.history.get("val_accuracy", [0])))])

    # Build the feature extractor: take output of the GAP layer
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("gap").output,
    )
    LOG.info("Feature extractor output dim: %d", feature_extractor.output_shape[-1])

    # Now extract features for ALL (image, aug_tag) combinations
    LOG.info("Building full extraction task list…")
    tasks = build_full_extraction_arrays(train_samples, test_samples, use_rotation=args.use_rotation)
    LOG.info("  total tasks: %d  (tag counts %s)",
             len(tasks), dict(Counter(t[2] for t in tasks)))

    # Batched feature extraction
    feats = []
    started = time.time()
    bs = max(8, args.batch_size)
    for batch_start in range(0, len(tasks), bs):
        batch = tasks[batch_start: batch_start + bs]
        imgs = np.stack([b[3] for b in batch]).astype(np.float32)
        pp = preprocess_fn(imgs)
        f = feature_extractor.predict(pp, verbose=0)
        feats.append(f)
        if (batch_start + bs) % (bs * 8) == 0:
            elapsed = time.time() - started
            done = min(batch_start + bs, len(tasks))
            rate = done / elapsed
            eta = (len(tasks) - done) / rate if rate > 0 else 0
            LOG.info("  %d/%d  (%.1f img/s) ETA %.1fs", done, len(tasks), rate, eta)
    feats_arr = np.vstack(feats)
    LOG.info("Feature matrix: %s", feats_arr.shape)

    # Build DataFrame
    df = pd.DataFrame({
        "source_id": [t[0] for t in tasks],
        "label": [t[1] for t in tasks],
        "aug_tag": [t[2] for t in tasks],
    })
    cnn_df = pd.DataFrame(feats_arr, columns=[f"cnn_{i}" for i in range(feats_arr.shape[1])])
    df = pd.concat([df, cnn_df], axis=1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix_part = "_rot_ft" if args.use_rotation else "_ft"
    out = args.out_dir / f"cnn_features_{args.backbone}{suffix_part}.pkl"
    df.to_pickle(out)
    LOG.info("Saved %s  (%d rows × %d cols)", out, len(df), df.shape[1])
    LOG.info("File size: %.1f MB", out.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
