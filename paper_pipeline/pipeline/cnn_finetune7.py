"""
Fine-tune a CNN backbone on the 7-class training split, then re-extract features.

Our ResNet-50 features come from a frozen ImageNet backbone that has never seen
a dermoscopic image, and the 7-class results show it: the CNN half added only
+0.03 balanced accuracy. Every published result in the 90%+ range fine-tunes the
backbone on lesion data. This module closes that gap.

Discipline:
  * The train/test split is NOT re-derived. Training source_ids are read from
    the existing feature pickles, so the fine-tuned backbone provably never sees
    a test image — the same guarantee `cnn_features7.py` gives for extraction.
  * Fine-tuning uses one row per *source image* (not per augmented variant) with
    class weights, so a class with 58 augmented copies per image cannot dominate
    the gradient purely by repetition.
  * Feature re-extraction then covers every (source_id, aug_tag) row, exactly as
    the frozen version did, so the downstream pipeline is unchanged.

CPU-only note: no GPU is available on this machine, so the defaults favour a
small backbone (EfficientNetB0) at reduced resolution. Sonuç et al. (Front. Med.
2026) report only +0.76 accuracy from 48x48 -> 224x224, so resolution is the
cheapest thing to trade away.

Usage:
    python paper_pipeline/pipeline/cnn_finetune7.py --suffix lesion_equalize
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_pipeline.pipeline.augmentation_transform import TAG_ORIGINAL, apply_tag
from paper_pipeline.pipeline.dataset import CLASSES7, SEGMENTED7_DIR

LOG = logging.getLogger("cnn_finetune7")

BOOKKEEPING = {"source_id", "aug_tag", "label", "lesion_id", "dx"}


def get_backbone(name: str, input_size: int):
    if name == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import (
            EfficientNetB0, preprocess_input)
        base = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg",
                              input_shape=(input_size, input_size, 3))
    elif name == "resnet50":
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        base = ResNet50(weights="imagenet", include_top=False, pooling="avg",
                        input_shape=(input_size, input_size, 3))
    else:
        raise ValueError(f"Unknown backbone {name!r}")
    return base, preprocess_input


def load_image(source_id: str, dx: str, aug_tag: str, size: int) -> np.ndarray | None:
    bgr = cv2.imread(str(SEGMENTED7_DIR / dx / f"{source_id}.png"), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if aug_tag != TAG_ORIGINAL:
        rgb = apply_tag(rgb, aug_tag)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def load_batch(records, size: int) -> tuple[np.ndarray, list[int]]:
    imgs, keep = [], []
    for i, r in enumerate(records):
        img = load_image(r["source_id"], r["dx"], r.get("aug_tag", TAG_ORIGINAL), size)
        if img is None:
            continue
        imgs.append(img)
        keep.append(i)
    if not imgs:
        return np.zeros((0, size, size, 3), dtype=np.float32), []
    return np.stack(imgs).astype(np.float32), keep


def build_model(base, n_classes: int, unfreeze_last_n: int, lr: float,
                augment: bool = True):
    """Fine-tuning head with on-GPU augmentation.

    Augmentation is applied as Keras layers rather than by pre-materialising the
    37,667 augmented rows: those would need ~23 GB of RAM at 224px on a 16 GB
    machine. Random flips/rotations per epoch give the backbone *more* variety
    than the fixed tag pool while keeping memory at one copy of the originals.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    base.trainable = True
    if unfreeze_last_n >= len(base.layers):
        LOG.info("unfreezing the ENTIRE backbone")
    else:
        for layer in base.layers[:-unfreeze_last_n]:
            layer.trainable = False
    # BatchNorm stays frozen even when everything else trains: its running
    # statistics come from ImageNet and small-batch updates destabilise them.
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    inp = layers.Input(shape=base.input_shape[1:])
    x = inp
    if augment:
        x = layers.RandomFlip("horizontal_and_vertical")(x)
        x = layers.RandomRotation(0.25)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomContrast(0.1)(x)
    x = base(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", name="head")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    trainable = sum(1 for l in base.layers if l.trainable)
    LOG.info("backbone layers: %d total, %d trainable (augment=%s)",
             len(base.layers), trainable, augment)
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", type=Path,
                    default=PROJECT_ROOT / "paper_pipeline" / "output" / "features7")
    ap.add_argument("--suffix", default="lesion_equalize")
    ap.add_argument("--lambdas", default="odd_harmonic")
    ap.add_argument("--backbone", default="efficientnetb0",
                    choices=["efficientnetb0", "resnet50"])
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--unfreeze", type=int, default=999,
                    help="Layers to unfreeze from the top; >= depth unfreezes all.")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--lr", type=float, default=5e-5)   # lower: whole backbone trains
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")

    import tensorflow as tf
    from sklearn.model_selection import GroupShuffleSplit

    lam = args.lambdas.split(",")[0].strip()
    tr_path = args.features_dir / f"features_train_{lam}_{args.suffix}.pkl"
    te_path = args.features_dir / f"features_test_{lam}_{args.suffix}.pkl"
    if not tr_path.is_file() or not te_path.is_file():
        LOG.error("Missing feature pickles for suffix=%s", args.suffix)
        return 2

    train_df = pd.read_pickle(tr_path)
    test_df = pd.read_pickle(te_path)

    # One row per source image for fine-tuning; the split comes from the pickles.
    uniq = (train_df[["source_id", "dx", "label", "lesion_id"]]
            .drop_duplicates("source_id").reset_index(drop=True))
    LOG.info("fine-tuning on %d unique training images (from %d augmented rows)",
             len(uniq), len(train_df))
    if set(uniq["source_id"]) & set(test_df["source_id"]):
        LOG.error("TEST IMAGES PRESENT IN FINE-TUNING SET — aborting")
        return 1

    # Held-out validation for early stopping, grouped by lesion so a repeat
    # photograph of a training lesion cannot be used to select the epoch.
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction,
                            random_state=args.random_state)
    tr_idx, va_idx = next(gss.split(uniq, uniq["label"], uniq["lesion_id"]))
    LOG.info("fine-tune split: %d train / %d val images", len(tr_idx), len(va_idx))

    base, preprocess_fn = get_backbone(args.backbone, args.input_size)
    model = build_model(base, len(CLASSES7), args.unfreeze, args.lr,
                        augment=not args.no_augment)

    def make_arrays(idx):
        recs = uniq.iloc[idx].to_dict("records")
        X, keep = load_batch(recs, args.input_size)
        y = np.array([recs[i]["label"] for i in keep], dtype=np.int64)
        return preprocess_fn(X), y

    LOG.info("loading images into memory (%d px)…", args.input_size)
    t0 = time.time()
    X_tr, y_tr = make_arrays(tr_idx)
    X_va, y_va = make_arrays(va_idx)
    LOG.info("loaded train %s val %s in %.1f min", X_tr.shape, X_va.shape,
             (time.time() - t0) / 60)

    counts = np.bincount(y_tr, minlength=len(CLASSES7)).astype(float)
    counts[counts == 0] = 1.0
    class_weight = {i: float(len(y_tr) / (len(CLASSES7) * c)) for i, c in enumerate(counts)}
    LOG.info("class weights: %s",
             {CLASSES7[i]: round(w, 2) for i, w in class_weight.items()})

    from sklearn.metrics import balanced_accuracy_score

    class BalancedAccuracy(tf.keras.callbacks.Callback):
        """Early stopping on plain accuracy would optimise for nevus, which is
        67% of the validation set. This reports the metric we actually report."""

        def __init__(self, Xv, yv):
            super().__init__()
            self.Xv, self.yv = Xv, yv

        def on_epoch_end(self, epoch, logs=None):
            pred = np.argmax(self.model.predict(self.Xv, verbose=0), axis=1)
            bal = balanced_accuracy_score(self.yv, pred)
            (logs if logs is not None else {})["val_bal_acc"] = bal
            LOG.info("  epoch %d  val_bal_acc=%.4f", epoch + 1, bal)

    ckpt = args.features_dir / f"finetuned_{args.backbone}_{args.suffix}.keras"
    callbacks = [
        BalancedAccuracy(X_va, y_va),
        tf.keras.callbacks.EarlyStopping(monitor="val_bal_acc", mode="max", patience=8,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3, min_lr=1e-7, verbose=1),
    ]
    LOG.info("training for up to %d epochs…", args.epochs)
    t0 = time.time()
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=args.epochs, batch_size=args.batch_size,
              class_weight=class_weight, callbacks=callbacks, verbose=2)
    LOG.info("fine-tuning done in %.1f min", (time.time() - t0) / 60)
    model.save(ckpt)
    LOG.info("saved %s", ckpt.name)

    del X_tr, X_va

    # Feature extractor = fine-tuned backbone up to the pooled embedding.
    extractor = tf.keras.Model(model.input, model.layers[-3].output)
    dim = extractor.output_shape[-1]
    LOG.info("re-extracting %d-d features from the fine-tuned backbone", dim)

    def extract(df: pd.DataFrame, label: str) -> pd.DataFrame:
        recs = df[["source_id", "aug_tag", "dx"]].to_dict("records")
        feats = np.zeros((len(recs), dim), dtype=np.float32)
        started = time.time()
        for start in range(0, len(recs), args.batch_size):
            chunk = recs[start:start + args.batch_size]
            X, keep = load_batch(chunk, args.input_size)
            if len(keep):
                feats[[start + i for i in keep]] = extractor.predict(
                    preprocess_fn(X), verbose=0)
            done = start + len(chunk)
            if done % (args.batch_size * 40) == 0 or done >= len(recs):
                rate = done / (time.time() - started)
                LOG.info("  %s %d/%d (%.1f img/s, ETA %.1f min)", label, done,
                         len(recs), rate, (len(recs) - done) / rate / 60)
        out = pd.DataFrame(feats, columns=[f"ftcnn_{i}" for i in range(dim)])
        out.insert(0, "aug_tag", df["aug_tag"].to_numpy())
        out.insert(0, "source_id", df["source_id"].to_numpy())
        return out

    for name, df in (("train", train_df), ("test", test_df)):
        cnn = extract(df, name.upper())
        assert (cnn["source_id"].to_numpy() == df["source_id"].to_numpy()).all()
        assert (cnn["aug_tag"].to_numpy() == df["aug_tag"].to_numpy()).all()
        merged = pd.concat(
            [df.reset_index(drop=True),
             cnn.drop(columns=["source_id", "aug_tag"]).reset_index(drop=True)], axis=1)
        out = args.features_dir / f"features_{name}_{lam}_{args.suffix}_ft.pkl"
        merged.to_pickle(out)
        LOG.info("%s: %s -> %s  (%d features)", name, df.shape, merged.shape,
                 merged.shape[1] - len(BOOKKEEPING))

    LOG.info("Done. Next: metadata7.py --suffix %s_ft, then train_eval7.py", args.suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
