"""
ResNet-50 features for the 7-class arm, aligned to the handcrafted rows.

Brings Track B to the same 2,509-feature representation as the binary paper:
    461 handcrafted+MKT  +  2,048 frozen ResNet-50  =  2,509

Alignment is guaranteed by construction: instead of re-deriving the split, this
reads the (source_id, aug_tag) pairs straight out of the handcrafted pickles
produced by `feature_extraction7.py` and computes a CNN vector for exactly those
rows. There is no way for the two halves to disagree about the split, the
augmentation policy, or the row order.

Outputs (per suffix):
    features7/cnn_<suffix>.pkl                      raw CNN rows
    features7/features_train_<lam>_<suffix>_hybrid.pkl   merged 2,509 features
    features7/features_test_<lam>_<suffix>_hybrid.pkl

Usage:
    python paper_pipeline/pipeline/cnn_features7.py --suffix lesion_equalize
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
from paper_pipeline.pipeline.dataset import SEGMENTED7_DIR

LOG = logging.getLogger("cnn_features7")

TARGET_SIZE = (224, 224)
DEFAULT_LAMBDA = "odd_harmonic"
BOOKKEEPING = {"source_id", "aug_tag", "label", "lesion_id", "dx"}


def get_backbone():
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg",
                     input_shape=(*TARGET_SIZE, 3))
    model.trainable = False
    return model, preprocess_input


def load_variant(source_id: str, dx: str, aug_tag: str) -> np.ndarray | None:
    """Load a segmented PNG and re-apply the exact augmentation tag used for
    the handcrafted row, so both halves describe the same pixels."""
    path = SEGMENTED7_DIR / dx / f"{source_id}.png"
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if aug_tag != TAG_ORIGINAL:
        rgb = apply_tag(rgb, aug_tag)
    return cv2.resize(rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)


def extract_for_frame(df: pd.DataFrame, model, preprocess_fn, batch_size: int,
                      shard_path: Path, label: str) -> pd.DataFrame:
    """CNN features for every row of `df`, checkpointed to `shard_path`."""
    if shard_path.is_file():
        cached = pd.read_pickle(shard_path)
        if len(cached) == len(df):
            LOG.info("  %s: %d rows already on disk — skipping", label, len(cached))
            return cached
        LOG.warning("  %s: cached %d rows != expected %d — recomputing",
                    label, len(cached), len(df))

    keys = df[["source_id", "aug_tag", "dx"]].to_dict("records")
    n = len(keys)
    feats = np.zeros((n, 2048), dtype=np.float32)
    started = time.time()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        imgs, valid = [], []
        for i in range(start, end):
            k = keys[i]
            img = load_variant(k["source_id"], k["dx"], k["aug_tag"])
            if img is None:
                LOG.warning("  unreadable: %s/%s", k["dx"], k["source_id"])
                continue
            imgs.append(img)
            valid.append(i)
        if imgs:
            batch = preprocess_fn(np.stack(imgs).astype(np.float32))
            feats[valid] = model.predict(batch, verbose=0)
        if end % (batch_size * 10) == 0 or end == n:
            elapsed = time.time() - started
            rate = end / elapsed
            LOG.info("  %s  %d/%d  (%.1f img/s, ETA %.1f min)",
                     label, end, n, rate, (n - end) / rate / 60.0)

    out = pd.DataFrame(feats, columns=[f"cnn_{i}" for i in range(2048)])
    out.insert(0, "aug_tag", df["aug_tag"].to_numpy())
    out.insert(0, "source_id", df["source_id"].to_numpy())
    out.to_pickle(shard_path)
    LOG.info("  %s: wrote %s (%d rows)", label, shard_path.name, len(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", type=Path,
                    default=PROJECT_ROOT / "paper_pipeline" / "output" / "features7")
    ap.add_argument("--suffix", default="lesion_equalize")
    ap.add_argument("--lambdas", default=DEFAULT_LAMBDA)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")

    lams = [s.strip() for s in args.lambdas.split(",") if s.strip()]
    LOG.info("Loading ResNet-50 (ImageNet, frozen, GAP → 2048)…")
    model, preprocess_fn = get_backbone()

    for lam in lams:
        tr_path = args.features_dir / f"features_train_{lam}_{args.suffix}.pkl"
        te_path = args.features_dir / f"features_test_{lam}_{args.suffix}.pkl"
        if not tr_path.is_file() or not te_path.is_file():
            LOG.error("Missing handcrafted pickles for λ=%s (%s)", lam, args.suffix)
            return 2

        train_df = pd.read_pickle(tr_path)
        test_df = pd.read_pickle(te_path)
        LOG.info("λ=%s  handcrafted train %s  test %s", lam, train_df.shape, test_df.shape)

        cnn_train = extract_for_frame(
            train_df, model, preprocess_fn, args.batch_size,
            args.features_dir / f"cnn_train_{args.suffix}.pkl", "TRAIN")
        cnn_test = extract_for_frame(
            test_df, model, preprocess_fn, args.batch_size,
            args.features_dir / f"cnn_test_{args.suffix}.pkl", "TEST")

        for name, hc, cnn in (("train", train_df, cnn_train), ("test", test_df, cnn_test)):
            if len(hc) != len(cnn):
                LOG.error("%s row mismatch: handcrafted=%d cnn=%d", name, len(hc), len(cnn))
                return 1
            # Positional concat is safe: cnn rows were built from hc in order.
            same_id = (hc["source_id"].to_numpy() == cnn["source_id"].to_numpy()).all()
            same_tag = (hc["aug_tag"].to_numpy() == cnn["aug_tag"].to_numpy()).all()
            if not (same_id and same_tag):
                LOG.error("%s alignment broken (source_id match=%s, aug_tag match=%s)",
                          name, same_id, same_tag)
                return 1

            merged = pd.concat(
                [hc.reset_index(drop=True),
                 cnn.drop(columns=["source_id", "aug_tag"]).reset_index(drop=True)],
                axis=1)
            n_feat = len([c for c in merged.columns if c not in BOOKKEEPING])
            out = args.features_dir / f"features_{name}_{lam}_{args.suffix}_hybrid.pkl"
            merged.to_pickle(out)
            LOG.info("  %s hybrid: %s → %d features (%d handcrafted + 2048 CNN)  %s",
                     name, merged.shape, n_feat, n_feat - 2048, out.name)

    LOG.info("Done. Next: python paper_pipeline/pipeline/train_eval7.py "
             "--suffix %s_hybrid", args.suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
