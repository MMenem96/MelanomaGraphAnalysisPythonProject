"""
Merge CNN features into each (handcrafted+MKT) pickle.

Reads:
    paper_pipeline/output/features/features_train_<λ>.pkl  (461 handcrafted+MKT cols)
    paper_pipeline/output/features/features_test_<λ>.pkl
    paper_pipeline/output/features/cnn_features_<backbone>.pkl  (2048 CNN cols)

Writes:
    paper_pipeline/output/features/features_train_<λ>_hybrid.pkl  (461 + 2048 = 2509 cols)
    paper_pipeline/output/features/features_test_<λ>_hybrid.pkl

Joins on (source_id, aug_tag). Asserts that every handcrafted row finds a
matching CNN row.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("merge_cnn")

BOOKKEEPING = {"source_id", "aug_tag", "label"}


def merge_one(handcrafted_path: Path, cnn_df: pd.DataFrame, label: str) -> pd.DataFrame:
    hc_df = pd.read_pickle(handcrafted_path)
    LOG.info("  %s: handcrafted %d rows × %d cols", label, len(hc_df), hc_df.shape[1])

    # CNN side: subset to rows whose (source_id, aug_tag) appears in handcrafted
    hc_keys = set(zip(hc_df["source_id"], hc_df["aug_tag"]))
    cnn_subset = cnn_df[cnn_df.apply(
        lambda r: (r["source_id"], r["aug_tag"]) in hc_keys, axis=1
    )].copy()
    LOG.info("  %s: CNN subset      %d rows × %d cols", label, len(cnn_subset), cnn_subset.shape[1])

    # Inner merge on (source_id, aug_tag, label)
    merged = hc_df.merge(
        cnn_subset.drop(columns=["label"]),
        on=["source_id", "aug_tag"],
        how="inner",
        validate="one_to_one",
    )
    LOG.info("  %s: MERGED          %d rows × %d cols", label, len(merged), merged.shape[1])

    if len(merged) != len(hc_df):
        raise RuntimeError(
            f"Row count mismatch after merge: handcrafted={len(hc_df)}, merged={len(merged)}. "
            "Some (source_id, aug_tag) pairs have no CNN counterpart."
        )
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path,
        default=PROJECT_ROOT / "paper_pipeline" / "output" / "features",
    )
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument(
        "--lambdas", default="low_pass,high_pass,dft,odd_harmonic",
    )
    parser.add_argument(
        "--suffix", default="hybrid",
        help="Output suffix: features_<split>_<lambda>_<suffix>.pkl",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    cnn_pkl = args.features_dir / f"cnn_features_{args.backbone}.pkl"
    if not cnn_pkl.is_file():
        LOG.error("CNN pickle not found: %s. Run cnn_features.py first.", cnn_pkl)
        return 2
    cnn_df = pd.read_pickle(cnn_pkl)
    LOG.info("Loaded CNN features: %s", cnn_pkl.name)
    LOG.info("  CNN shape: %s", cnn_df.shape)

    chosen = [s.strip() for s in args.lambdas.split(",") if s.strip()]
    for lam in chosen:
        LOG.info("======== λ = %s ========", lam)
        train_in = args.features_dir / f"features_train_{lam}.pkl"
        test_in = args.features_dir / f"features_test_{lam}.pkl"
        if not train_in.is_file() or not test_in.is_file():
            LOG.warning("Missing pickles for λ=%s; skipping", lam)
            continue

        merged_train = merge_one(train_in, cnn_df, label=f"{lam} train")
        merged_test = merge_one(test_in, cnn_df, label=f"{lam} test")

        train_out = args.features_dir / f"features_train_{lam}_{args.suffix}.pkl"
        test_out = args.features_dir / f"features_test_{lam}_{args.suffix}.pkl"
        merged_train.to_pickle(train_out)
        merged_test.to_pickle(test_out)
        LOG.info("Wrote %s and %s", train_out.name, test_out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
