"""
Fuse patient metadata (age, sex, lesion localization) into the 7-class features.

Sonuç et al. (Front. Med. 2026) report that adding exactly these three HAM10000
metadata fields lifts their stacked ensemble from 92.98% to 98.55% accuracy
(+5.57 points) under a lesion-ID-level split. The fields have been sitting in
`data/HAM10000_metadata.csv` unused by our pipeline.

Two deliberate differences from that paper:

  * **Real metadata only.** They generate synthetic metadata with SMOTENC and
    pair it with a randomly chosen same-class image, which they acknowledge
    "may introduce class-conditional feature associations not present in real
    clinical data". Here every augmented row inherits the *true* metadata of
    its source image, so no synthetic patient records are invented.
  * **Train-only fitting.** The age median used for imputation is computed on
    the training rows alone. Categorical vocabularies are fixed a priori from
    the dataset definition, so no test information reaches the encoder.

Adds 1 (age) + 3 (sex) + 15 (localization) = 19 columns.

Usage:
    python paper_pipeline/pipeline/metadata7.py --suffix lesion_equalize_hybrid
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG = logging.getLogger("metadata7")

METADATA_CSV = PROJECT_ROOT / "data" / "HAM10000_metadata.csv"

SEX_VALUES = ["male", "female", "unknown"]
LOCALIZATIONS = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital",
    "hand", "lower extremity", "neck", "scalp", "trunk", "unknown",
    "upper extremity",
]


def build_metadata_frame(age_median: float | None = None) -> tuple[pd.DataFrame, float]:
    """Return one row per image_id with encoded metadata, plus the age median used."""
    meta = pd.read_csv(METADATA_CSV)
    if age_median is None:
        age_median = float(meta["age"].median())

    out = pd.DataFrame({"source_id": meta["image_id"]})
    out["meta_age"] = meta["age"].fillna(age_median).astype(float)

    sex = meta["sex"].fillna("unknown").replace({"": "unknown"})
    for value in SEX_VALUES:
        out[f"meta_sex_{value}"] = (sex == value).astype(float)

    loc = meta["localization"].fillna("unknown").replace({"": "unknown"})
    unseen = set(loc.unique()) - set(LOCALIZATIONS)
    if unseen:
        LOG.warning("localization values not in the fixed vocabulary: %s", sorted(unseen))
    for value in LOCALIZATIONS:
        out[f"meta_loc_{value.replace(' ', '_')}"] = (loc == value).astype(float)

    return out, age_median


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", type=Path,
                    default=PROJECT_ROOT / "paper_pipeline" / "output" / "features7")
    ap.add_argument("--suffix", default="lesion_equalize_hybrid")
    ap.add_argument("--lambdas", default="odd_harmonic")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")

    for lam in [s.strip() for s in args.lambdas.split(",") if s.strip()]:
        tr_path = args.features_dir / f"features_train_{lam}_{args.suffix}.pkl"
        te_path = args.features_dir / f"features_test_{lam}_{args.suffix}.pkl"
        if not tr_path.is_file() or not te_path.is_file():
            LOG.error("Missing pickles for λ=%s suffix=%s", lam, args.suffix)
            return 2

        train_df = pd.read_pickle(tr_path)
        test_df = pd.read_pickle(te_path)

        # Age median from TRAINING images only.
        meta_raw = pd.read_csv(METADATA_CSV)
        train_ids = set(train_df["source_id"])
        age_median = float(meta_raw.loc[meta_raw["image_id"].isin(train_ids), "age"].median())
        LOG.info("age median (train only): %.1f", age_median)

        meta_df, _ = build_metadata_frame(age_median=age_median)
        meta_cols = [c for c in meta_df.columns if c != "source_id"]

        for name, df, path in (("train", train_df, tr_path), ("test", test_df, te_path)):
            merged = df.merge(meta_df, on="source_id", how="left", validate="many_to_one")
            missing = merged[meta_cols].isna().any(axis=1).sum()
            if missing:
                LOG.error("%s: %d rows found no metadata match", name, missing)
                return 1
            if len(merged) != len(df):
                LOG.error("%s: row count changed %d -> %d", name, len(df), len(merged))
                return 1

            out = args.features_dir / f"features_{name}_{lam}_{args.suffix}_meta.pkl"
            merged.to_pickle(out)
            n_feat = merged.shape[1] - 5   # source_id, aug_tag, label, lesion_id, dx
            LOG.info("%s: %s -> %s  (%d features, +%d metadata)",
                     name, df.shape, merged.shape, n_feat, len(meta_cols))

    LOG.info("Done. Next: train_eval7.py --suffix %s_meta", args.suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
