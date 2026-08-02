"""Regenerate the §5.1 per-family MI analysis using deterministic MI on the
full 2,509-feature pickle (including the 2,048 CNN features), so the paper's
family counts and median-MI numbers trace cleanly to a single CSV per λ.

Differences vs the earlier `feature_dominance_<λ>.csv` outputs:
  - Includes CNN features (those CSVs were handcrafted-only, 380 rows).
  - Pins `random_state=42` and `n_jobs=1` so MI is reproducible.
  - Classifies the 5 boundary/extent columns as Geometric (matching paper text).
  - Saves both:
      * `family_mi_full_<λ>.csv`  — per-feature MI ranking (2,509 rows)
      * `family_mi_selected_<λ>.csv` — only the SelectKBest top-360, per-feature
      * `family_mi_summary_<λ>.csv` — n + median MI per family for the top-360

Headline figures for the paper come from `family_mi_summary_<λ>.csv`.
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "features"
RESULTS_DIR  = PROJECT_ROOT / "paper_pipeline" / "output" / "results"

LAMBDAS      = ["low_pass", "high_pass", "dft", "odd_harmonic"]
N_FEATURES   = 360
RANDOM_STATE = 42
BOOKKEEPING  = {"source_id", "aug_tag", "label"}

# Token rules — matched to phase_a_full_grid.py but with the 5 boundary/extent
# features explicitly placed under Geometric (the paper's intent).
MKT_TOKENS = ("mdfkt", "_mkt_", "real_", "imag_", "magnitude_", "phase_",
              "_real_", "_imag_", "_magnitude_", "_phase_")
COLOR_TOKENS = ("_rgb_", "_hsv_", "_lab_", "_cielab_", "rgb_", "hsv_", "lab_",
                "cielab_", "channel_", "histogram_", "dominant_color",
                "color_asym", "color_uniform")
TEXTURE_TOKENS = ("glcm", "lbp_", "gabor", "wavelet", "gradient_",
                  "haralick", "_contrast", "_dissimilarity",
                  "_homogeneity", "_energy_", "_correlation_")
GEOMETRIC_TOKENS = ("area", "perimeter", "eccentric", "circularity",
                    "compactness", "convexity", "diameter", "asymmetry",
                    "fractal", "curvature", "smoothness", "hu_moment",
                    "extent", "boundary", "border_distance")
CNN_TOKENS = ("resnet", "cnn_", "_cnn", "deep_feat", "emb_",
              "efficientnet", "densenet")


def classify(name: str) -> str:
    s = name.lower()
    if any(t in s for t in MKT_TOKENS):      return "MKT"
    if any(t in s for t in CNN_TOKENS):      return "CNN"
    if any(t in s for t in COLOR_TOKENS):    return "Color"
    if any(t in s for t in TEXTURE_TOKENS):  return "Texture"
    if any(t in s for t in GEOMETRIC_TOKENS):return "Geometric"
    return "Unknown"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("mi_regen")
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for lam in LAMBDAS:
        log.info("λ = %s", lam)
        train_pkl = FEATURES_DIR / f"features_train_{lam}_hybrid.pkl"
        df = pd.read_pickle(train_pkl)
        cols = [c for c in df.columns if c not in BOOKKEEPING]
        X = np.nan_to_num(df[cols].to_numpy(dtype=np.float64),
                          nan=0.0, posinf=1e10, neginf=-1e10)
        y = df["label"].to_numpy(dtype=np.int64)

        mi = mutual_info_classif(X, y, random_state=RANDOM_STATE,
                                  n_neighbors=3, n_jobs=1)
        fam = [classify(c) for c in cols]
        full = pd.DataFrame({"feature": cols, "family": fam, "mi": mi})
        full = full.sort_values("mi", ascending=False).reset_index(drop=True)
        full.to_csv(RESULTS_DIR / f"family_mi_full_{lam}.csv", index=False)

        # Top-360 by MI (= what SelectKBest keeps)
        top = full.head(N_FEATURES).copy()
        top.to_csv(RESULTS_DIR / f"family_mi_selected_{lam}.csv", index=False)

        # Per-family summary for the SELECTED 360 features
        summary = (top.groupby("family")
                      .agg(n_selected=("mi", "size"),
                           median_mi=("mi", "median"),
                           mean_mi=("mi", "mean"),
                           max_mi=("mi", "max"))
                      .sort_values("median_mi", ascending=False))
        log.info("\n%s", summary.to_string())
        summary.to_csv(RESULTS_DIR / f"family_mi_summary_{lam}.csv")

        # Track headline figures (for paper §5.1)
        for fam_name, row in summary.iterrows():
            summary_rows.append({
                "lambda": lam,
                "family": fam_name,
                "n_selected_in_top360": int(row["n_selected"]),
                "median_mi": float(row["median_mi"]),
                "mean_mi": float(row["mean_mi"]),
                "max_mi": float(row["max_mi"]),
            })

    combined = pd.DataFrame(summary_rows)
    combined.to_csv(RESULTS_DIR / f"family_mi_combined_{ts}.csv", index=False)
    log.info("\nCombined summary written to family_mi_combined_%s.csv", ts)

    # Print headline numbers in paper format
    log.info("\n\nPAPER §5.1 NUMBERS:")
    for lam in LAMBDAS:
        sub = combined[combined["lambda"] == lam]
        log.info("\nλ = %s:", lam)
        for _, r in sub.iterrows():
            log.info("  %-9s n=%-3d median MI=%.3f", r["family"],
                     r["n_selected_in_top360"], r["median_mi"])


if __name__ == "__main__":
    main()
