"""
Mutual-information feature-family analysis for paper Section 5.1.

Reproduces the "Dominance of MKT Features After Feature Selection" figure:
groups selected features by family (MKT, color, texture, geometric),
ranks each family by median MI score, and renders a scatter+annotation plot.

Reads:
    paper_pipeline/output/features/features_train_<lambda>.pkl

For each λ, writes:
    paper_pipeline/output/figures/feature_dominance_<lambda>.png
    paper_pipeline/output/figures/feature_dominance_<lambda>.csv

The script also prints median MI per family for inclusion in the paper.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "features"
FIGURES_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "figures"

LOG = logging.getLogger("feature_family_analysis")


# Feature-name → family heuristic. Updated as we learn the actual keys.
MKT_TOKENS = (
    "mdfkt", "krawtchouk", "_mkt_", "real_", "imag_", "magnitude_", "phase_",
    "_real_", "_imag_", "_magnitude_", "_phase_",
)
COLOR_TOKENS = (
    "_rgb_", "_hsv_", "_lab_", "_cielab_", "rgb_", "hsv_", "lab_",
    "channel_mean", "channel_std", "channel_skew", "channel_kurt", "channel_min",
    "channel_max", "channel_entropy", "histogram_", "dominant_color",
    "color_asym", "color_uniform",
)
TEXTURE_TOKENS = (
    "glcm_", "lbp_", "gabor_", "wavelet_", "gradient_", "db1_", "haralick",
    "contrast", "dissimilarity", "homogeneity", "energy", "correlation", "asm",
)
GEOMETRIC_TOKENS = (
    "geom_", "area", "perimeter", "compactness", "eccentricity", "extent",
    "boundary_length", "border_", "convexity", "hu_moment", "fractal",
    "circularity", "asymmetry", "smoothness", "curvature",
)


def classify_feature(name: str) -> str:
    n = name.lower()
    # CNN features check first (cleanest token, no ambiguity)
    if n.startswith("cnn_") or "_cnn_" in n:
        return "CNN"
    if any(tok in n for tok in MKT_TOKENS):
        return "MKT"
    if any(tok in n for tok in TEXTURE_TOKENS):
        return "Texture"
    if any(tok in n for tok in COLOR_TOKENS):
        return "Color"
    if any(tok in n for tok in GEOMETRIC_TOKENS):
        return "Geometric"
    return "Other"


def analyse_one(lambda_name: str, top_k: int = 380) -> dict | None:
    train_pickle = FEATURES_DIR / f"features_train_{lambda_name}.pkl"
    if not train_pickle.is_file():
        LOG.warning("Missing %s — skipping", train_pickle.name)
        return None
    LOG.info("Loading %s", train_pickle.name)
    df = pd.read_pickle(train_pickle)
    feature_cols = [c for c in df.columns if c not in {"source_id", "aug_tag", "label"}]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    y = df["label"].to_numpy(dtype=np.int64)
    LOG.info("  shape: X=%s, y=%s", X.shape, y.shape)

    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=3)
    LOG.info("  MI computed: min=%.4f max=%.4f median=%.4f",
             mi.min(), mi.max(), np.median(mi))

    # Top-k by MI score
    order = np.argsort(mi)[::-1][: min(top_k, len(mi))]
    selected = [(feature_cols[i], float(mi[i])) for i in order]

    families: dict[str, list[float]] = defaultdict(list)
    for name, score in selected:
        families[classify_feature(name)].append(score)

    medians = {fam: float(np.median(scores)) for fam, scores in families.items()}
    counts = {fam: len(scores) for fam, scores in families.items()}
    LOG.info("  Family medians: %s", {f: f"{v:.4f}" for f, v in medians.items()})
    LOG.info("  Family counts : %s", counts)

    # Plot
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fam_order = sorted(families.keys(), key=lambda f: -medians.get(f, 0))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    palette = {"MKT": "tab:red", "Color": "tab:blue", "Texture": "tab:green",
               "Geometric": "tab:orange", "CNN": "tab:purple", "Other": "gray"}
    for i, fam in enumerate(fam_order):
        scores = families[fam]
        xs = i + (np.random.rand(len(scores)) - 0.5) * 0.5
        ax.scatter(xs, scores, alpha=0.5, color=palette.get(fam, "gray"), s=12)
        ax.scatter([i], [medians[fam]], color="black", s=80, marker="_", linewidths=3)
        ax.annotate(f"median={medians[fam]:.3f}\nn={counts[fam]}",
                    xy=(i, medians[fam]),
                    xytext=(i, max(scores) + 0.02),
                    ha="center", fontsize=8)

    ax.set_xticks(range(len(fam_order)))
    ax.set_xticklabels(fam_order)
    ax.set_ylabel("Mutual information score")
    ax.set_title(f"Feature-family MI dominance — λ = {lambda_name}  (top-{top_k} features)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = FIGURES_DIR / f"feature_dominance_{lambda_name}.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    LOG.info("  wrote %s", out_png.name)

    # Also save the per-feature MI as CSV
    rows = [{"feature": name, "mi": score, "family": classify_feature(name)}
            for name, score in selected]
    out_csv = FIGURES_DIR / f"feature_dominance_{lambda_name}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    LOG.info("  wrote %s", out_csv.name)

    return {
        "lambda": lambda_name,
        "medians": medians,
        "counts": counts,
        "fig": str(out_png),
        "csv": str(out_csv),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", type=str,
                        default="low_pass,high_pass,dft,odd_harmonic")
    parser.add_argument("--top-k", type=int, default=380)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    chosen = [s.strip() for s in args.lambdas.split(",") if s.strip()]
    results = []
    for lam in chosen:
        r = analyse_one(lam, top_k=args.top_k)
        if r is not None:
            results.append(r)

    print("\n================  Family medians per λ  ================")
    print(f"{'λ':<14} {'MKT':>8} {'Color':>8} {'Texture':>8} {'Geom':>8} {'CNN':>8} {'Other':>8}")
    for r in results:
        med = r["medians"]
        print(f"{r['lambda']:<14}"
              f" {med.get('MKT', 0):>8.4f}"
              f" {med.get('Color', 0):>8.4f}"
              f" {med.get('Texture', 0):>8.4f}"
              f" {med.get('Geometric', 0):>8.4f}"
              f" {med.get('CNN', 0):>8.4f}"
              f" {med.get('Other', 0):>8.4f}")
    # Also print the COUNT of selected features per family (paper Section 5.1).
    print()
    print(f"{'λ':<14} {'MKT':>5} {'Color':>5} {'Tex':>5} {'Geom':>5} {'CNN':>5} {'Other':>5}  (counts in top-k)")
    for r in results:
        c = r["counts"]
        print(f"{r['lambda']:<14}"
              f" {c.get('MKT', 0):>5}"
              f" {c.get('Color', 0):>5}"
              f" {c.get('Texture', 0):>5}"
              f" {c.get('Geometric', 0):>5}"
              f" {c.get('CNN', 0):>5}"
              f" {c.get('Other', 0):>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
