"""
Generate diagnostic figures for the paper from saved test predictions.

Reads:
    paper_pipeline/output/predictions/preds_<lambda>_<classifier>.npz

Writes:
    paper_pipeline/output/figures/roc_<lambda>_<classifier>.png
    paper_pipeline/output/figures/pr_<lambda>_<classifier>.png
    paper_pipeline/output/figures/confusion_<lambda>_<classifier>.png
    paper_pipeline/output/figures/prob_dist_<lambda>_<classifier>.png

By default, regenerates figures for the BEST classifier per λ as identified
by `summary_*.csv`. Pass `--all` to render every classifier × λ.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "predictions"
RESULTS_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "results"
FIGURES_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "figures"

LOG = logging.getLogger("generate_figures")


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def plot_roc(y_test, y_proba, title: str, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pr(y_test, y_proba, title: str, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(recall, precision, lw=2, label=f"AP = {pr_auc:.3f}")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion(y_test, y_pred, title: str, out_path: Path) -> None:
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["BKL (0)", "BCC (1)"])
    ax.set_yticklabels(["BKL (0)", "BCC (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_probability_dist(y_test, y_proba, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    bkl = y_proba[y_test == 0]
    bcc = y_proba[y_test == 1]
    ax.hist(bkl, bins=30, alpha=0.6, label=f"BKL (n={len(bkl)})", color="tab:blue")
    ax.hist(bcc, bins=30, alpha=0.6, label=f"BCC (n={len(bcc)})", color="tab:orange")
    ax.axvline(0.5, color="black", lw=1, linestyle="--", label="decision threshold = 0.5")
    ax.set_xlabel("Predicted probability P(class = BCC)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def render_one(pred_file: Path) -> None:
    data = np.load(pred_file, allow_pickle=True)
    classifier = str(data["classifier"])
    lambda_name = str(data["lambda_name"])
    y_test = np.asarray(data["y_test"]).astype(int)
    y_pred = np.asarray(data["y_pred"]).astype(int)
    y_proba_arr = np.asarray(data["y_proba"])
    has_proba = y_proba_arr.size > 0

    title_base = f"{classifier}  (λ = {lambda_name})"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    safe = f"{lambda_name}_{_safe(classifier)}"

    plot_confusion(y_test, y_pred, f"Confusion — {title_base}",
                   FIGURES_DIR / f"confusion_{safe}.png")
    LOG.info("  wrote confusion_%s.png", safe)

    if has_proba:
        plot_roc(y_test, y_proba_arr, f"ROC — {title_base}",
                 FIGURES_DIR / f"roc_{safe}.png")
        plot_pr(y_test, y_proba_arr, f"PR — {title_base}",
                FIGURES_DIR / f"pr_{safe}.png")
        plot_probability_dist(y_test, y_proba_arr, f"Probability — {title_base}",
                              FIGURES_DIR / f"prob_dist_{safe}.png")
        LOG.info("  wrote roc/pr/prob_dist for %s", safe)


def find_best_per_lambda(summary_csv: Path) -> list[tuple[str, str]]:
    """Return [(lambda_name, classifier), …] — the best classifier per λ."""
    df = pd.read_csv(summary_csv)
    out: list[tuple[str, str]] = []
    for lam, sub in df.groupby("lambda"):
        best = sub.loc[sub["test_accuracy"].idxmax()]
        out.append((str(best["lambda"]), str(best["classifier"])))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="Render every prediction file (default: best per λ only).")
    parser.add_argument("--summary-csv", type=Path, default=None,
                        help="Path to summary CSV (default: most recent).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    if not PREDICTIONS_DIR.is_dir():
        LOG.error("No predictions directory at %s — run train_eval first.",
                  PREDICTIONS_DIR)
        return 2

    pred_files = sorted(PREDICTIONS_DIR.glob("preds_*.npz"))
    if not pred_files:
        LOG.error("No prediction files found in %s.", PREDICTIONS_DIR)
        return 2

    if args.all:
        targets = pred_files
    else:
        summary_csv = args.summary_csv
        if summary_csv is None:
            summary_csv = max(RESULTS_DIR.glob("summary_*.csv"),
                              key=lambda p: p.stat().st_mtime,
                              default=None)
        if summary_csv is None:
            LOG.error("Could not find a summary CSV. Pass --summary-csv or use --all.")
            return 2
        best = find_best_per_lambda(summary_csv)
        targets = []
        for lam, clf in best:
            cand = PREDICTIONS_DIR / f"preds_{lam}_{_safe(clf)}.npz"
            if cand.is_file():
                targets.append(cand)
            else:
                LOG.warning("Missing prediction file: %s", cand.name)

    LOG.info("Rendering %d figure sets to %s", len(targets), FIGURES_DIR)
    for f in targets:
        LOG.info("Processing %s", f.name)
        render_one(f)

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
