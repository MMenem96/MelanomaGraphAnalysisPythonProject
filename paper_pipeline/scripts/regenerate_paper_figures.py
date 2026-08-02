"""Regenerate all paper figures from the canonical Phase A v2 outputs:
  Fig 11: mkt_vs_accuracy.png   — best acc per λ
  Fig 12: mkt_vs_auc.png        — best AUC per λ
  Fig 13: roc_curve.png         — ROC for MLP+odd_harmonic
  Fig 14: confusion_matrix.png  — CM for MLP+odd_harmonic
  Fig 15: pr_curve.png          — PR for MLP+odd_harmonic
  Fig 16: prob_distribution.png — probability histogram
  Fig 17: k_sweep_curve.png     — k-sensitivity sweep
  Fig (5.1) mkt_feature_dominance.png — per-family MI scatter

All outputs go to paper_pipeline/output/figures/ AND are copied to
Mohamed_Shoieb_Master_Paper_ElsevierV7/images/ so the LaTeX picks them up.
"""
from __future__ import annotations
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                              roc_curve)


PROJECT_ROOT  = Path(__file__).resolve().parents[2]
RESULTS_DIR   = PROJECT_ROOT / "paper_pipeline" / "output" / "results"
FIGURES_DIR   = PROJECT_ROOT / "paper_pipeline" / "output" / "figures"
PAPER_IMG_DIR = Path("/Users/mmoniem96/Desktop/Work/Master/"
                     "Mohamed_Shoieb_Master_Paper_ElsevierV7/images")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Canonical inputs — Phase A v2 with HEADLINE=odd_harmonic_MLP
SUMMARY_CSV  = RESULTS_DIR / "phase_a_summary_20260522_154502.csv"
PRED_NPZ     = RESULTS_DIR / "phase_a_predictions_20260522_154502.npz"
K_SWEEP_CSV  = RESULTS_DIR / "k_sweep_deterministic_odd_harmonic_MLP_20260522_155642.csv"
MI_SELECTED_DFT_CSV = RESULTS_DIR / "family_mi_selected_dft.csv"

HEADLINE_KEY = "odd_harmonic_MLP"


def save(fig, name):
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    # Copy to paper images/ so \includegraphics picks it up
    if PAPER_IMG_DIR.exists():
        shutil.copy(out, PAPER_IMG_DIR / name)
    print(f"→ {out} ({'copied' if PAPER_IMG_DIR.exists() else 'no copy'})")


def fig_mkt_vs(metric: str, ylabel: str, name: str):
    df = pd.read_csv(SUMMARY_CSV)
    LAMS = ["low_pass", "high_pass", "dft", "odd_harmonic"]
    LAM_LABEL = {
        "low_pass":   r"$\lambda_k\!=\!1/(k\!+\!1)$"+"\n(low-pass)",
        "high_pass":  r"$\lambda_k\!=\!1/(N\!-\!k)$"+"\n(high-pass)",
        "dft":        r"$\lambda_k\!=\!e^{i 2\pi k/N}$"+"\n(DFT-like)",
        "odd_harmonic": r"$\lambda_k\!=\!e^{i(2k\!+\!1)\pi/N}$"+"\n(odd-harm.)",
    }
    best_clf = []; best_val = []
    for lam in LAMS:
        sub = df[df["lambda"] == lam].sort_values(metric, ascending=False)
        best_val.append(sub.iloc[0][metric] * 100)
        best_clf.append(sub.iloc[0]["classifier"])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = ["#7FB3D5", "#76D7C4", "#F8C471", "#E67E22"]
    bars = ax.bar(range(len(LAMS)), best_val, color=colors, edgecolor="#2C3E50")
    for i, (b, v, c) in enumerate(zip(bars, best_val, best_clf)):
        ax.text(b.get_x() + b.get_width()/2, v + 0.08,
                f"{v:.2f}%\n{c}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(LAMS)))
    ax.set_xticklabels([LAM_LABEL[l] for l in LAMS], fontsize=9)
    ax.set_ylabel(ylabel + " (\%)")
    ymin = min(best_val) - 1.5
    ax.set_ylim(ymin, max(best_val) + 1.6)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, name)


def fig_roc_curve():
    data = np.load(PRED_NPZ)
    y_true  = data[f"{HEADLINE_KEY}_y_test"]
    y_proba = data[f"{HEADLINE_KEY}_y_proba"]
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    a = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot(fpr, tpr, color="#E67E22", lw=2.2, label=f"MLP + odd-harm. (AUC = {a:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("ROC curve")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    save(fig, "roc_curve.png")


def fig_confusion_matrix():
    data = np.load(PRED_NPZ)
    y_true = data[f"{HEADLINE_KEY}_y_test"]
    y_pred = data[f"{HEADLINE_KEY}_y_pred"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(cm, cmap="Oranges", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black",
                    fontsize=15, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["BKL\n(predicted)", "BCC\n(predicted)"])
    ax.set_yticklabels(["BKL\n(true)", "BCC\n(true)"])
    ax.set_title("Confusion matrix — MLP + odd-harmonic")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save(fig, "confusion_matrix.png")


def fig_pr_curve():
    data = np.load(PRED_NPZ)
    y_true  = data[f"{HEADLINE_KEY}_y_test"]
    y_proba = data[f"{HEADLINE_KEY}_y_proba"]
    p, r, _ = precision_recall_curve(y_true, y_proba)
    a = auc(r, p)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot(r, p, color="#27AE60", lw=2.2, label=f"MLP + odd-harm. (AP = {a:.4f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("Precision--Recall curve")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    save(fig, "pr_curve.png")


def fig_prob_distribution():
    data = np.load(PRED_NPZ)
    y_true  = data[f"{HEADLINE_KEY}_y_test"]
    y_proba = data[f"{HEADLINE_KEY}_y_proba"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bins = np.linspace(0, 1, 31)
    ax.hist(y_proba[y_true == 0], bins=bins, color="#3498DB", alpha=0.7,
            edgecolor="#1F4E79", label="BKL (true)")
    ax.hist(y_proba[y_true == 1], bins=bins, color="#E74C3C", alpha=0.7,
            edgecolor="#7B241C", label="BCC (true)")
    ax.set_xlabel("Predicted probability of BCC")
    ax.set_ylabel("Count")
    ax.set_title("Predicted-probability distribution")
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    save(fig, "prob_distribution.png")


def fig_k_sweep():
    df = pd.read_csv(K_SWEEP_CSV)
    fig, ax1 = plt.subplots(figsize=(6.8, 4.2))
    ax2 = ax1.twinx()
    ax1.plot(df["k_actual"], df["accuracy"] * 100, "o-", color="#E67E22",
             lw=2.0, label="Accuracy")
    ax2.plot(df["k_actual"], df["auc"] * 100, "s--", color="#2C3E50",
             lw=1.7, label="AUC")
    # Highlight k=360
    k_best = 360
    acc_best = df[df["k_actual"] == k_best]["accuracy"].iloc[0] * 100
    ax1.axvline(k_best, color="grey", ls=":", alpha=0.6)
    ax1.annotate(f"chosen $k=360$\nacc={acc_best:.2f}%",
                 xy=(k_best, acc_best), xytext=(k_best + 350, acc_best - 1.5),
                 fontsize=9, arrowprops=dict(arrowstyle="->", color="grey"))
    ax1.set_xlabel("Number of features kept ($k$)")
    ax1.set_ylabel("Test accuracy (\%)", color="#E67E22")
    ax2.set_ylabel("Test AUC (\%)", color="#2C3E50")
    ax1.set_xscale("log")
    ax1.set_xticks([100, 200, 360, 500, 800, 1200, 2000, 2509])
    ax1.set_xticklabels(["100", "200", "360", "500", "800", "1200", "2000", "2509"])
    ax1.tick_params(axis="y", colors="#E67E22")
    ax2.tick_params(axis="y", colors="#2C3E50")
    ax1.grid(linestyle=":", alpha=0.5)
    ax1.set_axisbelow(True)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower center", fontsize=10, ncol=2)
    ax1.set_title("$k$-sensitivity sweep — MLP + odd-harmonic")
    fig.tight_layout()
    save(fig, "k_sweep_curve.png")


def fig_mkt_feature_dominance():
    """Per-family scatter of MI scores for the top-360 features under λ=DFT."""
    df = pd.read_csv(MI_SELECTED_DFT_CSV)
    # Match the family order used in the paper text
    families = ["MKT", "Color", "Texture", "Geometric", "CNN"]
    palette = {"MKT": "#E67E22", "Color": "#3498DB", "Texture": "#27AE60",
                "Geometric": "#8E44AD", "CNN": "#7F8C8D"}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    rng = np.random.default_rng(42)
    for i, fam in enumerate(families):
        sub = df[df["family"] == fam]
        x = i + (rng.random(len(sub)) - 0.5) * 0.45  # jitter
        ax.scatter(x, sub["mi"], color=palette[fam], alpha=0.55, s=22,
                   edgecolors="none", label=f"{fam} (n={len(sub)})")
        med = sub["mi"].median()
        ax.hlines(med, i - 0.35, i + 0.35, color=palette[fam], lw=2.5)
        ax.text(i, med + 0.012, f"med={med:.3f}", ha="center", fontsize=9,
                color=palette[fam], fontweight="bold")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families)
    ax.set_ylabel("Mutual information with class label")
    ax.set_title(r"Per-family MI of the top-360 selected features ($\lambda$=DFT)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    save(fig, "mkt_feature_dominance.png")


def main():
    fig_mkt_vs("accuracy", "Best per-classifier accuracy", "mkt_vs_accuracy.png")
    fig_mkt_vs("auc",      "Best per-classifier AUC",      "mkt_vs_auc.png")
    fig_roc_curve()
    fig_confusion_matrix()
    fig_pr_curve()
    fig_prob_distribution()
    fig_k_sweep()
    fig_mkt_feature_dominance()
    print("All figures regenerated.")


if __name__ == "__main__":
    main()
