"""Regenerate Figure 6 (proposed_framework_figure.png) with the new headline
(MLP + odd-harmonic, 93.50% acc, 96.95% AUC).

Two-panel layout matching the existing figure:
  (a) Per-image feature extraction: image → mask → hair-aware inpainting →
      four MKT λ configurations → concatenate with handcrafted + ResNet-50 →
      2,509-d feature vector
  (b) Dataset-level protocol: 1,613-image set → stratified 80/20 split →
      BCC-only augmentation → SelectKBest 360/2,509 → 9-classifier sweep →
      headline metric block
"""
from __future__ import annotations
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.transforms import Bbox


OUT_PIPELINE = Path("/Users/mmoniem96/Desktop/Work/Master/Practical Project/"
                    "MelanomaGraphAnalysisV3/paper_pipeline/output/figures/"
                    "proposed_framework_figure.png")
OUT_PAPER    = Path("/Users/mmoniem96/Desktop/Work/Master/"
                    "Mohamed_Shoieb_Master_Paper_ElsevierV7/images/"
                    "proposed_framework_figure.png")


# Palette consistent with paper preamble (ARS palette)
NAVY      = "#2C3E50"
NAVY_TINT = "#F4F6F8"
GOLD      = "#E67E22"
GOLD_TINT = "#FDEBD0"
GREY_LT   = "#ECF0F1"
TEXT_COL  = "#34495E"


def rounded_box(ax, x, y, w, h, *, fill=NAVY_TINT, edge=NAVY, lw=1.0, alpha=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                          linewidth=lw, edgecolor=edge, facecolor=fill, alpha=alpha)
    ax.add_patch(box)


def text(ax, x, y, txt, *, fs=10, color=TEXT_COL, weight="normal",
         ha="center", va="center"):
    ax.text(x, y, txt, ha=ha, va=va, color=color, fontsize=fs,
            fontweight=weight, family="serif")


def arrow(ax, p1, p2, *, color=NAVY, lw=1.3, style="-|>", mut=14):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=color,
                                  linewidth=lw, mutation_scale=mut))


def panel_a(ax):
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.5)
    ax.set_aspect("equal"); ax.axis("off")

    text(ax, 0.2, 7.1, "(a)  Per-image feature extraction", fs=12, color=NAVY,
         weight="bold", ha="left")

    # Step 1: dermoscopic input
    rounded_box(ax, 0.2, 5.4, 1.7, 1.2, fill=GREY_LT)
    text(ax, 1.05, 6.4, "Dermoscopic", fs=9, weight="bold")
    text(ax, 1.05, 6.05, "image", fs=9)
    text(ax, 1.05, 5.55, r"$224\times224\times3$", fs=8, color=NAVY)

    # Step 2: mask × image
    rounded_box(ax, 2.3, 5.4, 1.6, 1.2, fill=NAVY_TINT)
    text(ax, 3.1, 6.4, "Lesion mask", fs=9, weight="bold")
    text(ax, 3.1, 6.05, "(HAM10000)", fs=8)
    text(ax, 3.1, 5.6, r"$I \odot M$", fs=8.5, color=NAVY)

    # Step 3: artifact removal (Telea)
    rounded_box(ax, 4.3, 5.4, 1.8, 1.2, fill=NAVY_TINT)
    text(ax, 5.2, 6.4, "Hair / artifact", fs=9, weight="bold")
    text(ax, 5.2, 6.05, "removal", fs=9)
    text(ax, 5.2, 5.6, "(Top-/Black-hat,", fs=8)
    text(ax, 5.2, 5.4, "Telea inpaint)", fs=8)

    # Step 4: branch
    arrow(ax, (1.9, 6.0), (2.3, 6.0))
    arrow(ax, (3.9, 6.0), (4.3, 6.0))
    arrow(ax, (6.1, 6.0), (6.5, 6.0))

    # MKT branch
    rounded_box(ax, 6.5, 5.2, 1.9, 1.7, fill=GOLD_TINT, edge=GOLD)
    text(ax, 7.45, 6.65, "MKT", fs=10, weight="bold", color=GOLD)
    text(ax, 7.45, 6.4, r"$\mathcal{K}_\Lambda\,I\,\mathcal{K}_\Lambda^\top$",
         fs=10, color=NAVY)
    text(ax, 7.45, 6.05, r"$N{=}64,\ p{=}0.5$", fs=8, color=NAVY)

    # Four eigenvalue configurations
    lam_x = [9.0, 9.0, 11.0, 11.0]
    lam_y = [6.7, 5.6, 6.7, 5.6]
    lam_lbl = [r"low-pass", r"high-pass", r"DFT-like",
               r"odd-harm. $\bigstar$"]
    lam_eq = [r"$1/(k\!+\!1)$", r"$1/(N\!-\!k)$",
              r"$e^{i 2\pi k/N}$", r"$e^{i(2k\!+\!1)\pi/N}$"]
    for i, (x, y, lbl, eq) in enumerate(zip(lam_x, lam_y, lam_lbl, lam_eq)):
        is_head = (i == 3)
        rounded_box(ax, x - 0.9, y - 0.45, 1.8, 0.9,
                     fill=GOLD if is_head else GREY_LT,
                     edge=GOLD if is_head else NAVY,
                     lw=1.8 if is_head else 1.0)
        text(ax, x, y + 0.15, lbl, fs=9,
             weight="bold" if is_head else "normal",
             color="white" if is_head else TEXT_COL)
        text(ax, x, y - 0.15, eq, fs=8.5,
             color="white" if is_head else NAVY)
    # Connect MKT to all 4 configs
    arrow(ax, (8.4, 6.05), (8.1, 6.7))
    arrow(ax, (8.4, 6.05), (8.1, 5.6))
    arrow(ax, (8.4, 6.05), (10.1, 6.7))
    arrow(ax, (8.4, 6.05), (10.1, 5.6))

    # Handcrafted branch
    rounded_box(ax, 6.5, 3.5, 5.5, 1.0, fill=NAVY_TINT)
    text(ax, 9.25, 4.2, "Handcrafted descriptors", fs=10, weight="bold",
         color=NAVY)
    text(ax, 9.25, 3.85,
         r"Geometric (18) $\cdot$ Color (138) $\cdot$ Texture (229)",
         fs=8.5)

    # CNN branch
    rounded_box(ax, 6.5, 2.0, 5.5, 1.0, fill=NAVY_TINT)
    text(ax, 9.25, 2.7, "Frozen ResNet-50", fs=10, weight="bold", color=NAVY)
    text(ax, 9.25, 2.35,
         r"ImageNet weights $\rightarrow$ 2{,}048-d descriptor",
         fs=8.5)

    arrow(ax, (5.2, 5.4), (5.2, 4.5))
    arrow(ax, (5.2, 4.0), (6.5, 4.0))
    arrow(ax, (5.2, 4.0), (5.2, 3.0))
    arrow(ax, (5.2, 2.5), (6.5, 2.5))

    # Concat → hybrid vector
    rounded_box(ax, 0.6, 0.6, 11, 1.0, fill=GOLD_TINT, edge=GOLD, lw=1.5)
    text(ax, 6.1, 1.25, "Hybrid feature vector", fs=11, weight="bold",
         color=GOLD)
    text(ax, 6.1, 0.85,
         r"$\mathbf{x} \in \mathbb{R}^{2{,}509}$ "
         r"$=$ MKT (76) $\oplus$ Handcrafted (385) "
         r"$\oplus$ ResNet-50 (2{,}048)",
         fs=9)
    arrow(ax, (9.0, 5.15), (6.1, 1.65))
    arrow(ax, (9.25, 3.5), (6.1, 1.65))
    arrow(ax, (9.25, 2.0), (6.1, 1.65))


def panel_b(ax):
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.5)
    ax.set_aspect("equal"); ax.axis("off")

    text(ax, 0.2, 7.1, "(b)  Dataset-level leak-free protocol", fs=12,
         color=NAVY, weight="bold", ha="left")

    # Step 1 dataset
    rounded_box(ax, 0.4, 5.4, 2.4, 1.2, fill=NAVY_TINT)
    text(ax, 1.6, 6.35, "HAM10000 subset", fs=9.5, weight="bold")
    text(ax, 1.6, 6.0, r"514 BCC $+$ 1{,}099 BKL", fs=8.5)
    text(ax, 1.6, 5.6, r"$= 1{,}613$ images", fs=8.5, color=NAVY)

    # Step 2 split
    rounded_box(ax, 3.4, 5.4, 2.6, 1.2, fill=NAVY_TINT)
    text(ax, 4.7, 6.35, "Stratified $80/20$ split", fs=9.5, weight="bold")
    text(ax, 4.7, 6.0, "on image identifiers", fs=8.5)
    text(ax, 4.7, 5.6, r"train $1{,}290$  |  test $323$", fs=8.5,
         color=NAVY)

    # Step 3 augmentation
    rounded_box(ax, 6.6, 5.4, 2.7, 1.2, fill=NAVY_TINT)
    text(ax, 7.95, 6.35, "BCC-only flip augmentation", fs=9.5, weight="bold")
    text(ax, 7.95, 6.0, r"411 $\rightarrow$ 879 BCC (orig + h/v-flip)",
         fs=8.5)
    text(ax, 7.95, 5.6, r"balanced train: $1{,}758$",
         fs=8.5, color=NAVY)

    # Step 4 selection + scaling
    rounded_box(ax, 9.9, 5.4, 1.9, 1.2, fill=NAVY_TINT)
    text(ax, 10.85, 6.35, "SelectKBest", fs=9.5, weight="bold")
    text(ax, 10.85, 6.0, r"$360 / 2{,}509$", fs=8.5)
    text(ax, 10.85, 5.6, "RobustScaler", fs=8.5, color=NAVY)

    arrow(ax, (2.8, 6.0), (3.4, 6.0))
    arrow(ax, (6.0, 6.0), (6.6, 6.0))
    arrow(ax, (9.3, 6.0), (9.9, 6.0))

    # Classifier sweep box
    rounded_box(ax, 0.4, 3.4, 11.4, 1.4, fill=NAVY_TINT)
    text(ax, 6.1, 4.5, r"$9$-classifier sweep on the $360$-feature representation",
         fs=10, weight="bold", color=NAVY)
    text(ax, 6.1, 4.1,
         "SVM (RBF), LightGBM, CatBoost, Gradient Boosting, Extra Trees,",
         fs=8.5)
    text(ax, 6.1, 3.75,
         r"KNN, Logistic Regression, MLP$^\bigstar$, Deep DNN  "
         r"$\cdot$  Stratified $5$-fold CV  $\cdot$  random_state $= 42$",
         fs=8.5)
    arrow(ax, (10.85, 5.4), (6.1, 4.8))

    # Headline result block
    rounded_box(ax, 1.0, 0.4, 10.2, 2.6, fill=GOLD, edge=NAVY, lw=2.0)
    text(ax, 6.1, 2.65, "HEADLINE", fs=11.5, weight="bold", color="white")
    text(ax, 6.1, 2.30,
         r"MLP  $\oplus$  $\lambda_k = e^{i(2k+1)\pi/N}$  (odd-harmonic, "
         r"$p = 0.5$, $N = 64$)",
         fs=10.5, weight="bold", color="white")
    # Metrics in grid
    metric_y = 1.55
    metric_pairs = [
        ("Acc",  "93.50%"),
        ("Sens", "91.26%"),
        ("Spec", "94.55%"),
        ("Prec", "88.68%"),
        ("F1",   "89.95%"),
        ("AUC",  "96.95%"),
    ]
    xs = [1.7, 3.4, 5.1, 6.8, 8.5, 10.2]
    for x, (k, v) in zip(xs, metric_pairs):
        text(ax, x, 1.75, k, fs=10.5, color="white", weight="bold")
        text(ax, x, 1.32, v, fs=12, color="white", weight="bold")
    text(ax, 6.1, 0.85, r"on a $323$-image held-out test set (103 BCC + 220 BKL)",
         fs=9, color="white", weight="bold")
    text(ax, 6.1, 0.55,
         r"95\% bootstrap CIs from $10{,}000$ resamples; "
         r"McNemar's vs runner-up $p = 1.0$",
         fs=8.5, color="white")
    arrow(ax, (6.1, 3.4), (6.1, 3.05))


def main():
    fig, axes = plt.subplots(2, 1, figsize=(14, 12.5))
    panel_a(axes[0])
    panel_b(axes[1])
    plt.tight_layout()
    fig.savefig(OUT_PIPELINE, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    if OUT_PAPER.parent.exists():
        shutil.copy(OUT_PIPELINE, OUT_PAPER)
        print(f"→ {OUT_PIPELINE}")
        print(f"→ {OUT_PAPER}")


if __name__ == "__main__":
    main()
