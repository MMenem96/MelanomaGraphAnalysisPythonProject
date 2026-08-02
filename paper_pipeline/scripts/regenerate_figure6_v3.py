"""Figure 6 v3 — single vertical workflow with numbered stages.

Goal: a cleaner, more publication-ready proposed-framework figure than v2.

Design changes vs v2:
  * Single vertical narrative (no a/b split) -- the eye reads top to bottom.
  * Six numbered stages with stage badges on the left margin:
      (1) Input  (2) Preprocessing  (3) MKT  (4) Fusion
      (5) Data + Training  (6) Headline result
  * Each stage in its own light-tinted band so the structure is obvious.
  * Larger preprocessing thumbnails, 4 MKT magnitude maps with the
    odd-harmonic configuration highlighted in gold throughout.
  * Manhattan-routed arrows (horizontal -> vertical, no diagonals).
  * Headline result as a prominent gold "card" at the bottom showing
    every metric, both CIs, McNemar callout, and test-set composition.

Run:
    cd /Users/mmoniem96/Desktop/Work/Master/Practical\\ Project/MelanomaGraphAnalysisV3
    python paper_pipeline/scripts/regenerate_figure6_v3.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Arc, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Circle,
)
from PIL import Image
from scipy.special import comb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_pipeline.pipeline.mkt_lambda import get_lambda_fn


OUT_PIPELINE = (PROJECT_ROOT / "paper_pipeline" / "output" / "figures"
                / "proposed_framework_figure_v3.png")
OUT_PAPER    = Path("/Users/mmoniem96/Desktop/Work/Master/"
                    "Mohamed_Shoieb_Master_Paper_ElsevierV7/images/"
                    "proposed_framework_figure_v3.png")
SAMPLE_BCC_PATH = (PROJECT_ROOT / "data" / "canonical" / "bcc_segmented"
                    / "ISIC_0024331.png")
PAPER_IMG_DIR = OUT_PAPER.parent
PREPROC_ORIG  = PAPER_IMG_DIR / "preprocessing_original_image.png"
PREPROC_MASK  = PAPER_IMG_DIR / "preprocessing_binary_mask.png"
PREPROC_ROI   = PAPER_IMG_DIR / "preprocessing_roi.png"
PREPROC_HAIR  = PAPER_IMG_DIR / "preprocessing_hair_detection.png"
PREPROC_RES   = PAPER_IMG_DIR / "preprocessing_result.png"


# ARS palette (matches paper preamble)
NAVY       = "#2C3E50"
NAVY_TINT  = "#F4F6F8"
NAVY_BAND  = "#EAEEF2"        # slightly darker tint for stage bands
GOLD       = "#E67E22"
GOLD_TINT  = "#FDEBD0"
GOLD_DEEP  = "#B7610F"
GREY_LT    = "#ECF0F1"
TEXT_COL   = "#34495E"
TEXT_SOFT  = "#5D6D7E"
WHITE      = "#FFFFFF"

N_MKT = 64
P_MKT = 0.5


# ---------- Krawtchouk MKT (same numerics as v2) -----------------------------
def _krawtchouk_poly(n, x, N, p):
    s = 0.0
    for j in range(n + 1):
        c1 = comb(x, j, exact=False)
        c2 = comb(N - x, n - j, exact=False)
        power = (p / (1 - p)) ** j if j > 0 else 1.0
        power = max(1e-10, min(power, 1e10))
        term = ((-1) ** j) * c1 * c2 * power
        if np.isfinite(term):
            s += term
    return s


def _normalized_krawtchouk(n, x, N, p):
    w_x  = comb(N, x, exact=False) * (p ** x) * ((1 - p) ** (N - x))
    norm = np.sqrt(comb(N, n, exact=False) * (p ** n) * ((1 - p) ** (N - n)))
    K = _krawtchouk_poly(n, x, N, p)
    if norm > 1e-10 and w_x > 0:
        return K * np.sqrt(w_x) / norm
    return 0.0


def build_K0(N: int, p: float) -> np.ndarray:
    K = np.zeros((N, N))
    for n in range(N):
        for x in range(N):
            K[n, x] = _normalized_krawtchouk(n, x, N - 1, p)
    Q, _ = np.linalg.qr(K.T)
    return Q.T


def apply_2d_mdfkt(K0, Lambda, f):
    temp = K0 @ f
    temp = Lambda[:, None] * temp
    return temp @ K0.T


def mkt_magnitude_map(image_gray_64: np.ndarray, lam_name: str,
                       K0: np.ndarray) -> np.ndarray:
    lam_fn = get_lambda_fn(lam_name)
    Lambda = lam_fn(image_gray_64.shape[0]).astype(complex)
    Y = apply_2d_mdfkt(K0, Lambda, image_gray_64.astype(float))
    mag = np.abs(Y)
    mag = np.log1p(mag)
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag


# ---------- Drawing primitives ----------------------------------------------
def rbox(ax, x, y, w, h, *, fill=WHITE, edge=NAVY, lw=1.0, rounding=0.10,
          alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                 boxstyle=f"round,pad=0,rounding_size={rounding}",
                                 linewidth=lw, edgecolor=edge, facecolor=fill,
                                 alpha=alpha))


def band(ax, x, y, w, h, *, fill=NAVY_BAND, alpha=1.0, rounding=0.18):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                 boxstyle=f"round,pad=0,rounding_size={rounding}",
                                 linewidth=0, facecolor=fill, alpha=alpha))


def text(ax, x, y, txt, *, fs=10, color=TEXT_COL, weight="normal",
         ha="center", va="center", family="serif"):
    ax.text(x, y, txt, ha=ha, va=va, color=color, fontsize=fs,
            fontweight=weight, family=family)


def arrow(ax, p1, p2, *, color=NAVY, lw=1.2, style="-|>", mut=14,
          connection="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=color,
                                  linewidth=lw, mutation_scale=mut,
                                  connectionstyle=connection))


def L_arrow(ax, p1, p2, *, color=NAVY, lw=1.2, mut=14, rev=False):
    """Manhattan-routed arrow: go horizontal first then vertical."""
    mid = (p2[0], p1[1]) if not rev else (p1[0], p2[1])
    ax.plot([p1[0], mid[0]], [p1[1], mid[1]], color=color, lw=lw,
            solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(mid, p2, arrowstyle="-|>", color=color,
                                  linewidth=lw, mutation_scale=mut))


def imshow_inset(ax, image, x, y, w, h, *, cmap=None, border=NAVY, lw=1.2):
    iax = ax.inset_axes([x, y, w, h], transform=ax.transData)
    if cmap is None:
        iax.imshow(image)
    else:
        iax.imshow(image, cmap=cmap)
    iax.set_xticks([]); iax.set_yticks([])
    for s in iax.spines.values():
        s.set_edgecolor(border); s.set_linewidth(lw)


def stage_badge(ax, x, y, n, *, r=0.30, color=NAVY):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor=color,
                         linewidth=1.2))
    text(ax, x, y, str(n), fs=11, color=WHITE, weight="bold")


def db_cylinder(ax, cx, cy, w, h, *, fill=NAVY_TINT, edge=NAVY, lw=1.0):
    eh = w * 0.32
    ax.add_patch(Polygon([(cx - w/2, cy - h/2 + eh/2),
                            (cx + w/2, cy - h/2 + eh/2),
                            (cx + w/2, cy + h/2 - eh/2),
                            (cx - w/2, cy + h/2 - eh/2)],
                          closed=True, facecolor=fill, edgecolor="none"))
    ax.add_patch(Ellipse((cx, cy + h/2 - eh/2), w, eh,
                          facecolor=fill, edgecolor=edge, linewidth=lw))
    ax.add_patch(Arc((cx, cy - h/2 + eh/2), w, eh,
                      theta1=180, theta2=360,
                      edgecolor=edge, linewidth=lw))
    ax.plot([cx - w/2, cx - w/2],
            [cy - h/2 + eh/2, cy + h/2 - eh/2],
            color=edge, linewidth=lw)
    ax.plot([cx + w/2, cx + w/2],
            [cy - h/2 + eh/2, cy + h/2 - eh/2],
            color=edge, linewidth=lw)
    for off in [0.18, -0.18]:
        ax.add_patch(Arc((cx, cy + off * h), w, eh,
                          theta1=180, theta2=360,
                          edgecolor=edge, linewidth=lw * 0.4, alpha=0.5))


# ---------- Stage helpers ----------------------------------------------------
def _load(path, mode="RGB", fallback=None):
    try:
        return np.array(Image.open(path).convert(mode))
    except Exception:
        return fallback


# ---------- Main layout ------------------------------------------------------
def render(K0, lesion_gray_64):
    fig = plt.figure(figsize=(14, 19))
    ax  = fig.add_subplot(111)
    ax.set_xlim(0, 14); ax.set_ylim(0, 19)
    ax.set_aspect("equal"); ax.axis("off")

    # ===== Title bar =====
    band(ax, 0.4, 18.0, 13.2, 0.75, fill=NAVY, rounding=0.18)
    text(ax, 7.0, 18.38,
         "Proposed BCC vs BKL Pipeline  -  hybrid MKT + handcrafted + ResNet-50",
         fs=13, color=WHITE, weight="bold")

    # ===== Stage band helper =====
    def stage_band(ystart, yh, n, title):
        band(ax, 0.4, ystart, 13.2, yh, fill=NAVY_BAND, alpha=0.55,
             rounding=0.18)
        stage_badge(ax, 0.95, ystart + yh - 0.45, n)
        text(ax, 1.45, ystart + yh - 0.45, title,
             fs=11.5, color=NAVY, weight="bold", ha="left")

    # ---------- Stage 1 — Input + Preprocessing ----------
    y1, h1 = 14.4, 3.30
    stage_band(y1, h1, 1, "Input  &  Preprocessing")

    # 5 preprocessing thumbnails + arrows
    steps = [
        (_load(PREPROC_ORIG, "RGB"),  None,   NAVY, "Dermoscopy",  r"$224{\times}224{\times}3$", False),
        (_load(PREPROC_MASK, "L"),    "gray", NAVY, "Lesion mask", r"$M \in \{0,1\}$",            False),
        (_load(PREPROC_ROI,  "RGB"),  None,   NAVY, "ROI",         r"$I \odot M$",                False),
        (_load(PREPROC_HAIR, "RGB"),  None,   NAVY, "Hair map",    r"Top/Black-hat",              False),
        (_load(PREPROC_RES,  "RGB"),  None,   GOLD, "Cleaned",     r"Telea inpaint",              True),
    ]
    th_w = 1.7; th_h = 1.7
    gap = 0.42
    total_w = 5 * th_w + 4 * gap
    x0 = (14 - total_w) / 2
    centres = []
    y_thumbs = y1 + 0.85
    for i, (img, cmap, border, title, sub, head) in enumerate(steps):
        if img is None:
            img = np.zeros((10, 10, 3), dtype=np.uint8)
        lw = 2.4 if head else 1.0
        imshow_inset(ax, img, x0, y_thumbs, th_w, th_h,
                      cmap=cmap, border=border, lw=lw)
        text(ax, x0 + th_w/2, y_thumbs - 0.27, title,
             fs=9, weight="bold",
             color=GOLD if head else NAVY)
        text(ax, x0 + th_w/2, y_thumbs - 0.52, sub,
             fs=8, color=TEXT_SOFT)
        centres.append(x0 + th_w/2)
        if i < len(steps) - 1:
            arrow(ax,
                  (x0 + th_w + 0.02, y_thumbs + th_h/2),
                  (x0 + th_w + gap - 0.02, y_thumbs + th_h/2),
                  lw=1.0)
        x0 += th_w + gap

    # ---------- Stage 2 — MKT transform ----------
    y2, h2 = 10.2, 4.0
    stage_band(y2, h2, 2, "MKT transform   "
                          r"$\boldsymbol{\Psi} = \mathcal{K}_\Lambda I \mathcal{K}_\Lambda^\top$,  "
                          r"$N=64,\ p=0.5$")

    # 4 MKT outputs in a row
    lam_specs = [
        ("low_pass",     "low-pass",   r"$\lambda_k=1/(k{+}1)$",        False),
        ("high_pass",    "high-pass",  r"$\lambda_k=1/(N{-}k)$",         False),
        ("dft",          "DFT-like",   r"$\lambda_k=e^{i 2\pi k/N}$",   False),
        ("odd_harmonic", r"odd-harmonic  $\star$",
                                       r"$\lambda_k=e^{i(2k{+}1)\pi/N}$", True),
    ]
    mkt_w = 1.85; mkt_h = 1.85
    mkt_gap = 0.65
    total_mw = 4 * mkt_w + 3 * mkt_gap
    mx0 = (14 - total_mw) / 2
    y_mkt = y2 + 1.05
    mkt_centres = []
    for (lam, label, eq, head) in lam_specs:
        mag = mkt_magnitude_map(lesion_gray_64, lam, K0)
        lw = 2.8 if head else 1.0
        border = GOLD if head else NAVY
        imshow_inset(ax, mag, mx0, y_mkt, mkt_w, mkt_h,
                      cmap="magma", border=border, lw=lw)
        text(ax, mx0 + mkt_w/2, y_mkt - 0.27, label,
             fs=9.5, weight="bold" if head else "normal",
             color=GOLD if head else NAVY)
        text(ax, mx0 + mkt_w/2, y_mkt - 0.50, eq,
             fs=8.5, color=TEXT_SOFT)
        text(ax, mx0 + mkt_w/2, y_mkt - 0.72,
             "76 features",
             fs=8, color=GOLD_DEEP if head else TEXT_SOFT,
             weight="bold" if head else "normal")
        mkt_centres.append((mx0 + mkt_w/2, y_mkt))
        mx0 += mkt_w + mkt_gap

    # arrow from cleaned thumbnail (centre 4 of stage 1) down to MKT operator
    cleaned_x = centres[4]
    L_arrow(ax,
            (cleaned_x, y1 + 0.85 - 0.05),
            (mkt_centres[3][0], y_mkt + mkt_h + 0.18),
            lw=1.0, color=GOLD)

    # ---------- Stage 3 — Feature fusion ----------
    y3, h3 = 8.4, 1.75
    stage_band(y3, h3, 3,
               "Feature fusion   "
               r"$\mathbf{x} \in \mathbb{R}^{2{,}509}$")

    # 5 family chips concatenated, gold for MKT
    families = [
        ("Geometric",   "18",     NAVY_TINT, NAVY),
        ("Color",       "138",    NAVY_TINT, NAVY),
        ("Texture",     "229",    NAVY_TINT, NAVY),
        ("MKT $\\star$", "76",    GOLD_TINT, GOLD),
        ("ResNet-50",   r"2{,}048", NAVY_TINT, NAVY),
    ]
    chip_h = 0.95
    weights = [1.0, 1.0, 1.0, 1.0, 2.4]      # ResNet wider to reflect 2048 dims
    total_w_units = sum(weights)
    chip_total_w = 12.0
    base_w = chip_total_w / total_w_units
    fx0 = (14 - chip_total_w) / 2
    fy0 = y3 + 0.25
    for (label, count, fill, edge), wt in zip(families, weights):
        w = base_w * wt
        rbox(ax, fx0, fy0, w, chip_h, fill=fill, edge=edge, lw=1.0)
        text(ax, fx0 + w/2, fy0 + 0.62, label, fs=9.5, weight="bold",
             color=edge)
        text(ax, fx0 + w/2, fy0 + 0.30, count, fs=10, weight="bold",
             color=edge)
        fx0 += w

    # arrow from middle of MKT row into the fusion strip
    arrow(ax, (7.0, y2 + 0.4), (7.0, y3 + h3 - 0.18),
          color=GOLD, lw=1.3)

    # ---------- Stage 4 — Data + Training ----------
    y4, h4 = 4.0, 4.30
    stage_band(y4, h4, 4, "Data + Training protocol")

    # HAM10000 cylinder
    cyl_cx = 1.75; cyl_cy = y4 + 2.55
    db_cylinder(ax, cyl_cx, cyl_cy, 1.4, 1.55, fill=NAVY_TINT, edge=NAVY)
    text(ax, cyl_cx, cyl_cy + 0.10, "HAM10000", fs=9.5, weight="bold",
         color=NAVY)
    text(ax, cyl_cx, cyl_cy - 0.15, r"514 BCC + 1{,}099 BKL", fs=8)
    text(ax, cyl_cx, cyl_cy - 0.38, r"= 1{,}613 images",
         fs=9, color=NAVY, weight="bold")
    text(ax, cyl_cx, cyl_cy - 0.92,
         r"dx-filter $\cap$ HAM10000 masks", fs=7.3, color=TEXT_SOFT)

    # Stratified split box
    sb_x = 4.0; sb_y = y4 + 2.0; sb_w = 2.3; sb_h = 1.4
    rbox(ax, sb_x, sb_y, sb_w, sb_h, fill=WHITE, edge=NAVY, lw=1.0)
    text(ax, sb_x + sb_w/2, sb_y + 1.10, "Stratified",
         fs=9.5, weight="bold", color=NAVY)
    text(ax, sb_x + sb_w/2, sb_y + 0.85, "80/20 split", fs=9, color=NAVY)
    text(ax, sb_x + sb_w/2, sb_y + 0.55,
         r"by image id, seed 42", fs=7.5, color=TEXT_SOFT)
    text(ax, sb_x + sb_w/2, sb_y + 0.25,
         r"train 1{,}290 | test 323",
         fs=8.5, weight="bold", color=NAVY)
    arrow(ax, (cyl_cx + 0.75, cyl_cy), (sb_x - 0.05, sb_y + sb_h/2))

    # Train branch (top) + Aug
    tb_x = 7.2; tb_y_train = y4 + 2.7; tb_y_test = y4 + 1.1
    tb_w = 2.4; tb_h = 1.05
    rbox(ax, tb_x, tb_y_train, tb_w, tb_h, fill=NAVY_TINT, edge=NAVY, lw=1.0)
    text(ax, tb_x + tb_w/2, tb_y_train + 0.78,
         r"Train  $n{=}1{,}290$", fs=9.5, weight="bold", color=NAVY)
    text(ax, tb_x + tb_w/2, tb_y_train + 0.50, r"411 BCC + 879 BKL", fs=8.5)
    text(ax, tb_x + tb_w/2, tb_y_train + 0.22,
         r"BCC h/v-flip $\to$ 879 BCC", fs=7.8, color=GOLD_DEEP, weight="bold")
    # Test branch (bottom)
    rbox(ax, tb_x, tb_y_test, tb_w, tb_h, fill=WHITE, edge=NAVY, lw=1.5)
    text(ax, tb_x + tb_w/2, tb_y_test + 0.78,
         r"Test  $n{=}323$", fs=9.5, weight="bold", color=NAVY)
    text(ax, tb_x + tb_w/2, tb_y_test + 0.50, r"103 BCC + 220 BKL", fs=8.5)
    text(ax, tb_x + tb_w/2, tb_y_test + 0.22,
         r"held out  -  touched once / model", fs=7.5, color=TEXT_SOFT)

    # split -> train & test
    L_arrow(ax, (sb_x + sb_w + 0.02, sb_y + sb_h/2),
                (tb_x - 0.02, tb_y_train + tb_h/2), lw=1.0)
    L_arrow(ax, (sb_x + sb_w + 0.02, sb_y + sb_h/2),
                (tb_x - 0.02, tb_y_test + tb_h/2), lw=1.0)

    # Selection box
    mi_x = 10.4; mi_y = y4 + 2.7; mi_w = 2.6; mi_h = 1.05
    rbox(ax, mi_x, mi_y, mi_w, mi_h, fill=NAVY_TINT, edge=NAVY, lw=1.0)
    text(ax, mi_x + mi_w/2, mi_y + 0.78,
         "SelectKBest (MI)", fs=9.5, weight="bold", color=NAVY)
    text(ax, mi_x + mi_w/2, mi_y + 0.50, r"360 / 2{,}509", fs=9.5,
         weight="bold", color=NAVY)
    text(ax, mi_x + mi_w/2, mi_y + 0.22,
         "RobustScaler, fit on train", fs=7.8, color=TEXT_SOFT)
    arrow(ax, (tb_x + tb_w + 0.02, tb_y_train + tb_h/2),
              (mi_x - 0.02, mi_y + mi_h/2))

    # train arrow with augmentation count
    text(ax, (tb_x + tb_w + mi_x) / 2, tb_y_train + tb_h/2 + 0.15,
         r"aug $\rightarrow 1{,}758$", fs=7.5, color=GOLD_DEEP,
         weight="bold")

    # Classifier block (bottom of stage 4)
    cl_x = 4.0; cl_y = y4 + 0.25; cl_w = 9.0; cl_h = 0.95
    rbox(ax, cl_x, cl_y, cl_w, cl_h, fill=WHITE, edge=NAVY, lw=1.0)
    text(ax, cl_x + cl_w/2, cl_y + 0.66,
         "9 classifiers, stratified 5-fold CV (dev only)",
         fs=9.5, weight="bold", color=NAVY)
    text(ax, cl_x + cl_w/2, cl_y + 0.35,
         "SVM (RBF) - LightGBM - CatBoost - Gradient Boosting - "
         "Extra Trees - KNN - Logistic Regression - "
         "$\\mathbf{MLP^{\\star}}$ - Deep DNN",
         fs=8.3, color=NAVY)
    arrow(ax, (mi_x + mi_w/2, mi_y), (cl_x + cl_w/2, cl_y + cl_h),
          color=NAVY, lw=1.1)
    # test arrow joins the classifier block from the left
    arrow(ax, (tb_x + tb_w + 0.02, tb_y_test + tb_h/2),
              (cl_x + cl_w * 0.35, cl_y + cl_h),
          color=TEXT_SOFT, lw=1.0, connection="arc3,rad=0.15")

    # arrow from fusion strip down into the data stage
    arrow(ax, (7.0, y3 + 0.10), (7.0, y4 + h4 - 0.18),
          color=NAVY, lw=1.2)

    # ---------- Stage 5 — Headline result ----------
    y5, h5 = 0.3, 3.45
    band(ax, 0.4, y5, 13.2, h5, fill=GOLD, rounding=0.18)
    stage_badge(ax, 0.95, y5 + h5 - 0.45, 5, color=NAVY)
    text(ax, 1.45, y5 + h5 - 0.45, "Headline result",
         fs=12, color=NAVY, weight="bold", ha="left")

    # row 1: model
    text(ax, 7.0, y5 + h5 - 0.95,
         r"MLP  $\oplus$  $\lambda_k = e^{i(2k+1)\pi/N}$  "
         r"(odd-harmonic, $p=0.5$, $N=64$)",
         fs=11.5, color=WHITE, weight="bold")

    # metrics grid
    metric_xs = [1.6, 3.7, 5.8, 7.9, 10.0, 12.1]
    metric_w  = 1.85; metric_h = 0.95
    keys = ["Acc",     "Sens",   "Spec",   "Prec",   "F1",     "AUC"]
    vals = ["93.50%",  "91.26%", "94.55%", "88.68%", "89.95%", "96.95%"]
    metric_y = y5 + 1.20
    for x, k, v in zip(metric_xs, keys, vals):
        rbox(ax, x - metric_w/2, metric_y, metric_w, metric_h,
             fill=WHITE, edge=NAVY, lw=1.0)
        text(ax, x, metric_y + metric_h * 0.72, k,
             fs=10, color=NAVY, weight="bold")
        text(ax, x, metric_y + metric_h * 0.30, v,
             fs=12.5, color=NAVY, weight="bold")

    # row 2: CI + McNemar + test composition
    text(ax, 7.0, y5 + 0.85,
         r"95\% bootstrap CI (10{,}000 resamples):  "
         r"Acc $\in [90.71, 95.98]$,  AUC $\in [95.13, 98.51]$",
         fs=9.5, color=WHITE)
    text(ax, 7.0, y5 + 0.52,
         r"McNemar vs runner-up (LightGBM + low-pass, $93.19\%$):  "
         r"exact $p = 1.0$",
         fs=9.5, color=WHITE)
    text(ax, 7.0, y5 + 0.22,
         r"evaluated once on 323-image held-out test set "
         r"(103 BCC + 220 BKL)",
         fs=9.5, color=WHITE, weight="bold")
    # arrow from classifier block into headline card
    arrow(ax, (7.0, cl_y), (7.0, y5 + h5 - 0.05),
          color=NAVY, lw=1.4)

    return fig


def main():
    img = Image.open(SAMPLE_BCC_PATH).convert("RGB")
    gray = img.convert("L").resize((N_MKT, N_MKT))
    lesion_gray_64 = np.array(gray, dtype=float)

    print("Building Krawtchouk K0 matrix (~5 s)...")
    K0 = build_K0(N_MKT, P_MKT)
    print(f"K0 shape: {K0.shape}")

    print("Composing Figure 6 v3...")
    fig = render(K0, lesion_gray_64)
    plt.tight_layout(pad=0.6)
    fig.savefig(OUT_PIPELINE, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {OUT_PIPELINE}")
    if OUT_PAPER.parent.exists():
        shutil.copy(OUT_PIPELINE, OUT_PAPER)
        print(f"-> {OUT_PAPER}")


if __name__ == "__main__":
    main()
