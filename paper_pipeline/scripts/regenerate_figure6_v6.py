"""Figure 6 v6 - PowerPoint-style methodology figure rebuilt in matplotlib.

Inspired by the reference figure the user shared. The aesthetic:
  * Database cylinder + stacked tilted dermoscopic image deck at the top.
  * Preprocessing chain (binary mask -> ROI -> hair map -> Telea cleaned).
  * THREE parallel feature branches:
      - MKT (4 lambda magnitude maps, each in its own colormap)
      - Handcrafted (Geom 18 + Color 138 + Texture 229 = 385)
      - Frozen ResNet-50 (rainbow CNN layer rectangles -> 2,048)
  * Convergence (+) -> hybrid R^{2,509} strip.
  * Dataset & training protocol row.
  * Headline result card + Labelled BCC/BKL test results grid.

Layout: landscape, figsize=(13.5, 10.5) at 300 DPI. Designed for
\\begin{figure*} (full text-width on Elsevier).

Run:
    cd /Users/mmoniem96/Desktop/Work/Master/Practical\\ Project/MelanomaGraphAnalysisV3
    python paper_pipeline/scripts/regenerate_figure6_v6.py
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
    Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon,
    Rectangle,
)
from PIL import Image
from scipy.special import comb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_pipeline.pipeline.mkt_lambda import get_lambda_fn


OUT_PIPELINE = (PROJECT_ROOT / "paper_pipeline" / "output" / "figures"
                / "proposed_framework_figure_v6.png")
OUT_PAPER    = Path("/Users/mmoniem96/Desktop/Work/Master/"
                    "Mohamed_Shoieb_Master_Paper_ElsevierV7/images/"
                    "proposed_framework_figure_v6.png")
PAPER_IMG_DIR = OUT_PAPER.parent

PREPROC_ORIG  = PAPER_IMG_DIR / "preprocessing_original_image.png"
PREPROC_MASK  = PAPER_IMG_DIR / "preprocessing_binary_mask.png"
PREPROC_ROI   = PAPER_IMG_DIR / "preprocessing_roi.png"
PREPROC_HAIR  = PAPER_IMG_DIR / "preprocessing_hair_detection.png"
PREPROC_RES   = PAPER_IMG_DIR / "preprocessing_result.png"

BCC_DIR = PROJECT_ROOT / "data" / "canonical" / "bcc_segmented"
BKL_DIR = PROJECT_ROOT / "data" / "canonical" / "bkl_segmented"


# ARS palette (consistent with paper preamble)
NAVY       = "#2C3E50"
NAVY_DEEP  = "#1B2D40"
NAVY_TINT  = "#F4F6F8"
NAVY_BAND  = "#E7ECF1"
GOLD       = "#E67E22"
GOLD_TINT  = "#FDEBD0"
GOLD_DEEP  = "#B7610F"
BLUE       = "#2874A6"
BLUE_TINT  = "#E7F0F8"
TEAL       = "#16A085"
PURPLE     = "#7D3C98"
TEXT_COL   = "#34495E"
TEXT_SOFT  = "#5D6D7E"
WHITE      = "#FFFFFF"

# Rainbow for ResNet-50 CNN layer bars
RAINBOW = ["#E74C3C", "#E67E22", "#F1C40F", "#27AE60", "#2980B9", "#8E44AD"]

# Per-lambda MKT colormap
MKT_CMAPS = {
    "low_pass":     "viridis",
    "high_pass":    "plasma",
    "dft":          "cividis",
    "odd_harmonic": "inferno",
}

N_MKT = 64
P_MKT = 0.5


# ---------- Krawtchouk MKT ---------------------------------------------------
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


def build_K0(N, p):
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


def mkt_magnitude_map(image_gray_64, lam_name, K0):
    lam_fn = get_lambda_fn(lam_name)
    Lambda = lam_fn(image_gray_64.shape[0]).astype(complex)
    Y = apply_2d_mdfkt(K0, Lambda, image_gray_64.astype(float))
    mag = np.abs(Y)
    mag = np.log1p(mag)
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag


# ---------- Drawing primitives ----------------------------------------------
def rbox(ax, x, y, w, h, *, fill=WHITE, edge=NAVY, lw=1.0, rounding=0.08,
          alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                 boxstyle=f"round,pad=0,rounding_size={rounding}",
                                 linewidth=lw, edgecolor=edge, facecolor=fill,
                                 alpha=alpha))


def text(ax, x, y, txt, *, fs=10, color=TEXT_COL, weight="normal",
         ha="center", va="center", family="serif", rotation=0):
    ax.text(x, y, txt, ha=ha, va=va, color=color, fontsize=fs,
            fontweight=weight, family=family, rotation=rotation)


def arrow(ax, p1, p2, *, color=GOLD, lw=2.4, mut=20, style="-|>",
          connection="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=color,
                                  linewidth=lw, mutation_scale=mut,
                                  connectionstyle=connection))


def thick_arrow(ax, p1, p2, *, color=GOLD, lw=4.0, mut=28,
                 connection="arc3,rad=0"):
    """Wide chunky arrow like the reference figure's purple arrows."""
    ax.add_patch(FancyArrowPatch(p1, p2,
                                  arrowstyle="-|>,head_width=0.7,head_length=1.0",
                                  color=color,
                                  linewidth=lw, mutation_scale=mut,
                                  connectionstyle=connection))


def imshow_inset(ax, image, x, y, w, h, *, cmap=None, border=NAVY, lw=1.2):
    iax = ax.inset_axes([x, y, w, h], transform=ax.transData)
    if cmap is None:
        iax.imshow(image)
    else:
        iax.imshow(image, cmap=cmap)
    iax.set_xticks([]); iax.set_yticks([])
    for s in iax.spines.values():
        s.set_edgecolor(border); s.set_linewidth(lw)


def db_cylinder(ax, cx, cy, w, h, *, fill=NAVY_TINT, edge=NAVY, lw=1.2):
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


def section_label(ax, x, y, txt, *, fs=12):
    text(ax, x, y, txt, fs=fs, color=NAVY, weight="bold", ha="left")
    ax.plot([x, x + len(txt) * fs / 80 + 1.0], [y - 0.15, y - 0.15],
            color=GOLD, lw=2.0, solid_capstyle="round")


# ---------- Helpers ---------------------------------------------------------
def _load_img(path, mode="RGB"):
    try:
        return np.array(Image.open(path).convert(mode))
    except Exception:
        return None


def _sample_images(folder, n, seed=42):
    rng = np.random.default_rng(seed)
    files = sorted([p for p in folder.iterdir() if p.suffix == ".png"])
    picks = rng.choice(files, size=min(n, len(files)), replace=False)
    return [np.array(Image.open(p).convert("RGB")) for p in picks]


def stacked_image_deck(ax, images, x0, y0, w, h, *, n=4, offset=0.18,
                        border=NAVY, lw=1.5):
    """Render a tilted stacked-photo deck (no actual rotation, just offset).
    Iterates back-to-front so the topmost image is the last drawn."""
    if not images:
        return
    n = min(n, len(images))
    for i in range(n):
        idx = (n - 1 - i)
        dx = offset * idx
        dy = -offset * idx
        imshow_inset(ax, images[i], x0 + dx, y0 + dy, w, h,
                      border=border, lw=lw)


def render_resnet_bars(ax, x0, y0, w, h):
    """Five rainbow rectangles representing ResNet-50 stages + GAP arrow."""
    # 5 stages with increasing channel count (display as growing height)
    layer_labels = ["conv1\n64", "res2\n256", "res3\n512", "res4\n1024",
                    "res5\n2048"]
    heights = [0.50, 0.65, 0.75, 0.85, 1.00]  # relative
    n = len(layer_labels)
    gap = 0.02
    bar_w = (w - (n + 1) * gap) / n - 0.05
    cx = x0 + gap
    max_h = h * 0.85
    base_y = y0 + 0.10
    for i, (label, rel_h) in enumerate(zip(layer_labels, heights)):
        bar_h = max_h * rel_h
        color = RAINBOW[i]
        rbox(ax, cx, base_y + (max_h - bar_h) / 2,
             bar_w, bar_h,
             fill=color, edge=color, lw=1.0, rounding=0.04)
        text(ax, cx + bar_w / 2,
             base_y + (max_h - bar_h) / 2 + bar_h / 2,
             label, fs=7.5, color=WHITE, weight="bold")
        cx += bar_w + gap
    # GAP final bar
    rbox(ax, cx, base_y + max_h * 0.20, bar_w * 0.8, max_h * 0.6,
         fill=RAINBOW[5], edge=RAINBOW[5], lw=1.0, rounding=0.04)
    text(ax, cx + bar_w * 0.4, base_y + max_h * 0.5,
         "GAP", fs=7.5, color=WHITE, weight="bold")


# ---------- Main figure -----------------------------------------------------
_mkt_cached = {}


def render(K0, lesion_gray_64):
    for lam in ["low_pass", "high_pass", "dft", "odd_harmonic"]:
        _mkt_cached[lam] = mkt_magnitude_map(lesion_gray_64, lam, K0)

    # Sample images for deck + result grid
    bcc_samples = _sample_images(BCC_DIR, 8, seed=7)
    bkl_samples = _sample_images(BKL_DIR, 8, seed=11)
    deck_pool = bcc_samples[:2] + bkl_samples[:2]  # 4 thumbnails for the deck

    fig = plt.figure(figsize=(13.5, 10.5), dpi=300)
    ax  = fig.add_subplot(111)
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 10.5)
    ax.set_aspect("equal"); ax.axis("off")

    # ==== Title bar ====
    rbox(ax, 0.20, 9.70, 13.1, 0.65, fill=NAVY_DEEP, edge=NAVY_DEEP,
         lw=0, rounding=0.10)
    text(ax, 6.75, 10.07,
         "Proposed BCC vs BKL CAD Pipeline   "
         "-   hybrid MKT + handcrafted + frozen ResNet-50",
         fs=13, color=WHITE, weight="bold")

    # ==== ROW 1 - Database & Preprocessing ====
    # y range 7.40 .. 9.55  (height 2.15)
    section_label(ax, 0.30, 9.45, "Database & Preprocessing", fs=12)

    # HAM10000 cylinder
    cyl_cx = 1.15; cyl_cy = 8.40
    db_cylinder(ax, cyl_cx, cyl_cy, 1.30, 1.50,
                fill=NAVY_TINT, edge=NAVY, lw=1.3)
    text(ax, cyl_cx, cyl_cy + 0.07, "HAM10000",
         fs=10, weight="bold", color=NAVY)
    text(ax, cyl_cx, cyl_cy - 0.16, r"514 BCC + 1{,}099 BKL", fs=7.5)
    text(ax, cyl_cx, cyl_cy - 0.35, r"= 1{,}613 images",
         fs=8.5, color=NAVY, weight="bold")
    text(ax, cyl_cx, cyl_cy - 1.00, "Database",
         fs=9, color=TEXT_SOFT)

    # Stacked dermoscopic deck
    deck_x = 2.40; deck_y = 7.90; deck_w = 1.10; deck_h = 1.10
    stacked_image_deck(ax, deck_pool, deck_x, deck_y, deck_w, deck_h,
                        n=4, offset=0.13, border=NAVY, lw=1.0)
    text(ax, deck_x + deck_w / 2, deck_y - 0.30,
         "Original\ndermoscopic images",
         fs=8.5, color=NAVY, weight="bold")
    thick_arrow(ax,
                (cyl_cx + 0.70, cyl_cy),
                (deck_x - 0.03, deck_y + deck_h / 2),
                color=GOLD)

    # Preprocessing chain - 5 thumbnails
    preproc_steps = [
        (_load_img(PREPROC_ORIG, "RGB"), None,   NAVY, "Original",  r"$224{\times}224{\times}3$"),
        (_load_img(PREPROC_MASK, "L"),   "gray", NAVY, "Mask",       r"$M$"),
        (_load_img(PREPROC_ROI,  "RGB"), None,   NAVY, "ROI",        r"$I \odot M$"),
        (_load_img(PREPROC_HAIR, "RGB"), None,   NAVY, "Hair map",   r"Top/Black-hat"),
        (_load_img(PREPROC_RES,  "RGB"), None,   GOLD, "Cleaned",    r"Telea inpaint"),
    ]
    pre_x0 = 4.55; pre_y = 7.90; pre_w = 1.30; pre_h = 1.10
    gap = 0.20
    cx = pre_x0
    centres = []
    for img, cmap, border, title, sub in preproc_steps:
        if img is None:
            img = np.zeros((10, 10, 3), dtype=np.uint8)
        head = (border == GOLD)
        lw = 2.4 if head else 1.0
        imshow_inset(ax, img, cx, pre_y, pre_w, pre_h,
                      cmap=cmap, border=border, lw=lw)
        text(ax, cx + pre_w / 2, pre_y - 0.13, title,
             fs=8.5, weight="bold",
             color=GOLD if head else NAVY)
        text(ax, cx + pre_w / 2, pre_y - 0.32, sub,
             fs=7.3, color=TEXT_SOFT)
        centres.append((cx + pre_w / 2, pre_y + pre_h / 2))
        cx += pre_w + gap
    # arrows between preprocessing steps (gold thick)
    for i in range(len(preproc_steps) - 1):
        a = (centres[i][0] + pre_w / 2 + 0.02, centres[i][1])
        b = (centres[i + 1][0] - pre_w / 2 - 0.02, centres[i + 1][1])
        thick_arrow(ax, a, b, color=GOLD, lw=2.5, mut=18)
    # Deck -> Preprocessing chain
    thick_arrow(ax,
                (deck_x + deck_w + 0.30, deck_y + deck_h / 2),
                (pre_x0 - 0.03, pre_y + pre_h / 2),
                color=GOLD)
    # downsample chip below preprocessing
    ds_w = 4.5; ds_h = 0.32
    ds_x = pre_x0 + (5 * pre_w + 4 * gap - ds_w) / 2
    ds_y = pre_y - 0.66
    rbox(ax, ds_x, ds_y, ds_w, ds_h, fill=NAVY_TINT, edge=NAVY, lw=0.8)
    text(ax, ds_x + ds_w / 2, ds_y + ds_h / 2,
         r"grayscale, downsample to $64 \times 64$ for MKT",
         fs=8.0, color=NAVY)

    # ==== ROW 2 - Feature extraction (3 parallel branches + convergence) ====
    # y range 4.20 .. 7.15  (height ~2.95)
    section_label(ax, 0.30, 7.05, "Feature Extraction & Selection", fs=12)

    # branch container box (background tint)
    branch_x0 = 0.30; branch_y0 = 4.30
    branch_w = 9.4; branch_h = 2.70
    rbox(ax, branch_x0, branch_y0, branch_w, branch_h,
         fill=NAVY_BAND, edge="none", lw=0, rounding=0.10, alpha=0.5)

    sub_branch_h = (branch_h - 0.30) / 3
    sub_gap = 0.10

    # ----- Branch A: MKT (top) -----
    bA_y = branch_y0 + branch_h - sub_branch_h - 0.10
    rbox(ax, branch_x0 + 0.10, bA_y, branch_w - 0.20, sub_branch_h,
         fill=GOLD_TINT, edge=GOLD, lw=1.4)
    text(ax, branch_x0 + 0.30, bA_y + sub_branch_h - 0.18,
         "MKT operator",
         fs=10, color=GOLD_DEEP, weight="bold", ha="left")
    text(ax, branch_x0 + 0.30, bA_y + sub_branch_h - 0.40,
         r"$\boldsymbol{\Psi} = \mathcal{K}_\Lambda I \mathcal{K}_\Lambda^\top$,  "
         r"$N=64$, $p=0.5$",
         fs=8.5, color=NAVY, ha="left")
    text(ax, branch_x0 + 0.30, bA_y + 0.15,
         r"$\Rightarrow$ 76 features per $\lambda$",
         fs=8.5, color=GOLD_DEEP, ha="left", weight="bold")
    # 4 lambda magnitude maps
    lam_specs = [
        ("low_pass",      "low-pass",            False),
        ("high_pass",     "high-pass",           False),
        ("dft",           "DFT-like",            False),
        ("odd_harmonic",  r"odd-harm.~$\star$",  True),
    ]
    th_x0 = branch_x0 + 3.10
    th_w  = 1.20
    th_h  = sub_branch_h - 0.40
    th_y  = bA_y + 0.20
    th_gap = 0.20
    for lam, label, head in lam_specs:
        accent = GOLD if head else NAVY
        lw = 2.6 if head else 1.0
        cmap = MKT_CMAPS[lam]
        imshow_inset(ax, _mkt_cached[lam], th_x0, th_y, th_w, th_h,
                      cmap=cmap, border=accent, lw=lw)
        text(ax, th_x0 + th_w / 2, th_y - 0.16, label,
             fs=8, weight="bold" if head else "normal",
             color=GOLD if head else NAVY)
        th_x0 += th_w + th_gap

    # ----- Branch B: Handcrafted (middle) -----
    bB_y = bA_y - sub_branch_h - sub_gap
    rbox(ax, branch_x0 + 0.10, bB_y, branch_w - 0.20, sub_branch_h,
         fill=NAVY_TINT, edge=NAVY, lw=1.2)
    text(ax, branch_x0 + 0.30, bB_y + sub_branch_h - 0.18,
         "Handcrafted descriptors",
         fs=10, color=NAVY, weight="bold", ha="left")
    text(ax, branch_x0 + 0.30, bB_y + sub_branch_h - 0.40,
         "spatial features", fs=8.5, color=TEXT_COL, ha="left")
    text(ax, branch_x0 + 0.30, bB_y + 0.15,
         r"$\Rightarrow$ 385 features",
         fs=8.5, color=NAVY, ha="left", weight="bold")
    # 3 sub-chips: Geom / Color / Texture
    sub_x0 = branch_x0 + 3.10
    sub_w = 1.20
    sub_h = sub_branch_h - 0.40
    sub_y = bB_y + 0.20
    sub_chips = [
        ("Geometric", "18",  "area, perimeter,\nasymmetry, fractal"),
        ("Color",     "138", "RGB, HSV, CIELAB\nstats + histograms"),
        ("Texture",   "229", "GLCM (4 angles),\nLBP, Gabor 8-orient"),
    ]
    chip_x = sub_x0
    for label, count, blurb in sub_chips:
        rbox(ax, chip_x, sub_y, sub_w * 1.5, sub_h,
             fill=WHITE, edge=NAVY, lw=0.9)
        text(ax, chip_x + sub_w * 0.75, sub_y + sub_h - 0.20,
             label, fs=9, weight="bold", color=NAVY)
        text(ax, chip_x + sub_w * 0.75, sub_y + sub_h - 0.42,
             count, fs=11, color=NAVY, weight="bold")
        for li, line in enumerate(blurb.split("\n")):
            text(ax, chip_x + sub_w * 0.75,
                 sub_y + 0.30 - li * 0.16,
                 line, fs=6.8, color=TEXT_SOFT)
        chip_x += sub_w * 1.5 + 0.15

    # ----- Branch C: ResNet-50 (bottom) -----
    bC_y = bB_y - sub_branch_h - sub_gap
    rbox(ax, branch_x0 + 0.10, bC_y, branch_w - 0.20, sub_branch_h,
         fill=BLUE_TINT, edge=BLUE, lw=1.2)
    text(ax, branch_x0 + 0.30, bC_y + sub_branch_h - 0.18,
         "Frozen ResNet-50",
         fs=10, color=BLUE, weight="bold", ha="left")
    text(ax, branch_x0 + 0.30, bC_y + sub_branch_h - 0.40,
         "ImageNet weights, no fine-tuning",
         fs=8.5, color=TEXT_COL, ha="left")
    text(ax, branch_x0 + 0.30, bC_y + 0.15,
         r"$\Rightarrow$ 2{,}048 features",
         fs=8.5, color=BLUE, ha="left", weight="bold")
    # Rainbow CNN bars
    render_resnet_bars(ax, branch_x0 + 3.05, bC_y + 0.05,
                        branch_w - 3.25, sub_branch_h - 0.10)

    # ----- Convergence (+) + Hybrid R^2509 strip -----
    cv_x = branch_x0 + branch_w + 0.30
    cv_cy = branch_y0 + branch_h / 2
    plus_r = 0.35
    ax.add_patch(Circle((cv_x, cv_cy), plus_r,
                          facecolor=GOLD, edgecolor=GOLD_DEEP, linewidth=1.5))
    text(ax, cv_x, cv_cy, "+", fs=22, color=WHITE, weight="bold")
    # branches -> +
    thick_arrow(ax, (branch_x0 + branch_w - 0.05, bA_y + sub_branch_h / 2),
                     (cv_x - plus_r, cv_cy), color=GOLD, lw=2.0, mut=16)
    thick_arrow(ax, (branch_x0 + branch_w - 0.05, bB_y + sub_branch_h / 2),
                     (cv_x - plus_r, cv_cy), color=NAVY, lw=2.0, mut=16)
    thick_arrow(ax, (branch_x0 + branch_w - 0.05, bC_y + sub_branch_h / 2),
                     (cv_x - plus_r, cv_cy), color=BLUE, lw=2.0, mut=16)
    # Hybrid R^2509 strip
    hv_x = cv_x + plus_r + 0.30
    hv_y = branch_y0 + 0.30
    hv_w = 13.30 - hv_x - 0.20
    hv_h = branch_h - 0.60
    rbox(ax, hv_x, hv_y, hv_w, hv_h, fill=GOLD, edge=GOLD_DEEP, lw=2.0)
    text(ax, hv_x + hv_w / 2, hv_y + hv_h * 0.66,
         "Hybrid feature", fs=10, color=WHITE, weight="bold")
    text(ax, hv_x + hv_w / 2, hv_y + hv_h * 0.45,
         "vector", fs=10, color=WHITE, weight="bold")
    text(ax, hv_x + hv_w / 2, hv_y + hv_h * 0.22,
         r"$\mathbf{x} \in \mathbb{R}^{2{,}509}$",
         fs=12, color=WHITE, weight="bold")
    # + -> hybrid bar
    thick_arrow(ax, (cv_x + plus_r, cv_cy),
                     (hv_x - 0.02, hv_y + hv_h / 2),
                color=GOLD_DEEP, lw=2.6, mut=20)

    # Preprocessing chain -> branches (long arrow on the left)
    thick_arrow(ax, (centres[-1][0], pre_y - 0.55),
                     (branch_x0 + 1.5, branch_y0 + branch_h + 0.03),
                color=GOLD)

    # ==== ROW 3 - Data + Training protocol ====
    # y range 2.35 .. 4.10  (height 1.75)
    section_label(ax, 0.30, 4.00, "Dataset & Training Protocol", fs=12)

    train_band_y = 2.55; train_band_h = 1.35
    rbox(ax, 0.30, train_band_y, 13.0, train_band_h,
         fill=NAVY_BAND, edge="none", lw=0, rounding=0.10, alpha=0.55)

    # Mini cylinder
    mini_cyl_cx = 1.05; mini_cyl_cy = train_band_y + 0.70
    db_cylinder(ax, mini_cyl_cx, mini_cyl_cy, 0.90, 1.10,
                fill=WHITE, edge=NAVY, lw=1.0)
    text(ax, mini_cyl_cx, mini_cyl_cy + 0.05, "HAM10000",
         fs=8, weight="bold", color=NAVY)
    text(ax, mini_cyl_cx, mini_cyl_cy - 0.15, r"1{,}613", fs=7.5, color=NAVY)

    # Stratified split
    sp_x = 2.30; sp_y = train_band_y + 0.20; sp_w = 1.85; sp_h = 0.95
    rbox(ax, sp_x, sp_y, sp_w, sp_h, fill=WHITE, edge=NAVY, lw=1.0)
    text(ax, sp_x + sp_w / 2, sp_y + sp_h - 0.18,
         "Stratified 80/20", fs=9, weight="bold", color=NAVY)
    text(ax, sp_x + sp_w / 2, sp_y + sp_h - 0.40,
         "by image id", fs=8)
    text(ax, sp_x + sp_w / 2, sp_y + sp_h - 0.62,
         r"random\_state = 42", fs=7.5, color=TEXT_SOFT)
    text(ax, sp_x + sp_w / 2, sp_y + 0.13,
         r"train 1{,}290 | test 323", fs=8, color=NAVY, weight="bold")
    thick_arrow(ax, (mini_cyl_cx + 0.5, mini_cyl_cy),
                     (sp_x - 0.02, sp_y + sp_h / 2), color=GOLD, lw=2.0)

    # Train aug
    ag_x = 4.45; ag_y = train_band_y + 0.60; ag_w = 1.95; ag_h = 0.65
    rbox(ax, ag_x, ag_y, ag_w, ag_h, fill=GOLD_TINT, edge=GOLD, lw=1.0)
    text(ax, ag_x + ag_w / 2, ag_y + ag_h - 0.18,
         "Train + h/v flip aug", fs=8.5, weight="bold", color=GOLD_DEEP)
    text(ax, ag_x + ag_w / 2, ag_y + 0.18,
         r"411 BCC $\rightarrow$ 879 ; balanced 1{,}758",
         fs=7.5, color=NAVY)

    # Test branch
    te_x = ag_x; te_y = train_band_y + 0.10; te_w = ag_w; te_h = 0.40
    rbox(ax, te_x, te_y, te_w, te_h, fill=WHITE, edge=NAVY, lw=1.3)
    text(ax, te_x + te_w / 2, te_y + te_h / 2,
         r"Test 323 (103 BCC + 220 BKL)  -  held out",
         fs=7.8, weight="bold", color=NAVY)
    thick_arrow(ax, (sp_x + sp_w + 0.02, sp_y + sp_h * 0.7),
                     (ag_x - 0.02, ag_y + ag_h / 2), color=GOLD, lw=1.8)
    thick_arrow(ax, (sp_x + sp_w + 0.02, sp_y + sp_h * 0.3),
                     (te_x - 0.02, te_y + te_h / 2), color=NAVY, lw=1.8)

    # MI selection
    mi_x = 6.70; mi_y = train_band_y + 0.20; mi_w = 1.95; mi_h = 0.95
    rbox(ax, mi_x, mi_y, mi_w, mi_h, fill=WHITE, edge=NAVY, lw=1.0)
    text(ax, mi_x + mi_w / 2, mi_y + mi_h - 0.20,
         "SelectKBest", fs=9, weight="bold", color=NAVY)
    text(ax, mi_x + mi_w / 2, mi_y + mi_h - 0.42,
         "(mutual info)", fs=7.8)
    text(ax, mi_x + mi_w / 2, mi_y + mi_h - 0.65,
         r"$k = 360 / 2{,}509$", fs=9, color=NAVY, weight="bold")
    text(ax, mi_x + mi_w / 2, mi_y + 0.13,
         "RobustScaler, fit on train", fs=7.3, color=TEXT_SOFT)
    thick_arrow(ax, (ag_x + ag_w + 0.02, ag_y + ag_h / 2),
                     (mi_x - 0.02, mi_y + mi_h / 2), color=GOLD, lw=2.0)

    # 9 classifiers
    cf_x = 8.95; cf_y = train_band_y + 0.20; cf_w = 4.20; cf_h = 0.95
    rbox(ax, cf_x, cf_y, cf_w, cf_h, fill=WHITE, edge=NAVY, lw=1.0)
    text(ax, cf_x + cf_w / 2, cf_y + cf_h - 0.18,
         "9-classifier sweep + stratified 5-fold CV",
         fs=9, weight="bold", color=NAVY)
    text(ax, cf_x + cf_w / 2, cf_y + cf_h - 0.42,
         "SVM (RBF) - LightGBM - CatBoost - GradBoost", fs=7.5)
    text(ax, cf_x + cf_w / 2, cf_y + cf_h - 0.60,
         "Extra Trees - KNN - Logistic Regression", fs=7.5)
    text(ax, cf_x + cf_w / 2, cf_y + 0.18,
         r"$\mathbf{MLP^{\star}}$ (headline) - Deep DNN",
         fs=8, weight="bold", color=GOLD_DEEP)
    thick_arrow(ax, (mi_x + mi_w + 0.02, mi_y + mi_h / 2),
                     (cf_x - 0.02, cf_y + cf_h / 2), color=GOLD, lw=2.0)

    # ==== ROW 4 - Headline + Labelled results grid ====
    # y range 0.10 .. 2.30
    section_label(ax, 0.30, 2.25,
                  "Headline Result   &   Labelled Test Outputs",
                  fs=12)

    # Headline gold card on the left
    hl_x = 0.30; hl_y = 0.20; hl_w = 6.6; hl_h = 1.95
    rbox(ax, hl_x, hl_y, hl_w, hl_h, fill=GOLD, edge=GOLD_DEEP, lw=2.0)
    text(ax, hl_x + hl_w / 2, hl_y + hl_h - 0.20,
         "HEADLINE   -   MLP + odd-harmonic MKT",
         fs=11, color=WHITE, weight="bold")
    text(ax, hl_x + hl_w / 2, hl_y + hl_h - 0.45,
         r"$\lambda_k = e^{i(2k+1)\pi/N}$, $p=0.5$, $N=64$",
         fs=9, color=WHITE)
    # 6 metric cells
    metric_xs = np.linspace(hl_x + 0.55, hl_x + hl_w - 0.55, 6)
    keys = ["Acc", "Sens", "Spec", "Prec", "F1", "AUC"]
    vals = ["93.50%", "91.26%", "94.55%", "88.68%", "89.95%", "96.95%"]
    mw = (hl_w - 1.10) / 6 - 0.04
    mh = 0.70
    my = hl_y + 0.55
    for cx, k, v in zip(metric_xs, keys, vals):
        rbox(ax, cx - mw / 2, my, mw, mh,
             fill=WHITE, edge=GOLD_DEEP, lw=1.0)
        text(ax, cx, my + mh * 0.72, k, fs=8, color=GOLD_DEEP,
             weight="bold")
        text(ax, cx, my + mh * 0.30, v, fs=10, color=NAVY, weight="bold")
    text(ax, hl_x + hl_w / 2, hl_y + 0.30,
         r"95\% CI: Acc $[90.71, 95.98]$ - AUC $[95.13, 98.51]$",
         fs=8.0, color=WHITE)
    text(ax, hl_x + hl_w / 2, hl_y + 0.13,
         r"McNemar vs LightGBM+low-pass ($93.19\%$): exact $p = 1.0$ -"
         r" test 323 (103+220)",
         fs=7.7, color=WHITE)

    # Result grid on the right - 6 test images with predicted labels
    rg_x = 7.20; rg_y = 0.20; rg_w = 6.10; rg_h = 1.95
    rbox(ax, rg_x, rg_y, rg_w, rg_h, fill=WHITE, edge=NAVY, lw=1.2)
    text(ax, rg_x + rg_w / 2, rg_y + rg_h - 0.18,
         "Proposed labelled outputs   (sample of 323-image test set)",
         fs=10, color=NAVY, weight="bold")
    # 2 rows x 3 cols of thumbnails
    # row 1: 3 BCC predictions; row 2: 3 BKL predictions
    n_cols = 3
    n_rows = 2
    grid_x0 = rg_x + 0.30
    grid_y0 = rg_y + 0.20
    cell_w = (rg_w - 0.60) / n_cols - 0.08
    cell_h = (rg_h - 0.60) / n_rows - 0.12
    img_h = cell_h - 0.18
    img_w = cell_w - 0.05
    # pick 3 BCC + 3 BKL
    row_samples = [bcc_samples[2:5], bkl_samples[2:5]]
    row_labels  = ["BCC", "BKL"]
    cur_y = grid_y0 + (n_rows - 1) * (cell_h + 0.12)
    for samples, lab in zip(row_samples, row_labels):
        cur_x = grid_x0
        for img in samples:
            imshow_inset(ax, img, cur_x, cur_y + 0.16, img_w, img_h,
                          border=GOLD if lab == "BCC" else NAVY, lw=1.4)
            # label chip
            chip_w = img_w * 0.85
            chip_x = cur_x + (img_w - chip_w) / 2
            chip_y = cur_y - 0.02
            chip_fill = GOLD_TINT if lab == "BCC" else NAVY_TINT
            chip_edge = GOLD if lab == "BCC" else NAVY
            rbox(ax, chip_x, chip_y, chip_w, 0.18,
                 fill=chip_fill, edge=chip_edge, lw=0.9, rounding=0.04)
            text(ax, chip_x + chip_w / 2, chip_y + 0.09,
                 f"{lab}  $\\checkmark$", fs=8,
                 color=GOLD_DEEP if lab == "BCC" else NAVY, weight="bold")
            cur_x += cell_w + 0.08
        cur_y -= cell_h + 0.12

    # Connect training row -> result row
    thick_arrow(ax, (cf_x + cf_w / 2, train_band_y),
                     (hl_x + hl_w / 2, hl_y + hl_h + 0.02),
                color=GOLD, lw=2.4, mut=20)

    return fig


def main():
    img = Image.open(BCC_DIR / "ISIC_0024331.png").convert("RGB")
    gray = img.convert("L").resize((N_MKT, N_MKT))
    lesion_gray_64 = np.array(gray, dtype=float)

    print("Building Krawtchouk K0 matrix (~5 s)...")
    K0 = build_K0(N_MKT, P_MKT)
    print(f"K0 shape: {K0.shape}")

    print("Composing Figure 6 v6 (PowerPoint-style)...")
    fig = render(K0, lesion_gray_64)
    fig.savefig(OUT_PIPELINE, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.10)
    plt.close(fig)
    print(f"-> {OUT_PIPELINE}")
    if OUT_PAPER.parent.exists():
        shutil.copy(OUT_PIPELINE, OUT_PAPER)
        print(f"-> {OUT_PAPER}")


if __name__ == "__main__":
    main()
