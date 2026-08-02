"""Figure 6 v5 - publication-grade methodology figure with parallel branches.

Design pattern follows the convention used by Elsevier / Springer / PMC
medical-AI methodology figures: each stage is a bordered card with a header
bar, sub-elements flow left-to-right within a card, and stages are stacked
top-to-bottom. The feature-extraction stage splits into THREE parallel
branches (MKT, Handcrafted, ResNet-50) that converge at a (+) node into the
hybrid vector.

Six stages:
  (1) Preprocessing chain  - 5 thumbnails + 64x64 downsample
  (2) Feature extraction   - 3 parallel branches converging at +
  (3) Data + training      - cylinder, split, augmentation, MI, 9 classifiers
  (4) Headline result      - MLP + odd-harm card with 6 metrics + CIs + McN

Sizing: figsize=(11, 13.5) at 300 DPI -> 3300 x 4050 px.
When placed in Overleaf at \\textwidth (~6.5 in) the figure scales to about
6.5 x 7.97 inches, which fits a single page comfortably.

Run:
    cd /Users/mmoniem96/Desktop/Work/Master/Practical\\ Project/MelanomaGraphAnalysisV3
    python paper_pipeline/scripts/regenerate_figure6_v5.py
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
)
from PIL import Image
from scipy.special import comb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_pipeline.pipeline.mkt_lambda import get_lambda_fn


OUT_PIPELINE = (PROJECT_ROOT / "paper_pipeline" / "output" / "figures"
                / "proposed_framework_figure_v5.png")
OUT_PAPER    = Path("/Users/mmoniem96/Desktop/Work/Master/"
                    "Mohamed_Shoieb_Master_Paper_ElsevierV7/images/"
                    "proposed_framework_figure_v5.png")
SAMPLE_BCC_PATH = (PROJECT_ROOT / "data" / "canonical" / "bcc_segmented"
                    / "ISIC_0024331.png")
PAPER_IMG_DIR = OUT_PAPER.parent
PREPROC_ORIG  = PAPER_IMG_DIR / "preprocessing_original_image.png"
PREPROC_MASK  = PAPER_IMG_DIR / "preprocessing_binary_mask.png"
PREPROC_ROI   = PAPER_IMG_DIR / "preprocessing_roi.png"
PREPROC_HAIR  = PAPER_IMG_DIR / "preprocessing_hair_detection.png"
PREPROC_RES   = PAPER_IMG_DIR / "preprocessing_result.png"


# ARS palette
NAVY       = "#2C3E50"
NAVY_DEEP  = "#1B2D40"
NAVY_TINT  = "#F4F6F8"
NAVY_BAND  = "#E7ECF1"
GOLD       = "#E67E22"
GOLD_TINT  = "#FDEBD0"
GOLD_DEEP  = "#B7610F"
TEAL       = "#16A085"
TEAL_TINT  = "#E8F8F3"
BLUE       = "#2874A6"
BLUE_TINT  = "#E7F0F8"
PURPLE     = "#7D3C98"
PURPLE_TINT= "#F0E6F4"
GREY_LT    = "#ECF0F1"
GREY_MED   = "#BDC3C7"
TEXT_COL   = "#34495E"
TEXT_SOFT  = "#5D6D7E"
WHITE      = "#FFFFFF"

N_MKT = 64
P_MKT = 0.5


# ---------- Krawtchouk MKT (unchanged) --------------------------------------
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
def rbox(ax, x, y, w, h, *, fill=WHITE, edge=NAVY, lw=1.0, rounding=0.10,
          alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                 boxstyle=f"round,pad=0,rounding_size={rounding}",
                                 linewidth=lw, edgecolor=edge, facecolor=fill,
                                 alpha=alpha))


def shadow_box(ax, x, y, w, h, *, depth=0.06, alpha=0.12, rounding=0.10):
    ax.add_patch(FancyBboxPatch((x + depth, y - depth), w, h,
                                 boxstyle=f"round,pad=0,rounding_size={rounding}",
                                 linewidth=0, facecolor="black", alpha=alpha))


def text(ax, x, y, txt, *, fs=10, color=TEXT_COL, weight="normal",
         ha="center", va="center", family="serif"):
    ax.text(x, y, txt, ha=ha, va=va, color=color, fontsize=fs,
            fontweight=weight, family=family)


def arrow(ax, p1, p2, *, color=NAVY, lw=1.3, mut=14, style="-|>",
          connection="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=color,
                                  linewidth=lw, mutation_scale=mut,
                                  connectionstyle=connection))


def manhattan(ax, p1, p2, *, color=NAVY, lw=1.3, mut=14, horiz_first=True):
    mid = (p2[0], p1[1]) if horiz_first else (p1[0], p2[1])
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


def stage_card(ax, x, y, w, h, n, title, *, accent=NAVY, header_h=0.55,
                  subtitle=None):
    shadow_box(ax, x, y, w, h, depth=0.06, alpha=0.09)
    rbox(ax, x, y, w, h, fill=WHITE, edge=accent, lw=1.3, rounding=0.10)
    # header bar
    rbox(ax, x + 0.05, y + h - header_h, w - 0.10, header_h - 0.05,
         fill=accent, edge=accent, lw=0, rounding=0.08)
    # badge
    ax.add_patch(Circle((x + 0.45, y + h - header_h/2 - 0.02),
                          0.22, facecolor=WHITE, edgecolor=accent,
                          linewidth=1.6))
    text(ax, x + 0.45, y + h - header_h/2 - 0.02, str(n),
         fs=12, color=accent, weight="bold")
    # title
    title_x = x + 0.85
    text(ax, title_x, y + h - header_h/2 - 0.02, title,
         fs=12.5, color=WHITE, weight="bold", ha="left")
    if subtitle is not None:
        # subtitle near right edge of header
        text(ax, x + w - 0.20, y + h - header_h/2 - 0.02, subtitle,
             fs=9.5, color="#D6DCE4", ha="right")
    # return inner content rectangle
    inner_x = x + 0.25
    inner_y = y + 0.25
    inner_w = w - 0.50
    inner_h = h - header_h - 0.45
    return (inner_x, inner_y, inner_w, inner_h)


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


def _load_img(path, mode="RGB"):
    try:
        return np.array(Image.open(path).convert(mode))
    except Exception:
        return None


# ---------- Stage 1 - Preprocessing -----------------------------------------
def render_stage_preproc(ax, inner):
    x, y, w, h = inner
    steps = [
        (_load_img(PREPROC_ORIG, "RGB"),  None,   NAVY, "Original",    r"$224{\times}224{\times}3$"),
        (_load_img(PREPROC_MASK, "L"),    "gray", NAVY, "Binary mask", r"HAM10000"),
        (_load_img(PREPROC_ROI,  "RGB"),   None,  NAVY, "Lesion ROI",  r"$I \odot M$"),
        (_load_img(PREPROC_HAIR, "RGB"),   None,  NAVY, "Hair map",    r"Top/Black-hat"),
        (_load_img(PREPROC_RES,  "RGB"),   None,  GOLD, "Cleaned",     r"Telea + $\sigma{=}0.8$"),
    ]
    n = len(steps)
    gap = 0.40
    th_w = (w - (n - 1) * gap) / n
    th_h = h - 0.95
    cx = x
    centres = []
    for i, (img, cmap, border, title, sub) in enumerate(steps):
        if img is None:
            img = np.zeros((10, 10, 3), dtype=np.uint8)
        head = (border == GOLD)
        lw = 2.6 if head else 1.1
        imshow_inset(ax, img, cx, y + 0.55, th_w, th_h,
                      cmap=cmap, border=border, lw=lw)
        text(ax, cx + th_w/2, y + 0.55 - 0.25, title,
             fs=10, weight="bold",
             color=GOLD if head else NAVY)
        text(ax, cx + th_w/2, y + 0.55 - 0.48, sub,
             fs=8.8, color=TEXT_SOFT)
        centres.append((cx + th_w/2, y + 0.55))
        if i < n - 1:
            arrow(ax,
                  (cx + th_w + 0.04, y + 0.55 + th_h/2),
                  (cx + th_w + gap - 0.04, y + 0.55 + th_h/2),
                  lw=1.1, mut=12)
        cx += th_w + gap

    # Downsample chip below
    ds_w = w * 0.35; ds_h = 0.40
    ds_x = x + (w - ds_w) / 2
    ds_y = y + 0.05
    rbox(ax, ds_x, ds_y, ds_w, ds_h, fill=NAVY_TINT, edge=NAVY, lw=1.0)
    text(ax, ds_x + ds_w/2, ds_y + ds_h/2,
         r"$\rightarrow$ grayscale, downsample to $64{\times}64$ for MKT",
         fs=9, color=NAVY)
    # arrow Cleaned -> downsample
    cleaned_centre = centres[-1]
    manhattan(ax, (cleaned_centre[0], cleaned_centre[1]),
                   (ds_x + ds_w * 0.85, ds_y + ds_h),
              lw=1.0, mut=12, horiz_first=False, color=GOLD)
    return ds_x + ds_w / 2, ds_y + ds_h / 2  # outflow point


# ---------- Stage 2 - Feature extraction (3 parallel branches) --------------
def render_stage_feat(ax, inner):
    """Three parallel branches converging at (+).

    Branch A: MKT  (4 lambda thumbs + 76 features)
    Branch B: Handcrafted  (Geom, Color, Texture chips + 385 features)
    Branch C: Frozen ResNet-50  (text + 2048 features)
    Convergence: (+) on the right -> R^2509 bar
    """
    x, y, w, h = inner

    # Vertical layout: 3 sub-bands of equal height
    branch_gap = 0.18
    avail_h = h
    branch_h = (avail_h - 2 * branch_gap) / 3

    branch_w = w * 0.78   # 78% of width for branches; right 22% = convergence
    cv_x = x + branch_w + 0.20  # convergence column

    # ---------- Branch A: MKT operator ----------
    bA_y = y + 2 * (branch_h + branch_gap)
    bA_box_x = x
    bA_box_w = branch_w
    bA_box_h = branch_h
    rbox(ax, bA_box_x, bA_y, bA_box_w, bA_box_h,
         fill=GOLD_TINT, edge=GOLD, lw=1.4)
    # Branch label on left
    text(ax, bA_box_x + 0.20, bA_y + bA_box_h - 0.30,
         "MKT", fs=11, color=GOLD_DEEP, weight="bold", ha="left")
    text(ax, bA_box_x + 0.20, bA_y + bA_box_h - 0.58,
         r"$\boldsymbol{\Psi} = \mathcal{K}_\Lambda I \mathcal{K}_\Lambda^\top$",
         fs=9.5, color=NAVY, ha="left")
    text(ax, bA_box_x + 0.20, bA_y + bA_box_h - 0.85,
         r"$N{=}64$, $p{=}0.5$",
         fs=8.5, color=TEXT_SOFT, ha="left")
    text(ax, bA_box_x + 0.20, bA_y + 0.27,
         r"$\Rightarrow$ 76 features",
         fs=10, color=GOLD_DEEP, weight="bold", ha="left")
    # 4 lambda magnitude thumbnails inside the branch
    lam_specs = [
        ("low_pass",     "low-pass",   False),
        ("high_pass",    "high-pass",  False),
        ("dft",          "DFT-like",   False),
        ("odd_harmonic", r"odd-harm.$\star$", True),
    ]
    th_x0 = bA_box_x + 1.85
    th_w  = (bA_box_w - 1.95 - 0.30) / 4
    th_h  = bA_box_h - 0.65
    th_y  = bA_y + 0.32
    th_gap_inner = 0.08
    actual_th_w = th_w - th_gap_inner
    for i, (lam, label, head) in enumerate(lam_specs):
        accent = GOLD if head else NAVY
        lw = 2.2 if head else 1.0
        imshow_inset(ax, _mkt_cached[lam], th_x0, th_y,
                      actual_th_w, th_h,
                      cmap="magma", border=accent, lw=lw)
        text(ax, th_x0 + actual_th_w/2, th_y - 0.20, label,
             fs=8.5, weight="bold" if head else "normal",
             color=GOLD if head else NAVY)
        th_x0 += th_w

    # ---------- Branch B: Handcrafted descriptors ----------
    bB_y = y + branch_h + branch_gap
    rbox(ax, x, bB_y, branch_w, branch_h,
         fill=NAVY_BAND, edge=NAVY, lw=1.2)
    text(ax, x + 0.20, bB_y + branch_h - 0.30,
         "Handcrafted", fs=11, color=NAVY, weight="bold", ha="left")
    text(ax, x + 0.20, bB_y + branch_h - 0.58,
         "spatial descriptors", fs=9.5, color=TEXT_COL, ha="left")
    text(ax, x + 0.20, bB_y + 0.27,
         r"$\Rightarrow$ 385 features", fs=10, color=NAVY,
         weight="bold", ha="left")
    # Three sub-chips inside
    sub_x = x + 1.85
    sub_specs = [
        ("Geometric", "18",  "area, perimeter,\nasymmetry, fractal"),
        ("Color",     "138", "RGB/HSV/CIELAB stats,\nhistograms, asymmetry"),
        ("Texture",   "229", "GLCM (4 angles, 5 props),\nLBP, Gabor 8-orient."),
    ]
    sub_w = (branch_w - 1.95 - 0.30) / 3
    sub_h = branch_h - 0.50
    sub_y = bB_y + 0.32
    sub_gap = 0.08
    for label, count, blurb in sub_specs:
        rbox(ax, sub_x, sub_y, sub_w - sub_gap, sub_h,
             fill=WHITE, edge=NAVY, lw=0.9)
        text(ax, sub_x + (sub_w - sub_gap)/2, sub_y + sub_h - 0.22,
             label, fs=9.5, weight="bold", color=NAVY)
        text(ax, sub_x + (sub_w - sub_gap)/2, sub_y + sub_h - 0.48,
             count, fs=11.5, weight="bold", color=NAVY)
        # blurb (multi-line)
        for i_line, line in enumerate(blurb.split("\n")):
            text(ax, sub_x + (sub_w - sub_gap)/2,
                 sub_y + 0.28 - i_line * 0.18,
                 line, fs=7.5, color=TEXT_SOFT)
        sub_x += sub_w

    # ---------- Branch C: Frozen ResNet-50 ----------
    bC_y = y
    rbox(ax, x, bC_y, branch_w, branch_h,
         fill=BLUE_TINT, edge=BLUE, lw=1.2)
    text(ax, x + 0.20, bC_y + branch_h - 0.30,
         "Frozen ResNet-50", fs=11, color=BLUE, weight="bold", ha="left")
    text(ax, x + 0.20, bC_y + branch_h - 0.58,
         "ImageNet weights, no fine-tuning", fs=9.5,
         color=TEXT_COL, ha="left")
    text(ax, x + 0.20, bC_y + 0.27,
         r"$\Rightarrow$ 2{,}048 features", fs=10, color=BLUE,
         weight="bold", ha="left")
    # Schematic CNN block + GAP arrow
    schem_x = x + 1.85
    schem_y_top = bC_y + branch_h - 0.20
    schem_y_bot = bC_y + 0.55
    # Layer rectangles representing the 5 ResNet stages
    layer_labels = ["conv1\n(64)", "res2\n(256)", "res3\n(512)",
                    "res4\n(1024)", "res5\n(2048)"]
    layer_widths = [0.45, 0.55, 0.62, 0.70, 0.80]
    sw_total = sum(layer_widths) + (len(layer_widths) - 1) * 0.08
    layer_x = schem_x
    schem_centre_y = (schem_y_top + schem_y_bot) / 2
    schem_h = (schem_y_top - schem_y_bot) * 0.85
    for label, lw in zip(layer_labels, layer_widths):
        rbox(ax, layer_x,
             schem_centre_y - schem_h * (0.4 + 0.04 * layer_widths.index(lw)),
             lw, schem_h * (0.8 + 0.08 * layer_widths.index(lw)),
             fill=WHITE, edge=BLUE, lw=0.9, rounding=0.03)
        text(ax, layer_x + lw/2, schem_centre_y, label,
             fs=7.0, color=BLUE)
        layer_x += lw + 0.08

    # GAP + 2048-d arrow
    text(ax, layer_x + 0.40, schem_centre_y + 0.05, "GAP",
         fs=9, color=BLUE, weight="bold", ha="center")
    text(ax, layer_x + 0.40, schem_centre_y - 0.20,
         r"$\rightarrow 2{,}048$-d", fs=8, color=TEXT_SOFT, ha="center")

    # ---------- Convergence (+) symbol + R^2509 strip ----------
    cv_y = y + h / 2 - 0.50  # vertical centre
    # Big plus circle
    plus_r = 0.42
    ax.add_patch(Circle((cv_x + 0.20, cv_y + 0.30), plus_r,
                          facecolor=GOLD, edgecolor=GOLD_DEEP,
                          linewidth=1.5))
    text(ax, cv_x + 0.20, cv_y + 0.30, r"$\boldsymbol{+}$",
         fs=18, color=WHITE, weight="bold")
    # Hybrid R^2509 bar
    hv_x = cv_x - 0.10
    hv_y = y + 0.05
    hv_w = w - branch_w - 0.20
    hv_h = 0.62
    rbox(ax, hv_x, hv_y, hv_w, hv_h, fill=GOLD, edge=GOLD_DEEP, lw=1.5)
    text(ax, hv_x + hv_w/2, hv_y + hv_h/2 + 0.10,
         r"$\mathbf{x} \in \mathbb{R}^{2{,}509}$",
         fs=12, color=WHITE, weight="bold")
    text(ax, hv_x + hv_w/2, hv_y + hv_h/2 - 0.16,
         r"hybrid feature vector",
         fs=8.5, color=WHITE)

    # Three branches -> (+)
    branch_right_x = x + branch_w
    bA_centre_y = bA_y + branch_h / 2
    bB_centre_y = bB_y + branch_h / 2
    bC_centre_y = bC_y + branch_h / 2
    manhattan(ax, (branch_right_x + 0.02, bA_centre_y),
                   (cv_x + 0.20 - plus_r, cv_y + 0.30),
              color=GOLD, lw=1.3, mut=14)
    manhattan(ax, (branch_right_x + 0.02, bB_centre_y),
                   (cv_x + 0.20 - plus_r, cv_y + 0.30),
              color=NAVY, lw=1.3, mut=14)
    manhattan(ax, (branch_right_x + 0.02, bC_centre_y),
                   (cv_x + 0.20 - plus_r, cv_y + 0.30),
              color=BLUE, lw=1.3, mut=14)
    # (+) -> R^2509 bar
    arrow(ax, (cv_x + 0.20, cv_y + 0.30 - plus_r),
              (cv_x + 0.20, hv_y + hv_h + 0.02),
          color=GOLD_DEEP, lw=1.6, mut=16)


# ---------- Stage 3 - Data + Training ---------------------------------------
def render_stage_data(ax, inner):
    x, y, w, h = inner

    # Top row: cylinder -> split -> train/test branches -> MI
    top_row_y = y + h - 1.85
    # cylinder
    cyl_cx = x + 1.10
    cyl_cy = top_row_y + 0.05
    db_cylinder(ax, cyl_cx, cyl_cy, 1.30, 1.60,
                fill=NAVY_TINT, edge=NAVY, lw=1.2)
    text(ax, cyl_cx, cyl_cy + 0.13, "HAM10000",
         fs=10.5, weight="bold", color=NAVY)
    text(ax, cyl_cx, cyl_cy - 0.12, r"514 BCC + 1{,}099 BKL", fs=8.5)
    text(ax, cyl_cx, cyl_cy - 0.36, r"= 1{,}613 images",
         fs=9.5, color=NAVY, weight="bold")
    text(ax, cyl_cx, cyl_cy - 1.05,
         r"$\mathrm{dx}$-filter $\cap$ masks",
         fs=8, color=TEXT_SOFT)

    # Split box
    sp_x = x + 2.75; sp_y = top_row_y - 0.25; sp_w = 1.9; sp_h = 1.10
    rbox(ax, sp_x, sp_y, sp_w, sp_h, fill=WHITE, edge=NAVY, lw=1.0)
    text(ax, sp_x + sp_w/2, sp_y + sp_h - 0.22,
         "Stratified split", fs=9.5, weight="bold", color=NAVY)
    text(ax, sp_x + sp_w/2, sp_y + sp_h - 0.45,
         r"80 / 20 by image id", fs=9)
    text(ax, sp_x + sp_w/2, sp_y + sp_h - 0.70,
         r"seed = 42", fs=8, color=TEXT_SOFT)
    text(ax, sp_x + sp_w/2, sp_y + 0.18,
         r"deterministic", fs=8, color=TEXT_SOFT)
    arrow(ax, (cyl_cx + 0.70, cyl_cy), (sp_x - 0.02, sp_y + sp_h/2), lw=1.1)

    # Train / test branches
    br_x = x + 4.90
    train_y = top_row_y + 0.15; test_y = top_row_y - 1.05
    br_w = 2.10; br_h = 0.95
    rbox(ax, br_x, train_y, br_w, br_h, fill=NAVY_TINT, edge=NAVY, lw=1.0)
    text(ax, br_x + br_w/2, train_y + br_h - 0.22,
         r"Train  $n{=}1{,}290$", fs=9.5, weight="bold", color=NAVY)
    text(ax, br_x + br_w/2, train_y + br_h - 0.50,
         r"411 BCC + 879 BKL", fs=8.5)
    text(ax, br_x + br_w/2, train_y + br_h - 0.78,
         r"BCC h/v-flip $\rightarrow$ 1{,}758",
         fs=8.3, color=GOLD_DEEP, weight="bold")
    rbox(ax, br_x, test_y, br_w, br_h, fill=WHITE, edge=NAVY, lw=1.5)
    text(ax, br_x + br_w/2, test_y + br_h - 0.22,
         r"Test  $n{=}323$", fs=9.5, weight="bold", color=NAVY)
    text(ax, br_x + br_w/2, test_y + br_h - 0.50,
         r"103 BCC + 220 BKL", fs=8.5)
    text(ax, br_x + br_w/2, test_y + br_h - 0.78,
         r"held out", fs=8, color=TEXT_SOFT)
    manhattan(ax, (sp_x + sp_w + 0.02, sp_y + sp_h/2),
                   (br_x - 0.02, train_y + br_h/2), lw=1.0)
    manhattan(ax, (sp_x + sp_w + 0.02, sp_y + sp_h/2),
                   (br_x - 0.02, test_y + br_h/2), lw=1.0)

    # MI box (right of train)
    mi_x = x + 7.30; mi_y = train_y; mi_w = 2.05; mi_h = br_h
    rbox(ax, mi_x, mi_y, mi_w, mi_h, fill=NAVY_TINT, edge=NAVY, lw=1.0)
    text(ax, mi_x + mi_w/2, mi_y + mi_h - 0.22,
         "SelectKBest (MI)", fs=9.5, weight="bold", color=NAVY)
    text(ax, mi_x + mi_w/2, mi_y + mi_h - 0.50,
         r"$k{=}360$ / 2{,}509", fs=10.5, weight="bold", color=NAVY)
    text(ax, mi_x + mi_w/2, mi_y + mi_h - 0.78,
         "RobustScaler, fit on train",
         fs=7.7, color=TEXT_SOFT)
    arrow(ax, (br_x + br_w + 0.02, train_y + br_h/2),
              (mi_x - 0.02, mi_y + mi_h/2), lw=1.1)

    # Classifier strip at bottom (full-width)
    cs_x = x; cs_y = y; cs_w = w; cs_h = 1.05
    rbox(ax, cs_x, cs_y, cs_w, cs_h, fill=WHITE, edge=NAVY, lw=1.1)
    text(ax, cs_x + cs_w/2, cs_y + cs_h - 0.22,
         "9 classifiers   -   stratified 5-fold cross-validation   -   "
         "test set touched once per model",
         fs=9.8, weight="bold", color=NAVY)
    text(ax, cs_x + cs_w/2, cs_y + cs_h - 0.55,
         "SVM (RBF)   -   LightGBM   -   CatBoost   -   "
         "Gradient Boosting   -   Extra Trees   -   KNN   -   "
         "Logistic Regression   -   "
         r"$\mathbf{MLP}\,\boldsymbol{\star}$   -   Deep DNN",
         fs=9, color=NAVY)
    text(ax, cs_x + cs_w/2, cs_y + 0.18,
         r"$\star$ = headline configuration (paired with odd-harmonic MKT)",
         fs=8, color=GOLD_DEEP)
    arrow(ax, (mi_x + mi_w/2, mi_y),
              (cs_x + cs_w * 0.65, cs_y + cs_h), lw=1.2)
    arrow(ax, (br_x + br_w/2, test_y),
              (cs_x + cs_w * 0.35, cs_y + cs_h),
          color=TEXT_SOFT, lw=1.0,
          connection="arc3,rad=0.10")


# ---------- Stage 4 - Headline result ---------------------------------------
def render_stage_headline(ax, inner, card_x, card_y, card_w, card_h):
    x, y, w, h = inner
    # Title row
    text(ax, x + w/2, y + h - 0.25,
         r"Headline configuration   --   MLP classifier $\oplus$ "
         r"odd-harmonic MKT   ($\lambda_k = e^{i(2k+1)\pi/N}$, "
         r"$p=0.5$, $N=64$)",
         fs=11.5, color=NAVY, weight="bold")
    # 6 metric cells
    metric_xs = np.linspace(x + 0.65, x + w - 0.65, 6)
    keys = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1",
            "AUC"]
    vals = ["93.50%", "91.26%", "94.55%", "88.68%", "89.95%", "96.95%"]
    mw = (w - 1.3) / 6 - 0.05
    mh = 0.90
    my = y + h - 1.60
    for cx, k, v in zip(metric_xs, keys, vals):
        rbox(ax, cx - mw/2, my, mw, mh,
             fill=GOLD_TINT, edge=GOLD, lw=1.2)
        text(ax, cx, my + mh * 0.72, k, fs=9, color=GOLD_DEEP,
             weight="bold")
        text(ax, cx, my + mh * 0.30, v, fs=12, color=NAVY,
             weight="bold")
    # CI + McNemar + test composition
    text(ax, x + w/2, y + 0.95,
         r"95\% bootstrap CI (10{,}000 resamples):   "
         r"Acc $\in [90.71, 95.98]$   -   AUC $\in [95.13, 98.51]$",
         fs=9.5, color=NAVY)
    text(ax, x + w/2, y + 0.62,
         r"McNemar's test vs runner-up (LightGBM + low-pass, $93.19\%$):   "
         r"exact $p = 1.0$",
         fs=9.5, color=NAVY)
    text(ax, x + w/2, y + 0.28,
         r"Evaluated once on the 323-image held-out test set "
         r"(103 BCC + 220 BKL)",
         fs=10, color=NAVY, weight="bold")
    # Gold accent underline
    ax.plot([card_x + 0.50, card_x + card_w - 0.50],
            [card_y + 0.16, card_y + 0.16],
            color=GOLD, lw=2.5, solid_capstyle="round")


# ---------- Layout glue ------------------------------------------------------
_mkt_cached = {}


def render(K0, lesion_gray_64):
    for lam in ["low_pass", "high_pass", "dft", "odd_harmonic"]:
        _mkt_cached[lam] = mkt_magnitude_map(lesion_gray_64, lam, K0)

    fig = plt.figure(figsize=(11, 13.5), dpi=300)
    ax  = fig.add_subplot(111)
    ax.set_xlim(0, 14); ax.set_ylim(0, 17.2)
    ax.set_aspect("equal"); ax.axis("off")

    # ===== Outer title bar =====
    rbox(ax, 0.20, 16.30, 13.6, 0.80,
         fill=NAVY_DEEP, edge=NAVY_DEEP, lw=0, rounding=0.10)
    text(ax, 7.0, 16.70,
         "Proposed BCC vs BKL CAD Pipeline",
         fs=14.5, color=WHITE, weight="bold")
    text(ax, 7.0, 16.42,
         r"Hybrid MKT + handcrafted + frozen ResNet-50  |  9 classifiers  |  "
         r"stratified 5-fold CV  |  bootstrap + McNemar inference",
         fs=9.5, color="#D6DCE4")

    # ===== Card layout =====
    margin_x = 0.20
    card_w = 13.6
    inter = 0.30  # gap between cards
    cur_y = 16.30 - 0.30   # below title bar

    # Stage 1 - Preprocessing
    s1_h = 2.40
    s1_y = cur_y - s1_h
    inner1 = stage_card(ax, margin_x, s1_y, card_w, s1_h, 1,
                         "Preprocessing chain",
                         accent=NAVY,
                         subtitle="dermoscopic image -> cleaned 64x64")
    out_preproc = render_stage_preproc(ax, inner1)
    cur_y = s1_y - inter

    # Stage 2 - Feature extraction (3 parallel branches)
    s2_h = 5.40
    s2_y = cur_y - s2_h
    inner2 = stage_card(ax, margin_x, s2_y, card_w, s2_h, 2,
                         "Feature extraction  -  three parallel branches",
                         accent=NAVY,
                         subtitle=r"convergence at $\oplus$ -> $\mathbf{x} \in \mathbb{R}^{2{,}509}$")
    render_stage_feat(ax, inner2)

    # Inter-stage arrow 1 -> 2 (gold to mark dataflow continuing the
    # cleaned path; arrow lands at the top of stage 2 in its centre)
    arrow(ax, (out_preproc[0], s1_y),
              (out_preproc[0], s2_y + s2_h + 0.04),
          color=GOLD, lw=1.8, mut=18)

    cur_y = s2_y - inter

    # Stage 3 - Data + Training
    s3_h = 3.90
    s3_y = cur_y - s3_h
    inner3 = stage_card(ax, margin_x, s3_y, card_w, s3_h, 3,
                         "Dataset & training protocol",
                         accent=NAVY,
                         subtitle="HAM10000 -> split -> aug -> select -> classify")
    render_stage_data(ax, inner3)

    # arrow 2 -> 3 from the R^2509 strip down to top of stage 3
    arrow(ax, (7.0, s2_y),
              (7.0, s3_y + s3_h + 0.04),
          color=NAVY, lw=1.8, mut=18)

    cur_y = s3_y - inter

    # Stage 4 - Headline result (gold-accented card)
    s4_h = 2.90
    s4_y = cur_y - s4_h
    inner4 = stage_card(ax, margin_x, s4_y, card_w, s4_h, 4,
                         "Headline result   -   single-touch held-out test",
                         accent=GOLD)
    render_stage_headline(ax, inner4, margin_x, s4_y, card_w, s4_h)

    # arrow 3 -> 4
    arrow(ax, (7.0, s3_y),
              (7.0, s4_y + s4_h + 0.04),
          color=GOLD, lw=1.8, mut=18)

    return fig


def main():
    img = Image.open(SAMPLE_BCC_PATH).convert("RGB")
    gray = img.convert("L").resize((N_MKT, N_MKT))
    lesion_gray_64 = np.array(gray, dtype=float)

    print("Building Krawtchouk K0 matrix (~5 s)...")
    K0 = build_K0(N_MKT, P_MKT)
    print(f"K0 shape: {K0.shape}")

    print("Composing Figure 6 v5 (parallel branches)...")
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
