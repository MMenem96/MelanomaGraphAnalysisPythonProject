"""Figure 6 v4 - publication-grade modular pipeline figure.

Design intent (per user request):
  * Complex but professional.
  * Each stage is broken into its own bordered MODULE CARD with a coloured
    title bar, internal sub-blocks, and a numbered badge.
  * Six modules connected by clear vertical/Manhattan arrows:
        (1) Input lesion
        (2) Preprocessing chain  (4 sub-steps)
        (3) MKT operator family  (4 lambda sub-cards)
        (4) Hybrid feature fusion (5 family chips + total dim)
        (5) Dataset & training protocol (cylinder + split + aug + MI + 9
            classifiers + 5-fold CV)
        (6) Headline result card  (model + 6 metrics + 2 CIs + McNemar +
            test-set composition)
  * figsize chosen so that at the paper's textwidth (~6.5 in) the body
    text remains >= 7 pt: figsize=(10.5, 16), dpi=300.
  * Backed by real Phase A v2 numbers - no fabricated values anywhere.

Run:
    cd /Users/mmoniem96/Desktop/Work/Master/Practical\\ Project/MelanomaGraphAnalysisV3
    python paper_pipeline/scripts/regenerate_figure6_v4.py
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
                / "proposed_framework_figure_v4.png")
OUT_PAPER    = Path("/Users/mmoniem96/Desktop/Work/Master/"
                    "Mohamed_Shoieb_Master_Paper_ElsevierV7/images/"
                    "proposed_framework_figure_v4.png")
SAMPLE_BCC_PATH = (PROJECT_ROOT / "data" / "canonical" / "bcc_segmented"
                    / "ISIC_0024331.png")
PAPER_IMG_DIR = OUT_PAPER.parent
PREPROC_ORIG  = PAPER_IMG_DIR / "preprocessing_original_image.png"
PREPROC_MASK  = PAPER_IMG_DIR / "preprocessing_binary_mask.png"
PREPROC_ROI   = PAPER_IMG_DIR / "preprocessing_roi.png"
PREPROC_HAIR  = PAPER_IMG_DIR / "preprocessing_hair_detection.png"
PREPROC_RES   = PAPER_IMG_DIR / "preprocessing_result.png"


# Palette (matches paper preamble ARS palette)
NAVY       = "#2C3E50"
NAVY_DEEP  = "#1B2D40"
NAVY_TINT  = "#F4F6F8"
NAVY_BAND  = "#E7ECF1"
GOLD       = "#E67E22"
GOLD_TINT  = "#FDEBD0"
GOLD_DEEP  = "#B7610F"
TEAL       = "#16A085"
PURPLE     = "#7D3C98"
BLUE       = "#2874A6"
GREY_LT    = "#ECF0F1"
GREY_MED   = "#BDC3C7"
TEXT_COL   = "#34495E"
TEXT_SOFT  = "#5D6D7E"
WHITE      = "#FFFFFF"

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


def shadow_box(ax, x, y, w, h, *, depth=0.06, alpha=0.15, rounding=0.08):
    """Drop-shadow under a box: render a slightly offset dark rectangle."""
    ax.add_patch(FancyBboxPatch((x + depth, y - depth), w, h,
                                 boxstyle=f"round,pad=0,rounding_size={rounding}",
                                 linewidth=0, facecolor="black", alpha=alpha))


def text(ax, x, y, txt, *, fs=10, color=TEXT_COL, weight="normal",
         ha="center", va="center", family="serif"):
    ax.text(x, y, txt, ha=ha, va=va, color=color, fontsize=fs,
            fontweight=weight, family=family)


def arrow(ax, p1, p2, *, color=NAVY, lw=1.3, style="-|>", mut=14,
          connection="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=color,
                                  linewidth=lw, mutation_scale=mut,
                                  connectionstyle=connection))


def manhattan_arrow(ax, p1, p2, *, color=NAVY, lw=1.3, mut=14,
                     horizontal_first=True):
    mid = (p2[0], p1[1]) if horizontal_first else (p1[0], p2[1])
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


def stage_badge(ax, x, y, n, *, r=0.30, fill=NAVY, edge=NAVY, txt_color=WHITE):
    ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=edge,
                         linewidth=1.5))
    text(ax, x, y, str(n), fs=12, color=txt_color, weight="bold")


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


def module_card(ax, x, y, w, h, n, title, *, accent=NAVY,
                 title_color=WHITE, header_h=0.55):
    """Draw a complete module card: shadow + body + header bar + numbered badge.
    Returns the inner content rectangle bounds for child rendering.
    """
    # Shadow
    shadow_box(ax, x, y, w, h, depth=0.06, alpha=0.10)
    # Body
    rbox(ax, x, y, w, h, fill=WHITE, edge=accent, lw=1.4, rounding=0.10)
    # Header band
    ax.add_patch(FancyBboxPatch((x + 0.06, y + h - header_h),
                                  w - 0.12, header_h - 0.04,
                                  boxstyle="round,pad=0,rounding_size=0.08",
                                  linewidth=0, facecolor=accent))
    # Numbered badge sits on top-left of header
    stage_badge(ax, x + 0.45, y + h - header_h/2,
                n, r=0.27, fill=WHITE, edge=accent, txt_color=accent)
    # Title text
    text(ax, x + w/2 + 0.10, y + h - header_h/2, title,
         fs=12.5, color=title_color, weight="bold")
    inner = (x + 0.18, y + 0.18, w - 0.36, h - header_h - 0.30)
    return inner


# ---------- Layout helpers ---------------------------------------------------
def hline(ax, x1, x2, y, *, color=GREY_MED, lw=0.6, dashes=(1.5, 1.5)):
    ax.plot([x1, x2], [y, y], color=color, lw=lw, dashes=dashes)


# ---------- Content of each module ------------------------------------------
def render_module_input(ax, inner, lesion_thumb):
    x, y, w, h = inner
    # Single thumbnail on the left, description on the right
    th_h = h - 0.20
    th_w = th_h
    th_x = x + 0.25
    th_y = y + 0.10
    imshow_inset(ax, lesion_thumb, th_x, th_y, th_w, th_h,
                  border=NAVY, lw=1.4)

    # Description block on the right
    rx = th_x + th_w + 0.55
    text(ax, rx, y + h - 0.20,
         "Dermoscopic image",
         fs=11, color=NAVY, weight="bold", ha="left")
    text(ax, rx, y + h - 0.55,
         r"Resolution: $224 \times 224 \times 3$",
         fs=10, ha="left")
    text(ax, rx, y + h - 0.85,
         "Source: HAM10000 Dataverse  (DOI: 10.7910/DVN/DBW86T)",
         fs=9.5, color=TEXT_SOFT, ha="left")
    text(ax, rx, y + h - 1.15,
         "Classes used: BCC = 514, BKL = 1{,}099",
         fs=10, ha="left", color=NAVY, weight="bold")
    text(ax, rx, y + h - 1.45,
         "Each image paired with a binary lesion mask",
         fs=9.5, color=TEXT_SOFT, ha="left")


def render_module_preproc(ax, inner):
    x, y, w, h = inner
    # 4 sub-steps in a row: Mask -> ROI -> Hair -> Cleaned
    steps = [
        (_load_img(PREPROC_MASK, "L"),    "gray", NAVY, "Apply mask", r"$I_{\mathrm{ROI}} = I \odot M$",        False),
        (_load_img(PREPROC_ROI,  "RGB"),   None,   NAVY, "Lesion ROI", r"$M$ from HAM10000",                       False),
        (_load_img(PREPROC_HAIR, "RGB"),   None,   NAVY, "Hair detect", r"Top-/Black-hat",                          False),
        (_load_img(PREPROC_RES,  "RGB"),   None,   GOLD, "Cleaned",    r"Telea inpaint $+ \mathcal{N}(0,\sigma{=}0.8)$", True),
    ]
    n = len(steps)
    th_w = (w - (n + 1) * 0.18 - 0.30) / n
    th_h = h - 0.95
    cx = x + 0.18
    for i, (img, cmap, border, title, sub, head) in enumerate(steps):
        if img is None:
            img = np.zeros((10, 10, 3), dtype=np.uint8)
        lw = 2.5 if head else 1.1
        imshow_inset(ax, img, cx, y + 0.55, th_w, th_h,
                      cmap=cmap, border=border, lw=lw)
        text(ax, cx + th_w/2, y + 0.55 - 0.22, title,
             fs=10, weight="bold",
             color=GOLD if head else NAVY)
        text(ax, cx + th_w/2, y + 0.55 - 0.46, sub,
             fs=8.7, color=TEXT_SOFT)
        if i < n - 1:
            arrow(ax,
                  (cx + th_w + 0.02, y + 0.55 + th_h/2),
                  (cx + th_w + 0.16, y + 0.55 + th_h/2),
                  lw=1.1)
        cx += th_w + 0.18

    # Downsample chip below
    ds_w = 3.4; ds_h = 0.50
    ds_x = x + (w - ds_w) / 2
    ds_y = y - 0.05
    rbox(ax, ds_x, ds_y, ds_w, ds_h, fill=NAVY_TINT, edge=NAVY, lw=1.0)
    text(ax, ds_x + ds_w/2, ds_y + ds_h/2,
         r"downsample to grayscale $64 \times 64$ for MKT",
         fs=9, color=NAVY)


def render_module_mkt(ax, inner):
    x, y, w, h = inner
    # Operator line at top
    text(ax, x + w/2, y + h - 0.30,
         r"$\boldsymbol{\Psi} = \mathcal{K}_\Lambda \, I \, \mathcal{K}_\Lambda^\top$"
         r"  -  separable 2-D MKT with $N{=}64$, $p{=}0.5$",
         fs=11, color=NAVY, weight="bold")

    # 4 lambda sub-cards
    lam_specs = [
        ("low_pass",     "low-pass",   r"$\lambda_k = 1/(k{+}1)$",        NAVY, False),
        ("high_pass",    "high-pass",  r"$\lambda_k = 1/(N{-}k)$",        NAVY, False),
        ("dft",          "DFT-like",   r"$\lambda_k = e^{i 2\pi k/N}$",   NAVY, False),
        ("odd_harmonic", r"odd-harmonic $\star$",
                                       r"$\lambda_k = e^{i(2k+1)\pi/N}$", GOLD, True),
    ]
    n = len(lam_specs)
    sub_w = (w - (n + 1) * 0.16 - 0.2) / n
    sub_h = h - 0.85
    cx = x + 0.18
    for (lam, label, eq, accent, head) in lam_specs:
        # Card outline
        body_fill = GOLD_TINT if head else WHITE
        body_edge = GOLD if head else NAVY
        body_lw   = 2.0 if head else 1.0
        if head:
            shadow_box(ax, cx, y + 0.05, sub_w, sub_h, depth=0.05, alpha=0.12)
        rbox(ax, cx, y + 0.05, sub_w, sub_h, fill=body_fill, edge=body_edge,
             lw=body_lw, rounding=0.08)
        # Sub-card title
        text(ax, cx + sub_w/2, y + sub_h - 0.20, label,
             fs=10.5, color=accent, weight="bold")
        text(ax, cx + sub_w/2, y + sub_h - 0.52, eq,
             fs=9.0, color=TEXT_COL)
        # Magnitude thumbnail inside
        img_w = sub_w - 0.30; img_h = sub_h - 1.30
        imshow_inset(ax, _mkt_cached[lam], cx + (sub_w - img_w)/2,
                       y + 0.40, img_w, img_h,
                       cmap="magma", border=accent,
                       lw=1.9 if head else 0.9)
        text(ax, cx + sub_w/2, y + 0.20,
             "76 features",
             fs=8.7, color=GOLD_DEEP if head else NAVY,
             weight="bold" if head else "normal")
        cx += sub_w + 0.16


def render_module_fusion(ax, inner):
    x, y, w, h = inner
    text(ax, x + w/2, y + h - 0.25,
         r"Hybrid feature vector  $\mathbf{x} \in \mathbb{R}^{2{,}509}$  ="
         r"  concat of 5 families",
         fs=11.5, color=NAVY, weight="bold")

    # 5 family chips: Geometric, Color, Texture, MKT (gold), ResNet-50
    chips = [
        ("Geometric", "18",      NAVY,  NAVY_TINT,  "shape area, perimeter, asymmetry, fractal"),
        ("Color",     "138",     NAVY,  NAVY_TINT,  "RGB/HSV/CIELAB stats, histograms, asymmetry"),
        ("Texture",   "229",     NAVY,  NAVY_TINT,  "GLCM (4 angles, 5 props), LBP, Gabor 8-orient."),
        (r"MKT $\star$","76",   GOLD,  GOLD_TINT,  r"Real/Imag/$|\cdot|$/arg per $\lambda$ + stats"),
        ("ResNet-50", r"2{,}048", BLUE, "#E7F0F8", "frozen ImageNet GAP descriptor"),
    ]
    weights = [1.0, 1.0, 1.0, 1.0, 2.5]   # ResNet wider
    total_units = sum(weights)
    strip_w = w - 0.36
    fx = x + 0.18
    chip_h = h - 1.30
    chip_y = y + 0.25
    for (label, count, edge, fill, sub), wt in zip(chips, weights):
        cw = strip_w * (wt / total_units)
        is_head = (edge == GOLD)
        if is_head:
            shadow_box(ax, fx, chip_y, cw, chip_h, depth=0.05, alpha=0.10)
        rbox(ax, fx, chip_y, cw, chip_h, fill=fill, edge=edge,
             lw=1.6 if is_head else 1.0, rounding=0.08)
        text(ax, fx + cw/2, chip_y + chip_h * 0.72, label,
             fs=10.5, color=edge, weight="bold")
        text(ax, fx + cw/2, chip_y + chip_h * 0.42, count,
             fs=12.5, color=edge, weight="bold")
        text(ax, fx + cw/2, chip_y + chip_h * 0.13, sub,
             fs=7.4, color=TEXT_SOFT)
        fx += cw

    # plus signs between chips
    fx = x + 0.18
    for i, wt in enumerate(weights[:-1]):
        cw = strip_w * (wt / total_units)
        fx += cw
        text(ax, fx - 0.01, chip_y + chip_h/2, r"$\oplus$",
             fs=14, color=NAVY, weight="bold")


def render_module_data(ax, inner):
    x, y, w, h = inner
    # ---- left: HAM10000 cylinder + summary ----
    cyl_cx = x + 1.40; cyl_cy = y + h - 1.70
    db_cylinder(ax, cyl_cx, cyl_cy, 1.45, 1.65,
                fill=NAVY_TINT, edge=NAVY, lw=1.2)
    text(ax, cyl_cx, cyl_cy + 0.10, "HAM10000", fs=10.5, weight="bold",
         color=NAVY)
    text(ax, cyl_cx, cyl_cy - 0.12, r"514 BCC + 1{,}099 BKL", fs=8.5)
    text(ax, cyl_cx, cyl_cy - 0.36, r"= 1{,}613 images",
         fs=9.5, color=NAVY, weight="bold")
    text(ax, cyl_cx, cyl_cy - 1.00,
         r"$\mathrm{dx}$-filter $\cap$ HAM10000 masks",
         fs=8, color=TEXT_SOFT)

    # ---- split ----
    sp_x = x + 3.10; sp_y = y + h - 2.20; sp_w = 2.1; sp_h = 0.95
    rbox(ax, sp_x, sp_y, sp_w, sp_h, fill=WHITE, edge=NAVY, lw=1.0)
    text(ax, sp_x + sp_w/2, sp_y + sp_h - 0.20, "Stratified",
         fs=9.5, weight="bold", color=NAVY)
    text(ax, sp_x + sp_w/2, sp_y + sp_h - 0.45, "80/20 on image id", fs=9)
    text(ax, sp_x + sp_w/2, sp_y + sp_h - 0.72,
         r"seed 42", fs=8, color=TEXT_SOFT)
    arrow(ax, (cyl_cx + 0.75, cyl_cy), (sp_x - 0.02, sp_y + sp_h/2), lw=1.1)

    # ---- train & test branches ----
    br_x = x + 5.55
    train_y = y + h - 1.40
    test_y  = y + h - 2.85
    br_w = 2.30; br_h = 1.05
    # Train
    rbox(ax, br_x, train_y, br_w, br_h, fill=NAVY_TINT, edge=NAVY, lw=1.0)
    text(ax, br_x + br_w/2, train_y + br_h - 0.22,
         r"Train  $n{=}1{,}290$", fs=10, weight="bold", color=NAVY)
    text(ax, br_x + br_w/2, train_y + br_h - 0.50,
         r"411 BCC + 879 BKL", fs=8.5)
    text(ax, br_x + br_w/2, train_y + br_h - 0.78,
         r"BCC h/v-flip $\rightarrow$ 1{,}758",
         fs=8.5, color=GOLD_DEEP, weight="bold")
    # Test
    rbox(ax, br_x, test_y, br_w, br_h, fill=WHITE, edge=NAVY, lw=1.5)
    text(ax, br_x + br_w/2, test_y + br_h - 0.22,
         r"Test  $n{=}323$", fs=10, weight="bold", color=NAVY)
    text(ax, br_x + br_w/2, test_y + br_h - 0.50,
         r"103 BCC + 220 BKL", fs=8.5)
    text(ax, br_x + br_w/2, test_y + br_h - 0.78,
         r"held out (single touch)",
         fs=8, color=TEXT_SOFT)

    manhattan_arrow(ax, (sp_x + sp_w + 0.02, sp_y + sp_h/2),
                         (br_x - 0.02, train_y + br_h/2),
                    lw=1.0)
    manhattan_arrow(ax, (sp_x + sp_w + 0.02, sp_y + sp_h/2),
                         (br_x - 0.02, test_y + br_h/2),
                    lw=1.0)

    # ---- MI + scaler box ----
    mi_x = x + 8.20; mi_y = train_y; mi_w = 2.30; mi_h = br_h
    rbox(ax, mi_x, mi_y, mi_w, mi_h, fill=NAVY_TINT, edge=NAVY, lw=1.0)
    text(ax, mi_x + mi_w/2, mi_y + mi_h - 0.22,
         "SelectKBest (MI)", fs=10, weight="bold", color=NAVY)
    text(ax, mi_x + mi_w/2, mi_y + mi_h - 0.50,
         r"$k{=}360$ / 2{,}509", fs=10.5, weight="bold", color=NAVY)
    text(ax, mi_x + mi_w/2, mi_y + mi_h - 0.78,
         "RobustScaler, fit on train",
         fs=8, color=TEXT_SOFT)
    arrow(ax, (br_x + br_w + 0.02, train_y + br_h/2),
              (mi_x - 0.02, mi_y + mi_h/2), lw=1.1)

    # ---- classifier strip at bottom ----
    cs_x = x + 0.10; cs_y = y + 0.05; cs_w = w - 0.20; cs_h = 0.95
    rbox(ax, cs_x, cs_y, cs_w, cs_h, fill=WHITE, edge=NAVY, lw=1.1)
    text(ax, cs_x + cs_w/2, cs_y + cs_h - 0.22,
         "9 classifiers   -   stratified 5-fold CV (dev only)   -   "
         "test set touched once per model",
         fs=10, weight="bold", color=NAVY)
    text(ax, cs_x + cs_w/2, cs_y + 0.27,
         "SVM (RBF)  -  LightGBM  -  CatBoost  -  Gradient Boosting  -  "
         "Extra Trees  -  KNN  -  Logistic Regression  -  "
         r"$\mathbf{MLP^{\star}}$  -  Deep DNN",
         fs=9, color=NAVY)
    # arrow MI -> classifier strip
    arrow(ax, (mi_x + mi_w/2, mi_y), (cs_x + cs_w * 0.65, cs_y + cs_h),
          lw=1.2)
    # arrow Test -> classifier strip
    arrow(ax, (br_x + br_w/2, test_y), (cs_x + cs_w * 0.35, cs_y + cs_h),
          color=TEXT_SOFT, lw=1.0)


def render_module_headline(ax, inner):
    x, y, w, h = inner
    # Title row
    text(ax, x + w/2, y + h - 0.30,
         r"Headline configuration  -  MLP classifier $\oplus$ "
         r"odd-harmonic MKT  ($\lambda_k = e^{i(2k+1)\pi/N}$, "
         r"$p=0.5$, $N=64$)",
         fs=11.5, color=NAVY, weight="bold")

    # 6 metric cells in a row
    metric_xs = np.linspace(x + 0.90, x + w - 0.90, 6)
    keys = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1",
            "AUC"]
    vals = ["93.50%", "91.26%", "94.55%", "88.68%", "89.95%", "96.95%"]
    mw = (w - 1.4) / 6 - 0.06
    mh = 0.95
    my = y + h - 1.80
    for cx, k, v in zip(metric_xs, keys, vals):
        rbox(ax, cx - mw/2, my, mw, mh, fill=NAVY_TINT,
             edge=NAVY, lw=1.1)
        text(ax, cx, my + mh * 0.70, k, fs=9, color=NAVY, weight="bold")
        text(ax, cx, my + mh * 0.28, v, fs=12.5, color=NAVY,
             weight="bold")

    # CI line
    text(ax, x + w/2, y + 1.30,
         r"95\% bootstrap CI (10{,}000 resamples):  "
         r"Acc $\in [90.71, 95.98]$  -  AUC $\in [95.13, 98.51]$",
         fs=9.5, color=NAVY)
    text(ax, x + w/2, y + 0.95,
         r"McNemar's test vs runner-up (LightGBM + low-pass, $93.19\%$):  "
         r"exact $p = 1.0$  (no significant difference)",
         fs=9.5, color=NAVY)
    text(ax, x + w/2, y + 0.55,
         r"Evaluated once on the 323-image held-out test set  "
         r"(103 BCC + 220 BKL)",
         fs=10, color=NAVY, weight="bold")
    # decorative gold underline
    ax.plot([x + 0.50, x + w - 0.50], [y + 0.30, y + 0.30],
            color=GOLD, lw=2.5, solid_capstyle="round")


# ---------- Asset loading ---------------------------------------------------
def _load_img(path, mode="RGB"):
    try:
        return np.array(Image.open(path).convert(mode))
    except Exception:
        return None


_mkt_cached = {}


# ---------- Main layout ------------------------------------------------------
def render(K0, lesion_gray_64, lesion_thumb):
    # Pre-compute MKT maps once
    for lam in ["low_pass", "high_pass", "dft", "odd_harmonic"]:
        _mkt_cached[lam] = mkt_magnitude_map(lesion_gray_64, lam, K0)

    fig = plt.figure(figsize=(10.5, 16.0), dpi=300)
    ax  = fig.add_subplot(111)
    ax.set_xlim(0, 14); ax.set_ylim(0, 21.5)
    ax.set_aspect("equal"); ax.axis("off")

    # ===== Outer title =====
    rbox(ax, 0.20, 20.55, 13.6, 0.80, fill=NAVY_DEEP, edge=NAVY_DEEP,
         lw=0, rounding=0.10)
    text(ax, 7.0, 20.94,
         "Proposed CAD Pipeline   -   BCC vs BKL on HAM10000",
         fs=14.5, color=WHITE, weight="bold")
    text(ax, 7.0, 20.66,
         r"Hybrid MKT + handcrafted + frozen ResNet-50 features  -  "
         r"9 classifiers  -  stratified 5-fold CV  -  bootstrap + McNemar inference",
         fs=9.5, color="#D6DCE4")

    # ===== Module 1 — Input =====
    m1_h = 2.20
    m1_y = 20.55 - 0.25 - m1_h
    inner = module_card(ax, 0.20, m1_y, 13.6, m1_h, 1,
                         "Input  -  dermoscopic image",
                         accent=NAVY)
    render_module_input(ax, inner, lesion_thumb)

    # ===== Module 2 — Preprocessing =====
    m2_h = 2.90
    m2_y = m1_y - 0.35 - m2_h
    inner = module_card(ax, 0.20, m2_y, 13.6, m2_h, 2,
                         "Preprocessing chain",
                         accent=NAVY)
    render_module_preproc(ax, inner)

    # ===== Module 3 — MKT =====
    m3_h = 3.50
    m3_y = m2_y - 0.35 - m3_h
    inner = module_card(ax, 0.20, m3_y, 13.6, m3_h, 3,
                         "MKT operator family   -   four eigenvalue settings",
                         accent=NAVY)
    render_module_mkt(ax, inner)

    # ===== Module 4 — Fusion =====
    m4_h = 2.05
    m4_y = m3_y - 0.35 - m4_h
    inner = module_card(ax, 0.20, m4_y, 13.6, m4_h, 4,
                         "Hybrid feature fusion",
                         accent=NAVY)
    render_module_fusion(ax, inner)

    # ===== Module 5 — Data & training =====
    m5_h = 4.10
    m5_y = m4_y - 0.35 - m5_h
    inner = module_card(ax, 0.20, m5_y, 13.6, m5_h, 5,
                         "Dataset & training protocol",
                         accent=NAVY)
    render_module_data(ax, inner)

    # ===== Module 6 — Headline =====
    m6_h = 3.20
    m6_y = m5_y - 0.35 - m6_h
    inner = module_card(ax, 0.20, m6_y, 13.6, m6_h, 6,
                         "Headline result   -   single-touch held-out test",
                         accent=GOLD)
    render_module_headline(ax, inner)

    # ===== Inter-module arrows (data flow) =====
    arrow_color = NAVY
    arrow_lw = 1.8
    # 1 -> 2
    arrow(ax, (7.0, m1_y), (7.0, m1_y - 0.30 + 0.04),
          color=arrow_color, lw=arrow_lw, mut=18)
    # 2 -> 3
    arrow(ax, (7.0, m2_y), (7.0, m2_y - 0.30 + 0.04),
          color=arrow_color, lw=arrow_lw, mut=18)
    # 3 -> 4
    arrow(ax, (7.0, m3_y), (7.0, m3_y - 0.30 + 0.04),
          color=arrow_color, lw=arrow_lw, mut=18)
    # 4 -> 5
    arrow(ax, (7.0, m4_y), (7.0, m4_y - 0.30 + 0.04),
          color=arrow_color, lw=arrow_lw, mut=18)
    # 5 -> 6
    arrow(ax, (7.0, m5_y), (7.0, m5_y - 0.30 + 0.04),
          color=GOLD, lw=arrow_lw + 0.3, mut=20)

    return fig


def main():
    img = Image.open(SAMPLE_BCC_PATH).convert("RGB")
    lesion_thumb = np.array(img.resize((256, 256)))

    gray = img.convert("L").resize((N_MKT, N_MKT))
    lesion_gray_64 = np.array(gray, dtype=float)

    print("Building Krawtchouk K0 matrix (~5 s)...")
    K0 = build_K0(N_MKT, P_MKT)
    print(f"K0 shape: {K0.shape}")

    print("Composing Figure 6 v4 (modular cards)...")
    fig = render(K0, lesion_gray_64, lesion_thumb)
    fig.savefig(OUT_PIPELINE, dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.10)
    plt.close(fig)
    print(f"-> {OUT_PIPELINE}")
    if OUT_PAPER.parent.exists():
        shutil.copy(OUT_PIPELINE, OUT_PAPER)
        print(f"-> {OUT_PAPER}")


if __name__ == "__main__":
    main()
