"""Figure 6 v2 — proposed_framework_figure.png with embedded lesion samples,
real MKT transform outputs for each of the four λ configurations, and the
locked MLP + odd-harmonic headline block.

Visual goals (per user choices):
  * Embed a real BCC lesion sample (canonical, segmented + preprocessed).
  * Show MKT magnitude maps for all four λ (low_pass / high_pass / dft /
    odd_harmonic) computed on the actual lesion at N=64, p=0.5.
  * Compact 6-metric headline bar with bootstrap CIs and McNemar callout.
  * ARS palette consistent with the paper's preamble.

Run:
    cd /Users/mmoniem96/Desktop/Work/Master/Practical\\ Project/MelanomaGraphAnalysisV3
    python paper_pipeline/scripts/regenerate_figure6_rich.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from scipy.special import comb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_pipeline.pipeline.mkt_lambda import get_lambda_fn

# Output destinations
OUT_PIPELINE = (PROJECT_ROOT / "paper_pipeline" / "output" / "figures"
                / "proposed_framework_figure.png")
OUT_PAPER    = Path("/Users/mmoniem96/Desktop/Work/Master/"
                    "Mohamed_Shoieb_Master_Paper_ElsevierV7/images/"
                    "proposed_framework_figure.png")

# Real sample lesion (canonical BCC)
SAMPLE_BCC_PATH = (PROJECT_ROOT / "data" / "canonical" / "bcc_segmented"
                    / "ISIC_0024331.png")
PAPER_IMG_DIR = OUT_PAPER.parent
PREPROC_ORIG = PAPER_IMG_DIR / "preprocessing_original_image.png"
PREPROC_MASK = PAPER_IMG_DIR / "preprocessing_binary_mask.png"
PREPROC_RESULT = PAPER_IMG_DIR / "preprocessing_result.png"

# Palette (matches paper preamble)
NAVY      = "#2C3E50"
NAVY_TINT = "#F4F6F8"
GOLD      = "#E67E22"
GOLD_TINT = "#FDEBD0"
GREY_LT   = "#ECF0F1"
TEXT_COL  = "#34495E"
WHITE     = "#FFFFFF"

N_MKT = 64
P_MKT = 0.5


# ---------- Krawtchouk transform (extracted from src/conventional_features.py)
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
    w_x = comb(N, x, exact=False) * (p ** x) * ((1 - p) ** (N - x))
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
    # Compress for display
    mag = np.log1p(mag)
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag


# ---------- Drawing helpers --------------------------------------------------
def rbox(ax, x, y, w, h, *, fill=NAVY_TINT, edge=NAVY, lw=1.0, alpha=1.0,
         rounding=0.05):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={rounding}",
                          linewidth=lw, edgecolor=edge, facecolor=fill,
                          alpha=alpha)
    ax.add_patch(box)


def text(ax, x, y, txt, *, fs=10, color=TEXT_COL, weight="normal",
         ha="center", va="center", family="serif"):
    ax.text(x, y, txt, ha=ha, va=va, color=color, fontsize=fs,
            fontweight=weight, family=family)


def arrow(ax, p1, p2, *, color=NAVY, lw=1.4, style="-|>", mut=15,
          connection="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=color,
                                  linewidth=lw, mutation_scale=mut,
                                  connectionstyle=connection))


def imshow_in_box(ax, image, x, y, w, h, *, cmap=None, border=NAVY, lw=1.5):
    # Use axes inset to place image in figure coordinates
    iax = ax.inset_axes([x, y, w, h], transform=ax.transData)
    if cmap is None:
        iax.imshow(image)
    else:
        iax.imshow(image, cmap=cmap)
    iax.set_xticks([]); iax.set_yticks([])
    for spine in iax.spines.values():
        spine.set_edgecolor(border)
        spine.set_linewidth(lw)


# ---------- Panel (a) — per-image feature extraction -------------------------
def panel_a(ax, K0, lesion_gray_64, lesion_rgb_thumb):
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    ax.set_aspect("equal"); ax.axis("off")

    # Panel title
    text(ax, 0.1, 8.6, "(a)  Per-image feature extraction", fs=13,
         color=NAVY, weight="bold", ha="left")

    # ---- Row 1: full preprocessing chain ------------------------------------
    # Original -> Mask -> ROI -> Hair detection -> Cleaned (Telea) -> 64x64
    y_top = 6.5
    h_thumb = 1.35; w_thumb = 1.35
    arrow_gap = 0.10

    # Helper: load image with fallback
    def _load(path, mode="RGB", fallback=lesion_rgb_thumb):
        try:
            return np.array(Image.open(path).convert(mode))
        except Exception:
            return fallback

    # Asset paths for the 5 preprocessing steps
    PREPROC_ROI = PAPER_IMG_DIR / "preprocessing_roi.png"
    PREPROC_HAIR_LOCAL = PAPER_IMG_DIR / "preprocessing_hair_detection.png"

    # Six preprocessing thumbnails: original, mask, ROI, hair map, cleaned,
    # downsampled 64x64. Each is followed by an arrow into the next.
    steps = [
        # (image, cmap, border, title, subtitle, is_headline_step)
        (_load(PREPROC_ORIG, "RGB"),               None,   NAVY, "Dermoscopic", r"$224{\times}224{\times}3$",                 False),
        (_load(PREPROC_MASK, "L", np.zeros_like(lesion_gray_64)), "gray", NAVY, "Lesion mask",  r"$M \in \{0,1\}$",                          False),
        (_load(PREPROC_ROI, "RGB"),                 None,   NAVY, "ROI",          r"$I \odot M$",                                 False),
        (_load(PREPROC_HAIR_LOCAL, "RGB"),          None,   NAVY, "Hair map",     r"Top/Black-hat",                               False),
        (_load(PREPROC_RESULT, "RGB"),              None,   GOLD, "Cleaned",      r"Telea inpaint",                               True),
        (lesion_gray_64,                            "gray", NAVY, "Downsample",   r"$64{\times}64$, gray",                        False),
    ]

    # Compute x positions: total 6 thumbs + 5 arrows + 1 final arrow into MKT
    # Available x: 0 .. ~9 for the strip, then MKT box from ~9.1 onward
    x = 0.30
    centres = []
    for i, (img, cmap, border, title, sub, is_head) in enumerate(steps):
        lw = 2.2 if is_head else 1.2
        imshow_in_box(ax, img, x, y_top, w_thumb, h_thumb,
                      cmap=cmap, border=border, lw=lw)
        text(ax, x + w_thumb / 2, y_top - 0.27, title,
             fs=8.5, weight="bold",
             color=GOLD if is_head else TEXT_COL)
        text(ax, x + w_thumb / 2, y_top - 0.52, sub,
             fs=7.5, color=NAVY)
        centres.append(x + w_thumb / 2)
        if i < len(steps) - 1:
            arrow(ax,
                  (x + w_thumb + 0.02, y_top + h_thumb / 2),
                  (x + w_thumb + arrow_gap + 0.18, y_top + h_thumb / 2),
                  lw=1.0)
        x += w_thumb + arrow_gap + 0.20

    # MKT operator box wraps everything below the strip + visualises the 4 outputs
    mkt_x0 = x + 0.10
    rbox(ax, mkt_x0, y_top - 0.85, 16.0 - mkt_x0 - 0.1, 2.55,
         fill=GOLD_TINT, edge=GOLD, lw=1.6)
    text(ax, (mkt_x0 + 16.0 - 0.1) / 2, y_top + 1.55,
         r"$\boldsymbol{\Psi}\!=\!\mathcal{K}_\Lambda\,I\,\mathcal{K}_\Lambda^\top$",
         fs=11.5, color=NAVY, weight="bold")
    text(ax, (mkt_x0 + 16.0 - 0.1) / 2, y_top + 1.15,
         r"MKT  $|\,N{=}64,\ p{=}0.5$",
         fs=9, color=NAVY)
    arrow(ax, (centres[-1] + w_thumb / 2 + 0.02, y_top + h_thumb / 2),
              (mkt_x0 + 0.05, y_top + h_thumb / 2), lw=1.0)

    # ---- Four λ panels (MKT outputs) inside the MKT box ----
    # Compute each magnitude map
    # Per-lambda colormaps so each MKT panel is visually distinct.
    lam_specs = [
        ("low_pass",     r"$\lambda_k=1/(k{+}1)$",         "low-pass",          "viridis"),
        ("high_pass",    r"$\lambda_k=1/(N{-}k)$",          "high-pass",         "plasma"),
        ("dft",          r"$\lambda_k=e^{i 2\pi k/N}$",     "DFT-like",          "cividis"),
        ("odd_harmonic", r"$\lambda_k=e^{i(2k{+}1)\pi/N}$", r"odd-harm.$\star$", "inferno"),
    ]
    panel_x = [9.6, 11.2, 12.8, 14.4]
    panel_y = y_top - 0.4
    panel_w = 1.35
    panel_h = 1.35
    for (lam, eq, label, cmap), px in zip(lam_specs, panel_x):
        mag = mkt_magnitude_map(lesion_gray_64, lam, K0)
        is_head = (lam == "odd_harmonic")
        imshow_in_box(ax, mag, px, panel_y, panel_w, panel_h,
                       cmap=cmap,
                       border=GOLD if is_head else NAVY,
                       lw=2.6 if is_head else 1.0)
        text(ax, px + panel_w / 2, panel_y - 0.18, label,
             fs=8.5, weight="bold" if is_head else "normal",
             color=GOLD if is_head else TEXT_COL)
        text(ax, px + panel_w / 2, panel_y - 0.36, eq, fs=7.5, color=NAVY)

    # ---- Row 2: feature families ----
    y_mid = 3.5
    # MKT features extracted
    rbox(ax, 0.4, y_mid, 4.6, 1.4, fill=GOLD_TINT, edge=GOLD, lw=1.3)
    text(ax, 2.7, y_mid + 1.05, "MKT statistics",
         fs=10, weight="bold", color=GOLD)
    text(ax, 2.7, y_mid + 0.65,
         r"$\Re,\Im,|\cdot|,\arg$ per channel",
         fs=8.5)
    text(ax, 2.7, y_mid + 0.30,
         "max, min, mean, std, energy",
         fs=8)
    text(ax, 2.7, y_mid + 0.05, "76 features per $\\lambda$",
         fs=8.5, color=NAVY, weight="bold")

    # Handcrafted
    rbox(ax, 5.6, y_mid, 4.6, 1.4, fill=NAVY_TINT, edge=NAVY, lw=1.2)
    text(ax, 7.9, y_mid + 1.05, "Handcrafted descriptors",
         fs=10, weight="bold", color=NAVY)
    text(ax, 7.9, y_mid + 0.65,
         r"Geometric (18)  $\cdot$  Color (138)  $\cdot$  Texture (229)",
         fs=8.5)
    text(ax, 7.9, y_mid + 0.30,
         "RGB/HSV/CIELAB stats, GLCM, LBP, Gabor",
         fs=8)
    text(ax, 7.9, y_mid + 0.05, "385 features", fs=8.5, color=NAVY,
         weight="bold")

    # CNN
    rbox(ax, 10.8, y_mid, 5.0, 1.4, fill=NAVY_TINT, edge=NAVY, lw=1.2)
    text(ax, 13.3, y_mid + 1.05, "Frozen ResNet-50",
         fs=10, weight="bold", color=NAVY)
    text(ax, 13.3, y_mid + 0.65,
         "ImageNet weights, no fine-tuning",
         fs=8.5)
    text(ax, 13.3, y_mid + 0.30,
         "global-avg-pool $\\rightarrow$ 2{,}048-d descriptor",
         fs=8)
    text(ax, 13.3, y_mid + 0.05, "2{,}048 features", fs=8.5, color=NAVY,
         weight="bold")

    # Arrows from row 1 to row 2
    arrow(ax, (12.6, y_top - 0.65), (2.7, y_mid + 1.4),
          connection="arc3,rad=-0.15", color=GOLD)
    arrow(ax, (5.8, y_top), (7.9, y_mid + 1.4))
    arrow(ax, (5.8, y_top), (13.3, y_mid + 1.4),
          connection="arc3,rad=0.05")

    # ---- Row 3: hybrid vector ----
    y_bot = 1.0
    rbox(ax, 0.4, y_bot, 15.4, 1.6, fill=GOLD, edge=NAVY, lw=2.0,
         rounding=0.08)
    text(ax, 8.1, y_bot + 1.20, "Hybrid feature vector",
         fs=12, weight="bold", color=WHITE)
    text(ax, 8.1, y_bot + 0.70,
         r"$\mathbf{x} \in \mathbb{R}^{2{,}509}$",
         fs=12.5, color=WHITE, weight="bold")
    text(ax, 8.1, y_bot + 0.25,
         r"$=$  MKT (76)  $\oplus$  Handcrafted (385)  $\oplus$  ResNet-50 (2{,}048)",
         fs=10, color=WHITE)
    arrow(ax, (2.7, y_mid), (4.5, y_bot + 1.6), color=GOLD)
    arrow(ax, (7.9, y_mid), (8.1, y_bot + 1.6))
    arrow(ax, (13.3, y_mid), (12.0, y_bot + 1.6))


# ---------- Panel (b) — dataset-level protocol ------------------------------
def db_cylinder(ax, cx, cy, w, h, *, fill=NAVY_TINT, edge=NAVY, lw=1.0):
    """Draw a database/dataset cylinder centred at (cx, cy) with width w and
    total height h. The shape is the classic flow-chart 'storage' icon:
    an ellipse on top + vertical sides + a half-ellipse arc at the bottom."""
    from matplotlib.patches import Ellipse, Polygon, Arc
    # Ellipse heights for top/bottom (the 'lid' of the cylinder)
    eh = w * 0.32
    # Body rectangle filled
    body = Polygon([(cx - w/2, cy - h/2 + eh/2),
                     (cx + w/2, cy - h/2 + eh/2),
                     (cx + w/2, cy + h/2 - eh/2),
                     (cx - w/2, cy + h/2 - eh/2)],
                    closed=True, facecolor=fill, edgecolor="none")
    ax.add_patch(body)
    # Top ellipse (the visible top opening of the cylinder)
    top = Ellipse((cx, cy + h/2 - eh/2), w, eh,
                   facecolor=fill, edgecolor=edge, linewidth=lw)
    ax.add_patch(top)
    # Bottom front arc (only the front half of the bottom ellipse is visible)
    bottom = Arc((cx, cy - h/2 + eh/2), w, eh,
                  theta1=180, theta2=360,
                  edgecolor=edge, linewidth=lw)
    ax.add_patch(bottom)
    # Left and right vertical side lines
    ax.plot([cx - w/2, cx - w/2],
            [cy - h/2 + eh/2, cy + h/2 - eh/2],
            color=edge, linewidth=lw)
    ax.plot([cx + w/2, cx + w/2],
            [cy - h/2 + eh/2, cy + h/2 - eh/2],
            color=edge, linewidth=lw)
    # Faint horizontal lines on the body to suggest 3D 'rings'
    for ring_offset in [0.18, -0.18]:
        ring = Arc((cx, cy + ring_offset * h), w, eh,
                    theta1=180, theta2=360,
                    edgecolor=edge, linewidth=lw * 0.4, alpha=0.5)
        ax.add_patch(ring)


def panel_b(ax):
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    ax.set_aspect("equal"); ax.axis("off")

    text(ax, 0.1, 8.6, "(b)  Dataset preparation and training protocol",
         fs=13, color=NAVY, weight="bold", ha="left")

    # Row 1: dataset → split → augmentation → SelectKBest
    y = 6.5
    h = 1.5; w = 3.5

    # 1. Dataset — drawn as a database cylinder symbol
    cyl_cx = 0.3 + w/2
    cyl_cy = y + h/2
    cyl_w = w * 0.78
    cyl_h = h * 1.4
    db_cylinder(ax, cyl_cx, cyl_cy, cyl_w, cyl_h, fill=NAVY_TINT, edge=NAVY,
                lw=1.2)
    text(ax, cyl_cx, cyl_cy + 0.10, "HAM10000", fs=10, weight="bold",
         color=NAVY)
    text(ax, cyl_cx, cyl_cy - 0.18, r"514 BCC $+$ 1{,}099 BKL",
         fs=8)
    text(ax, cyl_cx, cyl_cy - 0.42, r"$=\ 1{,}613$ images",
         fs=9, color=NAVY, weight="bold")
    text(ax, cyl_cx, y - 0.20,
         r"\texttt{dx} filtered $\cap$ HAM10000 masks", fs=7.5)

    arrow(ax, (0.3 + w + 0.05, y + h/2), (0.3 + w + 0.55, y + h/2))

    # 2. Stratified split
    rbox(ax, 4.3, y, w, h, fill=NAVY_TINT)
    text(ax, 4.3 + w/2, y + 1.10, "Stratified 80/20 split",
         fs=10, weight="bold", color=NAVY)
    text(ax, 4.3 + w/2, y + 0.75, "on image identifiers",
         fs=8.5)
    text(ax, 4.3 + w/2, y + 0.40,
         r"train 1{,}290  $|$  test 323",
         fs=9, color=NAVY, weight="bold")
    text(ax, 4.3 + w/2, y + 0.10, r"\texttt{random\_state} $= 42$",
         fs=7.5)

    arrow(ax, (4.3 + w + 0.05, y + h/2), (4.3 + w + 0.55, y + h/2))

    # 3. Augmentation
    rbox(ax, 8.3, y, w, h, fill=NAVY_TINT)
    text(ax, 8.3 + w/2, y + 1.10, "BCC-only flip augmentation",
         fs=10, weight="bold", color=NAVY)
    text(ax, 8.3 + w/2, y + 0.75,
         r"411 $\rightarrow$ 879 BCC (h/v-flip)",
         fs=8.5)
    text(ax, 8.3 + w/2, y + 0.40,
         r"balanced train $= 1{,}758$",
         fs=9, color=NAVY, weight="bold")
    text(ax, 8.3 + w/2, y + 0.10, "test set never augmented",
         fs=7.5)

    arrow(ax, (8.3 + w + 0.05, y + h/2), (8.3 + w + 0.55, y + h/2))

    # 4. SelectKBest + RobustScaler
    rbox(ax, 12.3, y, w, h, fill=NAVY_TINT)
    text(ax, 12.3 + w/2, y + 1.10, "Selection $+$ scaling",
         fs=10, weight="bold", color=NAVY)
    text(ax, 12.3 + w/2, y + 0.75, "SelectKBest (MI)",
         fs=8.5)
    text(ax, 12.3 + w/2, y + 0.40, r"360 / 2{,}509",
         fs=9, color=NAVY, weight="bold")
    text(ax, 12.3 + w/2, y + 0.10, "RobustScaler, fit on train",
         fs=7.5)

    # Row 2: 9-classifier sweep
    y2 = 4.4
    rbox(ax, 0.3, y2, 15.5, 1.4, fill=NAVY_TINT)
    text(ax, 8.0, y2 + 1.05,
         "9-classifier sweep on the 360-feature representation",
         fs=10.5, weight="bold", color=NAVY)
    text(ax, 8.0, y2 + 0.65,
         "SVM (RBF) $\\cdot$ LightGBM $\\cdot$ CatBoost $\\cdot$ Gradient Boosting $\\cdot$ Extra Trees",
         fs=8.5)
    text(ax, 8.0, y2 + 0.32,
         r"KNN $\cdot$ Logistic Regression $\cdot$ \textbf{MLP$^\bigstar$} $\cdot$ Deep DNN  ",
         fs=8.5)
    text(ax, 8.0, y2 + 0.05,
         r"Stratified 5-fold CV (dev only)  $\cdot$  random\_state $= 42$  $\cdot$  test set touched once / cell",
         fs=7.5, color=NAVY)
    arrow(ax, (14.05, y), (8.0, y2 + 1.4))

    # Headline result block — compact 6 metrics + 2 CIs + McNemar
    y3 = 0.5
    rbox(ax, 0.3, y3, 15.5, 3.5, fill=GOLD, edge=NAVY, lw=2.2,
         rounding=0.08)
    text(ax, 8.0, y3 + 3.10, "HEADLINE", fs=12, weight="bold",
         color=WHITE)
    text(ax, 8.0, y3 + 2.70,
         r"MLP  $\oplus$  $\lambda_k\!=\!e^{i(2k+1)\pi/N}$  (odd-harmonic, $p\!=\!0.5$, $N\!=\!64$)",
         fs=11, color=WHITE, weight="bold")

    # Metrics grid
    xs = [1.4, 3.8, 6.2, 8.6, 11.0, 13.4]
    keys  = ["Acc",   "Sens",   "Spec",   "Prec",   "F1",     "AUC"]
    vals  = ["93.50%","91.26%","94.55%","88.68%","89.95%","96.95%"]
    for x, k, v in zip(xs, keys, vals):
        text(ax, x + 0.8, y3 + 2.00, k, fs=10.5, color=WHITE, weight="bold")
        text(ax, x + 0.8, y3 + 1.50, v, fs=14, color=WHITE, weight="bold")

    # CI line
    text(ax, 8.0, y3 + 0.95,
         r"95\% bootstrap CI (10{,}000 resamples): "
         r"Acc $\in [90.71, 95.98]$ $\cdot$ AUC $\in [95.13, 98.51]$",
         fs=9, color=WHITE)
    text(ax, 8.0, y3 + 0.55,
         r"McNemar's vs runner-up (LightGBM $+$ low-pass, $93.19\%$): "
         r"exact $p = 1.0$",
         fs=9, color=WHITE)
    text(ax, 8.0, y3 + 0.18,
         r"on a 323-image held-out test set (103 BCC $+$ 220 BKL)",
         fs=9, color=WHITE, weight="bold")
    arrow(ax, (8.0, y2), (8.0, y3 + 3.5), color=NAVY, lw=2.0)


def main():
    # Load and prepare the sample lesion
    img = Image.open(SAMPLE_BCC_PATH).convert("RGB")
    lesion_rgb_thumb = np.array(img.resize((128, 128)))

    # Build the 64×64 grayscale input that the MKT actually sees
    gray = img.convert("L").resize((N_MKT, N_MKT))
    lesion_gray_64 = np.array(gray, dtype=float)

    print("Building Krawtchouk K0 matrix (this takes ~5 seconds)...")
    K0 = build_K0(N_MKT, P_MKT)
    print(f"K0 shape: {K0.shape}")

    print("Composing Figure 6...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 18))
    panel_a(axes[0], K0, lesion_gray_64, lesion_rgb_thumb)
    panel_b(axes[1])
    plt.tight_layout(pad=1.5)
    fig.savefig(OUT_PIPELINE, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"→ {OUT_PIPELINE}")
    if OUT_PAPER.parent.exists():
        shutil.copy(OUT_PIPELINE, OUT_PAPER)
        print(f"→ {OUT_PAPER}")


if __name__ == "__main__":
    main()
