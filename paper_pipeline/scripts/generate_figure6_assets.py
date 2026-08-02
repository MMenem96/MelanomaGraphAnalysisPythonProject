"""Generate the small PNG assets that the TikZ Figure 6 embeds:

  fig6_input.png         dermoscopic input (RGB, 256x256)
  fig6_mask.png          binary lesion mask
  fig6_hairmask.png      hair-removal mask
  fig6_preprocessed.png  post-Telea RGB
  fig6_downsampled.png   64x64 grayscale input to MKT
  fig6_mkt_lowpass.png   MKT magnitude map at λ_low_pass
  fig6_mkt_highpass.png  MKT magnitude map at λ_high_pass
  fig6_mkt_dft.png       MKT magnitude map at λ_dft
  fig6_mkt_oddharm.png   MKT magnitude map at λ_odd_harmonic
  fig6_resnet_block.png  small ResNet-50 schematic chip
  fig6_classifier_chip.png  classifier icon chip

All PNGs go to images/ in the paper folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.special import comb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_pipeline.pipeline.mkt_lambda import get_lambda_fn


OUT_DIR = Path("/Users/mmoniem96/Desktop/Work/Master/"
               "Mohamed_Shoieb_Master_Paper_ElsevierV7/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_BCC_PATH = (PROJECT_ROOT / "data" / "canonical" / "bcc_segmented"
                    / "ISIC_0024331.png")
PAPER_IMG_DIR = OUT_DIR
PREPROC_ORIG = PAPER_IMG_DIR / "preprocessing_original_image.png"
PREPROC_MASK = PAPER_IMG_DIR / "preprocessing_binary_mask.png"
PREPROC_RESULT = PAPER_IMG_DIR / "preprocessing_result.png"
PREPROC_HAIR = PAPER_IMG_DIR / "preprocessing_hair_detection.png"

N_MKT = 64
P_MKT = 0.5


# ---------- Krawtchouk transform (same as rich generator) -------------------
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
    mag = np.log1p(mag)
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag


def save_square(arr, out_path, *, cmap=None, dpi=200, border_color=None,
                 size_px=320):
    """Render array as a clean square PNG (no axis, optional border)."""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=dpi)
    ax.imshow(arr, cmap=cmap)
    ax.set_xticks([]); ax.set_yticks([])
    if border_color:
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
    else:
        ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    plt.close(fig)


def main():
    # Load source lesion
    img = Image.open(SAMPLE_BCC_PATH).convert("RGB")
    gray = img.convert("L").resize((N_MKT, N_MKT))
    lesion_gray_64 = np.array(gray, dtype=float)

    # 1. Re-export the preprocessing thumbnails with clean names (just copy
    #    them so the TikZ figure references stable filenames).
    if PREPROC_ORIG.exists():
        Image.open(PREPROC_ORIG).convert("RGB").save(OUT_DIR / "fig6_input.png")
    if PREPROC_MASK.exists():
        Image.open(PREPROC_MASK).convert("L").save(OUT_DIR / "fig6_mask.png")
    if PREPROC_HAIR.exists():
        Image.open(PREPROC_HAIR).convert("RGB").save(
            OUT_DIR / "fig6_hairmask.png")
    if PREPROC_RESULT.exists():
        Image.open(PREPROC_RESULT).convert("RGB").save(
            OUT_DIR / "fig6_preprocessed.png")

    # 2. 64x64 grayscale downsampled (input to MKT)
    save_square(lesion_gray_64, OUT_DIR / "fig6_downsampled.png",
                cmap="gray")

    # 3. Build K0 and the four MKT magnitude maps
    print("Building Krawtchouk K0 matrix (~5 sec)...")
    K0 = build_K0(N_MKT, P_MKT)

    print("Rendering MKT magnitude maps...")
    for lam_short, out_name in [
        ("low_pass", "fig6_mkt_lowpass.png"),
        ("high_pass", "fig6_mkt_highpass.png"),
        ("dft", "fig6_mkt_dft.png"),
        ("odd_harmonic", "fig6_mkt_oddharm.png"),
    ]:
        mag = mkt_magnitude_map(lesion_gray_64, lam_short, K0)
        save_square(mag, OUT_DIR / out_name, cmap="magma")

    print(f"Assets written to {OUT_DIR}")


if __name__ == "__main__":
    main()
