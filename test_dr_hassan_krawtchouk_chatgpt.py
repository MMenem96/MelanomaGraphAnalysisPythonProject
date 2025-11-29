"""
Stable MDFKT pipeline with visualization and orthonormality checks.
- Loads image from local path (grayscale)
- Resizes to NxN
- Builds orthonormal K0 via binomial-sum + QR
- Applies separable 2D MDFKT with separable lambda (vector)
- Extracts low-order MxM feature block and saves feature vector
- Visualizes original / reconstructed / MDFKT magnitude/phase / real/imag
- Prints orthonormality checks (K0 @ K0.T and K0.T @ K0)
"""

import numpy as np
from math import comb
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os

# === user image path (your local file) ===
INPUT_IMAGE_PATH = "data/bcc_segmented/ISIC_0024345_segmented.png"

# === parameters ===
N = 64          # transform size (NxN). choose 32/64/128 depending on needs
M = 8           # low-order block size to extract (MxM)
p = 0.5         # Krawtchouk probability parameter
LAMBDA_MODE = "inverse"  # "inverse" gives lambda_k = 1/(k+1) by default

# -------------------------
# Krawtchouk polynomial helpers (binomial-sum stable form)
# -------------------------
def krawtchouk_poly(n, x, N, p):
    s = 0.0
    for j in range(0, n + 1):
        s += ((-1) ** j) * comb(x, j) * comb(N - x, n - j) * ((p / (1 - p)) ** j)
    return s

def normalized_krawtchouk(n, x, N, p):
    # weight w_x = C(N, x) p^x (1-p)^(N-x)
    w_x = comb(N, x) * (p ** x) * ((1 - p) ** (N - x))
    # norm for order n: sqrt(C(N, n) p^n (1-p)^(N-n))
    denom = np.sqrt(comb(N, n) * (p ** n) * ((1 - p) ** (N - n)))
    return krawtchouk_poly(n, x, N, p) * np.sqrt(w_x) / denom

def compute_K0_matrix(N, p):
    """
    Build orthonormal K0 (N x N) using normalized_krawtchouk and QR orthonormalization.
    Note: follows your supervisor's convention using normalized_krawtchouk(..., N-1, p).
    """
    K = np.zeros((N, N), dtype=float)
    for n in range(N):
        for x in range(N):
            K[n, x] = normalized_krawtchouk(n, x, N - 1, p)
    # orthonormalize rows via QR on transpose
    Q, R = np.linalg.qr(K.T)
    K0 = Q.T
    return K0

# -------------------------
# Lambda builder
# -------------------------
def build_lambda_vector(N, mode="inverse", alpha=1, custom=None):
    n = np.arange(N, dtype=float)
    if mode == "inverse":
        return 1.0 / (n + 1.0)
    raise ValueError("unknown lambda mode")

# -------------------------
# 2D MDFKT (separable)
# -------------------------
def apply_2D_MDFKT(K0, lambda_vec, f):
    lambda_vec = np.asarray(lambda_vec)
    temp = K0 @ f
    temp = lambda_vec[:, None] * temp
    Y = temp @ K0.T
    return Y

def inverse_2D_MDFKT(K0, lambda_vec, Y):
    inv_l = 1.0 / np.asarray(lambda_vec)
    temp = Y @ K0
    temp = inv_l[:, None] * temp
    f_rec = K0.T @ temp
    return f_rec

# -------------------------
# Utility: load & preprocess image (grayscale, resize to NxN)
# -------------------------
def load_and_prepare_image(path, N):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize((N, N), Image.LANCZOS)
    arr = np.array(img).astype(np.float64)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr

# -------------------------
# Feature extraction pipeline
# -------------------------
def extract_features_from_image(path, N=64, M=8, p=0.5, lambda_mode="inverse"):
    f = load_and_prepare_image(path, N)
    K0 = compute_K0_matrix(N, p)
    lam = build_lambda_vector(N, mode=lambda_mode)
    Y = apply_2D_MDFKT(K0, lam, f)
    block = Y[:M, :M]
    feat = block.flatten().astype(np.float64)
    nrm = np.linalg.norm(feat)
    if nrm > 0:
        feat = feat / nrm
    return {"K0": K0, "Y": Y, "feature": feat, "f": f, "lambda": lam}

# -------------------------
# Visualization & orthonormality helpers
# -------------------------
def orthonormality_metrics(K0):
    I1 = K0 @ K0.T
    I2 = K0.T @ K0
    dev1 = np.max(np.abs(I1 - np.eye(K0.shape[0])))
    dev2 = np.max(np.abs(I2 - np.eye(K0.shape[0])))
    return dev1, dev2

def visualize_results(f, f_rec, Y, K0, N):
    plt.figure(figsize=(15,10))

    plt.subplot(2,3,1)
    plt.imshow(f, cmap='gray', origin='lower')
    plt.title("Original (grayscale)")
    plt.colorbar()

    plt.subplot(2,3,2)
    plt.imshow(f_rec.real, cmap='gray', origin='lower')
    plt.title("Reconstructed (real)")
    plt.colorbar()

    plt.subplot(2,3,3)
    plt.imshow(np.abs(Y), cmap='magma', origin='lower')
    plt.title("MDFKT Magnitude")
    plt.colorbar()

    plt.subplot(2,3,4)
    plt.imshow(np.angle(Y), cmap='twilight', origin='lower')
    plt.title("MDFKT Phase")
    plt.colorbar()

    plt.subplot(2,3,5)
    plt.imshow(Y.real, cmap='viridis', origin='lower')
    plt.title("MDFKT Real part")
    plt.colorbar()

    plt.subplot(2,3,6)
    plt.imshow(Y.imag, cmap='plasma', origin='lower')
    plt.title("MDFKT Imag part")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # show K0 as dataframe (rounded)
    try:
        import pandas as pd
        df = pd.DataFrame(np.round(K0, 6))
        print("K0 (rounded):")
        # 1) print a readable portion to console
        if df.size <= 2000:
            print(df.to_string())
        else:
            print("K0 (first 12x12):")
            print(df.iloc[:12, :12].to_string())

        # 2) save full K0 as CSV for inspection
        csv_path = "K0_matrix.csv"
        df.to_csv(csv_path, index=False)
        print(f"K0 saved to {csv_path}")

        # 3) save a heatmap image of K0 (openable in VS)
        plt.figure(figsize=(6,6))
        plt.imshow(K0, cmap='viridis', origin='lower')
        plt.colorbar()
        plt.title("K0 heatmap")
        heatmap_path = "K0_heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"K0 heatmap saved to {heatmap_path}")
    except Exception:
        print("K0 matrix (first 6 rows):")
        print(np.round(K0[:6, :6], 6))

# -------------------------
# Main demo
# -------------------------
if __name__ == "__main__":
    out = extract_features_from_image(INPUT_IMAGE_PATH, N=N, M=M, p=p, lambda_mode=LAMBDA_MODE)
    K0, Y, feat, f, lam = out["K0"], out["Y"], out["feature"], out["f"], out["lambda"]

    dev1, dev2 = orthonormality_metrics(K0)
    print("K0 orthonormality (K0@K0.T) max dev:", dev1)
    print("K0 orthonormality (K0.T@K0) max dev:", dev2)

    # reconstruction check
    f_rec = inverse_2D_MDFKT(K0, lam, Y)
    rec_err = np.max(np.abs(f - f_rec))
    print("reconstruction error (max abs):", rec_err)

    # visualize
    visualize_results(f, f_rec, Y, K0, N)

    # feature info
    print("feature length:", feat.size)
    np.save("krawtchouk_feature.npy", feat)
    print("feature saved to krawtchouk_feature.npy")
