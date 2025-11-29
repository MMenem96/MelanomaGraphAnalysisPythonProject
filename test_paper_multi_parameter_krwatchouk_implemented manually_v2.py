import numpy as np
import numpy as _np

from scipy.special import hyp2f1, gammaln, poch
import imageio
from PIL import Image

# Path to uploaded image (provided in this conversation)
UPLOADED_IMAGE_PATH = "data/bcc_segmented/ISIC_0024345_segmented.png"

# Maximum dimension to prevent numerical issues
MAX_DIMENSION = 32

# -------------------------
# Helper: Resize image if needed
# -------------------------
def resize_image_if_needed(img, max_dim=MAX_DIMENSION):
    """
    Resize image if either dimension exceeds max_dim.
    Maintains aspect ratio and uses high-quality resampling.
    
    Args:
        img: numpy array (H, W) or (H, W, C)
        max_dim: maximum allowed dimension
    
    Returns:
        resized image, original_shape, was_resized
    """
    original_shape = img.shape
    H, W = img.shape[:2]
    
    # Check if resizing is needed
    if H <= max_dim and W <= max_dim:
        return img, original_shape, False
    
    # Calculate new dimensions maintaining aspect ratio
    scale = max_dim / max(H, W)
    new_H = int(H * scale)
    new_W = int(W * scale)
    
    # Convert to PIL for high-quality resizing
    if img.ndim == 2:
        # Grayscale
        pil_img = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8))
        pil_resized = pil_img.resize((new_W, new_H), Image.LANCZOS)
        resized = np.array(pil_resized).astype(np.float64)
        if img.max() <= 1.0:
            resized = resized / 255.0
    else:
        # RGB/RGBA
        if img.max() <= 1.0:
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
        else:
            pil_img = Image.fromarray(img.astype(np.uint8))
        pil_resized = pil_img.resize((new_W, new_H), Image.LANCZOS)
        resized = np.array(pil_resized).astype(np.float64)
        if img.max() <= 1.0:
            resized = resized / 255.0
    
    print(f"Image resized from {original_shape[:2]} to {resized.shape[:2]} for numerical stability")
    return resized, original_shape, True


# -------------------------
# Helper: safe log binomial
# -------------------------
def compute_log_binomial_coefficient(total_count, chosen_count):
    return (
        gammaln(total_count + 1)
        - gammaln(chosen_count + 1)
        - gammaln(total_count - chosen_count + 1)
    )


# ====================================================
# 1D weighted Krawtchouk basis φ_n(i)  -- Eq.(4)
# ====================================================

def compute_weighted_krawtchouk_basis_1d(num_points: int, probability_p: float):
    if not (0.0 < probability_p < 1.0):
        raise ValueError("probability_p (p) must be in (0,1).")
    N = int(num_points)

    orders_n = np.arange(N)[:, None]     # (N,1)
    positions_i = np.arange(N)[None, :]  # (1,N)

    # --- compute 2F1(-n,-i; -N+1; z) by direct finite sum (stable recurrence) ---
    z = 1.0 / probability_p
    hypergeom_matrix = np.zeros((N, N), dtype=np.float64)
    for n in range(N):
        for i in range(N):
            K = min(n, i)
            s = 1.0  # k=0 term
            term = 1.0
            # recurrence: term_{k} = term_{k-1} * ((-n + k-1)*(-i + k-1)) / ((-N+1 + k-1) * k) * z
            for k in range(1, K + 1):
                num1 = -n + (k - 1)
                num2 = -i + (k - 1)
                den  = - (N - 1) + (k - 1)  # -N+1 + (k-1)
                # safe multiply/divide as floats
                term = term * (num1 * num2) / (den * k) * z
                s += term
            hypergeom_matrix[n, i] = s
    # --- end hypergeom ---

    log_binom = compute_log_binomial_coefficient(N - 1, positions_i)  # (1,N)
    log_numer = (
        log_binom
        + positions_i * np.log(probability_p)
        + (N - 1 - positions_i) * np.log(1.0 - probability_p)
    )
    sqrt_numer = np.exp(0.5 * log_numer)  # (1,N)

    log_term1 = orders_n * np.log(np.abs((probability_p - 1.0) / probability_p))  # (N,1)
    log_nfactorial = gammaln(orders_n + 1)  # (N,1)

    # exact closed-form for (-N+1)_n when start is negative integer:
    orders_n_int = orders_n.astype(int)  # (N,1)
    log_abs_poch_minus = gammaln(N) - gammaln(N - orders_n_int)   # (N,1)
    sign_poch_minus = np.where((orders_n_int % 2) == 1, -1.0, 1.0)  # (N,1)

    # combine log denominator (magnitude)
    log_denom = log_term1 + log_nfactorial - log_abs_poch_minus  # (N,1)

    # sqrt argument log = log_numer (1,N) - log_denom (N,1) -> broadcasting to (N,N)
    log_sqrt_arg = log_numer - log_denom

    # clip to avoid exp overflow
    LOG_SAFE_MAX = 700.0
    log_sqrt_arg_clipped = np.clip(log_sqrt_arg, -np.inf, LOG_SAFE_MAX)
    with np.errstate(over='ignore', invalid='ignore'):
        sqrt_factor = np.exp(0.5 * log_sqrt_arg_clipped)  # (N,N)

    # sign correction: from Pochhammer and from ((p-1)/p)^n if base negative
    sign_from_poch = np.where(sign_poch_minus < 0, -1.0, 1.0)  # (N,1)
    sign_from_pfactor = np.ones_like(orders_n, dtype=float)   # (N,1)
    if ((probability_p - 1.0) / probability_p) < 0:
        sign_from_pfactor = np.where((orders_n % 2) == 1, -1.0, 1.0)
    sign_correction = sign_from_poch * sign_from_pfactor  # (N,1)

    # assemble phi
    phi = hypergeom_matrix * sqrt_numer * sqrt_factor * sign_correction  # (N,N)
    phi = np.real_if_close(phi, tol=1e-9)
    return phi




# ====================================================
# Build lambda vector (multiparameter)
# ====================================================
def build_lambda_vector(num_orders: int, mode: str = "inverse", alpha: float = 0.12, custom: np.ndarray = None):
    n = np.arange(num_orders, dtype=float)
    if mode == "ones":
        return np.ones(num_orders, dtype=float)
    if mode == "decay":
        return np.exp(-alpha * n)
    if mode == "sign":
        return np.where((n % 2) == 0, 1.0, -1.0)
    if mode == "custom":
        if custom is None:
            raise ValueError("custom lambda vector required for mode='custom'")
        arr = np.asarray(custom, dtype=float)
        if arr.shape[0] != num_orders:
            raise ValueError("custom lambda must have length num_orders")
        return arr
    if mode == "inverse":
        return 1.0 / (n + 1.0)
    raise ValueError("unknown lambda mode")


# ====================================================
# Separable 2D MDFKT + apply λ as outer product
# ====================================================
def apply_separable_mdfkt_2d(image, probability_p=0.5, lambda_mode="inverse", lambda_alpha=0.12, max_dim=MAX_DIMENSION):
    image = np.asarray(image, dtype=np.float64)
    
    # Resize if needed
    image, original_shape, was_resized = resize_image_if_needed(image, max_dim=max_dim)
    
    H, W = image.shape[:2]

    basis_rows_phi = compute_weighted_krawtchouk_basis_1d(H, probability_p)  # (H,H)
    basis_cols_phi = compute_weighted_krawtchouk_basis_1d(W, probability_p)  # (W,W)

    lambda_rows = build_lambda_vector(H, mode=lambda_mode, alpha=lambda_alpha)
    lambda_cols = build_lambda_vector(W, mode=lambda_mode, alpha=lambda_alpha)

    def transform_channel(channel_2d):
        row_projected = basis_rows_phi @ channel_2d  # (H,W)
        coeffs = row_projected @ basis_cols_phi.T    # (H,W)
        coeffs = (lambda_rows[:, None] * lambda_cols[None, :]) * coeffs
        return coeffs

    if image.ndim == 2:
        coeffs = transform_channel(image)
    else:
        channels = []
        for ch in range(image.shape[2]):
            channels.append(transform_channel(image[:, :, ch].astype(np.float64)))
        coeffs = np.stack(channels, axis=-1)  # (H,W,C)

    return coeffs, basis_rows_phi, basis_cols_phi, lambda_rows, lambda_cols


# ====================================================
# Build full K_lambda (explicit Q Λ Q^T) using Kronecker
# ====================================================
def build_full_K_lambda(basis_rows_phi, basis_cols_phi, lambda_rows, lambda_cols):
    H = basis_rows_phi.shape[0]
    W = basis_cols_phi.shape[0]

    Q_full = np.kron(basis_cols_phi, basis_rows_phi)        # (H*W, H*W)
    Lambda_diag = np.kron(lambda_cols, lambda_rows)        # length H*W
    Lambda_full = np.diag(Lambda_diag)                     # (H*W, H*W)
    K_lambda = Q_full @ Lambda_full @ Q_full.T             # (H*W, H*W)
    return Q_full, Lambda_full, K_lambda


# ====================================================
# Feature extraction: low-order MxM block + L2 normalize
# ====================================================
def extract_krawtchouk_features_from_coeffs(coeffs, max_order_block=8, l2_normalize=True):
    H, W = coeffs.shape[:2]
    Mr = min(max_order_block, H)
    Mc = min(max_order_block, W)
    block = coeffs[:Mr, :Mc]
    vec = block.flatten().astype(float)
    if l2_normalize:
        nrm = np.linalg.norm(vec)
        if nrm > 0:
            vec = vec / nrm
    return vec


# ====================================================
# Validation helpers
# ====================================================
def orthonormality_metrics(phi):
    G1 = phi @ phi.T
    G2 = phi.T @ phi
    max_off1 = np.max(np.abs(G1 - np.eye(phi.shape[0])))
    max_off2 = np.max(np.abs(G2 - np.eye(phi.shape[0])))
    imag_max = np.max(np.abs(np.imag(phi)))
    return max_off1, max_off2, imag_max


# ====================================================
# Demo & validation run (uses uploaded image path)
# ====================================================
if __name__ == "__main__":
    p = 0.5
    # load image (grayscale or rgb)
    img = imageio.imread(UPLOADED_IMAGE_PATH)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    # convert to float in [0,1] if integer image
    if np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.float64) / np.iinfo(img.dtype).max
    else:
        img = img.astype(np.float64)

    print("Original image shape:", img.shape)

    # separable path (now with automatic resizing)
    coeffs, phi_r, phi_c, lam_r, lam_c = apply_separable_mdfkt_2d(
        img, 
        probability_p=p, 
        lambda_mode="inverse",
        max_dim=MAX_DIMENSION  # You can adjust this value
    )
    print("coeffs shape:", coeffs.shape)

    # orthonormality checks
    r1, r2, rim = orthonormality_metrics(phi_r)
    c1, c2, cim = orthonormality_metrics(phi_c)
    print(f"phi_rows max_offdiag (phi@phi.T): {r1:.3e}, (phi.T@phi): {r2:.3e}, imag_max: {rim:.3e}")
    print(f"phi_cols max_offdiag (phi@phi.T): {c1:.3e}, (phi.T@phi): {c2:.3e}, imag_max: {cim:.3e}")

    feats = extract_krawtchouk_features_from_coeffs(coeffs if coeffs.ndim==2 else coeffs[...,0], max_order_block=8)
    print("extracted feature length:", feats.size)