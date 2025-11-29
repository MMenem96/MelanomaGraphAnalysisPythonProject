import numpy as np
from scipy.special import hyp2f1, gammaln, poch

# Path to your uploaded paper (for reference)
UPLOADED_PAPER_PATH = "/mnt/data/dcf6c029-db88-48eb-8a9e-1b548308b81f.png"


# -------------------------
# Helper: safe log binomial
# -------------------------
def compute_log_binomial_coefficient(total_count, chosen_count):
    # log C(n, k) = gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)
    return (
        gammaln(total_count + 1)
        - gammaln(chosen_count + 1)
        - gammaln(total_count - chosen_count + 1)
    )


# ====================================================
# 1D weighted Krawtchouk basis φ_n(i)  -- exact Eq.(4)
# ====================================================
def compute_weighted_krawtchouk_basis_1d(num_points: int, probability_p: float):
    """
    Compute φ_n(i) for n,i = 0..N-1 using the paper's Eq.(4).
    Returns phi matrix of shape (N, N) where phi[n, i] = φ_n(i).
    """
    if not (0.0 < probability_p < 1.0):
        raise ValueError("probability_p (p) must be in (0,1).")
    N = int(num_points)

    # indices (vectorized)
    orders_n = np.arange(N)[:, None]     # shape (N,1)  -> n
    positions_i = np.arange(N)[None, :]  # shape (1,N)  -> i

    # polynomial core: 2F1(-n, -i; -N+1; 1/p)
    hypergeom_z = 1.0 / probability_p
    hypergeom_matrix = hyp2f1(-orders_n, -positions_i, -N + 1, hypergeom_z)  # (N,N)

    # numerator under sqrt: binomial * p^i * (1-p)^(N-1-i)
    log_binom = compute_log_binomial_coefficient(N - 1, positions_i)  # (1,N)
    log_numer = log_binom + positions_i * np.log(probability_p) + (N - 1 - positions_i) * np.log(1.0 - probability_p)
    # sqrt numerator
    sqrt_numer = np.exp(0.5 * log_numer)  # (1,N)

    # denominator under sqrt: ((p-1)/p)^n * n! / (-N+1)_n
    # compute log of magnitude of denominator pieces
    # term1: n * log(|(p-1)/p|)
    log_term1 = orders_n * np.log(np.abs((probability_p - 1.0) / probability_p))  # (N,1)

    # term2: log(n!) = gammaln(n+1)
    log_nfactorial = gammaln(orders_n + 1)  # (N,1)

    # term3: (-N+1)_n  (Pochhammer) may be negative -> track sign and magnitude
    poch_minus = poch(-N + 1, orders_n)    # (N,1)
    sign_poch_minus = np.sign(poch_minus)  # -1, 0, or +1
    abs_poch_minus = np.abs(poch_minus)
    # safe log of absolute
    with np.errstate(divide='ignore'):
        log_abs_poch_minus = np.where(abs_poch_minus > 0, np.log(abs_poch_minus), -np.inf)

    # combine log denominator (magnitude)
    log_denom = log_term1 + log_nfactorial - log_abs_poch_minus  # (N,1)

    # sqrt argument log = log_numer (1,N) - log_denom (N,1) -> broadcasting to (N,N)
    log_sqrt_arg = log_numer - log_denom
    with np.errstate(over='ignore', invalid='ignore'):
        sqrt_factor = np.exp(0.5 * log_sqrt_arg)  # (N,N)

    # sign correction: from Pochhammer and from ((p-1)/p)^n if base negative
    sign_from_poch = np.where(sign_poch_minus < 0, -1.0, 1.0)  # (N,1)
    sign_from_pfactor = np.ones_like(orders_n, dtype=float)   # (N,1)
    if ((probability_p - 1.0) / probability_p) < 0:
        # base negative => sign flips for odd n
        sign_from_pfactor = np.where((orders_n % 2) == 1, -1.0, 1.0)
    sign_correction = sign_from_poch * sign_from_pfactor  # (N,1)

    # assemble phi: hypergeom * sqrt_numer * sqrt_factor * sign_correction
    # sqrt_numer shape (1,N) broadcasts across rows; sign_correction (N,1) across cols
    phi = hypergeom_matrix * sqrt_numer * sqrt_factor * sign_correction  # (N,N)

    # remove tiny numerical imaginary parts
    phi = np.real_if_close(phi, tol=1e-9)
    return phi


# ====================================================
# Build lambda vector (multiparameter) - simple choices
# ====================================================
def build_lambda_vector(num_orders: int, mode: str = "inverse", alpha: float = 0.12, custom: np.ndarray = None):
    """
    Build lambda vector length N.
    modes: 'ones', 'decay', 'sign', 'custom'
    """
    n = np.arange(num_orders, dtype=float)
    if mode == "ones":
        return np.ones(num_orders, dtype=float)
    if mode == "decay":
        return np.exp(-alpha * n)
    if mode == "sign":
        return np.where((n % 2) == 0, 1.0, -1.0)
    if mode == "inverse":
         return 1.0 / (np.arange(num_orders) + 1)

    if mode == "custom":
        if custom is None:
            raise ValueError("custom lambda vector required for mode='custom'")
        arr = np.asarray(custom, dtype=float)
        if arr.shape[0] != num_orders:
            raise ValueError("custom lambda must have length num_orders")
        return arr
    raise ValueError("unknown lambda mode")


# ====================================================
# Separable 2D MDFKT + apply λ as outer product
# ====================================================
def apply_separable_mdfkt_2d(image, probability_p=0.5, lambda_mode="decay", lambda_alpha=0.12):
    """
    image: (H,W) grayscale or (H,W,C) RGB
    returns: coeffs (H,W) or (H,W,C), basis_rows_phi (H,H), basis_cols_phi (W,W), lambda_rows, lambda_cols
    """
    image = np.asarray(image, dtype=float)
    H, W = image.shape[:2]

    # 1D bases
    basis_rows_phi = compute_weighted_krawtchouk_basis_1d(H, probability_p)  # (H,H)
    basis_cols_phi = compute_weighted_krawtchouk_basis_1d(W, probability_p)  # (W,W)

    # lambdas
    lambda_rows = build_lambda_vector(H, mode=lambda_mode, alpha=lambda_alpha)
    lambda_cols = build_lambda_vector(W, mode=lambda_mode, alpha=lambda_alpha)

    def transform_channel(channel_2d):
        # project along rows: S[u, y] = sum_x phi_row[u,x] * F(x,y)
        row_projected = basis_rows_phi @ channel_2d  # (H,W)
        # project along columns: C[u,v] = sum_y row_projected[u,y] * phi_col[v,y]
        coeffs = row_projected @ basis_cols_phi.T    # (H,W)
        # apply separable lambda scaling (outer product)
        coeffs = (lambda_rows[:, None] * lambda_cols[None, :]) * coeffs
        return coeffs

    if image.ndim == 2:
        coeffs = transform_channel(image)
    else:
        channels = []
        for ch in range(image.shape[2]):
            channels.append(transform_channel(image[:, :, ch]))
        coeffs = np.stack(channels, axis=-1)  # (H,W,C)

    return coeffs, basis_rows_phi, basis_cols_phi, lambda_rows, lambda_cols


# ====================================================
# Feature extraction: low-order MxM block + L2 normalize
# ====================================================
def extract_krawtchouk_features_from_coeffs(coeffs, max_order_block=8, l2_normalize=True):
    """
    coeffs: (H,W) or (H,W,C)
    returns: 1D feature vector (flattened MxM block across channels)
    """
    H, W = coeffs.shape[:2]
    Mr = min(max_order_block, H)
    Mc = min(max_order_block, W)
    block = coeffs[:Mr, :Mc]           # (Mr,Mc) or (Mr,Mc,C)
    vec = block.flatten().astype(float)
    if l2_normalize:
        nrm = np.linalg.norm(vec)
        if nrm > 0:
            vec = vec / nrm
    return vec


# ====================================================
# Small demo (copy & run)
# ====================================================
if __name__ == "__main__":
    # test params
    p = 0.5
    H, W = 32, 24
    img_gray = np.random.rand(H, W)
    coeffs_gray, phi_r, phi_c, lam_r, lam_c = apply_separable_mdfkt_2d(img_gray, probability_p=p)
    feats_gray = extract_krawtchouk_features_from_coeffs(coeffs_gray, max_order_block=8)
    print("grayscale feature length:", feats_gray.size)

    img_rgb = np.random.rand(H, W, 3)
    coeffs_rgb, phi_r2, phi_c2, lam_r2, lam_c2 = apply_separable_mdfkt_2d(img_rgb, probability_p=p)
    feats_rgb = extract_krawtchouk_features_from_coeffs(coeffs_rgb, max_order_block=6)
    print("rgb feature length:", feats_rgb.size)

    # optional orthonormality check for basis phi (small N)
    test_phi = compute_weighted_krawtchouk_basis_1d(16, p)
    gram = test_phi @ test_phi.T
    max_offdiag = np.max(np.abs(gram - np.eye(16)))
    print("orthonormality check (N=16) max off-diag:", max_offdiag)
