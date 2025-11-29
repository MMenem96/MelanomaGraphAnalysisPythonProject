import numpy as np
from scipy.special import hyp2f1, gammaln, poch

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

    hypergeom_z = 1.0 / probability_p
    hypergeom_matrix = hyp2f1(-orders_n, -positions_i, -N + 1, hypergeom_z)  # (N,N)

    log_binom = compute_log_binomial_coefficient(N - 1, positions_i)  # (1,N)
    log_numer = log_binom + positions_i * np.log(probability_p) + (N - 1 - positions_i) * np.log(1.0 - probability_p)
    sqrt_numer = np.exp(0.5 * log_numer)  # (1,N)

    log_term1 = orders_n * np.log(np.abs((probability_p - 1.0) / probability_p))  # (N,1)
    log_nfactorial = gammaln(orders_n + 1)  # (N,1)

    poch_minus = poch(-N + 1, orders_n)    # (N,1)
    sign_poch_minus = np.sign(poch_minus)  # (N,1)
    abs_poch_minus = np.abs(poch_minus)
    with np.errstate(divide='ignore'):
        log_abs_poch_minus = np.where(abs_poch_minus > 0, np.log(abs_poch_minus), -np.inf)

    log_denom = log_term1 + log_nfactorial - log_abs_poch_minus  # (N,1)

    log_sqrt_arg = log_numer - log_denom
    with np.errstate(over='ignore', invalid='ignore'):
        sqrt_factor = np.exp(0.5 * log_sqrt_arg)  # (N,N)

    sign_from_poch = np.where(sign_poch_minus < 0, -1.0, 1.0)  # (N,1)
    sign_from_pfactor = np.ones_like(orders_n, dtype=float)   # (N,1)
    if ((probability_p - 1.0) / probability_p) < 0:
        sign_from_pfactor = np.where((orders_n % 2) == 1, -1.0, 1.0)
    sign_correction = sign_from_poch * sign_from_pfactor  # (N,1)

    phi = hypergeom_matrix * sqrt_numer * sqrt_factor * sign_correction  # (N,N)
    phi = np.real_if_close(phi, tol=1e-9)
    return phi


# ====================================================
# Build lambda vector (multiparameter)
# ====================================================
def build_lambda_vector(num_orders: int, mode: str = "decay", alpha: float = 0.12, custom: np.ndarray = None):
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
def apply_separable_mdfkt_2d(image, probability_p=0.5, lambda_mode="decay", lambda_alpha=0.12):
    image = np.asarray(image, dtype=float)
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
            channels.append(transform_channel(image[:, :, ch]))
        coeffs = np.stack(channels, axis=-1)  # (H,W,C)

    return coeffs, basis_rows_phi, basis_cols_phi, lambda_rows, lambda_cols


# ====================================================
# Build full K_lambda (explicit Q Λ Q^T) using Kronecker
# ====================================================
def build_full_K_lambda(basis_rows_phi, basis_cols_phi, lambda_rows, lambda_cols, vec_order='F'):
    """
    Returns:
      Q_full: (H*W, H*W)  -- Kronecker(Q_cols, Q_rows)
      Lambda_full: (H*W, H*W) -- diagonal matrix
      K_lambda: (H*W, H*W) = Q_full @ Lambda_full @ Q_full.T

    vec_order:
      'F' (Fortran/column-major) is the convention used here: vec stacks columns.
      If you use 'C' (row-major), Kron ordering must be adjusted.
    """
    H = basis_rows_phi.shape[0]
    W = basis_cols_phi.shape[0]

    # full orthonormal matrix (matches column-major vec ordering)
    Q_full = np.kron(basis_cols_phi, basis_rows_phi)        # (H*W, H*W)

    # diagonal Lambda built as Kronecker of column/row lambdas to match vec ordering
    Lambda_diag = np.kron(lambda_cols, lambda_rows)        # length H*W
    Lambda_full = np.diag(Lambda_diag)                     # (H*W, H*W)

    # full K_lambda
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
# Small demo (copy & run) - equality check between separable path and full K_lambda application
# ====================================================
if __name__ == "__main__":
    p = 0.5
    H, W = 8, 6   # small sizes for full matrix test
    img_gray = np.random.rand(H, W)

    # separable path
    coeffs_gray, phi_r, phi_c, lam_r, lam_c = apply_separable_mdfkt_2d(img_gray, probability_p=p, lambda_mode="decay")
    feats_gray = extract_krawtchouk_features_from_coeffs(coeffs_gray, max_order_block=6)
    print("separable coeffs.shape:", coeffs_gray.shape, "feature length:", feats_gray.size)

    # full K_lambda path
    Q_full, Lambda_full, K_lambda = build_full_K_lambda(phi_r, phi_c, lam_r, lam_c)
    vecF = img_gray.reshape((H * W,), order='F')       # column-major flatten
    out_vec_full = K_lambda @ vecF
    out_image_full = out_vec_full.reshape((H, W), order='F')

    # compare with separable coeffs
    diff = np.max(np.abs(out_image_full - coeffs_gray))
    print("max abs difference between separable and full K_lambda result:", diff)

    # minimal orthonormality check for phi
    test_phi = compute_weighted_krawtchouk_basis_1d(8, p)
    gram = test_phi @ test_phi.T
    max_offdiag = np.max(np.abs(gram - np.eye(8)))
    print("orthonormality check (N=8) max off-diag:", max_offdiag)
