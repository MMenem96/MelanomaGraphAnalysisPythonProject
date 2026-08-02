"""
The four MKT eigenvalue (λ) configurations reported in the paper.

Each function takes the transform size N and returns a length-N numpy array
of eigenvalues. The functions are used by feature_extraction.py to extract
four feature pickles in one pass over the images.

Mathematical notes:
  * "low_pass" matches the paper's `λ_k = 1/(k+1), k=0..N-1`. We compute it
    as `1.0/np.arange(1, N+1)` which is identical (no division by zero).
  * "high_pass" is `λ_k = 1/(N-k), k=0..N-1`. Smallest denominator is 1 at
    k=N-1, so it is always well-defined.
  * The two complex configurations are unit-modulus, so the resulting MKT
    is unitary — useful for phase-aware features.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

LambdaFn = Callable[[int], np.ndarray]


def lambda_low_pass(N: int) -> np.ndarray:
    """λ_k = 1/(k+1) for k=0..N-1 (paper's reciprocal low-pass)."""
    k = np.arange(1, N + 1, dtype=np.float64)
    return 1.0 / k


def lambda_high_pass(N: int) -> np.ndarray:
    """λ_k = 1/(N-k) for k=0..N-1 (paper's reciprocal high-pass)."""
    k = np.arange(N, dtype=np.float64)
    return 1.0 / (N - k)


def lambda_dft(N: int) -> np.ndarray:
    """λ_k = exp(i · 2π k / N) for k=0..N-1 (DFT-like complex)."""
    k = np.arange(N)
    return np.exp(1j * 2 * np.pi * k / N)


def lambda_odd_harmonic(N: int) -> np.ndarray:
    """λ_k = exp(i · (2k+1) π / N) for k=0..N-1 (odd-harmonic complex)."""
    k = np.arange(N)
    return np.exp(1j * (2 * k + 1) * np.pi / N)


# Mapping from short name (used in pickle filenames + CLI) to function.
# Ordered to match paper Tables 2–5.
LAMBDA_CONFIGS: dict[str, LambdaFn] = {
    "low_pass":     lambda_low_pass,        # paper Table 2
    "high_pass":    lambda_high_pass,       # paper Table 3
    "dft":          lambda_dft,             # paper Table 4
    "odd_harmonic": lambda_odd_harmonic,    # paper Table 5
}


def all_lambda_names() -> list[str]:
    return list(LAMBDA_CONFIGS.keys())


def get_lambda_fn(name: str) -> LambdaFn:
    if name not in LAMBDA_CONFIGS:
        raise KeyError(
            f"Unknown lambda config {name!r}. Known: {list(LAMBDA_CONFIGS)}"
        )
    return LAMBDA_CONFIGS[name]
