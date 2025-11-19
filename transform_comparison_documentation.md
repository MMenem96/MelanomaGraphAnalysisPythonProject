# Transform Methods Comparison for Skin Lesion Feature Extraction

**Project:** Melanoma Graph Analysis - BCC vs SK Classification  
**Author:** Mohamed Moniem  
**Date:** November 10, 2025  
**Purpose:** Comparative analysis of four transform-based feature extraction methods

---

## Table of Contents
1. [Overview](#overview)
2. [Krawtchouk Moments](#krawtchouk-moments)
3. [Fourier Transform](#fourier-transform)
4. [Discrete Cosine Transform (DCT)](#discrete-cosine-transform-dct)
5. [Hadamard Transform](#hadamard-transform)
6. [Comparison Table](#comparison-table)

---

## Overview

This document provides a comprehensive comparison of four mathematical transform methods used for feature extraction in skin lesion classification. Each method extracts 36 features from dermoscopy images to distinguish between Basal Cell Carcinoma (BCC) and Seborrheic Keratosis (SK).

**Common Preprocessing:**
- All methods use the same normalization strategy: **256×256 pixels**
- Lesion regions are extracted using binary masks
- Grayscale conversion applied before transform
- Aspect ratio preserved during resizing

---

## 1. Krawtchouk Moments

### Definition
Krawtchouk moments are a set of discrete orthogonal moments based on Krawtchouk polynomials. They are particularly effective for analyzing discrete digital images because they are defined on a discrete domain (unlike continuous Legendre or Zernike moments). Krawtchouk moments provide excellent shape description and are inherently invariant to geometric transformations.

### Mathematical Foundation

#### 1.1 Weighted Krawtchouk Polynomial

The weighted Krawtchouk polynomial $K_n(x; p, N)$ is defined as:

$$K_n(x; p, N) = \sum_{k=0}^{n} a_k \cdot \binom{x}{k} \cdot \binom{N-x}{n-k}$$

Where the coefficient $a_k$ is:

$$a_k = (-1)^k \cdot \binom{n}{k} \cdot p^{n-k} \cdot (1-p)^k$$

**Parameters:**
- $x$: Point at which to evaluate the polynomial (0 ≤ x ≤ N)
- $n$: Order of the polynomial (0 ≤ n ≤ N)
- $p$: Shape parameter controlling the distribution (0 < p < 1)
- $N$: Size of the discrete interval
- $\binom{a}{b}$: Binomial coefficient "a choose b"

#### 1.2 Weighting Function

The weighting function for orthogonality:

$$w(x; p, N) = \binom{N}{x} \cdot p^x \cdot (1-p)^{N-x}$$

This represents a binomial distribution that ensures orthogonality of the polynomials.

#### 1.3 Krawtchouk Moment Definition

The 2D Krawtchouk moment of order $(n, m)$ for an image $f(x, y)$ is:

$$Q_{nm} = \sum_{x=0}^{N_1-1} \sum_{y=0}^{N_2-1} f(x, y) \cdot K_n(x; p_1, N_1-1) \cdot K_m(y; p_2, N_2-1)$$

Where:
- $f(x, y)$: Image intensity at pixel (x, y)
- $N_1, N_2$: Image dimensions (height, width)
- $K_n, K_m$: Krawtchouk polynomials in x and y directions

#### 1.4 Rotation-Invariant Moments

To achieve rotation invariance, we compute normalized invariant moments:

$$\phi_1 = \frac{|Q_{20}| + |Q_{02}|}{Q_{00}^2}$$

$$\phi_2 = \frac{(Q_{20} - Q_{02})^2 + 4Q_{11}^2}{Q_{00}^4}$$

$$\phi_3 = \frac{|Q_{30}| + |Q_{12}|}{Q_{00}^{2.5}}$$

$$\phi_4 = \frac{|Q_{03}| + |Q_{21}|}{Q_{00}^{2.5}}$$

#### 1.5 Energy and Entropy

**Energy** (measures concentration of moment distribution):

$$E = \sum_{n=0}^{N_{\text{max}}} \sum_{m=0}^{N_{\text{max}}} |Q_{nm}|^2$$

**Entropy** (measures randomness in moment distribution):

$$H = -\sum_{n,m} P_{nm} \cdot \log_2(P_{nm})$$

Where $P_{nm} = \frac{|Q_{nm}|}{\sum |Q_{nm}|}$ (normalized probability)

### Implementation Parameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `max_order` | 4 | Maximum polynomial order (n + m ≤ 4) |
| `p1` | 0.5 | Shape parameter for x-direction (symmetric) |
| `p2` | 0.5 | Shape parameter for y-direction (symmetric) |
| `normalize_size` | 256 | Target size for lesion normalization |

### Features Extracted

1. **Raw Moments (15 features):** `krawtchouk_moment_{n}_{m}` where n+m ≤ 4
2. **Absolute Moments (15 features):** `krawtchouk_moment_abs_{n}_{m}`
3. **Rotation-Invariant Moments (4 features):** `krawtchouk_invariant_1` through `krawtchouk_invariant_4`
4. **Statistical Features (2 features):** `krawtchouk_energy`, `krawtchouk_entropy`

---

## 2. Fourier Transform

### Definition
The Fourier Transform decomposes an image into its frequency components, representing it as a sum of sinusoidal functions with different frequencies, amplitudes, and phases. It's widely used in signal processing, image compression, and pattern recognition. Low frequencies represent smooth regions, while high frequencies capture edges and fine details.

### Mathematical Foundation

#### 2.1 2D Discrete Fourier Transform (DFT)

For a discrete image $f(x, y)$ of size $M \times N$:

$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \cdot e^{-j2\pi(\frac{ux}{M} + \frac{vy}{N})}$$

Where:
- $F(u, v)$: Frequency domain representation
- $(u, v)$: Frequency coordinates
- $j = \sqrt{-1}$: Imaginary unit
- $e^{-j\theta} = \cos(\theta) - j\sin(\theta)$: Euler's formula

#### 2.2 Magnitude and Phase

The complex Fourier coefficient can be represented as:

$$F(u, v) = |F(u, v)| \cdot e^{j\phi(u,v)}$$

**Magnitude Spectrum:**

$$|F(u, v)| = \sqrt{\text{Re}(F(u,v))^2 + \text{Im}(F(u,v))^2}$$

**Phase Spectrum:**

$$\phi(u, v) = \arctan\left(\frac{\text{Im}(F(u,v))}{\text{Re}(F(u,v))}\right)$$

#### 2.3 Frequency Shift

To center low frequencies, we apply the shift theorem:

$$F_{\text{shifted}}(u, v) = F\left(u - \frac{M}{2}, v - \frac{N}{2}\right)$$

This moves the DC component (zero frequency) to the image center.

#### 2.4 Energy

Total frequency energy:

$$E_{\text{Fourier}} = \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} |F(u, v)|^2$$

By Parseval's theorem, this equals the spatial domain energy.

#### 2.5 Entropy

Frequency domain entropy:

$$H_{\text{Fourier}} = -\sum_{u,v} P(u,v) \cdot \log_2(P(u,v))$$

Where $P(u,v) = \frac{|F(u,v)|}{\sum |F|}$ (normalized magnitude distribution)

### Implementation Parameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `num_coefficients` | 36 | Total number of features to extract |
| `roi_size` | 20 | Region of interest around center (low frequencies) |
| `normalize_size` | 256 | Target size for fair comparison |

### Features Extracted

1. **Magnitude Coefficients (18 features):** `fourier_magnitude_{i}` - Top 18 magnitude values
2. **Phase Coefficients (18 features):** `fourier_phase_{i}` - Corresponding phase values
3. **Statistical Features (4 features):**
   - `fourier_energy`: Total frequency energy
   - `fourier_entropy`: Frequency distribution entropy
   - `fourier_mean_magnitude`: Average magnitude in ROI
   - `fourier_std_magnitude`: Standard deviation of magnitudes

---

## 3. Discrete Cosine Transform (DCT)

### Definition
The Discrete Cosine Transform (DCT) expresses a finite sequence of data points as a sum of cosine functions at different frequencies. Unlike Fourier (which uses complex exponentials), DCT uses only real numbers, making it computationally efficient. DCT is the foundation of JPEG image compression due to its excellent energy compaction property.

### Mathematical Foundation

#### 3.1 2D DCT Definition

The 2D DCT of an image $f(x, y)$ of size $M \times N$ is:

$$F(u, v) = \alpha(u) \alpha(v) \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \cdot \cos\left[\frac{\pi u}{2M}(2x+1)\right] \cdot \cos\left[\frac{\pi v}{2N}(2y+1)\right]$$

Where the normalization coefficients are:

$$\alpha(u) = \begin{cases}
\sqrt{\frac{1}{M}} & \text{if } u = 0 \\
\sqrt{\frac{2}{M}} & \text{if } u > 0
\end{cases}$$

$$\alpha(v) = \begin{cases}
\sqrt{\frac{1}{N}} & \text{if } v = 0 \\
\sqrt{\frac{2}{N}} & \text{if } v > 0
\end{cases}$$

#### 3.2 Separability Property

DCT can be computed efficiently using row-column decomposition:

$$F(u, v) = \text{DCT}_{\text{col}}\left(\text{DCT}_{\text{row}}(f(x, y))\right)$$

This reduces 2D DCT to two 1D DCT operations.

#### 3.3 Zigzag Scanning

Coefficients are extracted in zigzag order to prioritize low-frequency components:

```
0  → 1     5  → 6
   ↓   ↗  ↓      ↗
   2     4     7
   ↓  ↗     ↗  ↓
   3        8  → 9
```

This pattern captures energy-packed coefficients first (top-left = low frequency).

#### 3.4 Energy Compaction

DCT energy:

$$E_{\text{DCT}} = \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} F(u, v)^2$$

**Compaction Ratio** (efficiency measure):

$$\rho = \frac{E_{\text{low-freq}}}{E_{\text{total}}}$$

Where $E_{\text{low-freq}}$ is the energy in the top-left quarter of the DCT matrix. Higher ratio indicates better compaction.

#### 3.5 DC and AC Coefficients

- **DC Coefficient:** $F(0, 0)$ - represents average intensity
- **AC Coefficients:** All other $F(u, v)$ - represent variations from average

### Implementation Parameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `num_coefficients` | 36 | Number of DCT coefficients in zigzag order |
| `normalize_size` | 256 | Target size for fair comparison |
| `norm` | 'ortho' | Orthonormal normalization mode |

### Features Extracted

1. **DCT Coefficients (36 features):** `dct_coeff_{i}` - Zigzag-ordered coefficients
2. **Statistical Features (5 features):**
   - `dct_energy`: Total energy across all coefficients
   - `dct_mean`: Average absolute coefficient value
   - `dct_std`: Standard deviation of coefficients
   - `dct_max`: Maximum absolute coefficient
   - `dct_compaction_ratio`: Low-frequency energy ratio

---

## 4. Hadamard Transform

### Definition
The Hadamard Transform is a generalized Fourier transform that uses Walsh functions (square waves) instead of sinusoids. It operates on sequences whose length is a power of 2 and involves only additions and subtractions (no multiplications), making it extremely fast. Used in digital communications, cryptography, and image processing.

### Mathematical Foundation

#### 4.1 Hadamard Matrix

The Hadamard matrix $H_n$ is defined recursively:

**Base case:**
$$H_1 = \begin{bmatrix} 1 \end{bmatrix}$$

**Recursive definition:**
$$H_{2n} = \begin{bmatrix} H_n & H_n \\ H_n & -H_n \end{bmatrix}$$

**Examples:**

$$H_2 = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \quad H_4 = \begin{bmatrix} 1 & 1 & 1 & 1 \\ 1 & -1 & 1 & -1 \\ 1 & 1 & -1 & -1 \\ 1 & -1 & -1 & 1 \end{bmatrix}$$

#### 4.2 1D Fast Walsh-Hadamard Transform (FWHT)

For a vector $x$ of length $N = 2^k$:

$$X = H_N \cdot x$$

**Butterfly Algorithm** (O(N log N) complexity):

```
Stage 1: Pair adjacent elements, compute sum and difference
Stage 2: Apply same operation on pairs of pairs
...
Stage k: Combine final results
```

At each stage $s$, with stride $h = 2^s$:

$$\text{For } i \text{ in steps of } 2h: \quad \begin{cases} 
x[i:i+h] \leftarrow x[i:i+h] + x[i+h:i+2h] \\
x[i+h:i+2h] \leftarrow x[i:i+h] - x[i+h:i+2h]
\end{cases}$$

#### 4.3 2D Hadamard Transform

Computed using separable decomposition:

$$F(u, v) = \frac{1}{N} \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} f(x, y) \cdot H_u(x) \cdot H_v(y)$$

Where $H_u(x)$ are Walsh functions (rows of Hadamard matrix).

**Implementation:**
1. Apply 1D FWHT to each row
2. Apply 1D FWHT to each column of result
3. Normalize by $N$

#### 4.4 Sequency

Sequency is the Hadamard equivalent of frequency:
- **Sequency = 0:** Constant (DC component)
- **Low sequency:** Smooth patterns, slow transitions
- **High sequency:** Rapid changes, edges, texture

Sequency of Walsh function = number of zero-crossings per unit interval

#### 4.5 Energy and Low-Sequency Ratio

**Total Energy:**

$$E_{\text{Hadamard}} = \sum_{u=0}^{N-1} \sum_{v=0}^{N-1} F(u, v)^2$$

**Low-Sequency Ratio:**

$$\rho_{\text{low}} = \frac{E_{\text{low-seq}}}{E_{\text{total}}}$$

Where $E_{\text{low-seq}}$ is energy in top-left region (low sequency). High ratio indicates smooth, regular patterns.

### Implementation Parameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `num_coefficients` | 36 | Number of top Hadamard coefficients |
| `normalize_size` | 256 | Target size (padded to next power of 2) |
| `low_seq_size` | N/8 | Region size for low-sequency analysis |

### Features Extracted

1. **Hadamard Coefficients (36 features):** `hadamard_coeff_{i}` - Top 36 coefficients by magnitude
2. **Statistical Features (5 features):**
   - `hadamard_energy`: Total transform energy
   - `hadamard_mean`: Average absolute coefficient value
   - `hadamard_std`: Standard deviation of coefficients
   - `hadamard_max`: Maximum absolute coefficient
   - `hadamard_low_seq_ratio`: Low-sequency energy ratio

---

## Comparison Table

Complete this table after running training experiments with each transform method:

| Metric | Krawtchouk | Fourier | DCT | Hadamard |
|--------|-----------|---------|-----|----------|
| **AC (%)** | 96.5 | 93.1 | 94.7 | 92.0 |
| **SN (%)** | 96.6 | 98.1 | 97.1 | 93.2 |
| **SP (%)** | 96.3 | 89.0 | 92.7 | 91.1 |
| **PR (%)** | 95.7 | 88.2 | 91.7 | 89.7 |
| **F1 (%)** | 96.1 | 92.9 | 94.3 | 91.4 |
| **AUC (%)** | 99.2 | 98.3 | 98.1 | 97.7 |
