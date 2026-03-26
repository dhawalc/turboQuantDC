"""Lloyd-Max codebook generation for TurboQuant.

Implements the optimal scalar quantizer for the coordinate distribution
after random orthogonal rotation. Per Lemma 1 of the paper, each coordinate
of Pi*x follows a Beta distribution that converges to N(0, 1/d) for large d.

The Lloyd-Max algorithm finds centroids minimizing the expected MSE:
    C(f, b) = min sum_i integral_{b_{i-1}}^{b_i} |x - c_i|^2 f(x) dx

Reference: TurboQuant paper (arxiv 2504.19874), Section 3 / Eq. 4.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from scipy import integrate
from scipy.special import gamma as gamma_fn


# ---------------------------------------------------------------------------
# Distribution PDFs
# ---------------------------------------------------------------------------

def beta_pdf(x: float, d: int) -> float:
    """Exact Beta distribution PDF for coordinate of a unit-sphere vector.

    After random orthogonal rotation, each coordinate x_j of a unit vector
    on S^{d-1} follows (Lemma 1):
        f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
    for x in [-1, 1].

    Args:
        x: Coordinate value in [-1, 1].
        d: Ambient dimension (head dimension).

    Returns:
        PDF value at x.
    """
    if abs(x) >= 1.0:
        return 0.0
    normalization = gamma_fn(d / 2.0) / (math.sqrt(math.pi) * gamma_fn((d - 1) / 2.0))
    return normalization * (1.0 - x * x) ** ((d - 3) / 2.0)


def gaussian_pdf(x: float, d: int) -> float:
    """Gaussian N(0, 1/d) approximation of the coordinate distribution.

    In high dimensions (d >= 64), the exact Beta distribution converges to
    a normal distribution with variance 1/d. This is the recommended PDF
    for practical use.

    Args:
        x: Coordinate value.
        d: Ambient dimension (head dimension).

    Returns:
        PDF value at x.
    """
    sigma_sq = 1.0 / d
    return (1.0 / math.sqrt(2.0 * math.pi * sigma_sq)) * math.exp(
        -x * x / (2.0 * sigma_sq)
    )


# ---------------------------------------------------------------------------
# Lloyd-Max Solver
# ---------------------------------------------------------------------------

def solve_lloyd_max(
    d: int,
    bits: int,
    use_exact: bool = False,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve the Lloyd-Max optimal scalar quantization problem.

    Finds centroids c_1 <= ... <= c_{2^b} minimizing MSE against the
    coordinate distribution f(x) for dimension d at the given bit-width.

    Algorithm (paper Section 3):
        1. Initialize centroids uniformly in [-3.5/sqrt(d), 3.5/sqrt(d)]
        2. Repeat until convergence:
           a. Boundaries: b_i = (c_i + c_{i+1}) / 2
           b. Centroids: c_i = E[X | X in partition_i] via numerical integration
        3. Return (centroids, boundaries)

    Args:
        d: Ambient dimension (head dimension).
        bits: Number of bits per coordinate.
        use_exact: If True, use exact Beta PDF. Otherwise Gaussian N(0, 1/d).
        max_iter: Maximum number of Lloyd-Max iterations.
        tol: Convergence tolerance on max centroid shift.

    Returns:
        Tuple of (centroids: Tensor[2^bits], boundaries: Tensor[2^bits - 1]).
    """
    pdf = beta_pdf if use_exact else gaussian_pdf
    n_levels = 1 << bits  # 2^bits
    sigma = 1.0 / math.sqrt(d)

    # Integration bounds: extend well beyond the distribution support
    # 3x the initialization range to capture tail probability
    lo = -3.5 * sigma * 3  # -10.5 * sigma
    hi = 3.5 * sigma * 3  # +10.5 * sigma

    # Initialize centroids uniformly in [-3.5*sigma, 3.5*sigma]
    centroids = [
        -3.5 * sigma + (i + 0.5) * (7.0 * sigma / n_levels)
        for i in range(n_levels)
    ]

    for _ in range(max_iter):
        # Step a: compute boundaries as midpoints between adjacent centroids
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]

        # Step b: update centroids as conditional expectations
        # Partition edges: [lo, b_1, b_2, ..., b_{n-1}, hi]
        edges = [lo] + boundaries + [hi]
        new_centroids = []

        for i in range(n_levels):
            left = edges[i]
            right = edges[i + 1]

            # Numerator: integral of x * f(x) dx over partition i
            numerator, _ = integrate.quad(lambda x: x * pdf(x, d), left, right)
            # Denominator: integral of f(x) dx over partition i
            denominator, _ = integrate.quad(lambda x: pdf(x, d), left, right)

            if denominator > 1e-15:
                new_centroids.append(numerator / denominator)
            else:
                # Partition has negligible probability mass; keep old centroid
                new_centroids.append(centroids[i])

        # Check convergence: max shift of any centroid
        max_shift = max(
            abs(new_centroids[i] - centroids[i]) for i in range(n_levels)
        )
        centroids = new_centroids

        if max_shift < tol:
            break

    # Final boundaries from converged centroids
    boundaries = [
        (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
    ]

    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Codebook Class
# ---------------------------------------------------------------------------


class LloydMaxCodebook:
    """Precomputed Lloyd-Max codebook for a given (d, bits) configuration.

    The codebook is the core building block of PolarQuant (Stage 1). It maps
    each rotated coordinate to its nearest centroid index (b bits) and can
    reconstruct by looking up the centroid value.

    Attributes:
        d: Ambient dimension.
        bits: Bits per coordinate.
        n_levels: Number of quantization levels (2^bits).
        centroids: Tensor[n_levels] of centroid values, sorted ascending.
        boundaries: Tensor[n_levels - 1] of decision boundaries.
    """

    def __init__(self, d: int, bits: int, use_exact: bool = False):
        """Initialize codebook by solving the Lloyd-Max problem.

        Args:
            d: Ambient dimension (head dimension, e.g. 128).
            bits: Bits per coordinate (1-8 typical).
            use_exact: Use exact Beta PDF instead of Gaussian approximation.
        """
        self.d = d
        self.bits = bits
        self.n_levels = 1 << bits
        self.use_exact = use_exact

        self.centroids, self.boundaries = solve_lloyd_max(d, bits, use_exact)
        self._distortion: float | None = None

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Map values to nearest centroid indices.

        Uses brute-force nearest-neighbor against all centroids. For small
        n_levels (2-16 for 1-4 bits), this is efficient and exact.

        Args:
            x: Input tensor of any shape. Each element is a coordinate value.

        Returns:
            Integer tensor of same shape with centroid indices in [0, n_levels).
        """
        centroids = self.centroids.to(x.device)
        # (*, 1) - (n_levels,) -> (*, n_levels) -> argmin -> (*)
        return (x.unsqueeze(-1) - centroids).abs().argmin(dim=-1)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up centroid values from indices.

        Args:
            indices: Integer tensor of centroid indices in [0, n_levels).

        Returns:
            Float tensor of same shape with centroid values.
        """
        centroids = self.centroids.to(indices.device)
        return centroids[indices]

    def compute_distortion(self) -> float:
        """Compute the expected MSE per coordinate via numerical integration.

        D_coord = sum_i integral_{partition_i} |x - c_i|^2 f(x) dx

        For comparison with paper bounds:
            - b=1: ~0.36/d
            - b=2: ~0.117/d
            - b=3: ~0.03/d
            - b=4: ~0.009/d

        Returns:
            Expected MSE distortion per coordinate.
        """
        if self._distortion is not None:
            return self._distortion

        pdf = beta_pdf if self.use_exact else gaussian_pdf
        sigma = 1.0 / math.sqrt(self.d)
        lo = -3.5 * sigma * 3
        hi = 3.5 * sigma * 3

        centroids_list = self.centroids.tolist()
        boundaries_list = self.boundaries.tolist()
        edges = [lo] + boundaries_list + [hi]

        total_distortion = 0.0
        for i in range(self.n_levels):
            left = edges[i]
            right = edges[i + 1]
            c_i = centroids_list[i]

            val, _ = integrate.quad(
                lambda x, ci=c_i: (x - ci) ** 2 * pdf(x, self.d), left, right
            )
            total_distortion += val

        self._distortion = total_distortion
        return self._distortion

    def to(self, device: torch.device | str) -> "LloydMaxCodebook":
        """Move codebook tensors to the specified device.

        Args:
            device: Target device.

        Returns:
            Self (for chaining).
        """
        self.centroids = self.centroids.to(device)
        self.boundaries = self.boundaries.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"LloydMaxCodebook(d={self.d}, bits={self.bits}, "
            f"n_levels={self.n_levels})"
        )
