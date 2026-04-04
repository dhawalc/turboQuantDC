"""Learned rotation via PCA whitening for KV cache compression.

Instead of a random rotation (QR/WHT), compute the data-optimal rotation
from calibration vectors.  The eigenvectors of the covariance matrix:

    C = (1/n) * K^T @ K

yield a rotation V^T that *whitens* the data: coordinates become independent
with variance equal to the corresponding eigenvalue.  Sorting by eigenvalue
(descending) concentrates information into the first coordinates, enabling:

1.  Better scalar quantisation (lower MSE per bit)
2.  Adaptive bit allocation: more bits for high-variance coordinates
3.  Tail pruning: zero-out coordinates whose eigenvalue < threshold

The key question is **generalisation**: does V^T fitted on prompt A transfer
to prompt B?  If yes the calibration is practical.

Reference: information-theoretic optimal transform coding (Karhunen-Loeve).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .codebook import LloydMaxCodebook


# ---------------------------------------------------------------------------
# PCA / whitening rotation
# ---------------------------------------------------------------------------

def compute_pca_rotation(
    data: torch.Tensor,
    center: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute the PCA rotation matrix from calibration data.

    Args:
        data: Calibration vectors, shape (n, d).  Must have n >= d.
        center: Subtract the mean before computing covariance.

    Returns:
        Dict with:
            rotation: (d, d) orthogonal matrix V^T (rows = eigenvectors,
                      ordered by DECREASING eigenvalue)
            eigenvalues: (d,) sorted descending
            mean: (d,) mean vector (zero if center=False)
    """
    n, d = data.shape
    assert n >= 2, f"Need at least 2 calibration vectors, got {n}"

    data = data.float()

    if center:
        mean = data.mean(dim=0)
        data_c = data - mean
    else:
        mean = torch.zeros(d, device=data.device, dtype=torch.float32)
        data_c = data

    # Covariance: (d, d) = (1/(n-1)) * X^T @ X  (unbiased)
    C = (data_c.T @ data_c) / (n - 1)

    # Eigendecompose — torch.linalg.eigh returns ascending order
    eigenvalues, V = torch.linalg.eigh(C)

    # Flip to descending
    eigenvalues = eigenvalues.flip(0)
    V = V.flip(1)

    # Rotation is V^T: maps data into the eigenbasis
    rotation = V.T  # (d, d)

    return {
        "rotation": rotation,
        "eigenvalues": eigenvalues,
        "mean": mean,
    }


def compute_adaptive_bit_allocation(
    eigenvalues: torch.Tensor,
    target_avg_bits: float,
    min_bits: int = 1,
    max_bits: int = 6,
) -> torch.Tensor:
    """Water-filling bit allocation: more bits to high-variance coordinates.

    Uses the reverse water-filling solution from rate-distortion theory:

        R_i = max(0, 0.5 * log2(lambda_i / theta))

    where theta is the water level chosen so that sum(R_i) = d * target_avg_bits.

    In practice we discretise to integer bits and clamp to [min_bits, max_bits].

    Args:
        eigenvalues: (d,) sorted descending, must be positive.
        target_avg_bits: Average bits per coordinate.
        min_bits: Minimum bits for any coordinate.
        max_bits: Maximum bits for any coordinate.

    Returns:
        Integer tensor (d,) of per-coordinate bit allocations.
    """
    d = eigenvalues.shape[0]
    total_bits = d * target_avg_bits

    eigs = eigenvalues.clamp(min=1e-12).float()

    # Binary search for the water level theta
    lo_theta = eigs[-1].item() * 0.001
    hi_theta = eigs[0].item() * 10.0

    for _ in range(100):
        theta = (lo_theta + hi_theta) / 2.0
        # Continuous rate per coordinate
        rates = 0.5 * torch.log2(eigs / theta).clamp(min=0)
        total = rates.sum().item()
        if total > total_bits:
            lo_theta = theta
        else:
            hi_theta = theta

    # Discretise
    theta = (lo_theta + hi_theta) / 2.0
    rates = 0.5 * torch.log2(eigs / theta).clamp(min=0)

    # Round to nearest integer, then adjust to hit the budget exactly
    bits = rates.round().clamp(min=min_bits, max=max_bits).long()

    # Greedy adjustment to match budget
    deficit = int(total_bits) - bits.sum().item()
    if deficit > 0:
        # Add bits to coordinates with highest fractional part
        frac = rates - rates.floor()
        order = frac.argsort(descending=True)
        for i in range(min(abs(deficit), d)):
            idx = order[i].item()
            if bits[idx] < max_bits:
                bits[idx] += 1
    elif deficit < 0:
        # Remove bits from coordinates with lowest fractional part
        frac = rates - rates.floor()
        order = frac.argsort()
        for i in range(min(abs(deficit), d)):
            idx = order[i].item()
            if bits[idx] > min_bits:
                bits[idx] -= 1

    return bits


# ---------------------------------------------------------------------------
# PCA-based PolarQuant variant
# ---------------------------------------------------------------------------

class PCARotatedQuantizer(nn.Module):
    """PCA-rotation quantizer with optional adaptive bit allocation.

    Replaces the random rotation in PolarQuant with a learned PCA rotation.
    When adaptive_bits=True, each coordinate gets a custom codebook matched
    to its eigenvalue (variance).

    Args:
        d: Head dimension.
        bits: Uniform bits per coordinate (used when adaptive_bits=False).
        rotation_data: Dict from compute_pca_rotation().
        adaptive_bits: If True, use per-coordinate bit allocation.
        target_avg_bits: For adaptive mode, the average bits budget.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        bits: int,
        rotation_data: Dict[str, torch.Tensor],
        adaptive_bits: bool = False,
        target_avg_bits: float = 3.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.d = d
        self.bits = bits
        self.adaptive_bits = adaptive_bits

        # Store rotation parameters
        self.register_buffer("rotation", rotation_data["rotation"].to(device))
        self.register_buffer("eigenvalues", rotation_data["eigenvalues"].to(device))
        self.register_buffer("mean", rotation_data["mean"].to(device))

        # Whitening scale: after PCA rotation, coordinate i has variance =
        # eigenvalue_i.  The Lloyd-Max codebook is built for N(0, 1/d).
        # We scale each coordinate by sqrt(1/d) / sqrt(eigenvalue_i) so the
        # codebook's quantization boundaries match the actual data range.
        # On dequantize we multiply back by the inverse scale.
        target_var = 1.0 / d
        safe_eigs = rotation_data["eigenvalues"].clamp(min=1e-12).float()
        whiten_scale = (target_var / safe_eigs).sqrt()  # (d,)
        self.register_buffer("whiten_scale", whiten_scale.to(device))

        if adaptive_bits:
            self.bit_alloc = compute_adaptive_bit_allocation(
                rotation_data["eigenvalues"],
                target_avg_bits=target_avg_bits,
            )
            # Build per-coordinate codebooks grouped by bit-width
            # For efficiency, group coordinates with the same bit-width
            self._build_adaptive_codebooks(device)
        else:
            self.bit_alloc = torch.full((d,), bits, dtype=torch.long)
            self.codebook = LloydMaxCodebook(d, bits)
            self.codebook.to(device)
            self.register_buffer("centroids", self.codebook.centroids.to(device))

    def _build_adaptive_codebooks(self, device):
        """Build codebooks for each distinct bit-width in the allocation."""
        unique_bits = self.bit_alloc.unique().tolist()
        self.codebooks = {}
        self.centroid_map = {}

        for b in unique_bits:
            b = int(b)
            if b < 1:
                continue
            cb = LloydMaxCodebook(self.d, b)
            cb.to(device)
            self.codebooks[b] = cb
            self.centroid_map[b] = cb.centroids.to(device)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PCA rotation + whitening: y = ((x - mean) @ V) * whiten_scale.

        After this, each coordinate has variance ~= 1/d, matching the
        Lloyd-Max codebook's N(0, 1/d) assumption.
        """
        x_c = x - self.mean
        y = x_c @ self.rotation.T        # PCA coordinates (variance = eigenvalue_i)
        return y * self.whiten_scale      # whiten to N(0, 1/d)

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Undo whitening + inverse PCA rotation."""
        y_unwhitened = y / self.whiten_scale  # undo whitening
        return y_unwhitened @ self.rotation + self.mean

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize via PCA rotation + whitening + per-coordinate Lloyd-Max."""
        y = self.rotate(x)  # whitened PCA coordinates ~ N(0, 1/d)

        if not self.adaptive_bits:
            return self.codebook.quantize(y)

        # Adaptive: quantize each group of coordinates with its codebook
        squeeze = y.dim() == 1
        if squeeze:
            y = y.unsqueeze(0)

        batch = y.shape[0]
        indices = torch.zeros(batch, self.d, dtype=torch.long, device=y.device)

        for b_val, cb in self.codebooks.items():
            mask = (self.bit_alloc == b_val)
            if not mask.any():
                continue
            coords = y[:, mask]  # (batch, n_coords_at_this_bitwidth)
            indices[:, mask] = cb.quantize(coords)

        if squeeze:
            indices = indices.squeeze(0)
        return indices

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct vectors from codebook indices, undo whitening + rotation."""
        if not self.adaptive_bits:
            y_hat = self.centroids[indices]
            return self.unrotate(y_hat)

        squeeze = indices.dim() == 1
        if squeeze:
            indices = indices.unsqueeze(0)

        batch = indices.shape[0]
        y_hat = torch.zeros(batch, self.d, dtype=torch.float32, device=indices.device)

        for b_val, cb in self.codebooks.items():
            mask = (self.bit_alloc == b_val)
            if not mask.any():
                continue
            y_hat[:, mask] = cb.centroids[indices[:, mask]]

        if squeeze:
            y_hat = y_hat.squeeze(0)
        return self.unrotate(y_hat)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize + dequantize (for evaluation)."""
        indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        return x_hat, indices

    def effective_bits_per_coord(self) -> float:
        """Average bits per coordinate across the allocation."""
        return self.bit_alloc.float().mean().item()

    def variance_explained(self, n_coords: int = None) -> float:
        """Fraction of total variance captured by top n_coords components."""
        if n_coords is None:
            n_coords = self.d
        total = self.eigenvalues.sum().item()
        if total < 1e-12:
            return 1.0
        return self.eigenvalues[:n_coords].sum().item() / total
