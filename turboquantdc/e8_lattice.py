"""E8 Lattice Vector Quantization for KV Cache Compression.

Replaces per-coordinate scalar Lloyd-Max with 8-dimensional lattice VQ.
E8 achieves 14% lower MSE than scalar quantization at the same bit rate
(Zador's theorem: NSM_E8 = 0.07168 vs NSM_Z = 0.08333).

Algorithm: Conway-Sloane two-coset nearest point finding.
- E8 = D8 ∪ (D8 + 1/2), where D8 = {x ∈ Z^8 : sum(x) even}
- Nearest point is O(1) per 8D block (fixed dimension, pure arithmetic)
- No calibration data, no lookup table, no learned parameters

Usage:
    from turboquantdc.e8_lattice import E8Quantizer
    eq = E8Quantizer(scale=0.5)
    indices, recon = eq.quantize(x)  # x: (..., 8)

For KV cache (d=128): reshape to (..., 16, 8), quantize each 8D block.

References:
    - Conway & Sloane, "Sphere Packings, Lattices, and Groups" (Ch. 20)
    - QuIP# (Tseng et al., ICML 2024, arXiv 2402.04396)
    - Viazovska (2016): E8 optimal sphere packing in 8D (Fields Medal 2022)
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional


def nearest_d8(x: torch.Tensor) -> torch.Tensor:
    """Find nearest D8 lattice point (even integer lattice).

    D8 = {v ∈ Z^8 : sum(v) is even}

    Algorithm:
    1. Round each coordinate to nearest integer
    2. If sum is odd, flip the coordinate with smallest rounding margin
    """
    rounded = torch.round(x)
    # Check parity: sum must be even
    parity = rounded.sum(dim=-1) % 2  # 0 or 1
    needs_fix = parity != 0  # (...)

    if needs_fix.any():
        # Find coordinate with smallest rounding margin
        margin = (x - rounded).abs()  # (..., 8)
        # For positions that need fixing, find the coord to flip
        # Set margin to inf for positions that don't need fixing
        margin_masked = margin.clone()
        margin_masked[~needs_fix] = float('inf')
        flip_coord = margin_masked.argmin(dim=-1)  # (...)

        # Flip: if x > rounded, round up instead; if x < rounded, round down
        residual = x - rounded  # (..., 8)
        # Gather the residual at flip_coord
        flip_sign = torch.gather(residual, -1, flip_coord.unsqueeze(-1)).sign()
        # Where sign is 0, default to +1
        flip_sign = flip_sign.where(flip_sign != 0, torch.ones_like(flip_sign))

        # Apply fix only where needed
        fix = torch.zeros_like(rounded)
        fix.scatter_(-1, flip_coord.unsqueeze(-1),
                     flip_sign.where(needs_fix.unsqueeze(-1),
                                     torch.zeros_like(flip_sign)))
        rounded = rounded + fix

    return rounded


def nearest_e8(x: torch.Tensor) -> torch.Tensor:
    """Find nearest E8 lattice point.

    E8 = D8 ∪ (D8 + 1/2)
    Try both cosets, return whichever is closer.

    Args:
        x: (..., 8) tensor

    Returns:
        Nearest E8 lattice point, same shape as x
    """
    # Coset 1: nearest point in D8 (integer lattice, even sum)
    c1 = nearest_d8(x)

    # Coset 2: nearest point in D8 + 1/2 (half-integer lattice, even sum)
    c2 = nearest_d8(x - 0.5) + 0.5

    # Return whichever is closer
    d1 = ((x - c1) ** 2).sum(dim=-1, keepdim=True)
    d2 = ((x - c2) ** 2).sum(dim=-1, keepdim=True)
    return torch.where(d1 <= d2, c1, c2)


def nearest_e8_relaxed(x: torch.Tensor) -> torch.Tensor:
    """Find nearest point in relaxed E8 (no even-sum constraint).

    Removes the parity constraint to add codepoints near the origin,
    which is where WHT-rotated KV cache vectors concentrate.
    Expected: additional ~22% MSE reduction on KV data vs strict E8.

    Args:
        x: (..., 8) tensor

    Returns:
        Nearest relaxed-E8 lattice point
    """
    # Integer rounding (no parity constraint)
    r_int = torch.round(x)
    # Half-integer rounding (no parity constraint)
    r_half = torch.round(x - 0.5) + 0.5
    # Return closer
    d_int = ((x - r_int) ** 2).sum(dim=-1, keepdim=True)
    d_half = ((x - r_half) ** 2).sum(dim=-1, keepdim=True)
    return torch.where(d_int <= d_half, r_int, r_half)


class E8Quantizer:
    """E8 lattice vector quantizer for 8D sub-vectors.

    Quantizes d-dimensional vectors by reshaping to (d/8, 8) blocks
    and finding the nearest E8 lattice point for each block.

    The scale parameter controls the quantization step size:
    - Smaller scale = finer quantization = more bits = better quality
    - Larger scale = coarser quantization = fewer bits = more compression

    For b bits per dimension, scale ≈ 1 / (2^b * sqrt(8/12)) is a
    reasonable starting point. The optimal scale depends on the data
    distribution.

    Args:
        scale: Quantization step size (default 0.5 for ~2 bits/dim)
        relaxed: Use relaxed E8 (no parity constraint, better for KV cache)
    """

    def __init__(self, scale: float = 0.5, relaxed: bool = True):
        self.scale = scale
        self.relaxed = relaxed
        self._quantize_fn = nearest_e8_relaxed if relaxed else nearest_e8

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize input by finding nearest E8 lattice point.

        Args:
            x: (..., d) tensor where d is divisible by 8

        Returns:
            (lattice_points, reconstructed): both same shape as x
            lattice_points are the E8 lattice coordinates (can be used as indices)
            reconstructed = lattice_points * scale (the dequantized values)
        """
        *batch_shape, d = x.shape
        assert d % 8 == 0, f"Dimension {d} must be divisible by 8"

        # Scale input to lattice coordinates
        x_scaled = x / self.scale

        # Reshape to 8D blocks
        x_blocks = x_scaled.reshape(*batch_shape, d // 8, 8)

        # Find nearest lattice point per block
        lattice_pts = self._quantize_fn(x_blocks)

        # Reshape back and rescale
        lattice_pts = lattice_pts.reshape(*batch_shape, d)
        reconstructed = lattice_pts * self.scale

        return lattice_pts, reconstructed

    def dequantize(self, lattice_pts: torch.Tensor) -> torch.Tensor:
        """Dequantize lattice points back to original space."""
        return lattice_pts * self.scale


def calibrate_scale(x: torch.Tensor, target_bits: float = 3.0) -> float:
    """Estimate optimal E8 scale for a given target bit rate.

    For E8 at b bits/dim, the effective codebook size per 8D block
    is 2^(8b). The scale should place ~2^(8b) lattice points within
    the data's dynamic range.

    A simpler heuristic: scale = std(x) * alpha(b) where alpha is
    calibrated to match the desired MSE-rate tradeoff.

    Args:
        x: Calibration data (..., d) where d % 8 == 0
        target_bits: Target bits per dimension

    Returns:
        Recommended scale value
    """
    std = x.std().item()
    # Heuristic: at b bits/dim, there are 2^b levels per coordinate
    # E8 lattice spacing ≈ 1, so scale = std / (2^(b-1))
    n_levels = 2 ** target_bits
    scale = 2.0 * std / n_levels
    return max(scale, 1e-8)
