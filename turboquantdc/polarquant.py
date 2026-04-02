"""PolarQuant — Stage 1 MSE-optimal vector quantizer.

Implements Algorithm 1 (TurboQuant_mse) from the paper:
    1. Rotate input via random orthogonal matrix Pi
    2. Quantize each coordinate independently via Lloyd-Max codebook
    3. Dequantize by centroid lookup and inverse rotation

After rotation, coordinates are nearly independent and follow a concentrated
distribution (Beta -> Gaussian for large d), enabling optimal SCALAR quantization
of each coordinate independently. This is the key insight of TurboQuant.

Performance guarantee (Theorem 1):
    D_mse = E[||x - x_hat||^2] <= sqrt(3)*pi/2 * 1/4^b  (~2.72 / 4^b)

Reference: TurboQuant paper (arxiv 2504.19874), Algorithm 1 / Theorem 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .codebook import LloydMaxCodebook
from .rotation import (
    apply_wht_rotation,
    generate_rotation_matrix,
    generate_wht_rotation,
)


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class PolarQuant(nn.Module):
    """Stage 1: MSE-optimal vector quantizer via rotation + scalar quantization.

    The quantization pipeline for a unit vector x in R^d:
        Quantize:   x -> y = x @ Pi.T -> indices = nearest_centroid(y)
        Dequantize: indices -> y_hat = centroids[indices] -> x_hat = y_hat @ Pi

    For non-unit vectors, the caller is responsible for normalizing and
    storing the norm separately (handled by TurboQuantEstimator).

    Args:
        d: Head dimension (e.g. 128).
        bits: Bits per coordinate (1-4 typical).
        seed: Random seed for rotation matrix generation.
        device: Target device.
        rotation_type: "wht" (default when d is a power of 2; O(d log d)) or
            "qr" (Haar-uniform QR rotation; O(d^2), always valid).
            WHT is faster and accounts for ~98% of compression quality gain.
    """

    def __init__(
        self,
        d: int,
        bits: int,
        seed: int = 42,
        device: str | torch.device = "cpu",
        rotation_type: str | None = None,
    ):
        super().__init__()
        self.d = d
        self.bits = bits

        # Resolve rotation type: default to WHT when d is a power of 2
        if rotation_type is None:
            rotation_type = "wht" if _is_power_of_2(d) else "qr"
        if rotation_type not in ("wht", "qr"):
            raise ValueError(f"rotation_type must be 'wht' or 'qr', got {rotation_type!r}")
        if rotation_type == "wht" and not _is_power_of_2(d):
            raise ValueError(f"WHT requires d to be a power of 2, got d={d}")
        self.rotation_type = rotation_type

        if rotation_type == "wht":
            # Store WHT sign vector (d floats) — O(d) memory vs O(d^2) for QR
            wht_params = generate_wht_rotation(d, seed=seed, device="cpu")
            self.register_buffer("wht_signs", wht_params["signs"].to(device))
            # Build explicit Pi for API compatibility (tests, checkpointing)
            # Pi is derived from WHT; compute by applying WHT to the identity
            I_d = torch.eye(d, device="cpu")
            Pi_rows = apply_wht_rotation(I_d, {"signs": wht_params["signs"].cpu(), "d": d})
            self.register_buffer("Pi", Pi_rows.to(device))
        else:
            # QR: dense d x d orthogonal matrix
            Pi = generate_rotation_matrix(d, seed=seed, device="cpu")
            self.register_buffer("Pi", Pi.to(device))
            self.wht_signs = None  # not used for QR

        # Solve Lloyd-Max codebook for this (d, bits) configuration
        self.codebook = LloydMaxCodebook(d, bits)
        # Register centroids as buffer for device tracking
        self.register_buffer("centroids", self.codebook.centroids.to(device))

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random orthogonal rotation.

        For WHT (default when d is power of 2): O(d log d) butterfly transform.
        For QR: O(d^2) dense matrix multiply.

        After rotation, each coordinate y_j follows the concentrated
        distribution from Lemma 1, enabling scalar quantization.

        Args:
            x: Input vectors of shape (batch, d) or (d,).

        Returns:
            Rotated vectors of same shape.
        """
        if self.rotation_type == "wht":
            return apply_wht_rotation(x, {"signs": self.wht_signs, "d": self.d})
        return x @ self.Pi.T

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation.

        For WHT: O(d log d) inverse butterfly transform.
        For QR: O(d^2) dense matrix multiply (Pi^{-1} = Pi^T).

        Args:
            y: Rotated vectors of shape (batch, d) or (d,).

        Returns:
            Unrotated vectors of same shape.
        """
        if self.rotation_type == "wht":
            return apply_wht_rotation(y, {"signs": self.wht_signs, "d": self.d}, inverse=True)
        return y @ self.Pi

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize vectors to codebook indices.

        Algorithm 1, Quantize_mse:
            1. y = Pi @ x           (rotate)
            2. idx_j = argmin_k |y_j - c_k|  (nearest centroid per coordinate)

        Args:
            x: Input unit vectors of shape (batch, d) or (d,).

        Returns:
            Index tensor of shape (batch, d) or (d,) with values in [0, 2^bits).
        """
        y = self.rotate(x)
        # Use codebook's quantize which does brute-force nearest centroid
        return self.codebook.quantize(y)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct vectors from codebook indices.

        Algorithm 1, DeQuantize_mse:
            1. y_hat_j = c_{idx_j}  (centroid lookup)
            2. x_hat = Pi^T @ y_hat (unrotate)

        Args:
            indices: Index tensor of shape (batch, d) or (d,).

        Returns:
            Reconstructed vectors of same shape as indices, in float32.
        """
        # Centroid lookup using the registered buffer
        y_hat = self.centroids[indices]
        return self.unrotate(y_hat)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize and immediately dequantize (for training/evaluation).

        Args:
            x: Input unit vectors of shape (batch, d) or (d,).

        Returns:
            Tuple of (x_hat: reconstructed vectors, indices: codebook indices).
        """
        indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        return x_hat, indices
