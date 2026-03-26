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
from .rotation import generate_rotation_matrix


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
    """

    def __init__(
        self,
        d: int,
        bits: int,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.d = d
        self.bits = bits

        # Generate rotation matrix Pi (d x d orthogonal)
        Pi = generate_rotation_matrix(d, seed=seed, device="cpu")
        self.register_buffer("Pi", Pi.to(device))

        # Solve Lloyd-Max codebook for this (d, bits) configuration
        self.codebook = LloydMaxCodebook(d, bits)
        # Register centroids as buffer for device tracking
        self.register_buffer("centroids", self.codebook.centroids.to(device))

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random orthogonal rotation.

        y = x @ Pi.T  (equivalent to Pi @ x for each vector)

        After rotation, each coordinate y_j follows the concentrated
        distribution from Lemma 1, enabling scalar quantization.

        Args:
            x: Input vectors of shape (batch, d) or (d,).

        Returns:
            Rotated vectors of same shape.
        """
        return x @ self.Pi.T

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation.

        x = y @ Pi  (since Pi is orthogonal, Pi^{-1} = Pi^T, so Pi^T^{-1} = Pi)

        Args:
            y: Rotated vectors of shape (batch, d) or (d,).

        Returns:
            Unrotated vectors of same shape.
        """
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
