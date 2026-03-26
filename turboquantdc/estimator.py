"""TurboQuant combined estimator — MSE quantization + QJL bias correction.

Implements Algorithm 2 (TurboQuant_prod) from the paper, which combines:
    Stage 1: PolarQuant MSE quantization with (b-1) bits per coordinate
    Stage 2: QJL 1-bit sign correction on the residual

The combined estimator produces UNBIASED inner product estimates:
    E[<q, k_hat>] = <q, k>

with distortion:
    D_prod <= sqrt(3)*pi^2/d * ||q||^2 / 4^b

Storage per vector: (b-1)*d bits (MSE) + d bits (QJL) + 16 bits (residual norm)
                  = b*d + 16 bits total

For non-unit vectors: normalize, store ||x|| in FP16, rescale in estimator.

Reference: TurboQuant paper (arxiv 2504.19874), Algorithm 2 / Theorem 2.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .polarquant import PolarQuant
from .qjl import QJL


class TurboQuantEstimator(nn.Module):
    """Combined two-stage estimator for unbiased inner products.

    Bit budget allocation (from paper):
        Total b bits per coordinate = (b-1) bits MSE + 1 bit QJL
        Example: 3-bit total = 2-bit MSE + 1-bit QJL signs

    The estimator handles non-unit vectors by normalizing before quantization
    and storing the original norm for rescaling.

    Args:
        d: Head dimension (e.g. 128).
        bits: Total effective bits per coordinate.
        qjl_dim: QJL projection dimension m. Defaults to d.
        seed: Random seed for rotation and projection matrices.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        bits: int,
        qjl_dim: int | None = None,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.d = d
        self.bits = bits

        # Bit budget: (bits-1) for MSE, 1 for QJL, with floor at 1
        mse_bits = max(bits - 1, 1)
        self.mse_bits = mse_bits

        # Stage 1: PolarQuant with mse_bits
        self.polar = PolarQuant(d, mse_bits, seed=seed, device=device)

        # Stage 2: QJL with seed+1 to get independent random matrix
        self.qjl = QJL(d, m=qjl_dim, seed=seed + 1, device=device)

    def quantize(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compress a vector using both MSE and QJL stages.

        Algorithm 2, Quantize_prod:
            1. Store ||x|| and normalize
            2. idx = Quantize_mse(x_normalized)    (Stage 1)
            3. x_mse = DeQuantize_mse(idx)         (reconstruct for residual)
            4. r = x_normalized - x_mse            (residual)
            5. signs = sign(S @ r)                  (Stage 2: QJL)
            6. Store (idx, signs, ||r||, ||x||)

        Args:
            x: Input vectors of shape (batch, d) or (d,).

        Returns:
            Dict with keys:
                - mse_indices: Tensor of codebook indices, shape (batch, d)
                - qjl_signs: Tensor of sign bits {-1,+1}, shape (batch, m)
                - residual_norm: Tensor of ||r||, shape (batch,)
                - vec_norm: Tensor of ||x||, shape (batch,)
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        # Store original norm and normalize
        vec_norm = x.norm(dim=-1, keepdim=True)  # (batch, 1)
        x_normalized = x / (vec_norm + 1e-8)  # (batch, d)

        # Stage 1: MSE quantization
        mse_indices = self.polar.quantize(x_normalized)  # (batch, d)
        x_mse = self.polar.dequantize(mse_indices)  # (batch, d)

        # Compute residual: r = x_normalized - x_mse
        residual = x_normalized - x_mse  # (batch, d)
        residual_norm = residual.norm(dim=-1)  # (batch,)

        # Stage 2: QJL on residual
        qjl_signs = self.qjl.project_and_sign(residual)  # (batch, m)

        result = {
            "mse_indices": mse_indices,
            "qjl_signs": qjl_signs,
            "residual_norm": residual_norm,
            "vec_norm": vec_norm.squeeze(-1),  # (batch,)
        }

        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result

    def dequantize_mse(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct vector using MSE stage only (no QJL correction).

        This gives the biased but low-noise MSE reconstruction, rescaled
        by the original vector norm.

        Args:
            compressed: Output from quantize().

        Returns:
            Reconstructed vectors of shape (batch, d) or (d,).
        """
        x_mse = self.polar.dequantize(compressed["mse_indices"])
        vec_norm = compressed["vec_norm"]

        # Rescale by original norm
        if vec_norm.dim() == 0:
            return x_mse * vec_norm
        return x_mse * vec_norm.unsqueeze(-1)

    def inner_product(
        self,
        query: torch.Tensor,
        compressed: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Estimate <query, key> using the unbiased two-stage estimator.

        <q, k> ~ <q, k_mse> + ||r|| * sqrt(pi/2)/m * <S@q, signs>

        The full estimator accounts for the original key norm:
            <q, k> ~ ||k|| * (<q, k_mse_normalized> + QJL_correction)

        Args:
            query: Query vectors, shape (batch_q, d) or (d,).
            compressed: Output from quantize() for key vectors.

        Returns:
            Estimated inner products.
            - If query is (d,) and single key: scalar
            - If query is (batch_q, d) and keys are (batch_k, d):
              shape (batch_q, batch_k)
        """
        squeeze_q = query.dim() == 1
        if squeeze_q:
            query = query.unsqueeze(0)

        # Reconstruct MSE approximation (normalized, without vec_norm scaling)
        x_mse = self.polar.dequantize(compressed["mse_indices"])

        # Term 1: <query, k_mse> (MSE inner product)
        if x_mse.dim() == 1:
            # Single key: (batch_q, d) @ (d,) -> (batch_q,)
            term1 = query @ x_mse
        else:
            # Batched keys: (batch_q, d) @ (d, batch_k) -> (batch_q, batch_k)
            term1 = query @ x_mse.T

        # Term 2: QJL correction
        term2 = self.qjl.inner_product_correction(
            query=query,
            signs=compressed["qjl_signs"],
            residual_norm=compressed["residual_norm"],
        )

        # Combined estimator, scaled by original key norm
        vec_norm = compressed["vec_norm"]
        result = (term1 + term2) * vec_norm

        if squeeze_q and result.dim() > 0 and result.shape[0] == 1:
            result = result.squeeze(0)

        return result
