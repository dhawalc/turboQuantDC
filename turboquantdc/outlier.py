"""Outlier channel strategy for fractional bit rates.

Extends TurboQuant to support non-integer bit-widths (e.g. 2.5, 3.5) by
splitting channels into two groups after rotation:
    - n_high channels quantized at ceil(target_bits)
    - n_low channels quantized at floor(target_bits)

The rotation is applied once to the full d-dimensional vector, then channels
are split. This preserves the Gaussianization property that makes quantization
optimal — each coordinate of the rotated vector follows the same concentrated
distribution regardless of which group it is assigned to.

The key decomposition for inner products:
    <q, k> = <q_high, k_high> + <q_low, k_low>
where q_high/q_low and k_high/k_low are the rotated vectors split at the
same channel boundary.

Reference: TurboQuant paper (arxiv 2504.19874), non-integer bit precision
discussion. Implementation based on TheTom/turboquant_plus outlier strategy.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from .codebook import LloydMaxCodebook
from .qjl import QJL
from .rotation import generate_rotation_matrix


class OutlierTurboQuant(nn.Module):
    """TurboQuant with outlier channel strategy for fractional bit rates.

    Splits the d dimensions into two groups after rotation:
    - n_high channels quantized at ceil(target_bits) using full TurboQuant
    - n_low channels quantized at floor(target_bits) using full TurboQuant

    The split is determined by target_bits:
        frac = target_bits - floor(target_bits)
        n_high = round(d * frac)
        n_low = d - n_high

    Example (d=128, target_bits=2.5):
        n_high = 64 channels at 3-bit (2-bit MSE + 1-bit QJL)
        n_low = 64 channels at 2-bit (1-bit MSE + 1-bit QJL)
        Average: (64*3 + 64*2) / 128 = 2.5 bits/channel

    The rotation matrix is shared -- it is applied once to the full vector,
    then channels are split. This is important because rotation makes all
    channels equally informative (no "natural" outliers).

    For integer target_bits (e.g. 3.0), n_high=0 and the entire vector
    is quantized at floor(target_bits), behaving like standard TurboQuant.

    Args:
        d: Vector dimension.
        target_bits: Target average bits per coordinate (e.g., 2.5, 3.5).
        seed: Random seed.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        target_bits: float,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.d = d
        self.target_bits = target_bits

        low_bits = int(target_bits)   # floor
        high_bits = low_bits + 1      # ceil
        frac = target_bits - low_bits

        self.n_high = int(round(d * frac))
        self.n_low = d - self.n_high
        self.high_bits = high_bits
        self.low_bits = low_bits

        # Effective bit rate (may differ slightly from target due to rounding)
        if d > 0:
            self.effective_bits = (
                self.n_high * high_bits + self.n_low * low_bits
            ) / d
        else:
            self.effective_bits = target_bits

        # Shared rotation for the FULL d-dimensional vector
        Pi = generate_rotation_matrix(d, seed=seed + 2000, device="cpu")
        self.register_buffer("Pi", Pi.to(device))

        # --- High-bit channel group ---
        # MSE bits = total_bits - 1 (1 bit reserved for QJL)
        if self.n_high > 0:
            high_mse_bits = max(high_bits - 1, 1)
            self.high_codebook = LloydMaxCodebook(self.n_high, high_mse_bits)
            self.register_buffer(
                "high_centroids",
                self.high_codebook.centroids.to(device),
            )
            self.high_qjl = QJL(
                self.n_high, m=self.n_high, seed=seed, device=device
            )
        else:
            self.high_codebook = None
            self.high_qjl = None

        # --- Low-bit channel group ---
        if self.n_low > 0:
            low_mse_bits = max(low_bits - 1, 1)
            self.low_codebook = LloydMaxCodebook(self.n_low, low_mse_bits)
            self.register_buffer(
                "low_centroids",
                self.low_codebook.centroids.to(device),
            )
            self.low_qjl = QJL(
                self.n_low, m=self.n_low, seed=seed + 1000, device=device
            )
        else:
            self.low_codebook = None
            self.low_qjl = None

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared full-d rotation: y = x @ Pi.T."""
        return x @ self.Pi.T

    def _unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation: x = y @ Pi."""
        return y @ self.Pi

    def _split(self, y: torch.Tensor):
        """Split rotated vector into high and low channel groups."""
        y_high = y[..., : self.n_high]
        y_low = y[..., self.n_high :]
        return y_high, y_low

    def _quantize_group(
        self,
        y: torch.Tensor,
        codebook: LloydMaxCodebook,
        centroids_buf: torch.Tensor,
        qjl: QJL,
    ) -> Dict[str, torch.Tensor]:
        """Quantize a channel group using MSE + QJL (in already-rotated space).

        This replicates the TurboQuantEstimator pipeline but without
        applying rotation (since the shared rotation was already applied).

        Args:
            y: Already-rotated vectors for this channel group, (batch, n_ch).
            codebook: LloydMaxCodebook for this group.
            centroids_buf: Registered buffer of centroids on the correct device.
            qjl: QJL instance for this group.

        Returns:
            Dict with mse_indices, qjl_signs, residual_norm.
        """
        # MSE quantize in rotated space (no rotation needed — already done)
        mse_indices = codebook.quantize(y)         # (batch, n_ch)
        y_mse = centroids_buf[mse_indices]         # (batch, n_ch)

        # Residual in rotated space
        residual = y - y_mse                       # (batch, n_ch)
        residual_norm = residual.norm(dim=-1)      # (batch,)

        # QJL on residual
        qjl_signs = qjl.project_and_sign(residual)  # (batch, m)

        return {
            "mse_indices": mse_indices,
            "qjl_signs": qjl_signs,
            "residual_norm": residual_norm,
        }

    def _ip_group(
        self,
        q_ch: torch.Tensor,
        compressed: Dict[str, torch.Tensor],
        centroids_buf: torch.Tensor,
        qjl: QJL,
    ) -> torch.Tensor:
        """Estimate inner products for one channel group.

        Computes: <q_ch, k_ch> ~ <q_ch, k_mse> + QJL_correction

        Args:
            q_ch: Rotated query channels, (batch_q, n_ch).
            compressed: Output of _quantize_group for key channels.
            centroids_buf: Registered buffer of centroids.
            qjl: QJL instance for this group.

        Returns:
            (batch_q, batch_k) inner product estimates.
        """
        # MSE reconstruction in rotated space
        k_mse = centroids_buf[compressed["mse_indices"]]  # (batch_k, n_ch)

        # Term 1: MSE inner product
        if k_mse.dim() == 1:
            term1 = q_ch @ k_mse
        else:
            term1 = q_ch @ k_mse.T  # (batch_q, batch_k)

        # Term 2: QJL correction
        term2 = qjl.inner_product_correction(
            query=q_ch,
            signs=compressed["qjl_signs"],
            residual_norm=compressed["residual_norm"],
        )

        return term1 + term2

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def quantize(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Quantize vectors using outlier channel strategy.

        Pipeline:
            1. Store ||x||, normalize to unit vector
            2. Rotate: y = x_normalized @ Pi.T  (full d x d rotation)
            3. Split: y_high = y[..., :n_high], y_low = y[..., n_high:]
            4. Quantize each group with its own codebook + QJL

        Args:
            x: Input vectors, shape (batch, d) or (d,).

        Returns:
            Dict with keys:
                high_compressed: compressed data for high-bit channels (or None)
                low_compressed: compressed data for low-bit channels (or None)
                vec_norm: original vector norms, shape (batch,) or scalar
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        # Store norm and normalize
        vec_norm = x.norm(dim=-1)                  # (batch,)
        x_normalized = x / (vec_norm.unsqueeze(-1) + 1e-8)

        # Shared rotation on full vector
        y = self._rotate(x_normalized)             # (batch, d)

        # Split into channel groups
        y_high, y_low = self._split(y)

        # Quantize each group
        high_compressed = None
        if self.n_high > 0 and self.high_codebook is not None:
            high_compressed = self._quantize_group(
                y_high, self.high_codebook, self.high_centroids, self.high_qjl
            )

        low_compressed = None
        if self.n_low > 0 and self.low_codebook is not None:
            low_compressed = self._quantize_group(
                y_low, self.low_codebook, self.low_centroids, self.low_qjl
            )

        result = {
            "high_compressed": high_compressed,
            "low_compressed": low_compressed,
            "vec_norm": vec_norm,
        }

        if squeeze:
            result["vec_norm"] = vec_norm.squeeze(0)
            if high_compressed is not None:
                result["high_compressed"] = {
                    k: v.squeeze(0) for k, v in high_compressed.items()
                }
            if low_compressed is not None:
                result["low_compressed"] = {
                    k: v.squeeze(0) for k, v in low_compressed.items()
                }

        return result

    def inner_product(
        self,
        queries: torch.Tensor,
        compressed: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Estimate inner products using split-channel compressed vectors.

        The key insight: <q, k> = <q_high, k_high> + <q_low, k_low>
        where q_high/q_low are the rotated query split into the same channels.
        Each sub-inner-product is estimated by its own codebook + QJL.

        The full estimator accounts for the original key norm:
            <q, k> ~ ||k|| * (<q_rot_high, k_high_est> + <q_rot_low, k_low_est>)

        Args:
            queries: Query vectors, shape (batch_q, d) or (d,).
            compressed: Output from quantize() for key vectors.

        Returns:
            Estimated inner products.
            - If query is (d,) and single key: scalar
            - If query is (batch_q, d) and keys are (batch_k, d):
              shape (batch_q, batch_k)
        """
        squeeze_q = queries.dim() == 1
        if squeeze_q:
            queries = queries.unsqueeze(0)

        # Rotate query with the shared rotation
        q_rot = self._rotate(queries)              # (batch_q, d)

        # Split query into same channel groups
        q_high, q_low = self._split(q_rot)

        # Accumulate inner product from both groups
        result = torch.zeros(1, device=queries.device)  # will be replaced

        high_comp = compressed["high_compressed"]
        low_comp = compressed["low_compressed"]

        if self.n_high > 0 and high_comp is not None:
            ip_high = self._ip_group(
                q_high, high_comp, self.high_centroids, self.high_qjl
            )
            result = ip_high
        else:
            # Initialize to zero with correct shape
            # Determine batch_k from low_comp
            if low_comp is not None:
                bk = low_comp["residual_norm"]
                if bk.dim() == 0:
                    result = torch.zeros(
                        queries.shape[0], device=queries.device
                    )
                else:
                    result = torch.zeros(
                        queries.shape[0], bk.shape[0], device=queries.device
                    )

        if self.n_low > 0 and low_comp is not None:
            ip_low = self._ip_group(
                q_low, low_comp, self.low_centroids, self.low_qjl
            )
            if self.n_high > 0 and high_comp is not None:
                result = result + ip_low
            else:
                result = ip_low

        # Scale by original key norm
        vec_norm = compressed["vec_norm"]
        result = result * vec_norm

        if squeeze_q and result.dim() > 0 and result.shape[0] == 1:
            result = result.squeeze(0)

        return result

    def compression_ratio(self) -> float:
        """Compute compression ratio vs FP16.

        Storage per vector:
            High group: high_bits * n_high bits (MSE indices + QJL signs)
            Low group:  low_bits * n_low bits
            Overhead:   16 bits (vec_norm as FP16)
            Per-group QJL also stores residual_norm: 16 bits each

        Total = high_bits*n_high + low_bits*n_low + 16 (vec_norm)
              + 16 (high residual_norm, if n_high > 0)
              + 16 (low residual_norm, if n_low > 0)

        Baseline: d * 16 bits (FP16).
        """
        bits_per_vec = (
            self.n_high * self.high_bits
            + self.n_low * self.low_bits
            + 16  # vec_norm
        )
        # Add residual_norm overhead for each active group
        if self.n_high > 0:
            bits_per_vec += 16
        if self.n_low > 0:
            bits_per_vec += 16

        fp16_bits = self.d * 16
        return fp16_bits / bits_per_vec
