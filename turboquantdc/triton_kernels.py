"""Triton fused kernels for TurboQuant encode/decode/inner product.

Provides three fused kernels that replace the PyTorch hot path:

1. triton_quantize     -- Fused rotate + quantize + residual norm + QJL sign
2. triton_inner_product -- Fused rotate query + MSE IP + QJL correction + scale
3. triton_dequantize   -- Fused centroid lookup + inverse rotation + rescale

Wrapper class TritonTurboQuant is a drop-in replacement for TurboQuantEstimator.

Key optimization in quantize: precompute SR = S @ R^T on the host so that
    sign(S @ r) = sign(S @ R^T @ residual_rot) = sign(SR @ residual_rot)
avoids the costly inverse rotation during quantization. Since orthogonal
transforms preserve norms, ||r|| = ||residual_rot|| is computed cheaply.

Hardware target: RTX 4090 (SM 89), CUDA 12.8, Triton 3.6.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import triton
import triton.language as tl

from .codebook import LloydMaxCodebook
from .rotation import generate_qjl_matrix, generate_rotation_matrix


# ---------------------------------------------------------------------------
# Kernel 1: Fused Quantize (rotate + quantize + residual + QJL sign)
# ---------------------------------------------------------------------------


@triton.jit
def _quantize_kernel(
    # Inputs
    x_ptr,              # (batch, d) normalized input, float32, contiguous
    R_ptr,              # (d, d) rotation matrix, float32, contiguous (row-major)
    boundaries_ptr,     # (n_boundaries,) sorted boundaries
    centroids_ptr,      # (n_centroids,) sorted centroids
    SR_ptr,             # (m, d) precomputed S @ R^T, float32, contiguous
    # Outputs
    indices_ptr,        # (batch, d) int32 codebook indices
    signs_ptr,          # (batch, m) float32 sign bits {-1, +1}
    r_norm_ptr,         # (batch,) float32 residual norms
    # Dimensions
    d: tl.constexpr,
    m: tl.constexpr,
    n_centroids: tl.constexpr,
    n_boundaries: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Fused quantize: rotate -> boundary search -> residual -> QJL sign.

    Each program handles one vector (one row of the batch).
    BLOCK_D must be >= d and a power of 2. BLOCK_M must be >= m.
    All matrix-vector products are computed as single-tile reductions
    (valid for d, m <= 256).
    """
    pid = tl.program_id(0)  # batch index

    # Common offset vectors
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < d

    # ---- Phase 1: Rotate x -> y = x @ R^T ----
    # y[j] = sum_k x[k] * R[j, k]
    # Load full input vector
    x_vals = tl.load(x_ptr + pid * d + d_offs, mask=d_mask, other=0.0)

    # Matrix-vector product: y = R @ x (treating x as column vector)
    # R is (d, d), row-major: R[j, k] at offset j*d + k
    r_block = tl.load(
        R_ptr + d_offs[:, None] * d + d_offs[None, :],
        mask=d_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    y_vals = tl.sum(r_block * x_vals[None, :], axis=1)

    # ---- Phase 2: Quantize via boundary search ----
    # Count how many boundaries each y[j] exceeds -> gives centroid index.
    idx = tl.zeros([BLOCK_D], dtype=tl.int32)
    for b_i in range(n_boundaries):
        b_val = tl.load(boundaries_ptr + b_i)
        idx += tl.where(y_vals > b_val, 1, 0).to(tl.int32)

    # ---- Phase 3: Centroid lookup + residual in rotated space ----
    y_hat = tl.load(centroids_ptr + idx, mask=d_mask, other=0.0)
    res_rot = y_vals - y_hat

    # ||r|| = ||residual_rot|| (orthogonal transform preserves norms)
    r_norm_sq = tl.sum(res_rot * res_rot, axis=0)
    r_norm_val = tl.sqrt(r_norm_sq)

    # ---- Phase 4: QJL sign via precomputed SR = S @ R^T ----
    # sign(S @ r) = sign(SR @ residual_rot)
    # proj[i] = sum_k SR[i, k] * res_rot[k]
    m_offs = tl.arange(0, BLOCK_M)
    m_mask = m_offs < m

    sr_block = tl.load(
        SR_ptr + m_offs[:, None] * d + d_offs[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    proj = tl.sum(sr_block * res_rot[None, :], axis=1)
    sign_vals = tl.where(proj >= 0.0, 1.0, -1.0)

    # ---- Store outputs ----
    tl.store(indices_ptr + pid * d + d_offs, idx, mask=d_mask)
    tl.store(signs_ptr + pid * m + m_offs, sign_vals, mask=m_mask)
    tl.store(r_norm_ptr + pid, r_norm_val)


# ---------------------------------------------------------------------------
# Kernel 2: Fused Inner Product Estimation
# ---------------------------------------------------------------------------
#
# Strategy: The expensive query rotation (q @ R^T) and query projection
# (S @ q) are done ONCE per query in PyTorch (leveraging cuBLAS).
# The Triton kernel then handles the per-key gather+dot+scale which is
# memory-bound and benefits from fusion.
#
# This hybrid approach beats both:
# - Pure PyTorch: which launches 3+ separate kernels per query
# - Pure Triton: which redundantly recomputes query rotation per key
# ---------------------------------------------------------------------------


@triton.jit
def _inner_product_kernel(
    # Pre-rotated query
    q_rot_ptr,          # (n_queries, d) float32 -- already rotated
    # Pre-projected query
    sq_ptr,             # (n_queries, m) float32 -- S @ q, already computed
    # Compressed keys
    indices_ptr,        # (n_keys, d) int32
    centroids_ptr,      # (n_centroids,) float32
    signs_ptr,          # (n_keys, m) float32
    r_norm_ptr,         # (n_keys,) float32
    vec_norm_ptr,       # (n_keys,) float32
    # Output
    output_ptr,         # (n_queries, n_keys) float32
    # Scale constant: sqrt(pi/2) / m, precomputed as float
    qjl_scale,
    # Dimensions
    n_keys,
    d: tl.constexpr,
    m: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-key inner product: MSE dot + QJL dot + scale.

    Grid: (n_queries, ceil(n_keys / BLOCK_K)).
    Each program handles one query and BLOCK_K keys.
    The query rotation and S@q projection are precomputed by the caller.
    """
    q_idx = tl.program_id(0)
    k_block = tl.program_id(1)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < d
    m_offs = tl.arange(0, BLOCK_M)
    m_mask = m_offs < m

    # Load pre-rotated query and pre-projected S@q (once per program)
    q_rot = tl.load(q_rot_ptr + q_idx * d + d_offs, mask=d_mask, other=0.0)
    s_q = tl.load(sq_ptr + q_idx * m + m_offs, mask=m_mask, other=0.0)

    # Process BLOCK_K keys
    k_offs = tl.arange(0, BLOCK_K)
    for ki in range(BLOCK_K):
        k_idx = k_block * BLOCK_K + ki
        if k_idx < n_keys:
            # Term 1: MSE inner product = <q_rot, centroids[key_indices]>
            key_idx = tl.load(
                indices_ptr + k_idx * d + d_offs, mask=d_mask, other=0,
            )
            key_cents = tl.load(centroids_ptr + key_idx, mask=d_mask, other=0.0)
            ip_mse = tl.sum(key_cents * q_rot, axis=0)

            # Term 2: QJL correction = scale * r_norm * <S@q, signs>
            key_signs = tl.load(
                signs_ptr + k_idx * m + m_offs, mask=m_mask, other=0.0,
            )
            qjl_dot = tl.sum(s_q * key_signs, axis=0)
            r_norm_val = tl.load(r_norm_ptr + k_idx)
            correction = qjl_scale * r_norm_val * qjl_dot

            # Combined: (ip_mse + correction) * vec_norm
            vec_norm_val = tl.load(vec_norm_ptr + k_idx)
            result = (ip_mse + correction) * vec_norm_val

            tl.store(output_ptr + q_idx * n_keys + k_idx, result)


# ---------------------------------------------------------------------------
# Kernel 3: Fused Dequantize (centroid lookup + inverse rotation + rescale)
# ---------------------------------------------------------------------------


@triton.jit
def _dequantize_kernel(
    indices_ptr,        # (batch, d) int32
    centroids_ptr,      # (n_centroids,) float32
    R_ptr,              # (d, d) float32, contiguous row-major
    vec_norm_ptr,       # (batch,) float32
    output_ptr,         # (batch, d) float32
    # Dimensions
    d: tl.constexpr,
    n_centroids: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused dequantize: centroid lookup -> inverse rotation -> rescale.

    Each program handles one vector.
    x_hat = vec_norm * (y_hat @ R), where y_hat[j] = centroids[indices[j]].
    Inverse rotation: x = y_hat @ R means x[j] = sum_k y_hat[k] * R[k, j].
    """
    pid = tl.program_id(0)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < d

    # ---- Centroid lookup ----
    idx = tl.load(indices_ptr + pid * d + d_offs, mask=d_mask, other=0)
    y_hat = tl.load(centroids_ptr + idx, mask=d_mask, other=0.0)

    # ---- Inverse rotation: x[j] = sum_k y_hat[k] * R[k, j] ----
    # R^T[j, k] = R[k, j], so x = R^T^T @ y_hat... no.
    # y_hat @ R: x[j] = sum_k y_hat[k] * R[k, j]
    # Load R as (BLOCK_D, BLOCK_D), compute R^T @ y_hat
    # R[k, j] at offset k*d + j
    r_block = tl.load(
        R_ptr + d_offs[:, None] * d + d_offs[None, :],
        mask=d_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    # r_block[k, j] = R[k, j]
    # x[j] = sum_k y_hat[k] * R[k, j] = sum_k y_hat[k] * r_block[k, j]
    # = (r_block^T @ y_hat)[j] = sum_k r_block[k, j] * y_hat[k]
    # With broadcasting: r_block.T * y_hat[None, :] then sum over axis=1
    # But we have r_block as (BLOCK_D, BLOCK_D) where dim0=k, dim1=j
    # x[j] = sum over dim0 of (r_block[:, j] * y_hat[:])
    x_hat = tl.sum(r_block * y_hat[:, None], axis=0)

    # Rescale by original vector norm
    vn = tl.load(vec_norm_ptr + pid)
    x_hat = x_hat * vn

    tl.store(output_ptr + pid * d + d_offs, x_hat, mask=d_mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def _next_power_of_2(n: int) -> int:
    """Round up to the next power of 2."""
    p = 1
    while p < n:
        p <<= 1
    return p


def triton_quantize(
    x_normalized: torch.Tensor,
    R: torch.Tensor,
    boundaries: torch.Tensor,
    centroids: torch.Tensor,
    SR: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused quantize using Triton kernel.

    Args:
        x_normalized: (batch, d) normalized input vectors on CUDA, contiguous.
        R: (d, d) rotation matrix, contiguous.
        boundaries: (n_boundaries,) quantization boundaries.
        centroids: (n_centroids,) quantization centroids.
        SR: (m, d) precomputed S @ R^T matrix, contiguous.

    Returns:
        Tuple of (indices [int32], signs [float32], residual_norms [float32]).
    """
    batch, d = x_normalized.shape
    m = SR.shape[0]
    n_cent = centroids.shape[0]
    n_bound = boundaries.shape[0]
    BLOCK_D = _next_power_of_2(d)
    BLOCK_M = _next_power_of_2(m)

    indices = torch.empty(batch, d, dtype=torch.int32, device=x_normalized.device)
    signs = torch.empty(batch, m, dtype=torch.float32, device=x_normalized.device)
    r_norms = torch.empty(batch, dtype=torch.float32, device=x_normalized.device)

    _quantize_kernel[(batch,)](
        x_normalized, R, boundaries, centroids, SR,
        indices, signs, r_norms,
        d, m, n_cent, n_bound,
        BLOCK_D=BLOCK_D, BLOCK_M=BLOCK_M,
    )
    return indices, signs, r_norms


def triton_inner_product(
    queries: torch.Tensor,
    R: torch.Tensor,
    indices: torch.Tensor,
    centroids: torch.Tensor,
    signs: torch.Tensor,
    S: torch.Tensor,
    r_norms: torch.Tensor,
    vec_norms: torch.Tensor,
) -> torch.Tensor:
    """Hybrid Triton+cuBLAS inner product estimation.

    Precomputes query rotation and S@q via PyTorch (cuBLAS), then uses a
    lightweight Triton kernel for the per-key gather+dot+scale.

    Args:
        queries: (n_queries, d) query vectors, contiguous.
        R: (d, d) rotation matrix, contiguous.
        indices: (n_keys, d) int32 MSE codebook indices.
        centroids: (n_centroids,) centroid values.
        signs: (n_keys, m) QJL sign bits.
        S: (m, d) QJL projection matrix, contiguous.
        r_norms: (n_keys,) residual norms.
        vec_norms: (n_keys,) original vector norms.

    Returns:
        (n_queries, n_keys) inner product estimates.
    """
    n_queries = queries.shape[0]
    n_keys = indices.shape[0]
    d = queries.shape[1]
    m = signs.shape[1]
    BLOCK_D = _next_power_of_2(d)
    BLOCK_M = _next_power_of_2(m)
    BLOCK_K = 16  # keys per program -- tuned for RTX 4090

    # Precompute query rotation and projection via cuBLAS (fast for small n_queries)
    q_rot = (queries @ R.T).contiguous()   # (n_queries, d)
    s_q = (queries @ S.T).contiguous()     # (n_queries, m)

    qjl_scale = math.sqrt(math.pi / 2.0) / m

    output = torch.empty(
        n_queries, n_keys, dtype=torch.float32, device=queries.device,
    )

    grid = (n_queries, (n_keys + BLOCK_K - 1) // BLOCK_K)
    _inner_product_kernel[grid](
        q_rot, s_q,
        indices, centroids, signs, r_norms, vec_norms,
        output,
        qjl_scale,
        n_keys,
        d, m,
        BLOCK_D=BLOCK_D, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
    )
    return output


def triton_dequantize(
    indices: torch.Tensor,
    centroids: torch.Tensor,
    R: torch.Tensor,
    vec_norms: torch.Tensor,
) -> torch.Tensor:
    """Fused dequantize using Triton kernel.

    Args:
        indices: (batch, d) int32 MSE codebook indices.
        centroids: (n_centroids,) centroid values.
        R: (d, d) rotation matrix, contiguous.
        vec_norms: (batch,) original vector norms.

    Returns:
        (batch, d) dequantized and rescaled vectors.
    """
    batch, d = indices.shape
    n_cent = centroids.shape[0]
    BLOCK_D = _next_power_of_2(d)

    output = torch.empty(batch, d, dtype=torch.float32, device=indices.device)

    _dequantize_kernel[(batch,)](
        indices, centroids, R, vec_norms, output,
        d, n_cent,
        BLOCK_D=BLOCK_D,
    )
    return output


# ---------------------------------------------------------------------------
# Wrapper Class: Drop-in replacement for TurboQuantEstimator
# ---------------------------------------------------------------------------


class TritonTurboQuant:
    """Drop-in replacement for TurboQuantEstimator using fused Triton kernels.

    Same API as TurboQuantEstimator but fuses multiple kernel launches into
    single Triton kernels for significant speedup on GPU.

    Key optimization: precomputes SR = S @ R^T so that the QJL projection
    during quantization becomes sign(SR @ residual_rot), avoiding an extra
    d x d matrix-vector multiply per vector.

    Args:
        d: Head dimension (e.g. 128).
        bits: Total effective bits per coordinate (e.g. 3).
        qjl_dim: QJL projection dimension m. Defaults to d.
        seed: Random seed for rotation and projection matrices.
        device: Target device (must be 'cuda' or a CUDA device).
    """

    def __init__(
        self,
        d: int,
        bits: int = 3,
        qjl_dim: int | None = None,
        seed: int = 42,
        device: str | torch.device = "cuda",
    ):
        self.d = d
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.m = qjl_dim if qjl_dim is not None else d
        self.device = torch.device(device)

        # Generate rotation matrix Pi (d x d orthogonal) -- MUST be contiguous
        self.R = generate_rotation_matrix(
            d, seed=seed, device="cpu",
        ).contiguous().to(self.device)

        # Generate QJL projection matrix S (m x d)
        self.S = generate_qjl_matrix(
            d, m=self.m, seed=seed + 1, device="cpu",
        ).contiguous().to(self.device)

        # Precompute SR = S @ R^T for fused QJL in rotated space
        self.SR = (self.S @ self.R.T).contiguous()

        # Solve Lloyd-Max codebook
        codebook = LloydMaxCodebook(d, self.mse_bits)
        self.centroids = codebook.centroids.to(self.device)
        self.boundaries = codebook.boundaries.to(self.device)
        self.n_centroids = codebook.n_levels

    def quantize(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compress vectors using fused Triton quantization kernel.

        Args:
            x: Input vectors of shape (batch, d) or (d,).

        Returns:
            Dict with mse_indices, qjl_signs, residual_norm, vec_norm.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        # Normalize
        vec_norm = x.norm(dim=-1, keepdim=True)
        x_normalized = (x / (vec_norm + 1e-8)).contiguous()

        indices, signs, r_norms = triton_quantize(
            x_normalized, self.R, self.boundaries, self.centroids, self.SR,
        )

        result = {
            "mse_indices": indices,
            "qjl_signs": signs,
            "residual_norm": r_norms,
            "vec_norm": vec_norm.squeeze(-1),
        }

        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result

    def inner_product(
        self,
        query: torch.Tensor,
        compressed: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Estimate <query, key> using fused Triton inner product kernel.

        Args:
            query: Query vectors, shape (batch_q, d) or (d,).
            compressed: Output from quantize() for key vectors.

        Returns:
            Estimated inner products.
        """
        squeeze_q = query.dim() == 1
        if squeeze_q:
            query = query.unsqueeze(0)

        indices = compressed["mse_indices"]
        signs = compressed["qjl_signs"]
        r_norms = compressed["residual_norm"]
        vec_norms = compressed["vec_norm"]

        squeeze_k = indices.dim() == 1
        if squeeze_k:
            indices = indices.unsqueeze(0)
            signs = signs.unsqueeze(0)
            r_norms = r_norms.unsqueeze(0)
            vec_norms = vec_norms.unsqueeze(0)

        query = query.contiguous()

        result = triton_inner_product(
            query, self.R, indices, self.centroids, signs, self.S,
            r_norms, vec_norms,
        )

        if squeeze_q and result.dim() > 0 and result.shape[0] == 1:
            result = result.squeeze(0)
        if squeeze_k and result.dim() > 0 and result.shape[-1] == 1:
            result = result.squeeze(-1)

        return result

    def dequantize_mse(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct vectors using MSE path (centroid lookup + unrotate + rescale).

        Uses PyTorch for dequantization since the gather + cuBLAS matmul path
        is already highly optimized. The Triton dequantize kernel is available
        via triton_dequantize() for cases where single-kernel launch matters.

        Args:
            compressed: Output from quantize().

        Returns:
            Reconstructed vectors of shape (batch, d) or (d,).
        """
        indices = compressed["mse_indices"]
        vec_norms = compressed["vec_norm"]

        # Centroid lookup (gather)
        y_hat = self.centroids[indices.long()]

        # Inverse rotation: x_hat = y_hat @ R
        x_hat = y_hat @ self.R

        # Rescale by original norm
        if vec_norms.dim() == 0:
            return x_hat * vec_norms
        return x_hat * vec_norms.unsqueeze(-1)
