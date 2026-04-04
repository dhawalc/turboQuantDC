"""CUDA backend for TurboQuantDC dequantize and WHT kernels.

Drop-in replacement for triton_kernels.py functions. Provides:
  - cuda_dequantize()          -- same API as triton_dequantize()
  - cuda_dequantize_residual() -- same API as triton_dequantize_residual()
  - cuda_wht_rotate()          -- same API as triton_wht_rotate()
  - cuda_wht_unrotate()        -- same API as triton_wht_unrotate()
  - CUDATurboQuant             -- drop-in for TritonTurboQuant (dequantize only)

Backend selection order: CUDA -> Triton -> PyTorch fallback.
Set TURBOQUANTDC_BACKEND=cuda|triton|pytorch to force a backend.

Usage:
    from turboquantdc.cuda_kernels import cuda_dequantize, CUDATurboQuant

    # Direct function call
    output = cuda_dequantize(indices, centroids, R, vec_norms)

    # Class wrapper
    tq = CUDATurboQuant(d=128, bits=3, device="cuda")
    compressed = tq.quantize(x)
    x_hat = tq.dequantize_mse(compressed)
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend loading with fallback chain
# ---------------------------------------------------------------------------

_CUDA_DEQUANTIZE = None
_CUDA_WHT = None
_BACKEND_CHECKED = False


def _ensure_backend():
    """Load CUDA modules on first call. Thread-safe via Python GIL."""
    global _CUDA_DEQUANTIZE, _CUDA_WHT, _BACKEND_CHECKED
    if _BACKEND_CHECKED:
        return
    _BACKEND_CHECKED = True

    forced = os.environ.get("TURBOQUANTDC_BACKEND", "").lower()
    if forced == "pytorch":
        logger.info("TURBOQUANTDC_BACKEND=pytorch — skipping CUDA compilation")
        return
    if forced == "triton":
        logger.info("TURBOQUANTDC_BACKEND=triton — skipping CUDA compilation")
        return

    try:
        from .cuda.build import load_dequantize, load_wht
        _CUDA_DEQUANTIZE = load_dequantize()
        _CUDA_WHT = load_wht()
        if _CUDA_DEQUANTIZE is not None:
            logger.info("CUDA dequantize backend loaded")
        if _CUDA_WHT is not None:
            logger.info("CUDA WHT backend loaded (d up to 2048)")
    except Exception as e:
        logger.warning("Failed to load CUDA backend: %s", e)


def is_cuda_available() -> bool:
    """Check if CUDA dequantize kernel is available."""
    _ensure_backend()
    return _CUDA_DEQUANTIZE is not None


def is_cuda_wht_available() -> bool:
    """Check if CUDA WHT kernel is available."""
    _ensure_backend()
    return _CUDA_WHT is not None


# ---------------------------------------------------------------------------
# Dequantize functions
# ---------------------------------------------------------------------------


def cuda_dequantize(
    indices: torch.Tensor,
    centroids: torch.Tensor,
    R: torch.Tensor,
    vec_norms: torch.Tensor,
) -> torch.Tensor:
    """Fused dequantize using raw CUDA kernel.

    Same API as triton_dequantize from triton_kernels.py.

    Args:
        indices: (batch, d) int32 MSE codebook indices.
        centroids: (n_centroids,) centroid values.
        R: (d, d) rotation matrix, contiguous.
        vec_norms: (batch,) original vector norms.

    Returns:
        (batch, d) dequantized and rescaled vectors in float32.
    """
    _ensure_backend()

    if _CUDA_DEQUANTIZE is not None:
        return _CUDA_DEQUANTIZE.dequantize_mse(
            indices.contiguous().to(torch.int32),
            centroids.contiguous().float(),
            R.contiguous().float(),
            vec_norms.contiguous().float(),
        )

    # Fallback: try Triton
    try:
        from .triton_kernels import triton_dequantize
        return triton_dequantize(indices, centroids, R, vec_norms)
    except (ImportError, RuntimeError):
        pass

    # Fallback: PyTorch
    return _pytorch_dequantize(indices, centroids, R, vec_norms)


def cuda_dequantize_residual(
    indices: torch.Tensor,
    centroids: torch.Tensor,
    R: torch.Tensor,
    vec_norms: torch.Tensor,
    res_signs: torch.Tensor,
    res_scale: torch.Tensor,
) -> torch.Tensor:
    """Fused dequantize with residual correction using raw CUDA kernel.

    Same API as triton_dequantize_residual from triton_kernels.py.

    Args:
        indices: (batch, d) int32 MSE codebook indices.
        centroids: (n_centroids,) centroid values.
        R: (d, d) rotation matrix, contiguous.
        vec_norms: (batch,) corrected norms.
        res_signs: (batch, d) float32 residual signs {-1, 0, +1}.
        res_scale: (batch,) float32 residual scales.

    Returns:
        (batch, d) dequantized and rescaled vectors in float32.
    """
    _ensure_backend()

    if _CUDA_DEQUANTIZE is not None:
        return _CUDA_DEQUANTIZE.dequantize_residual(
            indices.contiguous().to(torch.int32),
            centroids.contiguous().float(),
            R.contiguous().float(),
            vec_norms.contiguous().float(),
            res_signs.contiguous().float(),
            res_scale.contiguous().float(),
        )

    # Fallback: try Triton
    try:
        from .triton_kernels import triton_dequantize_residual
        return triton_dequantize_residual(
            indices, centroids, R, vec_norms, res_signs, res_scale,
        )
    except (ImportError, RuntimeError):
        pass

    # Fallback: PyTorch
    return _pytorch_dequantize_residual(
        indices, centroids, R, vec_norms, res_signs, res_scale,
    )


# ---------------------------------------------------------------------------
# WHT functions
# ---------------------------------------------------------------------------


def cuda_wht_rotate(
    x: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    """Apply forward randomized WHT rotation using CUDA kernel.

    Computes: out = WHT(signs * x) / sqrt(d)

    Same API as triton_wht_rotate from triton_kernels.py.
    Supports d up to 2048 (vs 512 for Triton).

    Args:
        x: Input tensor, shape (batch, d) or (..., d). Must be on CUDA.
        signs: Random sign vector of shape (d,), values in {-1, +1}.

    Returns:
        Rotated tensor, same shape as x.
    """
    _ensure_backend()
    orig_shape = x.shape
    d = x.shape[-1]
    x_flat = x.reshape(-1, d).contiguous().float()

    if _CUDA_WHT is not None:
        result = _CUDA_WHT.wht(x_flat, signs.contiguous().float(), False)
        return result.reshape(orig_shape)

    # Fallback: try Triton (only d<=512)
    if d <= 512:
        try:
            from .triton_kernels import triton_wht_rotate
            return triton_wht_rotate(x, signs)
        except (ImportError, RuntimeError):
            pass

    # Fallback: PyTorch
    return _pytorch_wht(x_flat, signs, inverse=False).reshape(orig_shape)


def cuda_wht_unrotate(
    y: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    """Apply inverse randomized WHT rotation using CUDA kernel.

    Computes: out = signs * WHT(y) / sqrt(d)

    Same API as triton_wht_unrotate from triton_kernels.py.
    Supports d up to 2048 (vs 512 for Triton).

    Args:
        y: Rotated tensor, shape (batch, d) or (..., d). Must be on CUDA.
        signs: Random sign vector of shape (d,), values in {-1, +1}.

    Returns:
        Unrotated tensor, same shape as y.
    """
    _ensure_backend()
    orig_shape = y.shape
    d = y.shape[-1]
    y_flat = y.reshape(-1, d).contiguous().float()

    if _CUDA_WHT is not None:
        result = _CUDA_WHT.wht(y_flat, signs.contiguous().float(), True)
        return result.reshape(orig_shape)

    # Fallback: try Triton (only d<=512)
    if d <= 512:
        try:
            from .triton_kernels import triton_wht_unrotate
            return triton_wht_unrotate(y, signs)
        except (ImportError, RuntimeError):
            pass

    # Fallback: PyTorch
    return _pytorch_wht(y_flat, signs, inverse=True).reshape(orig_shape)


# ---------------------------------------------------------------------------
# PyTorch fallback implementations
# ---------------------------------------------------------------------------


def _pytorch_dequantize(
    indices: torch.Tensor,
    centroids: torch.Tensor,
    R: torch.Tensor,
    vec_norms: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch dequantize (gather + matmul)."""
    y_hat = centroids[indices.long()]
    x_hat = y_hat @ R
    return x_hat * vec_norms.unsqueeze(-1)


def _pytorch_dequantize_residual(
    indices: torch.Tensor,
    centroids: torch.Tensor,
    R: torch.Tensor,
    vec_norms: torch.Tensor,
    res_signs: torch.Tensor,
    res_scale: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch dequantize with residual correction."""
    y_hat = centroids[indices.long()]
    y_corrected = y_hat + res_signs * res_scale.unsqueeze(-1)
    x_hat = y_corrected @ R
    return x_hat * vec_norms.unsqueeze(-1)


def _pytorch_wht(
    x: torch.Tensor,
    signs: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Pure PyTorch WHT via iterative butterfly (O(d log d))."""
    d = x.shape[-1]
    result = x.clone()

    if not inverse:
        result = result * signs

    h = 1
    while h < d:
        xe = result.view(*result.shape[:-1], -1, 2, h)
        a = xe[..., 0, :].clone()
        b = xe[..., 1, :].clone()
        xe[..., 0, :] = a + b
        xe[..., 1, :] = a - b
        h *= 2

    result = result / math.sqrt(d)

    if inverse:
        result = result * signs

    return result


# ---------------------------------------------------------------------------
# CUDATurboQuant: drop-in replacement for TritonTurboQuant
# ---------------------------------------------------------------------------


class CUDATurboQuant:
    """Drop-in replacement for TritonTurboQuant using raw CUDA kernels.

    Provides the same API for dequantize operations. Quantize and inner_product
    still use Triton/PyTorch since those are less critical for the decode path.

    Backend selection (automatic):
        CUDA -> Triton -> PyTorch

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
        from .codebook import LloydMaxCodebook
        from .rotation import generate_rotation_matrix, generate_qjl_matrix

        self.d = d
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.m = qjl_dim if qjl_dim is not None else d
        self.device = torch.device(device)

        # Generate rotation matrix Pi (d x d orthogonal)
        self.R = generate_rotation_matrix(
            d, seed=seed, device="cpu",
        ).contiguous().to(self.device)

        # Generate QJL projection matrix S (m x d)
        self.S = generate_qjl_matrix(
            d, m=self.m, seed=seed + 1, device="cpu",
        ).contiguous().to(self.device)

        # Precompute SR = S @ R^T for fused QJL
        self.SR = (self.S @ self.R.T).contiguous()

        # Solve Lloyd-Max codebook
        codebook = LloydMaxCodebook(d, self.mse_bits)
        self.centroids = codebook.centroids.to(self.device)
        self.boundaries = codebook.boundaries.to(self.device)
        self.n_centroids = codebook.n_levels

        # Pre-load CUDA backend
        _ensure_backend()

        # Report active backend
        if _CUDA_DEQUANTIZE is not None:
            self._backend = "cuda"
        else:
            try:
                from .triton_kernels import triton_dequantize
                self._backend = "triton"
            except (ImportError, RuntimeError):
                self._backend = "pytorch"

        logger.info("CUDATurboQuant: d=%d, bits=%d, backend=%s",
                     d, bits, self._backend)

    @property
    def backend(self) -> str:
        """Return the active dequantize backend name."""
        return self._backend

    def quantize(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compress vectors. Delegates to Triton if available, else PyTorch.

        Same API as TritonTurboQuant.quantize().
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        vec_norm = x.norm(dim=-1, keepdim=True)
        x_normalized = (x / (vec_norm + 1e-8)).contiguous()

        # Try Triton quantize first
        try:
            from .triton_kernels import triton_quantize
            indices, signs, r_norms = triton_quantize(
                x_normalized, self.R, self.boundaries, self.centroids, self.SR,
            )
        except (ImportError, RuntimeError):
            # PyTorch fallback
            y = x_normalized @ self.R.T
            indices = self._pytorch_quantize(y)
            y_hat = self.centroids[indices]
            res_rot = y - y_hat
            r_norms = res_rot.norm(dim=-1)
            proj = res_rot @ self.SR.T
            signs = torch.where(proj >= 0, 1.0, -1.0)

        result = {
            "mse_indices": indices,
            "qjl_signs": signs,
            "residual_norm": r_norms,
            "vec_norm": vec_norm.squeeze(-1),
        }

        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result

    def _pytorch_quantize(self, y: torch.Tensor) -> torch.Tensor:
        """Brute-force boundary search in PyTorch."""
        indices = torch.zeros_like(y, dtype=torch.int32)
        for b_val in self.boundaries:
            indices += (y > b_val).int()
        return indices

    def dequantize(
        self,
        indices: torch.Tensor,
        centroids: torch.Tensor,
        rotation: torch.Tensor,
        vec_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize vectors. Direct function-call API.

        Args:
            indices: (batch, d) int32 codebook indices.
            centroids: (n_centroids,) centroid values.
            rotation: (d, d) rotation matrix.
            vec_norm: (batch,) vector norms.

        Returns:
            (batch, d) dequantized float32 vectors.
        """
        return cuda_dequantize(indices, centroids, rotation, vec_norm)

    def dequantize_mse(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct vectors using MSE path.

        Same API as TritonTurboQuant.dequantize_mse().
        """
        indices = compressed["mse_indices"]
        vec_norms = compressed["vec_norm"]

        squeeze = indices.dim() == 1
        if squeeze:
            indices = indices.unsqueeze(0)
            vec_norms = vec_norms.unsqueeze(0)

        result = cuda_dequantize(indices, self.centroids, self.R, vec_norms)

        if squeeze:
            result = result.squeeze(0)
        return result

    def inner_product(
        self,
        query: torch.Tensor,
        compressed: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Estimate <query, key> using hybrid approach.

        Same API as TritonTurboQuant.inner_product().
        Delegates to Triton/cuBLAS hybrid since the inner product kernel
        is not yet ported to raw CUDA.
        """
        try:
            from .triton_kernels import triton_inner_product
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

            result = triton_inner_product(
                query.contiguous(), self.R, indices, self.centroids, signs,
                self.S, r_norms, vec_norms,
            )

            if squeeze_q and result.dim() > 0 and result.shape[0] == 1:
                result = result.squeeze(0)
            if squeeze_k and result.dim() > 0 and result.shape[-1] == 1:
                result = result.squeeze(-1)

            return result

        except (ImportError, RuntimeError):
            # PyTorch fallback: dequantize then dot
            keys = self.dequantize_mse(compressed)
            if query.dim() == 1:
                query = query.unsqueeze(0)
            if keys.dim() == 1:
                keys = keys.unsqueeze(0)
            return query @ keys.T
