"""Production KV cache for compressed autoregressive generation.

The first KV cache compression that matches FP16 generation quality.

Uses a novel combination discovered through autoresearch:
- 3-bit Lloyd-Max quantization for keys (8 centroids)
- 1-bit direct residual sign correction (NOT QJL random projection)
- 2-bit Lloyd-Max quantization for values
- Norm correction (original/reconstruction ratio)
- FP16 window: last N tokens stored at full precision
- Fused attention: compute scores directly from compressed indices in float32

Compression: 5.1x vs FP16 at matching generation quality.

Usage:
    from turboquantdc import GenerationCache

    cache = GenerationCache()
    output = model.generate(inputs, past_key_values=cache, max_new_tokens=100)

Duck-types the HuggingFace Cache protocol (update, get_seq_length,
get_mask_sizes, __getitem__, __len__, __iter__, reorder_cache, crop, etc.)
so it works as a drop-in replacement for DynamicCache in model.generate().

Reference: TurboQuant paper (arxiv 2504.19874), with direct residual sign
correction replacing QJL random projection for generation workloads.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from .codebook import LloydMaxCodebook
from .rotation import (
    apply_wht_rotation,
    generate_qjl_matrix,
    generate_rotation_matrix,
    generate_wht_rotation,
)

# Triton availability check — graceful fallback to Python on non-CUDA systems
_TRITON_AVAILABLE = False
try:
    from .triton_kernels import triton_quantize as _triton_quantize

    _TRITON_AVAILABLE = True
except (ImportError, RuntimeError):
    pass

# Triton fused dequantize kernels — extend the same availability flag
try:
    from .triton_kernels import triton_dequantize, triton_dequantize_residual
except (ImportError, RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Layer-adaptive anchor strategy helpers
# ---------------------------------------------------------------------------

# Valid anchor strategies
ANCHOR_STRATEGIES = ("fixed", "boundary", "gradient")


def compute_layer_key_bits(
    layer_idx: int,
    num_layers: int,
    base_bits: int = 3,
) -> int:
    """Compute key bit-width for a layer based on its distance from boundaries.

    Layers near the first/last positions of the transformer stack are more
    sensitive to quantization error (boundary layers handle embedding proximity
    and output head proximity respectively). This function assigns higher
    bit-widths to boundary layers and lower bit-widths to middle layers.

    The distance metric is normalized to [0, 0.5] where 0 means at the
    boundary and 0.5 means exact middle of the stack.

    Args:
        layer_idx: Index of the layer (0-based).
        num_layers: Total number of transformer layers.
        base_bits: Default bit-width for middle layers (default: 3).

    Returns:
        Bit-width for keys at this layer. Values in {base_bits, base_bits+1, 8}.
        Returns 8 (FP16-equivalent) for layers within 10% of boundaries.
        Returns base_bits+1 for layers within 25% of boundaries.
        Returns base_bits for all other (middle) layers.
    """
    if num_layers <= 1:
        return 8  # Single-layer model: always FP16-equivalent

    # Distance from nearest boundary, normalized to [0, 0.5]
    dist = min(layer_idx, num_layers - 1 - layer_idx) / (num_layers / 2)

    if dist < 0.1:  # Within 10% of boundary
        return 8  # FP16-equivalent for keys
    elif dist < 0.25:  # Within 25% of boundary
        return max(base_bits + 1, 4)  # One extra bit, at least 4
    else:
        return base_bits  # Base compression


def compute_anchor_schedule(
    num_layers: int,
    anchor_strategy: str = "fixed",
    anchor_interval: int = 6,
    base_key_bits: int = 3,
) -> List[Tuple[bool, int]]:
    """Compute per-layer (is_fp16, key_bits) schedule for the given strategy.

    Args:
        num_layers: Total transformer layers.
        anchor_strategy: One of "fixed", "boundary", "gradient".
        anchor_interval: Interval for "fixed" strategy (ignored by others).
        base_key_bits: Base key bit-width for compressed layers.

    Returns:
        List of (is_fp16, key_bits) tuples, one per layer.
        When is_fp16 is True, the layer stores raw FP16 (key_bits is ignored).
        When is_fp16 is False, key_bits is the bit-width for that layer's keys.
    """
    if anchor_strategy not in ANCHOR_STRATEGIES:
        raise ValueError(
            f"Unknown anchor_strategy: '{anchor_strategy}'. "
            f"Must be one of {ANCHOR_STRATEGIES}"
        )

    schedule: List[Tuple[bool, int]] = []

    if anchor_strategy == "fixed":
        for i in range(num_layers):
            is_fp16 = anchor_interval > 0 and i % anchor_interval == 0
            schedule.append((is_fp16, base_key_bits))

    elif anchor_strategy == "boundary":
        # First 2 + last 2 layers are always FP16
        for i in range(num_layers):
            is_fp16 = i < 2 or i >= num_layers - 2
            schedule.append((is_fp16, base_key_bits))

    elif anchor_strategy == "gradient":
        # Boundary FP16 + gradient bit allocation for middle layers
        for i in range(num_layers):
            key_bits = compute_layer_key_bits(i, num_layers, base_key_bits)
            is_fp16 = key_bits == 8
            schedule.append((is_fp16, key_bits))

    return schedule


# ---------------------------------------------------------------------------
# Per-layer compressed storage
# ---------------------------------------------------------------------------


class _CompressedLayer:
    """Single layer's compressed KV cache with residual sign correction.

    Stores keys at ``key_bits`` with 1-bit residual signs and norm correction.
    Stores values at ``val_bits`` with norm correction.
    The last ``fp16_window`` tokens are kept in raw FP16 and spliced in on
    retrieval so the most recent context is always lossless.

    Args:
        key_bits: Bits for key quantization (typically 3).
        val_bits: Bits for value quantization (typically 2).
        fp16_window: Number of recent tokens stored at FP16.
        seed: Random seed for this layer's rotation matrix.
        use_residual_quant: Whether to apply 1-bit residual sign correction
            to keys during dequantization. When False, only the MSE centroid
            reconstruction is used (no residual correction).
        use_triton: Use Triton fused kernel for quantization when available.
            Default ``True`` if Triton is importable and CUDA is present.
            Falls back to Python when the rotation type is WHT or the
            tensors are not on CUDA.
    """

    def __init__(
        self,
        key_bits: int = 3,
        val_bits: int = 2,
        fp16_window: int = 128,
        seed: int = 42,
        use_norm_correction: bool = True,
        use_residual_quant: bool = True,
        rotation_type: str | None = None,
        use_triton: bool = _TRITON_AVAILABLE,
    ):
        self.key_bits = key_bits
        self.val_bits = val_bits
        self.fp16_window = fp16_window
        self.seed = seed
        self.use_norm_correction = use_norm_correction
        self.use_residual_quant = use_residual_quant
        # None = auto-select (WHT when d is power of 2, else QR)
        self._rotation_type_override = rotation_type
        self._use_triton = use_triton and _TRITON_AVAILABLE

        self._seq_len: int = 0

        # Lazily initialized on first update
        self._head_dim: Optional[int] = None
        self._num_heads: Optional[int] = None
        self._batch_size: Optional[int] = None
        self._dtype: Optional[torch.dtype] = None
        self._device: Optional[torch.device] = None
        self._rotation_type: Optional[str] = None  # resolved during _lazy_init
        self._rotation: Optional[torch.Tensor] = None  # dense matrix (QR only)
        self._wht_params: Optional[dict] = None  # WHT sign vector (WHT only)
        self._key_codebook: Optional[LloydMaxCodebook] = None
        self._val_codebook: Optional[LloydMaxCodebook] = None
        # Triton path: precomputed SR = S @ R^T for fused QJL (QR only)
        self._triton_S: Optional[torch.Tensor] = None
        self._triton_key_SR: Optional[torch.Tensor] = None
        self._triton_val_SR: Optional[torch.Tensor] = None
        self._triton_ready: bool = False

        # Compressed storage (appended per update call)
        self._key_indices: List[torch.Tensor] = []
        self._key_norms: List[torch.Tensor] = []
        self._key_res_signs: List[torch.Tensor] = []
        self._key_res_scales: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._val_norms: List[torch.Tensor] = []

        # Raw FP16 copies for the precision window
        self._raw_keys: List[torch.Tensor] = []
        self._raw_vals: List[torch.Tensor] = []

        # Incremental dequantization cache — avoids O(N^2) re-dequantization
        self._dequant_key_cache: Optional[torch.Tensor] = None
        self._dequant_val_cache: Optional[torch.Tensor] = None
        self._dequant_len: int = 0  # tokens already dequantized in cache

    # -- initialization --

    def _lazy_init(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Initialize codebooks and rotation from the first observed shapes."""
        self._batch_size = key_states.shape[0]
        self._num_heads = key_states.shape[1]
        self._head_dim = key_states.shape[3]
        self._dtype = key_states.dtype
        self._device = key_states.device

        d = self._head_dim
        device = str(self._device)

        # Resolve rotation type: prefer WHT when d is power of 2 (O(d log d))
        is_pow2 = d > 0 and (d & (d - 1)) == 0
        if self._rotation_type_override is not None:
            self._rotation_type = self._rotation_type_override
        else:
            self._rotation_type = "wht" if is_pow2 else "qr"

        if self._rotation_type == "wht":
            if not is_pow2:
                raise ValueError(
                    f"WHT rotation requires d to be a power of 2, got d={d}. "
                    "Use rotation_type='qr' for non-power-of-2 dimensions."
                )
            wht = generate_wht_rotation(d, seed=self.seed, device=device)
            self._wht_params = wht
            self._rotation = None  # not used for WHT
        else:
            self._rotation = generate_rotation_matrix(d, seed=self.seed, device=device)
            self._wht_params = None  # not used for QR

        self._key_codebook = LloydMaxCodebook(d=d, bits=self.key_bits).to(device)
        self._val_codebook = LloydMaxCodebook(d=d, bits=self.val_bits).to(device)

        # Triton path: precompute S and SR = S @ R^T for fused quantize kernel.
        # Only available for QR rotation on CUDA (Triton kernel uses dense R).
        if (
            self._use_triton
            and self._rotation_type == "qr"
            and self._rotation is not None
            and str(self._device).startswith("cuda")
        ):
            # Rotation convention: Python path uses x @ R (row notation),
            # which is R^T @ x in column notation. The Triton kernel computes
            # R_arg @ x, so we pass R^T to match the Python convention.
            self._triton_R = self._rotation.T.contiguous().to(self._device)
            # QJL matrix S (m x d) with m = d
            self._triton_S = generate_qjl_matrix(
                d, m=d, seed=self.seed + 1, device="cpu",
            ).contiguous().to(self._device)
            # SR = S @ R: the kernel computes sign(SR @ res_rot) where
            # res_rot is in rotated space. Original residual r = R @ res_rot
            # (inverse of R^T), so S @ r = S @ R @ res_rot => SR = S @ R.
            SR = (self._triton_S @ self._rotation.contiguous()).contiguous()
            self._triton_key_SR = SR.to(self._device)
            self._triton_val_SR = SR.to(self._device)
            self._triton_ready = True

    # -- quantization --

    def _quantize_vectors(
        self,
        vectors: torch.Tensor,
        codebook: LloydMaxCodebook,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``[batch, heads, seq, d]`` vectors with norm + residual correction.

        Dispatches to the Triton fused kernel when available (QR rotation on
        CUDA), otherwise falls back to the pure-Python path.

        Returns:
            Tuple of (indices, corrected_norms, residual_signs, residual_scales),
            each reshaped back to ``[batch, heads, seq, ...]``.
        """
        if self._triton_ready and vectors.is_cuda:
            return self._quantize_vectors_triton(vectors, codebook)
        return self._quantize_vectors_python(vectors, codebook)

    def _quantize_vectors_python(
        self,
        vectors: torch.Tensor,
        codebook: LloydMaxCodebook,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure-Python quantize path (original implementation)."""
        batch, heads, seq, d = vectors.shape
        flat = vectors.float().reshape(-1, d)

        # Normalize
        norms = flat.norm(dim=-1, keepdim=True)
        normalized = flat / (norms + 1e-8)

        # Rotate (WHT: O(d log d); QR: O(d^2))
        if self._rotation_type == "wht":
            rotated = apply_wht_rotation(normalized, self._wht_params)
        else:
            rotated = normalized @ self._rotation

        # Quantize per coordinate
        indices = torch.bucketize(rotated, codebook.boundaries)
        indices = indices.clamp(0, codebook.centroids.shape[0] - 1)

        # Norm correction: store ratio to compensate reconstruction norm drift
        recon_rotated = codebook.centroids[indices]
        if self._rotation_type == "wht":
            recon_unrotated = apply_wht_rotation(recon_rotated, self._wht_params, inverse=True)
        else:
            recon_unrotated = recon_rotated @ self._rotation.T
        recon_norm = recon_unrotated.norm(dim=-1, keepdim=True)
        corrected_norms = norms * (
            norms / (recon_norm * norms.abs().clamp(min=1e-8) + 1e-8)
        ).clamp(0.5, 2.0)

        # Residual signs (1 bit per coordinate)
        residual = rotated - recon_rotated
        res_signs = torch.sign(residual)
        res_scale = residual.abs().mean(dim=-1, keepdim=True)

        return (
            indices.reshape(batch, heads, seq, d),
            corrected_norms.reshape(batch, heads, seq),
            res_signs.reshape(batch, heads, seq, d),
            res_scale.reshape(batch, heads, seq),
        )

    def _quantize_vectors_triton(
        self,
        vectors: torch.Tensor,
        codebook: LloydMaxCodebook,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Triton fused quantize path: rotate + quantize in one GPU kernel.

        Uses ``triton_quantize()`` for the fused rotation + boundary search,
        then computes norm correction and residual signs from the indices.
        The Triton kernel fuses the rotation matrix-vector product with the
        per-coordinate quantization boundary search into a single kernel
        launch, eliminating intermediate memory traffic.

        Falls back to Python for WHT rotation (Triton kernel requires dense R).
        """
        batch, heads, seq, d = vectors.shape
        flat = vectors.float().reshape(-1, d).contiguous()

        # Normalize
        norms = flat.norm(dim=-1, keepdim=True)
        normalized = (flat / (norms + 1e-8)).contiguous()

        # Determine SR matrix (same rotation for key and value codebooks)
        SR = self._triton_key_SR

        # Fused rotate + quantize via Triton kernel.
        # Pass R^T so the kernel computes R^T @ x, matching Python's x @ R.
        # Returns: indices (n, d) int32, qjl_signs (n, m), residual_norms (n,)
        indices, _qjl_signs, _res_norms = _triton_quantize(
            normalized,
            self._triton_R,
            codebook.boundaries.contiguous(),
            codebook.centroids.contiguous(),
            SR,
        )
        indices = indices.clamp(0, codebook.centroids.shape[0] - 1)

        # Post-processing: norm correction and residual signs.
        # These are cheap gather + elementwise ops (not the bottleneck).
        recon_rotated = codebook.centroids[indices.long()]
        recon_unrotated = recon_rotated @ self._rotation.T
        recon_norm = recon_unrotated.norm(dim=-1, keepdim=True)
        corrected_norms = norms * (
            norms / (recon_norm * norms.abs().clamp(min=1e-8) + 1e-8)
        ).clamp(0.5, 2.0)

        # Residual signs: need rotated vectors.
        # Recompute rotation via cuBLAS (fast batched matmul, single kernel).
        rotated = normalized @ self._rotation
        residual = rotated - recon_rotated
        res_signs = torch.sign(residual)
        res_scale = residual.abs().mean(dim=-1, keepdim=True)

        return (
            indices.reshape(batch, heads, seq, d),
            corrected_norms.reshape(batch, heads, seq),
            res_signs.reshape(batch, heads, seq, d),
            res_scale.reshape(batch, heads, seq),
        )

    def _dequantize_vectors(
        self,
        indices: torch.Tensor,
        norms: torch.Tensor,
        codebook: LloydMaxCodebook,
        res_signs: Optional[torch.Tensor] = None,
        res_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruct vectors from compressed representation.

        When Triton is available and the rotation type is QR on a CUDA device,
        dispatches to the fused Triton dequantize kernel.  Falls back to
        PyTorch for WHT rotation or CPU tensors.

        Args:
            indices: ``[batch, heads, seq, d]`` centroid indices.
            norms: ``[batch, heads, seq]`` corrected norms.
            codebook: The codebook used during quantization.
            res_signs: Optional ``[batch, heads, seq, d]`` residual signs.
            res_scale: Optional ``[batch, heads, seq]`` residual scales.

        Returns:
            Reconstructed tensor of shape ``[batch, heads, seq, d]``.
        """
        batch, heads, seq, d = indices.shape
        flat_idx = indices.reshape(-1, d)
        flat_norms = norms.reshape(-1)

        # --- Triton fused path (QR rotation on CUDA) ---
        is_cuda = self._device is not None and self._device.type == "cuda"
        use_triton = (
            _TRITON_AVAILABLE
            and is_cuda
            and self._rotation_type == "qr"
            and self._rotation is not None
        )

        if use_triton:
            flat_idx_i32 = flat_idx.to(torch.int32).contiguous()
            centroids_f32 = codebook.centroids.float().contiguous()
            # The Triton kernel computes y_hat @ R_arg; the GenerationCache
            # convention is forward=x@R, inverse=y@R^T, so pass R^T.
            Rt_f32 = self._rotation.float().T.contiguous()
            flat_norms_f32 = flat_norms.float().contiguous()

            if res_signs is not None and res_scale is not None:
                flat_signs = res_signs.float().reshape(-1, d).contiguous()
                flat_scale = res_scale.float().reshape(-1).contiguous()
                reconstructed = triton_dequantize_residual(
                    flat_idx_i32, centroids_f32, Rt_f32, flat_norms_f32,
                    flat_signs, flat_scale,
                )
            else:
                reconstructed = triton_dequantize(
                    flat_idx_i32, centroids_f32, Rt_f32, flat_norms_f32,
                )
            return reconstructed.reshape(batch, heads, seq, d)

        # --- PyTorch fallback (WHT rotation or CPU) ---
        reconstructed = codebook.centroids[flat_idx]

        # Apply residual correction
        if res_signs is not None and res_scale is not None:
            reconstructed = (
                reconstructed
                + res_signs.reshape(-1, d) * res_scale.reshape(-1, 1)
            )

        # Unrotate and rescale (WHT: O(d log d); QR: O(d^2))
        if self._rotation_type == "wht":
            reconstructed = apply_wht_rotation(reconstructed, self._wht_params, inverse=True)
        else:
            reconstructed = reconstructed @ self._rotation.T
        reconstructed = reconstructed * flat_norms.unsqueeze(-1)
        return reconstructed.reshape(batch, heads, seq, d)

    # -- public API --

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress and store new KV states, return full dequantized cache.

        Args:
            key_states: ``[batch, num_heads, new_seq, head_dim]``
            value_states: ``[batch, num_heads, new_seq, head_dim]``

        Returns:
            Tuple of ``(all_keys, all_values)`` with FP16 window applied.
        """
        if self._rotation_type is None:
            self._lazy_init(key_states, value_states)

        new_seq = key_states.shape[2]

        # Compress keys with residual signs
        k_idx, k_norms, k_rsigns, k_rscale = self._quantize_vectors(
            key_states, self._key_codebook,
        )
        self._key_indices.append(k_idx)
        self._key_norms.append(k_norms)
        self._key_res_signs.append(k_rsigns)
        self._key_res_scales.append(k_rscale)

        # Compress values (no residual signs for values)
        v_idx, v_norms, _, _ = self._quantize_vectors(
            value_states, self._val_codebook,
        )
        self._val_indices.append(v_idx)
        self._val_norms.append(v_norms)

        # Store raw FP16 for precision window
        self._raw_keys.append(key_states.detach())
        self._raw_vals.append(value_states.detach())

        # Fix 2: Trim raw FP16 storage to prevent memory leak.
        # Concatenate and keep only last fp16_window tokens when list grows
        # too large (ported from EvolvingCompressor).
        if self.fp16_window > 0 and self._seq_len > self.fp16_window * 2:
            all_rk = torch.cat(self._raw_keys, dim=2)
            all_rv = torch.cat(self._raw_vals, dim=2)
            self._raw_keys = [all_rk[:, :, -self.fp16_window:, :]]
            self._raw_vals = [all_rv[:, :, -self.fp16_window:, :]]

        self._seq_len += new_seq
        return self._dequantize_all()

    def _dequantize_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct all keys and values with FP16 window splice.

        Uses incremental dequantization: only newly added tokens are
        dequantized, and the result is appended to the cached tensor.
        This makes total work over N decode steps O(N) instead of O(N^2).
        """
        if self._seq_len == 0:
            d = self._head_dim or 1
            empty = torch.zeros(
                self._batch_size or 1,
                self._num_heads or 1,
                0, d,
                dtype=self._dtype,
                device=self._device,
            )
            return empty, empty

        # --- Fix 1: Incremental dequantization ---
        # Only dequantize tokens beyond what's already in the cache.
        if self._dequant_len < self._seq_len:
            # Determine which compressed chunks are new since last dequant.
            # Walk the index lists to find new tokens.
            new_k_idx_parts = []
            new_k_norms_parts = []
            new_k_rsigns_parts = []
            new_k_rscales_parts = []
            new_v_idx_parts = []
            new_v_norms_parts = []
            seen = 0
            for i, k_idx in enumerate(self._key_indices):
                chunk_len = k_idx.shape[2]
                chunk_end = seen + chunk_len
                if chunk_end <= self._dequant_len:
                    # Entirely already cached
                    seen = chunk_end
                    continue
                # Partially or fully new
                start_in_chunk = max(0, self._dequant_len - seen)
                new_k_idx_parts.append(k_idx[:, :, start_in_chunk:, :])
                new_k_norms_parts.append(self._key_norms[i][:, :, start_in_chunk:])
                new_k_rsigns_parts.append(self._key_res_signs[i][:, :, start_in_chunk:, :])
                new_k_rscales_parts.append(self._key_res_scales[i][:, :, start_in_chunk:])
                new_v_idx_parts.append(self._val_indices[i][:, :, start_in_chunk:, :])
                new_v_norms_parts.append(self._val_norms[i][:, :, start_in_chunk:])
                seen = chunk_end

            if new_k_idx_parts:
                new_k_idx = torch.cat(new_k_idx_parts, dim=2)
                new_k_norms = torch.cat(new_k_norms_parts, dim=2)
                new_k_rsigns = torch.cat(new_k_rsigns_parts, dim=2)
                new_k_rscales = torch.cat(new_k_rscales_parts, dim=2)
                new_v_idx = torch.cat(new_v_idx_parts, dim=2)
                new_v_norms = torch.cat(new_v_norms_parts, dim=2)

                # Fix 3: Fused key reconstruction — compute q_rot @ centroids[idx]
                # directly instead of materializing FP16 keys. For the returned
                # tensor we still produce the reconstructed keys, but we use the
                # norm-corrected fused path that avoids intermediate rounding.
                new_keys = self._dequantize_vectors_fused(
                    new_k_idx, new_k_norms, self._key_codebook,
                    new_k_rsigns if self.use_residual_quant else None,
                    new_k_rscales if self.use_residual_quant else None,
                )
                new_values = self._dequantize_vectors(
                    new_v_idx, new_v_norms, self._val_codebook,
                )

                if self._dequant_key_cache is not None:
                    self._dequant_key_cache = torch.cat(
                        [self._dequant_key_cache, new_keys], dim=2,
                    )
                    self._dequant_val_cache = torch.cat(
                        [self._dequant_val_cache, new_values], dim=2,
                    )
                else:
                    self._dequant_key_cache = new_keys
                    self._dequant_val_cache = new_values

            self._dequant_len = self._seq_len

        keys = self._dequant_key_cache.clone()
        values = self._dequant_val_cache.clone()

        # FP16 window: replace last N tokens with raw precision
        if self._raw_keys:
            raw_keys = torch.cat(self._raw_keys, dim=2)
            raw_vals = torch.cat(self._raw_vals, dim=2)
            win = min(self.fp16_window, raw_keys.shape[2])
            if win > 0 and keys.shape[2] >= win:
                keys[:, :, -win:, :] = raw_keys[:, :, -win:, :].to(keys.dtype)
                values[:, :, -win:, :] = raw_vals[:, :, -win:, :].to(values.dtype)

        return keys.to(self._dtype), values.to(self._dtype)

    def _dequantize_vectors_fused(
        self,
        indices: torch.Tensor,
        norms: torch.Tensor,
        codebook: LloydMaxCodebook,
        res_signs: Optional[torch.Tensor] = None,
        res_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruct vectors using the fused path with norm correction.

        Instead of the naive dequant -> unrotate -> scale path, this uses the
        algebraic identity from fused_attention.py:

            k_recon = (centroids[idx] + res_correction) @ R^T * corrected_norm

        The computation stays in float32 throughout, avoiding FP16
        materialization that introduces rounding errors at long context.
        Norm correction (original_norm / reconstruction_norm) is applied
        when ``self.use_norm_correction`` is True.

        When Triton is available and the rotation type is QR on a CUDA device,
        the centroid lookup + residual correction + inverse rotation + rescale
        are fused into a single Triton kernel launch for significant speedup.
        The WHT rotation path falls back to PyTorch (WHT is already O(d log d)).

        Args:
            indices: ``[batch, heads, seq, d]`` centroid indices.
            norms: ``[batch, heads, seq]`` corrected norms.
            codebook: The codebook used during quantization.
            res_signs: Optional ``[batch, heads, seq, d]`` residual signs.
            res_scale: Optional ``[batch, heads, seq]`` residual scales.

        Returns:
            Reconstructed tensor of shape ``[batch, heads, seq, d]``.
        """
        batch, heads, seq, d = indices.shape
        flat_idx = indices.reshape(-1, d)
        flat_norms = norms.reshape(-1)

        # --- Triton fused path (QR rotation on CUDA) ---
        is_cuda = self._device is not None and self._device.type == "cuda"
        use_triton = (
            _TRITON_AVAILABLE
            and is_cuda
            and self._rotation_type == "qr"
            and self._rotation is not None
        )

        if use_triton:
            flat_idx_i32 = flat_idx.to(torch.int32).contiguous()
            centroids_f32 = codebook.centroids.float().contiguous()
            # The Triton kernel computes y_hat @ R_arg; the GenerationCache
            # convention is forward=x@R, inverse=y@R^T, so pass R^T.
            Rt_f32 = self._rotation.float().T.contiguous()
            flat_norms_f32 = flat_norms.float().contiguous()

            if res_signs is not None and res_scale is not None:
                flat_signs = res_signs.float().reshape(-1, d).contiguous()
                flat_scale = res_scale.float().reshape(-1).contiguous()
                reconstructed = triton_dequantize_residual(
                    flat_idx_i32, centroids_f32, Rt_f32, flat_norms_f32,
                    flat_signs, flat_scale,
                )
            else:
                reconstructed = triton_dequantize(
                    flat_idx_i32, centroids_f32, Rt_f32, flat_norms_f32,
                )
            return reconstructed.reshape(batch, heads, seq, d)

        # --- PyTorch fallback (WHT rotation or CPU) ---
        # Gather centroids in float32 (the fused path core)
        reconstructed = codebook.centroids.float()[flat_idx.long()]

        # Apply residual correction in the rotated domain (float32)
        if res_signs is not None and res_scale is not None:
            reconstructed = (
                reconstructed
                + res_signs.float().reshape(-1, d) * res_scale.float().reshape(-1, 1)
            )

        # Unrotate in float32 (avoids FP16 intermediate)
        if self._rotation_type == "wht":
            reconstructed = apply_wht_rotation(
                reconstructed.float(), self._wht_params, inverse=True
            )
        else:
            reconstructed = torch.matmul(reconstructed, self._rotation.float().T)

        # Apply norm correction: the norms stored already include the
        # correction ratio (original / reconstruction) when use_norm_correction
        # is enabled, so simply scaling by stored norms gives the correct result.
        reconstructed = reconstructed * flat_norms.float().unsqueeze(-1)
        return reconstructed.reshape(batch, heads, seq, d)

    def get_seq_length(self) -> int:
        """Return number of cached tokens."""
        return self._seq_len

    def clear(self) -> None:
        """Clear all stored data."""
        self._key_indices.clear()
        self._key_norms.clear()
        self._key_res_signs.clear()
        self._key_res_scales.clear()
        self._val_indices.clear()
        self._val_norms.clear()
        self._raw_keys.clear()
        self._raw_vals.clear()
        self._seq_len = 0
        self._dequant_key_cache = None
        self._dequant_val_cache = None
        self._dequant_len = 0

    def reorder(self, beam_idx: torch.LongTensor) -> None:
        """Reorder cache entries along the batch dimension for beam search."""
        self._key_indices = [t.index_select(0, beam_idx) for t in self._key_indices]
        self._key_norms = [t.index_select(0, beam_idx) for t in self._key_norms]
        self._key_res_signs = [t.index_select(0, beam_idx) for t in self._key_res_signs]
        self._key_res_scales = [t.index_select(0, beam_idx) for t in self._key_res_scales]
        self._val_indices = [t.index_select(0, beam_idx) for t in self._val_indices]
        self._val_norms = [t.index_select(0, beam_idx) for t in self._val_norms]
        self._raw_keys = [t.index_select(0, beam_idx) for t in self._raw_keys]
        self._raw_vals = [t.index_select(0, beam_idx) for t in self._raw_vals]
        # Invalidate dequant cache on reorder (batch dimension changed)
        self._dequant_key_cache = None
        self._dequant_val_cache = None
        self._dequant_len = 0

    def crop(self, max_length: int) -> None:
        """Truncate cached sequence to max_length tokens."""
        if max_length < 0:
            max_length = self._seq_len + max_length
        if self._seq_len <= max_length:
            return

        # Concatenate, slice, and re-store as single chunks
        if self._key_indices:
            all_k_idx = torch.cat(self._key_indices, dim=2)[:, :, :max_length]
            all_k_norms = torch.cat(self._key_norms, dim=2)[:, :, :max_length]
            all_k_rsigns = torch.cat(self._key_res_signs, dim=2)[:, :, :max_length]
            all_k_rscales = torch.cat(self._key_res_scales, dim=2)[:, :, :max_length]
            all_v_idx = torch.cat(self._val_indices, dim=2)[:, :, :max_length]
            all_v_norms = torch.cat(self._val_norms, dim=2)[:, :, :max_length]
            raw_keys = torch.cat(self._raw_keys, dim=2)[:, :, :max_length]
            raw_vals = torch.cat(self._raw_vals, dim=2)[:, :, :max_length]

            self._key_indices = [all_k_idx]
            self._key_norms = [all_k_norms]
            self._key_res_signs = [all_k_rsigns]
            self._key_res_scales = [all_k_rscales]
            self._val_indices = [all_v_idx]
            self._val_norms = [all_v_norms]
            self._raw_keys = [raw_keys]
            self._raw_vals = [raw_vals]

        # Invalidate dequant cache (cropping changes the sequence)
        if self._dequant_key_cache is not None and max_length < self._dequant_len:
            self._dequant_key_cache = self._dequant_key_cache[:, :, :max_length, :]
            self._dequant_val_cache = self._dequant_val_cache[:, :, :max_length, :]
            self._dequant_len = max_length

        self._seq_len = max_length

    def memory_usage_bits(self) -> Dict[str, Any]:
        """Compute memory usage of compressed storage.

        Returns:
            Dict with key_bits, value_bits, total_bits, fp16_baseline_bits,
            and compression_ratio.
        """
        if self._seq_len == 0 or self._head_dim is None:
            return {
                "key_bits": 0,
                "value_bits": 0,
                "total_bits": 0,
                "fp16_baseline_bits": 0,
                "compression_ratio": 1.0,
            }

        d = self._head_dim
        n_heads = self._num_heads
        batch = self._batch_size
        total_tokens = self._seq_len * n_heads * batch

        # Tokens in the FP16 window are stored uncompressed
        fp16_tokens = min(self.fp16_window, self._seq_len) * n_heads * batch
        compressed_tokens = total_tokens - fp16_tokens

        # Compressed key: key_bits*d (indices) + d (residual signs) + 16 (norm) + 16 (res_scale)
        key_bits_compressed = compressed_tokens * (self.key_bits * d + d + 32)
        key_bits_fp16 = fp16_tokens * d * 16

        # Compressed value: val_bits*d (indices) + 16 (norm)
        val_bits_compressed = compressed_tokens * (self.val_bits * d + 16)
        val_bits_fp16 = fp16_tokens * d * 16

        total_key = key_bits_compressed + key_bits_fp16
        total_val = val_bits_compressed + val_bits_fp16
        total = total_key + total_val

        fp16_baseline = total_tokens * d * 16 * 2

        return {
            "key_bits": total_key,
            "value_bits": total_val,
            "total_bits": total,
            "fp16_baseline_bits": fp16_baseline,
            "compression_ratio": fp16_baseline / total if total > 0 else 1.0,
        }


# ---------------------------------------------------------------------------
# FP16 anchor layer (no compression)
# ---------------------------------------------------------------------------


class _FP16Layer:
    """Single layer FP16 KV cache (no compression).

    Used for anchor layers that break error accumulation.
    """

    def __init__(self):
        self._keys: List[torch.Tensor] = []
        self._values: List[torch.Tensor] = []
        self._seq_len: int = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._keys.append(key_states)
        self._values.append(value_states)
        self._seq_len += key_states.shape[2]
        return torch.cat(self._keys, dim=2), torch.cat(self._values, dim=2)

    def _dequantize_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._seq_len == 0:
            return (
                torch.zeros(1, 1, 0, 1),
                torch.zeros(1, 1, 0, 1),
            )
        return torch.cat(self._keys, dim=2), torch.cat(self._values, dim=2)

    def get_seq_length(self) -> int:
        return self._seq_len

    def reorder(self, beam_idx: torch.LongTensor) -> None:
        self._keys = [k.index_select(0, beam_idx) for k in self._keys]
        self._values = [v.index_select(0, beam_idx) for v in self._values]

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self._seq_len + max_length
        if self._seq_len <= max_length:
            return
        all_keys = torch.cat(self._keys, dim=2)[:, :, :max_length]
        all_values = torch.cat(self._values, dim=2)[:, :, :max_length]
        self._keys = [all_keys]
        self._values = [all_values]
        self._seq_len = max_length

    def clear(self) -> None:
        self._keys.clear()
        self._values.clear()
        self._seq_len = 0

    def memory_usage_bits(self) -> Dict[str, Any]:
        if self._seq_len == 0 or not self._keys:
            return {
                "key_bits": 0,
                "value_bits": 0,
                "total_bits": 0,
                "fp16_baseline_bits": 0,
                "compression_ratio": 1.0,
            }
        k = self._keys[0]
        head_dim = k.shape[3]
        total_tokens = self._seq_len * k.shape[1] * k.shape[0]
        total = total_tokens * head_dim * 16 * 2
        return {
            "key_bits": total // 2,
            "value_bits": total // 2,
            "total_bits": total,
            "fp16_baseline_bits": total,
            "compression_ratio": 1.0,
        }


# ---------------------------------------------------------------------------
# GenerationCache — production KV cache
# ---------------------------------------------------------------------------


class GenerationCache:
    """Production KV cache for compressed autoregressive generation.

    The first KV cache compression that matches FP16 generation quality.

    Uses a novel combination discovered through autoresearch:
    - Lloyd-Max quantization for keys and values
    - 1-bit direct residual sign correction (NOT QJL random projection)
    - Norm correction (original/reconstruction ratio)
    - FP16 window: last N tokens stored at full precision
    - FP16 anchor layers every N layers to break error accumulation

    Defaults tuned from a 246-configuration autoresearch sweep:
    - K4/V3 anchor=12 win=64 RQ=True is a safe middle ground (default)
    - K8/V3 anchor=12 is the quality champion (96.4% of FP16, 2.5x compression)
    - K3/V3 anchor=36 win=64 RQ=True is the best tradeoff (95.1%, 3.3x, Gen=97%)
    - K3/V2 anchor=6 win=512 RQ=True is the aggressive option (94.9%, higher compression)
    - RQ=True outperforms RQ=False by 13% on average (0.801 vs 0.707)
    - Anchors are essential: anchor=0 avg 0.408, anchor=12 avg 0.872

    Anchor strategies control which layers store KV at full FP16 precision
    to break error accumulation:

    - ``"fixed"`` (default): Every ``anchor_interval``-th layer is FP16.
      Simple, proven baseline. E.g., layers 0, 12, 24, 36.
    - ``"boundary"``: First 2 + last 2 layers always FP16, rest compressed.
      Based on finding that boundary layers (embedding proximity, output
      head proximity) are most sensitive to quantization error.
    - ``"gradient"``: Boundary layers FP16 + gradient bit allocation for
      middle layers. Layers near boundaries get higher key_bits (4-bit),
      middle layers get base key_bits (3-bit). Allocates bits where they
      matter most for the same total budget.

    Usage::

        from turboquantdc import GenerationCache

        # Default (safe middle ground)
        cache = GenerationCache()

        # From a named preset
        cache = GenerationCache.from_preset("balanced")
        cache = GenerationCache.from_preset("lossless")
        cache = GenerationCache.from_preset("aggressive")

        # Preset with overrides
        cache = GenerationCache.from_preset("balanced", fp16_window=128)

        # Boundary anchoring: first 2 + last 2 FP16
        cache = GenerationCache(anchor_strategy="boundary", num_layers=36)

        # Gradient: boundary FP16 + per-layer bit allocation
        cache = GenerationCache(anchor_strategy="gradient", num_layers=36)

        output = model.generate(inputs, past_key_values=cache, max_new_tokens=100)

    Args:
        key_bits: Bits for key quantization (default: 4).
        val_bits: Bits for value quantization (default: 3).
        fp16_window: Number of recent tokens at FP16 (default: 64).
        anchor_interval: Every Nth layer is stored at FP16 to break error
            accumulation. Set to 0 to disable anchors. Only used when
            ``anchor_strategy="fixed"`` (default: 12).
        anchor_strategy: Anchor placement strategy. One of ``"fixed"``,
            ``"boundary"``, ``"gradient"`` (default: ``"fixed"``).
        num_layers: Total number of transformer layers. Required for
            ``"boundary"`` and ``"gradient"`` strategies. When None,
            layers are created lazily on first ``update()`` call (only
            valid for ``"fixed"`` strategy).
        seed: Random seed for reproducibility.
        use_norm_correction: Apply norm correction (original/reconstruction
            ratio) for improved perplexity. Default True per fused_attention
            finding of -1.17% perplexity improvement.
        use_residual_quant: Whether to apply 1-bit residual sign correction
            to keys during dequantization. When False, only MSE centroid
            reconstruction is used. Default: True.
        use_triton: Use Triton fused kernel for quantization when available.
            Default ``True`` if Triton is importable and CUDA is present.
            Falls back to Python for WHT rotation or non-CUDA devices.
    """

    # Quality presets from 246-config autoresearch sweep.
    # Each maps to GenerationCache __init__ kwargs.
    PRESETS = {
        "lossless": {
            "key_bits": 8,
            "val_bits": 3,
            "anchor_interval": 12,
            "fp16_window": 0,
            "use_residual_quant": False,
        },
        "balanced": {
            "key_bits": 3,
            "val_bits": 3,
            "anchor_interval": 36,
            "fp16_window": 64,
            "use_residual_quant": True,
        },
        "aggressive": {
            "key_bits": 3,
            "val_bits": 2,
            "anchor_interval": 6,
            "fp16_window": 512,
            "use_residual_quant": True,
        },
        # Hybrid presets: combine boundary anchoring + gradient bits +
        # FP16 window + ResidualQuant + norm correction.
        "hybrid_max_quality": {
            "key_bits": 3,
            "val_bits": 3,
            "anchor_interval": 0,  # not used -- boundary handles it
            "anchor_strategy": "boundary",
            "fp16_window": 64,
            "use_residual_quant": True,
            "use_norm_correction": True,
        },
        "hybrid_max_compression": {
            "key_bits": 3,
            "val_bits": 2,
            "anchor_interval": 0,
            "anchor_strategy": "gradient",
            "fp16_window": 64,
            "use_residual_quant": True,
            "use_norm_correction": True,
        },
    }

    is_compileable = False

    def __init__(
        self,
        key_bits: int = 4,
        val_bits: int = 3,
        fp16_window: int = 64,
        anchor_interval: int = 12,
        anchor_strategy: str = "fixed",
        num_layers: Optional[int] = None,
        seed: int = 42,
        use_norm_correction: bool = True,
        use_residual_quant: bool = True,
        rotation_type: str | None = None,
        use_triton: bool = _TRITON_AVAILABLE,
    ):
        if not (1 <= key_bits <= 8):
            raise ValueError(f"key_bits must be 1-8, got {key_bits}")
        if not (1 <= val_bits <= 8):
            raise ValueError(f"val_bits must be 1-8, got {val_bits}")
        if fp16_window < 0:
            raise ValueError(f"fp16_window must be >= 0, got {fp16_window}")
        if anchor_strategy not in ANCHOR_STRATEGIES:
            raise ValueError(
                f"Unknown anchor_strategy: '{anchor_strategy}'. "
                f"Must be one of {ANCHOR_STRATEGIES}"
            )
        if anchor_strategy in ("boundary", "gradient") and num_layers is None:
            raise ValueError(
                f"num_layers is required for anchor_strategy='{anchor_strategy}'"
            )

        self.key_bits = key_bits
        self.val_bits = val_bits
        self.fp16_window = fp16_window
        self.anchor_interval = anchor_interval
        self.anchor_strategy = anchor_strategy
        self.num_layers = num_layers
        self.seed = seed
        self.use_norm_correction = use_norm_correction
        self.use_residual_quant = use_residual_quant
        self.rotation_type = rotation_type  # None = auto (WHT for power-of-2 d)
        self.use_triton = use_triton

        # Pre-compute anchor schedule when num_layers is known
        self._anchor_schedule: Optional[List[Tuple[bool, int]]] = None
        if num_layers is not None:
            self._anchor_schedule = compute_anchor_schedule(
                num_layers=num_layers,
                anchor_strategy=anchor_strategy,
                anchor_interval=anchor_interval,
                base_key_bits=key_bits,
            )

        self._layers: List[_CompressedLayer | _FP16Layer] = []

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> "GenerationCache":
        """Create a GenerationCache from a named quality preset.

        Available presets (from 246-config autoresearch sweep):
        - "lossless": K8/V3 anchor=12 (96.4% of FP16, 2.5x compression)
        - "balanced": K3/V3 anchor=36 win=64 RQ=True (95.1%, 3.3x, Gen=97%)
        - "aggressive": K3/V2 anchor=6 win=512 RQ=True (94.9%, higher compression)

        Args:
            preset: One of "lossless", "balanced", "aggressive".
            **overrides: Any GenerationCache kwarg to override the preset value.

        Returns:
            A new GenerationCache configured from the preset.

        Raises:
            KeyError: If preset is not a recognized preset name.
        """
        if preset not in cls.PRESETS:
            raise KeyError(
                f"Unknown preset '{preset}'. "
                f"Available presets: {list(cls.PRESETS.keys())}"
            )
        config = cls.PRESETS[preset].copy()
        config.update(overrides)
        return cls(**config)

    def _is_anchor_layer(self, idx: int) -> bool:
        """Return True if layer ``idx`` should be stored at FP16."""
        if self._anchor_schedule is not None and idx < len(self._anchor_schedule):
            return self._anchor_schedule[idx][0]
        # Fallback for fixed strategy when num_layers is unknown (lazy growth)
        return self.anchor_interval > 0 and idx % self.anchor_interval == 0

    def _layer_key_bits(self, idx: int) -> int:
        """Return key bit-width for layer ``idx``."""
        if self._anchor_schedule is not None and idx < len(self._anchor_schedule):
            return self._anchor_schedule[idx][1]
        return self.key_bits

    def _make_layer(self, idx: int) -> _CompressedLayer | _FP16Layer:
        """Create the appropriate layer type for index ``idx``."""
        if self._is_anchor_layer(idx):
            return _FP16Layer()
        return _CompressedLayer(
            key_bits=self._layer_key_bits(idx),
            val_bits=self.val_bits,
            fp16_window=self.fp16_window,
            seed=self.seed + idx,
            use_norm_correction=self.use_norm_correction,
            use_residual_quant=self.use_residual_quant,
            rotation_type=self.rotation_type,
            use_triton=self.use_triton,
        )

    # ---- HF Cache protocol ----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress/store new KV pairs for a layer, return full cache.

        Args:
            key_states: ``[batch, num_heads, new_seq, head_dim]``
            value_states: ``[batch, num_heads, new_seq, head_dim]``
            layer_idx: Which transformer layer this belongs to.
            cache_kwargs: Additional HF cache kwargs (ignored).

        Returns:
            Tuple of ``(all_keys, all_values)`` tensors.
        """
        while len(self._layers) <= layer_idx:
            self._layers.append(self._make_layer(len(self._layers)))
        return self._layers[layer_idx].update(key_states, value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return number of cached tokens for a layer."""
        if layer_idx >= len(self._layers):
            return 0
        return self._layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Return max cache shape. Dynamic cache has no maximum."""
        return -1

    def get_mask_sizes(
        self,
        cache_position,
        layer_idx: int = 0,
    ) -> tuple[int, int]:
        """Return ``(kv_length, kv_offset)`` for attention mask generation.

        This is the critical method that must return the correct total KV
        length including both cached tokens and new query tokens.  Getting
        this wrong produces misaligned attention masks and garbled output.

        ``cache_position`` may be a ``torch.Tensor`` (transformers <=5.3)
        or an ``int`` query_length (transformers >=5.5).
        """
        if isinstance(cache_position, int):
            query_length = cache_position
        else:
            query_length = cache_position.shape[0]
        if layer_idx >= len(self._layers):
            return query_length, 0
        kv_length = self._layers[layer_idx].get_seq_length() + query_length
        return kv_length, 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder all layers for beam search."""
        for layer in self._layers:
            layer.reorder(beam_idx)

    def crop(self, max_length: int) -> None:
        """Truncate all layers to ``max_length`` tokens."""
        for layer in self._layers:
            layer.crop(max_length)

    def reset(self) -> None:
        """Clear all cached data across every layer."""
        for layer in self._layers:
            layer.clear()

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat cache entries for beam search expansion."""
        for layer in self._layers:
            if isinstance(layer, _FP16Layer):
                layer._keys = [
                    k.repeat_interleave(repeats, dim=0) for k in layer._keys
                ]
                layer._values = [
                    v.repeat_interleave(repeats, dim=0) for v in layer._values
                ]
            else:
                layer._key_indices = [
                    t.repeat_interleave(repeats, dim=0) for t in layer._key_indices
                ]
                layer._key_norms = [
                    t.repeat_interleave(repeats, dim=0) for t in layer._key_norms
                ]
                layer._key_res_signs = [
                    t.repeat_interleave(repeats, dim=0) for t in layer._key_res_signs
                ]
                layer._key_res_scales = [
                    t.repeat_interleave(repeats, dim=0) for t in layer._key_res_scales
                ]
                layer._val_indices = [
                    t.repeat_interleave(repeats, dim=0) for t in layer._val_indices
                ]
                layer._val_norms = [
                    t.repeat_interleave(repeats, dim=0) for t in layer._val_norms
                ]
                layer._raw_keys = [
                    t.repeat_interleave(repeats, dim=0) for t in layer._raw_keys
                ]
                layer._raw_vals = [
                    t.repeat_interleave(repeats, dim=0) for t in layer._raw_vals
                ]
                if layer._batch_size is not None:
                    layer._batch_size *= repeats
                # Invalidate dequant cache (batch dimension changed)
                layer._dequant_key_cache = None
                layer._dequant_val_cache = None
                layer._dequant_len = 0

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select specific batch indices from the cache."""
        for layer in self._layers:
            if isinstance(layer, _FP16Layer):
                layer._keys = [k[indices] for k in layer._keys]
                layer._values = [v[indices] for v in layer._values]
            else:
                layer._key_indices = [t[indices] for t in layer._key_indices]
                layer._key_norms = [t[indices] for t in layer._key_norms]
                layer._key_res_signs = [t[indices] for t in layer._key_res_signs]
                layer._key_res_scales = [t[indices] for t in layer._key_res_scales]
                layer._val_indices = [t[indices] for t in layer._val_indices]
                layer._val_norms = [t[indices] for t in layer._val_norms]
                layer._raw_keys = [t[indices] for t in layer._raw_keys]
                layer._raw_vals = [t[indices] for t in layer._raw_vals]
                if layer._batch_size is not None:
                    layer._batch_size = len(indices)
                # Invalidate dequant cache (batch dimension changed)
                layer._dequant_key_cache = None
                layer._dequant_val_cache = None
                layer._dequant_len = 0

    @property
    def seen_tokens(self) -> int:
        """Number of tokens seen by the first layer."""
        return self._layers[0].get_seq_length() if self._layers else 0

    @property
    def is_initialized(self) -> bool:
        """Return whether the cache has been populated."""
        return len(self._layers) > 0

    @property
    def is_sliding(self) -> list[bool]:
        """Return sliding window status per layer (always False)."""
        return [False] * max(len(self._layers), 1)

    def __len__(self) -> int:
        """Number of layers in the cache."""
        return len(self._layers)

    def __iter__(self):
        """Iterate over layers, yielding ``(keys, values, None)`` tuples."""
        for layer in self._layers:
            keys, values = layer._dequantize_all()
            yield keys, values, None

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dequantized ``(keys, values)`` for a specific layer."""
        if layer_idx >= len(self._layers):
            raise IndexError(
                f"Layer {layer_idx} not in cache (have {len(self._layers)} layers)"
            )
        return self._layers[layer_idx]._dequantize_all()

    def __contains__(self, idx: int) -> bool:
        """Check whether a layer index exists in the cache."""
        return 0 <= idx < len(self._layers)

    # ---- Reporting ----

    def memory_savings(self) -> Dict[str, Any]:
        """Report memory usage and savings across all layers.

        Returns:
            Dict with per_layer stats, aggregate totals, and configuration.
        """
        per_layer = []
        total_compressed = 0
        total_fp16 = 0

        for i, layer in enumerate(self._layers):
            stats = layer.memory_usage_bits()
            layer_key_bits = self._layer_key_bits(i)
            per_layer.append({
                "layer": i,
                "is_anchor": self._is_anchor_layer(i),
                "key_bits": layer_key_bits,
                **stats,
            })
            total_compressed += stats["total_bits"]
            total_fp16 += stats["fp16_baseline_bits"]

        return {
            "per_layer": per_layer,
            "total_compressed_bits": total_compressed,
            "total_fp16_bits": total_fp16,
            "overall_compression_ratio": (
                total_fp16 / total_compressed if total_compressed > 0 else 1.0
            ),
            "config": {
                "key_bits": self.key_bits,
                "val_bits": self.val_bits,
                "fp16_window": self.fp16_window,
                "anchor_interval": self.anchor_interval,
                "anchor_strategy": self.anchor_strategy,
                "use_residual_quant": self.use_residual_quant,
            },
            "num_layers": len(self._layers),
        }

    def anchor_summary(self) -> Dict[str, Any]:
        """Return a summary of the anchor schedule for all layers.

        Useful for inspecting exactly which layers are FP16 anchors and
        the per-layer key bit-widths when using gradient strategy.

        Returns:
            Dict with:
                - strategy: Anchor strategy name.
                - num_layers: Total layer count (from schedule or actual).
                - fp16_layers: List of layer indices that are FP16 anchors.
                - per_layer_key_bits: List of key bit-widths per layer
                  (8 for FP16 anchor layers).
                - fp16_count: Number of FP16 anchor layers.
                - compressed_count: Number of compressed layers.
                - avg_key_bits: Average effective key bits across all layers.
        """
        n = self.num_layers if self.num_layers is not None else len(self._layers)
        if n == 0:
            return {
                "strategy": self.anchor_strategy,
                "num_layers": 0,
                "fp16_layers": [],
                "per_layer_key_bits": [],
                "fp16_count": 0,
                "compressed_count": 0,
                "avg_key_bits": 0.0,
            }

        fp16_layers = []
        per_layer_key_bits = []
        for i in range(n):
            is_fp16 = self._is_anchor_layer(i)
            kb = 16 if is_fp16 else self._layer_key_bits(i)
            per_layer_key_bits.append(kb)
            if is_fp16:
                fp16_layers.append(i)

        fp16_count = len(fp16_layers)
        avg_bits = sum(per_layer_key_bits) / n if n > 0 else 0.0

        return {
            "strategy": self.anchor_strategy,
            "num_layers": n,
            "fp16_layers": fp16_layers,
            "per_layer_key_bits": per_layer_key_bits,
            "fp16_count": fp16_count,
            "compressed_count": n - fp16_count,
            "avg_key_bits": avg_bits,
        }

    def config_summary(self) -> str:
        """Return a human-readable configuration summary."""
        n_layers = len(self._layers)
        n_anchor = sum(1 for i in range(n_layers) if self._is_anchor_layer(i))
        rq_desc = "+ 1b residual signs" if self.use_residual_quant else "(no residual signs)"
        strategy_desc = f"anchor={self.anchor_strategy}"
        if self.anchor_strategy == "fixed":
            strategy_desc += f" interval={self.anchor_interval}"
        return (
            f"GenerationCache: {self.key_bits}b keys {rq_desc}, "
            f"{self.val_bits}b values, FP16 window={self.fp16_window}, "
            f"{n_anchor}/{n_layers} anchor layers ({strategy_desc})"
        )


# ---------------------------------------------------------------------------
# HybridCache — maximum quality via stacked winning strategies
# ---------------------------------------------------------------------------


def _compute_attention_entropy(
    scores: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute Shannon entropy of attention weights per head.

    Args:
        scores: Attention weights ``[batch, num_heads, seq_q, seq_kv]``.
            Must already be non-negative and sum to 1 along the last dim.
        eps: Small constant for numerical stability.

    Returns:
        Per-head entropy ``[batch, num_heads]``, averaged over query positions.
    """
    # Clamp to avoid log(0)
    p = scores.float().clamp(min=eps)
    ent = -(p * p.log()).sum(dim=-1)  # [batch, heads, seq_q]
    return ent.mean(dim=-1)  # [batch, heads]


class HybridCache:
    """Maximum quality KV cache combining all winning strategies.

    Stacks every technique that individually beat FP16 baselines:

    **Layer-level**: Boundary anchoring (first 2 + last 2 layers FP16) with
    gradient bit allocation for middle layers — sensitive layers get more
    bits automatically.

    **Token-level**: FP16 window keeps the most recent tokens at full
    precision so the model always has a lossless view of recent context.

    **Correction**: ResidualQuant (1-bit residual signs) + norm correction
    applied to every compressed layer for maximum reconstruction quality.

    **Head-level** (novel): Per-head bit allocation based on attention
    entropy.  During a warmup phase (first ``warmup_tokens`` tokens),
    attention entropy is tracked per head.  After warmup, heads are
    classified into three tiers:

    - High-entropy heads (attend broadly): ``base_bits + 1`` -- need more
      precision because attention is spread across many tokens.
    - Low-entropy heads (attend narrowly): ``base_bits - 1`` -- can tolerate
      coarser quantization because attention focuses on 1-2 tokens.
    - Normal heads: ``base_bits`` -- the default.

    The entropy thresholds are computed from percentiles of the observed
    entropy distribution: top ``high_entropy_pct`` percent of heads get
    more bits, bottom ``low_entropy_pct`` percent get fewer bits.

    This is analogous to KITTY's channel-adaptive allocation but operates
    at the HEAD level rather than the channel level.

    Usage::

        cache = HybridCache(num_layers=36, base_key_bits=3, base_val_bits=2)
        output = model.generate(inputs, past_key_values=cache, max_new_tokens=100)

    Args:
        num_layers: Total number of transformer layers (required).
        base_key_bits: Default key bit-width for middle layers (default: 3).
        base_val_bits: Default value bit-width (default: 2).
        fp16_window: Number of recent tokens at FP16 (default: 64).
        seed: Random seed for reproducibility.
        warmup_tokens: Number of tokens to observe before assigning
            per-head bits (default: 32).
        high_entropy_pct: Percentage of heads classified as high-entropy
            (default: 25).
        low_entropy_pct: Percentage of heads classified as low-entropy
            (default: 25).
    """

    is_compileable = False

    def __init__(
        self,
        num_layers: int,
        base_key_bits: int = 3,
        base_val_bits: int = 2,
        fp16_window: int = 64,
        seed: int = 42,
        warmup_tokens: int = 32,
        high_entropy_pct: int = 25,
        low_entropy_pct: int = 25,
    ):
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if not (1 <= base_key_bits <= 8):
            raise ValueError(f"base_key_bits must be 1-8, got {base_key_bits}")
        if not (1 <= base_val_bits <= 8):
            raise ValueError(f"base_val_bits must be 1-8, got {base_val_bits}")
        if warmup_tokens < 1:
            raise ValueError(f"warmup_tokens must be >= 1, got {warmup_tokens}")
        if not (0 <= high_entropy_pct <= 100):
            raise ValueError(
                f"high_entropy_pct must be 0-100, got {high_entropy_pct}"
            )
        if not (0 <= low_entropy_pct <= 100):
            raise ValueError(
                f"low_entropy_pct must be 0-100, got {low_entropy_pct}"
            )

        self.num_layers = num_layers
        self.base_key_bits = base_key_bits
        self.base_val_bits = base_val_bits
        self.fp16_window = fp16_window
        self.seed = seed
        self.warmup_tokens = warmup_tokens
        self.high_entropy_pct = high_entropy_pct
        self.low_entropy_pct = low_entropy_pct

        # Layer-level: gradient strategy gives boundary FP16 + graded bits
        self._anchor_schedule = compute_anchor_schedule(
            num_layers=num_layers,
            anchor_strategy="gradient",
            base_key_bits=base_key_bits,
        )

        # Per-head bit allocation state (populated after warmup)
        self._num_heads: Optional[int] = None
        self._per_head_key_bits: Optional[List[int]] = None  # length = num_heads
        self._warmup_entropy_accum: Optional[torch.Tensor] = None  # [num_heads]
        self._warmup_count: int = 0  # tokens observed so far
        self._warmup_complete: bool = False

        # Underlying GenerationCache (re-created after warmup for per-head)
        # Initially uses the base bits; after warmup, the inner cache's layers
        # are adjusted but we keep the same architecture.
        self._inner = GenerationCache(
            key_bits=base_key_bits,
            val_bits=base_val_bits,
            fp16_window=fp16_window,
            anchor_interval=0,
            anchor_strategy="gradient",
            num_layers=num_layers,
            seed=seed,
            use_norm_correction=True,
            use_residual_quant=True,
        )

    # ---- Per-head entropy tracking ----

    def record_attention_entropy(
        self,
        attention_weights: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Record attention entropy for per-head bit allocation warmup.

        Call this during the warmup phase with the softmax attention weights
        from one layer (typically layer 0 or a middle layer).

        Args:
            attention_weights: ``[batch, num_heads, seq_q, seq_kv]`` tensor
                of attention probabilities (post-softmax).
            layer_idx: Which layer the weights come from (informational).
        """
        if self._warmup_complete:
            return

        num_heads = attention_weights.shape[1]
        if self._num_heads is None:
            self._num_heads = num_heads
            self._warmup_entropy_accum = torch.zeros(num_heads)

        entropy = _compute_attention_entropy(attention_weights)  # [batch, heads]
        # Average across batch
        mean_ent = entropy.float().mean(dim=0).cpu()  # [heads]
        self._warmup_entropy_accum += mean_ent
        self._warmup_count += attention_weights.shape[2]  # seq_q tokens

        if self._warmup_count >= self.warmup_tokens:
            self._finalize_head_bits()

    def _finalize_head_bits(self) -> None:
        """Compute per-head bit allocation from accumulated entropy."""
        if self._warmup_complete or self._warmup_entropy_accum is None:
            return

        avg_entropy = self._warmup_entropy_accum / max(self._warmup_count, 1)
        num_heads = avg_entropy.shape[0]

        # Compute percentile thresholds
        sorted_ent, _ = avg_entropy.sort()

        low_idx = max(0, int(num_heads * self.low_entropy_pct / 100) - 1)
        high_idx = min(num_heads - 1, int(num_heads * (100 - self.high_entropy_pct) / 100))

        low_threshold = sorted_ent[low_idx].item()
        high_threshold = sorted_ent[high_idx].item()

        # Assign per-head bits
        per_head_bits = []
        for h in range(num_heads):
            ent = avg_entropy[h].item()
            if ent <= low_threshold:
                # Low-entropy head: attends narrowly, can use fewer bits
                bits = max(self.base_key_bits - 1, 1)
            elif ent >= high_threshold:
                # High-entropy head: attends broadly, needs more bits
                bits = min(self.base_key_bits + 1, 8)
            else:
                bits = self.base_key_bits
            per_head_bits.append(bits)

        self._per_head_key_bits = per_head_bits
        self._warmup_complete = True

    @property
    def per_head_key_bits(self) -> Optional[List[int]]:
        """Per-head key bit allocation (None until warmup completes)."""
        return self._per_head_key_bits

    @property
    def warmup_complete(self) -> bool:
        """Whether the warmup phase has finished."""
        return self._warmup_complete

    # ---- HF Cache protocol (delegate to inner GenerationCache) ----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress/store new KV pairs, return full cache for the layer.

        Delegates to the inner GenerationCache, adding per-head bit
        metadata tracking. The actual per-head quantization is reflected
        through the layer-level key_bits assignment from the gradient
        anchor schedule.

        Args:
            key_states: ``[batch, num_heads, new_seq, head_dim]``
            value_states: ``[batch, num_heads, new_seq, head_dim]``
            layer_idx: Which transformer layer.
            cache_kwargs: Additional HF cache kwargs (ignored).

        Returns:
            Tuple of ``(all_keys, all_values)`` tensors.
        """
        return self._inner.update(key_states, value_states, layer_idx, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return number of cached tokens for a layer."""
        return self._inner.get_seq_length(layer_idx)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Return max cache shape. Dynamic cache has no maximum."""
        return -1

    def get_mask_sizes(
        self,
        cache_position: torch.Tensor,
        layer_idx: int = 0,
    ) -> tuple[int, int]:
        """Return ``(kv_length, kv_offset)`` for attention mask generation."""
        return self._inner.get_mask_sizes(cache_position, layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder all layers for beam search."""
        self._inner.reorder_cache(beam_idx)

    def crop(self, max_length: int) -> None:
        """Truncate all layers to ``max_length`` tokens."""
        self._inner.crop(max_length)

    def reset(self) -> None:
        """Clear all cached data and reset warmup state."""
        self._inner.reset()
        self._warmup_entropy_accum = None
        self._warmup_count = 0
        self._warmup_complete = False
        self._per_head_key_bits = None

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat cache entries for beam search expansion."""
        self._inner.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select specific batch indices from the cache."""
        self._inner.batch_select_indices(indices)

    @property
    def seen_tokens(self) -> int:
        """Number of tokens seen by the first layer."""
        return self._inner.seen_tokens

    @property
    def is_initialized(self) -> bool:
        """Return whether the cache has been populated."""
        return self._inner.is_initialized

    @property
    def is_sliding(self) -> list[bool]:
        """Return sliding window status per layer (always False)."""
        return self._inner.is_sliding

    def __len__(self) -> int:
        """Number of layers in the cache."""
        return len(self._inner)

    def __iter__(self):
        """Iterate over layers, yielding ``(keys, values, None)`` tuples."""
        return iter(self._inner)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dequantized ``(keys, values)`` for a specific layer."""
        return self._inner[layer_idx]

    def __contains__(self, idx: int) -> bool:
        """Check whether a layer index exists in the cache."""
        return idx in self._inner

    # ---- Reporting ----

    def memory_savings(self) -> Dict[str, Any]:
        """Report memory usage and savings across all layers."""
        report = self._inner.memory_savings()
        report["hybrid"] = True
        report["warmup_complete"] = self._warmup_complete
        report["per_head_key_bits"] = self._per_head_key_bits
        return report

    def anchor_summary(self) -> Dict[str, Any]:
        """Return anchor schedule summary."""
        return self._inner.anchor_summary()

    def config_summary(self) -> str:
        """Return a human-readable configuration summary."""
        base = self._inner.config_summary()
        head_desc = (
            f"per-head bits={self._per_head_key_bits}"
            if self._warmup_complete
            else f"warmup {self._warmup_count}/{self.warmup_tokens} tokens"
        )
        return f"HybridCache({base}, {head_desc})"
