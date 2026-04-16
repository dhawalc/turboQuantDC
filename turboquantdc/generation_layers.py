from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch

from .block_rotation import GivensRotation, QuaternionRotation
from .codebook import LloydMaxCodebook
from .e8_lattice import E8Quantizer
from .rotation import apply_wht_rotation, generate_qjl_matrix, generate_rotation_matrix, generate_wht_rotation

_TRITON_AVAILABLE = False
try:
    from .triton_kernels import triton_quantize as _triton_quantize
    _TRITON_AVAILABLE = True
except (ImportError, RuntimeError):
    pass

try:
    from .triton_kernels import triton_dequantize, triton_dequantize_residual
except (ImportError, RuntimeError):
    pass

from .generation_strategy import compute_layer_key_bits, compute_anchor_schedule, ANCHOR_STRATEGIES

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

    @property
    def is_compileable(self) -> bool:
        return False

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
        center_before_quantize: bool = True,
        quantizer_type: str = "lloyd_max",
    ):
        self.key_bits = key_bits
        self.val_bits = val_bits
        self.fp16_window = fp16_window
        self.seed = seed
        self.use_norm_correction = use_norm_correction
        self.use_residual_quant = use_residual_quant
        self.center_before_quantize = center_before_quantize
        self.quantizer_type = quantizer_type  # "lloyd_max" or "e8"
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

        # Online mean tracking for mean-centering keys.
        # Shape: (batch, num_heads, 1, head_dim) when initialized.
        self._key_running_mean: Optional[torch.Tensor] = None
        self._key_running_count: int = 0

        # Compressed storage (appended per update call)
        self._key_indices: List[torch.Tensor] = []
        self._key_norms: List[torch.Tensor] = []
        self._key_res_signs: List[torch.Tensor] = []
        self._key_res_scales: List[torch.Tensor] = []
        self._key_means: List[torch.Tensor] = []  # per-chunk mean used at quantize time
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

        # Block-diagonal rotation module (for "givens" / "quaternion")
        self._block_rotation = None

        if self._rotation_type == "givens":
            self._block_rotation = GivensRotation(d, seed=self.seed, device=device)
            self._rotation = None
            self._wht_params = None
        elif self._rotation_type == "quaternion":
            self._block_rotation = QuaternionRotation(d, seed=self.seed, device=device)
            self._rotation = None
            self._wht_params = None
        elif self._rotation_type == "wht":
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

        # E8 lattice quantizers (used when quantizer_type="e8")
        self._key_e8: Optional[E8Quantizer] = None
        self._val_e8: Optional[E8Quantizer] = None
        if self.quantizer_type == "e8":
            # E8 scale calibrated for post-WHT unit vector distribution (std ≈ 1.0)
            self._key_e8 = E8Quantizer(
                scale=2.0 / (2 ** self.key_bits), relaxed=True
            )
            self._val_e8 = E8Quantizer(
                scale=2.0 / (2 ** self.val_bits), relaxed=True
            )

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

        # Rotate (WHT: O(d log d); QR: O(d^2); block: O(d))
        if self._block_rotation is not None:
            rotated = self._block_rotation.rotate(normalized)
        elif self._rotation_type == "wht":
            rotated = apply_wht_rotation(normalized, self._wht_params)
        else:
            rotated = normalized @ self._rotation

        # Quantize: per-coordinate (Lloyd-Max) or 8D blocks (E8 lattice)
        e8_q = self._key_e8 if codebook is self._key_codebook else self._val_e8
        if self.quantizer_type == "e8" and e8_q is not None:
            # E8 lattice VQ: adaptive scale based on actual data range
            actual_scale = max(1.0 * rotated.std().item() / (2 ** codebook.bits), 1e-8)
            e8_q_adj = E8Quantizer(scale=actual_scale, relaxed=True)
            _, recon_rotated = e8_q_adj.quantize(rotated)
            indices = recon_rotated  # store scaled reconstruction as "indices" (float)
        else:
            indices = torch.bucketize(rotated, codebook.boundaries)
            indices = indices.clamp(0, codebook.centroids.shape[0] - 1)
            recon_rotated = codebook.centroids[indices]

        # Norm correction: store ratio to compensate reconstruction norm drift
        if self._block_rotation is not None:
            recon_unrotated = self._block_rotation.unrotate(recon_rotated)
        elif self._rotation_type == "wht":
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
        if self.quantizer_type == "e8":
            # E8: "indices" are actually lattice point coordinates (float)
            reconstructed = flat_idx.float()
        else:
            reconstructed = codebook.centroids[flat_idx]

        # Apply residual correction
        if res_signs is not None and res_scale is not None:
            reconstructed = (
                reconstructed
                + res_signs.reshape(-1, d) * res_scale.reshape(-1, 1)
            )

        # Unrotate and rescale (WHT: O(d log d); QR: O(d^2); block: O(d))
        if self._block_rotation is not None:
            reconstructed = self._block_rotation.unrotate(reconstructed)
        elif self._rotation_type == "wht":
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

        # Mean-centering: subtract per-head running mean before quantization.
        # Softmax is shift-invariant so the mean is invisible to attention,
        # but removing it reduces variance -> better codebook utilization.
        keys_to_quantize = key_states
        if self.center_before_quantize:
            # Online mean update: mean_new = (mean_old * n + sum_new) / (n + new_seq)
            new_sum = key_states.float().sum(dim=2, keepdim=True)  # (B, H, 1, D)
            old_n = self._key_running_count
            new_n = old_n + new_seq
            if self._key_running_mean is None:
                self._key_running_mean = new_sum / new_n
            else:
                self._key_running_mean = (
                    self._key_running_mean * old_n + new_sum
                ) / new_n
            self._key_running_count = new_n
            # Center using running mean (broadcast over seq dim)
            keys_to_quantize = key_states.float() - self._key_running_mean
            # Store the mean snapshot used for this chunk (for dequantization)
            chunk_mean = self._key_running_mean.expand(
                key_states.shape[0], key_states.shape[1], new_seq, key_states.shape[3]
            ).clone()
        else:
            chunk_mean = torch.zeros_like(key_states)

        # Compress keys with residual signs
        k_idx, k_norms, k_rsigns, k_rscale = self._quantize_vectors(
            keys_to_quantize, self._key_codebook,
        )
        self._key_indices.append(k_idx)
        self._key_norms.append(k_norms)
        self._key_res_signs.append(k_rsigns)
        self._key_res_scales.append(k_rscale)
        self._key_means.append(chunk_mean.to(key_states.dtype))

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

    def compress_only(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Compress and store new KV states WITHOUT full dequantization.

        Same as :meth:`update` but skips ``_dequantize_all``, avoiding the
        O(N) GPU memory allocation of the incremental dequant cache.  Used by
        ``TurboRetrievalCache`` which only needs ``dequantize_selected``.

        Compressed indices are stored on CPU to keep GPU memory minimal.
        Only the FP16 window and codebooks stay on GPU.
        """
        if self._rotation_type is None:
            self._lazy_init(key_states, value_states)

        new_seq = key_states.shape[2]

        # Mean-centering (copy of update logic)
        keys_to_quantize = key_states
        if self.center_before_quantize:
            new_sum = key_states.float().sum(dim=2, keepdim=True)
            old_n = self._key_running_count
            new_n = old_n + new_seq
            if self._key_running_mean is None:
                self._key_running_mean = new_sum / new_n
            else:
                self._key_running_mean = (
                    self._key_running_mean * old_n + new_sum
                ) / new_n
            self._key_running_count = new_n
            keys_to_quantize = key_states.float() - self._key_running_mean
            chunk_mean = self._key_running_mean.expand(
                key_states.shape[0], key_states.shape[1], new_seq, key_states.shape[3]
            ).clone()
        else:
            chunk_mean = torch.zeros_like(key_states)

        # Compress keys
        k_idx, k_norms, k_rsigns, k_rscale = self._quantize_vectors(
            keys_to_quantize, self._key_codebook,
        )
        # Store compressed data on CPU to minimize GPU memory
        self._key_indices.append(k_idx.cpu())
        self._key_norms.append(k_norms.cpu())
        self._key_res_signs.append(k_rsigns.cpu())
        self._key_res_scales.append(k_rscale.cpu())
        self._key_means.append(chunk_mean.to(key_states.dtype).cpu())

        # Compress values
        v_idx, v_norms, _, _ = self._quantize_vectors(
            value_states, self._val_codebook,
        )
        self._val_indices.append(v_idx.cpu())
        self._val_norms.append(v_norms.cpu())

        # Store raw FP16 for precision window (keep on GPU for fast access)
        self._raw_keys.append(key_states.detach())
        self._raw_vals.append(value_states.detach())

        if self.fp16_window > 0 and self._seq_len > self.fp16_window * 2:
            all_rk = torch.cat(self._raw_keys, dim=2)
            all_rv = torch.cat(self._raw_vals, dim=2)
            self._raw_keys = [all_rk[:, :, -self.fp16_window:, :]]
            self._raw_vals = [all_rv[:, :, -self.fp16_window:, :]]

        self._seq_len += new_seq

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
            new_k_means_parts = []
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
                new_k_means_parts.append(self._key_means[i][:, :, start_in_chunk:, :])
                new_v_idx_parts.append(self._val_indices[i][:, :, start_in_chunk:, :])
                new_v_norms_parts.append(self._val_norms[i][:, :, start_in_chunk:])
                seen = chunk_end

            if new_k_idx_parts:
                new_k_idx = torch.cat(new_k_idx_parts, dim=2)
                new_k_norms = torch.cat(new_k_norms_parts, dim=2)
                new_k_rsigns = torch.cat(new_k_rsigns_parts, dim=2)
                new_k_rscales = torch.cat(new_k_rscales_parts, dim=2)
                new_k_means = torch.cat(new_k_means_parts, dim=2)
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
                # Add back the stored mean for mean-centered keys
                if self.center_before_quantize:
                    new_keys = new_keys + new_k_means.float()
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
        if self.quantizer_type == "e8":
            # E8: "indices" are lattice point coordinates, not codebook indices
            reconstructed = flat_idx.float()
        else:
            reconstructed = codebook.centroids.float()[flat_idx.long()]

        # Apply residual correction in the rotated domain (float32)
        if res_signs is not None and res_scale is not None:
            reconstructed = (
                reconstructed
                + res_signs.float().reshape(-1, d) * res_scale.float().reshape(-1, 1)
            )

        # Unrotate in float32 (avoids FP16 intermediate)
        if self._block_rotation is not None:
            reconstructed = self._block_rotation.unrotate(reconstructed.float())
        elif self._rotation_type == "wht":
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

    def dequantize_selected(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize only the specific token indices (for retrieval attention).
        
        Args:
            indices: 1D tensor of token indices to extract.
            
        Returns:
            Tuple of keys and values, shape `[batch, num_heads, len(indices), head_dim]`.
        """
        if self._seq_len == 0 or len(indices) == 0:
            d = self._head_dim or 1
            empty = torch.zeros(
                self._batch_size or 1,
                self._num_heads or 1,
                0, d,
                dtype=self._dtype,
                device=self._device,
            )
            return empty, empty

        # Ensure we have the concatenated chunks or concat them now
        all_k_idx = torch.cat(self._key_indices, dim=2)
        all_k_norms = torch.cat(self._key_norms, dim=2)
        all_k_rsigns = torch.cat(self._key_res_signs, dim=2) if self._key_res_signs else None
        all_k_rscales = torch.cat(self._key_res_scales, dim=2) if self._key_res_scales else None
        all_k_means = torch.cat(self._key_means, dim=2) if self._key_means else None
        
        all_v_idx = torch.cat(self._val_indices, dim=2)
        all_v_norms = torch.cat(self._val_norms, dim=2)

        # Select via index (may be on CPU if stored via compress_only)
        # indices shape is (N,) -> we index the seq_len dimension (dim=2)
        cpu_indices = indices.cpu() if indices.is_cuda else indices
        sel_k_idx = all_k_idx[:, :, cpu_indices, :]
        sel_k_norms = all_k_norms[:, :, cpu_indices]
        sel_k_rsigns = all_k_rsigns[:, :, cpu_indices, :] if all_k_rsigns is not None else None
        sel_k_rscales = all_k_rscales[:, :, cpu_indices] if all_k_rscales is not None else None
        sel_k_means = all_k_means[:, :, cpu_indices, :] if all_k_means is not None else None

        sel_v_idx = all_v_idx[:, :, cpu_indices, :]
        sel_v_norms = all_v_norms[:, :, cpu_indices]

        # Move selected subset to GPU for dequantization (codebooks are on GPU)
        dev = self._device
        sel_k_idx = sel_k_idx.to(dev)
        sel_k_norms = sel_k_norms.to(dev)
        if sel_k_rsigns is not None:
            sel_k_rsigns = sel_k_rsigns.to(dev)
        if sel_k_rscales is not None:
            sel_k_rscales = sel_k_rscales.to(dev)
        if sel_k_means is not None:
            sel_k_means = sel_k_means.to(dev)
        sel_v_idx = sel_v_idx.to(dev)
        sel_v_norms = sel_v_norms.to(dev)

        # Dequantize the selected subset
        keys = self._dequantize_vectors_fused(
            sel_k_idx, sel_k_norms, self._key_codebook,
            sel_k_rsigns if self.use_residual_quant else None,
            sel_k_rscales if self.use_residual_quant else None,
        )
        if self.center_before_quantize and sel_k_means is not None:
            keys = keys + sel_k_means.float()
            
        values = self._dequantize_vectors(
            sel_v_idx, sel_v_norms, self._val_codebook,
        )

        # Handle FP16 window replacement logic (if indices intersect the window)
        if self._raw_keys and self.fp16_window > 0:
            window_start = self._seq_len - self.fp16_window
            raw_keys = torch.cat(self._raw_keys, dim=2)
            raw_vals = torch.cat(self._raw_vals, dim=2)
            
            # Find which selected indices fall into the FP16 window
            mask = indices >= window_start
            if mask.any():
                # Map global indices to the offset within the raw_keys window
                # The raw_keys window contains the last `win` tokens
                win = raw_keys.shape[2]
                raw_start = self._seq_len - win
                
                # Filter out those that are in the true FP16 window
                # For each such index, its position in raw_keys is idx - raw_start
                valid_win_idx = indices[mask]
                rel_win_idx = valid_win_idx - raw_start
                
                # Need to update the output tensors
                # Find indices inside the `mask` array
                update_positions = torch.where(mask)[0]
                
                if (rel_win_idx >= 0).all() and (rel_win_idx < win).all():
                    keys[:, :, update_positions, :] = raw_keys[:, :, rel_win_idx, :].to(keys.dtype)
                    values[:, :, update_positions, :] = raw_vals[:, :, rel_win_idx, :].to(values.dtype)

        return keys.to(self._dtype), values.to(self._dtype)

    def clear(self) -> None:
        """Clear all stored data."""
        self._key_indices.clear()
        self._key_norms.clear()
        self._key_res_signs.clear()
        self._key_res_scales.clear()
        self._key_means.clear()
        self._val_indices.clear()
        self._val_norms.clear()
        self._raw_keys.clear()
        self._raw_vals.clear()
        self._seq_len = 0
        self._key_running_mean = None
        self._key_running_count = 0
        self._dequant_key_cache = None
        self._dequant_val_cache = None
        self._dequant_len = 0

    def reorder(self, beam_idx: torch.LongTensor) -> None:
        """Reorder cache entries along the batch dimension for beam search."""
        self._key_indices = [t.index_select(0, beam_idx) for t in self._key_indices]
        self._key_norms = [t.index_select(0, beam_idx) for t in self._key_norms]
        self._key_res_signs = [t.index_select(0, beam_idx) for t in self._key_res_signs]
        self._key_res_scales = [t.index_select(0, beam_idx) for t in self._key_res_scales]
        self._key_means = [t.index_select(0, beam_idx) for t in self._key_means]
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
            all_k_means = torch.cat(self._key_means, dim=2)[:, :, :max_length]
            all_v_idx = torch.cat(self._val_indices, dim=2)[:, :, :max_length]
            all_v_norms = torch.cat(self._val_norms, dim=2)[:, :, :max_length]
            raw_keys = torch.cat(self._raw_keys, dim=2)[:, :, :max_length]
            raw_vals = torch.cat(self._raw_vals, dim=2)[:, :, :max_length]

            self._key_indices = [all_k_idx]
            self._key_norms = [all_k_norms]
            self._key_res_signs = [all_k_rsigns]
            self._key_res_scales = [all_k_rscales]
            self._key_means = [all_k_means]
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


