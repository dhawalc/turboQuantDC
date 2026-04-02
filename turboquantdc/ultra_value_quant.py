"""1-bit value quantization with correction mechanisms.

Tom and community proved: "V compression is free. fp16-K + 2bit-V gives
perfect 1.000 cosine similarity." This module pushes the boundary to 1-bit
values to test whether that claim extends to the absolute limit.

Three correction methods for 1-bit values:

**Method A (scale):** 1-bit per coordinate + one FP16 scale per vector.
  Reconstruction: scale * centroid[sign(rotated[i])]
  Storage: 1 bit/coord + 16 bits/vector = ~1.125 bpc for d=128

**Method B (residual):** 1-bit primary + 1-bit residual signs (ResidualQuant).
  Stage 1: 1-bit Lloyd-Max quantization
  Stage 2: 1-bit residual sign correction
  Storage: 2 bits/coord + 16 bits/vector = ~2.125 bpc for d=128

**Method C (layer schedule):** Boundary layers at 3-bit V, middle at 1-bit.
  Average: ~1.2 bpc depending on layer count.

The key insight: V errors scale linearly through attention as softmax(QK^T) @ V.
If attention weights are sparse (most weight on 1-2 tokens), V errors on
low-weight tokens are effectively masked. This is why V compression is "free"
for attention fidelity, even if individual vector MSE is high.

Reference: TurboQuant paper (arxiv 2504.19874), Tom's community findings.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from .codebook import LloydMaxCodebook
from .rotation import (
    apply_wht_rotation,
    generate_rotation_matrix,
    generate_wht_rotation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def compute_value_layer_schedule(
    num_layers: int,
    base_val_bits: int = 1,
    boundary_val_bits: int = 3,
    boundary_count: int = 2,
) -> List[int]:
    """Compute per-layer V bit-width schedule with boundary protection.

    Boundary layers (first ``boundary_count`` + last ``boundary_count``) get
    ``boundary_val_bits``. Middle layers get ``base_val_bits``.

    Args:
        num_layers: Total transformer layers.
        base_val_bits: Bits for middle layer values (default: 1).
        boundary_val_bits: Bits for boundary layer values (default: 3).
        boundary_count: Number of boundary layers on each end (default: 2).

    Returns:
        List of int, one per layer, specifying V bit-width.
    """
    if num_layers <= 0:
        return []

    schedule = []
    for i in range(num_layers):
        is_boundary = (
            i < boundary_count or i >= num_layers - boundary_count
        )
        schedule.append(boundary_val_bits if is_boundary else base_val_bits)
    return schedule


# ---------------------------------------------------------------------------
# UltraValueQuantizer — 1-bit value quantization with corrections
# ---------------------------------------------------------------------------


class UltraValueQuantizer:
    """1-bit value quantizer with optional correction mechanisms.

    For a Gaussian N(0, 1/d) coordinate distribution, the optimal 1-bit
    Lloyd-Max centroids are approximately +/- 0.7979/sqrt(d). Each coordinate
    is mapped to one of these two centroids.

    Correction methods:
        - ``"none"``: Raw 1-bit, no correction. Cheapest but highest MSE.
        - ``"scale"``: Per-vector FP16 scale factor. Captures magnitude
          mismatch between original and quantized. Adds 16 bits per vector.
        - ``"residual"``: 1-bit residual signs + per-vector FP16 scale.
          Captures the sign of the quantization residual in the rotated domain.
          Adds d bits + 16 bits per vector (effective 2 bpc).

    Args:
        d: Head dimension.
        method: Correction method — one of "none", "scale", "residual".
        seed: Random seed for rotation matrix.
    """

    METHODS = ("none", "scale", "residual")

    def __init__(
        self,
        d: int,
        method: str = "scale",
        seed: int = 42,
    ):
        if method not in self.METHODS:
            raise ValueError(
                f"method must be one of {self.METHODS}, got {method!r}"
            )

        self.d = d
        self.method = method
        self.seed = seed

        # Build 1-bit codebook
        self.codebook = LloydMaxCodebook(d=d, bits=1)

        # Build rotation
        if _is_power_of_2(d):
            self._rotation_type = "wht"
            self._wht_params = generate_wht_rotation(d, seed=seed, device="cpu")
            self._rotation = None
        else:
            self._rotation_type = "qr"
            self._rotation = generate_rotation_matrix(d, seed=seed, device="cpu")
            self._wht_params = None

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply orthogonal rotation to input vectors."""
        if self._rotation_type == "wht":
            return apply_wht_rotation(x, self._wht_params)
        return x @ self._rotation

    def _unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation."""
        if self._rotation_type == "wht":
            return apply_wht_rotation(y, self._wht_params, inverse=True)
        return y @ self._rotation.T

    def quantize(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize value vectors to 1-bit with optional correction.

        Args:
            x: Input vectors of shape (n, d).

        Returns:
            Tuple of (x_hat, metadata) where x_hat is the reconstructed
            tensor and metadata contains the compressed representation.
        """
        n, d = x.shape
        assert d == self.d

        x_float = x.float()

        # Extract norms and normalize
        norms = x_float.norm(dim=-1, keepdim=True)
        normalized = x_float / (norms + 1e-8)

        # Rotate to decorrelated domain
        rotated = self._rotate(normalized)

        # 1-bit quantization: map to nearest centroid
        centroids = self.codebook.centroids.to(x.device)
        indices = self.codebook.quantize(rotated)  # 0 or 1
        recon_rotated = centroids[indices]  # centroid lookup

        metadata: Dict[str, torch.Tensor] = {
            "indices": indices,
            "norms": norms.squeeze(-1),
        }

        if self.method == "none":
            # Raw centroid reconstruction
            recon_unrotated = self._unrotate(recon_rotated)
            x_hat = recon_unrotated * norms

        elif self.method == "scale":
            # Method A: per-vector optimal scale factor
            # Find scale s that minimizes ||x_normalized - s * recon_normalized||^2
            # Solution: s = <x_rotated, recon_rotated> / ||recon_rotated||^2
            dot = (rotated * recon_rotated).sum(dim=-1, keepdim=True)
            recon_sq = (recon_rotated * recon_rotated).sum(dim=-1, keepdim=True)
            scale = dot / (recon_sq + 1e-8)
            scale = scale.clamp(0.1, 10.0)  # Prevent extreme scales

            scaled_recon = recon_rotated * scale
            recon_unrotated = self._unrotate(scaled_recon)
            x_hat = recon_unrotated * norms

            metadata["scale"] = scale.squeeze(-1)

        elif self.method == "residual":
            # Method B: 1-bit residual correction
            # Stage 1 residual in rotated domain
            residual = rotated - recon_rotated
            res_signs = torch.sign(residual)
            # Replace zeros with +1 to avoid dead corrections
            res_signs = torch.where(res_signs == 0, torch.ones_like(res_signs), res_signs)
            res_scale = residual.abs().mean(dim=-1, keepdim=True)

            # Apply residual correction
            corrected_rotated = recon_rotated + res_signs * res_scale
            recon_unrotated = self._unrotate(corrected_rotated)
            x_hat = recon_unrotated * norms

            metadata["residual_signs"] = res_signs
            metadata["residual_scale"] = res_scale.squeeze(-1)

        return x_hat, metadata

    def effective_bits_per_coord(self) -> float:
        """Compute effective bits per coordinate including overhead.

        Returns:
            Effective bits per coordinate value.
        """
        d = self.d
        if self.method == "none":
            # 1 bit per coord + 16 bits norm per vector
            return 1.0 + 16.0 / d
        elif self.method == "scale":
            # 1 bit per coord + 16 bits scale per vector (norm is implicit)
            return 1.0 + 16.0 / d
        elif self.method == "residual":
            # 1 bit primary + 1 bit residual signs + 16 bits scale per vector
            return 2.0 + 16.0 / d
        return 1.0

    def __repr__(self) -> str:
        return (
            f"UltraValueQuantizer(d={self.d}, method={self.method!r}, "
            f"bpc={self.effective_bits_per_coord():.3f})"
        )


# ---------------------------------------------------------------------------
# Value bit-width sweep
# ---------------------------------------------------------------------------


def sweep_value_bits(
    d: int = 128,
    num_tokens: int = 64,
    key_bits: int = 4,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Compare V=1, V=2, V=3, V=4 at fixed K bits.

    For each V bit-width, measures:
    - Value reconstruction MSE
    - Attention output cosine similarity (what actually matters)
    - Effective bits per coordinate

    Args:
        d: Head dimension.
        num_tokens: Number of tokens in the simulated sequence.
        key_bits: Fixed key bit-width.
        seed: Random seed.

    Returns:
        List of result dicts, sorted by effective_bpc.
    """
    torch.manual_seed(seed)

    # Generate test data: queries, keys, values
    queries = torch.randn(1, 1, 4, d)
    keys = torch.randn(1, 1, num_tokens, d)
    values = torch.randn(1, 1, num_tokens, d)

    # Reference attention output
    d_k = d
    scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    ref_output = torch.matmul(weights, values)

    results = []

    # Standard bit-widths using LloydMaxCodebook
    for bits in [2, 3, 4]:
        codebook = LloydMaxCodebook(d=d, bits=bits)

        # Build rotation for quantizing values
        if _is_power_of_2(d):
            wht_params = generate_wht_rotation(d, seed=seed, device="cpu")
            def rotate(x):
                return apply_wht_rotation(x, wht_params)
            def unrotate(y):
                return apply_wht_rotation(y, wht_params, inverse=True)
        else:
            R = generate_rotation_matrix(d, seed=seed, device="cpu")
            def rotate(x, R=R):
                return x @ R
            def unrotate(y, R=R):
                return y @ R.T

        flat_v = values.reshape(-1, d).float()
        v_norms = flat_v.norm(dim=-1, keepdim=True)
        v_normalized = flat_v / (v_norms + 1e-8)
        v_rotated = rotate(v_normalized)

        indices = codebook.quantize(v_rotated)
        recon_rotated = codebook.centroids[indices]
        recon_flat = unrotate(recon_rotated) * v_norms
        q_values = recon_flat.reshape(values.shape)

        # Metrics
        value_mse = (values.float() - q_values).pow(2).mean().item()
        test_output = torch.matmul(weights, q_values)
        flat_ref = ref_output.reshape(-1, d).float()
        flat_test = test_output.reshape(-1, d).float()
        attn_sim = torch.nn.functional.cosine_similarity(
            flat_ref, flat_test, dim=-1
        ).mean().item()

        results.append({
            "val_bits": bits,
            "method": "standard",
            "value_mse": value_mse,
            "attention_cosine_sim": attn_sim,
            "effective_bpc": float(bits) + 16.0 / d,
        })

    # 1-bit methods
    for method in ["none", "scale", "residual"]:
        quantizer = UltraValueQuantizer(d=d, method=method, seed=seed)
        flat_v = values.reshape(-1, d)
        q_flat, _ = quantizer.quantize(flat_v)
        q_values = q_flat.reshape(values.shape)

        value_mse = (values.float() - q_values).pow(2).mean().item()
        test_output = torch.matmul(weights, q_values)
        flat_ref = ref_output.reshape(-1, d).float()
        flat_test = test_output.reshape(-1, d).float()
        attn_sim = torch.nn.functional.cosine_similarity(
            flat_ref, flat_test, dim=-1
        ).mean().item()

        results.append({
            "val_bits": 1,
            "method": method,
            "value_mse": value_mse,
            "attention_cosine_sim": attn_sim,
            "effective_bpc": quantizer.effective_bits_per_coord(),
        })

    results.sort(key=lambda r: r["effective_bpc"])
    return results


# ---------------------------------------------------------------------------
# Ultra Value Cache Layer
# ---------------------------------------------------------------------------


class _UltraValueLayer:
    """Single layer of the ultra value cache.

    Keys: standard Lloyd-Max at ``key_bits`` with residual sign correction.
    Values: 1-bit UltraValueQuantizer with configurable correction method.

    Args:
        key_bits: Bits for key quantization.
        val_bits: Bits for value quantization (1 for ultra, 2-4 for boundary).
        val_method: Value correction method ("none", "scale", "residual").
        fp16_window: FP16 precision window for most recent tokens.
        seed: Random seed.
        use_residual_quant: Whether keys use 1-bit residual sign correction.
    """

    def __init__(
        self,
        key_bits: int = 4,
        val_bits: int = 1,
        val_method: str = "scale",
        fp16_window: int = 64,
        seed: int = 42,
        use_residual_quant: bool = True,
    ):
        self.key_bits = key_bits
        self.val_bits = val_bits
        self.val_method = val_method
        self.fp16_window = fp16_window
        self.seed = seed
        self.use_residual_quant = use_residual_quant

        self._seq_len: int = 0
        self._head_dim: Optional[int] = None
        self._num_heads: Optional[int] = None
        self._batch_size: Optional[int] = None
        self._dtype: Optional[torch.dtype] = None
        self._device: Optional[torch.device] = None

        # Rotation params (shared for K and V)
        self._rotation_type: Optional[str] = None
        self._rotation: Optional[torch.Tensor] = None
        self._wht_params: Optional[dict] = None

        # Codebooks
        self._key_codebook: Optional[LloydMaxCodebook] = None
        self._val_codebook: Optional[LloydMaxCodebook] = None
        self._val_quantizer: Optional[UltraValueQuantizer] = None

        # Compressed storage: keys
        self._key_indices: List[torch.Tensor] = []
        self._key_norms: List[torch.Tensor] = []
        self._key_res_signs: List[torch.Tensor] = []
        self._key_res_scales: List[torch.Tensor] = []

        # Compressed storage: values
        self._val_indices: List[torch.Tensor] = []
        self._val_norms: List[torch.Tensor] = []
        self._val_corrections: List[Dict[str, torch.Tensor]] = []

        # FP16 window
        self._raw_keys: List[torch.Tensor] = []
        self._raw_vals: List[torch.Tensor] = []

        # Incremental dequant cache
        self._dequant_key_cache: Optional[torch.Tensor] = None
        self._dequant_val_cache: Optional[torch.Tensor] = None
        self._dequant_len: int = 0

    def _lazy_init(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        self._batch_size = key_states.shape[0]
        self._num_heads = key_states.shape[1]
        self._head_dim = key_states.shape[3]
        self._dtype = key_states.dtype
        self._device = key_states.device
        d = self._head_dim
        device_str = str(self._device)

        # Rotation
        if _is_power_of_2(d):
            self._rotation_type = "wht"
            self._wht_params = generate_wht_rotation(d, seed=self.seed, device=device_str)
            self._rotation = None
        else:
            self._rotation_type = "qr"
            self._rotation = generate_rotation_matrix(d, seed=self.seed, device=device_str)
            self._wht_params = None

        # Key codebook
        self._key_codebook = LloydMaxCodebook(d=d, bits=self.key_bits).to(device_str)

        # Value quantizer: use UltraValueQuantizer for 1-bit, standard codebook otherwise
        if self.val_bits == 1:
            self._val_quantizer = UltraValueQuantizer(
                d=d, method=self.val_method, seed=self.seed,
            )
            self._val_codebook = None
        else:
            self._val_codebook = LloydMaxCodebook(d=d, bits=self.val_bits).to(device_str)
            self._val_quantizer = None

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        if self._rotation_type == "wht":
            return apply_wht_rotation(x, self._wht_params)
        return x @ self._rotation

    def _unrotate(self, y: torch.Tensor) -> torch.Tensor:
        if self._rotation_type == "wht":
            return apply_wht_rotation(y, self._wht_params, inverse=True)
        return y @ self._rotation.T

    def _quantize_keys(
        self, key_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize keys with rotation + Lloyd-Max + residual signs."""
        batch, heads, seq, d = key_states.shape
        flat = key_states.float().reshape(-1, d)

        norms = flat.norm(dim=-1, keepdim=True)
        normalized = flat / (norms + 1e-8)
        rotated = self._rotate(normalized)

        codebook = self._key_codebook
        indices = torch.bucketize(rotated, codebook.boundaries)
        indices = indices.clamp(0, codebook.centroids.shape[0] - 1)

        recon_rotated = codebook.centroids[indices]

        # Norm correction
        recon_unrotated = self._unrotate(recon_rotated)
        recon_norm = recon_unrotated.norm(dim=-1, keepdim=True)
        corrected_norms = norms * (
            norms / (recon_norm * norms.abs().clamp(min=1e-8) + 1e-8)
        ).clamp(0.5, 2.0)

        # Residual signs
        residual = rotated - recon_rotated
        res_signs = torch.sign(residual)
        res_scale = residual.abs().mean(dim=-1, keepdim=True)

        return (
            indices.reshape(batch, heads, seq, d),
            corrected_norms.reshape(batch, heads, seq),
            res_signs.reshape(batch, heads, seq, d),
            res_scale.reshape(batch, heads, seq),
        )

    def _dequantize_keys(
        self,
        indices: torch.Tensor,
        norms: torch.Tensor,
        res_signs: Optional[torch.Tensor] = None,
        res_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruct keys from compressed representation."""
        batch, heads, seq, d = indices.shape
        flat_idx = indices.reshape(-1, d)
        flat_norms = norms.reshape(-1)

        codebook = self._key_codebook
        reconstructed = codebook.centroids.float()[flat_idx.long()]

        if res_signs is not None and res_scale is not None and self.use_residual_quant:
            reconstructed = (
                reconstructed
                + res_signs.float().reshape(-1, d) * res_scale.float().reshape(-1, 1)
            )

        reconstructed = self._unrotate(reconstructed)
        reconstructed = reconstructed * flat_norms.float().unsqueeze(-1)
        return reconstructed.reshape(batch, heads, seq, d)

    def _quantize_values(
        self, value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize values using 1-bit UltraValueQuantizer or standard codebook."""
        batch, heads, seq, d = value_states.shape

        if self._val_quantizer is not None:
            # 1-bit ultra path
            flat = value_states.float().reshape(-1, d)
            recon_flat, metadata = self._val_quantizer.quantize(flat)
            recon = recon_flat.reshape(batch, heads, seq, d)
            return recon, metadata
        else:
            # Standard Lloyd-Max path
            flat = value_states.float().reshape(-1, d)
            norms = flat.norm(dim=-1, keepdim=True)
            normalized = flat / (norms + 1e-8)
            rotated = self._rotate(normalized)

            codebook = self._val_codebook
            indices = torch.bucketize(rotated, codebook.boundaries)
            indices = indices.clamp(0, codebook.centroids.shape[0] - 1)
            recon_rotated = codebook.centroids[indices]
            recon_unrotated = self._unrotate(recon_rotated) * norms
            recon = recon_unrotated.reshape(batch, heads, seq, d)
            metadata = {
                "indices": indices.reshape(batch, heads, seq, d),
                "norms": norms.squeeze(-1).reshape(batch, heads, seq),
            }
            return recon, metadata

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress and store new KV states, return full dequantized cache."""
        if self._rotation_type is None:
            self._lazy_init(key_states, value_states)

        new_seq = key_states.shape[2]

        # Compress keys
        k_idx, k_norms, k_rsigns, k_rscale = self._quantize_keys(key_states)
        self._key_indices.append(k_idx)
        self._key_norms.append(k_norms)
        self._key_res_signs.append(k_rsigns)
        self._key_res_scales.append(k_rscale)

        # Compress values
        v_recon, v_meta = self._quantize_values(value_states)
        self._val_corrections.append(v_meta)

        # Store raw FP16
        self._raw_keys.append(key_states.detach())
        self._raw_vals.append(value_states.detach())

        # Trim raw storage
        if self.fp16_window > 0 and self._seq_len > self.fp16_window * 2:
            all_rk = torch.cat(self._raw_keys, dim=2)
            all_rv = torch.cat(self._raw_vals, dim=2)
            self._raw_keys = [all_rk[:, :, -self.fp16_window:, :]]
            self._raw_vals = [all_rv[:, :, -self.fp16_window:, :]]

        self._seq_len += new_seq

        # Incremental dequant
        new_keys = self._dequantize_keys(
            k_idx, k_norms,
            k_rsigns if self.use_residual_quant else None,
            k_rscale if self.use_residual_quant else None,
        )

        if self._dequant_key_cache is not None:
            self._dequant_key_cache = torch.cat(
                [self._dequant_key_cache, new_keys], dim=2,
            )
            self._dequant_val_cache = torch.cat(
                [self._dequant_val_cache, v_recon], dim=2,
            )
        else:
            self._dequant_key_cache = new_keys
            self._dequant_val_cache = v_recon
        self._dequant_len = self._seq_len

        # Apply FP16 window
        keys = self._dequant_key_cache.clone()
        values = self._dequant_val_cache.clone()

        if self._raw_keys:
            raw_keys = torch.cat(self._raw_keys, dim=2)
            raw_vals = torch.cat(self._raw_vals, dim=2)
            win = min(self.fp16_window, raw_keys.shape[2])
            if win > 0 and keys.shape[2] >= win:
                keys[:, :, -win:, :] = raw_keys[:, :, -win:, :].to(keys.dtype)
                values[:, :, -win:, :] = raw_vals[:, :, -win:, :].to(values.dtype)

        return keys.to(self._dtype), values.to(self._dtype)

    def get_seq_length(self) -> int:
        return self._seq_len

    def clear(self) -> None:
        self._key_indices.clear()
        self._key_norms.clear()
        self._key_res_signs.clear()
        self._key_res_scales.clear()
        self._val_corrections.clear()
        self._raw_keys.clear()
        self._raw_vals.clear()
        self._seq_len = 0
        self._dequant_key_cache = None
        self._dequant_val_cache = None
        self._dequant_len = 0

    def reorder(self, beam_idx: torch.LongTensor) -> None:
        self._key_indices = [t.index_select(0, beam_idx) for t in self._key_indices]
        self._key_norms = [t.index_select(0, beam_idx) for t in self._key_norms]
        self._key_res_signs = [t.index_select(0, beam_idx) for t in self._key_res_signs]
        self._key_res_scales = [t.index_select(0, beam_idx) for t in self._key_res_scales]
        self._raw_keys = [t.index_select(0, beam_idx) for t in self._raw_keys]
        self._raw_vals = [t.index_select(0, beam_idx) for t in self._raw_vals]
        self._dequant_key_cache = None
        self._dequant_val_cache = None
        self._dequant_len = 0

    def memory_usage_bits(self) -> Dict[str, Any]:
        if self._seq_len == 0 or self._head_dim is None:
            return {
                "key_bits": 0, "value_bits": 0, "total_bits": 0,
                "fp16_baseline_bits": 0, "compression_ratio": 1.0,
            }
        d = self._head_dim
        n_heads = self._num_heads
        batch = self._batch_size
        total_tokens = self._seq_len * n_heads * batch

        fp16_tokens = min(self.fp16_window, self._seq_len) * n_heads * batch
        compressed_tokens = total_tokens - fp16_tokens

        # Key: key_bits*d + d (residual signs) + 32 (norm + res_scale)
        key_bits_c = compressed_tokens * (self.key_bits * d + d + 32)
        key_bits_fp16 = fp16_tokens * d * 16

        # Value: depends on method
        if self._val_quantizer is not None:
            bpc = self._val_quantizer.effective_bits_per_coord()
            val_bits_c = int(compressed_tokens * bpc * d)
        else:
            val_bits_c = compressed_tokens * (self.val_bits * d + 16)
        val_bits_fp16 = fp16_tokens * d * 16

        total_key = key_bits_c + key_bits_fp16
        total_val = val_bits_c + val_bits_fp16
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
# UltraValueCache — HF-compatible multi-layer cache
# ---------------------------------------------------------------------------


class UltraValueCache:
    """HF-compatible KV cache with 1-bit ultra value quantization.

    Keys: standard 3-4 bit PolarQuant with residual sign correction.
    Values: 1-bit UltraValueQuantizer (V compression is free per Tom).

    Supports per-layer V bit schedule for boundary protection (Method C).

    Duck-types the HuggingFace Cache protocol: update(), get_seq_length(),
    __getitem__, __len__, __iter__, reorder_cache().

    Args:
        key_bits: Key bit-width (default: 4).
        val_method: 1-bit correction method ("none", "scale", "residual").
        fp16_window: FP16 precision window.
        val_layer_schedule: Optional per-layer V bit-widths. When provided,
            layers with val_bits > 1 use standard Lloyd-Max instead of 1-bit.
        seed: Random seed.
        use_residual_quant: Whether keys use residual sign correction.
    """

    is_compileable = False

    def __init__(
        self,
        key_bits: int = 4,
        val_method: str = "scale",
        fp16_window: int = 64,
        val_layer_schedule: Optional[List[int]] = None,
        seed: int = 42,
        use_residual_quant: bool = True,
    ):
        self.key_bits = key_bits
        self.val_method = val_method
        self.fp16_window = fp16_window
        self.val_layer_schedule = val_layer_schedule
        self.seed = seed
        self.use_residual_quant = use_residual_quant

        self._layers: List[_UltraValueLayer] = []

    def _get_or_create_layer(self, layer_idx: int) -> _UltraValueLayer:
        """Get or lazily create a layer."""
        while len(self._layers) <= layer_idx:
            idx = len(self._layers)
            if self.val_layer_schedule is not None and idx < len(self.val_layer_schedule):
                val_bits = self.val_layer_schedule[idx]
            else:
                val_bits = 1  # Default to 1-bit ultra

            layer = _UltraValueLayer(
                key_bits=self.key_bits,
                val_bits=val_bits,
                val_method=self.val_method,
                fp16_window=self.fp16_window,
                seed=self.seed + idx,
                use_residual_quant=self.use_residual_quant,
            )
            self._layers.append(layer)
        return self._layers[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int = 0,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress and store new KV states for a layer.

        Args:
            key_states: [batch, heads, seq, d]
            value_states: [batch, heads, seq, d]
            layer_idx: Which transformer layer.
            cache_kwargs: Ignored (HF compatibility).

        Returns:
            Tuple of (all_keys, all_values) for this layer.
        """
        layer = self._get_or_create_layer(layer_idx)
        return layer.update(key_states, value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self._layers):
            return self._layers[layer_idx].get_seq_length()
        return 0

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):
        for layer in self._layers:
            keys = layer._dequant_key_cache
            values = layer._dequant_val_cache
            if keys is None:
                d = layer._head_dim or 1
                empty = torch.zeros(1, 1, 0, d)
                yield empty, empty
            else:
                yield keys, values

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        layer = self._layers[idx]
        keys = layer._dequant_key_cache
        values = layer._dequant_val_cache
        if keys is None:
            d = layer._head_dim or 1
            empty = torch.zeros(1, 1, 0, d)
            return empty, empty
        # Apply FP16 window
        k = keys.clone()
        v = values.clone()
        if layer._raw_keys:
            raw_k = torch.cat(layer._raw_keys, dim=2)
            raw_v = torch.cat(layer._raw_vals, dim=2)
            win = min(layer.fp16_window, raw_k.shape[2])
            if win > 0 and k.shape[2] >= win:
                k[:, :, -win:, :] = raw_k[:, :, -win:, :].to(k.dtype)
                v[:, :, -win:, :] = raw_v[:, :, -win:, :].to(v.dtype)
        if layer._dtype is not None:
            k = k.to(layer._dtype)
            v = v.to(layer._dtype)
        return k, v

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for layer in self._layers:
            layer.reorder(beam_idx)

    def memory_usage_bits(self, layer_idx: int = 0) -> Dict[str, Any]:
        if layer_idx < len(self._layers):
            return self._layers[layer_idx].memory_usage_bits()
        return {
            "key_bits": 0, "value_bits": 0, "total_bits": 0,
            "fp16_baseline_bits": 0, "compression_ratio": 1.0,
        }

    def total_memory_usage_bits(self) -> Dict[str, Any]:
        """Aggregate memory usage across all layers."""
        total_key = 0
        total_val = 0
        total_fp16 = 0
        for layer in self._layers:
            m = layer.memory_usage_bits()
            total_key += m["key_bits"]
            total_val += m["value_bits"]
            total_fp16 += m["fp16_baseline_bits"]
        total = total_key + total_val
        return {
            "key_bits": total_key,
            "value_bits": total_val,
            "total_bits": total,
            "fp16_baseline_bits": total_fp16,
            "compression_ratio": total_fp16 / total if total > 0 else 1.0,
        }

    def __repr__(self) -> str:
        n = len(self._layers)
        return (
            f"UltraValueCache(key_bits={self.key_bits}, "
            f"val_method={self.val_method!r}, "
            f"fp16_window={self.fp16_window}, "
            f"layers={n})"
        )
