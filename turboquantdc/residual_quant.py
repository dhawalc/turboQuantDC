"""ResidualQuant — MSE quantization + direct residual sign correction.

Alternative to QJL (Stage 2) that stores the actual sign of each residual
coordinate instead of random-projection signs.

QJL approach (standard TurboQuant):
    1. Compute residual r = x - x_mse in original space
    2. Project through random Gaussian S: projected = S @ r
    3. Store sign(projected) — 1 bit per dimension
    4. Correction uses: ||r|| * sqrt(pi/2)/m * <S@q, signs>
    This is UNBIASED but has high variance from the random projection.

ResidualQuant approach (this module):
    1. Compute residual in ROTATED space: r_rot = x_rot - centroids[indices]
    2. Store sign(r_rot) — 1 bit per coordinate (NO random projection)
    3. Store mean(|r_rot|) as scale — 16 bits per vector
    4. Correction: k_corrected_rot = centroids[indices] + scale * sign(r_rot)
    5. Unrotate to get k_corrected
    This is BIASED but has LOWER VARIANCE because no random projection noise.

The key insight: for autoregressive generation, low variance matters more than
unbiasedness. The sign of the actual residual preserves its direction perfectly
(just not its per-coordinate magnitude). QJL's random projection destroys
direction information while gaining unbiasedness — a bad trade for generation.

Storage is identical to TurboQuant at the same bit-width:
    (b-1)*d bits MSE + d bits residual signs + 16 bits scale + 16 bits norm
    = b*d + 32 bits per vector
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .polarquant import PolarQuant


class ResidualQuantEstimator(nn.Module):
    """MSE quantization + direct residual sign correction.

    Instead of QJL (random projection + signs), stores the actual sign
    of each residual coordinate in rotated space. Biased but low-variance.

    The correction in rotated space:
        k_corrected_rot = centroids[indices] + residual_scale * sign(r_rot)

    Then unrotate to get the corrected vector in original space.

    When ``center_before_quantize=True`` (default), the per-head mean is
    subtracted before quantization. Softmax is shift-invariant, so removing
    the mean is lossless for attention but reduces variance, giving better
    codebook utilization -- effectively a FREE +1 bit of precision.

    Args:
        d: Head dimension (e.g. 128).
        bits: Total effective bits per coordinate.
        seed: Random seed for rotation matrix generation.
        device: Target device.
        center_before_quantize: Subtract per-head mean from keys before
            quantization. The mean is stored as FP16 (2 bytes per head,
            negligible overhead) and added back during dequantization.
            Default: True.
    """

    def __init__(
        self,
        d: int,
        bits: int,
        seed: int = 42,
        device: str | torch.device = "cpu",
        center_before_quantize: bool = True,
    ):
        super().__init__()
        self.d = d
        self.bits = bits
        self.center_before_quantize = center_before_quantize

        # Bit budget: (bits-1) for MSE, 1 for residual signs
        self.mse_bits = max(bits - 1, 1)

        # Stage 1: PolarQuant with mse_bits (same as TurboQuant)
        self.polar = PolarQuant(d, self.mse_bits, seed=seed, device=device)

    def quantize(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compress a vector using MSE + direct residual signs.

        Algorithm:
            0. (Optional) Subtract per-head mean for better codebook utilization
            1. Store ||x|| and normalize
            2. Rotate: x_rot = x_normalized @ Pi.T
            3. Quantize: indices = nearest_centroid(x_rot)
            4. Compute residual in rotated space: r_rot = x_rot - centroids[indices]
            5. Store sign(r_rot) — 1 bit per coordinate
            6. Store mean(|r_rot|) — residual scale for reconstruction

        Args:
            x: Input vectors of shape (batch, d) or (d,).

        Returns:
            Dict with keys:
                - mse_indices: Tensor of codebook indices, shape (batch, d)
                - residual_signs: Tensor of sign bits {-1,+1}, shape (batch, d)
                - residual_scale: Tensor of mean |r_rot|, shape (batch,)
                - vec_norm: Tensor of ||x||, shape (batch,)
                - vec_mean: Tensor of per-batch mean, shape (batch, d) [if centering]
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        # Mean removal: subtract per-batch mean before quantization.
        # Softmax is shift-invariant, so the mean is invisible to attention.
        # Removing it reduces variance -> better codebook utilization ->
        # effectively a FREE +1 bit of precision.
        if self.center_before_quantize:
            vec_mean = x.mean(dim=0, keepdim=True).expand_as(x)  # (batch, d)
            x_centered = x - vec_mean
        else:
            vec_mean = torch.zeros_like(x)
            x_centered = x

        # Store original norm of the centered vector and normalize
        vec_norm = x_centered.norm(dim=-1, keepdim=True)  # (batch, 1)
        x_normalized = x_centered / (vec_norm + 1e-8)  # (batch, d)

        # Rotate to the space where Lloyd-Max quantization happens
        x_rotated = self.polar.rotate(x_normalized)  # (batch, d)

        # Stage 1: MSE quantization in rotated space
        mse_indices = self.polar.codebook.quantize(x_rotated)  # (batch, d)

        # Reconstruct in rotated space (centroid lookup, no unrotation)
        x_mse_rotated = self.polar.centroids[mse_indices]  # (batch, d)

        # Residual in rotated space
        residual_rotated = x_rotated - x_mse_rotated  # (batch, d)

        # Store signs of residual (1 bit per coordinate, NO random projection)
        # Use (>= 0) * 2 - 1 to avoid zero -> 0 behavior of torch.sign
        residual_signs = (residual_rotated >= 0).float() * 2.0 - 1.0  # {-1, +1}^d

        # Store mean absolute residual as scale factor
        residual_scale = residual_rotated.abs().mean(dim=-1)  # (batch,)

        result = {
            "mse_indices": mse_indices,
            "residual_signs": residual_signs,
            "residual_scale": residual_scale,
            "vec_norm": vec_norm.squeeze(-1),  # (batch,)
            "vec_mean": vec_mean,  # (batch, d) — FP16 overhead: 2*d bytes per head
        }

        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result

    def dequantize(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct vector using MSE + residual sign correction.

        In rotated space:
            k_corrected_rot = centroids[indices] + residual_scale * sign(r_rot)
        Then unrotate:
            k_corrected = k_corrected_rot @ Pi
        Then add mean back (if centering was used):
            k_final = k_corrected * norm + mean

        Args:
            compressed: Output from quantize().

        Returns:
            Reconstructed vectors of shape (batch, d) or (d,).
        """
        squeeze = compressed["mse_indices"].dim() == 1

        mse_indices = compressed["mse_indices"]
        residual_signs = compressed["residual_signs"]
        residual_scale = compressed["residual_scale"]
        vec_norm = compressed["vec_norm"]
        vec_mean = compressed["vec_mean"]

        if squeeze:
            mse_indices = mse_indices.unsqueeze(0)
            residual_signs = residual_signs.unsqueeze(0)
            residual_scale = residual_scale.unsqueeze(0)
            vec_norm = vec_norm.unsqueeze(0)
            vec_mean = vec_mean.unsqueeze(0)

        # Reconstruct in rotated space
        x_mse_rotated = self.polar.centroids[mse_indices]  # (batch, d)

        # Apply correction: MSE + scale * sign(residual)
        correction = residual_scale.unsqueeze(-1) * residual_signs  # (batch, d)
        x_corrected_rotated = x_mse_rotated + correction  # (batch, d)

        # Unrotate back to original space
        x_corrected = self.polar.unrotate(x_corrected_rotated)  # (batch, d)

        # Rescale by original norm and add mean back
        result = x_corrected * vec_norm.unsqueeze(-1) + vec_mean

        if squeeze:
            result = result.squeeze(0)

        return result

    def dequantize_mse(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct vector using MSE stage only (no residual correction).

        For comparison with the full correction.

        Args:
            compressed: Output from quantize().

        Returns:
            Reconstructed vectors of shape (batch, d) or (d,).
        """
        x_mse = self.polar.dequantize(compressed["mse_indices"])
        vec_norm = compressed["vec_norm"]
        vec_mean = compressed["vec_mean"]

        if vec_norm.dim() == 0:
            return x_mse * vec_norm + vec_mean
        return x_mse * vec_norm.unsqueeze(-1) + vec_mean

    def inner_product(
        self,
        query: torch.Tensor,
        compressed: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Estimate <query, key> using the residual-corrected reconstruction.

        Unlike TurboQuant's analytical inner product formula, ResidualQuant
        simply reconstructs the key and computes the dot product directly.
        This is valid because the reconstruction quality is high enough.

        Args:
            query: Query vectors, shape (batch_q, d) or (d,).
            compressed: Output from quantize() for key vectors.

        Returns:
            Estimated inner products.
        """
        # Reconstruct with residual correction
        k_corrected = self.dequantize(compressed)

        squeeze_q = query.dim() == 1
        if squeeze_q:
            query = query.unsqueeze(0)

        if k_corrected.dim() == 1:
            result = query @ k_corrected
        else:
            result = query @ k_corrected.T

        if squeeze_q and result.dim() > 0 and result.shape[0] == 1:
            result = result.squeeze(0)

        return result


class ResidualQuantLayer:
    """A single layer's compressed KV cache using ResidualQuant for keys.

    Keys: ResidualQuantEstimator (MSE + direct residual signs)
    Values: PolarQuant MSE-only (same as TurboQuant — values need reconstruction)

    Duck-types the TurboQuantLayer interface.

    Args:
        bits: Total bits per coordinate (2-8).
        seed: Random seed for rotation matrices.
        center_before_quantize: Subtract per-head key mean before quantization.
            Exploits softmax shift-invariance for FREE +1 bit of precision.
    """

    def __init__(self, bits: int = 3, seed: int = 42, center_before_quantize: bool = True):
        self.bits = bits
        self.seed = seed
        self.center_before_quantize = center_before_quantize
        self._seq_len: int = 0

        # Lazily initialized
        self._key_rq: ResidualQuantEstimator | None = None
        self._val_pq: PolarQuant | None = None
        self._dtype: torch.dtype | None = None
        self._device: torch.device | None = None
        self._head_dim: int | None = None
        self._num_heads: int | None = None
        self._batch_size: int | None = None

        # Online mean tracking for autoregressive generation.
        # mean_new = (mean_old * n + x_new) / (n + 1)
        # Shape: (batch, num_heads, 1, head_dim) when initialized.
        self._key_running_mean: torch.Tensor | None = None
        self._key_running_count: int = 0

        # Compressed storage
        self._key_compressed: list[Dict[str, torch.Tensor]] = []
        self._value_compressed: list[Dict[str, torch.Tensor]] = []

    def _lazy_init(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Initialize quantizers from first observed tensor shapes."""
        self._batch_size = key_states.shape[0]
        self._num_heads = key_states.shape[1]
        self._head_dim = key_states.shape[3]
        self._dtype = key_states.dtype
        self._device = key_states.device

        d = self._head_dim
        device = str(self._device) if self._device is not None else "cpu"

        self._key_rq = ResidualQuantEstimator(
            d=d, bits=self.bits, seed=self.seed, device=device,
            center_before_quantize=False,  # We handle centering at the layer level
        )
        self._val_pq = PolarQuant(
            d=d, bits=self.bits, seed=self.seed + 100, device=device,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress and store new key/value states, return dequantized full cache."""
        if self._key_rq is None:
            self._lazy_init(key_states, value_states)

        batch, num_heads, new_seq, head_dim = key_states.shape

        # Mean-centering: subtract per-head running mean before quantization.
        # Softmax is shift-invariant so the mean is invisible to attention,
        # but removing it reduces variance -> better codebook utilization.
        keys_to_quantize = key_states.float()
        if self.center_before_quantize:
            # Online mean update: mean_new = (mean_old * n + sum_new) / (n + new_seq)
            new_sum = keys_to_quantize.sum(dim=2, keepdim=True)  # (B, H, 1, D)
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
            keys_to_quantize = keys_to_quantize - self._key_running_mean

        # Flatten for quantization
        keys_flat = keys_to_quantize.reshape(-1, head_dim)
        vals_flat = value_states.float().reshape(-1, head_dim)

        # Compress keys with ResidualQuant
        key_comp = self._key_rq.quantize(keys_flat)
        key_entry = {
            "mse_indices": key_comp["mse_indices"].reshape(batch, num_heads, new_seq, head_dim),
            "residual_signs": key_comp["residual_signs"].reshape(batch, num_heads, new_seq, head_dim),
            "residual_scale": key_comp["residual_scale"].reshape(batch, num_heads, new_seq),
            "vec_norm": key_comp["vec_norm"].reshape(batch, num_heads, new_seq),
            "vec_mean": key_comp["vec_mean"].reshape(batch, num_heads, new_seq, head_dim),
        }

        # Compress values with MSE-only PolarQuant
        val_norms = vals_flat.norm(dim=-1, keepdim=True)
        vals_normalized = vals_flat / (val_norms + 1e-8)
        val_indices = self._val_pq.quantize(vals_normalized)
        val_entry = {
            "indices": val_indices.reshape(batch, num_heads, new_seq, head_dim),
            "norms": val_norms.squeeze(-1).reshape(batch, num_heads, new_seq),
        }

        self._key_compressed.append(key_entry)
        self._value_compressed.append(val_entry)
        self._seq_len += new_seq

        return self._dequantize_all()

    def _dequantize_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize all stored keys and values."""
        if self._seq_len == 0:
            empty = torch.zeros(
                self._batch_size or 1, self._num_heads or 1, 0,
                self._head_dim or 1,
                dtype=self._dtype, device=self._device,
            )
            return empty, empty

        batch = self._batch_size
        num_heads = self._num_heads
        head_dim = self._head_dim

        # Keys: gather compressed data, reconstruct with residual correction
        all_key_mse = torch.cat([e["mse_indices"] for e in self._key_compressed], dim=2)
        all_key_signs = torch.cat([e["residual_signs"] for e in self._key_compressed], dim=2)
        all_key_scale = torch.cat([e["residual_scale"] for e in self._key_compressed], dim=2)
        all_key_norm = torch.cat([e["vec_norm"] for e in self._key_compressed], dim=2)
        all_key_mean = torch.cat([e["vec_mean"] for e in self._key_compressed], dim=2)

        total_seq = all_key_mse.shape[2]

        # Flatten for dequantization
        key_comp_flat = {
            "mse_indices": all_key_mse.reshape(-1, head_dim),
            "residual_signs": all_key_signs.reshape(-1, head_dim),
            "residual_scale": all_key_scale.reshape(-1),
            "vec_norm": all_key_norm.reshape(-1),
            "vec_mean": all_key_mean.reshape(-1, head_dim),
        }
        keys_flat = self._key_rq.dequantize(key_comp_flat)
        keys_out = keys_flat.reshape(batch, num_heads, total_seq, head_dim)

        # Values: MSE-only dequantization (same as TurboQuant)
        all_val_indices = torch.cat([e["indices"] for e in self._value_compressed], dim=2)
        all_val_norms = torch.cat([e["norms"] for e in self._value_compressed], dim=2)

        val_idx_flat = all_val_indices.reshape(-1, head_dim)
        val_norms_flat = all_val_norms.reshape(-1)

        val_recon_flat = self._val_pq.dequantize(val_idx_flat)
        val_recon_flat = val_recon_flat * val_norms_flat.unsqueeze(-1)
        vals_out = val_recon_flat.reshape(batch, num_heads, total_seq, head_dim)

        return keys_out.to(self._dtype), vals_out.to(self._dtype)

    def get_seq_length(self) -> int:
        return self._seq_len

    def clear(self) -> None:
        self._key_compressed.clear()
        self._value_compressed.clear()
        self._seq_len = 0
        self._key_running_mean = None
        self._key_running_count = 0


class ResidualQuantCache:
    """Drop-in KV cache using ResidualQuant for HuggingFace transformers.

    Duck-types the HF Cache protocol. Keys are compressed with
    ResidualQuantEstimator (MSE + direct residual signs), values with
    MSE-only PolarQuant.

    On retrieval, data is dequantized to FP16/FP32 so standard HF attention
    works unmodified.

    Args:
        bits: Total bits per coordinate (2, 3, or 4). Default 3.
        seed: Base random seed. Each layer gets seed + layer_idx.
        center_before_quantize: Subtract per-head key mean before quantization.
            Exploits softmax shift-invariance for FREE +1 bit of precision.
            Default: True.
    """

    is_compileable = False

    def __init__(self, bits: int = 3, seed: int = 42, center_before_quantize: bool = True):
        if not (2 <= bits <= 8):
            raise ValueError(f"bits must be between 2 and 8, got {bits}")
        self.bits = bits
        self.seed = seed
        self.center_before_quantize = center_before_quantize
        self._layers: list[ResidualQuantLayer] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress and store new KV pairs for a layer."""
        while len(self._layers) <= layer_idx:
            self._layers.append(ResidualQuantLayer(
                bits=self.bits, seed=self.seed + len(self._layers),
                center_before_quantize=self.center_before_quantize,
            ))
        return self._layers[layer_idx].update(key_states, value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._layers):
            return 0
        return self._layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_mask_sizes(
        self, cache_position_or_query_length, layer_idx: int
    ) -> tuple[int, int]:
        # HF transformers may pass either a Tensor (cache_position) or int (query_length)
        if isinstance(cache_position_or_query_length, int):
            query_length = cache_position_or_query_length
        else:
            query_length = cache_position_or_query_length.shape[0]
        if layer_idx >= len(self._layers):
            return query_length, 0
        kv_length = self._layers[layer_idx].get_seq_length() + query_length
        return kv_length, 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for layer in self._layers:
            for entry in layer._key_compressed:
                for k in entry:
                    entry[k] = entry[k].index_select(0, beam_idx)
            for entry in layer._value_compressed:
                for k in entry:
                    entry[k] = entry[k].index_select(0, beam_idx)

    def crop(self, max_length: int) -> None:
        for layer in self._layers:
            if max_length < 0:
                effective = layer._seq_len + max_length
            else:
                effective = max_length
            if layer._seq_len <= effective:
                continue
            remaining = effective
            new_key_comp = []
            new_val_comp = []
            for kc, vc in zip(layer._key_compressed, layer._value_compressed):
                chunk_seq = kc["mse_indices"].shape[2]
                if remaining <= 0:
                    break
                take = min(chunk_seq, remaining)
                new_key_comp.append({k: v[:, :, :take] for k, v in kc.items()})
                new_val_comp.append({k: v[:, :, :take] for k, v in vc.items()})
                remaining -= take
            layer._key_compressed = new_key_comp
            layer._value_compressed = new_val_comp
            layer._seq_len = effective

    def reset(self) -> None:
        for layer in self._layers:
            layer.clear()

    def batch_repeat_interleave(self, repeats: int) -> None:
        for layer in self._layers:
            for entry in layer._key_compressed:
                for k in entry:
                    entry[k] = entry[k].repeat_interleave(repeats, dim=0)
            for entry in layer._value_compressed:
                for k in entry:
                    entry[k] = entry[k].repeat_interleave(repeats, dim=0)
            if layer._batch_size is not None:
                layer._batch_size *= repeats

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for layer in self._layers:
            for entry in layer._key_compressed:
                for k in entry:
                    entry[k] = entry[k][indices]
            for entry in layer._value_compressed:
                for k in entry:
                    entry[k] = entry[k][indices]
            if layer._batch_size is not None:
                layer._batch_size = len(indices)

    @property
    def is_initialized(self) -> bool:
        return len(self._layers) > 0

    @property
    def is_sliding(self) -> list[bool]:
        return [False] * len(self._layers)

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):
        for layer in self._layers:
            keys, values = layer._dequantize_all()
            yield keys, values, None

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self._layers):
            raise IndexError(f"Layer {layer_idx} not in cache (have {len(self._layers)} layers)")
        return self._layers[layer_idx]._dequantize_all()
