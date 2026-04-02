"""Residual Vector Quantization (2-stage RVQ).

Two-stage cascaded scalar quantization: stage 1 captures the bulk of the
signal, stage 2 quantizes the residual error with a codebook optimized for
the residual distribution.

Stage 1: Quantize with b1-bit Lloyd-Max codebook -> indices1, reconstruct x_hat1
Stage 2: Compute residual r = x - x_hat1, quantize r with SEPARATE b2-bit codebook
          optimized for the residual distribution -> indices2, reconstruct r_hat
Final reconstruction: x_final = x_hat1 + r_hat

The residual distribution after Lloyd-Max quantization of N(0, 1/d) is different
from the original -- it's concentrated near zero with lighter tails. The stage-2
codebook is optimized for this distribution by modeling it as N(0, sigma_residual)
where sigma_residual = sqrt(MSE_distortion_of_stage1).

Trade-offs vs single-stage quantization:
    For per-coordinate SCALAR Lloyd-Max, a single-stage quantizer with 2^b
    optimally-placed levels is better than two cascaded stages at the same
    total bit budget (well-known result in quantization theory). However,
    RVQ provides several practical benefits:

    1. Stage 2 significantly improves over stage 1 alone (~60% MSE reduction)
    2. Asymmetric bit allocation (e.g. 3+1 closely approaches K4 quality)
    3. Finer-grained bit-rate control without recomputing codebooks
    4. Foundation for future learned/vector codebook extensions where
       multi-stage DOES beat single-stage

Storage per vector: (stage1_bits + stage2_bits) * d bits for indices + 32 bits
for norm (same overhead structure as PolarQuant).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .codebook import LloydMaxCodebook, gaussian_pdf, solve_lloyd_max
from .polarquant import PolarQuant
from .rotation import generate_rotation_matrix


class ResidualVQ(nn.Module):
    """Two-stage Residual Vector Quantization.

    Stage 1: Lloyd-Max quantize at stage1_bits (captures bulk of signal)
    Stage 2: Lloyd-Max quantize the residual at stage2_bits (captures error)
    Total bits = stage1_bits + stage2_bits

    Both stages operate in rotated space (after random orthogonal rotation),
    sharing the same rotation matrix. The stage-2 codebook is optimized for
    the residual distribution N(0, sigma_residual) rather than the original
    N(0, 1/d).

    Args:
        d: Head dimension (e.g. 128).
        stage1_bits: Bits per coordinate for stage 1 (1-8).
        stage2_bits: Bits per coordinate for stage 2 (1-8).
        seed: Random seed for rotation matrix generation.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        stage1_bits: int = 2,
        stage2_bits: int = 2,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.d = d
        self.stage1_bits = stage1_bits
        self.stage2_bits = stage2_bits
        self.total_bits = stage1_bits + stage2_bits

        # Stage 1: PolarQuant (rotation + Lloyd-Max quantization)
        self.stage1 = PolarQuant(d, bits=stage1_bits, seed=seed, device=device)

        # Stage 2: separate codebook for the residual distribution
        # The residual after Lloyd-Max quantization of N(0, 1/d) has variance
        # equal to the per-coordinate MSE distortion of stage 1.
        self.stage2_codebook = self._build_residual_codebook(d, stage1_bits, stage2_bits)
        self.register_buffer(
            "stage2_centroids",
            self.stage2_codebook.centroids.to(device),
        )

    def _build_residual_codebook(
        self, d: int, stage1_bits: int, stage2_bits: int,
    ) -> LloydMaxCodebook:
        """Build optimal codebook for the residual distribution.

        After stage-1 Lloyd-Max quantization of coordinates drawn from N(0, 1/d),
        the residual r = x - x_hat has per-coordinate variance equal to the
        stage-1 MSE distortion. We model the residual as N(0, sigma_residual^2)
        and solve Lloyd-Max for that distribution.

        Since gaussian_pdf uses sigma^2 = 1/d_eff, we compute the effective
        dimension d_eff = 1/sigma_residual^2 so that N(0, 1/d_eff) matches the
        residual distribution.

        Returns:
            LloydMaxCodebook optimized for the residual distribution.
        """
        stage1_cb = LloydMaxCodebook(d, stage1_bits)
        mse_per_coord = stage1_cb.compute_distortion()

        # sigma_residual^2 = mse_per_coord, so d_eff = 1/mse_per_coord
        # Clamp to avoid division by zero / extreme values
        sigma_sq = max(mse_per_coord, 1e-12)
        d_eff = 1.0 / sigma_sq

        # d_eff must be a positive integer for LloydMaxCodebook
        # Round to nearest integer, minimum 2 to keep distribution well-behaved
        d_eff_int = max(int(round(d_eff)), 2)

        return LloydMaxCodebook(d_eff_int, stage2_bits)

    def quantize(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Two-stage quantization.

        Algorithm:
            1. Store ||x|| and normalize to unit sphere
            2. Rotate: x_rot = x_norm @ Pi.T
            3. Stage 1: quantize x_rot -> indices1, reconstruct x_hat1_rot
            4. Residual in rotated space: r_rot = x_rot - x_hat1_rot
            5. Stage 2: quantize r_rot with residual codebook -> indices2

        Args:
            x: Input vectors of shape (batch, d) or (d,).

        Returns:
            Dict with keys:
                - stage1_indices: Tensor[batch, d] in [0, 2^stage1_bits)
                - stage2_indices: Tensor[batch, d] in [0, 2^stage2_bits)
                - vec_norm: Tensor[batch] of ||x||
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        # Store norm and normalize
        vec_norm = x.norm(dim=-1, keepdim=True)  # (batch, 1)
        x_normalized = x / (vec_norm + 1e-8)  # (batch, d)

        # Rotate to quantization space
        x_rotated = self.stage1.rotate(x_normalized)  # (batch, d)

        # Stage 1: quantize in rotated space
        stage1_indices = self.stage1.codebook.quantize(x_rotated)  # (batch, d)
        x_hat1_rotated = self.stage1.centroids[stage1_indices]  # (batch, d)

        # Residual in rotated space
        residual_rotated = x_rotated - x_hat1_rotated  # (batch, d)

        # Stage 2: quantize residual with the residual-optimized codebook
        stage2_centroids = self.stage2_centroids.to(residual_rotated.device)
        stage2_indices = self.stage2_codebook.quantize(residual_rotated)  # (batch, d)

        result = {
            "stage1_indices": stage1_indices,
            "stage2_indices": stage2_indices,
            "vec_norm": vec_norm.squeeze(-1),
        }

        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result

    def dequantize(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct from two-stage indices.

        Algorithm:
            1. Stage 1 reconstruction: x_hat1_rot = centroids1[indices1]
            2. Stage 2 reconstruction: r_hat_rot = centroids2[indices2]
            3. Combined rotated: x_combined_rot = x_hat1_rot + r_hat_rot
            4. Unrotate: x_combined = x_combined_rot @ Pi
            5. Rescale by stored norm

        Args:
            compressed: Output from quantize().

        Returns:
            Reconstructed vectors of shape (batch, d) or (d,).
        """
        squeeze = compressed["stage1_indices"].dim() == 1

        stage1_indices = compressed["stage1_indices"]
        stage2_indices = compressed["stage2_indices"]
        vec_norm = compressed["vec_norm"]

        if squeeze:
            stage1_indices = stage1_indices.unsqueeze(0)
            stage2_indices = stage2_indices.unsqueeze(0)
            vec_norm = vec_norm.unsqueeze(0)

        # Stage 1 reconstruction in rotated space
        x_hat1_rotated = self.stage1.centroids[stage1_indices]  # (batch, d)

        # Stage 2 reconstruction in rotated space
        r_hat_rotated = self.stage2_centroids[stage2_indices]  # (batch, d)

        # Combined reconstruction in rotated space
        x_combined_rotated = x_hat1_rotated + r_hat_rotated  # (batch, d)

        # Unrotate
        x_combined = self.stage1.unrotate(x_combined_rotated)  # (batch, d)

        # Rescale by stored norm
        result = x_combined * vec_norm.unsqueeze(-1)

        if squeeze:
            result = result.squeeze(0)

        return result

    def dequantize_stage1_only(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct using only stage 1 (for ablation/comparison).

        Args:
            compressed: Output from quantize().

        Returns:
            Reconstructed vectors using stage 1 only.
        """
        squeeze = compressed["stage1_indices"].dim() == 1

        stage1_indices = compressed["stage1_indices"]
        vec_norm = compressed["vec_norm"]

        if squeeze:
            stage1_indices = stage1_indices.unsqueeze(0)
            vec_norm = vec_norm.unsqueeze(0)

        x_hat1_rotated = self.stage1.centroids[stage1_indices]
        x_hat1 = self.stage1.unrotate(x_hat1_rotated)
        result = x_hat1 * vec_norm.unsqueeze(-1)

        if squeeze:
            result = result.squeeze(0)

        return result

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize and immediately dequantize (for evaluation).

        Args:
            x: Input vectors of shape (batch, d) or (d,).

        Returns:
            Tuple of (x_hat: reconstructed vectors, compressed: dict of indices).
        """
        compressed = self.quantize(x)
        x_hat = self.dequantize(compressed)
        return x_hat, compressed


# ---------------------------------------------------------------------------
# RVQ Cache Layer
# ---------------------------------------------------------------------------


class ResidualVQLayer:
    """A single layer's compressed KV cache using ResidualVQ for keys.

    Keys: ResidualVQ (2-stage, e.g. 2+2 = 4 bits total)
    Values: PolarQuant MSE-only (e.g. 2-3 bits)

    Duck-types the same interface as ResidualQuantLayer.
    """

    def __init__(
        self,
        key_stage1_bits: int = 2,
        key_stage2_bits: int = 2,
        value_bits: int = 2,
        seed: int = 42,
        fp16_window: int = 0,
        anchor_interval: int = 0,
    ):
        self.key_stage1_bits = key_stage1_bits
        self.key_stage2_bits = key_stage2_bits
        self.value_bits = value_bits
        self.seed = seed
        self.fp16_window = fp16_window
        self.anchor_interval = anchor_interval
        self._seq_len: int = 0

        # Lazily initialized
        self._key_rvq: ResidualVQ | None = None
        self._val_pq: PolarQuant | None = None
        self._dtype: torch.dtype | None = None
        self._device: torch.device | None = None
        self._head_dim: int | None = None
        self._num_heads: int | None = None
        self._batch_size: int | None = None

        # Compressed storage
        self._key_compressed: List[Dict[str, torch.Tensor]] = []
        self._value_compressed: List[Dict[str, torch.Tensor]] = []

        # FP16 window storage
        self._fp16_keys: List[torch.Tensor] = []
        self._fp16_values: List[torch.Tensor] = []
        self._fp16_len: int = 0

        # Anchor storage (full precision periodic tokens)
        self._anchor_keys: List[torch.Tensor] = []
        self._anchor_values: List[torch.Tensor] = []
        self._anchor_positions: List[int] = []

    def _lazy_init(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Initialize quantizers from first observed tensor shapes."""
        self._batch_size = key_states.shape[0]
        self._num_heads = key_states.shape[1]
        self._head_dim = key_states.shape[3]
        self._dtype = key_states.dtype
        self._device = key_states.device

        d = self._head_dim
        device = str(self._device) if self._device is not None else "cpu"

        self._key_rvq = ResidualVQ(
            d=d,
            stage1_bits=self.key_stage1_bits,
            stage2_bits=self.key_stage2_bits,
            seed=self.seed,
            device=device,
        )
        self._val_pq = PolarQuant(
            d=d,
            bits=self.value_bits,
            seed=self.seed + 100,
            device=device,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress and store new key/value states, return dequantized full cache."""
        if self._key_rvq is None:
            self._lazy_init(key_states, value_states)

        batch, num_heads, new_seq, head_dim = key_states.shape

        for t in range(new_seq):
            pos = self._seq_len + t
            k_slice = key_states[:, :, t:t+1, :]  # (B, H, 1, D)
            v_slice = value_states[:, :, t:t+1, :]

            # Check if this is an anchor token
            if self.anchor_interval > 0 and pos % self.anchor_interval == 0:
                self._anchor_keys.append(k_slice)
                self._anchor_values.append(v_slice)
                self._anchor_positions.append(pos)
                self._seq_len += 1
                continue

            # Flatten for quantization: (B*H, D)
            k_flat = k_slice.float().reshape(-1, head_dim)
            v_flat = v_slice.float().reshape(-1, head_dim)

            # Compress keys with ResidualVQ
            key_comp = self._key_rvq.quantize(k_flat)
            key_entry = {
                "stage1_indices": key_comp["stage1_indices"].reshape(batch, num_heads, 1, head_dim),
                "stage2_indices": key_comp["stage2_indices"].reshape(batch, num_heads, 1, head_dim),
                "vec_norm": key_comp["vec_norm"].reshape(batch, num_heads, 1),
            }

            # Compress values with MSE-only PolarQuant
            val_norms = v_flat.norm(dim=-1, keepdim=True)
            v_normalized = v_flat / (val_norms + 1e-8)
            val_indices = self._val_pq.quantize(v_normalized)
            val_entry = {
                "indices": val_indices.reshape(batch, num_heads, 1, head_dim),
                "norms": val_norms.squeeze(-1).reshape(batch, num_heads, 1),
            }

            self._key_compressed.append(key_entry)
            self._value_compressed.append(val_entry)

        self._seq_len += new_seq - len([
            p for p in range(self._seq_len, self._seq_len + new_seq)
            if self.anchor_interval > 0 and p % self.anchor_interval == 0
        ])

        # Maintain FP16 window
        if self.fp16_window > 0:
            self._fp16_keys.append(key_states)
            self._fp16_values.append(value_states)
            self._fp16_len += new_seq
            self._trim_fp16_window()

        return self._dequantize_all()

    def _trim_fp16_window(self) -> None:
        """Keep only the last fp16_window tokens in the FP16 buffers."""
        while self._fp16_len > self.fp16_window and self._fp16_keys:
            first_seq = self._fp16_keys[0].shape[2]
            if self._fp16_len - first_seq >= self.fp16_window:
                self._fp16_keys.pop(0)
                self._fp16_values.pop(0)
                self._fp16_len -= first_seq
            else:
                trim = self._fp16_len - self.fp16_window
                self._fp16_keys[0] = self._fp16_keys[0][:, :, trim:]
                self._fp16_values[0] = self._fp16_values[0][:, :, trim:]
                self._fp16_len = self.fp16_window
                break

    def _dequantize_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize all stored keys and values."""
        batch = self._batch_size
        num_heads = self._num_heads
        head_dim = self._head_dim

        parts_k: List[torch.Tensor] = []
        parts_v: List[torch.Tensor] = []

        # Dequantize compressed tokens
        if self._key_compressed:
            all_s1 = torch.cat([e["stage1_indices"] for e in self._key_compressed], dim=2)
            all_s2 = torch.cat([e["stage2_indices"] for e in self._key_compressed], dim=2)
            all_kn = torch.cat([e["vec_norm"] for e in self._key_compressed], dim=2)

            comp_flat = {
                "stage1_indices": all_s1.reshape(-1, head_dim),
                "stage2_indices": all_s2.reshape(-1, head_dim),
                "vec_norm": all_kn.reshape(-1),
            }
            keys_flat = self._key_rvq.dequantize(comp_flat)
            comp_seq = all_s1.shape[2]
            parts_k.append(keys_flat.reshape(batch, num_heads, comp_seq, head_dim))

            # Values
            all_vi = torch.cat([e["indices"] for e in self._value_compressed], dim=2)
            all_vn = torch.cat([e["norms"] for e in self._value_compressed], dim=2)
            vi_flat = all_vi.reshape(-1, head_dim)
            vn_flat = all_vn.reshape(-1)
            v_recon = self._val_pq.dequantize(vi_flat)
            v_recon = v_recon * vn_flat.unsqueeze(-1)
            parts_v.append(v_recon.reshape(batch, num_heads, comp_seq, head_dim))

        # Add anchor tokens
        if self._anchor_keys:
            parts_k.extend(self._anchor_keys)
            parts_v.extend(self._anchor_values)

        # Add FP16 window tokens
        if self.fp16_window > 0 and self._fp16_keys:
            fp16_k = torch.cat(self._fp16_keys, dim=2)
            fp16_v = torch.cat(self._fp16_values, dim=2)
            parts_k.append(fp16_k)
            parts_v.append(fp16_v)

        if not parts_k:
            empty = torch.zeros(
                batch or 1, num_heads or 1, 0, head_dim or 1,
                dtype=self._dtype, device=self._device,
            )
            return empty, empty

        keys_out = torch.cat(parts_k, dim=2).to(self._dtype)
        vals_out = torch.cat(parts_v, dim=2).to(self._dtype)
        return keys_out, vals_out

    def get_seq_length(self) -> int:
        total = len(self._key_compressed) + len(self._anchor_keys)
        if self.fp16_window > 0:
            total += self._fp16_len
        return total

    def clear(self) -> None:
        self._key_compressed.clear()
        self._value_compressed.clear()
        self._fp16_keys.clear()
        self._fp16_values.clear()
        self._anchor_keys.clear()
        self._anchor_values.clear()
        self._anchor_positions.clear()
        self._seq_len = 0
        self._fp16_len = 0


# ---------------------------------------------------------------------------
# RVQ Cache (HF-compatible)
# ---------------------------------------------------------------------------


class ResidualVQCache:
    """Drop-in KV cache using 2-stage ResidualVQ for HuggingFace transformers.

    Duck-types the HF Cache protocol. Keys are compressed with ResidualVQ
    (2-stage Lloyd-Max), values with MSE-only PolarQuant.

    On retrieval, data is dequantized to FP16/FP32 so standard HF attention
    works unmodified.

    Args:
        key_stage1_bits: Bits per coordinate for key stage 1. Default 2.
        key_stage2_bits: Bits per coordinate for key stage 2. Default 2.
        value_bits: Bits per coordinate for values. Default 2.
        seed: Base random seed. Each layer gets seed + layer_idx.
        fp16_window: Number of recent tokens stored in full precision (0=off).
        anchor_interval: Store every Nth token at full precision (0=off).
    """

    is_compileable = False

    def __init__(
        self,
        key_stage1_bits: int = 2,
        key_stage2_bits: int = 2,
        value_bits: int = 2,
        seed: int = 42,
        fp16_window: int = 0,
        anchor_interval: int = 0,
    ):
        if not (1 <= key_stage1_bits <= 8):
            raise ValueError(f"key_stage1_bits must be 1-8, got {key_stage1_bits}")
        if not (1 <= key_stage2_bits <= 8):
            raise ValueError(f"key_stage2_bits must be 1-8, got {key_stage2_bits}")
        if not (1 <= value_bits <= 8):
            raise ValueError(f"value_bits must be 1-8, got {value_bits}")

        self.key_stage1_bits = key_stage1_bits
        self.key_stage2_bits = key_stage2_bits
        self.value_bits = value_bits
        self.seed = seed
        self.fp16_window = fp16_window
        self.anchor_interval = anchor_interval
        self._layers: List[ResidualVQLayer] = []

    @property
    def total_key_bits(self) -> int:
        return self.key_stage1_bits + self.key_stage2_bits

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Dict | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress and store new KV pairs for a layer."""
        while len(self._layers) <= layer_idx:
            self._layers.append(ResidualVQLayer(
                key_stage1_bits=self.key_stage1_bits,
                key_stage2_bits=self.key_stage2_bits,
                value_bits=self.value_bits,
                seed=self.seed + len(self._layers),
                fp16_window=self.fp16_window,
                anchor_interval=self.anchor_interval,
            ))
        return self._layers[layer_idx].update(key_states, value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._layers):
            return 0
        return self._layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int,
    ) -> Tuple[int, int]:
        if layer_idx >= len(self._layers):
            return cache_position.shape[0], 0
        query_length = cache_position.shape[0]
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
            total = len(layer._key_compressed)
            if max_length < 0:
                effective = total + max_length
            else:
                effective = max_length
            if total <= effective:
                continue
            layer._key_compressed = layer._key_compressed[:effective]
            layer._value_compressed = layer._value_compressed[:effective]

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
    def is_sliding(self) -> list:
        return [False] * len(self._layers)

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):
        for layer in self._layers:
            keys, values = layer._dequantize_all()
            yield keys, values, None

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self._layers):
            raise IndexError(f"Layer {layer_idx} not in cache (have {len(self._layers)} layers)")
        return self._layers[layer_idx]._dequantize_all()
