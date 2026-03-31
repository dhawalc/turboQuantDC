"""UltimateCache -- the full TurboQuantDC compression stack for generation.

Combines all breakthroughs discovered during the project:
1. FP16 anchor layers (every anchor_interval layers) -- breaks error accumulation
2. ResidualQuant for compressed layers' keys (direct signs, not QJL)
3. Asymmetric K/V (keys at key_bits, values at val_bits)
4. Residual windowing (last fp16_window tokens stored at FP16)

Duck-types the HF Cache protocol for model.generate().

Usage:
    from turboquantdc.ultimate_cache import UltimateCache
    cache = UltimateCache(
        num_layers=36,
        key_bits=4, val_bits=2,
        anchor_interval=12,
        fp16_window=128,
    )
    output = model.generate(inputs, past_key_values=cache, max_new_tokens=100)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .polarquant import PolarQuant
from .residual_quant import ResidualQuantEstimator


# ---------------------------------------------------------------------------
# Per-layer implementations
# ---------------------------------------------------------------------------

class FP16Layer:
    """FP16 anchor layer -- no compression at all."""

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
            return torch.zeros(1, 1, 0, 1), torch.zeros(1, 1, 0, 1)
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
                "key_bits": 0, "value_bits": 0, "total_bits": 0,
                "fp16_baseline_bits": 0, "compression_ratio": 1.0,
            }
        k = self._keys[0]
        total_tokens = self._seq_len * k.shape[1] * k.shape[0]
        head_dim = k.shape[3]
        total = total_tokens * head_dim * 16 * 2
        return {
            "key_bits": total // 2, "value_bits": total // 2,
            "total_bits": total, "fp16_baseline_bits": total,
            "compression_ratio": 1.0,
        }


class AsymmetricCompressedLayer:
    """Compressed layer with asymmetric K/V bit-widths and optional residual window.

    Keys: either ResidualQuant (MSE + residual signs) or MSE-only PolarQuant
    Values: MSE-only PolarQuant at val_bits

    If fp16_window > 0, the last fp16_window tokens are stored at FP16 and
    merged with compressed tokens on dequantization.
    """

    def __init__(
        self,
        key_bits: int = 4,
        val_bits: int = 2,
        use_residual_quant: bool = False,
        fp16_window: int = 0,
        seed: int = 42,
    ):
        self.key_bits = key_bits
        self.val_bits = val_bits
        self.use_residual_quant = use_residual_quant
        self.fp16_window = fp16_window
        self.seed = seed
        self._seq_len: int = 0

        # Lazily initialized
        self._key_rq: Optional[ResidualQuantEstimator] = None
        self._key_pq: Optional[PolarQuant] = None
        self._val_pq: Optional[PolarQuant] = None
        self._dtype: Optional[torch.dtype] = None
        self._device: Optional[torch.device] = None
        self._head_dim: Optional[int] = None
        self._num_heads: Optional[int] = None
        self._batch_size: Optional[int] = None

        # Compressed storage
        self._key_compressed: List[Dict[str, torch.Tensor]] = []
        self._value_compressed: List[Dict[str, torch.Tensor]] = []

        # FP16 window storage (recent tokens at full precision)
        self._fp16_keys: List[torch.Tensor] = []
        self._fp16_values: List[torch.Tensor] = []
        self._fp16_seq_len: int = 0

    def _lazy_init(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self._batch_size = key_states.shape[0]
        self._num_heads = key_states.shape[1]
        self._head_dim = key_states.shape[3]
        self._dtype = key_states.dtype
        self._device = key_states.device

        d = self._head_dim
        device = str(self._device) if self._device is not None else "cpu"

        if self.use_residual_quant:
            self._key_rq = ResidualQuantEstimator(
                d=d, bits=self.key_bits, seed=self.seed, device=device,
            )
        else:
            self._key_pq = PolarQuant(
                d=d, bits=self.key_bits, seed=self.seed, device=device,
            )

        self._val_pq = PolarQuant(
            d=d, bits=self.val_bits, seed=self.seed + 100, device=device,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._val_pq is None:
            self._lazy_init(key_states, value_states)

        batch, num_heads, new_seq, head_dim = key_states.shape

        if self.fp16_window > 0:
            # Add new tokens to FP16 window first
            self._fp16_keys.append(key_states)
            self._fp16_values.append(value_states)
            self._fp16_seq_len += new_seq
            self._seq_len += new_seq

            # If FP16 window exceeds limit, spill oldest tokens to compressed
            self._spill_fp16_to_compressed()
        else:
            # No windowing, compress everything
            self._compress_and_store(key_states, value_states)
            self._seq_len += new_seq

        return self._dequantize_all()

    def _spill_fp16_to_compressed(self) -> None:
        """Move excess FP16 tokens to compressed storage."""
        if self._fp16_seq_len <= self.fp16_window:
            return

        # Concatenate all FP16 tokens
        all_fp16_keys = torch.cat(self._fp16_keys, dim=2)
        all_fp16_vals = torch.cat(self._fp16_values, dim=2)

        # Split: tokens to compress vs tokens to keep at FP16
        n_to_compress = self._fp16_seq_len - self.fp16_window
        keys_to_compress = all_fp16_keys[:, :, :n_to_compress]
        vals_to_compress = all_fp16_vals[:, :, :n_to_compress]
        keys_to_keep = all_fp16_keys[:, :, n_to_compress:]
        vals_to_keep = all_fp16_vals[:, :, n_to_compress:]

        # Compress the spilled tokens
        self._compress_and_store(keys_to_compress, vals_to_compress)

        # Update FP16 window
        self._fp16_keys = [keys_to_keep]
        self._fp16_values = [vals_to_keep]
        self._fp16_seq_len = self.fp16_window

    def _compress_and_store(
        self, key_states: torch.Tensor, value_states: torch.Tensor,
    ) -> None:
        """Compress key/value states and add to compressed storage."""
        batch, num_heads, new_seq, head_dim = key_states.shape

        keys_flat = key_states.float().reshape(-1, head_dim)
        vals_flat = value_states.float().reshape(-1, head_dim)

        # Compress keys
        if self.use_residual_quant and self._key_rq is not None:
            key_comp = self._key_rq.quantize(keys_flat)
            key_entry = {
                "mse_indices": key_comp["mse_indices"].reshape(batch, num_heads, new_seq, head_dim),
                "residual_signs": key_comp["residual_signs"].reshape(batch, num_heads, new_seq, head_dim),
                "residual_scale": key_comp["residual_scale"].reshape(batch, num_heads, new_seq),
                "vec_norm": key_comp["vec_norm"].reshape(batch, num_heads, new_seq),
                "type": "residual_quant",
            }
        else:
            key_norms = keys_flat.norm(dim=-1, keepdim=True)
            keys_normalized = keys_flat / (key_norms + 1e-8)
            key_indices = self._key_pq.quantize(keys_normalized)
            key_entry = {
                "mse_indices": key_indices.reshape(batch, num_heads, new_seq, head_dim),
                "vec_norm": key_norms.squeeze(-1).reshape(batch, num_heads, new_seq),
                "type": "mse_only",
            }

        # Compress values (always MSE-only)
        val_norms = vals_flat.norm(dim=-1, keepdim=True)
        vals_normalized = vals_flat / (val_norms + 1e-8)
        val_indices = self._val_pq.quantize(vals_normalized)
        val_entry = {
            "indices": val_indices.reshape(batch, num_heads, new_seq, head_dim),
            "norms": val_norms.squeeze(-1).reshape(batch, num_heads, new_seq),
        }

        self._key_compressed.append(key_entry)
        self._value_compressed.append(val_entry)

    def _dequantize_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._seq_len == 0:
            empty = torch.zeros(
                self._batch_size or 1, self._num_heads or 1, 0,
                self._head_dim or 1,
                dtype=self._dtype, device=self._device,
            )
            return empty, empty

        head_dim = self._head_dim
        batch = self._batch_size
        num_heads = self._num_heads
        key_parts = []
        val_parts = []

        # Dequantize compressed tokens
        if self._key_compressed:
            # Keys
            for entry in self._key_compressed:
                mse_indices = entry["mse_indices"]
                chunk_seq = mse_indices.shape[2]
                flat_mse = mse_indices.reshape(-1, head_dim)

                if entry.get("type") == "residual_quant" and self._key_rq is not None:
                    flat_signs = entry["residual_signs"].reshape(-1, head_dim)
                    flat_scale = entry["residual_scale"].reshape(-1)
                    flat_norm = entry["vec_norm"].reshape(-1)
                    comp_dict = {
                        "mse_indices": flat_mse,
                        "residual_signs": flat_signs,
                        "residual_scale": flat_scale,
                        "vec_norm": flat_norm,
                    }
                    recon = self._key_rq.dequantize(comp_dict)
                else:
                    flat_norm = entry["vec_norm"].reshape(-1)
                    recon = self._key_pq.dequantize(flat_mse)
                    recon = recon * flat_norm.unsqueeze(-1)

                key_parts.append(recon.reshape(batch, num_heads, chunk_seq, head_dim))

            # Values
            for entry in self._value_compressed:
                indices = entry["indices"]
                chunk_seq = indices.shape[2]
                flat_idx = indices.reshape(-1, head_dim)
                flat_norms = entry["norms"].reshape(-1)
                recon = self._val_pq.dequantize(flat_idx)
                recon = recon * flat_norms.unsqueeze(-1)
                val_parts.append(recon.reshape(batch, num_heads, chunk_seq, head_dim))

        # Append FP16 window tokens (uncompressed)
        if self._fp16_seq_len > 0:
            fp16_k = torch.cat(self._fp16_keys, dim=2)
            fp16_v = torch.cat(self._fp16_values, dim=2)
            key_parts.append(fp16_k)
            val_parts.append(fp16_v)

        if not key_parts:
            empty = torch.zeros(
                batch, num_heads, 0, head_dim,
                dtype=self._dtype, device=self._device,
            )
            return empty, empty

        keys_out = torch.cat(key_parts, dim=2)
        vals_out = torch.cat(val_parts, dim=2)
        return keys_out.to(self._dtype), vals_out.to(self._dtype)

    def get_seq_length(self) -> int:
        return self._seq_len

    def reorder(self, beam_idx: torch.LongTensor) -> None:
        for entry in self._key_compressed:
            for k in list(entry.keys()):
                if k != "type" and isinstance(entry[k], torch.Tensor):
                    entry[k] = entry[k].index_select(0, beam_idx)
        for entry in self._value_compressed:
            for k in entry:
                if isinstance(entry[k], torch.Tensor):
                    entry[k] = entry[k].index_select(0, beam_idx)
        self._fp16_keys = [k.index_select(0, beam_idx) for k in self._fp16_keys]
        self._fp16_values = [v.index_select(0, beam_idx) for v in self._fp16_values]

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self._seq_len + max_length
        if self._seq_len <= max_length:
            return
        # Simple: dequantize, slice, re-store as FP16
        keys, vals = self._dequantize_all()
        keys = keys[:, :, :max_length]
        vals = vals[:, :, :max_length]
        self._key_compressed.clear()
        self._value_compressed.clear()
        self._fp16_keys = [keys]
        self._fp16_values = [vals]
        self._fp16_seq_len = max_length
        self._seq_len = max_length

    def clear(self) -> None:
        self._key_compressed.clear()
        self._value_compressed.clear()
        self._fp16_keys.clear()
        self._fp16_values.clear()
        self._seq_len = 0
        self._fp16_seq_len = 0

    def memory_usage_bits(self) -> Dict[str, Any]:
        if self._seq_len == 0 or self._head_dim is None:
            return {
                "key_bits": 0, "value_bits": 0, "total_bits": 0,
                "fp16_baseline_bits": 0, "compression_ratio": 0.0,
            }

        n_heads = self._num_heads
        batch_sz = self._batch_size
        d = self._head_dim

        # Compressed tokens
        compressed_tokens = (self._seq_len - self._fp16_seq_len) * n_heads * batch_sz
        fp16_tokens = self._fp16_seq_len * n_heads * batch_sz

        # Key cost per compressed token
        if self.use_residual_quant:
            key_bpt = self.key_bits * d + 16 + 16  # MSE + signs + scale + norm
        else:
            key_bpt = self.key_bits * d + 16  # MSE + norm

        val_bpt = self.val_bits * d + 16  # MSE + norm

        key_bits = compressed_tokens * key_bpt + fp16_tokens * 16 * d
        val_bits = compressed_tokens * val_bpt + fp16_tokens * 16 * d
        total = key_bits + val_bits
        fp16_baseline = self._seq_len * n_heads * batch_sz * d * 16 * 2

        return {
            "key_bits": key_bits,
            "value_bits": val_bits,
            "total_bits": total,
            "fp16_baseline_bits": fp16_baseline,
            "compression_ratio": fp16_baseline / total if total > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# UltimateCache -- the combined stack
# ---------------------------------------------------------------------------

class UltimateCache:
    """The full TurboQuantDC compression stack for generation.

    Combines all breakthroughs:
    1. FP16 anchor layers (every anchor_interval layers)
    2. ResidualQuant for compressed layers' keys (direct signs, not QJL)
    3. Asymmetric K/V (keys at key_bits, values at val_bits)
    4. Residual window (last fp16_window tokens at FP16 within compressed layers)

    Duck-types HF Cache protocol for model.generate().

    Args:
        num_layers: Total transformer layers.
        key_bits: Bits for key compression in compressed layers.
        val_bits: Bits for value compression in compressed layers.
        anchor_interval: Every Nth layer is FP16 (0 = no anchors).
        fp16_window: Keep last N tokens at FP16 in compressed layers (0 = none).
        use_residual_quant: Use ResidualQuant for keys (vs MSE-only PolarQuant).
        seed: Base random seed.
    """

    is_compileable = False

    def __init__(
        self,
        num_layers: int = 36,
        key_bits: int = 4,
        val_bits: int = 2,
        anchor_interval: int = 12,
        fp16_window: int = 0,
        use_residual_quant: bool = False,
        seed: int = 42,
    ):
        self.num_layers = num_layers
        self.key_bits = key_bits
        self.val_bits = val_bits
        self.anchor_interval = anchor_interval
        self.fp16_window = fp16_window
        self.use_residual_quant = use_residual_quant
        self.seed = seed

        # Determine anchor layers
        if anchor_interval > 0:
            self._fp16_layers = {
                i for i in range(num_layers) if i % anchor_interval == 0
            }
        else:
            self._fp16_layers = set()

        # Create all layers
        self._layers: List[FP16Layer | AsymmetricCompressedLayer] = []
        for i in range(num_layers):
            if i in self._fp16_layers:
                self._layers.append(FP16Layer())
            else:
                self._layers.append(AsymmetricCompressedLayer(
                    key_bits=key_bits,
                    val_bits=val_bits,
                    use_residual_quant=use_residual_quant,
                    fp16_window=fp16_window,
                    seed=seed + i,
                ))

    @property
    def fp16_layer_count(self) -> int:
        return len(self._fp16_layers)

    @property
    def compressed_layer_count(self) -> int:
        return self.num_layers - self.fp16_layer_count

    def config_summary(self) -> str:
        parts = []
        if self.anchor_interval > 0:
            parts.append(f"anchor-{self.anchor_interval}")
        else:
            parts.append("no-anchors")
        if self.use_residual_quant:
            parts.append(f"ResQ-K{self.key_bits}")
        else:
            parts.append(f"MSE-K{self.key_bits}")
        parts.append(f"MSE-V{self.val_bits}")
        if self.fp16_window > 0:
            parts.append(f"win{self.fp16_window}")
        fp16_pct = 100 * self.fp16_layer_count / self.num_layers if self.num_layers > 0 else 0
        parts.append(f"({fp16_pct:.0f}%FP16)")
        return " + ".join(parts)

    # ---- HF Cache protocol ----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self._layers) <= layer_idx:
            idx = len(self._layers)
            if idx in self._fp16_layers:
                self._layers.append(FP16Layer())
            else:
                self._layers.append(AsymmetricCompressedLayer(
                    key_bits=self.key_bits,
                    val_bits=self.val_bits,
                    use_residual_quant=self.use_residual_quant,
                    fp16_window=self.fp16_window,
                    seed=self.seed + idx,
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
    ) -> tuple[int, int]:
        if layer_idx >= len(self._layers):
            return cache_position.shape[0], 0
        query_length = cache_position.shape[0]
        kv_length = self._layers[layer_idx].get_seq_length() + query_length
        return kv_length, 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for layer in self._layers:
            layer.reorder(beam_idx)

    def crop(self, max_length: int) -> None:
        for layer in self._layers:
            layer.crop(max_length)

    def reset(self) -> None:
        for layer in self._layers:
            layer.clear()

    def batch_repeat_interleave(self, repeats: int) -> None:
        for layer in self._layers:
            if isinstance(layer, FP16Layer):
                layer._keys = [k.repeat_interleave(repeats, dim=0) for k in layer._keys]
                layer._values = [v.repeat_interleave(repeats, dim=0) for v in layer._values]
            else:
                for entry in layer._key_compressed:
                    for k in list(entry.keys()):
                        if k != "type" and isinstance(entry[k], torch.Tensor):
                            entry[k] = entry[k].repeat_interleave(repeats, dim=0)
                for entry in layer._value_compressed:
                    for k in entry:
                        if isinstance(entry[k], torch.Tensor):
                            entry[k] = entry[k].repeat_interleave(repeats, dim=0)
                layer._fp16_keys = [k.repeat_interleave(repeats, dim=0) for k in layer._fp16_keys]
                layer._fp16_values = [v.repeat_interleave(repeats, dim=0) for v in layer._fp16_values]

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for layer in self._layers:
            if isinstance(layer, FP16Layer):
                layer._keys = [k[indices] for k in layer._keys]
                layer._values = [v[indices] for v in layer._values]
            else:
                for entry in layer._key_compressed:
                    for k in list(entry.keys()):
                        if k != "type" and isinstance(entry[k], torch.Tensor):
                            entry[k] = entry[k][indices]
                for entry in layer._value_compressed:
                    for k in entry:
                        if isinstance(entry[k], torch.Tensor):
                            entry[k] = entry[k][indices]
                layer._fp16_keys = [k[indices] for k in layer._fp16_keys]
                layer._fp16_values = [v[indices] for v in layer._fp16_values]

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
            raise IndexError(f"Layer {layer_idx} not in cache (have {len(self._layers)})")
        return self._layers[layer_idx]._dequantize_all()

    # ---- Reporting ----

    def memory_savings(self) -> Dict[str, Any]:
        per_layer = []
        total_compressed = 0
        total_fp16 = 0

        for i, layer in enumerate(self._layers):
            stats = layer.memory_usage_bits()
            is_anchor = i in self._fp16_layers
            per_layer.append({"layer": i, "is_fp16_anchor": is_anchor, **stats})
            total_compressed += stats["total_bits"]
            total_fp16 += stats["fp16_baseline_bits"]

        return {
            "per_layer": per_layer,
            "total_compressed_bits": total_compressed,
            "total_fp16_bits": total_fp16,
            "overall_compression_ratio": (
                total_fp16 / total_compressed if total_compressed > 0 else 0.0
            ),
            "config": self.config_summary(),
        }

    def theoretical_compression_ratio(self) -> float:
        """Compute theoretical compression ratio before any data is stored."""
        if self.num_layers == 0:
            return 1.0

        n_fp16 = self.fp16_layer_count
        n_comp = self.compressed_layer_count

        # FP16: 32 bits per coordinate (K + V each at 16-bit)
        fp16_cost = n_fp16 * 32

        # Compressed: key_bits + val_bits per coordinate + overhead
        if self.use_residual_quant:
            # ResQ keys: key_bits * d + residual_signs * d + scale(16) + norm(16)
            # But per coordinate: key_bits + 32/d
            comp_key_per_coord = self.key_bits + 32 / 128
        else:
            comp_key_per_coord = self.key_bits + 16 / 128

        comp_val_per_coord = self.val_bits + 16 / 128
        comp_cost = n_comp * (comp_key_per_coord + comp_val_per_coord)

        # Adjust for FP16 window (approximate)
        if self.fp16_window > 0 and n_comp > 0:
            # Window effect reduces compression for recent tokens
            # This is sequence-length dependent; approximate for 512 tokens
            window_fraction = min(self.fp16_window / 512.0, 1.0)
            comp_cost_windowed = n_comp * (
                (1 - window_fraction) * (comp_key_per_coord + comp_val_per_coord)
                + window_fraction * 32
            )
            comp_cost = comp_cost_windowed

        total_cost = fp16_cost + comp_cost
        baseline = self.num_layers * 32

        return baseline / total_cost if total_cost > 0 else 1.0
