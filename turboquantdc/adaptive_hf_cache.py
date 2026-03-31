"""HF-compatible KV cache with per-layer bit-width control and FP16 anchors.

Some transformer layers store KV at FP16 (no compression), others use TurboQuant.
This breaks the error accumulation chain -- every Nth layer gets perfect FP16 KV,
resetting the drift that causes garbled generation.

Anchor strategies:
    "interval": Every Nth layer is FP16 (e.g., layers 0, 6, 12, 18, 24, 30).
    "early":    First N layers are FP16 (rest compressed).
    "tail":     Last N layers are FP16 (rest compressed).

Duck-types the HF Cache protocol (update, get_seq_length, __getitem__, __len__,
__iter__, etc.) just like TurboQuantCache does.

Usage:
    from turboquantdc.adaptive_hf_cache import AdaptiveHFCache
    from transformers import AutoModelForCausalLM

    cache = AdaptiveHFCache(
        num_layers=36,
        compressed_bits=4,
        anchor_interval=6,
        anchor_mode="interval",
        mse_only=True,
    )
    output = model.generate(
        inputs,
        max_new_tokens=100,
        past_key_values=cache,
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .hf_integration import TurboQuantLayer


class FP16Layer:
    """A single layer's FP16 KV cache storage (no compression).

    Stores raw key/value tensors at full precision. Implements the same
    interface as TurboQuantLayer for duck-typing.
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
        """Store raw FP16 tensors and return full cache.

        Args:
            key_states: [batch, num_heads, new_seq, head_dim]
            value_states: [batch, num_heads, new_seq, head_dim]

        Returns:
            Tuple of (all_keys, all_values) tensors.
        """
        self._keys.append(key_states)
        self._values.append(value_states)
        self._seq_len += key_states.shape[2]

        all_keys = torch.cat(self._keys, dim=2)
        all_values = torch.cat(self._values, dim=2)
        return all_keys, all_values

    def _dequantize_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return all stored keys and values (no dequantization needed)."""
        if self._seq_len == 0:
            return (
                torch.zeros(1, 1, 0, 1),
                torch.zeros(1, 1, 0, 1),
            )
        all_keys = torch.cat(self._keys, dim=2)
        all_values = torch.cat(self._values, dim=2)
        return all_keys, all_values

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
        # Concatenate, slice, and re-store as single chunk
        all_keys = torch.cat(self._keys, dim=2)[:, :, :max_length]
        all_values = torch.cat(self._values, dim=2)[:, :, :max_length]
        self._keys = [all_keys]
        self._values = [all_values]
        self._seq_len = max_length

    def memory_usage_bits(self) -> Dict[str, int]:
        if self._seq_len == 0 or not self._keys:
            return {
                "key_bits": 0,
                "value_bits": 0,
                "total_bits": 0,
                "fp16_baseline_bits": 0,
                "compression_ratio": 1.0,
            }
        k = self._keys[0]
        batch = k.shape[0]
        n_heads = k.shape[1]
        head_dim = k.shape[3]
        total_tokens = self._seq_len * n_heads * batch
        # FP16 = 16 bits per coordinate, keys + values
        total = total_tokens * head_dim * 16 * 2
        return {
            "key_bits": total // 2,
            "value_bits": total // 2,
            "total_bits": total,
            "fp16_baseline_bits": total,
            "compression_ratio": 1.0,
        }

    def clear(self) -> None:
        self._keys.clear()
        self._values.clear()
        self._seq_len = 0


class AdaptiveHFCache:
    """HF-compatible KV cache with per-layer bit-width control.

    Some layers store KV at FP16 (no compression), others use TurboQuant.
    This breaks the error accumulation chain -- every Nth layer gets
    perfect FP16 KV, resetting the drift.

    Args:
        num_layers: Total transformer layers.
        compressed_bits: Bit-width for compressed layers (4 recommended).
        anchor_interval: Every Nth layer is FP16 (e.g., 6 means layers
            0, 6, 12, 18, 24, 30 are FP16).
        anchor_mode: "interval" (every Nth), "early" (first N layers FP16),
            or "tail" (last N layers FP16).
        n_early_fp16: Number of early/tail layers at FP16 (for "early"/"tail").
        mse_only: Use MSE-only for compressed layers (skip QJL).
        seed: Base random seed.
    """

    is_compileable = False

    def __init__(
        self,
        num_layers: int,
        compressed_bits: int = 4,
        anchor_interval: int = 6,
        anchor_mode: str = "interval",
        n_early_fp16: int = 6,
        mse_only: bool = True,
        seed: int = 42,
    ):
        self.num_layers = num_layers
        self.compressed_bits = compressed_bits
        self.anchor_interval = anchor_interval
        self.anchor_mode = anchor_mode
        self.n_early_fp16 = n_early_fp16
        self.mse_only = mse_only
        self.seed = seed

        # Compute which layers are FP16 anchors
        self._fp16_layers = self._compute_fp16_layers()

        # Pre-create all layers
        self._layers: List[FP16Layer | TurboQuantLayer] = []
        for i in range(num_layers):
            if i in self._fp16_layers:
                self._layers.append(FP16Layer())
            else:
                self._layers.append(
                    TurboQuantLayer(
                        bits=compressed_bits,
                        seed=seed + i,
                        mse_only=mse_only,
                    )
                )

    def _compute_fp16_layers(self) -> set:
        """Determine which layers are FP16 anchors."""
        if self.anchor_mode == "interval":
            return {i for i in range(self.num_layers) if i % self.anchor_interval == 0}
        elif self.anchor_mode == "early":
            return set(range(min(self.n_early_fp16, self.num_layers)))
        elif self.anchor_mode == "tail":
            start = max(0, self.num_layers - self.n_early_fp16)
            return set(range(start, self.num_layers))
        else:
            raise ValueError(f"Unknown anchor_mode: {self.anchor_mode}")

    @property
    def fp16_layer_count(self) -> int:
        return len(self._fp16_layers)

    @property
    def compressed_layer_count(self) -> int:
        return self.num_layers - self.fp16_layer_count

    @property
    def fp16_fraction(self) -> float:
        return self.fp16_layer_count / self.num_layers if self.num_layers > 0 else 0.0

    def config_summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        fp16_count = self.fp16_layer_count
        comp_count = self.compressed_layer_count
        pct = 100.0 * self.fp16_fraction
        return (
            f"{self.anchor_mode} anchor: {fp16_count}/{self.num_layers} FP16 "
            f"({pct:.0f}%), {comp_count} TQ-{self.compressed_bits}"
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

        FP16 layers store raw tensors. Compressed layers use TurboQuantLayer.
        """
        # Lazily extend if model has more layers than we expected
        while len(self._layers) <= layer_idx:
            idx = len(self._layers)
            if idx in self._fp16_layers:
                self._layers.append(FP16Layer())
            else:
                self._layers.append(
                    TurboQuantLayer(
                        bits=self.compressed_bits,
                        seed=self.seed + idx,
                        mse_only=self.mse_only,
                    )
                )

        return self._layers[layer_idx].update(key_states, value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._layers):
            return 0
        return self._layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int
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
                # TurboQuantLayer
                for entry in layer._key_compressed:
                    for key in entry:
                        entry[key] = entry[key].repeat_interleave(repeats, dim=0)
                for entry in layer._value_compressed:
                    for key in entry:
                        entry[key] = entry[key].repeat_interleave(repeats, dim=0)
                if layer._batch_size is not None:
                    layer._batch_size *= repeats

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for layer in self._layers:
            if isinstance(layer, FP16Layer):
                layer._keys = [k[indices] for k in layer._keys]
                layer._values = [v[indices] for v in layer._values]
            else:
                for entry in layer._key_compressed:
                    for key in entry:
                        entry[key] = entry[key][indices]
                for entry in layer._value_compressed:
                    for key in entry:
                        entry[key] = entry[key][indices]
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
            raise IndexError(
                f"Layer {layer_idx} not in cache (have {len(self._layers)} layers)"
            )
        return self._layers[layer_idx]._dequantize_all()

    # ---- Memory reporting ----

    def memory_savings(self) -> Dict[str, Any]:
        """Report memory usage and savings across all layers."""
        per_layer = []
        total_compressed = 0
        total_fp16 = 0

        for i, layer in enumerate(self._layers):
            stats = layer.memory_usage_bits()
            is_anchor = i in self._fp16_layers
            per_layer.append({
                "layer": i,
                "is_fp16_anchor": is_anchor,
                **stats,
            })
            total_compressed += stats["total_bits"]
            total_fp16 += stats["fp16_baseline_bits"]

        return {
            "per_layer": per_layer,
            "total_compressed_bits": total_compressed,
            "total_fp16_bits": total_fp16,
            "overall_compression_ratio": (
                total_fp16 / total_compressed if total_compressed > 0 else 0.0
            ),
            "bits": self.compressed_bits,
            "num_layers": len(self._layers),
            "fp16_anchor_count": self.fp16_layer_count,
            "compressed_layer_count": self.compressed_layer_count,
            "anchor_mode": self.anchor_mode,
        }

    def effective_compression_ratio(self) -> float:
        """Compute the effective compression ratio accounting for FP16 anchors.

        This is the theoretical ratio based on layer counts and bit-widths,
        not actual stored data (works even before any data is stored).
        """
        if self.num_layers == 0:
            return 1.0

        n_fp16 = self.fp16_layer_count
        n_comp = self.compressed_layer_count
        b = self.compressed_bits

        # FP16 baseline: 16 bits per coordinate, keys + values = 32 bits
        # Compressed (MSE-only): b bits per coordinate, keys + values = 2*b bits + overhead
        # Approximate: each FP16 layer uses 32*d bits/token, compressed uses ~2*b*d bits/token

        fp16_cost = n_fp16 * 32  # bits per coordinate pair
        if self.mse_only:
            comp_cost = n_comp * (2 * b + 32 / 128)  # approx overhead
        else:
            comp_cost = n_comp * (2 * b + 1 + 48 / 128)  # QJL adds ~1 bit
        total_cost = fp16_cost + comp_cost
        baseline_cost = self.num_layers * 32

        return baseline_cost / total_cost if total_cost > 0 else 1.0
