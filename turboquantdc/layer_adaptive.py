"""Layer-adaptive KV cache with per-layer bit-width assignment.

Not all transformer layers are equally sensitive to KV cache compression.
The last few layers (closest to the output) are most quality-critical.
By assigning higher bit-widths to critical layers and lower bit-widths
to less critical ones, we can achieve near-lossless quality at higher
overall compression ratios.

Strategies:
    "tail_preserve": Last N layers at higher bits (or FP16), rest at base_bits.
    "gradient": Linearly increasing bits from first to last layer.
    "custom": User-specified per-layer bit-widths.

Key insight (from TheTom/turboquant_plus): Keeping the last 8 layers at
full precision while compressing the rest achieves q8_0-equivalent quality
at 3.5x compression.

Reference: TheTom/turboquant_plus "layer-adaptive mode 2".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch

from .kv_cache import TurboQuantKVCache


class FP16Cache:
    """Simple FP16 KV cache (no compression) with the same interface as TurboQuantKVCache.

    Used for critical layers where quantization error is unacceptable.
    Stores raw FP16 key and value tensors without any compression.

    Args:
        d_key: Key head dimension.
        d_value: Value head dimension.
        device: Target device.
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        device: str | torch.device = "cpu",
    ):
        self.d_key = d_key
        self.d_value = d_value
        self.device = device
        self._keys: List[torch.Tensor] = []
        self._values: List[torch.Tensor] = []

    @property
    def seq_len(self) -> int:
        """Number of tokens in the cache."""
        return sum(
            k.shape[0] if k.dim() > 1 else 1 for k in self._keys
        )

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Store raw FP16 tensors without compression.

        Args:
            keys: Key vectors, shape (batch, d_key) or (d_key,).
            values: Value vectors, shape (batch, d_value) or (d_value,).
        """
        if keys.dim() == 1:
            keys = keys.unsqueeze(0)
            values = values.unsqueeze(0)
        self._keys.append(keys.to(self.device))
        self._values.append(values.to(self.device))

    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        """Exact FP16 dot product -- no quantization error.

        Args:
            queries: Query vectors, shape (n_queries, d_key) or (d_key,).

        Returns:
            Attention scores. Shape depends on input:
            - (d_key,) query -> (seq_len,)
            - (n_queries, d_key) queries -> (n_queries, seq_len)
        """
        if not self._keys:
            if queries.dim() == 1:
                return torch.zeros(0, device=self.device)
            return torch.zeros(
                queries.shape[0], 0, device=self.device
            )
        all_keys = torch.cat(self._keys, dim=0)  # (seq_len, d_key)
        if queries.dim() == 1:
            return all_keys @ queries  # (seq_len,)
        return queries @ all_keys.T  # (n_queries, seq_len)

    def get_values(self) -> torch.Tensor:
        """Return all stored values concatenated.

        Returns:
            Values tensor of shape (seq_len, d_value).
        """
        if not self._values:
            return torch.zeros(0, self.d_value, device=self.device)
        return torch.cat(self._values, dim=0)

    def memory_usage_bits(self) -> Dict[str, Union[int, float]]:
        """Compute memory usage for this FP16 cache.

        Returns:
            Dict with total_bits, fp16_baseline_bits, compression_ratio.
        """
        n = self.seq_len
        total = n * (self.d_key + self.d_value) * 16
        return {
            "total_bits": total,
            "fp16_baseline_bits": total,
            "compression_ratio": 1.0 if n > 0 else 0.0,
        }

    def clear(self) -> None:
        """Clear all cached key-value pairs."""
        self._keys.clear()
        self._values.clear()


class LayerAdaptiveKVCache:
    """Multi-layer KV cache with per-layer bit-width assignment.

    Different transformer layers get different compression levels:
    - Critical layers (typically last N): higher bits or uncompressed (FP16)
    - Standard layers: normal TurboQuant compression
    - Optional: aggressive layers (first N): lower bits

    Strategies:
        "tail_preserve": Last `n_preserve` layers at `preserve_bits`, rest at `base_bits`.
        "gradient": Linearly increasing bits from first to last layer.
        "custom": User-specified per-layer bit-widths.

    Args:
        num_layers: Number of transformer layers.
        d_key: Key head dimension.
        d_value: Value head dimension.
        strategy: Compression strategy ("tail_preserve", "gradient", "custom").
        base_bits: Default bit-width for compressed layers (default: 3).
        preserve_bits: Bit-width for preserved layers (default: 0 = FP16).
        n_preserve: Number of tail layers to preserve (default: 8).
        bits_schedule: List of per-layer bit-widths for "custom" strategy.
        seed: Random seed.
        device: Target device.
    """

    def __init__(
        self,
        num_layers: int,
        d_key: int,
        d_value: int,
        strategy: str = "tail_preserve",
        base_bits: int = 3,
        preserve_bits: int = 0,
        n_preserve: int = 8,
        bits_schedule: Optional[List[int]] = None,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        self.num_layers = num_layers
        self.d_key = d_key
        self.d_value = d_value
        self.strategy = strategy
        self.device = device

        # Compute per-layer bit schedule
        self.bits_schedule = self._compute_schedule(
            strategy, num_layers, base_bits, preserve_bits,
            n_preserve, bits_schedule,
        )

        # Create per-layer caches
        # bits=0 means FP16 (no compression)
        self.layer_caches: List[Union[FP16Cache, TurboQuantKVCache]] = []
        for layer_idx in range(num_layers):
            bits = self.bits_schedule[layer_idx]
            if bits == 0:
                self.layer_caches.append(
                    FP16Cache(d_key, d_value, device)
                )
            else:
                self.layer_caches.append(
                    TurboQuantKVCache(
                        d_key, d_value, bits=bits,
                        seed=seed + layer_idx, device=device,
                    )
                )

    def append(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Append KV pair for a specific layer.

        Args:
            layer_idx: Layer index in [0, num_layers).
            keys: Key vectors, shape (batch, d_key) or (d_key,).
            values: Value vectors, shape (batch, d_value) or (d_value,).
        """
        self.layer_caches[layer_idx].append(keys, values)

    def attention_scores(self, layer_idx: int, queries: torch.Tensor) -> torch.Tensor:
        """Get attention scores for a specific layer.

        Args:
            layer_idx: Layer index in [0, num_layers).
            queries: Query vectors, shape (n_queries, d_key) or (d_key,).

        Returns:
            Attention scores, shape (n_queries, seq_len) or (seq_len,).
        """
        return self.layer_caches[layer_idx].attention_scores(queries)

    def get_values(self, layer_idx: int) -> torch.Tensor:
        """Get reconstructed values for a specific layer.

        Args:
            layer_idx: Layer index in [0, num_layers).

        Returns:
            Reconstructed values, shape (seq_len, d_value).
        """
        return self.layer_caches[layer_idx].get_values()

    def memory_usage_bits(self) -> Dict[str, Union[int, float, List]]:
        """Aggregate memory usage across all layers.

        Returns:
            Dict with:
                - total_bits: Grand total across all layers.
                - fp16_baseline_bits: What full FP16 would cost.
                - compression_ratio: fp16_baseline / total.
                - per_layer: List of per-layer memory dicts.
        """
        per_layer = []
        total_bits = 0
        fp16_baseline_bits = 0

        for cache in self.layer_caches:
            usage = cache.memory_usage_bits()
            per_layer.append(usage)
            total_bits += usage["total_bits"]
            fp16_baseline_bits += usage["fp16_baseline_bits"]

        return {
            "total_bits": total_bits,
            "fp16_baseline_bits": fp16_baseline_bits,
            "compression_ratio": (
                fp16_baseline_bits / total_bits if total_bits > 0 else 0.0
            ),
            "per_layer": per_layer,
        }

    def effective_compression(self) -> float:
        """Overall compression ratio across all layers.

        Returns:
            Ratio of FP16 baseline to actual storage. Higher is better.
            Returns 0.0 if no data stored.
        """
        usage = self.memory_usage_bits()
        return usage["compression_ratio"]

    def clear(self, layer_idx: Optional[int] = None) -> None:
        """Clear cached key-value pairs.

        Args:
            layer_idx: If specified, clear only that layer.
                       If None, clear all layers.
        """
        if layer_idx is not None:
            self.layer_caches[layer_idx].clear()
        else:
            for cache in self.layer_caches:
                cache.clear()

    @staticmethod
    def _compute_schedule(
        strategy: str,
        num_layers: int,
        base_bits: int,
        preserve_bits: int,
        n_preserve: int,
        bits_schedule: Optional[List[int]],
    ) -> List[int]:
        """Compute per-layer bit-width schedule.

        Args:
            strategy: One of "tail_preserve", "gradient", "custom".
            num_layers: Number of transformer layers.
            base_bits: Default bit-width for compressed layers.
            preserve_bits: Bit-width for preserved layers (0 = FP16).
            n_preserve: Number of tail layers to preserve.
            bits_schedule: Explicit per-layer schedule for "custom" strategy.

        Returns:
            List of per-layer bit-widths. 0 means FP16 (no compression).
        """
        if strategy == "tail_preserve":
            n_compressed = max(num_layers - n_preserve, 0)
            n_preserved = num_layers - n_compressed
            schedule = [base_bits] * n_compressed + [preserve_bits] * n_preserved
        elif strategy == "gradient":
            # Linear interpolation from base_bits to preserve_bits
            schedule = []
            for i in range(num_layers):
                t = i / max(num_layers - 1, 1)
                raw = base_bits + (preserve_bits - base_bits) * t
                rounded = round(raw)
                # Clamp: 0 means FP16, otherwise must be in [2, 4]
                if rounded <= 1:
                    schedule.append(0)
                else:
                    schedule.append(max(2, min(4, rounded)))
            return schedule
        elif strategy == "custom":
            if bits_schedule is None:
                raise ValueError(
                    "bits_schedule must be provided for 'custom' strategy"
                )
            if len(bits_schedule) != num_layers:
                raise ValueError(
                    f"bits_schedule length ({len(bits_schedule)}) must match "
                    f"num_layers ({num_layers})"
                )
            schedule = list(bits_schedule)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return schedule


def recommended_schedule(model_name: str, base_bits: int = 3) -> Dict[str, Union[int, str]]:
    """Return recommended layer-adaptive config for known models.

    Based on empirical findings from TheTom/turboquant_plus: keeping the
    last N layers at full precision while compressing the rest achieves
    near-lossless quality at significantly higher compression than uniform.

    Args:
        model_name: Model identifier (case-insensitive, partial match).
        base_bits: Default bit-width for compressed layers.

    Returns:
        Dict with keys: num_layers, strategy, base_bits, preserve_bits, n_preserve.

    Raises:
        KeyError: If model_name is not recognized.
    """
    configs = {
        "qwen2.5-3b": {"num_layers": 36, "n_preserve": 6},
        "qwen2.5-14b": {"num_layers": 48, "n_preserve": 8},
        "qwen3.5-27b": {"num_layers": 64, "n_preserve": 8},
        "llama-3-8b": {"num_layers": 32, "n_preserve": 6},
        "llama-3-70b": {"num_layers": 80, "n_preserve": 10},
    }

    key = model_name.lower()
    if key not in configs:
        available = ", ".join(sorted(configs.keys()))
        raise KeyError(
            f"Unknown model: '{model_name}'. Available: {available}"
        )

    cfg = configs[key]
    return {
        "num_layers": cfg["num_layers"],
        "strategy": "tail_preserve",
        "base_bits": base_bits,
        "preserve_bits": 0,  # FP16
        "n_preserve": cfg["n_preserve"],
    }


def estimate_memory(
    num_layers: int,
    d_key: int,
    d_value: int,
    seq_len: int,
    schedule: List[int],
) -> Dict[str, Union[float, List]]:
    """Estimate memory usage for a given schedule without creating caches.

    Computes theoretical memory for each layer based on bit-width:
    - FP16 (bits=0): 16 * (d_key + d_value) bits per token
    - Compressed (bits=b): key uses b*d_key + 32 bits, value uses b*d_value + 16 bits

    Args:
        num_layers: Number of transformer layers.
        d_key: Key head dimension.
        d_value: Value head dimension.
        seq_len: Sequence length (tokens per layer).
        schedule: Per-layer bit-widths (0 = FP16).

    Returns:
        Dict with:
            - fp16_gb: Total FP16 baseline memory in GB.
            - compressed_gb: Total compressed memory in GB.
            - ratio: Overall compression ratio.
            - per_layer: List of per-layer dicts with bits, bits_per_token,
              layer_mb.
    """
    if len(schedule) != num_layers:
        raise ValueError(
            f"schedule length ({len(schedule)}) must match "
            f"num_layers ({num_layers})"
        )

    fp16_bits_per_token = (d_key + d_value) * 16
    fp16_total_bits = num_layers * seq_len * fp16_bits_per_token

    per_layer = []
    compressed_total_bits = 0

    for layer_idx in range(num_layers):
        bits = schedule[layer_idx]
        if bits == 0:
            # FP16: no compression
            layer_bits_per_token = fp16_bits_per_token
        else:
            # Key: (bits-1)*d_key [MSE] + d_key [QJL] + 16 [r_norm] + 16 [v_norm]
            #    = bits*d_key + 32
            # Value: bits*d_value [MSE] + 16 [v_norm]
            key_bits = bits * d_key + 32
            value_bits = bits * d_value + 16
            layer_bits_per_token = key_bits + value_bits

        layer_total_bits = seq_len * layer_bits_per_token
        compressed_total_bits += layer_total_bits

        per_layer.append({
            "layer": layer_idx,
            "bits": bits,
            "bits_per_token": layer_bits_per_token,
            "layer_mb": layer_total_bits / 8 / 1024 / 1024,
        })

    bits_to_gb = 8 * 1024 * 1024 * 1024  # bits per GB

    return {
        "fp16_gb": fp16_total_bits / bits_to_gb,
        "compressed_gb": compressed_total_bits / bits_to_gb,
        "ratio": fp16_total_bits / compressed_total_bits if compressed_total_bits > 0 else 0.0,
        "per_layer": per_layer,
    }
