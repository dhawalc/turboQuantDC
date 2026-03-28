"""Temporal Decay Compression for TurboQuant KV cache.

Older tokens in the KV cache receive diminishing attention and can tolerate
lower quantization precision. This module divides the cache into three tiers:

    Tier 0 (hot):    Last ``hot_window`` tokens at ``hot_bits`` (e.g. 4-bit)
    Tier 1 (warm):   Next ``warm_window`` tokens at ``warm_bits`` (e.g. 3-bit)
    Tier 2 (cold):   Everything older at ``cold_bits`` (e.g. 2-bit)

When a token ages out of its tier it is re-quantized at the lower precision.
Re-quantization reconstructs the vector from the higher-precision cache and
re-compresses it with the lower-precision cache.  The QJL correction is lost
during reconstruction -- that is the quality cost of demotion.

At long context the majority of tokens reside in the cold tier, yielding
30-34 % additional memory savings on top of base TurboQuant compression.

Example at 32K tokens (hot=512, warm=4096):
    Hot  (512  tokens, 4-bit):  1.6 % of cache
    Warm (4096 tokens, 3-bit): 12.8 % of cache
    Cold (27392 tokens, 2-bit): 85.6 % of cache
    Weighted average ~ 2.2 bits vs uniform 3-bit = ~27 % savings

Reference: TheTom/turboquant_plus temporal decay extension.
"""

from __future__ import annotations

from typing import Dict

import torch

from .kv_cache import TurboQuantKVCache


class TemporalDecayCache:
    """KV cache with temporal decay -- older entries get progressively compressed.

    Args:
        d_key: Key head dimension.
        d_value: Value head dimension.
        hot_bits: Bit-width for recent tokens (default: 4).
        warm_bits: Bit-width for medium-age tokens (default: 3).
        cold_bits: Bit-width for old tokens (default: 2).
        hot_window: Number of recent tokens kept in the hot tier (default: 512).
        warm_window: Number of tokens kept in the warm tier (default: 4096).
        seed: Random seed for reproducibility.
        device: Target device.
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        hot_bits: int = 4,
        warm_bits: int = 3,
        cold_bits: int = 2,
        hot_window: int = 512,
        warm_window: int = 4096,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        self.d_key = d_key
        self.d_value = d_value
        self.hot_bits = hot_bits
        self.warm_bits = warm_bits
        self.cold_bits = cold_bits
        self.hot_window = hot_window
        self.warm_window = warm_window
        self.device = device

        # Three separate caches for each tier.  Different seeds ensure each
        # tier uses independent rotation / QJL projection matrices.
        self.hot_cache = TurboQuantKVCache(
            d_key, d_value, bits=hot_bits, seed=seed, device=device,
        )
        self.warm_cache = TurboQuantKVCache(
            d_key, d_value, bits=warm_bits, seed=seed + 100, device=device,
        )
        self.cold_cache = TurboQuantKVCache(
            d_key, d_value, bits=cold_bits, seed=seed + 200, device=device,
        )

        self._total_tokens = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def seq_len(self) -> int:
        """Total number of tokens across all tiers."""
        return self._total_tokens

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Add new tokens to the hot tier, triggering decay if needed.

        Args:
            keys: Key vectors, shape ``(batch, d_key)`` or ``(d_key,)``.
            values: Value vectors, shape ``(batch, d_value)`` or ``(d_value,)``.
        """
        # Determine how many tokens are being appended.
        if keys.dim() == 1:
            n_tokens = 1
        else:
            n_tokens = keys.shape[0]

        self.hot_cache.append(keys, values)
        self._total_tokens += n_tokens

        # Decay overflows through the tier chain.
        self._maybe_decay()

    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        """Compute attention scores across all three tiers.

        Scores are returned in sequence order: cold, then warm, then hot.

        Args:
            queries: Query vectors, shape ``(n_queries, d_key)`` or ``(d_key,)``.

        Returns:
            Attention scores, shape ``(n_queries, seq_len)`` or ``(seq_len,)``.
        """
        scores_parts = []

        if self.cold_cache.seq_len > 0:
            scores_parts.append(self.cold_cache.attention_scores(queries))
        if self.warm_cache.seq_len > 0:
            scores_parts.append(self.warm_cache.attention_scores(queries))
        if self.hot_cache.seq_len > 0:
            scores_parts.append(self.hot_cache.attention_scores(queries))

        if not scores_parts:
            if queries.dim() == 1:
                return torch.zeros(0, device=queries.device)
            return torch.zeros(queries.shape[0], 0, device=queries.device)

        return torch.cat(scores_parts, dim=-1)

    def get_values(self) -> torch.Tensor:
        """Reconstruct all values in sequence order (cold, warm, hot).

        Returns:
            Reconstructed values, shape ``(seq_len, d_value)``.
        """
        parts = []
        if self.cold_cache.seq_len > 0:
            parts.append(self.cold_cache.get_values())
        if self.warm_cache.seq_len > 0:
            parts.append(self.warm_cache.get_values())
        if self.hot_cache.seq_len > 0:
            parts.append(self.hot_cache.get_values())
        if not parts:
            return torch.zeros(0, self.d_value, device=self.device)
        return torch.cat(parts, dim=0)

    def memory_usage_bits(self) -> Dict[str, object]:
        """Memory statistics across all tiers.

        Returns:
            Dict with per-tier and aggregate bit counts, token counts,
            compression ratio vs FP16, and savings vs uniform warm-bits.
        """
        hot_usage = self.hot_cache.memory_usage_bits()
        warm_usage = self.warm_cache.memory_usage_bits()
        cold_usage = self.cold_cache.memory_usage_bits()

        total = (
            hot_usage["total_bits"]
            + warm_usage["total_bits"]
            + cold_usage["total_bits"]
        )
        fp16 = (
            hot_usage["fp16_baseline_bits"]
            + warm_usage["fp16_baseline_bits"]
            + cold_usage["fp16_baseline_bits"]
        )

        # Compute what uniform warm-bits (default 3-bit) would cost for the
        # same number of tokens.  We use the warm_bits tier as the baseline
        # because that is the "standard" TurboQuant precision.
        n_total = self._total_tokens
        if n_total > 0:
            # Approximate uniform cost: same formula as TurboQuantKVCache uses.
            # key bits: warm_bits * d_key + 32 (norms)
            # value bits: warm_bits * d_value + 16 (norm)
            uniform_bits_per_token = (
                self.warm_bits * self.d_key
                + 32  # key norms (vec + residual)
                + self.warm_bits * self.d_value
                + 16  # value norm
            )
            uniform_total = n_total * uniform_bits_per_token
            savings_pct = (
                (1.0 - total / uniform_total) * 100.0 if uniform_total > 0 else 0.0
            )
        else:
            uniform_total = 0
            savings_pct = 0.0

        return {
            "hot_bits": hot_usage["total_bits"],
            "warm_bits": warm_usage["total_bits"],
            "cold_bits": cold_usage["total_bits"],
            "total_bits": total,
            "fp16_baseline_bits": fp16,
            "compression_ratio": fp16 / total if total > 0 else 0.0,
            "hot_tokens": self._count_tokens(self.hot_cache),
            "warm_tokens": self._count_tokens(self.warm_cache),
            "cold_tokens": self._count_tokens(self.cold_cache),
            "uniform_baseline_bits": uniform_total,
            "savings_vs_uniform_pct": savings_pct,
        }

    def clear(self) -> None:
        """Clear all cached key-value pairs across every tier."""
        self.hot_cache.clear()
        self.warm_cache.clear()
        self.cold_cache.clear()
        self._total_tokens = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _maybe_decay(self) -> None:
        """Move tokens between tiers when they exceed window sizes."""
        # Hot -> Warm overflow
        while self._count_tokens(self.hot_cache) > self.hot_window:
            self._demote_oldest(self.hot_cache, self.warm_cache)

        # Warm -> Cold overflow
        while self._count_tokens(self.warm_cache) > self.warm_window:
            self._demote_oldest(self.warm_cache, self.cold_cache)

    def _demote_oldest(
        self,
        source: TurboQuantKVCache,
        dest: TurboQuantKVCache,
    ) -> None:
        """Reconstruct the oldest entry from *source* and re-quantize into *dest*.

        This is lossy re-quantization:
        1. Pop the oldest compressed entry from *source*.
        2. Reconstruct the key via MSE dequantization (QJL correction is lost).
        3. Reconstruct the value via PolarQuant dequantization.
        4. Re-compress and append into *dest* at the lower precision.

        The quality cost is that the QJL bias correction from the source tier
        is discarded.  This is acceptable because old tokens receive very
        little attention weight.
        """
        # ---- Pop the oldest entry from source's internal lists ----
        key_entry = source._key_store.pop(0)
        val_indices = source._value_indices.pop(0)
        val_norms = source._value_norms.pop(0)

        # ---- Reconstruct keys ----
        # key_entry is a dict with mse_indices, qjl_signs, residual_norm, vec_norm.
        # We reconstruct via MSE dequantization (Stage 1 only) and rescale by
        # the stored vector norm.  This mirrors TurboQuantEstimator.dequantize_mse.
        mse_indices = key_entry["mse_indices"]
        vec_norm = key_entry["vec_norm"]

        # Ensure 2-D for dequantization
        squeeze_key = mse_indices.dim() == 1
        if squeeze_key:
            mse_indices = mse_indices.unsqueeze(0)
            vec_norm = vec_norm.unsqueeze(0)

        key_recon_normalized = source.key_quantizer.polar.dequantize(mse_indices)
        key_recon = key_recon_normalized * vec_norm.unsqueeze(-1)

        if squeeze_key:
            key_recon = key_recon.squeeze(0)

        # ---- Reconstruct values ----
        squeeze_val = val_indices.dim() == 1
        if squeeze_val:
            val_indices = val_indices.unsqueeze(0)
            val_norms = val_norms.unsqueeze(0)

        val_recon_normalized = source.value_quantizer.dequantize(val_indices)
        val_recon = val_recon_normalized * val_norms.unsqueeze(-1)

        if squeeze_val:
            val_recon = val_recon.squeeze(0)

        # ---- Re-compress into destination cache ----
        dest.append(key_recon, val_recon)

    @staticmethod
    def _count_tokens(cache: TurboQuantKVCache) -> int:
        """Count the actual number of tokens stored in a cache.

        A single ``_key_store`` entry may hold a batch of tokens, so we sum
        across all entries.
        """
        total = 0
        for ks in cache._key_store:
            mi = ks["mse_indices"]
            if mi.dim() > 1:
                total += mi.shape[0]
            else:
                total += 1
        return total
