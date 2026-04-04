"""Attention-aware adaptive bit allocation for TurboQuant KV cache.

Allocates bits based on token importance for future attention. Tokens that
consistently receive high attention scores are kept at higher precision,
while tokens that receive little attention are compressed more aggressively.

The attention distribution in transformer models follows a power law: a small
fraction of tokens receive the vast majority of attention weight. By matching
compression level to importance, we can achieve uniform-3-bit quality at
significantly lower effective bit-rates.

Tier structure (configurable):
    Tier 0 (critical): FP16 -- no compression
    Tier 1 (important): 4-bit ResidualQuant
    Tier 2 (normal): 3-bit ResidualQuant
    Tier 3 (unimportant): 2-bit or 1-bit

Importance scoring:
    Exponential moving average of attention weights received by each token.
    Updated after each decode step from the actual attention distribution.

Usage::

    from turboquantdc.adaptive_bits import AdaptiveBitsCache

    cache = AdaptiveBitsCache(
        tier_thresholds=[0.05, 0.20, 0.50],  # top 5%, next 15%, next 30%, bottom 50%
        tier_bits=[16, 4, 3, 2],
    )
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .codebook import LloydMaxCodebook
from .rotation import generate_rotation_matrix, generate_wht_rotation, apply_wht_rotation


# ---------------------------------------------------------------------------
# Importance scorer
# ---------------------------------------------------------------------------


class ImportanceScorer:
    """Track per-token importance via exponential moving average of attention.

    Each token's importance is the EMA of the total attention weight it
    receives across all queries in each decode step.

    Args:
        ema_decay: Decay factor for exponential moving average (default: 0.9).
            Higher values give more weight to recent attention patterns.
    """

    def __init__(self, ema_decay: float = 0.9):
        self.ema_decay = ema_decay
        self._scores: Optional[torch.Tensor] = None
        self._update_count: int = 0

    @property
    def scores(self) -> Optional[torch.Tensor]:
        """Current importance scores, shape (seq_len,)."""
        return self._scores

    @property
    def seq_len(self) -> int:
        return 0 if self._scores is None else self._scores.shape[0]

    def update(self, attention_weights: torch.Tensor, query_window: int = 64) -> None:
        """Update importance scores from attention weights.

        Uses the LAST query_window queries for decode-relevant importance.
        This avoids the causal averaging artifact where all tokens appear
        equally important when averaging across the full causal mask.

        Args:
            attention_weights: Attention weight matrix.
                Shape: (batch, heads, query_len, kv_len) or (heads, query_len, kv_len).
                Each row should sum to 1 (softmax output).
            query_window: Number of last query positions to use (default: 64).
        """
        # Normalize to (heads, query_len, kv_len)
        if attention_weights.dim() == 4:
            attn = attention_weights.mean(dim=0)  # average across batch
        elif attention_weights.dim() == 3:
            attn = attention_weights
        else:
            raise ValueError(
                f"Expected 3D or 4D attention weights, got {attention_weights.dim()}D"
            )

        # Use last query_window queries (decode-relevant)
        n_queries = min(query_window, attn.shape[1])
        attn = attn[:, -n_queries:, :]  # (heads, n_queries, kv_len)

        # Average attention received by each KV token from decode queries across heads
        # attn: (heads, n_queries, kv_len) -> mean over queries -> (heads, kv_len) -> mean over heads -> (kv_len,)
        received = attn.mean(dim=1).mean(dim=0)  # (kv_len,)

        # Normalize to sum to 1
        received = received / received.sum().clamp(min=1e-10)

        kv_len = received.shape[0]

        if self._scores is None or self._scores.shape[0] < kv_len:
            # First update or sequence grew -- initialize new positions
            new_scores = torch.zeros(kv_len, device=received.device, dtype=torch.float32)
            if self._scores is not None:
                old_len = self._scores.shape[0]
                new_scores[:old_len] = self._scores.to(new_scores.device)
            self._scores = new_scores

        # EMA update: score = decay * old_score + (1 - decay) * new_received
        self._scores[:kv_len] = (
            self.ema_decay * self._scores[:kv_len]
            + (1 - self.ema_decay) * received.to(self._scores.device)
        )
        self._update_count += 1

    def classify_tiers(
        self,
        tier_thresholds: List[float],
    ) -> torch.Tensor:
        """Classify tokens into importance tiers.

        Args:
            tier_thresholds: Cumulative percentile thresholds.
                E.g., [0.05, 0.20, 0.50] means:
                - Tier 0: top 5% of tokens by importance
                - Tier 1: next 15% (5% to 20%)
                - Tier 2: next 30% (20% to 50%)
                - Tier 3: bottom 50%

        Returns:
            Tensor of tier assignments, shape (seq_len,), values 0..len(thresholds).
        """
        if self._scores is None:
            return torch.zeros(0, dtype=torch.long)

        n = self._scores.shape[0]
        tiers = torch.full((n,), len(tier_thresholds), dtype=torch.long)

        # Sort by importance (descending)
        sorted_indices = torch.argsort(self._scores, descending=True)

        # Assign tiers based on cumulative thresholds
        prev_cutoff = 0
        for tier_id, threshold in enumerate(tier_thresholds):
            cutoff = int(math.ceil(threshold * n))
            tiers[sorted_indices[prev_cutoff:cutoff]] = tier_id
            prev_cutoff = cutoff

        return tiers

    def reset(self) -> None:
        """Reset all importance scores."""
        self._scores = None
        self._update_count = 0


# ---------------------------------------------------------------------------
# Attention distribution analyzer
# ---------------------------------------------------------------------------


def analyze_attention_distribution(
    attention_weights: torch.Tensor,
    top_k_percentiles: List[float] = [0.01, 0.05, 0.10, 0.20, 0.50],
    query_window: int = 64,
) -> Dict[str, Any]:
    """Analyze the power-law distribution of attention across tokens.

    Uses the LAST query_window query positions to analyze attention patterns,
    simulating the decode phase. This is important because with causal
    attention, averaging across all query positions washes out the signal
    (each query only sees its prefix, making the aggregate look uniform).

    The decode-relevant view: how attention from the most recent queries
    distributes across all KV tokens. This reveals the true power-law
    structure that adaptive compression can exploit.

    Args:
        attention_weights: Full attention matrix from a forward pass.
            Shape: (batch, heads, seq_len, seq_len) or list of such tensors
            (one per layer).
        top_k_percentiles: Cumulative percentiles to analyze.
        query_window: Number of last query positions to analyze (default: 64).

    Returns:
        Dict with:
        - per_layer: list of per-layer stats
        - aggregate: stats averaged across all layers
        - power_law_strength: how concentrated attention is (higher = more power-law)
    """
    if isinstance(attention_weights, (list, tuple)):
        layers = attention_weights
    else:
        layers = [attention_weights]

    per_layer_stats = []

    for layer_idx, attn in enumerate(layers):
        if attn is None:
            continue

        # attn: (batch, heads, q_len, kv_len)
        if attn.dim() == 3:
            attn = attn.unsqueeze(0)

        batch, heads, q_len, kv_len = attn.shape

        # Use the LAST query_window queries to analyze attention patterns.
        # These represent the decode-phase view: "when generating text,
        # which KV tokens does the model actually look at?"
        n_queries = min(query_window, q_len)
        decode_attn = attn[:, :, -n_queries:, :]  # (batch, heads, n_queries, kv_len)

        # Average attention received by each KV token from decode queries.
        # For each head, average across queries -> (batch, heads, kv_len)
        # Then average across heads and batch -> (kv_len,)
        per_token_attention = decode_attn.mean(dim=2).mean(dim=(0, 1))  # (kv_len,)

        # Normalize to a proper distribution
        total_attn = per_token_attention.sum().clamp(min=1e-10)
        received_avg = per_token_attention / total_attn

        # Sort descending
        sorted_vals, sorted_idx = torch.sort(received_avg, descending=True)
        cumulative = torch.cumsum(sorted_vals, dim=0)
        total = cumulative[-1].item()

        # Compute concentration at each percentile
        concentration = {}
        for pct in top_k_percentiles:
            k = max(1, int(pct * kv_len))
            attn_captured = cumulative[min(k - 1, kv_len - 1)].item() / max(total, 1e-10)
            concentration[f"top_{pct:.0%}"] = attn_captured

        # Gini coefficient (measure of inequality)
        n = kv_len
        if n > 1:
            sorted_asc = torch.sort(received_avg)[0]
            index = torch.arange(1, n + 1, dtype=torch.float32, device=sorted_asc.device)
            gini = (2 * (index * sorted_asc).sum() / (n * sorted_asc.sum().clamp(min=1e-10)) - (n + 1) / n).item()
            gini = max(0.0, gini)  # clamp to non-negative
        else:
            gini = 0.0

        # Entropy of the attention distribution
        entropy = -(received_avg * torch.log(received_avg + 1e-10)).sum().item()
        max_entropy = math.log(max(kv_len, 1))
        normalized_entropy = entropy / max(max_entropy, 1e-10)

        per_layer_stats.append({
            "layer": layer_idx,
            "kv_len": kv_len,
            "concentration": concentration,
            "gini": gini,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "max_attention": sorted_vals[0].item(),
            "min_attention": sorted_vals[-1].item(),
            "median_attention": sorted_vals[kv_len // 2].item() if kv_len > 0 else 0.0,
        })

    # Aggregate across layers
    if per_layer_stats:
        agg_concentration = {}
        for key in per_layer_stats[0]["concentration"]:
            vals = [s["concentration"][key] for s in per_layer_stats]
            agg_concentration[key] = sum(vals) / len(vals)

        avg_gini = sum(s["gini"] for s in per_layer_stats) / len(per_layer_stats)
        avg_entropy = sum(s["normalized_entropy"] for s in per_layer_stats) / len(per_layer_stats)
    else:
        agg_concentration = {}
        avg_gini = 0.0
        avg_entropy = 0.0

    return {
        "per_layer": per_layer_stats,
        "aggregate": {
            "concentration": agg_concentration,
            "avg_gini": avg_gini,
            "avg_normalized_entropy": avg_entropy,
        },
        "power_law_strength": avg_gini,  # Gini coefficient: 0 = uniform, 1 = all attention on one token
        "n_layers_analyzed": len(per_layer_stats),
    }


# ---------------------------------------------------------------------------
# Adaptive Bits Cache
# ---------------------------------------------------------------------------


class AdaptiveBitsCache:
    """KV cache with attention-aware adaptive bit allocation.

    Tokens are classified into importance tiers based on attention patterns.
    Higher-importance tokens get more bits, lower-importance tokens get fewer.
    This exploits the power-law attention distribution to achieve better
    quality at the same effective bit-rate, or same quality at fewer bits.

    The cache starts with all tokens at the highest compressed tier (tier 1).
    After initial warmup, it reclassifies tokens periodically and adjusts
    their compression level.

    Args:
        tier_thresholds: Cumulative percentile boundaries for tiers.
            Default [0.05, 0.20, 0.50] = top 5%, next 15%, next 30%, bottom 50%.
        tier_bits: Bits for each tier. Default [16, 4, 3, 2].
            First element is for the most important tokens.
        ema_decay: EMA decay for importance scoring (default: 0.9).
        reclassify_interval: Reclassify tokens every N decode steps (default: 16).
        warmup_steps: Number of decode steps before first reclassification (default: 32).
        d: Head dimension (required).
        seed: Random seed for reproducibility.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        tier_thresholds: Optional[List[float]] = None,
        tier_bits: Optional[List[int]] = None,
        ema_decay: float = 0.9,
        reclassify_interval: int = 16,
        warmup_steps: int = 32,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        self.d = d
        self.tier_thresholds = tier_thresholds or [0.05, 0.20, 0.50]
        self.tier_bits = tier_bits or [16, 4, 3, 2]
        self.reclassify_interval = reclassify_interval
        self.warmup_steps = warmup_steps
        self.seed = seed
        self.device = device

        assert len(self.tier_bits) == len(self.tier_thresholds) + 1, (
            f"Need {len(self.tier_thresholds) + 1} tier_bits entries, "
            f"got {len(self.tier_bits)}"
        )

        self.scorer = ImportanceScorer(ema_decay=ema_decay)

        # Per-tier codebooks (lazily initialized for non-FP16 tiers)
        self._codebooks: Dict[int, LloydMaxCodebook] = {}
        self._rotations: Dict[int, Any] = {}

        # Storage: each token's data
        self._keys_fp16: List[torch.Tensor] = []  # original FP16 keys
        self._values_fp16: List[torch.Tensor] = []  # original FP16 values
        self._tier_assignments: Optional[torch.Tensor] = None
        self._step_count: int = 0
        self._seq_len: int = 0

    def _get_codebook(self, bits: int) -> LloydMaxCodebook:
        """Get or create a codebook for the given bit-width."""
        if bits not in self._codebooks:
            cb = LloydMaxCodebook(d=self.d, bits=bits)
            self._codebooks[bits] = cb
        return self._codebooks[bits]

    def _get_rotation(self, bits: int) -> torch.Tensor:
        """Get or create a rotation matrix for the given bit-width."""
        if bits not in self._rotations:
            seed = self.seed + bits * 100
            rot = generate_rotation_matrix(self.d, seed=seed, device=str(self.device))
            self._rotations[bits] = rot
        return self._rotations[bits]

    def append_tokens(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Add new tokens to the cache.

        Args:
            keys: (batch, d) or (d,) key vectors.
            values: (batch, d) or (d,) value vectors.
        """
        if keys.dim() == 1:
            keys = keys.unsqueeze(0)
            values = values.unsqueeze(0)

        self._keys_fp16.append(keys.detach().float())
        self._values_fp16.append(values.detach().float())
        self._seq_len += keys.shape[0]

    def update_importance(self, attention_weights: torch.Tensor) -> None:
        """Update importance scores from attention weights.

        Args:
            attention_weights: (batch, heads, query_len, kv_len) attention matrix.
        """
        self.scorer.update(attention_weights)
        self._step_count += 1

    def reclassify(self) -> torch.Tensor:
        """Reclassify all tokens into tiers based on current importance.

        Returns:
            Tier assignments tensor, shape (seq_len,).
        """
        self._tier_assignments = self.scorer.classify_tiers(self.tier_thresholds)
        return self._tier_assignments

    def get_compressed_keys(self) -> torch.Tensor:
        """Get keys with per-tier compression applied.

        Returns:
            (seq_len, d) tensor of keys at mixed precision.
        """
        if self._seq_len == 0:
            return torch.zeros(0, self.d, device=self.device)

        all_keys = torch.cat(self._keys_fp16, dim=0)  # (seq_len, d)

        if self._tier_assignments is None:
            return all_keys  # no reclassification yet, return FP16

        result = torch.zeros_like(all_keys)

        for tier_id, bits in enumerate(self.tier_bits):
            mask = self._tier_assignments == tier_id
            if not mask.any():
                continue

            tier_keys = all_keys[mask]

            if bits >= 16:
                # FP16 tier: no compression
                result[mask] = tier_keys
            else:
                # Quantize and dequantize at this bit-width
                cb = self._get_codebook(bits)
                rot = self._get_rotation(bits)

                norms = tier_keys.norm(dim=-1, keepdim=True)
                normalized = tier_keys / (norms + 1e-8)
                rotated = normalized @ rot

                indices = torch.bucketize(rotated, cb.boundaries)
                indices = indices.clamp(0, cb.centroids.shape[0] - 1)

                reconstructed = cb.centroids[indices]
                unrotated = reconstructed @ rot.T
                result[mask] = unrotated * norms

        return result

    def get_compressed_values(self) -> torch.Tensor:
        """Get values with per-tier compression (MSE only).

        Returns:
            (seq_len, d) tensor of values at mixed precision.
        """
        if self._seq_len == 0:
            return torch.zeros(0, self.d, device=self.device)

        all_values = torch.cat(self._values_fp16, dim=0)

        if self._tier_assignments is None:
            return all_values

        result = torch.zeros_like(all_values)

        for tier_id, bits in enumerate(self.tier_bits):
            mask = self._tier_assignments == tier_id
            if not mask.any():
                continue

            tier_vals = all_values[mask]

            if bits >= 16:
                result[mask] = tier_vals
            else:
                cb = self._get_codebook(bits)
                rot = self._get_rotation(bits)

                norms = tier_vals.norm(dim=-1, keepdim=True)
                normalized = tier_vals / (norms + 1e-8)
                rotated = normalized @ rot

                indices = torch.bucketize(rotated, cb.boundaries)
                indices = indices.clamp(0, cb.centroids.shape[0] - 1)

                reconstructed = cb.centroids[indices]
                unrotated = reconstructed @ rot.T
                result[mask] = unrotated * norms

        return result

    def effective_bits(self) -> float:
        """Compute the weighted-average effective bits across all tokens.

        Returns:
            Average bits per coordinate across all tokens.
        """
        if self._tier_assignments is None or self._seq_len == 0:
            return 16.0  # no reclassification yet

        total = 0.0
        for tier_id, bits in enumerate(self.tier_bits):
            count = (self._tier_assignments == tier_id).sum().item()
            total += count * bits
        return total / max(self._seq_len, 1)

    def tier_distribution(self) -> Dict[str, Any]:
        """Report how many tokens are in each tier.

        Returns:
            Dict with per-tier token counts and percentages.
        """
        if self._tier_assignments is None:
            return {"tiers": [], "total": self._seq_len}

        tiers = []
        for tier_id, bits in enumerate(self.tier_bits):
            count = (self._tier_assignments == tier_id).sum().item()
            pct = count / max(self._seq_len, 1)
            tiers.append({
                "tier": tier_id,
                "bits": bits,
                "count": count,
                "percentage": pct,
            })

        return {
            "tiers": tiers,
            "total": self._seq_len,
            "effective_bits": self.effective_bits(),
        }

    def clear(self) -> None:
        """Reset all state."""
        self._keys_fp16.clear()
        self._values_fp16.clear()
        self._tier_assignments = None
        self._step_count = 0
        self._seq_len = 0
        self.scorer.reset()
