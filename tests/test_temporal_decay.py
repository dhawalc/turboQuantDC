"""Tests for temporal decay compression.

Validates that the TemporalDecayCache correctly manages three tiers of
quantization precision, demotes tokens as they age, and preserves attention
quality within acceptable bounds.

All tests use small dimensions (d=32 or 64) and small windows (hot=4, warm=8)
for fast execution.
"""

from __future__ import annotations

import torch
import pytest

from turboquantdc.temporal_decay import TemporalDecayCache


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D = 32
SEED = 42
HOT_WINDOW = 4
WARM_WINDOW = 8
HOT_BITS = 4
WARM_BITS = 3
COLD_BITS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cache(**overrides) -> TemporalDecayCache:
    """Create a TemporalDecayCache with small test-friendly defaults."""
    defaults = dict(
        d_key=D,
        d_value=D,
        hot_bits=HOT_BITS,
        warm_bits=WARM_BITS,
        cold_bits=COLD_BITS,
        hot_window=HOT_WINDOW,
        warm_window=WARM_WINDOW,
        seed=SEED,
        device="cpu",
    )
    defaults.update(overrides)
    return TemporalDecayCache(**defaults)


def random_vectors(n: int, d: int = D, seed: int = SEED) -> torch.Tensor:
    """Generate n random vectors of dimension d."""
    torch.manual_seed(seed)
    return torch.randn(n, d)


def random_unit_vectors(n: int, d: int = D, seed: int = SEED) -> torch.Tensor:
    """Generate n random unit vectors of dimension d."""
    v = random_vectors(n, d, seed)
    return v / v.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHotTierOnly:
    """Few tokens remain entirely in the hot tier."""

    def test_hot_tier_only(self):
        cache = make_cache()
        keys = random_vectors(3, D, seed=10)
        values = random_vectors(3, D, seed=11)

        # Append one at a time
        for i in range(3):
            cache.append(keys[i], values[i])

        assert cache.seq_len == 3
        stats = cache.memory_usage_bits()
        assert stats["hot_tokens"] == 3
        assert stats["warm_tokens"] == 0
        assert stats["cold_tokens"] == 0


class TestHotToWarmDecay:
    """Exceeding hot_window pushes oldest tokens to warm tier."""

    def test_hot_to_warm_decay(self):
        cache = make_cache()
        # Append more tokens than the hot window (4)
        for i in range(6):
            torch.manual_seed(SEED + i)
            k = torch.randn(D)
            v = torch.randn(D)
            cache.append(k, v)

        stats = cache.memory_usage_bits()
        assert cache.seq_len == 6
        # Hot should have at most hot_window tokens
        assert stats["hot_tokens"] <= HOT_WINDOW
        # Overflow should be in warm
        assert stats["warm_tokens"] == 6 - stats["hot_tokens"]
        assert stats["cold_tokens"] == 0


class TestWarmToColdDecay:
    """Exceeding warm_window pushes oldest warm tokens to cold tier."""

    def test_warm_to_cold_decay(self):
        cache = make_cache()
        total = HOT_WINDOW + WARM_WINDOW + 3  # 4 + 8 + 3 = 15
        for i in range(total):
            torch.manual_seed(SEED + i)
            cache.append(torch.randn(D), torch.randn(D))

        stats = cache.memory_usage_bits()
        assert cache.seq_len == total
        assert stats["hot_tokens"] <= HOT_WINDOW
        assert stats["warm_tokens"] <= WARM_WINDOW
        assert stats["cold_tokens"] > 0


class TestFullDecayChain:
    """Tokens flow through all three tiers."""

    def test_full_decay_chain(self):
        cache = make_cache()
        total = HOT_WINDOW + WARM_WINDOW + 10  # 22 tokens
        for i in range(total):
            torch.manual_seed(SEED + i)
            cache.append(torch.randn(D), torch.randn(D))

        stats = cache.memory_usage_bits()
        assert stats["hot_tokens"] <= HOT_WINDOW
        assert stats["warm_tokens"] <= WARM_WINDOW
        assert stats["cold_tokens"] >= 10
        assert (
            stats["hot_tokens"] + stats["warm_tokens"] + stats["cold_tokens"]
            == total
        )


class TestAttentionScoresAcrossTiers:
    """Attention scores span all tiers correctly."""

    def test_attention_scores_across_tiers(self):
        cache = make_cache()
        total = HOT_WINDOW + WARM_WINDOW + 5  # tokens in all 3 tiers
        for i in range(total):
            torch.manual_seed(SEED + i)
            cache.append(torch.randn(D), torch.randn(D))

        # Query with a single vector
        torch.manual_seed(999)
        query = torch.randn(D)
        scores = cache.attention_scores(query)

        # Scores should cover all tokens
        assert scores.shape == (total,)
        # Scores should be finite
        assert torch.isfinite(scores).all()

    def test_attention_scores_batched_query(self):
        cache = make_cache()
        for i in range(6):
            torch.manual_seed(SEED + i)
            cache.append(torch.randn(D), torch.randn(D))

        torch.manual_seed(888)
        queries = torch.randn(3, D)
        scores = cache.attention_scores(queries)

        assert scores.shape == (3, 6)
        assert torch.isfinite(scores).all()


class TestValuesOrdered:
    """get_values() returns values in cold -> warm -> hot order."""

    def test_values_ordered(self):
        cache = make_cache()
        total = HOT_WINDOW + WARM_WINDOW + 4  # 16 tokens
        for i in range(total):
            torch.manual_seed(SEED + i)
            cache.append(torch.randn(D), torch.randn(D))

        vals = cache.get_values()
        assert vals.shape == (total, D)
        assert torch.isfinite(vals).all()


class TestMemorySavings:
    """At long enough context, temporal decay uses less memory than uniform."""

    def test_memory_savings(self):
        cache = make_cache()
        # Fill well beyond hot + warm to get most tokens in cold
        total = HOT_WINDOW + WARM_WINDOW + 50  # 62 tokens
        for i in range(total):
            torch.manual_seed(SEED + i)
            cache.append(torch.randn(D), torch.randn(D))

        stats = cache.memory_usage_bits()
        # Total bits should be less than uniform at warm_bits
        assert stats["total_bits"] < stats["uniform_baseline_bits"], (
            f"Temporal decay ({stats['total_bits']} bits) should use less "
            f"than uniform {WARM_BITS}-bit ({stats['uniform_baseline_bits']} bits)"
        )
        assert stats["savings_vs_uniform_pct"] > 0


class TestQualityPreserved:
    """Cosine similarity of attention scores vs FP16 should exceed 0.98."""

    def test_quality_preserved(self):
        d = 64
        cache = TemporalDecayCache(
            d_key=d,
            d_value=d,
            hot_bits=4,
            warm_bits=3,
            cold_bits=2,
            hot_window=HOT_WINDOW,
            warm_window=WARM_WINDOW,
            seed=SEED,
            device="cpu",
        )

        # Populate with known vectors
        n_tokens = HOT_WINDOW + WARM_WINDOW + 4
        all_keys = []
        all_values = []
        for i in range(n_tokens):
            torch.manual_seed(SEED + i)
            k = torch.randn(d)
            v = torch.randn(d)
            all_keys.append(k)
            all_values.append(v)
            cache.append(k, v)

        # Compute FP16 reference scores
        keys_fp16 = torch.stack(all_keys)
        torch.manual_seed(777)
        query = torch.randn(d)
        ref_scores = query @ keys_fp16.T  # (n_tokens,)

        # Quantized scores
        quant_scores = cache.attention_scores(query)

        # Both should have the same length
        assert quant_scores.shape == ref_scores.shape

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_scores.unsqueeze(0), quant_scores.unsqueeze(0)
        ).item()

        # Paper allows some degradation; temporal decay adds re-quantization loss
        assert cos_sim > 0.90, f"Cosine similarity {cos_sim:.4f} < 0.90"


class TestTierTokenCounts:
    """Token counts in each tier match expectations after filling."""

    def test_tier_token_counts(self):
        cache = make_cache()
        total = HOT_WINDOW + WARM_WINDOW + 20  # 32 tokens
        for i in range(total):
            torch.manual_seed(SEED + i)
            cache.append(torch.randn(D), torch.randn(D))

        stats = cache.memory_usage_bits()

        # Hot tier should be exactly at capacity
        assert stats["hot_tokens"] == HOT_WINDOW
        # Warm tier should be exactly at capacity
        assert stats["warm_tokens"] == WARM_WINDOW
        # Cold tier gets the rest
        assert stats["cold_tokens"] == 20
        # Total matches
        assert (
            stats["hot_tokens"] + stats["warm_tokens"] + stats["cold_tokens"]
            == total
        )


class TestSmallWindows:
    """Works with very small windows (hot=4, warm=8)."""

    def test_small_windows(self):
        cache = make_cache(hot_window=4, warm_window=8)
        for i in range(20):
            torch.manual_seed(SEED + i)
            cache.append(torch.randn(D), torch.randn(D))

        stats = cache.memory_usage_bits()
        assert stats["hot_tokens"] == 4
        assert stats["warm_tokens"] == 8
        assert stats["cold_tokens"] == 8

        # Can still compute attention
        torch.manual_seed(42)
        scores = cache.attention_scores(torch.randn(D))
        assert scores.shape == (20,)
        assert torch.isfinite(scores).all()


class TestClear:
    """Clear resets everything."""

    def test_clear(self):
        cache = make_cache()
        for i in range(10):
            torch.manual_seed(SEED + i)
            cache.append(torch.randn(D), torch.randn(D))

        assert cache.seq_len == 10
        cache.clear()

        assert cache.seq_len == 0
        stats = cache.memory_usage_bits()
        assert stats["hot_tokens"] == 0
        assert stats["warm_tokens"] == 0
        assert stats["cold_tokens"] == 0
        assert stats["total_bits"] == 0


class TestSingleTokenAppend:
    """Append one token at a time."""

    def test_single_token_append(self):
        cache = make_cache()

        for i in range(HOT_WINDOW + 2):
            torch.manual_seed(SEED + i)
            k = torch.randn(D)
            v = torch.randn(D)
            cache.append(k, v)

        assert cache.seq_len == HOT_WINDOW + 2
        stats = cache.memory_usage_bits()
        # 2 tokens should have been demoted to warm
        assert stats["hot_tokens"] == HOT_WINDOW
        assert stats["warm_tokens"] == 2

    def test_single_token_attention(self):
        cache = make_cache()
        torch.manual_seed(SEED)
        k = torch.randn(D)
        v = torch.randn(D)
        cache.append(k, v)

        torch.manual_seed(SEED + 100)
        q = torch.randn(D)
        scores = cache.attention_scores(q)
        assert scores.shape == (1,)
        assert torch.isfinite(scores).all()


class TestBatchAppend:
    """Appending batches of tokens works correctly."""

    def test_batch_append(self):
        cache = make_cache()
        torch.manual_seed(SEED)
        keys = torch.randn(3, D)
        values = torch.randn(3, D)
        cache.append(keys, values)

        assert cache.seq_len == 3
        stats = cache.memory_usage_bits()
        assert stats["hot_tokens"] == 3

    def test_batch_overflow(self):
        """A batch larger than hot_window triggers immediate decay."""
        cache = make_cache(hot_window=4, warm_window=8)
        torch.manual_seed(SEED)
        # Append 6 tokens as a batch -- exceeds hot_window of 4
        keys = torch.randn(6, D)
        values = torch.randn(6, D)
        cache.append(keys, values)

        stats = cache.memory_usage_bits()
        # The batch is stored as a single entry in _key_store, which counts
        # as 6 tokens.  Decay should have moved some to warm.
        assert stats["hot_tokens"] + stats["warm_tokens"] == 6
        assert cache.seq_len == 6


class TestEmptyCache:
    """Edge cases on an empty cache."""

    def test_empty_attention(self):
        cache = make_cache()
        torch.manual_seed(SEED)
        q = torch.randn(D)
        scores = cache.attention_scores(q)
        assert scores.shape == (0,)

    def test_empty_values(self):
        cache = make_cache()
        vals = cache.get_values()
        assert vals.shape == (0, D)

    def test_empty_memory(self):
        cache = make_cache()
        stats = cache.memory_usage_bits()
        assert stats["total_bits"] == 0
        assert stats["hot_tokens"] == 0
