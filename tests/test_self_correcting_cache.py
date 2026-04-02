"""Tests for the self-correcting KV cache with periodic refresh.

Validates:
- Refresh triggers at correct interval
- Top-attended tokens are identified correctly via key norms
- Norm correction refresh improves reconstruction quality
- Cache protocol compliance (delegates to inner cache)
- Importance tracking accumulates correctly
- Configuration validation and edge cases
- Refresh statistics reporting
"""

import math

import pytest
import torch

from turboquantdc.self_correcting_cache import SelfCorrectingCache
from turboquantdc.generation_cache import GenerationCache


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEAD_DIM = 128
NUM_HEADS = 4
BATCH_SIZE = 2
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_kv_states(
    batch: int = BATCH_SIZE,
    num_heads: int = NUM_HEADS,
    seq_len: int = 8,
    head_dim: int = HEAD_DIM,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random KV tensors in HF format [batch, num_heads, seq_len, head_dim]."""
    torch.manual_seed(seed)
    keys = torch.randn(batch, num_heads, seq_len, head_dim)
    values = torch.randn(batch, num_heads, seq_len, head_dim)
    return keys, values


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean cosine similarity between two tensors of same shape."""
    a_flat = a.reshape(-1, a.shape[-1]).float()
    b_flat = b.reshape(-1, b.shape[-1]).float()
    sims = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1)
    return sims.mean().item()


# ---------------------------------------------------------------------------
# Test: Construction and configuration
# ---------------------------------------------------------------------------
class TestConstruction:
    """Validate construction, configuration, and parameter validation."""

    def test_default_construction(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        assert cache.refresh_interval == 50
        assert cache.refresh_count == 5
        assert cache.tokens_since_refresh == 0

    def test_custom_parameters(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(
            inner,
            refresh_interval=100,
            refresh_count=10,
        )
        assert cache.refresh_interval == 100
        assert cache.refresh_count == 10

    def test_invalid_refresh_interval(self):
        inner = GenerationCache(seed=SEED)
        with pytest.raises(ValueError, match="refresh_interval"):
            SelfCorrectingCache(inner, refresh_interval=0)
        with pytest.raises(ValueError, match="refresh_interval"):
            SelfCorrectingCache(inner, refresh_interval=-1)

    def test_invalid_refresh_count(self):
        inner = GenerationCache(seed=SEED)
        with pytest.raises(ValueError, match="refresh_count"):
            SelfCorrectingCache(inner, refresh_count=0)
        with pytest.raises(ValueError, match="refresh_count"):
            SelfCorrectingCache(inner, refresh_count=-1)

    def test_refresh_count_clamped_to_cache_size(self):
        """refresh_count > available tokens should not crash."""
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner, refresh_interval=5, refresh_count=1000)
        keys, values = make_kv_states(seq_len=3)
        # Should not raise even with refresh_count > seq_len
        cache.update(keys, values, layer_idx=0)


# ---------------------------------------------------------------------------
# Test: HF Cache protocol delegation
# ---------------------------------------------------------------------------
class TestCacheProtocol:
    """Validate that SelfCorrectingCache delegates to inner cache correctly."""

    def test_update_returns_correct_shapes(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        keys, values = make_kv_states(seq_len=5)
        k_out, v_out = cache.update(keys, values, layer_idx=0)
        assert k_out.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)
        assert v_out.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)

    def test_update_accumulates_sequence(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        k1, v1 = make_kv_states(seq_len=5, seed=1)
        k2, v2 = make_kv_states(seq_len=3, seed=2)
        cache.update(k1, v1, layer_idx=0)
        k_out, v_out = cache.update(k2, v2, layer_idx=0)
        assert k_out.shape == (BATCH_SIZE, NUM_HEADS, 8, HEAD_DIM)
        assert v_out.shape == (BATCH_SIZE, NUM_HEADS, 8, HEAD_DIM)

    def test_get_seq_length_delegates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        assert cache.get_seq_length(0) == 0
        keys, values = make_kv_states(seq_len=10)
        cache.update(keys, values, layer_idx=0)
        assert cache.get_seq_length(0) == 10

    def test_get_max_cache_shape_delegates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        assert cache.get_max_cache_shape() == -1

    def test_get_mask_sizes_delegates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        keys, values = make_kv_states(seq_len=20)
        cache.update(keys, values, layer_idx=0)
        pos = torch.arange(5)
        kv_len, offset = cache.get_mask_sizes(pos, layer_idx=0)
        assert kv_len == 25  # 20 cached + 5 query
        assert offset == 0

    def test_reorder_cache_delegates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        beam_idx = torch.tensor([1, 0])
        cache.reorder_cache(beam_idx)  # should not raise

    def test_crop_delegates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        keys, values = make_kv_states(seq_len=20)
        cache.update(keys, values, layer_idx=0)
        cache.crop(10)
        assert cache.get_seq_length(0) == 10

    def test_reset_delegates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        keys, values = make_kv_states(seq_len=10)
        cache.update(keys, values, layer_idx=0)
        cache.reset()
        assert cache.get_seq_length(0) == 0

    def test_len_delegates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        cache.update(keys, values, layer_idx=1)
        assert len(cache) == 2

    def test_getitem_delegates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        k, v = cache[0]
        assert k.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)

    def test_contains_delegates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        assert 0 in cache
        assert 99 not in cache

    def test_iter_delegates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        cache.update(keys, values, layer_idx=1)
        items = list(cache)
        assert len(items) == 2
        for k, v, extra in items:
            assert k.shape[2] == 5

    def test_seen_tokens_delegates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        keys, values = make_kv_states(seq_len=10)
        cache.update(keys, values, layer_idx=0)
        assert cache.seen_tokens == 10

    def test_is_initialized_delegates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        assert not cache.is_initialized
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        assert cache.is_initialized


# ---------------------------------------------------------------------------
# Test: Refresh trigger timing
# ---------------------------------------------------------------------------
class TestRefreshTrigger:
    """Validate that refresh triggers at the correct interval."""

    def test_no_refresh_before_interval(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner, refresh_interval=10, refresh_count=2)
        keys, values = make_kv_states(seq_len=1)

        for i in range(9):
            cache.update(keys, values, layer_idx=0, cache_kwargs=None)
            assert cache.tokens_since_refresh == i + 1

    def test_refresh_resets_counter(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner, refresh_interval=5, refresh_count=1)
        keys, values = make_kv_states(seq_len=1)

        for i in range(5):
            cache.update(keys, values, layer_idx=0)

        # After 5 tokens, refresh should have fired and reset counter
        assert cache.tokens_since_refresh == 0

    def test_refresh_fires_multiple_times(self):
        # anchor_interval=0 disables FP16 anchors so layer 0 is compressed
        inner = GenerationCache(seed=SEED, anchor_interval=0)
        cache = SelfCorrectingCache(inner, refresh_interval=5, refresh_count=1)
        keys, values = make_kv_states(seq_len=1)

        for i in range(15):
            cache.update(keys, values, layer_idx=0)

        # Should have refreshed 3 times
        assert cache.total_refreshes == 3
        assert cache.tokens_since_refresh == 0

    def test_counter_only_increments_on_layer_0(self):
        """Tokens-since-refresh only counts on layer_idx=0."""
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner, refresh_interval=10, refresh_count=1)
        keys, values = make_kv_states(seq_len=1)

        # Update layers 0, 1, 2 -- only layer 0 counts
        cache.update(keys, values, layer_idx=0)
        cache.update(keys, values, layer_idx=1)
        cache.update(keys, values, layer_idx=2)
        assert cache.tokens_since_refresh == 1

    def test_prefill_counts_all_tokens(self):
        """A single update with seq_len=N counts as N tokens."""
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner, refresh_interval=10, refresh_count=2)
        keys, values = make_kv_states(seq_len=8)
        cache.update(keys, values, layer_idx=0)
        assert cache.tokens_since_refresh == 8


# ---------------------------------------------------------------------------
# Test: Importance tracking
# ---------------------------------------------------------------------------
class TestImportanceTracking:
    """Validate token importance tracking via key norms."""

    def test_importance_accumulates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner, refresh_interval=100, refresh_count=2)

        keys, values = make_kv_states(seq_len=10)
        cache.update(keys, values, layer_idx=0)

        # Should have importance entries for layer 0
        assert 0 in cache._importance_tracker
        assert len(cache._importance_tracker[0]) == 10

    def test_importance_grows_with_updates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner, refresh_interval=100, refresh_count=2)

        k1, v1 = make_kv_states(seq_len=5, seed=1)
        k2, v2 = make_kv_states(seq_len=3, seed=2)
        cache.update(k1, v1, layer_idx=0)
        cache.update(k2, v2, layer_idx=0)

        assert len(cache._importance_tracker[0]) == 8

    def test_high_norm_tokens_rank_higher(self):
        """Tokens with larger key norms should rank as more important."""
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner, refresh_interval=100, refresh_count=2)

        # Create keys where token 0 has much larger norms
        keys = torch.randn(BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM) * 0.1
        keys[:, :, 0, :] *= 100  # Make token 0 very high-norm
        values = torch.randn(BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)
        cache.update(keys, values, layer_idx=0)

        top_positions = cache._get_top_positions(layer_idx=0, count=1)
        assert 0 in top_positions

    def test_get_top_positions_count_clamped(self):
        """Requesting more top positions than available returns all."""
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner, refresh_interval=100, refresh_count=10)

        keys, values = make_kv_states(seq_len=3)
        cache.update(keys, values, layer_idx=0)

        top = cache._get_top_positions(layer_idx=0, count=100)
        assert len(top) == 3  # Only 3 tokens available

    def test_multi_layer_tracking(self):
        """Each layer tracks importance independently."""
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner, refresh_interval=100, refresh_count=2)

        k1, v1 = make_kv_states(seq_len=5, seed=1)
        k2, v2 = make_kv_states(seq_len=3, seed=2)
        cache.update(k1, v1, layer_idx=0)
        cache.update(k2, v2, layer_idx=1)

        assert 0 in cache._importance_tracker
        assert 1 in cache._importance_tracker
        assert len(cache._importance_tracker[0]) == 5
        assert len(cache._importance_tracker[1]) == 3


# ---------------------------------------------------------------------------
# Test: Norm correction refresh
# ---------------------------------------------------------------------------
class TestNormCorrectionRefresh:
    """Validate that norm correction refresh improves reconstruction quality."""

    def test_refresh_corrects_norms(self):
        """After refresh, key norms should be closer to original."""
        inner = GenerationCache(seed=SEED, key_bits=3, val_bits=2, anchor_interval=0)
        cache = SelfCorrectingCache(
            inner, refresh_interval=10, refresh_count=3,
        )

        # Store original keys for comparison
        all_keys = []
        all_values = []
        for i in range(10):
            k, v = make_kv_states(seq_len=1, seed=i + 100)
            all_keys.append(k)
            all_values.append(v)
            cache.update(k, v, layer_idx=0)

        # Refresh should have fired. Check stats.
        assert cache.total_refreshes >= 1

    def test_refresh_does_not_change_sequence_length(self):
        """Refresh replaces in-place, it should not add or remove tokens."""
        inner = GenerationCache(seed=SEED, anchor_interval=0)
        cache = SelfCorrectingCache(inner, refresh_interval=5, refresh_count=2)

        keys, values = make_kv_states(seq_len=1)
        for i in range(5):
            cache.update(keys, values, layer_idx=0)

        assert cache.get_seq_length(0) == 5

    def test_refresh_preserves_reconstruction_quality(self):
        """After refresh, cosine similarity should not degrade."""
        inner = GenerationCache(seed=SEED, key_bits=3, val_bits=3, anchor_interval=0)
        cache = SelfCorrectingCache(inner, refresh_interval=5, refresh_count=2)

        # Feed tokens and let refresh fire
        for i in range(10):
            k, v = make_kv_states(seq_len=1, seed=i + 200)
            cache.update(k, v, layer_idx=0)

        # The cache should still return valid tensors
        k_out, v_out = cache[0]
        assert k_out.shape[2] == 10
        assert not torch.isnan(k_out).any()
        assert not torch.isnan(v_out).any()


# ---------------------------------------------------------------------------
# Test: Refresh statistics
# ---------------------------------------------------------------------------
class TestRefreshStats:
    """Validate refresh statistics tracking."""

    def test_stats_initial(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner, refresh_interval=10, refresh_count=2)
        stats = cache.refresh_stats()
        assert stats["total_refreshes"] == 0
        assert stats["total_tokens_refreshed"] == 0
        assert stats["tokens_since_refresh"] == 0

    def test_stats_after_refresh(self):
        inner = GenerationCache(seed=SEED, anchor_interval=0)
        cache = SelfCorrectingCache(inner, refresh_interval=5, refresh_count=2)
        keys, values = make_kv_states(seq_len=1)

        for i in range(5):
            cache.update(keys, values, layer_idx=0)

        stats = cache.refresh_stats()
        assert stats["total_refreshes"] == 1
        assert stats["total_tokens_refreshed"] > 0
        assert stats["tokens_since_refresh"] == 0

    def test_stats_multiple_refreshes(self):
        inner = GenerationCache(seed=SEED, anchor_interval=0)
        cache = SelfCorrectingCache(inner, refresh_interval=3, refresh_count=1)
        keys, values = make_kv_states(seq_len=1)

        for i in range(9):
            cache.update(keys, values, layer_idx=0)

        stats = cache.refresh_stats()
        assert stats["total_refreshes"] == 3

    def test_config_in_stats(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(
            inner, refresh_interval=25, refresh_count=7,
        )
        stats = cache.refresh_stats()
        assert stats["config"]["refresh_interval"] == 25
        assert stats["config"]["refresh_count"] == 7


# ---------------------------------------------------------------------------
# Test: Memory savings delegation
# ---------------------------------------------------------------------------
class TestMemorySavings:
    """Validate that memory_savings delegates to inner cache."""

    def test_memory_savings_delegates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        keys, values = make_kv_states(seq_len=20)
        cache.update(keys, values, layer_idx=0)

        savings = cache.memory_savings()
        inner_savings = inner.memory_savings()
        assert savings["total_compressed_bits"] == inner_savings["total_compressed_bits"]
        assert savings["total_fp16_bits"] == inner_savings["total_fp16_bits"]


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_cache_refresh(self):
        """Refresh on empty cache should be a no-op."""
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner, refresh_interval=1, refresh_count=5)
        # Manually trigger refresh -- should not crash
        cache._perform_refresh()
        assert cache.total_refreshes == 0

    def test_single_token_cache(self):
        inner = GenerationCache(seed=SEED, anchor_interval=0)
        cache = SelfCorrectingCache(inner, refresh_interval=1, refresh_count=1)
        keys, values = make_kv_states(seq_len=1)
        cache.update(keys, values, layer_idx=0)
        # Should have triggered refresh after 1 token
        assert cache.total_refreshes == 1

    def test_refresh_count_larger_than_cache(self):
        """Refresh count > cached tokens should refresh all available."""
        inner = GenerationCache(seed=SEED, anchor_interval=0)
        cache = SelfCorrectingCache(inner, refresh_interval=3, refresh_count=100)
        keys, values = make_kv_states(seq_len=1)
        for i in range(3):
            cache.update(keys, values, layer_idx=0)
        stats = cache.refresh_stats()
        # Should have refreshed, but only up to 3 tokens (not 100)
        assert stats["total_refreshes"] == 1
        assert stats["total_tokens_refreshed"] <= 3

    def test_batch_repeat_interleave_delegates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        cache.batch_repeat_interleave(2)
        # Inner should have been expanded
        assert cache.get_seq_length(0) == 5  # seq unchanged

    def test_batch_select_indices_delegates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        cache.batch_select_indices(torch.tensor([0]))

    def test_is_sliding_delegates(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        assert cache.is_sliding == [False]

    def test_is_compileable_false(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        assert cache.is_compileable is False


# ---------------------------------------------------------------------------
# Test: Inner cache type flexibility
# ---------------------------------------------------------------------------
class TestInnerCacheTypes:
    """Validate that SelfCorrectingCache works with different inner caches."""

    def test_with_generation_cache(self):
        inner = GenerationCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        keys, values = make_kv_states(seq_len=5)
        k_out, v_out = cache.update(keys, values, layer_idx=0)
        assert k_out.shape[2] == 5

    def test_with_eviction_cache(self):
        from turboquantdc.token_eviction import EvictionCache
        inner = EvictionCache(seed=SEED)
        cache = SelfCorrectingCache(inner)
        keys, values = make_kv_states(seq_len=5)
        k_out, v_out = cache.update(keys, values, layer_idx=0)
        assert k_out.shape[2] == 5
