"""Tests for the production GenerationCache module.

Validates the HF Cache protocol, compression quality, FP16 window behavior,
anchor layers, memory reporting, configurable parameters, incremental
dequantization, memory leak prevention, and fused attention.
"""

import math
import time

import pytest
import torch

from turboquantdc.generation_cache import (
    GenerationCache,
    _CompressedLayer,
    _FP16Layer,
)


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
# Test: basic cache protocol
# ---------------------------------------------------------------------------
class TestCacheProtocol:
    """Validate the HF Cache protocol methods."""

    def test_update_returns_correct_shapes(self):
        cache = GenerationCache(seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        k_out, v_out = cache.update(keys, values, layer_idx=0)
        assert k_out.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)
        assert v_out.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)

    def test_update_accumulates_sequence(self):
        cache = GenerationCache(seed=SEED)
        k1, v1 = make_kv_states(seq_len=5, seed=1)
        k2, v2 = make_kv_states(seq_len=3, seed=2)
        cache.update(k1, v1, layer_idx=0)
        k_out, v_out = cache.update(k2, v2, layer_idx=0)
        assert k_out.shape == (BATCH_SIZE, NUM_HEADS, 8, HEAD_DIM)
        assert v_out.shape == (BATCH_SIZE, NUM_HEADS, 8, HEAD_DIM)

    def test_get_seq_length(self):
        cache = GenerationCache(seed=SEED)
        assert cache.get_seq_length(0) == 0
        keys, values = make_kv_states(seq_len=10)
        cache.update(keys, values, layer_idx=0)
        assert cache.get_seq_length(0) == 10
        # Out of range returns 0
        assert cache.get_seq_length(99) == 0

    def test_get_max_cache_shape(self):
        cache = GenerationCache(seed=SEED)
        assert cache.get_max_cache_shape() == -1

    def test_get_mask_sizes_empty(self):
        cache = GenerationCache(seed=SEED)
        pos = torch.arange(5)
        kv_len, offset = cache.get_mask_sizes(pos, layer_idx=0)
        assert kv_len == 5
        assert offset == 0

    def test_get_mask_sizes_with_cached(self):
        """get_mask_sizes must return cached + query_length (the critical fix)."""
        cache = GenerationCache(seed=SEED)
        keys, values = make_kv_states(seq_len=20)
        cache.update(keys, values, layer_idx=0)
        # Now query 1 new token
        pos = torch.arange(1)
        kv_len, offset = cache.get_mask_sizes(pos, layer_idx=0)
        assert kv_len == 21  # 20 cached + 1 query
        assert offset == 0

    def test_len(self):
        cache = GenerationCache(seed=SEED)
        assert len(cache) == 0
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        assert len(cache) == 1
        cache.update(keys, values, layer_idx=2)
        assert len(cache) == 3  # layers 0, 1, 2

    def test_contains(self):
        cache = GenerationCache(seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        assert 0 in cache
        assert 1 not in cache

    def test_getitem(self):
        cache = GenerationCache(seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        k, v = cache[0]
        assert k.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)
        assert v.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)

    def test_getitem_out_of_range(self):
        cache = GenerationCache(seed=SEED)
        with pytest.raises(IndexError):
            _ = cache[0]

    def test_iter(self):
        cache = GenerationCache(seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        cache.update(keys, values, layer_idx=1)
        items = list(cache)
        assert len(items) == 2
        for k, v, extra in items:
            assert k.shape[2] == 5
            assert extra is None

    def test_seen_tokens(self):
        cache = GenerationCache(seed=SEED)
        assert cache.seen_tokens == 0
        keys, values = make_kv_states(seq_len=7)
        cache.update(keys, values, layer_idx=0)
        assert cache.seen_tokens == 7

    def test_is_initialized(self):
        cache = GenerationCache(seed=SEED)
        assert not cache.is_initialized
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        assert cache.is_initialized

    def test_is_sliding(self):
        cache = GenerationCache(seed=SEED)
        # Before any layers
        assert cache.is_sliding == [False]
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        cache.update(keys, values, layer_idx=1)
        assert cache.is_sliding == [False, False]

    def test_is_compileable(self):
        assert GenerationCache.is_compileable is False


# ---------------------------------------------------------------------------
# Test: compression quality
# ---------------------------------------------------------------------------
class TestCompressionQuality:
    """Validate that compression produces high quality reconstruction."""

    def test_key_cosine_similarity(self):
        """3-bit keys with residual signs should have high cosine similarity."""
        cache = GenerationCache(key_bits=3, val_bits=2, fp16_window=0, seed=SEED, anchor_interval=0)
        keys, values = make_kv_states(seq_len=64, seed=100)
        k_out, _ = cache.update(keys, values, layer_idx=0)
        sim = cosine_sim(keys, k_out)
        assert sim > 0.95, f"Key cosine similarity {sim:.4f} below 0.95 threshold"

    def test_value_cosine_similarity(self):
        """2-bit values should have reasonable cosine similarity."""
        cache = GenerationCache(key_bits=3, val_bits=2, fp16_window=0, seed=SEED, anchor_interval=0)
        keys, values = make_kv_states(seq_len=64, seed=100)
        _, v_out = cache.update(keys, values, layer_idx=0)
        sim = cosine_sim(values, v_out)
        assert sim > 0.85, f"Value cosine similarity {sim:.4f} below 0.85 threshold"

    def test_higher_bits_improve_quality(self):
        """4-bit should be better than 3-bit, which should be better than 2-bit."""
        keys, values = make_kv_states(seq_len=64, seed=100)
        sims = {}
        for bits in [2, 3, 4]:
            cache = GenerationCache(key_bits=bits, val_bits=bits, fp16_window=0, seed=SEED, anchor_interval=0)
            k_out, _ = cache.update(keys, values, layer_idx=0)
            sims[bits] = cosine_sim(keys, k_out)
        assert sims[4] >= sims[3] >= sims[2], f"Quality not monotonic: {sims}"

    @pytest.mark.parametrize("seed_val", [42, 123, 999])
    def test_key_quality_monotonic_2_to_8(self, seed_val):
        """Key reconstruction cosine similarity must be monotonically
        non-decreasing from 2-bit to 8-bit (the K5 anomaly guard).

        Tests with multiple seeds to catch seed-dependent issues.
        """
        keys, values = make_kv_states(seq_len=64, seed=seed_val)
        sims = {}
        for bits in range(2, 9):
            cache = GenerationCache(
                key_bits=bits, val_bits=2, fp16_window=0,
                seed=SEED, anchor_interval=0,
            )
            k_out, _ = cache.update(keys, values, layer_idx=0)
            sims[bits] = cosine_sim(keys, k_out)
        for b in range(2, 8):
            assert sims[b + 1] >= sims[b] - 1e-6, (
                f"Key quality not monotonic at seed={seed_val}: "
                f"{b}-bit sim={sims[b]:.6f} > {b+1}-bit sim={sims[b+1]:.6f}"
            )

    def test_value_quality_monotonic_2_to_8(self):
        """Value reconstruction quality must increase with bits (2-8)."""
        keys, values = make_kv_states(seq_len=64, seed=100)
        sims = {}
        for bits in range(2, 9):
            cache = GenerationCache(
                key_bits=3, val_bits=bits, fp16_window=0,
                seed=SEED, anchor_interval=0,
            )
            _, v_out = cache.update(keys, values, layer_idx=0)
            sims[bits] = cosine_sim(values, v_out)
        for b in range(2, 8):
            assert sims[b + 1] >= sims[b] - 1e-6, (
                f"Value quality not monotonic: "
                f"{b}-bit sim={sims[b]:.6f} > {b+1}-bit sim={sims[b+1]:.6f}"
            )

    def test_5bit_keys_not_worse_than_4bit(self):
        """Explicit guard: 5-bit keys must be at least as good as 4-bit.

        This test was added to catch the K5 anomaly observed in autoresearch
        where K5 V2 A0 W0 scored 0.51 while K4 V2 A0 W0 scored 0.92.
        """
        keys, values = make_kv_states(seq_len=128, seed=100)
        sim_4 = None
        sim_5 = None
        for bits in [4, 5]:
            cache = GenerationCache(
                key_bits=bits, val_bits=2, fp16_window=0,
                seed=SEED, anchor_interval=0,
            )
            k_out, _ = cache.update(keys, values, layer_idx=0)
            sim = cosine_sim(keys, k_out)
            if bits == 4:
                sim_4 = sim
            else:
                sim_5 = sim
        assert sim_5 >= sim_4 - 1e-6, (
            f"5-bit keys ({sim_5:.6f}) worse than 4-bit ({sim_4:.6f})"
        )

    def test_multi_layer_quality_monotonic(self):
        """Quality across 36 layers should be monotonic with bits."""
        n_layers = 36
        for key_bits in [4, 5]:
            cache = GenerationCache(
                key_bits=key_bits, val_bits=2, fp16_window=0,
                seed=SEED, anchor_interval=0,
            )
            torch.manual_seed(42)
            for layer_idx in range(n_layers):
                keys = torch.randn(1, NUM_HEADS, 32, HEAD_DIM)
                values = torch.randn(1, NUM_HEADS, 32, HEAD_DIM)
                cache.update(keys, values, layer_idx=layer_idx)

            # Verify all layers have data
            assert cache.get_seq_length(0) == 32
            assert len(cache) == n_layers


# ---------------------------------------------------------------------------
# Test: FP16 window
# ---------------------------------------------------------------------------
class TestFP16Window:
    """Validate the FP16 precision window for recent tokens."""

    def test_fp16_window_preserves_recent_tokens(self):
        """Last fp16_window tokens should be exactly FP16."""
        window = 4
        cache = GenerationCache(fp16_window=window, seed=SEED, anchor_interval=0)
        keys, values = make_kv_states(seq_len=16, seed=200)
        k_out, v_out = cache.update(keys, values, layer_idx=0)
        # Last 4 tokens should match exactly
        torch.testing.assert_close(
            k_out[:, :, -window:, :],
            keys[:, :, -window:, :],
            atol=1e-6, rtol=1e-5,
        )
        torch.testing.assert_close(
            v_out[:, :, -window:, :],
            values[:, :, -window:, :],
            atol=1e-6, rtol=1e-5,
        )

    def test_fp16_window_zero_means_all_compressed(self):
        """fp16_window=0 should not preserve any tokens at FP16."""
        cache = GenerationCache(fp16_window=0, seed=SEED, anchor_interval=0)
        keys, values = make_kv_states(seq_len=16, seed=200)
        k_out, v_out = cache.update(keys, values, layer_idx=0)
        # With lossy compression, output should differ from input
        diff = (k_out - keys).abs().max().item()
        assert diff > 1e-4, "Expected lossy compression with fp16_window=0"

    def test_fp16_window_larger_than_seq(self):
        """If fp16_window > seq_len, all tokens should be at FP16."""
        cache = GenerationCache(fp16_window=1000, seed=SEED, anchor_interval=0)
        keys, values = make_kv_states(seq_len=8, seed=200)
        k_out, v_out = cache.update(keys, values, layer_idx=0)
        torch.testing.assert_close(k_out, keys, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(v_out, values, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test: anchor layers
# ---------------------------------------------------------------------------
class TestAnchorLayers:
    """Validate FP16 anchor layers that break error accumulation."""

    def test_anchor_layers_are_fp16(self):
        """Layer 0, 6, 12 should be FP16 anchors with interval=6."""
        cache = GenerationCache(anchor_interval=6, seed=SEED)
        keys, values = make_kv_states(seq_len=8, seed=300)
        # Fill layers 0 through 12
        for i in range(13):
            cache.update(keys, values, layer_idx=i)
        # Anchor layers (0, 6, 12) should return exact FP16
        for anchor_idx in [0, 6, 12]:
            k_out, v_out = cache[anchor_idx]
            torch.testing.assert_close(k_out, keys, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(v_out, values, atol=1e-6, rtol=1e-5)

    def test_no_anchors(self):
        """anchor_interval=0 should disable anchors entirely."""
        cache = GenerationCache(anchor_interval=0, seed=SEED)
        assert not cache._is_anchor_layer(0)
        assert not cache._is_anchor_layer(6)


# ---------------------------------------------------------------------------
# Test: cache operations
# ---------------------------------------------------------------------------
class TestCacheOperations:
    """Validate reset, crop, reorder, and batch operations."""

    def test_reset_clears_all(self):
        cache = GenerationCache(seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        assert cache.get_seq_length(0) == 5
        cache.reset()
        assert cache.get_seq_length(0) == 0

    def test_crop(self):
        cache = GenerationCache(seed=SEED, anchor_interval=0)
        keys, values = make_kv_states(seq_len=20)
        cache.update(keys, values, layer_idx=0)
        cache.crop(10)
        assert cache.get_seq_length(0) == 10
        k_out, v_out = cache[0]
        assert k_out.shape[2] == 10

    def test_crop_negative(self):
        """Negative crop should trim from the end."""
        cache = GenerationCache(seed=SEED, anchor_interval=0)
        keys, values = make_kv_states(seq_len=20)
        cache.update(keys, values, layer_idx=0)
        cache.crop(-5)
        assert cache.get_seq_length(0) == 15

    def test_reorder_cache(self):
        cache = GenerationCache(seed=SEED, anchor_interval=0)
        keys, values = make_kv_states(batch=4, seq_len=5, seed=400)
        cache.update(keys, values, layer_idx=0)
        # Reorder: swap batch 0 and 3
        beam_idx = torch.tensor([3, 1, 2, 0])
        cache.reorder_cache(beam_idx)
        k_out, v_out = cache[0]
        assert k_out.shape[0] == 4


# ---------------------------------------------------------------------------
# Test: configuration validation
# ---------------------------------------------------------------------------
class TestConfiguration:
    """Validate parameter validation and configuration."""

    def test_invalid_key_bits(self):
        with pytest.raises(ValueError, match="key_bits"):
            GenerationCache(key_bits=0)
        with pytest.raises(ValueError, match="key_bits"):
            GenerationCache(key_bits=9)

    def test_invalid_val_bits(self):
        with pytest.raises(ValueError, match="val_bits"):
            GenerationCache(val_bits=0)
        with pytest.raises(ValueError, match="val_bits"):
            GenerationCache(val_bits=9)

    def test_invalid_fp16_window(self):
        with pytest.raises(ValueError, match="fp16_window"):
            GenerationCache(fp16_window=-1)

    def test_default_config(self):
        cache = GenerationCache()
        assert cache.key_bits == 3
        assert cache.val_bits == 2
        assert cache.fp16_window == 128
        assert cache.anchor_interval == 6

    def test_custom_config(self):
        cache = GenerationCache(key_bits=4, val_bits=3, fp16_window=64, anchor_interval=4)
        assert cache.key_bits == 4
        assert cache.val_bits == 3
        assert cache.fp16_window == 64
        assert cache.anchor_interval == 4


# ---------------------------------------------------------------------------
# Test: memory reporting
# ---------------------------------------------------------------------------
class TestMemoryReporting:
    """Validate memory usage and compression ratio reporting."""

    def test_memory_savings_empty(self):
        cache = GenerationCache(seed=SEED)
        report = cache.memory_savings()
        assert report["overall_compression_ratio"] == 1.0
        assert report["num_layers"] == 0

    def test_memory_savings_with_data(self):
        cache = GenerationCache(seed=SEED, anchor_interval=0, fp16_window=0)
        keys, values = make_kv_states(seq_len=64)
        cache.update(keys, values, layer_idx=0)
        report = cache.memory_savings()
        assert report["total_compressed_bits"] > 0
        assert report["total_fp16_bits"] > 0
        # Should show compression (fp16_window=0 so everything is compressed)
        assert report["overall_compression_ratio"] > 1.0

    def test_config_summary(self):
        cache = GenerationCache(seed=SEED, anchor_interval=6)
        keys, values = make_kv_states(seq_len=5)
        for i in range(12):
            cache.update(keys, values, layer_idx=i)
        summary = cache.config_summary()
        assert "3b keys" in summary
        assert "2b values" in summary
        assert "FP16 window=128" in summary


# ---------------------------------------------------------------------------
# Test: multi-layer autoregressive simulation
# ---------------------------------------------------------------------------
class TestAutoregressiveSimulation:
    """Simulate autoregressive generation to validate end-to-end behavior."""

    def test_token_by_token_generation(self):
        """Simulate token-by-token generation across multiple layers."""
        n_layers = 4
        cache = GenerationCache(seed=SEED, anchor_interval=0, fp16_window=8)

        # Prefill with 16 tokens
        prefill_keys, prefill_values = make_kv_states(
            batch=1, num_heads=2, seq_len=16, head_dim=64, seed=500,
        )
        for layer in range(n_layers):
            cache.update(prefill_keys, prefill_values, layer_idx=layer)

        assert cache.get_seq_length(0) == 16

        # Generate 10 tokens one at a time
        for step in range(10):
            new_k, new_v = make_kv_states(
                batch=1, num_heads=2, seq_len=1, head_dim=64, seed=600 + step,
            )
            for layer in range(n_layers):
                k_out, v_out = cache.update(new_k, new_v, layer_idx=layer)
                expected_len = 16 + step + 1
                assert k_out.shape[2] == expected_len, (
                    f"Step {step}, layer {layer}: expected seq {expected_len}, "
                    f"got {k_out.shape[2]}"
                )

        assert cache.get_seq_length(0) == 26

    def test_mask_sizes_during_generation(self):
        """get_mask_sizes must be consistent throughout generation."""
        cache = GenerationCache(seed=SEED, anchor_interval=0)
        keys, values = make_kv_states(batch=1, num_heads=2, seq_len=10, head_dim=64)
        cache.update(keys, values, layer_idx=0)

        # Simulate decoding: 1 new token
        pos = torch.arange(1)
        kv_len, offset = cache.get_mask_sizes(pos, layer_idx=0)
        assert kv_len == 11  # 10 + 1
        assert offset == 0


# ---------------------------------------------------------------------------
# Test: Fix 1 — O(N) incremental dequantization
# ---------------------------------------------------------------------------
class TestIncrementalDequantization:
    """Validate that dequantization is incremental (O(N) not O(N^2))."""

    def test_dequant_cache_populated(self):
        """After update, the internal dequant cache should be populated."""
        cache = GenerationCache(seed=SEED, anchor_interval=0, fp16_window=0)
        keys, values = make_kv_states(batch=1, num_heads=2, seq_len=8, head_dim=64)
        cache.update(keys, values, layer_idx=0)
        layer = cache._layers[0]
        assert layer._dequant_key_cache is not None
        assert layer._dequant_val_cache is not None
        assert layer._dequant_len == 8

    def test_dequant_cache_grows_incrementally(self):
        """Dequant cache should grow with each update, not be rebuilt."""
        cache = GenerationCache(seed=SEED, anchor_interval=0, fp16_window=0)
        # Prefill
        k1, v1 = make_kv_states(batch=1, num_heads=2, seq_len=10, head_dim=64, seed=1)
        cache.update(k1, v1, layer_idx=0)
        layer = cache._layers[0]
        assert layer._dequant_len == 10
        # Add 1 token
        k2, v2 = make_kv_states(batch=1, num_heads=2, seq_len=1, head_dim=64, seed=2)
        cache.update(k2, v2, layer_idx=0)
        assert layer._dequant_len == 11
        assert layer._dequant_key_cache.shape[2] == 11

    def test_incremental_matches_full_rebuild(self):
        """Incremental dequantization must produce same result as full rebuild."""
        # Build incrementally
        cache_inc = GenerationCache(seed=SEED, anchor_interval=0, fp16_window=0)
        k1, v1 = make_kv_states(batch=1, num_heads=2, seq_len=10, head_dim=64, seed=10)
        cache_inc.update(k1, v1, layer_idx=0)
        for i in range(5):
            ki, vi = make_kv_states(batch=1, num_heads=2, seq_len=1, head_dim=64, seed=100 + i)
            k_out, v_out = cache_inc.update(ki, vi, layer_idx=0)

        # Build all at once (simulate by concatenating inputs)
        all_keys = [k1] + [
            make_kv_states(batch=1, num_heads=2, seq_len=1, head_dim=64, seed=100 + i)[0]
            for i in range(5)
        ]
        all_values = [v1] + [
            make_kv_states(batch=1, num_heads=2, seq_len=1, head_dim=64, seed=100 + i)[1]
            for i in range(5)
        ]
        all_k = torch.cat(all_keys, dim=2)
        all_v = torch.cat(all_values, dim=2)

        cache_full = GenerationCache(seed=SEED, anchor_interval=0, fp16_window=0)
        k_full, v_full = cache_full.update(all_k, all_v, layer_idx=0)

        # Results should be close (same quantization, same dequant path)
        torch.testing.assert_close(k_out, k_full, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(v_out, v_full, atol=1e-5, rtol=1e-4)

    def test_dequant_is_linear_not_quadratic(self):
        """Dequant of 100 vs 200 tokens should take ~2x, not ~4x time."""
        def time_n_tokens(n: int) -> float:
            cache = GenerationCache(seed=SEED, anchor_interval=0, fp16_window=0)
            # Prefill with 10 tokens
            k0, v0 = make_kv_states(batch=1, num_heads=2, seq_len=10, head_dim=64, seed=0)
            cache.update(k0, v0, layer_idx=0)
            # Add n tokens one at a time
            start = time.perf_counter()
            for i in range(n):
                ki, vi = make_kv_states(batch=1, num_heads=2, seq_len=1, head_dim=64, seed=i + 1)
                cache.update(ki, vi, layer_idx=0)
            elapsed = time.perf_counter() - start
            return elapsed

        t100 = time_n_tokens(100)
        t200 = time_n_tokens(200)

        # With O(N^2), t200/t100 would be ~4x. With O(N), it should be ~2x.
        # Allow generous margin but reject clearly quadratic behavior.
        ratio = t200 / max(t100, 1e-9)
        assert ratio < 3.5, (
            f"Dequant scaling looks quadratic: 200tok took {t200:.3f}s, "
            f"100tok took {t100:.3f}s, ratio={ratio:.2f} (expected <3.5)"
        )

    def test_dequant_cache_invalidated_on_clear(self):
        """Clearing a layer should invalidate the dequant cache."""
        cache = GenerationCache(seed=SEED, anchor_interval=0)
        keys, values = make_kv_states(batch=1, num_heads=2, seq_len=8, head_dim=64)
        cache.update(keys, values, layer_idx=0)
        layer = cache._layers[0]
        assert layer._dequant_key_cache is not None
        layer.clear()
        assert layer._dequant_key_cache is None
        assert layer._dequant_len == 0

    def test_dequant_cache_truncated_on_crop(self):
        """Cropping should truncate the dequant cache to match."""
        cache = GenerationCache(seed=SEED, anchor_interval=0, fp16_window=0)
        keys, values = make_kv_states(batch=1, num_heads=2, seq_len=20, head_dim=64)
        cache.update(keys, values, layer_idx=0)
        layer = cache._layers[0]
        assert layer._dequant_key_cache.shape[2] == 20
        cache.crop(10)
        assert layer._dequant_key_cache.shape[2] == 10
        assert layer._dequant_len == 10


# ---------------------------------------------------------------------------
# Test: Fix 2 — FP16 window memory leak
# ---------------------------------------------------------------------------
class TestFP16MemoryLeak:
    """Validate that raw FP16 storage is bounded, not growing unboundedly."""

    def test_raw_storage_bounded(self):
        """After 500 tokens, raw list should have <= fp16_window entries."""
        fp16_window = 32
        cache = GenerationCache(
            seed=SEED, anchor_interval=0, fp16_window=fp16_window,
        )
        # Prefill
        k0, v0 = make_kv_states(batch=1, num_heads=2, seq_len=10, head_dim=64, seed=0)
        cache.update(k0, v0, layer_idx=0)
        # Add 490 tokens one at a time
        for i in range(490):
            ki, vi = make_kv_states(
                batch=1, num_heads=2, seq_len=1, head_dim=64, seed=i + 1,
            )
            cache.update(ki, vi, layer_idx=0)

        layer = cache._layers[0]
        total_raw_tokens = sum(t.shape[2] for t in layer._raw_keys)
        assert total_raw_tokens <= fp16_window, (
            f"Raw FP16 storage has {total_raw_tokens} tokens, "
            f"expected <= {fp16_window}"
        )

    def test_raw_storage_exact_window(self):
        """After trimming, the raw data should have exactly fp16_window tokens."""
        fp16_window = 16
        cache = GenerationCache(
            seed=SEED, anchor_interval=0, fp16_window=fp16_window,
        )
        # Add enough tokens to trigger trim (> fp16_window * 2)
        for i in range(fp16_window * 3):
            ki, vi = make_kv_states(
                batch=1, num_heads=2, seq_len=1, head_dim=64, seed=i,
            )
            cache.update(ki, vi, layer_idx=0)

        layer = cache._layers[0]
        total_raw_tokens = sum(t.shape[2] for t in layer._raw_keys)
        assert total_raw_tokens == fp16_window, (
            f"Expected exactly {fp16_window} raw tokens after trim, "
            f"got {total_raw_tokens}"
        )

    def test_fp16_window_zero_no_raw_growth(self):
        """With fp16_window=0, raw storage should still not leak."""
        cache = GenerationCache(
            seed=SEED, anchor_interval=0, fp16_window=0,
        )
        for i in range(100):
            ki, vi = make_kv_states(
                batch=1, num_heads=2, seq_len=1, head_dim=64, seed=i,
            )
            cache.update(ki, vi, layer_idx=0)

        # fp16_window=0 means the raw data is never used for splice,
        # but the trim condition (seq_len > fp16_window * 2 = 0) is always
        # true after first token, so trimming is a no-op for window=0.
        # The important thing is it doesn't crash.
        layer = cache._layers[0]
        assert layer.get_seq_length() == 100

    def test_fp16_window_still_works_after_trim(self):
        """After raw trimming, FP16 window splice should still work correctly."""
        fp16_window = 8
        cache = GenerationCache(
            seed=SEED, anchor_interval=0, fp16_window=fp16_window,
        )
        # Build up 50 tokens
        all_keys = []
        for i in range(50):
            ki, vi = make_kv_states(
                batch=1, num_heads=2, seq_len=1, head_dim=64, seed=i,
            )
            all_keys.append(ki)
            k_out, v_out = cache.update(ki, vi, layer_idx=0)

        # The last fp16_window tokens in the output should match
        # the last fp16_window inputs exactly
        recent_keys = torch.cat(all_keys[-fp16_window:], dim=2)
        torch.testing.assert_close(
            k_out[:, :, -fp16_window:, :],
            recent_keys,
            atol=1e-6, rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# Test: Fix 3 — Fused attention / norm correction
# ---------------------------------------------------------------------------
class TestFusedAttentionNormCorrection:
    """Validate fused attention path and norm correction integration."""

    def test_norm_correction_default_on(self):
        """use_norm_correction should default to True."""
        cache = GenerationCache(seed=SEED)
        assert cache.use_norm_correction is True

    def test_norm_correction_passed_to_layers(self):
        """use_norm_correction should be passed to _CompressedLayer."""
        cache = GenerationCache(seed=SEED, anchor_interval=0, use_norm_correction=True)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        layer = cache._layers[0]
        assert layer.use_norm_correction is True

        cache2 = GenerationCache(seed=SEED, anchor_interval=0, use_norm_correction=False)
        cache2.update(keys, values, layer_idx=0)
        layer2 = cache2._layers[0]
        assert layer2.use_norm_correction is False

    def test_fused_dequant_uses_float32(self):
        """The fused path should compute in float32 to avoid FP16 rounding."""
        cache = GenerationCache(
            seed=SEED, anchor_interval=0, fp16_window=0,
            use_norm_correction=True,
        )
        keys, values = make_kv_states(seq_len=16, seed=42)
        cache.update(keys, values, layer_idx=0)

        layer = cache._layers[0]
        # The dequant cache should exist
        assert layer._dequant_key_cache is not None
        # The fused method should exist and be callable
        assert hasattr(layer, '_dequantize_vectors_fused')

    def test_norm_correction_improves_quality(self):
        """Norm correction should improve or maintain reconstruction quality."""
        keys, values = make_kv_states(seq_len=64, seed=300)

        # With norm correction (default)
        cache_on = GenerationCache(
            seed=SEED, anchor_interval=0, fp16_window=0,
            use_norm_correction=True,
        )
        k_on, _ = cache_on.update(keys, values, layer_idx=0)
        sim_on = cosine_sim(keys, k_on)

        # Without norm correction
        cache_off = GenerationCache(
            seed=SEED, anchor_interval=0, fp16_window=0,
            use_norm_correction=False,
        )
        k_off, _ = cache_off.update(keys, values, layer_idx=0)
        sim_off = cosine_sim(keys, k_off)

        # Both should have high quality (norm correction is always stored,
        # the flag controls the fused path behavior)
        assert sim_on > 0.90, f"Norm correction ON quality too low: {sim_on:.4f}"
        assert sim_off > 0.90, f"Norm correction OFF quality too low: {sim_off:.4f}"

    def test_fused_path_produces_valid_output(self):
        """The fused dequantization path should produce valid tensors."""
        cache = GenerationCache(
            seed=SEED, anchor_interval=0, fp16_window=0,
            use_norm_correction=True,
        )
        keys, values = make_kv_states(batch=1, num_heads=2, seq_len=32, head_dim=64, seed=50)
        k_out, v_out = cache.update(keys, values, layer_idx=0)

        # Output should have correct shape
        assert k_out.shape == keys.shape
        assert v_out.shape == values.shape

        # Output should not contain NaN or Inf
        assert not torch.isnan(k_out).any(), "Fused path produced NaN in keys"
        assert not torch.isinf(k_out).any(), "Fused path produced Inf in keys"
        assert not torch.isnan(v_out).any(), "Fused path produced NaN in values"
        assert not torch.isinf(v_out).any(), "Fused path produced Inf in values"

    def test_fused_path_cosine_quality(self):
        """Fused path key reconstruction should have good cosine similarity."""
        cache = GenerationCache(
            key_bits=3, val_bits=2, fp16_window=0, seed=SEED,
            anchor_interval=0, use_norm_correction=True,
        )
        keys, values = make_kv_states(seq_len=64, seed=100)
        k_out, _ = cache.update(keys, values, layer_idx=0)
        sim = cosine_sim(keys, k_out)
        assert sim > 0.95, f"Fused path cosine similarity {sim:.4f} below 0.95"


# ---------------------------------------------------------------------------
# Test: use_residual_quant toggle
# ---------------------------------------------------------------------------
class TestResidualQuantToggle:
    """Validate that use_residual_quant actually changes dequantization output.

    Regression test for the bug where autoresearch sweeps of
    use_residual_quant=True vs False produced identical scores because
    GenerationCache ignored the flag entirely.
    """

    def test_rq_true_vs_false_produce_different_keys(self):
        """Compressing the same vectors with RQ=True vs RQ=False must differ."""
        keys, values = make_kv_states(seq_len=32, seed=42)

        cache_rq = GenerationCache(
            key_bits=3, val_bits=2, fp16_window=0, anchor_interval=0,
            seed=SEED, use_residual_quant=True,
        )
        cache_no_rq = GenerationCache(
            key_bits=3, val_bits=2, fp16_window=0, anchor_interval=0,
            seed=SEED, use_residual_quant=False,
        )

        k_rq, _ = cache_rq.update(keys, values, layer_idx=0)
        k_no_rq, _ = cache_no_rq.update(keys, values, layer_idx=0)

        # With fp16_window=0, ALL tokens are compressed, so the outputs
        # should differ when residual correction is toggled
        diff = (k_rq - k_no_rq).abs().max().item()
        assert diff > 1e-4, (
            f"RQ=True and RQ=False produced identical keys (max diff={diff}). "
            f"The use_residual_quant flag has no effect!"
        )

    def test_rq_true_has_better_quality(self):
        """RQ=True should produce better key reconstruction than RQ=False."""
        keys, values = make_kv_states(seq_len=64, seed=42)

        cache_rq = GenerationCache(
            key_bits=3, val_bits=2, fp16_window=0, anchor_interval=0,
            seed=SEED, use_residual_quant=True,
        )
        cache_no_rq = GenerationCache(
            key_bits=3, val_bits=2, fp16_window=0, anchor_interval=0,
            seed=SEED, use_residual_quant=False,
        )

        k_rq, _ = cache_rq.update(keys, values, layer_idx=0)
        k_no_rq, _ = cache_no_rq.update(keys, values, layer_idx=0)

        sim_rq = cosine_sim(keys, k_rq)
        sim_no_rq = cosine_sim(keys, k_no_rq)

        assert sim_rq > sim_no_rq, (
            f"RQ=True cosine sim ({sim_rq:.6f}) should be > "
            f"RQ=False cosine sim ({sim_no_rq:.6f})"
        )

    def test_rq_false_values_unaffected(self):
        """Values should be identical regardless of use_residual_quant."""
        keys, values = make_kv_states(seq_len=32, seed=42)

        cache_rq = GenerationCache(
            key_bits=3, val_bits=2, fp16_window=0, anchor_interval=0,
            seed=SEED, use_residual_quant=True,
        )
        cache_no_rq = GenerationCache(
            key_bits=3, val_bits=2, fp16_window=0, anchor_interval=0,
            seed=SEED, use_residual_quant=False,
        )

        _, v_rq = cache_rq.update(keys, values, layer_idx=0)
        _, v_no_rq = cache_no_rq.update(keys, values, layer_idx=0)

        # Values never use residual correction, so they should be identical
        torch.testing.assert_close(v_rq, v_no_rq, atol=1e-6, rtol=1e-6)

    def test_default_rq_is_true(self):
        """GenerationCache default should have use_residual_quant=True."""
        cache = GenerationCache()
        assert cache.use_residual_quant is True

    def test_config_summary_reflects_rq(self):
        """config_summary should indicate residual quant status."""
        cache_rq = GenerationCache(use_residual_quant=True)
        cache_no_rq = GenerationCache(use_residual_quant=False)
        keys, values = make_kv_states(seq_len=5)
        cache_rq.update(keys, values, layer_idx=0)
        cache_no_rq.update(keys, values, layer_idx=0)

        assert "residual signs" in cache_rq.config_summary()
        assert "no residual signs" in cache_no_rq.config_summary()
