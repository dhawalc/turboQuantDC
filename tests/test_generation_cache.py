"""Tests for the production GenerationCache module.

Validates the HF Cache protocol, compression quality, FP16 window behavior,
anchor layers, memory reporting, and configurable parameters.
"""

import math

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
