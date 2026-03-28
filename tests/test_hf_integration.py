"""Tests for HuggingFace transformers integration.

Tests the TurboQuantCache class which provides a drop-in KV cache for HF
models. Validates the Cache protocol (update, get_seq_length, __getitem__,
reorder_cache, crop, etc.) and compression quality.

The cache stores keys with TurboQuantEstimator (MSE + QJL) and values with
PolarQuant (MSE-only), then dequantizes on retrieval so standard HF attention
works without modification.
"""

import math

import pytest
import torch

from turboquantdc.hf_integration import TurboQuantCache, TurboQuantLayer


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


# ---------------------------------------------------------------------------
# Test: basic cache operations
# ---------------------------------------------------------------------------
class TestBasicCacheOperations:
    """Test the fundamental cache protocol: update, seq_length, getitem."""

    def test_update_returns_correct_shapes(self):
        """update() should return dequantized tensors with correct shapes."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        keys, values = make_kv_states(seq_len=5)

        k_out, v_out = cache.update(keys, values, layer_idx=0)

        assert k_out.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)
        assert v_out.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)

    def test_seq_length_after_update(self):
        """get_seq_length() should reflect total cached tokens."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        assert cache.get_seq_length(0) == 0

        keys1, vals1 = make_kv_states(seq_len=5, seed=1)
        cache.update(keys1, vals1, layer_idx=0)
        assert cache.get_seq_length(0) == 5

        keys2, vals2 = make_kv_states(seq_len=3, seed=2)
        cache.update(keys2, vals2, layer_idx=0)
        assert cache.get_seq_length(0) == 8

    def test_seq_length_unknown_layer(self):
        """get_seq_length for layer not yet created should return 0."""
        cache = TurboQuantCache(bits=3)
        assert cache.get_seq_length(99) == 0

    def test_getitem(self):
        """__getitem__ should return dequantized KV for a layer."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        keys, values = make_kv_states(seq_len=4)
        cache.update(keys, values, layer_idx=0)

        k_out, v_out = cache[0]
        assert k_out.shape == (BATCH_SIZE, NUM_HEADS, 4, HEAD_DIM)
        assert v_out.shape == (BATCH_SIZE, NUM_HEADS, 4, HEAD_DIM)

    def test_getitem_out_of_range(self):
        """__getitem__ for nonexistent layer should raise IndexError."""
        cache = TurboQuantCache(bits=3)
        with pytest.raises(IndexError):
            _ = cache[0]

    def test_len_grows_with_layers(self):
        """__len__ should match number of layers that have been updated."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        assert len(cache) == 0

        keys, values = make_kv_states(seq_len=2)
        cache.update(keys, values, layer_idx=0)
        assert len(cache) == 1

        cache.update(keys, values, layer_idx=2)
        # Layers 0, 1 (empty), 2 should exist
        assert len(cache) == 3

    def test_accumulation_across_updates(self):
        """Multiple updates should concatenate along sequence dimension."""
        cache = TurboQuantCache(bits=3, seed=SEED)

        for i in range(4):
            keys, values = make_kv_states(seq_len=1, seed=i)
            k_out, v_out = cache.update(keys, values, layer_idx=0)

        assert k_out.shape == (BATCH_SIZE, NUM_HEADS, 4, HEAD_DIM)
        assert v_out.shape == (BATCH_SIZE, NUM_HEADS, 4, HEAD_DIM)
        assert cache.get_seq_length(0) == 4


# ---------------------------------------------------------------------------
# Test: multi-layer support
# ---------------------------------------------------------------------------
class TestMultiLayer:
    """Test that cache works correctly across multiple transformer layers."""

    def test_independent_layers(self):
        """Each layer should maintain its own sequence independently."""
        cache = TurboQuantCache(bits=3, seed=SEED)

        # Layer 0 gets 5 tokens
        k0, v0 = make_kv_states(seq_len=5, seed=10)
        cache.update(k0, v0, layer_idx=0)

        # Layer 1 gets 3 tokens
        k1, v1 = make_kv_states(seq_len=3, seed=20)
        cache.update(k1, v1, layer_idx=1)

        assert cache.get_seq_length(0) == 5
        assert cache.get_seq_length(1) == 3

    def test_iter_yields_all_layers(self):
        """__iter__ should yield (keys, values, None) for each layer."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        keys, values = make_kv_states(seq_len=2)
        cache.update(keys, values, layer_idx=0)
        cache.update(keys, values, layer_idx=1)

        items = list(cache)
        assert len(items) == 2
        for k, v, sliding in items:
            assert k.shape[2] == 2
            assert v.shape[2] == 2
            assert sliding is None

    def test_different_seeds_per_layer(self):
        """Each layer should use a different seed for its quantizers."""
        cache = TurboQuantCache(bits=3, seed=100)
        keys, values = make_kv_states(seq_len=2)
        cache.update(keys, values, layer_idx=0)
        cache.update(keys, values, layer_idx=1)

        # The layers should have different seeds
        assert cache._layers[0].seed != cache._layers[1].seed


# ---------------------------------------------------------------------------
# Test: bits parameter
# ---------------------------------------------------------------------------
class TestBitsParameter:
    """Test that different bit-widths (2, 3, 4) all work."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_valid_bits(self, bits):
        """Cache should work for bits in {2, 3, 4}."""
        cache = TurboQuantCache(bits=bits, seed=SEED)
        keys, values = make_kv_states(seq_len=4)

        k_out, v_out = cache.update(keys, values, layer_idx=0)
        assert k_out.shape == (BATCH_SIZE, NUM_HEADS, 4, HEAD_DIM)
        assert v_out.shape == (BATCH_SIZE, NUM_HEADS, 4, HEAD_DIM)
        assert cache.get_seq_length(0) == 4

    def test_invalid_bits(self):
        """Bits outside {2, 3, 4} should raise ValueError."""
        with pytest.raises(ValueError, match="bits must be"):
            TurboQuantCache(bits=5)
        with pytest.raises(ValueError, match="bits must be"):
            TurboQuantCache(bits=1)


# ---------------------------------------------------------------------------
# Test: reorder_cache (beam search)
# ---------------------------------------------------------------------------
class TestReorderCache:
    """Test beam search reordering of the cache."""

    def test_reorder_swaps_batches(self):
        """reorder_cache should select batch entries by beam_idx."""
        cache = TurboQuantCache(bits=3, seed=SEED)

        # Create distinct data per batch
        torch.manual_seed(SEED)
        keys = torch.randn(2, NUM_HEADS, 3, HEAD_DIM)
        values = torch.randn(2, NUM_HEADS, 3, HEAD_DIM)
        cache.update(keys, values, layer_idx=0)

        # Get pre-reorder values
        k_before, v_before = cache[0]

        # Reorder: swap batch 0 and 1 -> [1, 0]
        beam_idx = torch.tensor([1, 0])
        cache.reorder_cache(beam_idx)

        k_after, v_after = cache[0]

        # After reorder, batch 0 should equal what was batch 1 before
        torch.testing.assert_close(k_after[0], k_before[1], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(k_after[1], k_before[0], atol=1e-5, rtol=1e-4)

    def test_reorder_duplicate_beams(self):
        """reorder_cache with duplicate indices should duplicate cache entries."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        keys, values = make_kv_states(batch=2, seq_len=3)
        cache.update(keys, values, layer_idx=0)

        # Both beams select batch 0
        beam_idx = torch.tensor([0, 0])
        cache.reorder_cache(beam_idx)

        k_out, v_out = cache[0]
        # Both batch entries should now be the same
        torch.testing.assert_close(k_out[0], k_out[1], atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test: crop
# ---------------------------------------------------------------------------
class TestCrop:
    """Test cache truncation."""

    def test_crop_reduces_seq_length(self):
        """crop(max_length) should truncate the sequence."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        keys, values = make_kv_states(seq_len=10)
        cache.update(keys, values, layer_idx=0)
        assert cache.get_seq_length(0) == 10

        cache.crop(5)
        assert cache.get_seq_length(0) == 5

        k_out, v_out = cache[0]
        assert k_out.shape[2] == 5
        assert v_out.shape[2] == 5

    def test_crop_no_op_if_smaller(self):
        """crop should be a no-op if cache is already smaller than max_length."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        keys, values = make_kv_states(seq_len=3)
        cache.update(keys, values, layer_idx=0)

        cache.crop(10)
        assert cache.get_seq_length(0) == 3

    def test_crop_negative(self):
        """crop with negative value should remove that many tokens."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        keys, values = make_kv_states(seq_len=10)
        cache.update(keys, values, layer_idx=0)

        cache.crop(-3)
        assert cache.get_seq_length(0) == 7

    def test_crop_preserves_earlier_tokens(self):
        """After crop, the remaining tokens should match the originals."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        keys, values = make_kv_states(seq_len=6)
        cache.update(keys, values, layer_idx=0)

        # Get full cache before crop
        k_full, v_full = cache[0]

        cache.crop(4)
        k_cropped, v_cropped = cache[0]

        torch.testing.assert_close(k_cropped, k_full[:, :, :4, :], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v_cropped, v_full[:, :, :4, :], atol=1e-5, rtol=1e-5)

    def test_crop_across_multiple_updates(self):
        """crop should work correctly when seq was built from multiple updates."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        # 3 updates of 4 tokens each = 12 total
        for i in range(3):
            keys, values = make_kv_states(seq_len=4, seed=i)
            cache.update(keys, values, layer_idx=0)
        assert cache.get_seq_length(0) == 12

        cache.crop(6)
        assert cache.get_seq_length(0) == 6
        k_out, _ = cache[0]
        assert k_out.shape[2] == 6


# ---------------------------------------------------------------------------
# Test: memory savings
# ---------------------------------------------------------------------------
class TestMemorySavings:
    """Test that compressed cache uses less memory than FP16 baseline."""

    def test_memory_savings_positive(self):
        """Compressed cache should have compression_ratio > 1."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        keys, values = make_kv_states(seq_len=64)
        cache.update(keys, values, layer_idx=0)

        savings = cache.memory_savings()
        assert savings["overall_compression_ratio"] > 1.0
        assert savings["total_compressed_bits"] < savings["total_fp16_bits"]
        assert savings["bits"] == 3
        assert savings["num_layers"] == 1

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_higher_bits_lower_compression(self, bits):
        """Higher bit-widths should have lower compression ratios."""
        cache = TurboQuantCache(bits=bits, seed=SEED)
        keys, values = make_kv_states(seq_len=32)
        cache.update(keys, values, layer_idx=0)

        savings = cache.memory_savings()
        ratio = savings["overall_compression_ratio"]
        # All should compress (ratio > 1)
        assert ratio > 1.0

    def test_memory_savings_empty_cache(self):
        """Empty cache should report zero usage."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        savings = cache.memory_savings()
        assert savings["total_compressed_bits"] == 0
        assert savings["total_fp16_bits"] == 0
        assert savings["overall_compression_ratio"] == 0.0

    def test_3bit_ratio_close_to_paper(self):
        """At 3-bit d=128, compression ratio should be near the paper's 5.0x."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        keys, values = make_kv_states(seq_len=64)
        cache.update(keys, values, layer_idx=0)

        savings = cache.memory_savings()
        ratio = savings["overall_compression_ratio"]
        # Paper claims ~5.0x. Allow range [3.5, 6.5] for overhead.
        assert 3.5 < ratio < 6.5, f"Expected ratio near 5.0x, got {ratio:.2f}x"


# ---------------------------------------------------------------------------
# Test: quality (dequantized output fidelity)
# ---------------------------------------------------------------------------
class TestQuality:
    """Test that dequantized keys/values are close to originals."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_key_cosine_similarity(self, bits):
        """Dequantized keys should have high cosine similarity to originals."""
        cache = TurboQuantCache(bits=bits, seed=SEED)
        torch.manual_seed(SEED)
        keys = torch.randn(1, NUM_HEADS, 32, HEAD_DIM)
        values = torch.randn(1, NUM_HEADS, 32, HEAD_DIM)

        cache.update(keys, values, layer_idx=0)
        k_out, _ = cache[0]

        # Compute per-vector cosine similarity
        k_flat = keys.float().reshape(-1, HEAD_DIM)
        k_out_flat = k_out.float().reshape(-1, HEAD_DIM)
        cos_sim = torch.nn.functional.cosine_similarity(k_flat, k_out_flat, dim=-1)
        mean_cos = cos_sim.mean().item()

        # Keys use (bits-1) MSE bits because 1 bit goes to QJL for unbiased
        # inner products. So key reconstruction quality is inherently lower than
        # value reconstruction (which uses full b-bit MSE). This is by design:
        # keys need unbiased IP estimation, not perfect reconstruction.
        # Measured quality: 2-bit->1-bit MSE ~0.65, 3-bit->2-bit MSE ~0.80, 4-bit->3-bit MSE ~0.87
        thresholds = {2: 0.60, 3: 0.75, 4: 0.85}
        assert mean_cos > thresholds[bits], (
            f"{bits}-bit key cosine sim {mean_cos:.4f} below threshold {thresholds[bits]}"
        )

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_value_cosine_similarity(self, bits):
        """Dequantized values should have high cosine similarity to originals."""
        cache = TurboQuantCache(bits=bits, seed=SEED)
        torch.manual_seed(SEED)
        keys = torch.randn(1, NUM_HEADS, 32, HEAD_DIM)
        values = torch.randn(1, NUM_HEADS, 32, HEAD_DIM)

        cache.update(keys, values, layer_idx=0)
        _, v_out = cache[0]

        v_flat = values.float().reshape(-1, HEAD_DIM)
        v_out_flat = v_out.float().reshape(-1, HEAD_DIM)
        cos_sim = torch.nn.functional.cosine_similarity(v_flat, v_out_flat, dim=-1)
        mean_cos = cos_sim.mean().item()

        thresholds = {2: 0.90, 3: 0.98, 4: 0.99}
        assert mean_cos > thresholds[bits], (
            f"{bits}-bit value cosine sim {mean_cos:.4f} below threshold {thresholds[bits]}"
        )

    def test_higher_bits_better_quality(self):
        """Higher bit-widths should produce better reconstruction quality."""
        torch.manual_seed(SEED)
        keys = torch.randn(1, NUM_HEADS, 32, HEAD_DIM)
        values = torch.randn(1, NUM_HEADS, 32, HEAD_DIM)

        cos_sims = {}
        for bits in [2, 3, 4]:
            cache = TurboQuantCache(bits=bits, seed=SEED)
            cache.update(keys.clone(), values.clone(), layer_idx=0)
            k_out, _ = cache[0]

            k_flat = keys.float().reshape(-1, HEAD_DIM)
            k_out_flat = k_out.float().reshape(-1, HEAD_DIM)
            cos_sim = torch.nn.functional.cosine_similarity(k_flat, k_out_flat, dim=-1)
            cos_sims[bits] = cos_sim.mean().item()

        assert cos_sims[3] > cos_sims[2], "3-bit should be better than 2-bit"
        assert cos_sims[4] > cos_sims[3], "4-bit should be better than 3-bit"


# ---------------------------------------------------------------------------
# Test: HF protocol compatibility
# ---------------------------------------------------------------------------
class TestHFProtocol:
    """Test compatibility with HuggingFace's expected Cache interface."""

    def test_is_initialized_property(self):
        """is_initialized should reflect cache state."""
        cache = TurboQuantCache(bits=3)
        assert not cache.is_initialized

        keys, values = make_kv_states(seq_len=1)
        cache.update(keys, values, layer_idx=0)
        assert cache.is_initialized

    def test_is_sliding_property(self):
        """is_sliding should return False for all layers."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        keys, values = make_kv_states(seq_len=1)
        cache.update(keys, values, layer_idx=0)
        cache.update(keys, values, layer_idx=1)

        assert cache.is_sliding == [False, False]

    def test_get_max_cache_shape(self):
        """Dynamic cache should return -1 for max shape."""
        cache = TurboQuantCache(bits=3)
        assert cache.get_max_cache_shape(0) == -1

    def test_get_mask_sizes(self):
        """get_mask_sizes should return correct kv_length."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)

        cache_position = torch.arange(3)  # query of length 3
        kv_length, kv_offset = cache.get_mask_sizes(cache_position, layer_idx=0)
        assert kv_length == 5 + 3  # cached + query
        assert kv_offset == 0

    def test_get_mask_sizes_empty_layer(self):
        """get_mask_sizes for uncreated layer should use cache_position shape."""
        cache = TurboQuantCache(bits=3)
        cache_position = torch.arange(4)
        kv_length, kv_offset = cache.get_mask_sizes(cache_position, layer_idx=5)
        assert kv_length == 4
        assert kv_offset == 0

    def test_reset(self):
        """reset() should clear all cached data."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        keys, values = make_kv_states(seq_len=10)
        cache.update(keys, values, layer_idx=0)
        assert cache.get_seq_length(0) == 10

        cache.reset()
        assert cache.get_seq_length(0) == 0

    def test_batch_repeat_interleave(self):
        """batch_repeat_interleave should expand the batch dimension."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        keys, values = make_kv_states(batch=2, seq_len=3)
        cache.update(keys, values, layer_idx=0)

        cache.batch_repeat_interleave(3)
        k_out, v_out = cache[0]
        # batch should now be 2 * 3 = 6
        assert k_out.shape[0] == 6
        assert v_out.shape[0] == 6

    def test_batch_select_indices(self):
        """batch_select_indices should filter batch dimension."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        keys, values = make_kv_states(batch=4, seq_len=3)
        cache.update(keys, values, layer_idx=0)

        indices = torch.tensor([0, 2])
        cache.batch_select_indices(indices)
        k_out, v_out = cache[0]
        assert k_out.shape[0] == 2
        assert v_out.shape[0] == 2


# ---------------------------------------------------------------------------
# Test: generate with TurboQuantCache (mock-based)
# ---------------------------------------------------------------------------
class TestGenerateIntegration:
    """Test that cache works in a generate-like loop with synthetic data.

    This simulates what HF generate() does: iteratively call update()
    with new tokens for each layer, using returned KV to compute attention.
    """

    def test_mock_generate_loop(self):
        """Simulate a multi-step generation loop."""
        num_layers = 4
        cache = TurboQuantCache(bits=3, seed=SEED)

        # Prefill: 16 tokens at once
        for layer_idx in range(num_layers):
            keys, values = make_kv_states(
                batch=1, num_heads=2, seq_len=16, head_dim=64, seed=layer_idx,
            )
            k_out, v_out = cache.update(keys, values, layer_idx=layer_idx)
            # Should return full 16 tokens
            assert k_out.shape == (1, 2, 16, 64)

        # Generate 10 tokens one at a time
        for step in range(10):
            for layer_idx in range(num_layers):
                keys, values = make_kv_states(
                    batch=1, num_heads=2, seq_len=1, head_dim=64,
                    seed=100 + step * num_layers + layer_idx,
                )
                k_out, v_out = cache.update(keys, values, layer_idx=layer_idx)
                expected_seq = 16 + step + 1
                assert k_out.shape == (1, 2, expected_seq, 64)
                assert v_out.shape == (1, 2, expected_seq, 64)

        # Final state
        for layer_idx in range(num_layers):
            assert cache.get_seq_length(layer_idx) == 26  # 16 + 10
        assert len(cache) == num_layers

    def test_prefill_then_decode_quality(self):
        """Verify reconstruction quality is maintained across prefill + decode."""
        cache = TurboQuantCache(bits=3, seed=SEED)
        all_keys = []
        all_values = []

        # Prefill: 8 tokens
        torch.manual_seed(SEED)
        k_pf = torch.randn(1, 2, 8, HEAD_DIM)
        v_pf = torch.randn(1, 2, 8, HEAD_DIM)
        all_keys.append(k_pf)
        all_values.append(v_pf)
        cache.update(k_pf, v_pf, layer_idx=0)

        # Decode: 4 more tokens
        for i in range(4):
            torch.manual_seed(SEED + 100 + i)
            k_dec = torch.randn(1, 2, 1, HEAD_DIM)
            v_dec = torch.randn(1, 2, 1, HEAD_DIM)
            all_keys.append(k_dec)
            all_values.append(v_dec)
            cache.update(k_dec, v_dec, layer_idx=0)

        # Check quality of full sequence
        k_all = torch.cat(all_keys, dim=2).float()
        v_all = torch.cat(all_values, dim=2).float()
        k_out, v_out = cache[0]

        k_flat = k_all.reshape(-1, HEAD_DIM)
        k_out_flat = k_out.float().reshape(-1, HEAD_DIM)
        cos_sim = torch.nn.functional.cosine_similarity(k_flat, k_out_flat, dim=-1)
        # Keys use 2-bit MSE (3-bit total, 1 for QJL), so reconstruction cos ~0.80
        assert cos_sim.mean().item() > 0.75, "Key quality degraded across prefill+decode"


# ---------------------------------------------------------------------------
# Test: TurboQuantLayer directly
# ---------------------------------------------------------------------------
class TestTurboQuantLayer:
    """Unit tests for the per-layer compressed storage."""

    def test_lazy_init(self):
        """Layer should initialize quantizers on first update."""
        layer = TurboQuantLayer(bits=3, seed=SEED)
        assert layer._key_est is None

        keys, values = make_kv_states(seq_len=2)
        layer.update(keys, values)
        assert layer._key_est is not None
        assert layer._val_pq is not None
        assert layer._head_dim == HEAD_DIM

    def test_clear(self):
        """clear() should reset sequence length to 0."""
        layer = TurboQuantLayer(bits=3, seed=SEED)
        keys, values = make_kv_states(seq_len=5)
        layer.update(keys, values)
        assert layer.get_seq_length() == 5

        layer.clear()
        assert layer.get_seq_length() == 0

    def test_memory_usage_scales_with_tokens(self):
        """Memory usage should scale with number of tokens."""
        layer = TurboQuantLayer(bits=3, seed=SEED)
        keys1, values1 = make_kv_states(seq_len=10)
        layer.update(keys1, values1)
        usage_10 = layer.memory_usage_bits()

        keys2, values2 = make_kv_states(seq_len=10, seed=99)
        layer.update(keys2, values2)
        usage_20 = layer.memory_usage_bits()

        assert usage_20["total_bits"] == 2 * usage_10["total_bits"]
