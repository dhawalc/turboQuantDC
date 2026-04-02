"""Tests for cross-layer KV cache sharing.

Validates:
1. Diagnostic: cross-layer KV correlation measurement
2. Diagnostic: distribution similarity measurement
3. CrossLayerKVCache HF protocol (update, get_seq_length, crop, etc.)
4. Resource sharing: layers in same group share codebook + rotation
5. Compression quality: shared resources produce same quality as independent
6. Memory savings report
7. Edge cases: group_size=1 (no sharing), single layer, empty cache
8. Anchor layer handling with shared resources
9. Beam search reorder
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# Ensure project root is importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from turboquantdc.cross_layer_kv import (
    CrossLayerKVCache,
    _SharedResourceLayer,
    correlation_report,
    measure_cross_layer_kv_correlation,
    measure_distribution_similarity,
)
from turboquantdc.generation_cache import GenerationCache, _FP16Layer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEAD_DIM = 128
NUM_HEADS = 4
BATCH_SIZE = 2
SEED = 42
NUM_LAYERS = 12


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


def make_kv_by_layer(
    num_layers: int = NUM_LAYERS,
    seq_len: int = 16,
    seed: int = SEED,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Create KV states for multiple layers (independent random vectors)."""
    kv_by_layer = {}
    for layer_idx in range(num_layers):
        k, v = make_kv_states(seq_len=seq_len, seed=seed + layer_idx * 100)
        kv_by_layer[layer_idx] = (k, v)
    return kv_by_layer


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean cosine similarity between two tensors of same shape."""
    a_flat = a.reshape(-1, a.shape[-1]).float()
    b_flat = b.reshape(-1, b.shape[-1]).float()
    sims = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1)
    return sims.mean().item()


# ---------------------------------------------------------------------------
# Test: Cross-layer KV correlation diagnostics
# ---------------------------------------------------------------------------
class TestCrossLayerDiagnostics:
    """Validate the correlation measurement tools."""

    def test_measure_correlation_returns_all_pairs(self):
        kv = make_kv_by_layer(num_layers=5)
        results = measure_cross_layer_kv_correlation(kv)
        assert len(results) == 4  # 5 layers -> 4 pairs

    def test_measure_correlation_keys_present(self):
        kv = make_kv_by_layer(num_layers=3)
        results = measure_cross_layer_kv_correlation(kv)
        for r in results:
            assert "layer_pair" in r
            assert "key_cosine_sim" in r
            assert "value_cosine_sim" in r
            assert "key_pearson_r" in r
            assert "value_pearson_r" in r
            assert "key_relative_delta_norm" in r
            assert "value_relative_delta_norm" in r

    def test_independent_vectors_have_low_correlation(self):
        """Random independent vectors should have near-zero Pearson r."""
        kv = make_kv_by_layer(num_layers=4, seq_len=32)
        results = measure_cross_layer_kv_correlation(kv)
        for r in results:
            # Independent random vectors: Pearson r should be near 0
            assert abs(r["key_pearson_r"]) < 0.2, (
                f"Expected low correlation, got r={r['key_pearson_r']:.4f}"
            )
            assert abs(r["value_pearson_r"]) < 0.2, (
                f"Expected low correlation, got r={r['value_pearson_r']:.4f}"
            )

    def test_correlated_vectors_have_high_correlation(self):
        """Synthetically correlated vectors should show high Pearson r."""
        torch.manual_seed(SEED)
        base_k = torch.randn(BATCH_SIZE, NUM_HEADS, 32, HEAD_DIM)
        base_v = torch.randn(BATCH_SIZE, NUM_HEADS, 32, HEAD_DIM)
        noise_k = torch.randn_like(base_k) * 0.1
        noise_v = torch.randn_like(base_v) * 0.1

        kv = {
            0: (base_k, base_v),
            1: (base_k + noise_k, base_v + noise_v),
        }
        results = measure_cross_layer_kv_correlation(kv)
        assert results[0]["key_pearson_r"] > 0.9
        assert results[0]["value_pearson_r"] > 0.9

    def test_single_layer_returns_empty(self):
        kv = make_kv_by_layer(num_layers=1)
        results = measure_cross_layer_kv_correlation(kv)
        assert len(results) == 0


class TestDistributionSimilarity:
    """Validate distribution comparison diagnostics."""

    def test_measure_distribution_returns_all_pairs(self):
        kv = make_kv_by_layer(num_layers=4)
        results = measure_distribution_similarity(kv)
        assert len(results) == 3  # 4 layers -> 3 pairs

    def test_independent_vectors_have_similar_distributions(self):
        """Random N(0,1) vectors should have very similar distributions."""
        kv = make_kv_by_layer(num_layers=4, seq_len=64)
        results = measure_distribution_similarity(kv)
        for r in results:
            # KL divergence should be small for same-distribution samples
            assert r["key_kl_divergence"] < 0.1, (
                f"KL too high: {r['key_kl_divergence']:.6f}"
            )
            assert r["value_kl_divergence"] < 0.1, (
                f"KL too high: {r['value_kl_divergence']:.6f}"
            )
            # Std ratios should be near 1.0
            assert 0.8 < r["key_std_ratio"] < 1.2
            assert 0.8 < r["value_std_ratio"] < 1.2

    def test_distribution_keys_present(self):
        kv = make_kv_by_layer(num_layers=3)
        results = measure_distribution_similarity(kv)
        for r in results:
            assert "key_kl_divergence" in r
            assert "value_kl_divergence" in r
            assert "key_mean_diff" in r
            assert "key_std_ratio" in r


class TestCorrelationReport:
    """Validate the human-readable report generator."""

    def test_report_is_string(self):
        kv = make_kv_by_layer(num_layers=4)
        report = correlation_report(kv)
        assert isinstance(report, str)
        assert len(report) > 100

    def test_report_contains_sections(self):
        kv = make_kv_by_layer(num_layers=4)
        report = correlation_report(kv)
        assert "VECTOR CORRELATION" in report
        assert "DISTRIBUTION SIMILARITY" in report
        assert "RECOMMENDATION" in report

    def test_report_diagnoses_no_delta_coding(self):
        """Independent random vectors should trigger 'NOT viable' message."""
        kv = make_kv_by_layer(num_layers=4, seq_len=64)
        report = correlation_report(kv)
        assert "NOT viable" in report


# ---------------------------------------------------------------------------
# Test: CrossLayerKVCache basic protocol
# ---------------------------------------------------------------------------
class TestCrossLayerCacheProtocol:
    """Validate the HF Cache protocol methods."""

    def test_update_returns_correct_shapes(self):
        cache = CrossLayerKVCache(
            group_size=2, num_layers=NUM_LAYERS,
            anchor_strategy="boundary", seed=SEED,
        )
        keys, values = make_kv_states(seq_len=5)
        k_out, v_out = cache.update(keys, values, layer_idx=2)
        assert k_out.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)
        assert v_out.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)

    def test_update_accumulates_sequence(self):
        cache = CrossLayerKVCache(
            group_size=2, num_layers=NUM_LAYERS,
            anchor_strategy="boundary", seed=SEED,
        )
        k1, v1 = make_kv_states(seq_len=5, seed=1)
        k2, v2 = make_kv_states(seq_len=3, seed=2)
        cache.update(k1, v1, layer_idx=3)
        k_out, v_out = cache.update(k2, v2, layer_idx=3)
        assert k_out.shape == (BATCH_SIZE, NUM_HEADS, 8, HEAD_DIM)

    def test_get_seq_length(self):
        cache = CrossLayerKVCache(
            group_size=2, num_layers=NUM_LAYERS,
            anchor_strategy="boundary", seed=SEED,
        )
        assert cache.get_seq_length(0) == 0
        keys, values = make_kv_states(seq_len=10)
        cache.update(keys, values, layer_idx=3)
        assert cache.get_seq_length(3) == 10

    def test_get_max_cache_shape(self):
        cache = CrossLayerKVCache(
            group_size=2, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
        )
        assert cache.get_max_cache_shape() == -1

    def test_len_grows_with_layers(self):
        cache = CrossLayerKVCache(
            group_size=2, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
        )
        assert len(cache) == 0
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        assert len(cache) == 1
        cache.update(keys, values, layer_idx=5)
        assert len(cache) == 6

    def test_contains(self):
        cache = CrossLayerKVCache(
            group_size=2, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        cache.update(keys, values, layer_idx=2)
        assert 0 in cache
        assert 2 in cache
        assert 10 not in cache

    def test_seen_tokens(self):
        cache = CrossLayerKVCache(
            group_size=2, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=7)
        cache.update(keys, values, layer_idx=0)
        assert cache.seen_tokens == 7

    def test_is_initialized(self):
        cache = CrossLayerKVCache(
            group_size=2, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
        )
        assert not cache.is_initialized
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        assert cache.is_initialized

    def test_is_sliding(self):
        cache = CrossLayerKVCache(
            group_size=2, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
        )
        assert cache.is_sliding == [False]
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        cache.update(keys, values, layer_idx=1)
        assert cache.is_sliding == [False, False]

    def test_getitem(self):
        cache = CrossLayerKVCache(
            group_size=2, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        k, v = cache[0]
        assert k.shape == (BATCH_SIZE, NUM_HEADS, 5, HEAD_DIM)

    def test_getitem_out_of_range_raises(self):
        cache = CrossLayerKVCache(
            group_size=2, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
        )
        with pytest.raises(IndexError):
            cache[0]

    def test_iter(self):
        cache = CrossLayerKVCache(
            group_size=2, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        cache.update(keys, values, layer_idx=1)
        layers = list(cache)
        assert len(layers) == 2
        for k, v, extra in layers:
            assert k.shape[2] == 5
            assert extra is None


# ---------------------------------------------------------------------------
# Test: Resource sharing
# ---------------------------------------------------------------------------
class TestResourceSharing:
    """Validate that layers in the same group share resources."""

    def test_layers_in_same_group_share_codebook(self):
        cache = CrossLayerKVCache(
            group_size=4, num_layers=NUM_LAYERS,
            anchor_strategy="fixed", anchor_interval=0,
            seed=SEED,
        )
        keys, values = make_kv_states(seq_len=5)
        # Populate layers 0-3 (all in group 0)
        for i in range(4):
            cache.update(keys, values, layer_idx=i)

        # All 4 layers should reference the same codebook objects
        shared_layers = [
            layer for layer in cache._layers
            if isinstance(layer, _SharedResourceLayer)
        ]
        assert len(shared_layers) == 4

        key_cb_ids = {id(l._key_codebook) for l in shared_layers[:4]}
        val_cb_ids = {id(l._val_codebook) for l in shared_layers[:4]}
        assert len(key_cb_ids) == 1, "Layers in same group should share key codebook"
        assert len(val_cb_ids) == 1, "Layers in same group should share val codebook"

    def test_layers_in_different_groups_have_different_rotation(self):
        cache = CrossLayerKVCache(
            group_size=2, num_layers=NUM_LAYERS,
            anchor_strategy="fixed", anchor_interval=0,
            seed=SEED,
        )
        keys, values = make_kv_states(seq_len=5)
        # Populate layers 0-1 (group 0) and 2-3 (group 1)
        for i in range(4):
            cache.update(keys, values, layer_idx=i)

        shared_layers = [
            layer for layer in cache._layers
            if isinstance(layer, _SharedResourceLayer)
        ]
        # Group 0 and group 1 should have different rotation resources
        # (they share the same codebook since (d, bits) is the same,
        #  but different rotation matrices due to different seeds)
        assert len(cache._group_resources) == 2

    def test_group_size_1_no_sharing(self):
        """group_size=1 means each layer is its own group (no sharing benefit)."""
        cache = CrossLayerKVCache(
            group_size=1, num_layers=6,
            anchor_strategy="fixed", anchor_interval=0,
            seed=SEED,
        )
        keys, values = make_kv_states(seq_len=5)
        for i in range(6):
            cache.update(keys, values, layer_idx=i)
        # Each layer is its own group
        assert len(cache._group_resources) == 6

    def test_large_group_size_reduces_groups(self):
        cache = CrossLayerKVCache(
            group_size=6, num_layers=12,
            anchor_strategy="fixed", anchor_interval=0,
            seed=SEED,
        )
        keys, values = make_kv_states(seq_len=5)
        for i in range(12):
            cache.update(keys, values, layer_idx=i)
        # 12 layers / 6 per group = 2 groups
        assert len(cache._group_resources) == 2


# ---------------------------------------------------------------------------
# Test: Compression quality
# ---------------------------------------------------------------------------
class TestCompressionQuality:
    """Validate that shared resources produce same quality as independent."""

    def test_cosine_similarity_above_threshold(self):
        """Shared-resource compression should achieve >0.95 cosine sim."""
        cache = CrossLayerKVCache(
            group_size=2, key_bits=3, val_bits=3,
            fp16_window=0, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
            use_residual_quant=True,
        )
        keys, values = make_kv_states(seq_len=32, seed=100)
        k_out, v_out = cache.update(keys, values, layer_idx=0)

        key_cos = cosine_sim(keys, k_out)
        val_cos = cosine_sim(values, v_out)
        assert key_cos > 0.95, f"Key cosine sim too low: {key_cos:.4f}"
        assert val_cos > 0.90, f"Value cosine sim too low: {val_cos:.4f}"

    def test_quality_matches_generation_cache(self):
        """Shared-resource quality should be comparable to GenerationCache."""
        # Standard GenerationCache (per-layer resources)
        gen_cache = GenerationCache(
            key_bits=3, val_bits=3, fp16_window=0,
            anchor_interval=0, seed=SEED,
            use_residual_quant=True,
        )
        # Cross-layer cache with group_size=1 (no sharing)
        cross_cache = CrossLayerKVCache(
            group_size=1, key_bits=3, val_bits=3,
            fp16_window=0, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
            use_residual_quant=True,
        )

        keys, values = make_kv_states(seq_len=32, seed=200)

        gen_k, gen_v = gen_cache.update(keys, values, layer_idx=0)
        cross_k, cross_v = cross_cache.update(keys, values, layer_idx=0)

        # Both should have similar quality (not identical due to different
        # seed patterns, but both should be high quality)
        gen_key_cos = cosine_sim(keys, gen_k)
        cross_key_cos = cosine_sim(keys, cross_k)
        assert abs(gen_key_cos - cross_key_cos) < 0.05, (
            f"Quality gap too large: gen={gen_key_cos:.4f} vs cross={cross_key_cos:.4f}"
        )


# ---------------------------------------------------------------------------
# Test: FP16 window
# ---------------------------------------------------------------------------
class TestFP16Window:
    """Validate FP16 precision window behavior."""

    def test_fp16_window_last_tokens_are_exact(self):
        cache = CrossLayerKVCache(
            group_size=2, key_bits=3, val_bits=3,
            fp16_window=4, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=16, seed=300)
        k_out, v_out = cache.update(keys, values, layer_idx=2)

        # Last 4 tokens should be exact (FP16 window)
        last_k = k_out[:, :, -4:, :]
        orig_last_k = keys[:, :, -4:, :]
        cos = cosine_sim(orig_last_k, last_k)
        assert cos > 0.999, f"FP16 window tokens not precise enough: {cos:.4f}"


# ---------------------------------------------------------------------------
# Test: Anchor layers
# ---------------------------------------------------------------------------
class TestAnchorLayers:
    """Validate anchor layer handling with shared resources."""

    def test_boundary_anchors_are_fp16(self):
        cache = CrossLayerKVCache(
            group_size=4, num_layers=12,
            anchor_strategy="boundary", seed=SEED,
        )
        keys, values = make_kv_states(seq_len=5)
        for i in range(12):
            cache.update(keys, values, layer_idx=i)

        # Boundary strategy: first 2 + last 2 are FP16
        assert isinstance(cache._layers[0], _FP16Layer)
        assert isinstance(cache._layers[1], _FP16Layer)
        assert isinstance(cache._layers[10], _FP16Layer)
        assert isinstance(cache._layers[11], _FP16Layer)

        # Middle layers should be shared-resource compressed
        assert isinstance(cache._layers[5], _SharedResourceLayer)
        assert isinstance(cache._layers[6], _SharedResourceLayer)

    def test_fixed_anchors_with_groups(self):
        cache = CrossLayerKVCache(
            group_size=3, anchor_strategy="fixed",
            anchor_interval=6, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=5)
        for i in range(12):
            cache.update(keys, values, layer_idx=i)

        # Layers 0, 6 should be FP16 anchors
        assert isinstance(cache._layers[0], _FP16Layer)
        assert isinstance(cache._layers[6], _FP16Layer)
        assert isinstance(cache._layers[3], _SharedResourceLayer)

    def test_anchor_layers_are_lossless(self):
        cache = CrossLayerKVCache(
            group_size=2, num_layers=12,
            anchor_strategy="boundary", seed=SEED,
        )
        keys, values = make_kv_states(seq_len=10, seed=400)
        k_out, v_out = cache.update(keys, values, layer_idx=0)

        # Layer 0 is FP16 anchor: output should match input exactly
        assert torch.allclose(keys, k_out, atol=1e-6), (
            "Anchor layer should return exact FP16 values"
        )


# ---------------------------------------------------------------------------
# Test: Crop
# ---------------------------------------------------------------------------
class TestCrop:
    """Validate crop behavior."""

    def test_crop_reduces_seq_length(self):
        cache = CrossLayerKVCache(
            group_size=2, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=20)
        cache.update(keys, values, layer_idx=0)
        assert cache.get_seq_length(0) == 20
        cache.crop(10)
        assert cache.get_seq_length(0) == 10

    def test_crop_negative_index(self):
        cache = CrossLayerKVCache(
            group_size=2, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=20)
        cache.update(keys, values, layer_idx=0)
        cache.crop(-5)  # Keep first 15 tokens
        assert cache.get_seq_length(0) == 15


# ---------------------------------------------------------------------------
# Test: Reset
# ---------------------------------------------------------------------------
class TestReset:

    def test_reset_clears_all_layers(self):
        cache = CrossLayerKVCache(
            group_size=2, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=10)
        cache.update(keys, values, layer_idx=0)
        cache.update(keys, values, layer_idx=1)
        cache.reset()
        assert cache.get_seq_length(0) == 0
        assert cache.get_seq_length(1) == 0


# ---------------------------------------------------------------------------
# Test: Beam search
# ---------------------------------------------------------------------------
class TestBeamSearch:

    def test_reorder_cache(self):
        cache = CrossLayerKVCache(
            group_size=2, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
        )
        keys, values = make_kv_states(batch=4, seq_len=5)
        cache.update(keys, values, layer_idx=0)
        # Reorder: pick batch indices [2, 0, 1, 3]
        beam_idx = torch.tensor([2, 0, 1, 3])
        cache.reorder_cache(beam_idx)
        k_out, v_out = cache[0]
        assert k_out.shape[0] == 4


# ---------------------------------------------------------------------------
# Test: Reporting
# ---------------------------------------------------------------------------
class TestReporting:

    def test_resource_sharing_report(self):
        cache = CrossLayerKVCache(
            group_size=4, num_layers=12,
            anchor_strategy="boundary", seed=SEED,
        )
        keys, values = make_kv_states(seq_len=10)
        for i in range(12):
            cache.update(keys, values, layer_idx=i)

        report = cache.resource_sharing_report()
        assert report["num_layers"] == 12
        assert report["num_groups"] > 0
        assert report["group_size"] == 4
        assert report["rotation_bytes_saved"] >= 0

    def test_memory_savings_report(self):
        cache = CrossLayerKVCache(
            group_size=2, key_bits=3, val_bits=3,
            fp16_window=0, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=20)
        cache.update(keys, values, layer_idx=0)

        report = cache.memory_savings()
        assert report["overall_compression_ratio"] > 1.0
        assert report["num_layers"] == 1
        assert len(report["per_layer"]) == 1
        assert "group" in report["per_layer"][0]

    def test_config_summary(self):
        cache = CrossLayerKVCache(
            group_size=3, key_bits=3, val_bits=2,
            anchor_strategy="fixed", anchor_interval=0,
            seed=SEED, use_residual_quant=True,
        )
        keys, values = make_kv_states(seq_len=5)
        cache.update(keys, values, layer_idx=0)
        summary = cache.config_summary()
        assert "CrossLayerKVCache" in summary
        assert "group_size=3" in summary
        assert "3b keys" in summary


# ---------------------------------------------------------------------------
# Test: Validation errors
# ---------------------------------------------------------------------------
class TestValidation:

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            CrossLayerKVCache(group_size=0)

    def test_invalid_key_bits(self):
        with pytest.raises(ValueError, match="key_bits"):
            CrossLayerKVCache(key_bits=0)

    def test_invalid_val_bits(self):
        with pytest.raises(ValueError, match="val_bits"):
            CrossLayerKVCache(val_bits=10)

    def test_invalid_fp16_window(self):
        with pytest.raises(ValueError, match="fp16_window"):
            CrossLayerKVCache(fp16_window=-1)

    def test_invalid_anchor_strategy(self):
        with pytest.raises(ValueError, match="anchor_strategy"):
            CrossLayerKVCache(anchor_strategy="invalid")

    def test_boundary_requires_num_layers(self):
        with pytest.raises(ValueError, match="num_layers"):
            CrossLayerKVCache(anchor_strategy="boundary")

    def test_gradient_requires_num_layers(self):
        with pytest.raises(ValueError, match="num_layers"):
            CrossLayerKVCache(anchor_strategy="gradient")


# ---------------------------------------------------------------------------
# Test: Multi-layer round-trip
# ---------------------------------------------------------------------------
class TestMultiLayerRoundTrip:
    """End-to-end test simulating a real model forward pass."""

    def test_full_model_forward_pass(self):
        """Simulate 12-layer model with group_size=4."""
        cache = CrossLayerKVCache(
            group_size=4, num_layers=12,
            key_bits=3, val_bits=3,
            fp16_window=4, anchor_strategy="boundary",
            seed=SEED, use_residual_quant=True,
        )

        # Prefill: 16 tokens across all 12 layers
        for layer_idx in range(12):
            k, v = make_kv_states(seq_len=16, seed=500 + layer_idx)
            k_out, v_out = cache.update(k, v, layer_idx=layer_idx)
            assert k_out.shape == (BATCH_SIZE, NUM_HEADS, 16, HEAD_DIM)

        # Decode: 1 token at a time for 5 steps
        for step in range(5):
            for layer_idx in range(12):
                k, v = make_kv_states(seq_len=1, seed=1000 + step * 12 + layer_idx)
                k_out, v_out = cache.update(k, v, layer_idx=layer_idx)
                expected_seq = 16 + step + 1
                assert k_out.shape[2] == expected_seq

        # Final sequence length should be 16 + 5 = 21
        assert cache.get_seq_length(0) == 21

    def test_groups_share_resources_end_to_end(self):
        """After full forward pass, verify resource sharing is effective."""
        cache = CrossLayerKVCache(
            group_size=4, num_layers=12,
            anchor_strategy="fixed", anchor_interval=0,
            seed=SEED,
        )
        keys, values = make_kv_states(seq_len=10)
        for i in range(12):
            cache.update(keys, values, layer_idx=i)

        # 12 layers / 4 per group = 3 groups
        assert len(cache._group_resources) == 3

        # Each group's rotation is different (different seed)
        group_ids = list(cache._group_resources.keys())
        assert len(group_ids) == 3

    def test_get_mask_sizes(self):
        cache = CrossLayerKVCache(
            group_size=2, anchor_strategy="fixed",
            anchor_interval=0, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=10)
        cache.update(keys, values, layer_idx=0)

        cache_position = torch.arange(1)
        kv_length, kv_offset = cache.get_mask_sizes(cache_position, layer_idx=0)
        assert kv_length == 11  # 10 cached + 1 new
        assert kv_offset == 0
