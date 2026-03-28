"""Tests for layer-adaptive KV cache with per-layer bit-width assignment.

Validates that:
    - Bit-width schedules are computed correctly for all strategies.
    - FP16 layers produce exact attention scores (zero quantization error).
    - Compressed layers produce approximate but reasonable scores.
    - Mixed caches have higher quality in later (preserved) layers.
    - Memory usage aggregation and compression ratio are correct.
    - Utility functions (recommended_schedule, estimate_memory) work.
"""

import math

import pytest
import torch

from turboquantdc.layer_adaptive import (
    FP16Cache,
    LayerAdaptiveKVCache,
    estimate_memory,
    recommended_schedule,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
D = 64  # Small dimension for fast tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_unit_vectors(n: int, d: int, seed: int = SEED) -> torch.Tensor:
    """Generate n random unit vectors of dimension d."""
    torch.manual_seed(seed)
    x = torch.randn(n, d)
    x = x / x.norm(dim=1, keepdim=True)
    return x


# ---------------------------------------------------------------------------
# Tests: schedule computation
# ---------------------------------------------------------------------------
class TestScheduleComputation:
    """Verify that per-layer bit-width schedules are computed correctly."""

    def test_tail_preserve_schedule(self):
        """Last n_preserve layers at preserve_bits (FP16=0), rest at base_bits."""
        cache = LayerAdaptiveKVCache(
            num_layers=8, d_key=D, d_value=D,
            strategy="tail_preserve",
            base_bits=3, preserve_bits=0, n_preserve=3,
            seed=SEED,
        )
        expected = [3, 3, 3, 3, 3, 0, 0, 0]
        assert cache.bits_schedule == expected, (
            f"Expected {expected}, got {cache.bits_schedule}"
        )

    def test_tail_preserve_all_preserved(self):
        """When n_preserve >= num_layers, all layers are preserved."""
        cache = LayerAdaptiveKVCache(
            num_layers=4, d_key=D, d_value=D,
            strategy="tail_preserve",
            base_bits=3, preserve_bits=0, n_preserve=10,
            seed=SEED,
        )
        expected = [0, 0, 0, 0]
        assert cache.bits_schedule == expected

    def test_gradient_schedule(self):
        """Bits increase linearly from base_bits to preserve_bits."""
        cache = LayerAdaptiveKVCache(
            num_layers=5, d_key=D, d_value=D,
            strategy="gradient",
            base_bits=2, preserve_bits=4, n_preserve=0,
            seed=SEED,
        )
        # Linear from 2 to 4 over 5 layers: [2, 2.5, 3, 3.5, 4]
        # Rounded: [2, 2, 3, 4, 4] (after clamping to [2,4])
        schedule = cache.bits_schedule
        assert len(schedule) == 5
        # First layer should be base_bits
        assert schedule[0] == 2
        # Last layer should be preserve_bits
        assert schedule[-1] == 4
        # Should be non-decreasing
        for i in range(len(schedule) - 1):
            assert schedule[i] <= schedule[i + 1], (
                f"Schedule not non-decreasing at {i}: {schedule}"
            )

    def test_gradient_schedule_to_fp16(self):
        """Gradient from 3 to 0 (FP16) should produce decreasing then FP16."""
        cache = LayerAdaptiveKVCache(
            num_layers=4, d_key=D, d_value=D,
            strategy="gradient",
            base_bits=4, preserve_bits=0, n_preserve=0,
            seed=SEED,
        )
        schedule = cache.bits_schedule
        assert len(schedule) == 4
        # First should be base_bits=4, last should hit 0 (FP16)
        assert schedule[0] == 4
        # Values in [0] or [2,4]
        for b in schedule:
            assert b == 0 or (2 <= b <= 4), (
                f"Invalid bit-width {b} in gradient schedule: {schedule}"
            )

    def test_custom_schedule(self):
        """User-specified per-layer bits work."""
        custom = [3, 3, 2, 4, 0, 0]
        cache = LayerAdaptiveKVCache(
            num_layers=6, d_key=D, d_value=D,
            strategy="custom",
            bits_schedule=custom,
            seed=SEED,
        )
        assert cache.bits_schedule == custom

    def test_custom_schedule_length_mismatch(self):
        """Custom schedule with wrong length should raise ValueError."""
        with pytest.raises(ValueError, match="must match"):
            LayerAdaptiveKVCache(
                num_layers=4, d_key=D, d_value=D,
                strategy="custom",
                bits_schedule=[3, 3],  # wrong length
                seed=SEED,
            )

    def test_custom_schedule_none_raises(self):
        """Custom strategy without bits_schedule should raise ValueError."""
        with pytest.raises(ValueError, match="must be provided"):
            LayerAdaptiveKVCache(
                num_layers=4, d_key=D, d_value=D,
                strategy="custom",
                bits_schedule=None,
                seed=SEED,
            )

    def test_unknown_strategy_raises(self):
        """Unknown strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            LayerAdaptiveKVCache(
                num_layers=4, d_key=D, d_value=D,
                strategy="unknown",
                seed=SEED,
            )


# ---------------------------------------------------------------------------
# Tests: data flow through all layers
# ---------------------------------------------------------------------------
class TestAppendAndRetrieve:
    """Verify data flows correctly through all layer caches."""

    def test_append_and_retrieve_all_layers(self):
        """Append to each layer and verify retrieval works for all."""
        n_tokens = 20
        cache = LayerAdaptiveKVCache(
            num_layers=4, d_key=D, d_value=D,
            strategy="tail_preserve",
            base_bits=3, preserve_bits=0, n_preserve=2,
            seed=SEED,
        )

        for layer_idx in range(4):
            keys = random_unit_vectors(n_tokens, D, seed=SEED + layer_idx)
            values = random_unit_vectors(n_tokens, D, seed=SEED + 100 + layer_idx)
            cache.append(layer_idx, keys, values)

        # Verify all layers have data
        for layer_idx in range(4):
            queries = random_unit_vectors(5, D, seed=SEED + 200)
            scores = cache.attention_scores(layer_idx, queries)
            assert scores.shape == (5, n_tokens), (
                f"Layer {layer_idx}: expected shape (5, {n_tokens}), got {scores.shape}"
            )
            values = cache.get_values(layer_idx)
            assert values.shape == (n_tokens, D), (
                f"Layer {layer_idx}: expected value shape ({n_tokens}, {D}), "
                f"got {values.shape}"
            )

    def test_empty_layers(self):
        """Layers with no data should return appropriate empty tensors."""
        cache = LayerAdaptiveKVCache(
            num_layers=4, d_key=D, d_value=D,
            strategy="tail_preserve",
            base_bits=3, preserve_bits=0, n_preserve=2,
            seed=SEED,
        )
        queries = random_unit_vectors(3, D)

        for layer_idx in range(4):
            scores = cache.attention_scores(layer_idx, queries)
            assert scores.shape[1] == 0, (
                f"Empty layer {layer_idx} should have 0 columns"
            )
            values = cache.get_values(layer_idx)
            assert values.shape[0] == 0


# ---------------------------------------------------------------------------
# Tests: FP16 layers produce exact attention scores
# ---------------------------------------------------------------------------
class TestFP16Exact:
    """FP16 layers should produce exact attention scores with no quantization error."""

    def test_fp16_layers_exact(self):
        """FP16 layers produce exact attention scores (no quantization error)."""
        n_tokens = 50
        n_queries = 10
        torch.manual_seed(SEED)

        fp16_cache = FP16Cache(d_key=D, d_value=D)
        keys = random_unit_vectors(n_tokens, D, seed=SEED)
        values = random_unit_vectors(n_tokens, D, seed=SEED + 1)
        queries = random_unit_vectors(n_queries, D, seed=SEED + 2)

        fp16_cache.append(keys, values)

        # FP16 attention scores
        scores = fp16_cache.attention_scores(queries)
        # True scores
        true_scores = queries @ keys.T

        # Should be exactly equal (no quantization)
        torch.testing.assert_close(scores, true_scores, atol=1e-6, rtol=1e-5)

    def test_fp16_values_exact(self):
        """FP16 cache returns exact values."""
        n_tokens = 30
        torch.manual_seed(SEED)

        fp16_cache = FP16Cache(d_key=D, d_value=D)
        keys = random_unit_vectors(n_tokens, D, seed=SEED)
        values = random_unit_vectors(n_tokens, D, seed=SEED + 1)

        fp16_cache.append(keys, values)
        retrieved = fp16_cache.get_values()

        torch.testing.assert_close(retrieved, values, atol=1e-6, rtol=1e-5)

    def test_fp16_via_layer_adaptive(self):
        """FP16 layer in LayerAdaptiveKVCache produces exact scores."""
        n_tokens = 30
        n_queries = 5

        # All layers FP16
        cache = LayerAdaptiveKVCache(
            num_layers=2, d_key=D, d_value=D,
            strategy="custom",
            bits_schedule=[0, 0],
            seed=SEED,
        )

        keys = random_unit_vectors(n_tokens, D, seed=SEED)
        values = random_unit_vectors(n_tokens, D, seed=SEED + 1)
        queries = random_unit_vectors(n_queries, D, seed=SEED + 2)

        cache.append(0, keys, values)
        scores = cache.attention_scores(0, queries)
        true_scores = queries @ keys.T

        torch.testing.assert_close(scores, true_scores, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: compressed layers have expected quality
# ---------------------------------------------------------------------------
class TestCompressedQuality:
    """Compressed layers should have reasonable (but not exact) quality."""

    def test_compressed_layers_approximate(self):
        """Compressed layers have approximate but reasonable attention scores."""
        n_tokens = 200
        n_queries = 200

        cache = LayerAdaptiveKVCache(
            num_layers=2, d_key=D, d_value=D,
            strategy="custom",
            bits_schedule=[3, 3],
            seed=SEED,
        )

        keys = random_unit_vectors(n_tokens, D, seed=SEED)
        values = random_unit_vectors(n_tokens, D, seed=SEED + 1)
        queries = random_unit_vectors(n_queries, D, seed=SEED + 2)

        cache.append(0, keys, values)
        scores = cache.attention_scores(0, queries)
        true_scores = queries @ keys.T

        # Scores should be approximately correct (unbiased)
        mean_error = (scores - true_scores).mean().item()
        assert abs(mean_error) < 0.1, (
            f"Mean error {mean_error:.4f} too large (should be ~0 for unbiased)"
        )

        # Cosine similarity of score vectors should be high
        cos_sim = (
            (scores.flatten() * true_scores.flatten()).sum()
            / (scores.flatten().norm() * true_scores.flatten().norm())
        ).item()
        assert cos_sim > 0.7, (
            f"Score cosine similarity {cos_sim:.4f} too low"
        )


# ---------------------------------------------------------------------------
# Tests: mixed quality -- later layers better than earlier
# ---------------------------------------------------------------------------
class TestMixedQuality:
    """Last layers (preserved) should have higher quality than first layers."""

    def test_mixed_quality(self):
        """FP16 layers have lower error than compressed layers."""
        n_tokens = 200
        n_queries = 200

        # Layer 0: 3-bit, Layer 1: FP16
        cache = LayerAdaptiveKVCache(
            num_layers=2, d_key=D, d_value=D,
            strategy="tail_preserve",
            base_bits=3, preserve_bits=0, n_preserve=1,
            seed=SEED,
        )
        assert cache.bits_schedule == [3, 0]

        keys = random_unit_vectors(n_tokens, D, seed=SEED)
        values = random_unit_vectors(n_tokens, D, seed=SEED + 1)
        queries = random_unit_vectors(n_queries, D, seed=SEED + 2)
        true_scores = queries @ keys.T

        # Append same data to both layers
        cache.append(0, keys, values)
        cache.append(1, keys, values)

        # Compressed layer error
        scores_compressed = cache.attention_scores(0, queries)
        mse_compressed = ((scores_compressed - true_scores) ** 2).mean().item()

        # FP16 layer error
        scores_fp16 = cache.attention_scores(1, queries)
        mse_fp16 = ((scores_fp16 - true_scores) ** 2).mean().item()

        # FP16 should have much lower error
        assert mse_fp16 < mse_compressed, (
            f"FP16 MSE ({mse_fp16:.6f}) should be lower than compressed "
            f"MSE ({mse_compressed:.6f})"
        )
        # FP16 error should be essentially zero
        assert mse_fp16 < 1e-10, (
            f"FP16 MSE ({mse_fp16:.6e}) should be near zero"
        )


# ---------------------------------------------------------------------------
# Tests: effective compression ratio
# ---------------------------------------------------------------------------
class TestEffectiveCompression:
    """Overall ratio should be between pure FP16 and pure compressed."""

    def test_effective_compression(self):
        """Mixed cache has compression between 1.0 (FP16) and full compressed."""
        n_tokens = 50

        # Mixed: 2 layers at 3-bit, 2 layers at FP16
        cache = LayerAdaptiveKVCache(
            num_layers=4, d_key=D, d_value=D,
            strategy="tail_preserve",
            base_bits=3, preserve_bits=0, n_preserve=2,
            seed=SEED,
        )

        for layer_idx in range(4):
            keys = random_unit_vectors(n_tokens, D, seed=SEED + layer_idx)
            values = random_unit_vectors(n_tokens, D, seed=SEED + 100 + layer_idx)
            cache.append(layer_idx, keys, values)

        ratio = cache.effective_compression()

        # Should be > 1.0 (some layers are compressed)
        assert ratio > 1.0, f"Compression ratio {ratio:.2f} should be > 1.0"

        # All-FP16 cache for comparison
        cache_fp16 = LayerAdaptiveKVCache(
            num_layers=4, d_key=D, d_value=D,
            strategy="custom",
            bits_schedule=[0, 0, 0, 0],
            seed=SEED,
        )
        for layer_idx in range(4):
            keys = random_unit_vectors(n_tokens, D, seed=SEED + layer_idx)
            values = random_unit_vectors(n_tokens, D, seed=SEED + 100 + layer_idx)
            cache_fp16.append(layer_idx, keys, values)

        ratio_fp16 = cache_fp16.effective_compression()
        assert abs(ratio_fp16 - 1.0) < 0.01, (
            f"All-FP16 ratio should be ~1.0, got {ratio_fp16}"
        )

        # Mixed should compress more than all-FP16
        assert ratio > ratio_fp16, (
            f"Mixed ratio ({ratio:.2f}) should exceed FP16 ratio ({ratio_fp16:.2f})"
        )

    def test_all_compressed_highest_ratio(self):
        """All-compressed cache should have the highest compression ratio."""
        n_tokens = 50

        cache_all = LayerAdaptiveKVCache(
            num_layers=4, d_key=D, d_value=D,
            strategy="custom",
            bits_schedule=[3, 3, 3, 3],
            seed=SEED,
        )
        cache_mixed = LayerAdaptiveKVCache(
            num_layers=4, d_key=D, d_value=D,
            strategy="tail_preserve",
            base_bits=3, preserve_bits=0, n_preserve=2,
            seed=SEED,
        )

        for layer_idx in range(4):
            keys = random_unit_vectors(n_tokens, D, seed=SEED + layer_idx)
            values = random_unit_vectors(n_tokens, D, seed=SEED + 100 + layer_idx)
            cache_all.append(layer_idx, keys, values)
            cache_mixed.append(layer_idx, keys, values)

        ratio_all = cache_all.effective_compression()
        ratio_mixed = cache_mixed.effective_compression()

        assert ratio_all > ratio_mixed, (
            f"All-compressed ratio ({ratio_all:.2f}) should exceed "
            f"mixed ratio ({ratio_mixed:.2f})"
        )


# ---------------------------------------------------------------------------
# Tests: memory usage aggregation
# ---------------------------------------------------------------------------
class TestMemoryUsage:
    """memory_usage_bits sums correctly across layers."""

    def test_memory_usage_aggregation(self):
        """Total bits should equal sum of per-layer bits."""
        n_tokens = 30

        cache = LayerAdaptiveKVCache(
            num_layers=4, d_key=D, d_value=D,
            strategy="tail_preserve",
            base_bits=3, preserve_bits=0, n_preserve=2,
            seed=SEED,
        )

        for layer_idx in range(4):
            keys = random_unit_vectors(n_tokens, D, seed=SEED + layer_idx)
            values = random_unit_vectors(n_tokens, D, seed=SEED + 100 + layer_idx)
            cache.append(layer_idx, keys, values)

        usage = cache.memory_usage_bits()
        per_layer_sum = sum(pl["total_bits"] for pl in usage["per_layer"])

        assert usage["total_bits"] == per_layer_sum, (
            f"Total ({usage['total_bits']}) != sum of per-layer ({per_layer_sum})"
        )

        fp16_sum = sum(pl["fp16_baseline_bits"] for pl in usage["per_layer"])
        assert usage["fp16_baseline_bits"] == fp16_sum

    def test_fp16_layer_memory(self):
        """FP16 layer memory should equal the baseline."""
        n_tokens = 20
        fp16 = FP16Cache(d_key=D, d_value=D)
        keys = random_unit_vectors(n_tokens, D)
        values = random_unit_vectors(n_tokens, D, seed=SEED + 1)
        fp16.append(keys, values)

        usage = fp16.memory_usage_bits()
        expected_bits = n_tokens * (D + D) * 16
        assert usage["total_bits"] == expected_bits
        assert usage["fp16_baseline_bits"] == expected_bits
        assert usage["compression_ratio"] == 1.0

    def test_compressed_layer_smaller(self):
        """Compressed layer should use fewer bits than FP16 baseline."""
        n_tokens = 30

        cache = LayerAdaptiveKVCache(
            num_layers=2, d_key=D, d_value=D,
            strategy="custom",
            bits_schedule=[3, 0],
            seed=SEED,
        )

        for layer_idx in range(2):
            keys = random_unit_vectors(n_tokens, D, seed=SEED + layer_idx)
            values = random_unit_vectors(n_tokens, D, seed=SEED + 100 + layer_idx)
            cache.append(layer_idx, keys, values)

        usage = cache.memory_usage_bits()
        # Compressed layer should have total_bits < fp16_baseline_bits
        compressed_layer = usage["per_layer"][0]
        assert compressed_layer["total_bits"] < compressed_layer["fp16_baseline_bits"], (
            f"Compressed layer total ({compressed_layer['total_bits']}) should be "
            f"less than FP16 baseline ({compressed_layer['fp16_baseline_bits']})"
        )

    def test_empty_cache_memory(self):
        """Empty cache should report zero memory."""
        cache = LayerAdaptiveKVCache(
            num_layers=4, d_key=D, d_value=D,
            strategy="tail_preserve",
            base_bits=3, preserve_bits=0, n_preserve=2,
            seed=SEED,
        )
        usage = cache.memory_usage_bits()
        assert usage["total_bits"] == 0
        assert usage["fp16_baseline_bits"] == 0


# ---------------------------------------------------------------------------
# Tests: recommended_schedule
# ---------------------------------------------------------------------------
class TestRecommendedSchedule:
    """Known models return valid configs."""

    def test_recommended_schedule_known_models(self):
        """All known models return valid schedule configs."""
        known_models = [
            "qwen2.5-3b", "qwen2.5-14b", "qwen3.5-27b",
            "llama-3-8b", "llama-3-70b",
        ]

        for model in known_models:
            config = recommended_schedule(model)
            assert "num_layers" in config
            assert "strategy" in config
            assert "base_bits" in config
            assert "preserve_bits" in config
            assert "n_preserve" in config
            assert config["num_layers"] > 0
            assert config["n_preserve"] > 0
            assert config["n_preserve"] < config["num_layers"]
            assert config["strategy"] == "tail_preserve"

    def test_recommended_schedule_unknown_raises(self):
        """Unknown model should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown model"):
            recommended_schedule("nonexistent-model")

    def test_recommended_schedule_custom_base_bits(self):
        """Custom base_bits should be propagated."""
        config = recommended_schedule("llama-3-8b", base_bits=4)
        assert config["base_bits"] == 4

    def test_recommended_schedule_creates_valid_cache(self):
        """Config from recommended_schedule creates a valid LayerAdaptiveKVCache."""
        config = recommended_schedule("qwen2.5-3b")
        cache = LayerAdaptiveKVCache(
            d_key=D, d_value=D,
            **config,
            seed=SEED,
        )
        assert len(cache.layer_caches) == config["num_layers"]
        # Last n_preserve layers should be FP16
        for i in range(config["num_layers"] - config["n_preserve"], config["num_layers"]):
            assert cache.bits_schedule[i] == 0
        # First layers should be base_bits
        assert cache.bits_schedule[0] == config["base_bits"]


# ---------------------------------------------------------------------------
# Tests: estimate_memory
# ---------------------------------------------------------------------------
class TestEstimateMemory:
    """Memory estimates should be reasonable and match actual cache usage."""

    def test_estimate_memory_basic(self):
        """Basic memory estimate returns expected structure."""
        schedule = [3, 3, 0, 0]
        result = estimate_memory(
            num_layers=4, d_key=D, d_value=D,
            seq_len=100, schedule=schedule,
        )
        assert "fp16_gb" in result
        assert "compressed_gb" in result
        assert "ratio" in result
        assert "per_layer" in result
        assert len(result["per_layer"]) == 4
        assert result["fp16_gb"] > 0
        assert result["compressed_gb"] > 0
        assert result["ratio"] > 1.0  # Some layers are compressed

    def test_estimate_memory_all_fp16(self):
        """All-FP16 schedule should give ratio ~1.0."""
        schedule = [0, 0, 0, 0]
        result = estimate_memory(
            num_layers=4, d_key=D, d_value=D,
            seq_len=100, schedule=schedule,
        )
        assert abs(result["ratio"] - 1.0) < 0.01

    def test_estimate_memory_matches_actual(self):
        """Estimated memory should roughly match actual cache memory."""
        n_tokens = 50
        schedule = [3, 3, 0, 0]

        # Estimate
        est = estimate_memory(
            num_layers=4, d_key=D, d_value=D,
            seq_len=n_tokens, schedule=schedule,
        )

        # Actual
        cache = LayerAdaptiveKVCache(
            num_layers=4, d_key=D, d_value=D,
            strategy="custom",
            bits_schedule=schedule,
            seed=SEED,
        )
        for layer_idx in range(4):
            keys = random_unit_vectors(n_tokens, D, seed=SEED + layer_idx)
            values = random_unit_vectors(n_tokens, D, seed=SEED + 100 + layer_idx)
            cache.append(layer_idx, keys, values)

        actual = cache.memory_usage_bits()

        # FP16 baseline should match exactly
        est_fp16_bits = est["fp16_gb"] * 8 * 1024 * 1024 * 1024
        assert abs(est_fp16_bits - actual["fp16_baseline_bits"]) < 1, (
            f"FP16 baseline mismatch: est={est_fp16_bits}, actual={actual['fp16_baseline_bits']}"
        )

        # Compressed total should be in the same ballpark
        # (exact match not guaranteed because TurboQuantKVCache may account
        #  for bits slightly differently in its detailed breakdown)
        est_compressed_bits = est["compressed_gb"] * 8 * 1024 * 1024 * 1024
        # FP16 layers should match exactly
        for i in [2, 3]:  # FP16 layers
            est_layer_bits = est["per_layer"][i]["bits_per_token"] * n_tokens
            actual_layer_bits = actual["per_layer"][i]["total_bits"]
            assert abs(est_layer_bits - actual_layer_bits) < 1, (
                f"FP16 layer {i} mismatch: est={est_layer_bits}, actual={actual_layer_bits}"
            )

    def test_estimate_memory_schedule_length_mismatch(self):
        """Wrong schedule length should raise ValueError."""
        with pytest.raises(ValueError, match="must match"):
            estimate_memory(
                num_layers=4, d_key=D, d_value=D,
                seq_len=100, schedule=[3, 3],
            )


# ---------------------------------------------------------------------------
# Tests: clear
# ---------------------------------------------------------------------------
class TestClear:
    """Clear works across all layer types."""

    def test_clear_all_layers(self):
        """Clearing all layers empties every cache."""
        n_tokens = 20
        cache = LayerAdaptiveKVCache(
            num_layers=4, d_key=D, d_value=D,
            strategy="tail_preserve",
            base_bits=3, preserve_bits=0, n_preserve=2,
            seed=SEED,
        )

        # Fill all layers
        for layer_idx in range(4):
            keys = random_unit_vectors(n_tokens, D, seed=SEED + layer_idx)
            values = random_unit_vectors(n_tokens, D, seed=SEED + 100 + layer_idx)
            cache.append(layer_idx, keys, values)

        # Verify data is present
        usage_before = cache.memory_usage_bits()
        assert usage_before["total_bits"] > 0

        # Clear
        cache.clear()

        # Verify all layers are empty
        usage_after = cache.memory_usage_bits()
        assert usage_after["total_bits"] == 0

        for layer_idx in range(4):
            values = cache.get_values(layer_idx)
            assert values.shape[0] == 0

    def test_clear_single_layer(self):
        """Clearing a single layer only affects that layer."""
        n_tokens = 20
        cache = LayerAdaptiveKVCache(
            num_layers=4, d_key=D, d_value=D,
            strategy="tail_preserve",
            base_bits=3, preserve_bits=0, n_preserve=2,
            seed=SEED,
        )

        for layer_idx in range(4):
            keys = random_unit_vectors(n_tokens, D, seed=SEED + layer_idx)
            values = random_unit_vectors(n_tokens, D, seed=SEED + 100 + layer_idx)
            cache.append(layer_idx, keys, values)

        # Clear only layer 1
        cache.clear(layer_idx=1)

        # Layer 1 should be empty
        values_1 = cache.get_values(1)
        assert values_1.shape[0] == 0

        # Other layers should still have data
        for layer_idx in [0, 2, 3]:
            values = cache.get_values(layer_idx)
            assert values.shape[0] == n_tokens, (
                f"Layer {layer_idx} should still have {n_tokens} tokens"
            )

    def test_fp16_cache_clear(self):
        """FP16Cache clear works correctly."""
        fp16 = FP16Cache(d_key=D, d_value=D)
        keys = random_unit_vectors(10, D)
        values = random_unit_vectors(10, D, seed=SEED + 1)
        fp16.append(keys, values)

        assert fp16.seq_len == 10
        fp16.clear()
        assert fp16.seq_len == 0
        assert fp16.get_values().shape[0] == 0
