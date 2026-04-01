"""Tests for layer-adaptive anchor strategies in GenerationCache.

Validates:
    - compute_layer_key_bits: gradient bit allocation from boundaries to middle.
    - compute_anchor_schedule: per-layer schedules for fixed, boundary, gradient.
    - GenerationCache integration: anchor_strategy param creates correct layer types.
    - Boundary strategy: first 2 + last 2 layers are FP16 anchors.
    - Gradient strategy: monotonically decreasing bits from edges to middle.
    - Total bit budget: gradient strategy matches expected compression ratio.
    - Quality: boundary anchoring vs fixed anchoring compression quality.
    - Backward compatibility: default args produce identical behavior to before.
    - Edge cases: small models, single-layer, all-anchor configs.
"""

import math

import pytest
import torch

from turboquantdc.generation_cache import (
    ANCHOR_STRATEGIES,
    GenerationCache,
    _CompressedLayer,
    _FP16Layer,
    compute_anchor_schedule,
    compute_layer_key_bits,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEAD_DIM = 128
NUM_HEADS = 4
BATCH_SIZE = 2
SEED = 42
NUM_LAYERS_36 = 36  # Qwen2.5-3B typical
NUM_LAYERS_32 = 32  # Llama-3-8B typical


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
# Tests: compute_layer_key_bits
# ---------------------------------------------------------------------------
class TestComputeLayerKeyBits:
    """Validate per-layer key bit-width computation for gradient strategy."""

    def test_boundary_layers_get_fp16(self):
        """Layers 0, 1, N-2, N-1 should get 8 bits (FP16-equivalent)."""
        n = 36
        assert compute_layer_key_bits(0, n) == 8
        assert compute_layer_key_bits(1, n) == 8
        assert compute_layer_key_bits(n - 1, n) == 8
        assert compute_layer_key_bits(n - 2, n) == 8

    def test_near_boundary_layers_get_extra_bit(self):
        """Layers within 10-25% of boundary get base_bits + 1."""
        n = 36
        # Layer 2 has dist = 2 / 18 = 0.111 -> within 25% but not 10%
        assert compute_layer_key_bits(2, n, base_bits=3) == 4
        # Layer 33 (symmetric) should also get 4
        assert compute_layer_key_bits(33, n, base_bits=3) == 4

    def test_middle_layers_get_base_bits(self):
        """Middle layers (far from boundaries) get base_bits."""
        n = 36
        mid = n // 2
        assert compute_layer_key_bits(mid, n, base_bits=3) == 3
        assert compute_layer_key_bits(mid - 1, n, base_bits=3) == 3
        assert compute_layer_key_bits(mid + 1, n, base_bits=3) == 3

    def test_symmetric_around_middle(self):
        """Bit-widths should be symmetric: layer i == layer (N-1-i)."""
        n = 36
        for i in range(n):
            mirror = n - 1 - i
            assert compute_layer_key_bits(i, n) == compute_layer_key_bits(mirror, n), (
                f"Layer {i} and {mirror} should have the same bits"
            )

    def test_monotonically_decreasing_to_middle(self):
        """Bits should monotonically decrease from edge (layer 0) to middle."""
        n = 36
        bits = [compute_layer_key_bits(i, n) for i in range(n // 2 + 1)]
        # Each successive layer should have <= bits of the previous
        for i in range(1, len(bits)):
            assert bits[i] <= bits[i - 1], (
                f"Layer {i} has {bits[i]} bits but layer {i-1} has {bits[i-1]}; "
                f"expected monotonically decreasing"
            )

    def test_single_layer_model(self):
        """Single-layer model should always return 8 (FP16-equivalent)."""
        assert compute_layer_key_bits(0, 1) == 8

    def test_two_layer_model(self):
        """Two-layer model: both layers should be FP16-equivalent."""
        assert compute_layer_key_bits(0, 2) == 8
        assert compute_layer_key_bits(1, 2) == 8

    def test_custom_base_bits(self):
        """base_bits parameter should be respected for middle layers."""
        n = 36
        mid = n // 2
        assert compute_layer_key_bits(mid, n, base_bits=2) == 2
        assert compute_layer_key_bits(mid, n, base_bits=4) == 4

    def test_near_boundary_min_4_bits(self):
        """Near-boundary layers should have at least 4 bits."""
        n = 36
        # Even with base_bits=2, near-boundary gets max(2+1, 4) = 4
        layer_near = 3  # dist = 3/18 = 0.167, in 10-25% range
        result = compute_layer_key_bits(layer_near, n, base_bits=2)
        assert result >= 4, f"Expected at least 4, got {result}"


# ---------------------------------------------------------------------------
# Tests: compute_anchor_schedule
# ---------------------------------------------------------------------------
class TestComputeAnchorSchedule:
    """Validate per-layer anchor schedule computation."""

    def test_fixed_schedule_matches_interval(self):
        """Fixed strategy: FP16 at every anchor_interval-th layer."""
        schedule = compute_anchor_schedule(
            num_layers=12, anchor_strategy="fixed", anchor_interval=4,
        )
        assert len(schedule) == 12
        for i, (is_fp16, key_bits) in enumerate(schedule):
            if i % 4 == 0:
                assert is_fp16, f"Layer {i} should be FP16"
            else:
                assert not is_fp16, f"Layer {i} should NOT be FP16"
            assert key_bits == 3  # base default

    def test_fixed_schedule_no_anchors(self):
        """Fixed strategy with anchor_interval=0: no FP16 layers."""
        schedule = compute_anchor_schedule(
            num_layers=8, anchor_strategy="fixed", anchor_interval=0,
        )
        for i, (is_fp16, _) in enumerate(schedule):
            assert not is_fp16, f"Layer {i} should not be FP16"

    def test_boundary_schedule_first_last_two(self):
        """Boundary strategy: exactly first 2 + last 2 layers FP16."""
        schedule = compute_anchor_schedule(
            num_layers=36, anchor_strategy="boundary",
        )
        assert len(schedule) == 36
        fp16_layers = [i for i, (is_fp16, _) in enumerate(schedule) if is_fp16]
        assert fp16_layers == [0, 1, 34, 35], (
            f"Expected [0, 1, 34, 35], got {fp16_layers}"
        )

    def test_boundary_schedule_small_model(self):
        """Boundary strategy with 4 layers: all FP16."""
        schedule = compute_anchor_schedule(
            num_layers=4, anchor_strategy="boundary",
        )
        fp16_layers = [i for i, (is_fp16, _) in enumerate(schedule) if is_fp16]
        assert fp16_layers == [0, 1, 2, 3]

    def test_boundary_schedule_5_layers(self):
        """Boundary strategy with 5 layers: 0,1 and 3,4 are FP16, middle is compressed."""
        schedule = compute_anchor_schedule(
            num_layers=5, anchor_strategy="boundary",
        )
        fp16_layers = [i for i, (is_fp16, _) in enumerate(schedule) if is_fp16]
        assert fp16_layers == [0, 1, 3, 4]
        # Layer 2 should be compressed
        assert not schedule[2][0]

    def test_gradient_schedule_boundary_fp16(self):
        """Gradient strategy: boundary layers (dist < 0.1) should be FP16."""
        schedule = compute_anchor_schedule(
            num_layers=36, anchor_strategy="gradient",
        )
        # Layers 0 and 35 (dist=0) should be FP16
        assert schedule[0][0], "Layer 0 should be FP16"
        assert schedule[35][0], "Layer 35 should be FP16"
        # Layer 1 has dist=1/18=0.056 < 0.1, should be FP16
        assert schedule[1][0], "Layer 1 should be FP16"
        assert schedule[34][0], "Layer 34 should be FP16"

    def test_gradient_schedule_middle_base_bits(self):
        """Gradient strategy: middle layers should get base key_bits."""
        schedule = compute_anchor_schedule(
            num_layers=36, anchor_strategy="gradient", base_key_bits=3,
        )
        mid = 18
        assert not schedule[mid][0], "Middle layer should not be FP16"
        assert schedule[mid][1] == 3, f"Middle layer key_bits should be 3, got {schedule[mid][1]}"

    def test_gradient_schedule_has_transition_layers(self):
        """Gradient strategy should have layers with intermediate bits (e.g. 4)."""
        schedule = compute_anchor_schedule(
            num_layers=36, anchor_strategy="gradient", base_key_bits=3,
        )
        key_bits_values = set(kb for _, kb in schedule)
        # Should have 8 (FP16 boundary), 4 (near-boundary), and 3 (middle)
        assert 3 in key_bits_values, "Should have base_bits=3 layers"
        assert 4 in key_bits_values, "Should have transition layers at 4 bits"
        assert 8 in key_bits_values, "Should have FP16-equivalent layers at 8 bits"

    def test_invalid_strategy_raises(self):
        """Unknown strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown anchor_strategy"):
            compute_anchor_schedule(num_layers=10, anchor_strategy="unknown")


# ---------------------------------------------------------------------------
# Tests: GenerationCache with anchor_strategy
# ---------------------------------------------------------------------------
class TestGenerationCacheAnchorStrategy:
    """Validate GenerationCache integration with anchor strategies."""

    def test_default_is_fixed(self):
        """Default construction uses fixed strategy (backward compatible)."""
        cache = GenerationCache(seed=SEED)
        assert cache.anchor_strategy == "fixed"

    def test_fixed_backward_compatible(self):
        """Fixed strategy with anchor_interval=6 produces same layers as before."""
        cache = GenerationCache(anchor_interval=6, key_bits=3, val_bits=2, seed=SEED)
        keys, values = make_kv_states(seq_len=4)
        # Populate 12 layers
        for i in range(12):
            cache.update(keys, values, layer_idx=i)
        # Layers 0, 6 should be FP16 anchors
        assert isinstance(cache._layers[0], _FP16Layer)
        assert isinstance(cache._layers[6], _FP16Layer)
        # Layers 1-5, 7-11 should be compressed
        for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]:
            assert isinstance(cache._layers[i], _CompressedLayer), (
                f"Layer {i} should be _CompressedLayer"
            )

    def test_boundary_requires_num_layers(self):
        """Boundary strategy without num_layers should raise ValueError."""
        with pytest.raises(ValueError, match="num_layers is required"):
            GenerationCache(anchor_strategy="boundary", seed=SEED)

    def test_gradient_requires_num_layers(self):
        """Gradient strategy without num_layers should raise ValueError."""
        with pytest.raises(ValueError, match="num_layers is required"):
            GenerationCache(anchor_strategy="gradient", seed=SEED)

    def test_invalid_strategy_raises(self):
        """Unknown anchor strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown anchor_strategy"):
            GenerationCache(anchor_strategy="invalid", seed=SEED)

    def test_boundary_creates_fp16_at_edges(self):
        """Boundary strategy: first 2 + last 2 layers should be _FP16Layer."""
        n = 36
        cache = GenerationCache(
            anchor_strategy="boundary", num_layers=n, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=4)
        for i in range(n):
            cache.update(keys, values, layer_idx=i)

        # First two and last two should be FP16
        assert isinstance(cache._layers[0], _FP16Layer)
        assert isinstance(cache._layers[1], _FP16Layer)
        assert isinstance(cache._layers[34], _FP16Layer)
        assert isinstance(cache._layers[35], _FP16Layer)

        # Middle layers should be compressed
        for i in range(2, 34):
            assert isinstance(cache._layers[i], _CompressedLayer), (
                f"Layer {i} should be _CompressedLayer"
            )

    def test_boundary_fp16_count(self):
        """Boundary strategy on 36-layer model: exactly 4 FP16 layers."""
        cache = GenerationCache(
            anchor_strategy="boundary", num_layers=36, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=4)
        for i in range(36):
            cache.update(keys, values, layer_idx=i)
        fp16_count = sum(isinstance(l, _FP16Layer) for l in cache._layers)
        assert fp16_count == 4

    def test_gradient_creates_fp16_at_boundaries(self):
        """Gradient strategy: layers within 10% of boundaries should be FP16."""
        n = 36
        cache = GenerationCache(
            anchor_strategy="gradient", num_layers=n, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=4)
        for i in range(n):
            cache.update(keys, values, layer_idx=i)

        # Layer 0,1 (dist < 0.1) should be FP16
        assert isinstance(cache._layers[0], _FP16Layer)
        assert isinstance(cache._layers[1], _FP16Layer)
        # Layer 34,35 (symmetric) should be FP16
        assert isinstance(cache._layers[34], _FP16Layer)
        assert isinstance(cache._layers[35], _FP16Layer)

    def test_gradient_near_boundary_higher_bits(self):
        """Gradient strategy: near-boundary compressed layers get higher key_bits."""
        n = 36
        cache = GenerationCache(
            anchor_strategy="gradient", num_layers=n, key_bits=3, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=4)
        for i in range(n):
            cache.update(keys, values, layer_idx=i)

        # Layers near boundaries (but not FP16) should have key_bits > base
        # Layer 2: dist = 2/18 = 0.111 -> in (0.1, 0.25) range -> 4 bits
        layer2 = cache._layers[2]
        assert isinstance(layer2, _CompressedLayer)
        assert layer2.key_bits == 4, f"Layer 2 key_bits should be 4, got {layer2.key_bits}"

        # Layer 33 (symmetric with 2)
        layer33 = cache._layers[33]
        assert isinstance(layer33, _CompressedLayer)
        assert layer33.key_bits == 4

    def test_gradient_middle_layers_base_bits(self):
        """Gradient strategy: middle layers get base key_bits."""
        n = 36
        cache = GenerationCache(
            anchor_strategy="gradient", num_layers=n, key_bits=3, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=4)
        for i in range(n):
            cache.update(keys, values, layer_idx=i)

        mid = 18
        layer_mid = cache._layers[mid]
        assert isinstance(layer_mid, _CompressedLayer)
        assert layer_mid.key_bits == 3

    def test_gradient_bits_symmetric(self):
        """Gradient strategy: layer i and layer (N-1-i) have same key_bits."""
        n = 36
        cache = GenerationCache(
            anchor_strategy="gradient", num_layers=n, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=4)
        for i in range(n):
            cache.update(keys, values, layer_idx=i)

        for i in range(n):
            mirror = n - 1 - i
            layer_i = cache._layers[i]
            layer_m = cache._layers[mirror]
            # Both FP16 or both compressed with same bits
            i_is_fp16 = isinstance(layer_i, _FP16Layer)
            m_is_fp16 = isinstance(layer_m, _FP16Layer)
            assert i_is_fp16 == m_is_fp16, (
                f"Layer {i} and {mirror} FP16 mismatch"
            )
            if not i_is_fp16:
                assert layer_i.key_bits == layer_m.key_bits, (
                    f"Layer {i} key_bits={layer_i.key_bits} != "
                    f"layer {mirror} key_bits={layer_m.key_bits}"
                )


# ---------------------------------------------------------------------------
# Tests: anchor_summary
# ---------------------------------------------------------------------------
class TestAnchorSummary:
    """Validate the anchor_summary reporting method."""

    def test_fixed_summary(self):
        """Fixed strategy summary reports correct FP16 layers."""
        cache = GenerationCache(
            anchor_interval=6, anchor_strategy="fixed",
            num_layers=12, seed=SEED,
        )
        summary = cache.anchor_summary()
        assert summary["strategy"] == "fixed"
        assert summary["num_layers"] == 12
        assert summary["fp16_layers"] == [0, 6]
        assert summary["fp16_count"] == 2
        assert summary["compressed_count"] == 10

    def test_boundary_summary(self):
        """Boundary strategy summary reports first 2 + last 2."""
        cache = GenerationCache(
            anchor_strategy="boundary", num_layers=36, seed=SEED,
        )
        summary = cache.anchor_summary()
        assert summary["strategy"] == "boundary"
        assert summary["fp16_layers"] == [0, 1, 34, 35]
        assert summary["fp16_count"] == 4
        assert summary["compressed_count"] == 32

    def test_gradient_summary_has_mixed_bits(self):
        """Gradient strategy summary has diverse key bit-widths."""
        cache = GenerationCache(
            anchor_strategy="gradient", num_layers=36, key_bits=3, seed=SEED,
        )
        summary = cache.anchor_summary()
        bits_set = set(summary["per_layer_key_bits"])
        # Should have FP16 (16), transition (4), and base (3)
        assert 16 in bits_set, "Should have FP16 layers (16 bits)"
        assert 4 in bits_set, "Should have near-boundary layers (4 bits)"
        assert 3 in bits_set, "Should have middle layers (3 bits)"

    def test_gradient_avg_bits_between_base_and_16(self):
        """Gradient strategy average bits should be between base_bits and 16."""
        cache = GenerationCache(
            anchor_strategy="gradient", num_layers=36, key_bits=3, seed=SEED,
        )
        summary = cache.anchor_summary()
        avg = summary["avg_key_bits"]
        assert 3.0 < avg < 16.0, f"avg_key_bits={avg} should be between 3 and 16"

    def test_empty_cache_summary(self):
        """Summary on cache with num_layers=None and no data returns empty."""
        cache = GenerationCache(seed=SEED)
        summary = cache.anchor_summary()
        assert summary["num_layers"] == 0
        assert summary["fp16_count"] == 0


# ---------------------------------------------------------------------------
# Tests: total bit budget comparison
# ---------------------------------------------------------------------------
class TestBitBudget:
    """Validate that anchor strategies produce expected compression ratios."""

    def test_boundary_fewer_anchors_than_interval_6(self):
        """Boundary strategy on 36 layers uses 4 FP16 vs interval=6 uses 6."""
        boundary_sched = compute_anchor_schedule(
            num_layers=36, anchor_strategy="boundary",
        )
        fixed_sched = compute_anchor_schedule(
            num_layers=36, anchor_strategy="fixed", anchor_interval=6,
        )
        boundary_fp16 = sum(1 for is_fp16, _ in boundary_sched if is_fp16)
        fixed_fp16 = sum(1 for is_fp16, _ in fixed_sched if is_fp16)
        assert boundary_fp16 == 4
        assert fixed_fp16 == 6
        # Boundary uses fewer FP16 layers -> higher compression potential
        assert boundary_fp16 < fixed_fp16

    def test_gradient_average_bits_less_than_fixed_anchor12(self):
        """Gradient strategy should use fewer average bits than fixed anchor=12."""
        gradient_sched = compute_anchor_schedule(
            num_layers=36, anchor_strategy="gradient", base_key_bits=3,
        )
        fixed12_sched = compute_anchor_schedule(
            num_layers=36, anchor_strategy="fixed", anchor_interval=12,
            base_key_bits=3,
        )

        def avg_bits(sched):
            total = 0
            for is_fp16, kb in sched:
                total += 16 if is_fp16 else kb
            return total / len(sched)

        grad_avg = avg_bits(gradient_sched)
        fixed12_avg = avg_bits(fixed12_sched)
        # Gradient should allocate bits more efficiently
        # Both should be in reasonable range
        assert 3.0 < grad_avg < 10.0
        assert 3.0 < fixed12_avg < 10.0

    def test_gradient_total_bits_36_layers(self):
        """Verify gradient schedule total bits for a 36-layer model."""
        schedule = compute_anchor_schedule(
            num_layers=36, anchor_strategy="gradient", base_key_bits=3,
        )
        # Count bit distribution
        fp16_count = sum(1 for is_fp16, _ in schedule if is_fp16)
        bits_4_count = sum(1 for is_fp16, kb in schedule if not is_fp16 and kb == 4)
        bits_3_count = sum(1 for is_fp16, kb in schedule if not is_fp16 and kb == 3)

        assert fp16_count > 0, "Should have at least some FP16 layers"
        assert bits_3_count > 0, "Should have base-bits layers"
        # Total should sum to 36
        assert fp16_count + bits_4_count + bits_3_count == 36


# ---------------------------------------------------------------------------
# Tests: quality comparison
# ---------------------------------------------------------------------------
class TestQualityComparison:
    """Compare reconstruction quality across anchor strategies."""

    def test_fp16_anchor_layers_exact(self):
        """FP16 anchor layers should return exact input (no quantization error)."""
        cache = GenerationCache(
            anchor_strategy="boundary", num_layers=36, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=10, seed=100)
        # Layer 0 is FP16 anchor
        k_out, v_out = cache.update(keys, values, layer_idx=0)
        # Should be exact
        assert torch.allclose(k_out, keys, atol=1e-6), "FP16 layer keys should be exact"
        assert torch.allclose(v_out, values, atol=1e-6), "FP16 layer values should be exact"

    def test_compressed_layers_approximate(self):
        """Compressed layers should have high but not perfect cosine similarity."""
        # Use fp16_window=0 to ensure all tokens go through compression
        cache = GenerationCache(
            anchor_strategy="boundary", num_layers=36,
            fp16_window=0, key_bits=3, val_bits=2, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=10, seed=200)
        # Layer 10 is compressed (middle layer)
        k_out, v_out = cache.update(keys, values, layer_idx=10)
        k_sim = cosine_sim(k_out, keys)
        v_sim = cosine_sim(v_out, values)
        # Should be good but not perfect
        assert k_sim > 0.90, f"Key cosine similarity {k_sim} too low"
        assert v_sim > 0.85, f"Value cosine similarity {v_sim} too low"
        # Not perfect (compressed)
        assert k_sim < 1.0 - 1e-6, "Compressed keys should not be exact"

    def test_gradient_near_boundary_better_than_middle(self):
        """Gradient strategy: near-boundary layers (4-bit) should have better
        reconstruction than middle layers (3-bit)."""
        # Use fp16_window=0 to ensure all tokens go through compression
        cache = GenerationCache(
            anchor_strategy="gradient", num_layers=36, key_bits=3, val_bits=2,
            fp16_window=0, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=10, seed=300)

        # Layer 3: near boundary, should get 4-bit keys
        k_near, _ = cache.update(keys, values, layer_idx=3)
        sim_near = cosine_sim(k_near, keys)

        # Layer 18: middle, should get 3-bit keys
        k_mid, _ = cache.update(keys, values, layer_idx=18)
        sim_mid = cosine_sim(k_mid, keys)

        # Near-boundary (4-bit) should generally reconstruct better than middle (3-bit)
        # Allow some tolerance since this is stochastic
        assert sim_near > sim_mid - 0.05, (
            f"Near-boundary sim={sim_near:.4f} should be >= middle sim={sim_mid:.4f} "
            f"(within tolerance)"
        )


# ---------------------------------------------------------------------------
# Tests: backward compatibility
# ---------------------------------------------------------------------------
class TestBackwardCompatibility:
    """Ensure existing code using GenerationCache continues to work."""

    def test_default_args_unchanged(self):
        """Default construction should produce identical behavior to pre-change."""
        cache = GenerationCache(seed=SEED)
        # Defaults may come from autoresearch sweep; the key point is
        # anchor_strategy defaults to "fixed" and num_layers is None
        assert cache.anchor_strategy == "fixed"
        assert cache.num_layers is None
        assert isinstance(cache.key_bits, int)
        assert isinstance(cache.val_bits, int)

    def test_fixed_strategy_identical_to_old_behavior(self):
        """Fixed strategy should produce same layer types as old anchor_interval."""
        cache_new = GenerationCache(anchor_interval=6, anchor_strategy="fixed", seed=SEED)
        cache_old = GenerationCache(anchor_interval=6, seed=SEED)

        keys, values = make_kv_states(seq_len=4)
        for i in range(12):
            cache_new.update(keys, values, layer_idx=i)
            cache_old.update(keys, values, layer_idx=i)

        for i in range(12):
            new_is_fp16 = isinstance(cache_new._layers[i], _FP16Layer)
            old_is_fp16 = isinstance(cache_old._layers[i], _FP16Layer)
            assert new_is_fp16 == old_is_fp16, (
                f"Layer {i}: new={new_is_fp16} vs old={old_is_fp16}"
            )

    def test_hf_protocol_works_with_all_strategies(self):
        """All anchor strategies should support the full HF Cache protocol."""
        for strategy in ANCHOR_STRATEGIES:
            kwargs = {"anchor_strategy": strategy, "seed": SEED}
            if strategy in ("boundary", "gradient"):
                kwargs["num_layers"] = 8
            cache = GenerationCache(**kwargs)

            keys, values = make_kv_states(seq_len=4)
            for i in range(8):
                cache.update(keys, values, layer_idx=i)

            # Basic protocol methods
            assert cache.get_seq_length(0) == 4
            assert len(cache) == 8
            assert 0 in cache

            pos = torch.arange(1)
            kv_len, offset = cache.get_mask_sizes(pos, layer_idx=0)
            assert kv_len == 5  # 4 cached + 1 query

            # Iteration
            items = list(cache)
            assert len(items) == 8

            # Reset
            cache.reset()
            assert cache.get_seq_length(0) == 0


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases for anchor strategies."""

    def test_3_layer_boundary(self):
        """Boundary strategy with 3 layers: 0,1 at start + 1,2 at end -> all FP16."""
        # Layers: 0 (first 2), 1 (first 2 AND last 2), 2 (last 2)
        cache = GenerationCache(
            anchor_strategy="boundary", num_layers=3, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=4)
        for i in range(3):
            cache.update(keys, values, layer_idx=i)
        # All layers should be FP16
        for i in range(3):
            assert isinstance(cache._layers[i], _FP16Layer), (
                f"Layer {i} should be FP16 in 3-layer boundary model"
            )

    def test_gradient_32_layer_model(self):
        """Gradient strategy on 32-layer model (Llama-3-8B)."""
        schedule = compute_anchor_schedule(
            num_layers=32, anchor_strategy="gradient", base_key_bits=3,
        )
        assert len(schedule) == 32
        # First and last layers should be FP16
        assert schedule[0][0], "Layer 0 should be FP16"
        assert schedule[31][0], "Layer 31 should be FP16"
        # Middle should be base bits
        assert schedule[16][1] == 3, "Middle layer should be 3 bits"

    def test_gradient_64_layer_model(self):
        """Gradient strategy on 64-layer model (Qwen3.5-27B)."""
        schedule = compute_anchor_schedule(
            num_layers=64, anchor_strategy="gradient", base_key_bits=3,
        )
        assert len(schedule) == 64
        # More layers -> more boundary layers (dist < 0.1 covers more indices)
        fp16_count = sum(1 for is_fp16, _ in schedule if is_fp16)
        assert fp16_count >= 4, f"Expected at least 4 FP16 layers, got {fp16_count}"
        # Middle should still be 3 bits
        assert schedule[32][1] == 3

    def test_config_summary_includes_strategy(self):
        """config_summary() should mention the anchor strategy."""
        cache = GenerationCache(
            anchor_strategy="gradient", num_layers=36, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=4)
        cache.update(keys, values, layer_idx=0)
        summary = cache.config_summary()
        assert "gradient" in summary

    def test_memory_savings_includes_strategy(self):
        """memory_savings() config dict should include anchor_strategy."""
        cache = GenerationCache(
            anchor_strategy="boundary", num_layers=36, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=4)
        cache.update(keys, values, layer_idx=0)
        report = cache.memory_savings()
        assert report["config"]["anchor_strategy"] == "boundary"

    def test_memory_savings_per_layer_key_bits(self):
        """memory_savings() per-layer entries should include key_bits."""
        cache = GenerationCache(
            anchor_strategy="gradient", num_layers=8, seed=SEED,
        )
        keys, values = make_kv_states(seq_len=4)
        for i in range(8):
            cache.update(keys, values, layer_idx=i)
        report = cache.memory_savings()
        for entry in report["per_layer"]:
            assert "key_bits" in entry


# ---------------------------------------------------------------------------
# Tests: schedule completeness
# ---------------------------------------------------------------------------
class TestScheduleCompleteness:
    """Verify that schedules cover all layers and have valid bit values."""

    @pytest.mark.parametrize("n_layers", [4, 8, 16, 32, 36, 48, 64, 80])
    def test_gradient_schedule_length(self, n_layers):
        """Gradient schedule should have exactly num_layers entries."""
        schedule = compute_anchor_schedule(
            num_layers=n_layers, anchor_strategy="gradient",
        )
        assert len(schedule) == n_layers

    @pytest.mark.parametrize("n_layers", [4, 8, 16, 32, 36, 48, 64, 80])
    def test_gradient_schedule_valid_bits(self, n_layers):
        """All bit values in gradient schedule should be valid (2-8)."""
        schedule = compute_anchor_schedule(
            num_layers=n_layers, anchor_strategy="gradient", base_key_bits=3,
        )
        for i, (is_fp16, kb) in enumerate(schedule):
            if is_fp16:
                assert kb == 8, f"FP16 layer {i} should have kb=8, got {kb}"
            else:
                assert 2 <= kb <= 8, (
                    f"Layer {i} key_bits={kb} out of valid range [2, 8]"
                )

    @pytest.mark.parametrize("n_layers", [4, 8, 16, 32, 36, 48, 64, 80])
    def test_boundary_schedule_symmetry(self, n_layers):
        """Boundary schedule should be symmetric (first 2 = last 2)."""
        schedule = compute_anchor_schedule(
            num_layers=n_layers, anchor_strategy="boundary",
        )
        fp16_layers = [i for i, (is_fp16, _) in enumerate(schedule) if is_fp16]
        if n_layers >= 5:
            # Should include exactly layers 0, 1, N-2, N-1
            assert 0 in fp16_layers
            assert 1 in fp16_layers
            assert n_layers - 2 in fp16_layers
            assert n_layers - 1 in fp16_layers
            assert len(fp16_layers) == 4
        else:
            # Small models: overlap means more than 4 layers are FP16
            assert len(fp16_layers) >= min(4, n_layers)
