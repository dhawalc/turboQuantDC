"""Tests for AdaptiveGenerationCache."""

import pytest
import torch

from turboquantdc.adaptive_generation_cache import (
    AdaptiveGenerationCache,
    _AdaptiveLayer,
    _FP16AnchorLayer,
)


class TestAdaptiveLayer:
    """Test the per-layer adaptive compressed storage."""

    def test_fp16_only_short_sequence(self):
        """Sequence shorter than buffer stays at FP16."""
        layer = _AdaptiveLayer(
            hot_window=4, fp16_buffer_size=16,
            tier_bits=[16, 4, 3, 1],
            tier_thresholds=[0.05, 0.20, 0.80],
        )
        keys = torch.randn(1, 2, 10, 128)
        vals = torch.randn(1, 2, 10, 128)
        k, v = layer.update(keys, vals)
        assert k.shape == (1, 2, 10, 128)
        assert v.shape == (1, 2, 10, 128)
        assert layer.get_seq_length() == 10
        # All tokens in FP16 buffer
        assert layer.effective_bits() == 16.0

    def test_flush_triggers(self):
        """Tokens exceeding buffer size get compressed."""
        layer = _AdaptiveLayer(
            hot_window=4, fp16_buffer_size=8,
            tier_bits=[16, 4, 3, 1],
            tier_thresholds=[0.05, 0.20, 0.80],
        )
        keys = torch.randn(1, 2, 20, 128)
        vals = torch.randn(1, 2, 20, 128)
        k, v = layer.update(keys, vals)
        assert k.shape == (1, 2, 20, 128)
        assert layer.get_seq_length() == 20
        # Some tokens should be compressed
        assert layer.effective_bits() < 16.0
        assert layer._compressed_len > 0

    def test_incremental_updates(self):
        """Multiple small updates accumulate correctly."""
        layer = _AdaptiveLayer(
            hot_window=4, fp16_buffer_size=8,
            tier_bits=[16, 4, 3, 1],
            tier_thresholds=[0.05, 0.20, 0.80],
        )
        for i in range(15):
            keys = torch.randn(1, 2, 1, 128)
            vals = torch.randn(1, 2, 1, 128)
            k, v = layer.update(keys, vals)
        assert layer.get_seq_length() == 15
        assert k.shape == (1, 2, 15, 128)

    def test_clear_resets_state(self):
        """Clear resets all state."""
        layer = _AdaptiveLayer(
            hot_window=4, fp16_buffer_size=8,
            tier_bits=[16, 4, 3, 1],
            tier_thresholds=[0.05, 0.20, 0.80],
        )
        keys = torch.randn(1, 2, 20, 128)
        vals = torch.randn(1, 2, 20, 128)
        layer.update(keys, vals)
        layer.clear()
        assert layer.get_seq_length() == 0
        assert layer._compressed_len == 0

    def test_importance_updates(self):
        """Importance scorer receives attention weights."""
        layer = _AdaptiveLayer(
            hot_window=4, fp16_buffer_size=8,
            tier_bits=[16, 4, 3, 1],
            tier_thresholds=[0.05, 0.20, 0.80],
        )
        # Create attention weights
        attn = torch.rand(1, 2, 10, 10)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        layer.update_importance(attn)
        assert layer._scorer.scores is not None
        assert layer._scorer.seq_len == 10


class TestAdaptiveGenerationCache:
    """Test the unified cache system."""

    def test_basic_construction(self):
        """Cache can be constructed with default params."""
        cache = AdaptiveGenerationCache(
            num_layers=8,
            tier_thresholds=[0.05, 0.20, 0.80],
            tier_bits=[16, 4, 3, 1],
        )
        assert len(cache) == 0
        assert not cache.is_initialized

    def test_boundary_layers(self):
        """Boundary layers are FP16, others are adaptive."""
        cache = AdaptiveGenerationCache(
            num_layers=8,
            boundary_layers=2,
        )
        # Trigger layer creation
        d = 64
        for layer_idx in range(8):
            keys = torch.randn(1, 2, 5, d)
            vals = torch.randn(1, 2, 5, d)
            cache.update(keys, vals, layer_idx)

        # Check layer types
        assert isinstance(cache._layers[0], _FP16AnchorLayer)  # first 2
        assert isinstance(cache._layers[1], _FP16AnchorLayer)
        assert isinstance(cache._layers[2], _AdaptiveLayer)    # middle
        assert isinstance(cache._layers[5], _AdaptiveLayer)
        assert isinstance(cache._layers[6], _FP16AnchorLayer)  # last 2
        assert isinstance(cache._layers[7], _FP16AnchorLayer)

    def test_hf_protocol(self):
        """Cache duck-types the HF Cache protocol."""
        cache = AdaptiveGenerationCache(
            num_layers=4,
            boundary_layers=1,
        )
        d = 64
        keys = torch.randn(1, 2, 10, d)
        vals = torch.randn(1, 2, 10, d)

        # update returns (keys, values)
        k, v = cache.update(keys, vals, layer_idx=0)
        assert k.shape == (1, 2, 10, d)
        assert v.shape == (1, 2, 10, d)

        # get_seq_length
        assert cache.get_seq_length(0) == 10
        assert cache.get_seq_length(1) == 0  # layer 1 not populated yet

        # get_mask_sizes
        kv_len, offset = cache.get_mask_sizes(5, layer_idx=0)
        assert kv_len == 15  # 10 cached + 5 new query
        assert offset == 0

        # __len__
        assert len(cache) == 1

        # __contains__
        assert 0 in cache
        assert 1 not in cache

        # is_initialized, is_sliding
        assert cache.is_initialized
        assert not any(cache.is_sliding)

    def test_generation_roundtrip(self):
        """Full generation-like loop: update all layers per step."""
        num_layers = 4
        cache = AdaptiveGenerationCache(
            num_layers=num_layers,
            boundary_layers=1,
            hot_window=4,
            fp16_buffer_size=8,
        )
        d = 64
        batch, heads = 1, 2

        # Simulate 20 decode steps
        for step in range(20):
            for layer_idx in range(num_layers):
                keys = torch.randn(batch, heads, 1, d)
                vals = torch.randn(batch, heads, 1, d)
                k, v = cache.update(keys, vals, layer_idx)
                assert k.shape[2] == step + 1

        assert cache.get_seq_length(0) == 20
        assert cache.effective_bits() < 16.0

    def test_tier_differentiation(self):
        """Different importance scores produce different tier assignments."""
        cache = AdaptiveGenerationCache(
            num_layers=1,
            boundary_layers=0,
            hot_window=4,
            fp16_buffer_size=16,
            tier_thresholds=[0.05, 0.20, 0.80],
            tier_bits=[16, 4, 3, 1],
        )
        d = 128
        batch, heads = 1, 2
        seq = 100

        # Power-law attention: first 5 tokens get 80% of attention
        attn = torch.zeros(batch, heads, seq, seq)
        for q in range(seq):
            weights = torch.zeros(seq)
            weights[:5] = 0.8 / 5
            weights[max(0, q-2):q+1] = 0.2 / min(3, q+1)
            weights[:q+1] = weights[:q+1] / weights[:q+1].sum()
            attn[0, 0, q, :q+1] = weights[:q+1]
            attn[0, 1, q, :q+1] = weights[:q+1]

        cache.update_importance(attn, 0)

        keys = torch.randn(batch, heads, seq, d)
        vals = torch.randn(batch, heads, seq, d)
        cache.update(keys, vals, 0)

        layer = cache._layers[0]
        assert layer._token_tiers is not None
        # Should have multiple tiers represented
        unique_tiers = layer._token_tiers.unique()
        assert len(unique_tiers) >= 2, f"Only {len(unique_tiers)} tiers used"

    def test_crop(self):
        """Crop reduces sequence length."""
        cache = AdaptiveGenerationCache(
            num_layers=2,
            boundary_layers=0,
            hot_window=4,
            fp16_buffer_size=8,
        )
        d = 64
        keys = torch.randn(1, 2, 20, d)
        vals = torch.randn(1, 2, 20, d)
        cache.update(keys, vals, 0)
        cache.update(keys, vals, 1)

        cache.crop(10)
        assert cache.get_seq_length(0) == 10
        assert cache.get_seq_length(1) == 10

    def test_reset(self):
        """Reset clears all layers."""
        cache = AdaptiveGenerationCache(
            num_layers=2,
            boundary_layers=0,
        )
        d = 64
        keys = torch.randn(1, 2, 10, d)
        vals = torch.randn(1, 2, 10, d)
        cache.update(keys, vals, 0)
        cache.reset()
        assert cache.get_seq_length(0) == 0
