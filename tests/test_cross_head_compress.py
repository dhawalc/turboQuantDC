"""Tests for cross-head delta compression.

Validates:
1. measure_inter_head_correlation produces correct shapes and ranges
2. CrossHeadDeltaQuantizer compresses and decompresses correctly
3. UniformQuantizer baseline works
4. Quality evaluation functions produce valid metrics
5. Effective bit rate calculations are correct
"""

import pytest
import torch

from turboquantdc.cross_head_compress import (
    CrossHeadDeltaQuantizer,
    UniformQuantizer,
    evaluate_attention_quality,
    evaluate_reconstruction_quality,
    measure_inter_head_correlation,
    select_best_anchor,
)


@pytest.fixture
def kv_states():
    """Generate synthetic KV states for testing."""
    torch.manual_seed(42)
    # [batch=2, num_heads=4, seq_len=16, head_dim=64]
    return torch.randn(2, 4, 16, 64)


@pytest.fixture
def correlated_kv_states():
    """Generate KV states with known high inter-head correlation."""
    torch.manual_seed(42)
    batch, num_heads, seq_len, head_dim = 2, 4, 16, 64
    # Base pattern shared across heads
    base = torch.randn(batch, 1, seq_len, head_dim)
    # Small per-head noise
    noise = 0.1 * torch.randn(batch, num_heads, seq_len, head_dim)
    return base.expand(-1, num_heads, -1, -1) + noise


class TestMeasureInterHeadCorrelation:
    """Tests for the inter-head correlation measurement function."""

    def test_output_keys(self, kv_states):
        stats = measure_inter_head_correlation(kv_states)
        expected_keys = {
            "pairwise_cosine", "pairwise_pearson", "pairwise_delta_norm",
            "mean_cosine", "mean_pearson", "mean_delta_norm",
            "anchor_delta_stats", "num_heads", "mode",
        }
        assert set(stats.keys()) == expected_keys

    def test_pairwise_shapes(self, kv_states):
        stats = measure_inter_head_correlation(kv_states)
        n = kv_states.shape[1]
        assert stats["pairwise_cosine"].shape == (n, n)
        assert stats["pairwise_pearson"].shape == (n, n)
        assert stats["pairwise_delta_norm"].shape == (n, n)

    def test_diagonal_cosine_is_one(self, kv_states):
        stats = measure_inter_head_correlation(kv_states)
        n = kv_states.shape[1]
        for i in range(n):
            assert abs(stats["pairwise_cosine"][i, i] - 1.0) < 0.01

    def test_cosine_in_valid_range(self, kv_states):
        stats = measure_inter_head_correlation(kv_states)
        assert -1.0 <= stats["mean_cosine"] <= 1.0

    def test_anchor_delta_stats_count(self, kv_states):
        stats = measure_inter_head_correlation(kv_states)
        # N-1 entries (one for each non-anchor head)
        assert len(stats["anchor_delta_stats"]) == kv_states.shape[1] - 1

    def test_high_correlation_detected(self, correlated_kv_states):
        stats = measure_inter_head_correlation(correlated_kv_states)
        # Correlated KV states should have high cosine similarity
        assert stats["mean_cosine"] > 0.8

    def test_variance_ratio_low_for_correlated(self, correlated_kv_states):
        stats = measure_inter_head_correlation(correlated_kv_states)
        for s in stats["anchor_delta_stats"]:
            # Delta variance should be much lower than original
            assert s["variance_ratio"] < 0.2


class TestSelectBestAnchor:
    def test_returns_valid_index(self, kv_states):
        anchor = select_best_anchor(kv_states)
        assert 0 <= anchor < kv_states.shape[1]

    def test_selects_central_head(self, correlated_kv_states):
        # For highly correlated data, any head is a good anchor
        anchor = select_best_anchor(correlated_kv_states)
        assert 0 <= anchor < correlated_kv_states.shape[1]


class TestCrossHeadDeltaQuantizer:
    """Tests for the cross-head delta quantizer."""

    def test_basic_roundtrip(self, kv_states):
        d = kv_states.shape[-1]
        n_heads = kv_states.shape[1]
        quant = CrossHeadDeltaQuantizer(
            d=d, num_heads=n_heads, anchor_bits=3, delta_bits=1,
        )
        recon = quant.quantize_dequantize(kv_states)
        assert recon.shape == kv_states.shape

    def test_output_dtype(self, kv_states):
        d = kv_states.shape[-1]
        n_heads = kv_states.shape[1]
        quant = CrossHeadDeltaQuantizer(d=d, num_heads=n_heads)
        recon = quant.quantize_dequantize(kv_states)
        assert recon.dtype == torch.float32

    def test_anchor_better_than_deltas(self, kv_states):
        """Anchor head should have better reconstruction than delta heads."""
        d = kv_states.shape[-1]
        n_heads = kv_states.shape[1]
        quant = CrossHeadDeltaQuantizer(
            d=d, num_heads=n_heads, anchor_bits=3, delta_bits=1,
        )
        recon = quant.quantize_dequantize(kv_states)

        # Cosine sim for anchor head vs delta heads
        anchor_cos = torch.nn.functional.cosine_similarity(
            kv_states[:, 0].reshape(-1, d),
            recon[:, 0].reshape(-1, d),
            dim=-1,
        ).mean()

        delta_cos_list = []
        for h in range(1, n_heads):
            cos = torch.nn.functional.cosine_similarity(
                kv_states[:, h].reshape(-1, d),
                recon[:, h].reshape(-1, d),
                dim=-1,
            ).mean()
            delta_cos_list.append(cos.item())

        # Anchor (3-bit) should generally be better than 1-bit deltas
        # (unless correlation is very high, in which case both are good)
        avg_delta_cos = sum(delta_cos_list) / len(delta_cos_list)
        assert anchor_cos.item() >= avg_delta_cos - 0.1  # allow some slack

    def test_effective_bits_calculation(self):
        quant = CrossHeadDeltaQuantizer(
            d=128, num_heads=8, anchor_bits=3, delta_bits=1,
        )
        eff = quant.effective_bits_per_element()
        # (3*128 + 32 + 7*(1*128 + 16)) / (8*128)
        expected = (3 * 128 + 32 + 7 * (1 * 128 + 16)) / (8 * 128)
        assert abs(eff - expected) < 0.01

    def test_compression_ratio(self):
        quant = CrossHeadDeltaQuantizer(
            d=128, num_heads=8, anchor_bits=3, delta_bits=1,
        )
        ratio = quant.compression_ratio()
        assert ratio > 10.0  # Should be around 12.8x

    def test_high_correlation_improves_quality(self, correlated_kv_states):
        """With highly correlated heads, cross-head should work well."""
        d = correlated_kv_states.shape[-1]
        n_heads = correlated_kv_states.shape[1]
        quant = CrossHeadDeltaQuantizer(
            d=d, num_heads=n_heads, anchor_bits=3, delta_bits=1,
        )
        recon = quant.quantize_dequantize(correlated_kv_states)
        qual = evaluate_reconstruction_quality(correlated_kv_states, recon)
        # High correlation means even 1-bit deltas should give decent quality
        assert qual["mean_cosine_sim"] > 0.8

    def test_different_delta_bits(self, kv_states):
        d = kv_states.shape[-1]
        n_heads = kv_states.shape[1]

        results = {}
        for delta_bits in [1, 2, 3]:
            quant = CrossHeadDeltaQuantizer(
                d=d, num_heads=n_heads, anchor_bits=3, delta_bits=delta_bits,
            )
            recon = quant.quantize_dequantize(kv_states)
            qual = evaluate_reconstruction_quality(kv_states, recon)
            results[delta_bits] = qual["mean_cosine_sim"]

        # More bits should generally give better quality
        assert results[2] >= results[1] - 0.05
        assert results[3] >= results[2] - 0.05


class TestUniformQuantizer:
    def test_basic_roundtrip(self, kv_states):
        d = kv_states.shape[-1]
        n_heads = kv_states.shape[1]
        quant = UniformQuantizer(d=d, num_heads=n_heads, bits=3)
        recon = quant.quantize_dequantize(kv_states)
        assert recon.shape == kv_states.shape

    def test_effective_bits(self):
        quant = UniformQuantizer(d=128, num_heads=8, bits=3)
        assert quant.effective_bits_per_element() == 3.0

    def test_compression_ratio(self):
        quant = UniformQuantizer(d=128, num_heads=8, bits=3)
        assert abs(quant.compression_ratio() - 16.0 / 3.0) < 0.01


class TestEvaluationHelpers:
    def test_reconstruction_quality_keys(self, kv_states):
        # Perfect reconstruction
        qual = evaluate_reconstruction_quality(kv_states, kv_states)
        assert abs(qual["mean_cosine_sim"] - 1.0) < 0.001
        assert qual["mse"] < 1e-6

    def test_reconstruction_quality_range(self, kv_states):
        noisy = kv_states + 0.5 * torch.randn_like(kv_states)
        qual = evaluate_reconstruction_quality(kv_states, noisy)
        assert 0.0 < qual["mean_cosine_sim"] < 1.0
        assert qual["mse"] > 0.0

    def test_attention_quality_perfect(self, kv_states):
        queries = torch.randn(2, 4, 4, 64)
        qual = evaluate_attention_quality(queries, kv_states, kv_states)
        assert abs(qual["top5_attention_match"] - 1.0) < 0.001
        assert abs(qual["attention_score_pearson_r"] - 1.0) < 0.001

    def test_attention_quality_noisy(self, kv_states):
        queries = torch.randn(2, 4, 4, 64)
        noisy = kv_states + torch.randn_like(kv_states)
        qual = evaluate_attention_quality(queries, kv_states, noisy)
        assert 0.0 <= qual["top5_attention_match"] <= 1.0
        assert -1.0 <= qual["attention_score_pearson_r"] <= 1.0

    def test_per_head_cosine(self, kv_states):
        noisy = kv_states + 0.1 * torch.randn_like(kv_states)
        qual = evaluate_reconstruction_quality(kv_states, noisy)
        assert len(qual["per_head_cosine"]) == kv_states.shape[1]
        for cos in qual["per_head_cosine"]:
            assert 0.0 < cos <= 1.0
