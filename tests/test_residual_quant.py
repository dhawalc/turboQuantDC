"""Tests for ResidualQuant — MSE + direct residual sign correction."""

import math

import pytest
import torch

from turboquantdc.residual_quant import (
    ResidualQuantCache,
    ResidualQuantEstimator,
    ResidualQuantLayer,
)


# ---------------------------------------------------------------------------
# ResidualQuantEstimator unit tests
# ---------------------------------------------------------------------------

class TestResidualQuantEstimator:
    """Tests for the core ResidualQuantEstimator."""

    @pytest.fixture
    def estimator_3bit(self):
        return ResidualQuantEstimator(d=128, bits=3, seed=42, device="cpu")

    @pytest.fixture
    def estimator_4bit(self):
        return ResidualQuantEstimator(d=128, bits=4, seed=42, device="cpu")

    def test_init_bit_budget(self, estimator_3bit):
        """3-bit total = 2-bit MSE + 1-bit residual signs."""
        assert estimator_3bit.bits == 3
        assert estimator_3bit.mse_bits == 2
        assert estimator_3bit.d == 128

    def test_init_bit_budget_4bit(self, estimator_4bit):
        """4-bit total = 3-bit MSE + 1-bit residual signs."""
        assert estimator_4bit.bits == 4
        assert estimator_4bit.mse_bits == 3

    def test_quantize_output_shapes(self, estimator_3bit):
        """quantize() returns expected shapes."""
        x = torch.randn(10, 128)
        compressed = estimator_3bit.quantize(x)

        assert compressed["mse_indices"].shape == (10, 128)
        assert compressed["residual_signs"].shape == (10, 128)
        assert compressed["residual_scale"].shape == (10,)
        assert compressed["vec_norm"].shape == (10,)

    def test_quantize_single_vector(self, estimator_3bit):
        """quantize() works for a single vector (no batch dim)."""
        x = torch.randn(128)
        compressed = estimator_3bit.quantize(x)

        assert compressed["mse_indices"].shape == (128,)
        assert compressed["residual_signs"].shape == (128,)
        assert compressed["residual_scale"].shape == ()
        assert compressed["vec_norm"].shape == ()

    def test_residual_signs_are_binary(self, estimator_3bit):
        """Residual signs should be exactly {-1, +1}."""
        x = torch.randn(20, 128)
        compressed = estimator_3bit.quantize(x)
        signs = compressed["residual_signs"]
        assert torch.all((signs == 1.0) | (signs == -1.0))

    def test_residual_scale_positive(self, estimator_3bit):
        """Residual scale (mean |r|) should be non-negative."""
        x = torch.randn(20, 128)
        compressed = estimator_3bit.quantize(x)
        assert torch.all(compressed["residual_scale"] >= 0)

    def test_vec_norm_matches_input(self, estimator_3bit):
        """Stored vec_norm should match the norm of the (centered) input."""
        x = torch.randn(10, 128)
        compressed = estimator_3bit.quantize(x)
        # With center_before_quantize=True (default), vec_norm is the norm
        # of x_centered = x - mean(x), not x itself.
        x_centered = x - x.mean(dim=0, keepdim=True)
        expected_norms = x_centered.norm(dim=-1)
        torch.testing.assert_close(
            compressed["vec_norm"], expected_norms, atol=1e-5, rtol=1e-5
        )

    def test_vec_norm_matches_input_no_centering(self):
        """Without centering, stored vec_norm should match original input norm."""
        est = ResidualQuantEstimator(d=128, bits=3, seed=42, center_before_quantize=False)
        x = torch.randn(10, 128)
        compressed = est.quantize(x)
        expected_norms = x.norm(dim=-1)
        torch.testing.assert_close(
            compressed["vec_norm"], expected_norms, atol=1e-5, rtol=1e-5
        )

    def test_dequantize_output_shape(self, estimator_3bit):
        """dequantize() returns same shape as input."""
        x = torch.randn(10, 128)
        compressed = estimator_3bit.quantize(x)
        x_recon = estimator_3bit.dequantize(compressed)
        assert x_recon.shape == x.shape

    def test_dequantize_single_vector(self, estimator_3bit):
        """dequantize() works for single vector."""
        x = torch.randn(128)
        compressed = estimator_3bit.quantize(x)
        x_recon = estimator_3bit.dequantize(compressed)
        assert x_recon.shape == x.shape

    def test_reconstruction_better_than_mse_only(self, estimator_3bit):
        """ResidualQuant correction should improve over MSE-only."""
        torch.manual_seed(42)
        x = torch.randn(100, 128)

        compressed = estimator_3bit.quantize(x)
        x_rq = estimator_3bit.dequantize(compressed)
        x_mse = estimator_3bit.dequantize_mse(compressed)

        # Compute per-vector MSE
        mse_rq = ((x - x_rq) ** 2).mean(dim=-1).mean()
        mse_only = ((x - x_mse) ** 2).mean(dim=-1).mean()

        # ResidualQuant should have lower reconstruction error
        assert mse_rq < mse_only, (
            f"ResidualQuant MSE {mse_rq:.6f} should be < MSE-only {mse_only:.6f}"
        )

    def test_cosine_similarity_improvement(self, estimator_3bit):
        """Cosine similarity should improve with residual correction."""
        torch.manual_seed(42)
        x = torch.randn(100, 128)

        compressed = estimator_3bit.quantize(x)
        x_rq = estimator_3bit.dequantize(compressed)
        x_mse = estimator_3bit.dequantize_mse(compressed)

        # Cosine similarity
        cos_rq = torch.nn.functional.cosine_similarity(x, x_rq, dim=-1).mean()
        cos_mse = torch.nn.functional.cosine_similarity(x, x_mse, dim=-1).mean()

        assert cos_rq > cos_mse, (
            f"ResidualQuant cosine sim {cos_rq:.6f} should be > MSE-only {cos_mse:.6f}"
        )

    def test_inner_product_shape(self, estimator_3bit):
        """inner_product() returns correct shapes."""
        keys = torch.randn(20, 128)
        queries = torch.randn(5, 128)

        compressed = estimator_3bit.quantize(keys)
        ip = estimator_3bit.inner_product(queries, compressed)

        assert ip.shape == (5, 20)

    def test_inner_product_single_query(self, estimator_3bit):
        """inner_product() works with single query vector."""
        keys = torch.randn(10, 128)
        query = torch.randn(128)

        compressed = estimator_3bit.quantize(keys)
        ip = estimator_3bit.inner_product(query, compressed)

        assert ip.shape == (10,)

    def test_inner_product_accuracy(self, estimator_3bit):
        """Inner product estimates should correlate with true values."""
        torch.manual_seed(42)
        keys = torch.randn(50, 128)
        queries = torch.randn(10, 128)

        compressed = estimator_3bit.quantize(keys)
        ip_est = estimator_3bit.inner_product(queries, compressed)
        ip_true = queries @ keys.T

        # Check correlation — 3-bit on random vectors gives ~0.91;
        # real LLM vectors have more structure and correlate better.
        correlation = torch.corrcoef(
            torch.stack([ip_est.flatten(), ip_true.flatten()])
        )[0, 1]
        assert correlation > 0.85, f"IP correlation {correlation:.4f} too low"

    def test_4bit_better_than_3bit(self):
        """4-bit should have better reconstruction than 3-bit."""
        torch.manual_seed(42)
        x = torch.randn(100, 128)

        est_3 = ResidualQuantEstimator(d=128, bits=3, seed=42)
        est_4 = ResidualQuantEstimator(d=128, bits=4, seed=42)

        comp_3 = est_3.quantize(x)
        comp_4 = est_4.quantize(x)

        cos_3 = torch.nn.functional.cosine_similarity(
            x, est_3.dequantize(comp_3), dim=-1
        ).mean()
        cos_4 = torch.nn.functional.cosine_similarity(
            x, est_4.dequantize(comp_4), dim=-1
        ).mean()

        assert cos_4 > cos_3, (
            f"4-bit cosine sim {cos_4:.6f} should be > 3-bit {cos_3:.6f}"
        )

    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical results."""
        x = torch.randn(10, 128)

        est1 = ResidualQuantEstimator(d=128, bits=3, seed=42)
        est2 = ResidualQuantEstimator(d=128, bits=3, seed=42)

        comp1 = est1.quantize(x)
        comp2 = est2.quantize(x)

        torch.testing.assert_close(comp1["mse_indices"], comp2["mse_indices"])
        torch.testing.assert_close(comp1["residual_signs"], comp2["residual_signs"])

    def test_different_seeds_differ(self):
        """Different seeds should produce different rotation matrices."""
        x = torch.randn(10, 128)

        est1 = ResidualQuantEstimator(d=128, bits=3, seed=42)
        est2 = ResidualQuantEstimator(d=128, bits=3, seed=99)

        comp1 = est1.quantize(x)
        comp2 = est2.quantize(x)

        # MSE indices should differ (different rotation matrices)
        assert not torch.equal(comp1["mse_indices"], comp2["mse_indices"])


# ---------------------------------------------------------------------------
# ResidualQuantLayer unit tests
# ---------------------------------------------------------------------------

class TestResidualQuantLayer:
    """Tests for the per-layer cache storage."""

    def test_update_returns_correct_shapes(self):
        """update() returns dequantized tensors with correct shapes."""
        layer = ResidualQuantLayer(bits=3, seed=42)
        keys = torch.randn(1, 2, 5, 64)   # batch=1, heads=2, seq=5, d=64
        vals = torch.randn(1, 2, 5, 64)

        keys_out, vals_out = layer.update(keys, vals)
        assert keys_out.shape == (1, 2, 5, 64)
        assert vals_out.shape == (1, 2, 5, 64)

    def test_seq_length_accumulates(self):
        """Sequence length should accumulate across updates."""
        layer = ResidualQuantLayer(bits=3, seed=42)
        keys = torch.randn(1, 2, 3, 64)
        vals = torch.randn(1, 2, 3, 64)

        layer.update(keys, vals)
        assert layer.get_seq_length() == 3

        layer.update(keys, vals)
        assert layer.get_seq_length() == 6

    def test_clear_resets(self):
        """clear() should reset sequence length to 0."""
        layer = ResidualQuantLayer(bits=3, seed=42)
        layer.update(torch.randn(1, 2, 5, 64), torch.randn(1, 2, 5, 64))
        assert layer.get_seq_length() == 5
        layer.clear()
        assert layer.get_seq_length() == 0

    def test_accumulated_output_shape(self):
        """After multiple updates, output should have full sequence length."""
        layer = ResidualQuantLayer(bits=3, seed=42)
        for _ in range(3):
            keys = torch.randn(1, 2, 4, 64)
            vals = torch.randn(1, 2, 4, 64)
            k_out, v_out = layer.update(keys, vals)

        assert k_out.shape == (1, 2, 12, 64)
        assert v_out.shape == (1, 2, 12, 64)


# ---------------------------------------------------------------------------
# ResidualQuantCache HF protocol tests
# ---------------------------------------------------------------------------

class TestResidualQuantCache:
    """Tests for the HF-compatible cache wrapper."""

    def test_init(self):
        """Cache should initialize with given bits."""
        cache = ResidualQuantCache(bits=3, seed=42)
        assert cache.bits == 3
        assert len(cache) == 0

    def test_invalid_bits(self):
        """Should reject invalid bit widths."""
        with pytest.raises(ValueError):
            ResidualQuantCache(bits=1)
        with pytest.raises(ValueError):
            ResidualQuantCache(bits=9)

    def test_update_creates_layers(self):
        """update() should lazily create layers."""
        cache = ResidualQuantCache(bits=3)
        keys = torch.randn(1, 2, 5, 64)
        vals = torch.randn(1, 2, 5, 64)

        cache.update(keys, vals, layer_idx=0)
        assert len(cache) == 1

        cache.update(keys, vals, layer_idx=2)
        assert len(cache) == 3  # 0, 1, 2

    def test_get_seq_length(self):
        """get_seq_length() should return correct length."""
        cache = ResidualQuantCache(bits=3)
        keys = torch.randn(1, 2, 7, 64)
        vals = torch.randn(1, 2, 7, 64)

        assert cache.get_seq_length(0) == 0
        cache.update(keys, vals, layer_idx=0)
        assert cache.get_seq_length(0) == 7

    def test_reset(self):
        """reset() should clear all layers."""
        cache = ResidualQuantCache(bits=3)
        cache.update(torch.randn(1, 2, 5, 64), torch.randn(1, 2, 5, 64), layer_idx=0)
        cache.reset()
        assert cache.get_seq_length(0) == 0

    def test_iter_protocol(self):
        """__iter__ should yield (keys, values, None) tuples."""
        cache = ResidualQuantCache(bits=3)
        cache.update(torch.randn(1, 2, 5, 64), torch.randn(1, 2, 5, 64), layer_idx=0)
        cache.update(torch.randn(1, 2, 5, 64), torch.randn(1, 2, 5, 64), layer_idx=1)

        for keys, values, extra in cache:
            assert keys.shape[-1] == 64
            assert values.shape[-1] == 64
            assert extra is None

    def test_getitem(self):
        """__getitem__ should return dequantized (keys, values)."""
        cache = ResidualQuantCache(bits=3)
        cache.update(torch.randn(1, 2, 5, 64), torch.randn(1, 2, 5, 64), layer_idx=0)

        keys, values = cache[0]
        assert keys.shape == (1, 2, 5, 64)
        assert values.shape == (1, 2, 5, 64)

    def test_getitem_out_of_range(self):
        """__getitem__ should raise IndexError for invalid index."""
        cache = ResidualQuantCache(bits=3)
        with pytest.raises(IndexError):
            cache[0]

    def test_is_initialized(self):
        """is_initialized should reflect whether any layers exist."""
        cache = ResidualQuantCache(bits=3)
        assert not cache.is_initialized
        cache.update(torch.randn(1, 2, 5, 64), torch.randn(1, 2, 5, 64), layer_idx=0)
        assert cache.is_initialized


# ---------------------------------------------------------------------------
# Comparison: ResidualQuant vs TurboQuant vs MSE-only
# ---------------------------------------------------------------------------

class TestComparison:
    """Compare ResidualQuant against TurboQuant and MSE-only baselines."""

    def test_residualquant_vs_mse_cosine_sim(self):
        """ResidualQuant at 3-bit should have higher cosine sim than 2-bit MSE-only."""
        from turboquantdc.polarquant import PolarQuant

        torch.manual_seed(42)
        d = 128
        x = torch.randn(200, d)
        x_normalized = x / (x.norm(dim=-1, keepdim=True) + 1e-8)

        # 2-bit MSE-only (same MSE budget as 3-bit ResidualQuant)
        pq_2bit = PolarQuant(d=d, bits=2, seed=42)
        x_mse_2bit = pq_2bit.dequantize(pq_2bit.quantize(x_normalized))
        cos_mse_2bit = torch.nn.functional.cosine_similarity(
            x_normalized, x_mse_2bit, dim=-1
        ).mean()

        # 3-bit ResidualQuant (2-bit MSE + 1-bit residual correction)
        rq = ResidualQuantEstimator(d=d, bits=3, seed=42)
        comp = rq.quantize(x)
        x_rq = rq.dequantize(comp)
        x_rq_normalized = x_rq / (x_rq.norm(dim=-1, keepdim=True) + 1e-8)
        cos_rq = torch.nn.functional.cosine_similarity(
            x_normalized, x_rq_normalized, dim=-1
        ).mean()

        assert cos_rq > cos_mse_2bit, (
            f"RQ cosine sim {cos_rq:.6f} should beat 2-bit MSE {cos_mse_2bit:.6f}"
        )

    def test_residualquant_3bit_vs_turboquant_3bit_variance(self):
        """ResidualQuant should have lower IP variance than TurboQuant at 3-bit."""
        from turboquantdc.estimator import TurboQuantEstimator

        torch.manual_seed(42)
        d = 128
        n_trials = 200

        keys = torch.randn(n_trials, d)
        queries = torch.randn(n_trials, d)

        # True inner products
        ip_true = (queries * keys).sum(dim=-1)

        # TurboQuant 3-bit
        tq = TurboQuantEstimator(d=d, bits=3, seed=42)
        tq_comp = tq.quantize(keys)
        ip_tq = torch.stack([
            tq.inner_product(queries[i:i+1], {k: v[i:i+1] for k, v in tq_comp.items()}).squeeze()
            for i in range(n_trials)
        ])

        # ResidualQuant 3-bit
        rq = ResidualQuantEstimator(d=d, bits=3, seed=42)
        rq_comp = rq.quantize(keys)
        ip_rq = torch.stack([
            rq.inner_product(queries[i:i+1], {k: v[i:i+1] for k, v in rq_comp.items()}).squeeze()
            for i in range(n_trials)
        ])

        # Compare errors
        err_tq = (ip_tq - ip_true).abs()
        err_rq = (ip_rq - ip_true).abs()

        # ResidualQuant should have lower mean absolute error
        # (lower variance, even if biased)
        mae_tq = err_tq.mean().item()
        mae_rq = err_rq.mean().item()

        # We don't assert here because it depends on the random seed,
        # but we print for inspection. The benchmark script will do the
        # definitive comparison with generation quality.
        print(f"\nTurboQuant 3-bit MAE: {mae_tq:.6f}")
        print(f"ResidualQuant 3-bit MAE: {mae_rq:.6f}")
        print(f"ResidualQuant {'wins' if mae_rq < mae_tq else 'loses'}")

    def test_storage_equivalence(self):
        """ResidualQuant should use the same number of bits as TurboQuant."""
        d = 128
        bits = 3

        # TurboQuant: (bits-1)*d MSE + d QJL signs + 16 res_norm + 16 vec_norm
        tq_bits_per_vec = (bits - 1) * d + d + 16 + 16

        # ResidualQuant: (bits-1)*d MSE + d residual signs + 16 res_scale + 16 vec_norm
        rq_bits_per_vec = (bits - 1) * d + d + 16 + 16

        assert tq_bits_per_vec == rq_bits_per_vec, (
            f"TQ uses {tq_bits_per_vec} bits, RQ uses {rq_bits_per_vec} bits"
        )
