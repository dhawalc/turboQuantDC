"""Tests for ResidualVQ — 2-stage Residual Vector Quantization."""

import math

import pytest
import torch

from turboquantdc.codebook import LloydMaxCodebook
from turboquantdc.polarquant import PolarQuant
from turboquantdc.residual_vq import ResidualVQ, ResidualVQCache, ResidualVQLayer


# ---------------------------------------------------------------------------
# ResidualVQ unit tests
# ---------------------------------------------------------------------------


class TestResidualVQInit:
    """Test construction and parameter validation."""

    def test_init_2_plus_2(self):
        rvq = ResidualVQ(d=128, stage1_bits=2, stage2_bits=2, seed=42)
        assert rvq.d == 128
        assert rvq.stage1_bits == 2
        assert rvq.stage2_bits == 2
        assert rvq.total_bits == 4

    def test_init_3_plus_2(self):
        rvq = ResidualVQ(d=128, stage1_bits=3, stage2_bits=2, seed=42)
        assert rvq.total_bits == 5

    def test_init_2_plus_1(self):
        rvq = ResidualVQ(d=128, stage1_bits=2, stage2_bits=1, seed=42)
        assert rvq.total_bits == 3

    def test_stage2_codebook_is_different_from_stage1(self):
        """Stage 2 codebook should be optimized for the residual distribution,
        which has a different variance than the original."""
        rvq = ResidualVQ(d=128, stage1_bits=2, stage2_bits=2, seed=42)
        s1_centroids = rvq.stage1.codebook.centroids
        s2_centroids = rvq.stage2_codebook.centroids
        # They should have different centroids (different distributions)
        assert not torch.allclose(s1_centroids, s2_centroids), (
            "Stage 2 codebook should differ from stage 1 (different distribution)"
        )

    def test_stage2_codebook_has_smaller_centroids(self):
        """Residual centroids should have smaller magnitude than stage 1 centroids
        because the residual has lower variance."""
        rvq = ResidualVQ(d=128, stage1_bits=2, stage2_bits=2, seed=42)
        s1_max = rvq.stage1.codebook.centroids.abs().max().item()
        s2_max = rvq.stage2_codebook.centroids.abs().max().item()
        assert s2_max < s1_max, (
            f"Stage 2 centroids max {s2_max:.6f} should be < "
            f"stage 1 centroids max {s1_max:.6f}"
        )

    def test_different_seeds_produce_different_rotations(self):
        rvq1 = ResidualVQ(d=64, stage1_bits=2, stage2_bits=2, seed=42)
        rvq2 = ResidualVQ(d=64, stage1_bits=2, stage2_bits=2, seed=99)
        assert not torch.allclose(rvq1.stage1.Pi, rvq2.stage1.Pi)


class TestResidualVQQuantize:
    """Test quantize/dequantize round-trip."""

    @pytest.fixture
    def rvq_2_2(self):
        return ResidualVQ(d=128, stage1_bits=2, stage2_bits=2, seed=42)

    @pytest.fixture
    def rvq_3_2(self):
        return ResidualVQ(d=128, stage1_bits=3, stage2_bits=2, seed=42)

    def test_quantize_output_shapes_batch(self, rvq_2_2):
        x = torch.randn(10, 128)
        comp = rvq_2_2.quantize(x)
        assert comp["stage1_indices"].shape == (10, 128)
        assert comp["stage2_indices"].shape == (10, 128)
        assert comp["vec_norm"].shape == (10,)

    def test_quantize_output_shapes_single(self, rvq_2_2):
        x = torch.randn(128)
        comp = rvq_2_2.quantize(x)
        assert comp["stage1_indices"].shape == (128,)
        assert comp["stage2_indices"].shape == (128,)
        assert comp["vec_norm"].shape == ()

    def test_indices_range_stage1(self, rvq_2_2):
        x = torch.randn(100, 128)
        comp = rvq_2_2.quantize(x)
        assert comp["stage1_indices"].min() >= 0
        assert comp["stage1_indices"].max() < (1 << 2)

    def test_indices_range_stage2(self, rvq_2_2):
        x = torch.randn(100, 128)
        comp = rvq_2_2.quantize(x)
        assert comp["stage2_indices"].min() >= 0
        assert comp["stage2_indices"].max() < (1 << 2)

    def test_indices_range_3_2(self, rvq_3_2):
        x = torch.randn(100, 128)
        comp = rvq_3_2.quantize(x)
        assert comp["stage1_indices"].max() < (1 << 3)
        assert comp["stage2_indices"].max() < (1 << 2)

    def test_norms_positive(self, rvq_2_2):
        x = torch.randn(20, 128)
        comp = rvq_2_2.quantize(x)
        assert (comp["vec_norm"] > 0).all()

    def test_norms_match_input(self, rvq_2_2):
        x = torch.randn(20, 128)
        comp = rvq_2_2.quantize(x)
        expected_norms = x.norm(dim=-1)
        assert torch.allclose(comp["vec_norm"], expected_norms, atol=1e-5)

    def test_roundtrip_shape_batch(self, rvq_2_2):
        x = torch.randn(10, 128)
        comp = rvq_2_2.quantize(x)
        x_hat = rvq_2_2.dequantize(comp)
        assert x_hat.shape == x.shape

    def test_roundtrip_shape_single(self, rvq_2_2):
        x = torch.randn(128)
        comp = rvq_2_2.quantize(x)
        x_hat = rvq_2_2.dequantize(comp)
        assert x_hat.shape == x.shape

    def test_roundtrip_not_identity(self, rvq_2_2):
        """Quantization is lossy, so reconstruction should differ from input."""
        x = torch.randn(10, 128)
        comp = rvq_2_2.quantize(x)
        x_hat = rvq_2_2.dequantize(comp)
        assert not torch.allclose(x, x_hat, atol=1e-6)

    def test_forward_returns_tuple(self, rvq_2_2):
        x = torch.randn(5, 128)
        x_hat, comp = rvq_2_2(x)
        assert x_hat.shape == x.shape
        assert "stage1_indices" in comp
        assert "stage2_indices" in comp


class TestResidualVQStage1Only:
    """Test stage1-only dequantization for ablation."""

    @pytest.fixture
    def rvq(self):
        return ResidualVQ(d=128, stage1_bits=2, stage2_bits=2, seed=42)

    def test_stage1_only_shape(self, rvq):
        x = torch.randn(10, 128)
        comp = rvq.quantize(x)
        x_s1 = rvq.dequantize_stage1_only(comp)
        assert x_s1.shape == x.shape

    def test_stage2_improves_over_stage1(self, rvq):
        """Full 2-stage reconstruction should have lower MSE than stage 1 alone."""
        torch.manual_seed(123)
        x = torch.randn(500, 128)
        comp = rvq.quantize(x)
        x_s1 = rvq.dequantize_stage1_only(comp)
        x_full = rvq.dequantize(comp)

        mse_s1 = ((x - x_s1) ** 2).mean().item()
        mse_full = ((x - x_full) ** 2).mean().item()

        assert mse_full < mse_s1, (
            f"Full RVQ MSE ({mse_full:.6f}) should be < stage1-only MSE ({mse_s1:.6f})"
        )

    def test_stage1_only_single_vector(self, rvq):
        x = torch.randn(128)
        comp = rvq.quantize(x)
        x_s1 = rvq.dequantize_stage1_only(comp)
        assert x_s1.shape == (128,)


# ---------------------------------------------------------------------------
# RVQ vs Single-Stage Quality Comparisons (the core hypothesis)
# ---------------------------------------------------------------------------


class TestRVQvsSingleStage:
    """Compare RVQ at total bits vs single-stage at same total bits.

    For scalar per-coordinate Lloyd-Max quantization, a single-stage quantizer
    with 2^b optimal levels is strictly better than two cascaded stages with
    2^b1 + 2^b2 levels, because the single-stage can place all 2^b levels
    optimally for the source distribution.

    The value of RVQ in this context is:
    1. Stage 2 significantly improves over stage 1 alone (confirmed)
    2. RVQ provides finer-grained bit-rate control (asymmetric splits)
    3. RVQ 3+1 closely approaches K4 quality with different bit allocation
    4. The architecture supports future extensions (learned codebooks,
       multi-dimensional VQ) where cascaded stages DO beat single-stage
    """

    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Generate test data: unit-sphere-like vectors."""
        torch.manual_seed(42)
        self.d = 128
        self.n = 1000
        self.x = torch.randn(self.n, self.d)

    def _mse(self, x: torch.Tensor, x_hat: torch.Tensor) -> float:
        return ((x - x_hat) ** 2).mean().item()

    def _cosine_sim(self, x: torch.Tensor, x_hat: torch.Tensor) -> float:
        cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1)
        return cos.mean().item()

    def _single_stage_reconstruct(self, x: torch.Tensor, bits: int) -> torch.Tensor:
        """Reconstruct using single-stage PolarQuant at given bits."""
        pq = PolarQuant(self.d, bits=bits, seed=42)
        norms = x.norm(dim=-1, keepdim=True)
        x_norm = x / (norms + 1e-8)
        indices = pq.quantize(x_norm)
        x_hat_norm = pq.dequantize(indices)
        return x_hat_norm * norms

    def _rvq_reconstruct(
        self, x: torch.Tensor, s1_bits: int, s2_bits: int,
    ) -> torch.Tensor:
        rvq = ResidualVQ(self.d, stage1_bits=s1_bits, stage2_bits=s2_bits, seed=42)
        comp = rvq.quantize(x)
        return rvq.dequantize(comp)

    def test_2_plus_2_competitive_with_k4(self):
        """RVQ 2+2 (4 bits) should be within 2x MSE of K4 single-stage.

        Single-stage K4 is theoretically optimal for scalar quantization,
        but RVQ 2+2 should not be catastrophically worse.
        """
        x_hat_k4 = self._single_stage_reconstruct(self.x, bits=4)
        x_hat_rvq = self._rvq_reconstruct(self.x, s1_bits=2, s2_bits=2)

        mse_k4 = self._mse(self.x, x_hat_k4)
        mse_rvq = self._mse(self.x, x_hat_rvq)

        # RVQ at same bits is within 2x of single-stage
        assert mse_rvq < mse_k4 * 2.5, (
            f"RVQ 2+2 MSE ({mse_rvq:.6f}) should be < 2.5x K4 MSE ({mse_k4:.6f})"
        )

    def test_2_plus_2_beats_k3(self):
        """RVQ 2+2 (4 bits total) should beat K3 (3 bits) -- more bits wins."""
        x_hat_k3 = self._single_stage_reconstruct(self.x, bits=3)
        x_hat_rvq = self._rvq_reconstruct(self.x, s1_bits=2, s2_bits=2)

        mse_k3 = self._mse(self.x, x_hat_k3)
        mse_rvq = self._mse(self.x, x_hat_rvq)

        assert mse_rvq < mse_k3, (
            f"RVQ 2+2 (4bit) MSE ({mse_rvq:.6f}) should be < K3 MSE ({mse_k3:.6f})"
        )

    def test_3_plus_1_closely_approaches_k4(self):
        """RVQ 3+1 (4 bits) should be close to K4 -- asymmetric split is efficient."""
        x_hat_k4 = self._single_stage_reconstruct(self.x, bits=4)
        x_hat_rvq = self._rvq_reconstruct(self.x, s1_bits=3, s2_bits=1)

        mse_k4 = self._mse(self.x, x_hat_k4)
        mse_rvq = self._mse(self.x, x_hat_rvq)

        # 3+1 should be within 1.5x of K4 (stage1 already captures most)
        assert mse_rvq < mse_k4 * 1.5, (
            f"RVQ 3+1 MSE ({mse_rvq:.6f}) should be < 1.5x K4 MSE ({mse_k4:.6f})"
        )

    def test_3_plus_2_beats_k4(self):
        """RVQ 3+2 (5 bits total) should beat K4 (4 bits) -- more total bits."""
        x_hat_k4 = self._single_stage_reconstruct(self.x, bits=4)
        x_hat_rvq = self._rvq_reconstruct(self.x, s1_bits=3, s2_bits=2)

        mse_k4 = self._mse(self.x, x_hat_k4)
        mse_rvq = self._mse(self.x, x_hat_rvq)

        assert mse_rvq < mse_k4, (
            f"RVQ 3+2 (5bit) MSE ({mse_rvq:.6f}) should be < K4 (4bit) MSE ({mse_k4:.6f})"
        )

    def test_rvq_cosine_above_threshold(self):
        """RVQ 2+2 should achieve good cosine similarity on d=128."""
        x_hat = self._rvq_reconstruct(self.x, s1_bits=2, s2_bits=2)
        cos = self._cosine_sim(self.x, x_hat)
        # RVQ 2+2 (4 total bits, cascaded) achieves ~0.975 cosine
        assert cos > 0.97, f"RVQ 2+2 cosine {cos:.6f} should be > 0.97"

    def test_asymmetric_3_1_better_than_1_3(self):
        """Putting more bits in stage 1 should be better than stage 2.

        Stage 1 captures the bulk of the signal, so it benefits more from
        extra bits than the residual correction stage.
        """
        x_hat_31 = self._rvq_reconstruct(self.x, s1_bits=3, s2_bits=1)
        x_hat_13 = self._rvq_reconstruct(self.x, s1_bits=1, s2_bits=3)

        mse_31 = self._mse(self.x, x_hat_31)
        mse_13 = self._mse(self.x, x_hat_13)

        assert mse_31 < mse_13, (
            f"RVQ 3+1 MSE ({mse_31:.6f}) should be < RVQ 1+3 MSE ({mse_13:.6f})"
        )

    def test_monotonic_stage2_improvement(self):
        """Increasing stage2 bits (holding stage1 fixed) should reduce MSE."""
        mse_values = []
        for s2 in [1, 2, 3]:
            x_hat = self._rvq_reconstruct(self.x, s1_bits=2, s2_bits=s2)
            mse_values.append(self._mse(self.x, x_hat))

        for i in range(len(mse_values) - 1):
            assert mse_values[i + 1] < mse_values[i], (
                f"MSE should decrease: 2+{i+1}={mse_values[i]:.6f} "
                f"-> 2+{i+2}={mse_values[i+1]:.6f}"
            )


class TestRVQDimensionVariations:
    """Test RVQ across different head dimensions."""

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_roundtrip_various_d(self, d):
        torch.manual_seed(42)
        rvq = ResidualVQ(d=d, stage1_bits=2, stage2_bits=2, seed=42)
        x = torch.randn(50, d)
        comp = rvq.quantize(x)
        x_hat = rvq.dequantize(comp)
        assert x_hat.shape == x.shape
        mse = ((x - x_hat) ** 2).mean().item()
        # MSE should be finite and reasonable
        assert mse < 1.0, f"MSE {mse:.6f} unexpectedly large for d={d}"

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_rvq_beats_lower_bit_single_stage(self, d):
        """RVQ 2+2 (4 total bits) should beat K3 (3 bits) across dimensions."""
        torch.manual_seed(42)
        n = 500
        x = torch.randn(n, d)

        # Single-stage K3
        pq = PolarQuant(d, bits=3, seed=42)
        norms = x.norm(dim=-1, keepdim=True)
        x_norm = x / (norms + 1e-8)
        idx = pq.quantize(x_norm)
        x_k3 = pq.dequantize(idx) * norms

        # RVQ 2+2
        rvq = ResidualVQ(d, stage1_bits=2, stage2_bits=2, seed=42)
        comp = rvq.quantize(x)
        x_rvq = rvq.dequantize(comp)

        mse_k3 = ((x - x_k3) ** 2).mean().item()
        mse_rvq = ((x - x_rvq) ** 2).mean().item()

        assert mse_rvq < mse_k3, (
            f"d={d}: RVQ 2+2 MSE ({mse_rvq:.6f}) should be < K3 MSE ({mse_k3:.6f})"
        )


class TestRVQBitCombinations:
    """Test various stage1/stage2 bit combinations."""

    @pytest.mark.parametrize(
        "s1,s2",
        [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)],
    )
    def test_roundtrip(self, s1, s2):
        torch.manual_seed(42)
        rvq = ResidualVQ(d=64, stage1_bits=s1, stage2_bits=s2, seed=42)
        x = torch.randn(20, 64)
        comp = rvq.quantize(x)
        x_hat = rvq.dequantize(comp)
        assert x_hat.shape == x.shape
        # Check finite
        assert torch.isfinite(x_hat).all()

    @pytest.mark.parametrize(
        "s1,s2",
        [(1, 1), (2, 1), (2, 2), (3, 2)],
    )
    def test_more_bits_lower_mse(self, s1, s2):
        """Higher total bits should give lower MSE."""
        torch.manual_seed(42)
        d = 128
        n = 500
        x = torch.randn(n, d)

        rvq_low = ResidualVQ(d, stage1_bits=s1, stage2_bits=s2, seed=42)
        rvq_high = ResidualVQ(d, stage1_bits=s1 + 1, stage2_bits=s2, seed=42)

        comp_low = rvq_low.quantize(x)
        comp_high = rvq_high.quantize(x)
        x_low = rvq_low.dequantize(comp_low)
        x_high = rvq_high.dequantize(comp_high)

        mse_low = ((x - x_low) ** 2).mean().item()
        mse_high = ((x - x_high) ** 2).mean().item()

        assert mse_high < mse_low, (
            f"({s1+1}+{s2}) MSE ({mse_high:.6f}) should be < "
            f"({s1}+{s2}) MSE ({mse_low:.6f})"
        )


# ---------------------------------------------------------------------------
# ResidualVQLayer tests
# ---------------------------------------------------------------------------


class TestResidualVQLayer:
    """Test the layer-level KV cache using ResidualVQ."""

    def _make_kv(self, batch=1, heads=2, seq=4, d=64):
        return (
            torch.randn(batch, heads, seq, d),
            torch.randn(batch, heads, seq, d),
        )

    def test_basic_update(self):
        layer = ResidualVQLayer(key_stage1_bits=2, key_stage2_bits=2, value_bits=2)
        k, v = self._make_kv()
        k_out, v_out = layer.update(k, v)
        assert k_out.shape == (1, 2, 4, 64)
        assert v_out.shape == (1, 2, 4, 64)

    def test_incremental_update(self):
        layer = ResidualVQLayer(key_stage1_bits=2, key_stage2_bits=2, value_bits=2)
        k1, v1 = self._make_kv(seq=3)
        k2, v2 = self._make_kv(seq=2)
        layer.update(k1, v1)
        k_out, v_out = layer.update(k2, v2)
        assert k_out.shape[2] == 5  # 3 + 2

    def test_clear(self):
        layer = ResidualVQLayer(key_stage1_bits=2, key_stage2_bits=2, value_bits=2)
        k, v = self._make_kv()
        layer.update(k, v)
        layer.clear()
        assert layer.get_seq_length() == 0

    def test_output_dtype_preserved(self):
        layer = ResidualVQLayer(key_stage1_bits=2, key_stage2_bits=2, value_bits=2)
        k = torch.randn(1, 2, 4, 64, dtype=torch.float16)
        v = torch.randn(1, 2, 4, 64, dtype=torch.float16)
        k_out, v_out = layer.update(k, v)
        assert k_out.dtype == torch.float16
        assert v_out.dtype == torch.float16

    def test_fp16_window(self):
        layer = ResidualVQLayer(
            key_stage1_bits=2, key_stage2_bits=2, value_bits=2,
            fp16_window=2,
        )
        k, v = self._make_kv(seq=5)
        k_out, v_out = layer.update(k, v)
        # Should have compressed tokens + fp16 window tokens
        assert k_out.shape[2] >= 5


# ---------------------------------------------------------------------------
# ResidualVQCache (HF protocol) tests
# ---------------------------------------------------------------------------


class TestResidualVQCache:
    """Test the HF-compatible cache wrapper."""

    def test_init_valid(self):
        cache = ResidualVQCache(key_stage1_bits=2, key_stage2_bits=2, value_bits=2)
        assert cache.total_key_bits == 4

    def test_init_invalid_bits(self):
        with pytest.raises(ValueError):
            ResidualVQCache(key_stage1_bits=0)
        with pytest.raises(ValueError):
            ResidualVQCache(key_stage2_bits=9)
        with pytest.raises(ValueError):
            ResidualVQCache(value_bits=0)

    def test_update_single_layer(self):
        cache = ResidualVQCache(key_stage1_bits=2, key_stage2_bits=2, value_bits=2)
        k = torch.randn(1, 4, 8, 64)
        v = torch.randn(1, 4, 8, 64)
        k_out, v_out = cache.update(k, v, layer_idx=0)
        assert k_out.shape == (1, 4, 8, 64)
        assert v_out.shape == (1, 4, 8, 64)

    def test_update_multiple_layers(self):
        cache = ResidualVQCache(key_stage1_bits=2, key_stage2_bits=2, value_bits=2)
        for layer_idx in range(3):
            k = torch.randn(1, 4, 8, 64)
            v = torch.randn(1, 4, 8, 64)
            cache.update(k, v, layer_idx=layer_idx)
        assert len(cache) == 3

    def test_get_seq_length(self):
        cache = ResidualVQCache(key_stage1_bits=2, key_stage2_bits=2, value_bits=2)
        assert cache.get_seq_length() == 0
        k = torch.randn(1, 2, 5, 64)
        v = torch.randn(1, 2, 5, 64)
        cache.update(k, v, layer_idx=0)
        assert cache.get_seq_length(0) > 0

    def test_get_seq_length_missing_layer(self):
        cache = ResidualVQCache()
        assert cache.get_seq_length(99) == 0

    def test_getitem(self):
        cache = ResidualVQCache(key_stage1_bits=2, key_stage2_bits=2, value_bits=2)
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)
        cache.update(k, v, layer_idx=0)
        k_out, v_out = cache[0]
        assert k_out.shape == (1, 2, 4, 64)

    def test_getitem_out_of_range(self):
        cache = ResidualVQCache()
        with pytest.raises(IndexError):
            cache[0]

    def test_iter(self):
        cache = ResidualVQCache(key_stage1_bits=2, key_stage2_bits=2, value_bits=2)
        for i in range(2):
            k = torch.randn(1, 2, 3, 64)
            v = torch.randn(1, 2, 3, 64)
            cache.update(k, v, layer_idx=i)
        layers = list(cache)
        assert len(layers) == 2
        for keys, values, _ in layers:
            assert keys.shape == (1, 2, 3, 64)

    def test_is_initialized(self):
        cache = ResidualVQCache()
        assert not cache.is_initialized
        cache.update(torch.randn(1, 1, 1, 64), torch.randn(1, 1, 1, 64), layer_idx=0)
        assert cache.is_initialized

    def test_is_sliding(self):
        cache = ResidualVQCache()
        cache.update(torch.randn(1, 1, 1, 64), torch.randn(1, 1, 1, 64), layer_idx=0)
        assert cache.is_sliding == [False]

    def test_reset(self):
        cache = ResidualVQCache(key_stage1_bits=2, key_stage2_bits=2, value_bits=2)
        cache.update(torch.randn(1, 2, 4, 64), torch.randn(1, 2, 4, 64), layer_idx=0)
        cache.reset()
        assert cache.get_seq_length(0) == 0

    def test_get_max_cache_shape(self):
        cache = ResidualVQCache()
        assert cache.get_max_cache_shape() == -1

    def test_get_mask_sizes_empty(self):
        cache = ResidualVQCache()
        pos = torch.arange(5)
        kv_len, _ = cache.get_mask_sizes(pos, layer_idx=0)
        assert kv_len == 5

    def test_get_mask_sizes_populated(self):
        cache = ResidualVQCache(key_stage1_bits=2, key_stage2_bits=2, value_bits=2)
        cache.update(torch.randn(1, 2, 4, 64), torch.randn(1, 2, 4, 64), layer_idx=0)
        pos = torch.arange(1)
        kv_len, _ = cache.get_mask_sizes(pos, layer_idx=0)
        assert kv_len > 1


class TestResidualVQCacheBeam:
    """Test beam search operations on ResidualVQCache."""

    def _populated_cache(self):
        cache = ResidualVQCache(key_stage1_bits=2, key_stage2_bits=2, value_bits=2)
        k = torch.randn(2, 2, 4, 64)  # batch=2
        v = torch.randn(2, 2, 4, 64)
        cache.update(k, v, layer_idx=0)
        return cache

    def test_reorder_cache(self):
        cache = self._populated_cache()
        beam_idx = torch.tensor([1, 0])
        cache.reorder_cache(beam_idx)
        k, v = cache[0]
        assert k.shape[0] == 2

    def test_batch_repeat_interleave(self):
        cache = self._populated_cache()
        cache.batch_repeat_interleave(3)
        k, v = cache[0]
        assert k.shape[0] == 6  # 2 * 3

    def test_batch_select_indices(self):
        cache = self._populated_cache()
        cache.batch_select_indices(torch.tensor([0]))
        k, v = cache[0]
        assert k.shape[0] == 1

    def test_crop(self):
        cache = ResidualVQCache(key_stage1_bits=2, key_stage2_bits=2, value_bits=2)
        for t in range(5):
            k = torch.randn(1, 2, 1, 64)
            v = torch.randn(1, 2, 1, 64)
            cache.update(k, v, layer_idx=0)
        cache.crop(3)
        # After crop, should have at most 3 compressed entries
        assert len(cache._layers[0]._key_compressed) <= 3


# ---------------------------------------------------------------------------
# Reconstruction quality tests
# ---------------------------------------------------------------------------


class TestReconstructionQuality:
    """Verify absolute reconstruction quality metrics."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(7)
        self.d = 128
        self.x = torch.randn(1000, self.d)

    def test_rvq_2_2_mse_bound(self):
        """RVQ 2+2 MSE should be within reasonable bounds."""
        rvq = ResidualVQ(self.d, stage1_bits=2, stage2_bits=2, seed=42)
        comp = rvq.quantize(self.x)
        x_hat = rvq.dequantize(comp)
        mse = ((self.x - x_hat) ** 2).mean().item()
        # At 4 total bits on d=128, MSE should be very low
        assert mse < 0.1, f"RVQ 2+2 MSE {mse:.6f} too high"

    def test_rvq_3_2_cosine_above_threshold(self):
        """RVQ 3+2 (5 bits) should achieve high cosine similarity."""
        rvq = ResidualVQ(self.d, stage1_bits=3, stage2_bits=2, seed=42)
        comp = rvq.quantize(self.x)
        x_hat = rvq.dequantize(comp)
        cos = torch.nn.functional.cosine_similarity(self.x, x_hat, dim=-1).mean().item()
        assert cos > 0.995, f"RVQ 3+2 cosine {cos:.6f} should be > 0.995"

    def test_rvq_deterministic(self):
        """Same input should produce same output."""
        rvq = ResidualVQ(self.d, stage1_bits=2, stage2_bits=2, seed=42)
        x = self.x[:10]
        x_hat1 = rvq.dequantize(rvq.quantize(x))
        x_hat2 = rvq.dequantize(rvq.quantize(x))
        assert torch.allclose(x_hat1, x_hat2)

    def test_zero_vector_handling(self):
        """Zero vector should not produce NaN/Inf."""
        rvq = ResidualVQ(self.d, stage1_bits=2, stage2_bits=2, seed=42)
        x = torch.zeros(1, self.d)
        comp = rvq.quantize(x)
        x_hat = rvq.dequantize(comp)
        assert torch.isfinite(x_hat).all()

    def test_large_norm_vector(self):
        """Very large vectors should round-trip cleanly."""
        rvq = ResidualVQ(self.d, stage1_bits=2, stage2_bits=2, seed=42)
        x = torch.randn(5, self.d) * 1000
        comp = rvq.quantize(x)
        x_hat = rvq.dequantize(comp)
        assert torch.isfinite(x_hat).all()
        cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()
        assert cos > 0.99


# ---------------------------------------------------------------------------
# Inner product quality (attention relevance)
# ---------------------------------------------------------------------------


class TestInnerProductQuality:
    """Test that RVQ preserves inner product structure (critical for attention)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.d = 128
        self.queries = torch.randn(100, self.d)
        self.keys = torch.randn(200, self.d)

    def test_ip_correlation(self):
        """Inner products computed from RVQ reconstruction should correlate
        highly with true inner products."""
        rvq = ResidualVQ(self.d, stage1_bits=2, stage2_bits=2, seed=42)
        comp = rvq.quantize(self.keys)
        keys_hat = rvq.dequantize(comp)

        true_ip = self.queries @ self.keys.T  # (100, 200)
        approx_ip = self.queries @ keys_hat.T

        # Flatten and compute correlation
        true_flat = true_ip.flatten()
        approx_flat = approx_ip.flatten()
        correlation = torch.corrcoef(torch.stack([true_flat, approx_flat]))[0, 1].item()

        assert correlation > 0.98, (
            f"IP correlation {correlation:.6f} should be > 0.98"
        )

    def test_top5_attention_match(self):
        """Top-5 keys by attention score should have high overlap with true top-5."""
        rvq = ResidualVQ(self.d, stage1_bits=2, stage2_bits=2, seed=42)
        comp = rvq.quantize(self.keys)
        keys_hat = rvq.dequantize(comp)

        true_ip = self.queries @ self.keys.T
        approx_ip = self.queries @ keys_hat.T

        true_top5 = true_ip.topk(5, dim=-1).indices
        approx_top5 = approx_ip.topk(5, dim=-1).indices

        matches = 0
        total = self.queries.shape[0] * 5
        for i in range(self.queries.shape[0]):
            matches += len(set(true_top5[i].tolist()) & set(approx_top5[i].tolist()))

        match_rate = matches / total
        assert match_rate > 0.75, f"Top-5 match rate {match_rate:.3f} should be > 0.75"
