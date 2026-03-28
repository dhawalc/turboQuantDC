"""Tests for the outlier channel strategy (fractional bit rates).

OutlierTurboQuant splits channels into two groups after rotation to achieve
non-integer average bit rates. This test suite validates channel splitting,
compression ratios, inner product quality, unbiasedness, and comparison
against pure integer bit-width baselines.

Reference: TurboQuant paper, non-integer bit precision section.
"""

import pytest
import torch

from turboquantdc.outlier import OutlierTurboQuant
from turboquantdc.estimator import TurboQuantEstimator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DIM = 128
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_unit_vectors(n: int, d: int, seed: int = SEED) -> torch.Tensor:
    """Generate n random unit vectors of dimension d."""
    torch.manual_seed(seed)
    x = torch.randn(n, d)
    x = x / x.norm(dim=1, keepdim=True)
    return x


def element_wise_ips(
    oq: OutlierTurboQuant,
    queries: torch.Tensor,
    compressed: dict,
) -> torch.Tensor:
    """Compute element-wise inner products <q_i, k_i> via diagonal."""
    return torch.diagonal(oq.inner_product(queries, compressed))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChannelSplit:
    """Verify that channel splitting is computed correctly."""

    def test_outlier_2_5_bit(self):
        """2.5-bit mode: d=128 should give n_high=64, n_low=64."""
        oq = OutlierTurboQuant(d=128, target_bits=2.5, seed=SEED)
        assert oq.n_high == 64, f"Expected n_high=64, got {oq.n_high}"
        assert oq.n_low == 64, f"Expected n_low=64, got {oq.n_low}"
        assert oq.high_bits == 3
        assert oq.low_bits == 2

    def test_outlier_3_5_bit(self):
        """3.5-bit mode: d=128 should give n_high=64, n_low=64."""
        oq = OutlierTurboQuant(d=128, target_bits=3.5, seed=SEED)
        assert oq.n_high == 64, f"Expected n_high=64, got {oq.n_high}"
        assert oq.n_low == 64, f"Expected n_low=64, got {oq.n_low}"
        assert oq.high_bits == 4
        assert oq.low_bits == 3
        # Effective bits should be exactly 3.5 for d=128
        assert abs(oq.effective_bits - 3.5) < 1e-6


class TestCompressionRatio:
    """Verify compression ratios for fractional bit rates."""

    def test_outlier_compression_ratio_2_5_bit(self):
        """2.5-bit should give compression ratio around 5.5-6.5x."""
        oq = OutlierTurboQuant(d=128, target_bits=2.5, seed=SEED)
        ratio = oq.compression_ratio()
        # 128*16 / (64*3 + 64*2 + 16 + 16 + 16) = 2048 / 368 ~ 5.56
        assert 5.0 < ratio < 7.0, (
            f"2.5-bit compression ratio: {ratio:.2f}, expected ~5.5-6.0"
        )

    def test_outlier_compression_ratio_3_5_bit(self):
        """3.5-bit should give compression ratio around 4.0-5.0x."""
        oq = OutlierTurboQuant(d=128, target_bits=3.5, seed=SEED)
        ratio = oq.compression_ratio()
        # 128*16 / (64*4 + 64*3 + 16 + 16 + 16) = 2048 / 496 ~ 4.13
        assert 3.5 < ratio < 5.0, (
            f"3.5-bit compression ratio: {ratio:.2f}, expected ~4.0-4.5"
        )


class TestInnerProductQuality:
    """Validate inner product estimation quality."""

    def test_outlier_inner_product_quality(self):
        """Cosine sim of IP estimates vs true IPs should be > 0.90.

        Note: the baseline TurboQuantEstimator test uses > 0.85 for 3-bit.
        The outlier strategy at 3.5-bit should exceed that threshold.
        We use 0.90 as a comfortable margin above the baseline while
        accounting for finite-sample variance with n=500 pairs.
        """
        oq = OutlierTurboQuant(d=DEFAULT_DIM, target_bits=3.5, seed=SEED)
        n = 500
        keys = random_unit_vectors(n, DEFAULT_DIM, seed=42)
        queries = random_unit_vectors(n, DEFAULT_DIM, seed=77)

        true_ips = (queries * keys).sum(dim=1)  # (n,)
        compressed = oq.quantize(keys)
        est_ips = element_wise_ips(oq, queries, compressed)

        cos_sim = (
            (true_ips * est_ips).sum()
            / (true_ips.norm() * est_ips.norm() + 1e-8)
        ).item()

        assert cos_sim > 0.90, (
            f"IP cosine similarity: {cos_sim:.4f} (expected > 0.90)"
        )


class TestUnbiased:
    """The combined estimator should be unbiased: E[error] ~ 0."""

    def test_outlier_unbiased(self):
        """Mean error of IP estimates over 1000 pairs should be near 0."""
        torch.manual_seed(SEED)
        oq = OutlierTurboQuant(d=DEFAULT_DIM, target_bits=2.5, seed=SEED)
        n = 1000
        keys = random_unit_vectors(n, DEFAULT_DIM, seed=42)
        queries = random_unit_vectors(n, DEFAULT_DIM, seed=77)

        true_ips = (queries * keys).sum(dim=1)
        compressed = oq.quantize(keys)
        est_ips = element_wise_ips(oq, queries, compressed)

        mean_error = (est_ips - true_ips).mean().item()
        assert abs(mean_error) < 0.05, (
            f"Mean IP estimation error: {mean_error:.4f} (should be ~0)"
        )


class TestFloorCeilComparison:
    """Fractional bit rate quality should be between floor and ceil."""

    def _distortion(self, estimator_fn, keys, queries, true_ips):
        """Compute mean squared IP distortion for a quantizer."""
        compressed = estimator_fn(keys)
        est_ips = torch.diagonal(estimator_fn.ip(queries, compressed))
        return ((est_ips - true_ips) ** 2).mean().item()

    def test_outlier_better_than_floor(self):
        """2.5-bit quality should exceed pure 2-bit."""
        n = 1000
        keys = random_unit_vectors(n, DEFAULT_DIM, seed=42)
        queries = random_unit_vectors(n, DEFAULT_DIM, seed=77)
        true_ips = (queries * keys).sum(dim=1)

        # Pure 2-bit baseline
        est_2bit = TurboQuantEstimator(d=DEFAULT_DIM, bits=2, seed=SEED)
        comp_2 = est_2bit.quantize(keys)
        ips_2 = torch.diagonal(est_2bit.inner_product(queries, comp_2))
        distortion_2bit = ((ips_2 - true_ips) ** 2).mean().item()

        # 2.5-bit outlier
        oq = OutlierTurboQuant(d=DEFAULT_DIM, target_bits=2.5, seed=SEED)
        comp_25 = oq.quantize(keys)
        ips_25 = element_wise_ips(oq, queries, comp_25)
        distortion_25bit = ((ips_25 - true_ips) ** 2).mean().item()

        assert distortion_25bit < distortion_2bit, (
            f"2.5-bit distortion ({distortion_25bit:.6f}) should be less than "
            f"2-bit ({distortion_2bit:.6f})"
        )

    def test_outlier_worse_than_ceil(self):
        """2.5-bit quality should be below pure 3-bit."""
        n = 1000
        keys = random_unit_vectors(n, DEFAULT_DIM, seed=42)
        queries = random_unit_vectors(n, DEFAULT_DIM, seed=77)
        true_ips = (queries * keys).sum(dim=1)

        # Pure 3-bit baseline
        est_3bit = TurboQuantEstimator(d=DEFAULT_DIM, bits=3, seed=SEED)
        comp_3 = est_3bit.quantize(keys)
        ips_3 = torch.diagonal(est_3bit.inner_product(queries, comp_3))
        distortion_3bit = ((ips_3 - true_ips) ** 2).mean().item()

        # 2.5-bit outlier
        oq = OutlierTurboQuant(d=DEFAULT_DIM, target_bits=2.5, seed=SEED)
        comp_25 = oq.quantize(keys)
        ips_25 = element_wise_ips(oq, queries, comp_25)
        distortion_25bit = ((ips_25 - true_ips) ** 2).mean().item()

        assert distortion_25bit > distortion_3bit, (
            f"2.5-bit distortion ({distortion_25bit:.6f}) should be greater than "
            f"3-bit ({distortion_3bit:.6f})"
        )


class TestEffectiveBits:
    """Verify effective_bits matches target within rounding."""

    def test_outlier_effective_bits(self):
        """effective_bits should match target_bits within 1/d tolerance."""
        for target in [2.5, 3.0, 3.25, 3.5, 3.75]:
            oq = OutlierTurboQuant(d=DEFAULT_DIM, target_bits=target, seed=SEED)
            tolerance = 1.0 / DEFAULT_DIM + 1e-6  # rounding error up to 1 channel
            assert abs(oq.effective_bits - target) < tolerance, (
                f"target_bits={target}, effective_bits={oq.effective_bits:.4f}, "
                f"diff={abs(oq.effective_bits - target):.6f}, "
                f"tolerance={tolerance:.6f}"
            )


class TestVariousDims:
    """Outlier strategy should work for various dimensions."""

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_outlier_various_dims(self, dim):
        """Quantize + inner_product should work for d=64, 128, 256."""
        oq = OutlierTurboQuant(d=dim, target_bits=2.5, seed=SEED)
        n = 50
        keys = random_unit_vectors(n, dim, seed=42)
        queries = random_unit_vectors(10, dim, seed=77)

        compressed = oq.quantize(keys)
        scores = oq.inner_product(queries, compressed)
        assert scores.shape == (10, n), (
            f"Expected shape (10, {n}), got {scores.shape}"
        )


class TestSingleVector:
    """Single vector (1D input) should work."""

    def test_outlier_single_vector(self):
        """Quantize and inner_product with 1D input tensors."""
        oq = OutlierTurboQuant(d=DEFAULT_DIM, target_bits=3.5, seed=SEED)

        torch.manual_seed(SEED)
        key = torch.randn(DEFAULT_DIM)
        key = key / key.norm()
        query = torch.randn(DEFAULT_DIM)
        query = query / query.norm()

        compressed = oq.quantize(key)
        result = oq.inner_product(query, compressed)

        # Should be a scalar
        assert result.dim() == 0, (
            f"Expected scalar output for 1D inputs, got shape {result.shape}"
        )

        # Sanity: should be a reasonable inner product value
        true_ip = (query * key).sum().item()
        assert abs(result.item() - true_ip) < 0.5, (
            f"Estimated IP: {result.item():.4f}, true IP: {true_ip:.4f}"
        )


class TestIntegerFallback:
    """Integer target_bits should behave like standard TurboQuant."""

    def test_outlier_integer_fallback(self):
        """target_bits=3.0 should have n_high=0 and use only low group at 3-bit."""
        oq = OutlierTurboQuant(d=DEFAULT_DIM, target_bits=3.0, seed=SEED)
        assert oq.n_high == 0, f"Expected n_high=0 for integer bits, got {oq.n_high}"
        assert oq.n_low == DEFAULT_DIM
        assert oq.low_bits == 3
        assert abs(oq.effective_bits - 3.0) < 1e-6

        # Should still produce valid inner products
        n = 100
        keys = random_unit_vectors(n, DEFAULT_DIM, seed=42)
        queries = random_unit_vectors(n, DEFAULT_DIM, seed=77)

        true_ips = (queries * keys).sum(dim=1)
        compressed = oq.quantize(keys)
        est_ips = element_wise_ips(oq, queries, compressed)

        mean_error = (est_ips - true_ips).mean().item()
        assert abs(mean_error) < 0.1, (
            f"Integer fallback mean error: {mean_error:.4f} (should be ~0)"
        )


class TestGPU:
    """Test on CUDA when available."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_outlier_gpu(self):
        """Full pipeline should work on GPU."""
        oq = OutlierTurboQuant(
            d=DEFAULT_DIM, target_bits=2.5, seed=SEED, device="cuda"
        )
        n = 100
        keys = random_unit_vectors(n, DEFAULT_DIM, seed=42).cuda()
        queries = random_unit_vectors(10, DEFAULT_DIM, seed=77).cuda()

        compressed = oq.quantize(keys)
        scores = oq.inner_product(queries, compressed)

        assert scores.device.type == "cuda"
        assert scores.shape == (10, n)
