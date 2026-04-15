"""Tests for E8 lattice vector quantization.

Tests cover:
1. E8 lattice point validity (even sum constraint)
2. Nearest point finding (both strict and relaxed)
3. Quantizer roundtrip accuracy
4. MSE comparison vs scalar quantization
5. Integration with WHT rotation pipeline
"""

import pytest
import torch
import math
from turboquantdc.e8_lattice import (
    nearest_d8,
    nearest_e8,
    nearest_e8_relaxed,
    E8Quantizer,
    calibrate_scale,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def random_8d(device):
    """Random 8D vectors for testing."""
    torch.manual_seed(42)
    return torch.randn(1000, 8, device=device)


@pytest.fixture
def random_128d(device):
    """Random 128D vectors (typical head_dim)."""
    torch.manual_seed(42)
    return torch.randn(500, 128, device=device)


# ── D8 Lattice Tests ──────────────────────────────────────────────────────

class TestNearestD8:
    def test_output_is_integer(self, random_8d):
        result = nearest_d8(random_8d)
        assert torch.allclose(result, result.round(), atol=1e-6), \
            "D8 points must have integer coordinates"

    def test_even_sum(self, random_8d):
        result = nearest_d8(random_8d)
        sums = result.sum(dim=-1)
        assert (sums % 2 == 0).all(), \
            "D8 points must have even coordinate sum"

    def test_exact_lattice_point_unchanged(self, device):
        """A point already on D8 should map to itself."""
        pt = torch.tensor([[2.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], device=device)
        assert pt.sum() % 2 == 0  # verify it's on D8
        result = nearest_d8(pt)
        assert torch.allclose(result, pt)

    def test_batch_shape(self, device):
        x = torch.randn(3, 5, 8, device=device)
        result = nearest_d8(x)
        assert result.shape == (3, 5, 8)


# ── E8 Lattice Tests ──────────────────────────────────────────────────────

class TestNearestE8:
    def test_is_e8_point(self, random_8d):
        """E8 = D8 ∪ (D8 + 1/2). Result must be in one coset."""
        result = nearest_e8(random_8d)
        # Check: either all integers with even sum, or all half-integers with even sum
        is_int = torch.allclose(result, result.round(), atol=1e-6)
        is_half = torch.allclose(result - 0.5, (result - 0.5).round(), atol=1e-6)
        for i in range(result.shape[0]):
            pt = result[i]
            r = pt.round()
            h = (pt - 0.5).round()
            int_check = torch.allclose(pt, r, atol=1e-6) and r.sum() % 2 == 0
            half_check = torch.allclose(pt, h + 0.5, atol=1e-6) and (h + 0.5).sum() % 2 == 0
            assert int_check or half_check, \
                f"Point {pt.tolist()} is not on E8 lattice"

    def test_closer_than_d8(self, random_8d):
        """E8 nearest should be at least as close as D8 nearest."""
        e8_pt = nearest_e8(random_8d)
        d8_pt = nearest_d8(random_8d)
        e8_dist = ((random_8d - e8_pt) ** 2).sum(dim=-1)
        d8_dist = ((random_8d - d8_pt) ** 2).sum(dim=-1)
        assert (e8_dist <= d8_dist + 1e-6).all(), \
            "E8 should be at least as close as D8"

    def test_zero_input(self, device):
        x = torch.zeros(1, 8, device=device)
        result = nearest_e8(x)
        assert torch.allclose(result, torch.zeros(1, 8, device=device))


class TestNearestE8Relaxed:
    def test_closer_than_strict(self, random_8d):
        """Relaxed E8 should be at least as close as strict E8."""
        strict = nearest_e8(random_8d)
        relaxed = nearest_e8_relaxed(random_8d)
        strict_dist = ((random_8d - strict) ** 2).sum(dim=-1)
        relaxed_dist = ((random_8d - relaxed) ** 2).sum(dim=-1)
        assert (relaxed_dist <= strict_dist + 1e-6).all(), \
            "Relaxed E8 should be at least as close as strict E8"

    def test_lower_mse(self, random_8d):
        """Relaxed E8 should have lower or equal MSE."""
        strict = nearest_e8(random_8d)
        relaxed = nearest_e8_relaxed(random_8d)
        mse_strict = ((random_8d - strict) ** 2).mean().item()
        mse_relaxed = ((random_8d - relaxed) ** 2).mean().item()
        assert mse_relaxed <= mse_strict + 1e-6


# ── E8Quantizer Tests ────────────────────────────────────────────────────

class TestE8Quantizer:
    def test_roundtrip_shape(self, random_128d):
        eq = E8Quantizer(scale=0.1)
        lp, recon = eq.quantize(random_128d)
        assert lp.shape == random_128d.shape
        assert recon.shape == random_128d.shape

    def test_dequantize_matches(self, random_128d):
        eq = E8Quantizer(scale=0.1)
        lp, recon = eq.quantize(random_128d)
        recon2 = eq.dequantize(lp)
        assert torch.allclose(recon, recon2)

    def test_dimension_must_be_divisible_by_8(self, device):
        eq = E8Quantizer(scale=0.1)
        x = torch.randn(10, 7, device=device)
        with pytest.raises(AssertionError, match="divisible by 8"):
            eq.quantize(x)

    def test_smaller_scale_lower_mse(self, random_128d):
        """Smaller scale = finer quantization = lower MSE."""
        eq_coarse = E8Quantizer(scale=1.0)
        eq_fine = E8Quantizer(scale=0.1)
        _, recon_coarse = eq_coarse.quantize(random_128d)
        _, recon_fine = eq_fine.quantize(random_128d)
        mse_coarse = ((random_128d - recon_coarse) ** 2).mean().item()
        mse_fine = ((random_128d - recon_fine) ** 2).mean().item()
        assert mse_fine < mse_coarse, \
            f"Finer scale should give lower MSE: {mse_fine} vs {mse_coarse}"

    def test_relaxed_vs_strict(self, random_128d):
        """Relaxed mode should have lower or equal MSE."""
        scale = 0.1
        eq_strict = E8Quantizer(scale=scale, relaxed=False)
        eq_relaxed = E8Quantizer(scale=scale, relaxed=True)
        _, recon_strict = eq_strict.quantize(random_128d)
        _, recon_relaxed = eq_relaxed.quantize(random_128d)
        mse_strict = ((random_128d - recon_strict) ** 2).mean().item()
        mse_relaxed = ((random_128d - recon_relaxed) ** 2).mean().item()
        assert mse_relaxed <= mse_strict + 1e-6

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_various_dimensions(self, d, device):
        eq = E8Quantizer(scale=0.1)
        x = torch.randn(100, d, device=device)
        lp, recon = eq.quantize(x)
        assert recon.shape == x.shape

    def test_cosine_similarity(self, random_128d):
        """E8 quantization should preserve direction well."""
        eq = E8Quantizer(scale=0.05)
        _, recon = eq.quantize(random_128d)
        cos = torch.nn.functional.cosine_similarity(
            random_128d, recon, dim=-1
        ).mean().item()
        assert cos > 0.95, f"Cosine similarity {cos} too low"


# ── Calibrate Scale Tests ────────────────────────────────────────────────

class TestCalibrateScale:
    def test_positive(self, random_128d):
        scale = calibrate_scale(random_128d, target_bits=3.0)
        assert scale > 0

    def test_lower_bits_larger_scale(self, random_128d):
        s2 = calibrate_scale(random_128d, target_bits=2.0)
        s3 = calibrate_scale(random_128d, target_bits=3.0)
        s4 = calibrate_scale(random_128d, target_bits=4.0)
        assert s2 > s3 > s4, \
            f"Lower bits should give larger scale: {s2} > {s3} > {s4}"


# ── Integration with WHT Pipeline ────────────────────────────────────────

class TestE8WithWHT:
    def test_wht_e8_roundtrip(self, device):
        """Full pipeline: WHT rotate → E8 quantize → inverse WHT."""
        from turboquantdc.rotation import fast_wht

        d = 128
        x = torch.randn(200, d, device=device)
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        unit = x / norms

        # WHT rotate
        rotated = fast_wht(unit)

        # E8 quantize
        scale = 2.0 * rotated.std().item() / 8  # ~3-bit equivalent
        eq = E8Quantizer(scale=scale, relaxed=True)
        _, recon_rot = eq.quantize(rotated)

        # Inverse WHT (unnormalized: fast_wht(fast_wht(x)) = d*x)
        recon_unit = fast_wht(recon_rot) / d
        recon = recon_unit * norms

        cos = torch.nn.functional.cosine_similarity(x, recon, dim=-1).mean().item()
        assert cos > 0.98, f"WHT+E8 roundtrip cosine {cos} too low"

    def test_e8_beats_scalar_on_mse(self, device):
        """E8 should have lower MSE than scalar Lloyd-Max on WHT-rotated data."""
        from turboquantdc.rotation import fast_wht
        from turboquantdc.codebook import LloydMaxCodebook

        d = 128
        torch.manual_seed(42)
        x = torch.randn(500, d, device=device)
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        unit = x / norms
        rotated = fast_wht(unit)

        # Scalar Lloyd-Max
        cb = LloydMaxCodebook(d, 3)
        indices = cb.quantize(rotated)
        recon_scalar = cb.centroids.to(device)[indices]
        mse_scalar = ((rotated - recon_scalar) ** 2).mean().item()

        # E8
        scale = 2.0 * rotated.std().item() / 8
        eq = E8Quantizer(scale=scale, relaxed=True)
        _, recon_e8 = eq.quantize(rotated)
        mse_e8 = ((rotated - recon_e8) ** 2).mean().item()

        assert mse_e8 < mse_scalar, \
            f"E8 MSE {mse_e8} should be lower than scalar {mse_scalar}"
