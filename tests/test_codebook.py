"""Tests for Lloyd-Max codebook generation.

Tests the codebook module against known properties of Lloyd-Max optimal
scalar quantizers for the N(0, 1/d) distribution (Gaussian approximation
of the Beta distribution on the unit hypersphere after random rotation).

Mathematical reference: MATH_SPEC.md sections 4 and 16.
"""

import math

import pytest
import torch

from turboquantdc.codebook import LloydMaxCodebook

# ---------------------------------------------------------------------------
# Constants from the paper
# ---------------------------------------------------------------------------
DEFAULT_DIM = 128  # paper's primary test dimension
SIGMA = 1.0 / math.sqrt(DEFAULT_DIM)  # std dev of coordinate distribution

# Known centroid values (in units of 1/sqrt(d)) — MATH_SPEC.md section 16
KNOWN_CENTROIDS_SIGMA_UNITS = {
    1: [0.7979],                   # +- sqrt(2/pi)
    2: [0.4528, 1.5104],          # +-0.4528, +-1.5104
}

# Known MSE distortion per-coordinate C(f_X, b) ~ value/d
# From MATH_SPEC.md section 4: D_mse ~ 0.36, 0.117, 0.03, 0.009
MSE_PER_COORD = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

# MSE upper bound: D_mse <= sqrt(3)*pi/2 * 1/4^b
MSE_BOUND_FACTOR = math.sqrt(3) * math.pi / 2  # ~ 2.721


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(params=[1, 2, 3, 4], ids=lambda b: f"bits={b}")
def bits(request):
    return request.param


@pytest.fixture
def codebook_1bit():
    """Lloyd-Max codebook for 1-bit quantization, d=128."""
    return LloydMaxCodebook(d=DEFAULT_DIM, bits=1)


@pytest.fixture
def codebook_2bit():
    """Lloyd-Max codebook for 2-bit quantization, d=128."""
    return LloydMaxCodebook(d=DEFAULT_DIM, bits=2)


@pytest.fixture
def codebook(bits):
    """Lloyd-Max codebook parametrized over bit-widths."""
    return LloydMaxCodebook(d=DEFAULT_DIM, bits=bits)


# ---------------------------------------------------------------------------
# Tests: structural properties
# ---------------------------------------------------------------------------
class TestCodebookStructure:
    """Verify that the codebook has the correct number of centroids,
    boundaries, and that they satisfy basic ordering constraints."""

    def test_num_centroids(self, bits, codebook):
        """Codebook must have exactly 2^bits centroids."""
        expected = 2 ** bits
        centroids = codebook.centroids
        assert centroids.shape == (expected,), (
            f"Expected {expected} centroids for {bits}-bit, got {centroids.shape}"
        )

    def test_num_boundaries(self, bits, codebook):
        """Codebook must have exactly 2^bits - 1 boundaries."""
        expected = 2 ** bits - 1
        boundaries = codebook.boundaries
        assert boundaries.shape == (expected,), (
            f"Expected {expected} boundaries for {bits}-bit, got {boundaries.shape}"
        )

    def test_centroids_sorted(self, codebook):
        """Centroids must be in strictly ascending order."""
        c = codebook.centroids
        diffs = c[1:] - c[:-1]
        assert torch.all(diffs > 0), "Centroids must be strictly sorted ascending"

    def test_boundaries_sorted(self, codebook):
        """Boundaries must be in strictly ascending order."""
        b = codebook.boundaries
        if b.numel() > 1:
            diffs = b[1:] - b[:-1]
            assert torch.all(diffs > 0), "Boundaries must be strictly sorted ascending"

    def test_centroids_are_tensors(self, codebook):
        """Centroids and boundaries must be torch.Tensor."""
        assert isinstance(codebook.centroids, torch.Tensor)
        assert isinstance(codebook.boundaries, torch.Tensor)


# ---------------------------------------------------------------------------
# Tests: symmetry properties
# ---------------------------------------------------------------------------
class TestCodebookSymmetry:
    """The N(0, 1/d) distribution is symmetric around 0, so Lloyd-Max
    centroids must also be symmetric around 0."""

    def test_centroid_sum_near_zero(self, codebook):
        """Sum of all centroids should be approximately 0 (symmetry)."""
        total = codebook.centroids.sum().item()
        assert abs(total) < 1e-6, (
            f"Centroid sum should be ~0 (symmetry), got {total}"
        )

    def test_centroids_symmetric_pairwise(self, codebook, bits):
        """Centroid i should equal -centroid(n-1-i) for all i."""
        c = codebook.centroids
        n = 2 ** bits
        for i in range(n // 2):
            assert abs(c[i].item() + c[n - 1 - i].item()) < 1e-6, (
                f"Centroid pair ({i}, {n-1-i}) not symmetric: "
                f"{c[i].item()} vs {-c[n-1-i].item()}"
            )

    def test_boundaries_symmetric(self, codebook, bits):
        """Boundaries should be symmetric: b_i = -b_{n-2-i}."""
        b = codebook.boundaries
        n_bounds = 2 ** bits - 1
        for i in range(n_bounds // 2):
            assert abs(b[i].item() + b[n_bounds - 1 - i].item()) < 1e-6, (
                f"Boundary pair ({i}, {n_bounds-1-i}) not symmetric"
            )

    def test_middle_boundary_is_zero(self, codebook, bits):
        """For even number of centroids, the middle boundary should be 0."""
        n_bounds = 2 ** bits - 1
        # All our bit-widths give even number of centroids (2,4,8,16)
        mid_idx = n_bounds // 2
        assert abs(codebook.boundaries[mid_idx].item()) < 1e-6, (
            f"Middle boundary should be 0, got {codebook.boundaries[mid_idx].item()}"
        )


# ---------------------------------------------------------------------------
# Tests: boundary = midpoint between adjacent centroids (Voronoi)
# ---------------------------------------------------------------------------
class TestBoundaryMidpoints:
    """In Lloyd-Max, boundaries are midpoints between adjacent centroids."""

    def test_boundaries_are_midpoints(self, codebook, bits):
        """Each boundary b_i should equal (c_i + c_{i+1}) / 2."""
        c = codebook.centroids
        b = codebook.boundaries
        for i in range(len(b)):
            expected = (c[i].item() + c[i + 1].item()) / 2.0
            actual = b[i].item()
            assert abs(actual - expected) < 1e-6, (
                f"Boundary {i}: expected {expected} (midpoint), got {actual}"
            )


# ---------------------------------------------------------------------------
# Tests: known centroid values
# ---------------------------------------------------------------------------
class TestKnownCentroidValues:
    """Verify centroid values against known analytical solutions."""

    def test_1bit_centroids(self, codebook_1bit):
        """1-bit centroids should be +-0.7979/sqrt(d) = +-sqrt(2/pi)/sqrt(d)."""
        c = codebook_1bit.centroids
        expected_abs = math.sqrt(2.0 / math.pi) / math.sqrt(DEFAULT_DIM)
        assert c.shape == (2,)
        # c[0] should be negative, c[1] positive
        assert abs(c[0].item() + expected_abs) < 1e-4, (
            f"1-bit centroid[0]: expected {-expected_abs}, got {c[0].item()}"
        )
        assert abs(c[1].item() - expected_abs) < 1e-4, (
            f"1-bit centroid[1]: expected {expected_abs}, got {c[1].item()}"
        )

    def test_2bit_centroids(self, codebook_2bit):
        """2-bit centroids should be +-0.4528/sqrt(d) and +-1.5104/sqrt(d)."""
        c = codebook_2bit.centroids
        sigma = SIGMA  # 1/sqrt(128)
        assert c.shape == (4,)

        expected_inner = 0.4528 * sigma
        expected_outer = 1.5104 * sigma

        # Sorted: [-outer, -inner, +inner, +outer]
        assert abs(c[0].item() + expected_outer) < 1e-3 * sigma, (
            f"2-bit centroid[0]: expected {-expected_outer}, got {c[0].item()}"
        )
        assert abs(c[1].item() + expected_inner) < 1e-3 * sigma, (
            f"2-bit centroid[1]: expected {-expected_inner}, got {c[1].item()}"
        )
        assert abs(c[2].item() - expected_inner) < 1e-3 * sigma, (
            f"2-bit centroid[2]: expected {expected_inner}, got {c[2].item()}"
        )
        assert abs(c[3].item() - expected_outer) < 1e-3 * sigma, (
            f"2-bit centroid[3]: expected {expected_outer}, got {c[3].item()}"
        )


# ---------------------------------------------------------------------------
# Tests: distortion properties
# ---------------------------------------------------------------------------
class TestDistortion:
    """Verify MSE distortion of the codebook against theoretical bounds."""

    def test_distortion_decreases_with_bits(self):
        """More bits should give lower quantization distortion."""
        distortions = []
        for b in [1, 2, 3, 4]:
            cb = LloydMaxCodebook(d=DEFAULT_DIM, bits=b)
            # Quantize a large sample from N(0, 1/d) and measure MSE
            torch.manual_seed(42)
            samples = torch.randn(10000) * SIGMA
            indices = cb.quantize(samples)
            reconstructed = cb.dequantize(indices)
            mse = ((samples - reconstructed) ** 2).mean().item()
            distortions.append(mse)

        for i in range(len(distortions) - 1):
            assert distortions[i] > distortions[i + 1], (
                f"Distortion should decrease: bits={i+1} gave {distortions[i]}, "
                f"bits={i+2} gave {distortions[i+1]}"
            )

    @pytest.mark.parametrize("bits_val,expected_mse_per_coord", [
        (1, 0.36),
        (2, 0.117),
        (3, 0.03),
        (4, 0.009),
    ])
    def test_mse_matches_paper_values(self, bits_val, expected_mse_per_coord):
        """Per-coordinate MSE should match the paper's tabulated values.

        C(f_X, b) ~ 0.36/d, 0.117/d, 0.03/d, 0.009/d for b=1,2,3,4.
        We sample from N(0, 1/d) and measure. Allow 1.5x slack for finite samples.
        """
        cb = LloydMaxCodebook(d=DEFAULT_DIM, bits=bits_val)
        torch.manual_seed(42)
        n_samples = 50000
        samples = torch.randn(n_samples) * SIGMA
        indices = cb.quantize(samples)
        reconstructed = cb.dequantize(indices)
        mse_per_coord = ((samples - reconstructed) ** 2).mean().item()
        expected = expected_mse_per_coord / DEFAULT_DIM

        # Allow 1.5x slack (finite sample variance)
        assert mse_per_coord < expected * 1.5, (
            f"Per-coord MSE at {bits_val}-bit: {mse_per_coord} > "
            f"1.5 * {expected} = {expected * 1.5}"
        )
        # Also check it's not absurdly low (sanity)
        assert mse_per_coord > expected * 0.3, (
            f"Per-coord MSE at {bits_val}-bit suspiciously low: "
            f"{mse_per_coord} < 0.3 * {expected}"
        )

    def test_mse_below_theoretical_bound(self, bits, codebook):
        """Total MSE (d * C) should be below sqrt(3)*pi/2 * 1/4^b.

        This is Theorem 1 from the paper.
        """
        torch.manual_seed(42)
        n_samples = 50000
        samples = torch.randn(n_samples) * SIGMA
        indices = codebook.quantize(samples)
        reconstructed = codebook.dequantize(indices)
        mse_per_coord = ((samples - reconstructed) ** 2).mean().item()

        # D_mse = d * C(f_X, b) <= sqrt(3)*pi/2 * 1/4^b
        d_mse = mse_per_coord * DEFAULT_DIM
        upper_bound = MSE_BOUND_FACTOR / (4 ** bits)

        assert d_mse < upper_bound * 1.1, (
            f"D_mse={d_mse} exceeds theoretical bound {upper_bound} at {bits}-bit"
        )


# ---------------------------------------------------------------------------
# Tests: quantize / dequantize roundtrip
# ---------------------------------------------------------------------------
class TestQuantizeDequantize:
    """Verify that the quantize/dequantize cycle works correctly."""

    def test_roundtrip_maps_to_centroid(self, codebook):
        """Dequantize(quantize(x)) should return the nearest centroid."""
        torch.manual_seed(42)
        samples = torch.randn(1000) * SIGMA
        indices = codebook.quantize(samples)
        reconstructed = codebook.dequantize(indices)

        # Each reconstructed value should be one of the centroids
        centroids = codebook.centroids
        for val in reconstructed:
            dists = (centroids - val).abs()
            assert dists.min().item() < 1e-6, (
                f"Reconstructed value {val.item()} is not a centroid"
            )

    def test_quantize_returns_valid_indices(self, bits, codebook):
        """Quantize should return indices in [0, 2^bits - 1]."""
        torch.manual_seed(42)
        samples = torch.randn(1000) * SIGMA
        indices = codebook.quantize(samples)

        assert indices.min() >= 0, f"Negative index found: {indices.min()}"
        assert indices.max() < 2 ** bits, (
            f"Index {indices.max()} >= {2 ** bits}"
        )

    def test_quantize_selects_nearest_centroid(self, codebook):
        """Quantize should select the nearest centroid for each input."""
        centroids = codebook.centroids
        # Test with exact centroid values — should map to themselves
        indices = codebook.quantize(centroids)
        reconstructed = codebook.dequantize(indices)
        assert torch.allclose(reconstructed, centroids, atol=1e-6), (
            "Quantizing centroids should produce the same centroids"
        )

    def test_dequantize_with_known_indices(self, codebook):
        """Dequantize of index i should return centroid[i]."""
        centroids = codebook.centroids
        n = centroids.shape[0]
        indices = torch.arange(n)
        reconstructed = codebook.dequantize(indices)
        assert torch.allclose(reconstructed, centroids, atol=1e-6)

    def test_batch_quantize_shape(self, bits, codebook):
        """Quantize should handle batch inputs and preserve shape."""
        torch.manual_seed(42)
        batch_size = 256
        samples = torch.randn(batch_size) * SIGMA
        indices = codebook.quantize(samples)
        assert indices.shape == (batch_size,), (
            f"Expected shape ({batch_size},), got {indices.shape}"
        )

    def test_multidim_quantize(self, bits, codebook):
        """Quantize should handle (N, d) input, quantizing per-element."""
        torch.manual_seed(42)
        N, d = 100, DEFAULT_DIM
        samples = torch.randn(N, d) * SIGMA
        indices = codebook.quantize(samples)
        assert indices.shape == (N, d), (
            f"Expected shape ({N}, {d}), got {indices.shape}"
        )
        reconstructed = codebook.dequantize(indices)
        assert reconstructed.shape == (N, d)


# ---------------------------------------------------------------------------
# Tests: Gaussian vs exact Beta distribution
# ---------------------------------------------------------------------------
class TestGaussianVsBeta:
    """For d >= 128, Gaussian and exact Beta codebooks should agree closely."""

    @pytest.mark.parametrize("bits_val", [1, 2, 3])
    def test_gaussian_beta_centroids_close(self, bits_val):
        """Gaussian-approximation centroids should closely match
        exact-Beta centroids for d=128."""
        cb_gauss = LloydMaxCodebook(d=DEFAULT_DIM, bits=bits_val, use_exact=False)
        cb_beta = LloydMaxCodebook(d=DEFAULT_DIM, bits=bits_val, use_exact=True)

        gauss_c = cb_gauss.centroids
        beta_c = cb_beta.centroids
        max_diff = (gauss_c - beta_c).abs().max().item()

        # Should agree to within 5% of centroid scale
        centroid_scale = gauss_c.abs().max().item()
        relative_diff = max_diff / centroid_scale if centroid_scale > 1e-10 else max_diff
        assert relative_diff < 0.05, (
            f"Gaussian vs Beta centroids differ by {relative_diff*100:.1f}% "
            f"at {bits_val}-bit, d={DEFAULT_DIM}"
        )


# ---------------------------------------------------------------------------
# Tests: different dimensions
# ---------------------------------------------------------------------------
class TestDimensionVariation:
    """Verify codebook works correctly across different dimensions."""

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_centroid_scales_with_dimension(self, dim):
        """Centroid magnitudes should scale as 1/sqrt(d)."""
        cb = LloydMaxCodebook(d=dim, bits=2)
        c = cb.centroids
        # Outer centroid should be ~1.51/sqrt(d)
        outer = c[-1].item()
        expected = 1.5104 / math.sqrt(dim)
        assert abs(outer - expected) / expected < 0.05, (
            f"Outer centroid at d={dim}: expected ~{expected}, got {outer}"
        )

    def test_codebook_at_dim_64(self):
        """Codebook should work at d=64 (some model architectures use this)."""
        cb = LloydMaxCodebook(d=64, bits=3)
        assert cb.centroids.shape == (8,)
        assert cb.boundaries.shape == (7,)
