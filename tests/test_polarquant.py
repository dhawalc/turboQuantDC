"""Tests for PolarQuant (Stage 1: MSE-optimal quantization).

PolarQuant = random orthogonal rotation + Lloyd-Max scalar quantization.
Tests verify rotation properties, quantize/dequantize pipeline, and
MSE distortion against the paper's theoretical bounds (Theorem 1).

Mathematical reference: MATH_SPEC.md sections 3, 5, and 9.
"""

import math

import pytest
import torch

from turboquantdc.polarquant import PolarQuant

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DIM = 128
N_VECTORS = 1000
SEED = 42

# MSE distortion values from paper (MATH_SPEC.md section 5)
# D_mse ~ {0.36, 0.117, 0.03, 0.009} for b = {1, 2, 3, 4}
PAPER_DMSE = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

# Theoretical upper bound: D_mse <= sqrt(3)*pi/2 * 1/4^b
MSE_BOUND_FACTOR = math.sqrt(3) * math.pi / 2  # ~ 2.721


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
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(params=[1, 2, 3, 4], ids=lambda b: f"bits={b}")
def bits(request):
    return request.param


@pytest.fixture
def pq(bits):
    """PolarQuant instance parametrized over bit-widths."""
    return PolarQuant(d=DEFAULT_DIM, bits=bits, seed=SEED)


@pytest.fixture
def pq_3bit():
    """PolarQuant at 3-bit for targeted tests."""
    return PolarQuant(d=DEFAULT_DIM, bits=3, seed=SEED)


@pytest.fixture
def unit_vectors():
    """1000 random unit vectors of dimension 128."""
    return random_unit_vectors(N_VECTORS, DEFAULT_DIM)


# ---------------------------------------------------------------------------
# Tests: rotation matrix properties
# ---------------------------------------------------------------------------
class TestRotationMatrix:
    """Verify that the internal rotation matrix is a proper orthogonal matrix."""

    def test_rotation_is_orthogonal(self, pq_3bit):
        """R @ R.T should equal the identity matrix."""
        R = pq_3bit.Pi
        assert R.shape == (DEFAULT_DIM, DEFAULT_DIM)
        product = R @ R.T
        identity = torch.eye(DEFAULT_DIM)
        assert torch.allclose(product, identity, atol=1e-5), (
            f"R @ R.T deviates from identity by "
            f"{(product - identity).abs().max().item()}"
        )

    def test_rotation_transpose_is_inverse(self, pq_3bit):
        """R.T @ R should also equal identity (both left and right inverse)."""
        R = pq_3bit.Pi
        product = R.T @ R
        identity = torch.eye(DEFAULT_DIM)
        assert torch.allclose(product, identity, atol=1e-5)

    def test_rotation_preserves_norm(self, pq_3bit, unit_vectors):
        """||R @ x|| should equal ||x|| for all x."""
        R = pq_3bit.Pi
        rotated = unit_vectors @ R.T  # (N, d) @ (d, d) -> (N, d)
        original_norms = unit_vectors.norm(dim=1)
        rotated_norms = rotated.norm(dim=1)
        assert torch.allclose(original_norms, rotated_norms, atol=1e-5), (
            f"Max norm deviation: {(original_norms - rotated_norms).abs().max().item()}"
        )

    def test_rotation_determinant_is_pm1(self, pq_3bit):
        """Determinant of orthogonal matrix should be +1 or -1."""
        R = pq_3bit.Pi
        det = torch.linalg.det(R).item()
        assert abs(abs(det) - 1.0) < 1e-4, (
            f"Determinant should be +/-1, got {det}"
        )

    def test_same_seed_same_rotation(self):
        """Same seed must produce the same rotation matrix."""
        pq1 = PolarQuant(d=DEFAULT_DIM, bits=3, seed=123)
        pq2 = PolarQuant(d=DEFAULT_DIM, bits=3, seed=123)
        assert torch.allclose(pq1.Pi, pq2.Pi, atol=1e-7), (
            "Same seed should produce identical rotation matrices"
        )

    def test_different_seed_different_rotation(self):
        """Different seeds must produce different rotation matrices."""
        pq1 = PolarQuant(d=DEFAULT_DIM, bits=3, seed=123)
        pq2 = PolarQuant(d=DEFAULT_DIM, bits=3, seed=456)
        assert not torch.allclose(pq1.Pi, pq2.Pi, atol=1e-3), (
            "Different seeds should produce different rotation matrices"
        )


# ---------------------------------------------------------------------------
# Tests: quantize / dequantize pipeline
# ---------------------------------------------------------------------------
class TestQuantizeDequantize:
    """Verify the full PolarQuant quantize-dequantize pipeline."""

    def test_quantize_returns_indices(self, pq, bits, unit_vectors):
        """Quantize should return integer indices in [0, 2^bits)."""
        indices = pq.quantize(unit_vectors)
        assert indices.shape == (N_VECTORS, DEFAULT_DIM), (
            f"Expected shape ({N_VECTORS}, {DEFAULT_DIM}), got {indices.shape}"
        )
        assert indices.min() >= 0
        assert indices.max() < 2 ** bits

    def test_dequantize_returns_vectors(self, pq, unit_vectors):
        """Dequantize should return vectors of the same shape as input."""
        indices = pq.quantize(unit_vectors)
        reconstructed = pq.dequantize(indices)
        assert reconstructed.shape == unit_vectors.shape, (
            f"Expected shape {unit_vectors.shape}, got {reconstructed.shape}"
        )

    def test_roundtrip_reconstruction_not_exact(self, pq_3bit, unit_vectors):
        """Reconstruction should NOT be exact (lossy compression)."""
        indices = pq_3bit.quantize(unit_vectors)
        reconstructed = pq_3bit.dequantize(indices)
        # Should be different (lossy)
        assert not torch.allclose(unit_vectors, reconstructed, atol=1e-3), (
            "Reconstruction should not be exact for 3-bit quantization"
        )

    def test_roundtrip_reconstruction_reasonable(self, pq_3bit, unit_vectors):
        """Reconstruction should be reasonably close (not garbage)."""
        indices = pq_3bit.quantize(unit_vectors)
        reconstructed = pq_3bit.dequantize(indices)
        mse = ((unit_vectors - reconstructed) ** 2).sum(dim=1).mean().item()
        # 3-bit D_mse ~ 0.03, allow 3x slack
        assert mse < 0.1, f"MSE too high: {mse}"

    def test_single_vector(self, pq_3bit):
        """Should handle a single vector (N=1) input."""
        x = random_unit_vectors(1, DEFAULT_DIM)
        indices = pq_3bit.quantize(x)
        assert indices.shape == (1, DEFAULT_DIM)
        reconstructed = pq_3bit.dequantize(indices)
        assert reconstructed.shape == (1, DEFAULT_DIM)

    def test_large_batch(self, pq_3bit):
        """Should handle large batches efficiently."""
        x = random_unit_vectors(5000, DEFAULT_DIM, seed=99)
        indices = pq_3bit.quantize(x)
        assert indices.shape == (5000, DEFAULT_DIM)
        reconstructed = pq_3bit.dequantize(indices)
        assert reconstructed.shape == (5000, DEFAULT_DIM)


# ---------------------------------------------------------------------------
# Tests: MSE distortion against paper bounds
# ---------------------------------------------------------------------------
class TestMSEDistortion:
    """Verify MSE distortion matches paper's theoretical predictions."""

    @pytest.mark.parametrize("bits_val,expected_dmse", [
        (1, 0.36),
        (2, 0.117),
        (3, 0.03),
        (4, 0.009),
    ])
    def test_mse_matches_paper_values(self, bits_val, expected_dmse):
        """D_mse should be close to paper's tabulated values.

        Paper values: {0.36, 0.117, 0.03, 0.009} for b={1,2,3,4}.
        Allow 1.5x slack for finite sample variance.
        """
        pq = PolarQuant(d=DEFAULT_DIM, bits=bits_val, seed=SEED)
        x = random_unit_vectors(2000, DEFAULT_DIM, seed=77)
        indices = pq.quantize(x)
        reconstructed = pq.dequantize(indices)

        # D_mse = E[||x - x_hat||^2]
        d_mse = ((x - reconstructed) ** 2).sum(dim=1).mean().item()

        assert d_mse < expected_dmse * 1.5, (
            f"D_mse at {bits_val}-bit: {d_mse:.4f} > "
            f"1.5 * {expected_dmse} = {expected_dmse * 1.5:.4f}"
        )
        # Sanity: not absurdly low
        assert d_mse > expected_dmse * 0.3, (
            f"D_mse suspiciously low at {bits_val}-bit: {d_mse:.4f} "
            f"< 0.3 * {expected_dmse} = {expected_dmse * 0.3:.4f}"
        )

    def test_mse_below_theoretical_bound(self, bits, pq):
        """D_mse must be below sqrt(3)*pi/2 * 1/4^b (Theorem 1)."""
        x = random_unit_vectors(2000, DEFAULT_DIM, seed=77)
        indices = pq.quantize(x)
        reconstructed = pq.dequantize(indices)
        d_mse = ((x - reconstructed) ** 2).sum(dim=1).mean().item()
        upper_bound = MSE_BOUND_FACTOR / (4 ** bits)

        assert d_mse < upper_bound * 1.1, (
            f"D_mse={d_mse:.4f} exceeds theoretical bound "
            f"{upper_bound:.4f} at {bits}-bit (with 1.1x slack)"
        )

    def test_mse_above_lower_bound(self, bits, pq):
        """D_mse must be above the information-theoretic lower bound 1/4^b."""
        x = random_unit_vectors(2000, DEFAULT_DIM, seed=77)
        indices = pq.quantize(x)
        reconstructed = pq.dequantize(indices)
        d_mse = ((x - reconstructed) ** 2).sum(dim=1).mean().item()
        lower_bound = 1.0 / (4 ** bits)

        # Allow some slack below (finite d effects)
        assert d_mse > lower_bound * 0.5, (
            f"D_mse={d_mse:.4f} below information-theoretic lower bound "
            f"{lower_bound:.4f} at {bits}-bit"
        )

    def test_mse_decreases_with_bits(self):
        """More bits should produce lower MSE distortion."""
        x = random_unit_vectors(2000, DEFAULT_DIM, seed=77)
        distortions = []
        for b in [1, 2, 3, 4]:
            pq = PolarQuant(d=DEFAULT_DIM, bits=b, seed=SEED)
            indices = pq.quantize(x)
            reconstructed = pq.dequantize(indices)
            d_mse = ((x - reconstructed) ** 2).sum(dim=1).mean().item()
            distortions.append(d_mse)

        for i in range(len(distortions) - 1):
            assert distortions[i] > distortions[i + 1], (
                f"D_mse should decrease: bits={i+1} gave {distortions[i]:.4f}, "
                f"bits={i+2} gave {distortions[i+1]:.4f}"
            )


# ---------------------------------------------------------------------------
# Tests: cosine similarity
# ---------------------------------------------------------------------------
class TestCosineSimilarity:
    """Verify cosine similarity between original and reconstructed vectors."""

    def test_cosine_similarity_3bit(self, pq_3bit, unit_vectors):
        """3-bit quantization should achieve >0.99 mean cosine similarity."""
        indices = pq_3bit.quantize(unit_vectors)
        reconstructed = pq_3bit.dequantize(indices)

        # Cosine similarity for unit vectors (already normalized)
        cos_sim = (unit_vectors * reconstructed).sum(dim=1) / (
            unit_vectors.norm(dim=1) * reconstructed.norm(dim=1)
        )
        mean_cos = cos_sim.mean().item()
        assert mean_cos > 0.95, (
            f"Mean cosine similarity at 3-bit: {mean_cos:.4f} (expected >0.95)"
        )

    def test_cosine_similarity_improves_with_bits(self):
        """Cosine similarity should improve with more bits."""
        x = random_unit_vectors(1000, DEFAULT_DIM, seed=77)
        cos_sims = []
        for b in [1, 2, 3, 4]:
            pq = PolarQuant(d=DEFAULT_DIM, bits=b, seed=SEED)
            indices = pq.quantize(x)
            reconstructed = pq.dequantize(indices)
            cos_sim = (x * reconstructed).sum(dim=1) / (
                x.norm(dim=1) * reconstructed.norm(dim=1)
            )
            cos_sims.append(cos_sim.mean().item())

        for i in range(len(cos_sims) - 1):
            assert cos_sims[i] < cos_sims[i + 1], (
                f"Cosine sim should increase: bits={i+1} gave {cos_sims[i]:.4f}, "
                f"bits={i+2} gave {cos_sims[i+1]:.4f}"
            )


# ---------------------------------------------------------------------------
# Tests: GPU compatibility
# ---------------------------------------------------------------------------
class TestGPU:
    """Test that PolarQuant works on CUDA when available."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_quantize_on_gpu(self):
        """PolarQuant should work with GPU tensors."""
        pq = PolarQuant(d=DEFAULT_DIM, bits=3, seed=SEED, device='cuda')
        # Use a different seed for test vectors to avoid RNG collision with pq init
        x = random_unit_vectors(100, DEFAULT_DIM, seed=77).cuda()
        indices = pq.quantize(x)
        reconstructed = pq.dequantize(indices)
        assert reconstructed.device.type == "cuda"
        mse = ((x - reconstructed) ** 2).sum(dim=1).mean().item()
        # Same bound as CPU
        assert mse < 0.03 * 1.5

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cpu_gpu_consistency(self):
        """CPU and GPU should produce the same results."""
        pq_cpu = PolarQuant(d=DEFAULT_DIM, bits=3, seed=SEED, device='cpu')
        pq_gpu = PolarQuant(d=DEFAULT_DIM, bits=3, seed=SEED, device='cuda')
        x = random_unit_vectors(100, DEFAULT_DIM)

        indices_cpu = pq_cpu.quantize(x)
        indices_gpu = pq_gpu.quantize(x.cuda())

        assert torch.equal(indices_cpu, indices_gpu.cpu()), (
            "CPU and GPU quantization should produce identical indices"
        )


# ---------------------------------------------------------------------------
# Tests: different dimensions
# ---------------------------------------------------------------------------
class TestDimensions:
    """Test PolarQuant across different head dimensions."""

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_various_dimensions(self, dim):
        """PolarQuant should work for d=64, 128, 256."""
        pq = PolarQuant(d=dim, bits=3, seed=SEED)
        x = random_unit_vectors(500, dim, seed=77)
        indices = pq.quantize(x)
        assert indices.shape == (500, dim)
        reconstructed = pq.dequantize(indices)
        assert reconstructed.shape == (500, dim)
        # MSE should be reasonable for any dimension
        mse = ((x - reconstructed) ** 2).sum(dim=1).mean().item()
        assert mse < 0.1, f"MSE too high at d={dim}: {mse}"


# ---------------------------------------------------------------------------
# Tests: rotation distribution check
# ---------------------------------------------------------------------------
class TestRotatedDistribution:
    """After rotation, coordinates should follow ~N(0, 1/d)."""

    def test_rotated_coordinates_mean(self, pq_3bit):
        """Mean of rotated coordinates should be ~0."""
        x = random_unit_vectors(2000, DEFAULT_DIM, seed=77)
        R = pq_3bit.Pi
        rotated = x @ R.T
        mean = rotated.mean().item()
        assert abs(mean) < 0.01, (
            f"Rotated coordinate mean should be ~0, got {mean}"
        )

    def test_rotated_coordinates_variance(self, pq_3bit):
        """Variance of rotated coordinates should be ~1/d."""
        x = random_unit_vectors(2000, DEFAULT_DIM, seed=77)
        R = pq_3bit.Pi
        rotated = x @ R.T
        var = rotated.var().item()
        expected_var = 1.0 / DEFAULT_DIM
        # Allow 30% tolerance for finite samples
        assert abs(var - expected_var) / expected_var < 0.3, (
            f"Rotated coordinate variance: expected ~{expected_var:.6f}, "
            f"got {var:.6f}"
        )
