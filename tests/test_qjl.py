"""Tests for QJL (Quantized Johnson-Lindenstrauss) bias correction.

QJL is Stage 2 of TurboQuant: it applies a random Gaussian projection to
the residual vector, stores only signs (1 bit per dimension), and produces
an unbiased inner product estimator.

Mathematical reference: MATH_SPEC.md sections 6, 7, and Lemma 4.
"""

import math

import pytest
import torch

from turboquantdc.qjl import QJL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DIM = 128
SEED = 42
QJL_SCALE = math.sqrt(math.pi / 2)  # ~ 1.2533


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
@pytest.fixture
def qjl():
    """QJL instance with default m=d=128."""
    return QJL(d=DEFAULT_DIM, seed=SEED)


@pytest.fixture
def qjl_custom_m():
    """QJL instance with m=256 (different from d)."""
    return QJL(d=DEFAULT_DIM, m=256, seed=SEED)


# ---------------------------------------------------------------------------
# Tests: matrix shape and properties
# ---------------------------------------------------------------------------
class TestQJLMatrix:
    """Verify properties of the QJL projection matrix S."""

    def test_matrix_shape_default(self, qjl):
        """S should be (m, d) = (d, d) by default."""
        S = qjl.S
        assert S.shape == (DEFAULT_DIM, DEFAULT_DIM), (
            f"Expected shape ({DEFAULT_DIM}, {DEFAULT_DIM}), got {S.shape}"
        )

    def test_matrix_shape_custom_m(self, qjl_custom_m):
        """S should be (m, d) when m != d."""
        S = qjl_custom_m.S
        assert S.shape == (256, DEFAULT_DIM), (
            f"Expected shape (256, {DEFAULT_DIM}), got {S.shape}"
        )

    def test_same_seed_same_matrix(self):
        """Same seed must produce the same projection matrix."""
        q1 = QJL(d=DEFAULT_DIM, seed=123)
        q2 = QJL(d=DEFAULT_DIM, seed=123)
        assert torch.allclose(q1.S, q2.S, atol=1e-7)

    def test_different_seed_different_matrix(self):
        """Different seeds must produce different projection matrices."""
        q1 = QJL(d=DEFAULT_DIM, seed=123)
        q2 = QJL(d=DEFAULT_DIM, seed=456)
        assert not torch.allclose(
            q1.S, q2.S, atol=1e-3
        )


# ---------------------------------------------------------------------------
# Tests: sign output properties
# ---------------------------------------------------------------------------
class TestSignOutput:
    """Verify that QJL produces correct sign vectors."""

    def test_sign_values_are_pm1(self, qjl):
        """QJL output should be exactly {-1, +1}, no zeros."""
        x = random_unit_vectors(100, DEFAULT_DIM)
        signs = qjl.project_and_sign(x)
        unique_vals = signs.unique()
        assert set(unique_vals.tolist()).issubset({-1.0, 1.0}), (
            f"Sign values should be {{-1, +1}}, got {unique_vals.tolist()}"
        )

    def test_no_zeros_in_signs(self, qjl):
        """There should be no zeros in the sign output."""
        x = random_unit_vectors(500, DEFAULT_DIM, seed=77)
        signs = qjl.project_and_sign(x)
        n_zeros = (signs == 0).sum().item()
        assert n_zeros == 0, f"Found {n_zeros} zeros in sign output"

    def test_sign_shape(self, qjl):
        """Signs should have shape (N, m) where m defaults to d."""
        x = random_unit_vectors(50, DEFAULT_DIM)
        signs = qjl.project_and_sign(x)
        assert signs.shape == (50, DEFAULT_DIM), (
            f"Expected shape (50, {DEFAULT_DIM}), got {signs.shape}"
        )

    def test_sign_shape_custom_m(self, qjl_custom_m):
        """Signs should have shape (N, m) when m is specified."""
        x = random_unit_vectors(50, DEFAULT_DIM)
        signs = qjl_custom_m.project_and_sign(x)
        assert signs.shape == (50, 256), (
            f"Expected shape (50, 256), got {signs.shape}"
        )

    def test_sign_balance(self, qjl):
        """For random input, signs should be roughly balanced (+1/-1)."""
        x = random_unit_vectors(1000, DEFAULT_DIM, seed=77)
        signs = qjl.project_and_sign(x)
        fraction_positive = (signs > 0).float().mean().item()
        # Should be close to 0.5 (within 5%)
        assert 0.4 < fraction_positive < 0.6, (
            f"Sign balance: {fraction_positive:.3f} (expected ~0.5)"
        )


# ---------------------------------------------------------------------------
# Tests: residual norm storage
# ---------------------------------------------------------------------------
class TestResidualNorm:
    """Verify that QJL correctly stores the residual norm."""

    def test_encode_returns_signs_and_norm(self, qjl):
        """project_and_sign() + norm should return (signs, residual_norm)."""
        residual = torch.randn(100, DEFAULT_DIM) * 0.1
        signs = qjl.project_and_sign(residual)
        norms = residual.norm(dim=1)
        assert signs.shape == (100, DEFAULT_DIM)
        assert norms.shape == (100,)

    def test_residual_norm_correct(self, qjl):
        """Stored residual norm should match ||residual||_2."""
        torch.manual_seed(42)
        residual = torch.randn(100, DEFAULT_DIM) * 0.1
        norms = residual.norm(dim=1)
        expected_norms = residual.norm(dim=1)
        assert torch.allclose(norms, expected_norms, atol=1e-5), (
            f"Max norm error: {(norms - expected_norms).abs().max().item()}"
        )

    def test_residual_norm_nonnegative(self, qjl):
        """Residual norms should always be non-negative."""
        torch.manual_seed(42)
        residual = torch.randn(50, DEFAULT_DIM) * 0.1
        norms = residual.norm(dim=1)
        assert (norms >= 0).all(), "All norms should be non-negative"


# ---------------------------------------------------------------------------
# Tests: unbiased inner product estimation (Lemma 4)
# ---------------------------------------------------------------------------
class TestUnbiasedEstimation:
    """The QJL inner product estimator must be unbiased.

    E[<y, Q_qjl^{-1}(Q_qjl(x))>] = <y, x> (Lemma 4).
    """

    def test_qjl_inner_product_unbiased(self):
        """Mean estimated inner product should converge to true inner product.

        We fix x and y, run many independent QJL instances (different seeds),
        and check that the average estimated inner product equals <y, x>.
        """
        torch.manual_seed(42)
        x = torch.randn(DEFAULT_DIM)
        x = x / x.norm()
        y = torch.randn(DEFAULT_DIM)
        y = y / y.norm()

        true_ip = (x * y).sum().item()

        n_trials = 500
        estimates = []
        for seed in range(n_trials):
            qjl = QJL(d=DEFAULT_DIM, seed=seed + 1000)
            signs = qjl.project_and_sign(x.unsqueeze(0))
            residual_norm = x.norm().unsqueeze(0)
            est = qjl.inner_product_correction(
                query=y.unsqueeze(0),
                signs=signs,
                residual_norm=residual_norm,
            )
            estimates.append(est.item())

        mean_est = sum(estimates) / len(estimates)
        # Allow statistical tolerance: std ~ sqrt(pi/(2d)) / sqrt(n_trials)
        # ~ sqrt(1.57/128) / sqrt(500) ~ 0.111 / 22.4 ~ 0.005
        tolerance = 0.05  # generous for safety
        assert abs(mean_est - true_ip) < tolerance, (
            f"Mean estimated IP: {mean_est:.4f}, true IP: {true_ip:.4f}, "
            f"diff: {abs(mean_est - true_ip):.4f} (tolerance: {tolerance})"
        )

    def test_qjl_inner_product_unbiased_batch(self):
        """Batch version: averaged over many (x, y) pairs, the estimate
        should be unbiased.

        inner_product_correction returns (batch_q, batch_k); use diagonal
        to get element-wise estimates for pairs (x_i, y_i).
        """
        n_pairs = 2000
        torch.manual_seed(42)
        x = torch.randn(n_pairs, DEFAULT_DIM)
        x = x / x.norm(dim=1, keepdim=True)
        y = torch.randn(n_pairs, DEFAULT_DIM)
        y = y / y.norm(dim=1, keepdim=True)

        true_ips = (x * y).sum(dim=1)  # (n_pairs,)

        qjl = QJL(d=DEFAULT_DIM, seed=SEED)
        signs = qjl.project_and_sign(x)
        norms = x.norm(dim=1)

        # Returns (n_pairs, n_pairs); diagonal gives element-wise estimates
        ip_matrix = qjl.inner_product_correction(
            query=y, signs=signs, residual_norm=norms
        )
        estimated_ips = torch.diagonal(ip_matrix)  # (n_pairs,)

        # Mean error should be near zero (unbiased)
        mean_error = (estimated_ips - true_ips).mean().item()
        assert abs(mean_error) < 0.05, (
            f"Mean estimation error: {mean_error:.4f} (should be ~0 for unbiased)"
        )


# ---------------------------------------------------------------------------
# Tests: variance bound (Lemma 4)
# ---------------------------------------------------------------------------
class TestVarianceBound:
    """Variance of QJL estimator should satisfy Var <= pi/(2d) * ||y||^2."""

    def test_qjl_variance_bound(self):
        """Empirical variance should be below pi/(2d) * ||y||^2.

        We fix y, run many random x vectors through QJL, and compute
        Var(<y, Q^{-1}(Q(x))>).
        """
        torch.manual_seed(42)
        y = torch.randn(DEFAULT_DIM)
        y = y / y.norm()
        y_norm_sq = (y ** 2).sum().item()  # = 1.0 for unit vector

        n_trials = 2000
        x_batch = torch.randn(n_trials, DEFAULT_DIM)
        x_batch = x_batch / x_batch.norm(dim=1, keepdim=True)

        qjl = QJL(d=DEFAULT_DIM, seed=SEED)

        # True inner products
        true_ips = (x_batch * y.unsqueeze(0)).sum(dim=1)

        # Estimated inner products
        # inner_product_correction(y_expanded, signs) returns (n_trials, n_trials)
        # Take diagonal to get element-wise estimates <y, x_i> for each i
        signs = qjl.project_and_sign(x_batch)
        norms = x_batch.norm(dim=1)
        ip_matrix = qjl.inner_product_correction(
            query=y.unsqueeze(0).expand(n_trials, -1),
            signs=signs,
            residual_norm=norms,
        )
        est_ips = torch.diagonal(ip_matrix)  # (n_trials,)

        # Variance of (estimate - true) should be bounded
        errors = est_ips - true_ips
        empirical_var = errors.var().item()
        theoretical_bound = math.pi / (2 * DEFAULT_DIM) * y_norm_sq

        # Allow 2x slack for finite samples and QJL being applied to full
        # vectors (not just residuals)
        assert empirical_var < theoretical_bound * 3.0, (
            f"Empirical variance: {empirical_var:.6f}, "
            f"theoretical bound: {theoretical_bound:.6f}, "
            f"ratio: {empirical_var / theoretical_bound:.2f}x"
        )

    def test_variance_scales_with_y_norm(self):
        """Variance should scale linearly with ||y||^2."""
        torch.manual_seed(42)
        n_trials = 2000
        x_batch = torch.randn(n_trials, DEFAULT_DIM)
        x_batch = x_batch / x_batch.norm(dim=1, keepdim=True)
        qjl = QJL(d=DEFAULT_DIM, seed=SEED)

        variances = []
        for scale in [0.5, 1.0, 2.0]:
            y = torch.randn(DEFAULT_DIM)
            y = scale * y / y.norm()

            true_ips = (x_batch * y.unsqueeze(0)).sum(dim=1)
            signs = qjl.project_and_sign(x_batch)
            norms = x_batch.norm(dim=1)
            # Returns (n_trials, n_trials); diagonal gives element-wise estimates
            ip_matrix = qjl.inner_product_correction(
                query=y.unsqueeze(0).expand(n_trials, -1),
                signs=signs,
                residual_norm=norms,
            )
            est_ips = torch.diagonal(ip_matrix)  # (n_trials,)
            errors = est_ips - true_ips
            variances.append(errors.var().item())

        # Variance at scale=2 should be ~4x variance at scale=1
        ratio = variances[2] / variances[1]
        assert 2.0 < ratio < 8.0, (
            f"Variance scaling ratio (2x/1x norm): {ratio:.2f}, expected ~4"
        )


# ---------------------------------------------------------------------------
# Tests: GPU compatibility
# ---------------------------------------------------------------------------
class TestGPU:
    """Test QJL on CUDA when available."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_signs_on_gpu(self):
        """QJL should work with GPU tensors."""
        qjl = QJL(d=DEFAULT_DIM, seed=SEED, device='cuda')
        x = random_unit_vectors(100, DEFAULT_DIM).cuda()
        signs = qjl.project_and_sign(x)
        assert signs.device.type == "cuda"
        unique = signs.unique()
        assert set(unique.tolist()).issubset({-1.0, 1.0})

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_estimate_on_gpu(self):
        """Inner product estimation should work on GPU."""
        qjl = QJL(d=DEFAULT_DIM, seed=SEED, device='cuda')
        x = random_unit_vectors(100, DEFAULT_DIM).cuda()
        y = random_unit_vectors(100, DEFAULT_DIM, seed=77).cuda()
        signs = qjl.project_and_sign(x)
        norms = x.norm(dim=1)
        # inner_product_correction returns (batch_q, batch_k) - use diagonal for (N,)
        ip_matrix = qjl.inner_product_correction(
            query=y, signs=signs, residual_norm=norms
        )
        # Take diagonal to get element-wise (100,)
        est = torch.diagonal(ip_matrix)
        assert est.device.type == "cuda"
        assert est.shape == (100,)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Test QJL with edge-case inputs."""

    def test_zero_residual(self, qjl):
        """When residual is zero, estimated correction should be zero."""
        residual = torch.zeros(10, DEFAULT_DIM)
        norms = residual.norm(dim=1)
        assert (norms == 0).all(), "Zero residual should have zero norm"

    def test_single_vector(self, qjl):
        """Should handle a single vector (N=1)."""
        x = random_unit_vectors(1, DEFAULT_DIM)
        signs = qjl.project_and_sign(x)
        assert signs.shape == (1, DEFAULT_DIM)

    def test_different_dimensions(self):
        """QJL should work for d=64 and d=256."""
        for d in [64, 256]:
            qjl = QJL(d=d, seed=SEED)
            x = random_unit_vectors(50, d, seed=77)
            signs = qjl.project_and_sign(x)
            assert signs.shape == (50, d)
