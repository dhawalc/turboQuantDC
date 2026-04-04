"""Tests for CUDA dequantize and WHT kernels.

Validates numerical correctness of raw CUDA kernels against PyTorch reference
and Triton kernels. Tests cover:
  - Dequantize MSE: centroid lookup + inverse rotation + rescale
  - Dequantize Residual: centroid + correction + inverse rotation + rescale
  - WHT forward and inverse for d=64 through d=2048
  - CUDATurboQuant drop-in wrapper
  - Backend fallback chain
"""

import math
import os

import pytest
import torch

# Ensure CUDA_HOME is set for JIT compilation
if "CUDA_HOME" not in os.environ:
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cuda_deq():
    """Load CUDA dequantize module."""
    from turboquantdc.cuda.build import load_dequantize
    mod = load_dequantize()
    if mod is None:
        pytest.skip("CUDA dequantize kernel failed to compile")
    return mod


@pytest.fixture(scope="module")
def cuda_wht():
    """Load CUDA WHT module."""
    from turboquantdc.cuda.build import load_wht
    mod = load_wht()
    if mod is None:
        pytest.skip("CUDA WHT kernel failed to compile")
    return mod


# ---------------------------------------------------------------------------
# Dequantize MSE tests
# ---------------------------------------------------------------------------


class TestDequantizeMSE:
    """Test CUDA dequantize MSE kernel correctness."""

    @pytest.mark.parametrize("d", [128, 256])
    @pytest.mark.parametrize("batch", [1, 10, 100, 1000])
    @pytest.mark.parametrize("n_centroids", [4, 8, 16])
    def test_matches_pytorch(self, cuda_deq, d, batch, n_centroids):
        """CUDA dequantize must match PyTorch gather+matmul."""
        torch.manual_seed(42)
        centroids = torch.randn(n_centroids, device="cuda")
        R = torch.linalg.qr(torch.randn(d, d))[0].cuda().contiguous()
        indices = torch.randint(0, n_centroids, (batch, d),
                                dtype=torch.int32, device="cuda")
        vec_norms = torch.randn(batch, device="cuda").abs() + 0.1

        out_cuda = cuda_deq.dequantize_mse(indices, centroids, R, vec_norms)

        # PyTorch reference
        y_hat = centroids[indices.long()]
        out_ref = (y_hat @ R) * vec_norms.unsqueeze(-1)

        torch.testing.assert_close(out_cuda, out_ref, atol=1e-5, rtol=1e-5)

    def test_identity_rotation(self, cuda_deq):
        """With identity rotation, output = centroids[indices] * norm."""
        d = 128
        centroids = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
        R = torch.eye(d, device="cuda").contiguous()
        indices = torch.tensor([[0, 1, 2, 3] * 32], dtype=torch.int32, device="cuda")
        vec_norms = torch.tensor([2.0], device="cuda")

        out = cuda_deq.dequantize_mse(indices, centroids, R, vec_norms)
        expected = centroids[indices.long()] * 2.0
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("d", [64, 96, 192])
    def test_generic_kernel_path(self, cuda_deq, d):
        """Non-template dimensions should use the generic kernel and still be correct."""
        batch = 50
        n_centroids = 8
        torch.manual_seed(42)
        centroids = torch.randn(n_centroids, device="cuda")
        R = torch.linalg.qr(torch.randn(d, d))[0].cuda().contiguous()
        indices = torch.randint(0, n_centroids, (batch, d),
                                dtype=torch.int32, device="cuda")
        vec_norms = torch.randn(batch, device="cuda").abs() + 0.1

        out_cuda = cuda_deq.dequantize_mse(indices, centroids, R, vec_norms)
        y_hat = centroids[indices.long()]
        out_ref = (y_hat @ R) * vec_norms.unsqueeze(-1)
        torch.testing.assert_close(out_cuda, out_ref, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Dequantize Residual tests
# ---------------------------------------------------------------------------


class TestDequantizeResidual:
    """Test CUDA dequantize residual kernel correctness."""

    @pytest.mark.parametrize("d", [128, 256])
    @pytest.mark.parametrize("batch", [1, 100, 1000])
    def test_matches_pytorch(self, cuda_deq, d, batch):
        """CUDA residual dequantize must match PyTorch reference."""
        torch.manual_seed(42)
        n_centroids = 8
        centroids = torch.randn(n_centroids, device="cuda")
        R = torch.linalg.qr(torch.randn(d, d))[0].cuda().contiguous()
        indices = torch.randint(0, n_centroids, (batch, d),
                                dtype=torch.int32, device="cuda")
        vec_norms = torch.randn(batch, device="cuda").abs() + 0.1
        res_signs = torch.where(
            torch.rand(batch, d, device="cuda") > 0.5, 1.0, -1.0)
        res_scale = torch.rand(batch, device="cuda") * 0.1

        out_cuda = cuda_deq.dequantize_residual(
            indices, centroids, R, vec_norms, res_signs, res_scale)

        y_hat = centroids[indices.long()]
        y_corr = y_hat + res_signs * res_scale.unsqueeze(-1)
        out_ref = (y_corr @ R) * vec_norms.unsqueeze(-1)

        torch.testing.assert_close(out_cuda, out_ref, atol=1e-5, rtol=1e-5)

    def test_zero_residual_equals_mse(self, cuda_deq):
        """With zero residual scale, result should equal MSE-only dequantize."""
        d = 128
        batch = 50
        n_centroids = 8
        torch.manual_seed(42)
        centroids = torch.randn(n_centroids, device="cuda")
        R = torch.linalg.qr(torch.randn(d, d))[0].cuda().contiguous()
        indices = torch.randint(0, n_centroids, (batch, d),
                                dtype=torch.int32, device="cuda")
        vec_norms = torch.randn(batch, device="cuda").abs() + 0.1
        res_signs = torch.ones(batch, d, device="cuda")
        res_scale = torch.zeros(batch, device="cuda")

        out_res = cuda_deq.dequantize_residual(
            indices, centroids, R, vec_norms, res_signs, res_scale)
        out_mse = cuda_deq.dequantize_mse(indices, centroids, R, vec_norms)

        torch.testing.assert_close(out_res, out_mse, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# WHT tests
# ---------------------------------------------------------------------------


class TestWHT:
    """Test CUDA Walsh-Hadamard Transform kernel correctness."""

    @pytest.mark.parametrize("d", [64, 128, 256, 512, 1024, 2048])
    def test_roundtrip(self, cuda_wht, d):
        """Forward + inverse WHT should recover the original vector."""
        batch = 100
        torch.manual_seed(42)
        from turboquantdc.rotation import generate_wht_rotation
        params = generate_wht_rotation(d, seed=42, device="cuda")
        signs = params["signs"]
        x = torch.randn(batch, d, device="cuda")

        y = cuda_wht.wht(x.contiguous(), signs.contiguous(), False)
        x_recovered = cuda_wht.wht(y.contiguous(), signs.contiguous(), True)

        torch.testing.assert_close(x_recovered, x, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("d", [64, 128, 256, 512])
    def test_matches_pytorch_reference(self, cuda_wht, d):
        """CUDA WHT must match PyTorch iterative butterfly."""
        batch = 100
        torch.manual_seed(42)
        from turboquantdc.rotation import generate_wht_rotation, apply_wht_rotation
        params = generate_wht_rotation(d, seed=42, device="cuda")
        signs = params["signs"]
        x = torch.randn(batch, d, device="cuda")

        out_cuda = cuda_wht.wht(x.contiguous(), signs.contiguous(), False)
        out_ref = apply_wht_rotation(x, params, inverse=False)

        torch.testing.assert_close(out_cuda, out_ref, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("d", [64, 128, 256, 512])
    def test_inverse_matches_pytorch(self, cuda_wht, d):
        """CUDA inverse WHT must match PyTorch reference."""
        batch = 100
        torch.manual_seed(42)
        from turboquantdc.rotation import generate_wht_rotation, apply_wht_rotation
        params = generate_wht_rotation(d, seed=42, device="cuda")
        signs = params["signs"]
        x = torch.randn(batch, d, device="cuda")

        out_cuda = cuda_wht.wht(x.contiguous(), signs.contiguous(), True)
        out_ref = apply_wht_rotation(x, params, inverse=True)

        torch.testing.assert_close(out_cuda, out_ref, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("d", [1024, 2048])
    def test_extended_dimensions_orthogonal(self, cuda_wht, d):
        """For extended d=1024/2048, verify WHT preserves vector norms (orthogonality)."""
        batch = 50
        torch.manual_seed(42)
        from turboquantdc.rotation import generate_wht_rotation
        params = generate_wht_rotation(d, seed=42, device="cuda")
        signs = params["signs"]
        x = torch.randn(batch, d, device="cuda")

        y = cuda_wht.wht(x.contiguous(), signs.contiguous(), False)

        # Orthogonal transform preserves norms
        x_norms = x.norm(dim=-1)
        y_norms = y.norm(dim=-1)
        torch.testing.assert_close(x_norms, y_norms, atol=1e-4, rtol=1e-4)

    def test_single_vector(self, cuda_wht):
        """Test with batch=1."""
        d = 128
        from turboquantdc.rotation import generate_wht_rotation
        params = generate_wht_rotation(d, seed=42, device="cuda")
        signs = params["signs"]
        x = torch.randn(1, d, device="cuda")

        y = cuda_wht.wht(x.contiguous(), signs.contiguous(), False)
        x_rec = cuda_wht.wht(y.contiguous(), signs.contiguous(), True)
        torch.testing.assert_close(x_rec, x, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# CUDATurboQuant wrapper tests
# ---------------------------------------------------------------------------


class TestCUDATurboQuant:
    """Test CUDATurboQuant drop-in replacement."""

    def test_dequantize_mse_api(self):
        """CUDATurboQuant.dequantize_mse should produce valid results."""
        from turboquantdc.cuda_kernels import CUDATurboQuant
        tq = CUDATurboQuant(d=128, bits=3, device="cuda")

        x = torch.randn(50, 128, device="cuda")
        compressed = tq.quantize(x)
        x_hat = tq.dequantize_mse(compressed)

        assert x_hat.shape == x.shape
        # Cosine similarity should be reasonable for 3-bit
        cos_sim = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean()
        assert cos_sim > 0.8, f"Cosine similarity too low: {cos_sim:.4f}"

    def test_backend_detection(self):
        """CUDATurboQuant should report the correct backend."""
        from turboquantdc.cuda_kernels import CUDATurboQuant
        tq = CUDATurboQuant(d=128, bits=3, device="cuda")
        assert tq.backend in ("cuda", "triton", "pytorch")

    def test_direct_dequantize_api(self):
        """CUDATurboQuant.dequantize should work with explicit arguments."""
        from turboquantdc.cuda_kernels import CUDATurboQuant
        tq = CUDATurboQuant(d=128, bits=3, device="cuda")

        batch = 20
        d = 128
        n_centroids = tq.n_centroids
        indices = torch.randint(0, n_centroids, (batch, d),
                                dtype=torch.int32, device="cuda")
        vec_norms = torch.ones(batch, device="cuda")

        out = tq.dequantize(indices, tq.centroids, tq.R, vec_norms)
        assert out.shape == (batch, d)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Backend fallback tests
# ---------------------------------------------------------------------------


class TestBackendFallback:
    """Test that cuda_kernels falls back gracefully."""

    def test_is_cuda_available(self):
        from turboquantdc.cuda_kernels import is_cuda_available
        # Should be True since we are on a CUDA machine with compilation working
        assert is_cuda_available() is True

    def test_is_cuda_wht_available(self):
        from turboquantdc.cuda_kernels import is_cuda_wht_available
        assert is_cuda_wht_available() is True

    def test_pytorch_fallback_dequantize(self):
        """PyTorch fallback should produce correct results."""
        from turboquantdc.cuda_kernels import _pytorch_dequantize
        d = 128
        n_centroids = 8
        batch = 10
        torch.manual_seed(42)
        centroids = torch.randn(n_centroids, device="cuda")
        R = torch.linalg.qr(torch.randn(d, d))[0].cuda().contiguous()
        indices = torch.randint(0, n_centroids, (batch, d),
                                dtype=torch.int32, device="cuda")
        vec_norms = torch.ones(batch, device="cuda")

        out = _pytorch_dequantize(indices, centroids, R, vec_norms)
        y_hat = centroids[indices.long()]
        expected = (y_hat @ R) * vec_norms.unsqueeze(-1)
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    def test_pytorch_fallback_wht(self):
        """PyTorch WHT fallback should match the rotation module."""
        from turboquantdc.cuda_kernels import _pytorch_wht
        from turboquantdc.rotation import generate_wht_rotation, apply_wht_rotation
        d = 128
        params = generate_wht_rotation(d, seed=42, device="cuda")
        signs = params["signs"]
        x = torch.randn(10, d, device="cuda")

        out = _pytorch_wht(x, signs, inverse=False)
        ref = apply_wht_rotation(x, params, inverse=False)
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)
