"""Tests for Triton fused kernels.

Validates that TritonTurboQuant produces results matching the PyTorch
TurboQuantEstimator baseline, across various dimensions, bit-widths,
and batch sizes. Also benchmarks speedup.
"""

import math

import pytest
import torch

from turboquantdc import TurboQuantEstimator
from turboquantdc.triton_kernels import (
    TritonTurboQuant,
    triton_dequantize,
    triton_inner_product,
    triton_quantize,
)


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pytorch_estimator(d, bits):
    return TurboQuantEstimator(d=d, bits=bits, device=DEVICE)


def make_triton_estimator(d, bits):
    return TritonTurboQuant(d=d, bits=bits, device=DEVICE)


# ---------------------------------------------------------------------------
# Correctness Tests
# ---------------------------------------------------------------------------


class TestTritonQuantizeMatchesPyTorch:
    """Triton quantize output matches TurboQuantEstimator.quantize()."""

    @pytest.mark.parametrize("d", [64, 128, 256])
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_indices_match_exactly(self, d, bits):
        """MSE indices from Triton must exactly match PyTorch."""
        torch.manual_seed(42)
        x = torch.randn(100, d, device=DEVICE)

        est = make_pytorch_estimator(d, bits)
        tri = make_triton_estimator(d, bits)

        c_pt = est.quantize(x)
        c_tri = tri.quantize(x)

        match_rate = (c_pt["mse_indices"] == c_tri["mse_indices"]).float().mean()
        assert match_rate == 1.0, f"Index match rate: {match_rate:.4f}"

    @pytest.mark.parametrize("d", [64, 128, 256])
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_signs_match_exactly(self, d, bits):
        """QJL sign bits from Triton must exactly match PyTorch."""
        torch.manual_seed(42)
        x = torch.randn(100, d, device=DEVICE)

        est = make_pytorch_estimator(d, bits)
        tri = make_triton_estimator(d, bits)

        c_pt = est.quantize(x)
        c_tri = tri.quantize(x)

        match_rate = (c_pt["qjl_signs"] == c_tri["qjl_signs"]).float().mean()
        assert match_rate == 1.0, f"Sign match rate: {match_rate:.4f}"

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_residual_norms_close(self, d):
        """Residual norms must be within floating-point tolerance."""
        torch.manual_seed(42)
        x = torch.randn(100, d, device=DEVICE)

        est = make_pytorch_estimator(d, 3)
        tri = make_triton_estimator(d, 3)

        c_pt = est.quantize(x)
        c_tri = tri.quantize(x)

        max_diff = (c_pt["residual_norm"] - c_tri["residual_norm"]).abs().max()
        assert max_diff < 1e-5, f"Residual norm max diff: {max_diff:.8f}"

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_vec_norms_identical(self, d):
        """Vector norms must be identical (both use PyTorch norm)."""
        torch.manual_seed(42)
        x = torch.randn(100, d, device=DEVICE)

        est = make_pytorch_estimator(d, 3)
        tri = make_triton_estimator(d, 3)

        c_pt = est.quantize(x)
        c_tri = tri.quantize(x)

        max_diff = (c_pt["vec_norm"] - c_tri["vec_norm"]).abs().max()
        assert max_diff < 1e-7, f"Vec norm max diff: {max_diff:.10f}"


class TestTritonInnerProductMatchesPyTorch:
    """Triton inner product matches TurboQuantEstimator.inner_product()."""

    @pytest.mark.parametrize("d", [64, 128, 256])
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_ip_relative_error(self, d, bits):
        """Inner product relative error must be < 1e-5."""
        torch.manual_seed(42)
        n_keys = 50
        n_queries = 4
        x = torch.randn(n_keys, d, device=DEVICE)
        q = torch.randn(n_queries, d, device=DEVICE)

        est = make_pytorch_estimator(d, bits)
        tri = make_triton_estimator(d, bits)

        c_pt = est.quantize(x)
        c_tri = tri.quantize(x)

        ip_pt = est.inner_product(q, c_pt)
        ip_tri = tri.inner_product(q, c_tri)

        rel_err = (ip_pt - ip_tri).abs().max() / (ip_pt.abs().max() + 1e-8)
        assert rel_err < 1e-5, f"IP relative error: {rel_err:.2e}"

    def test_ip_single_query_single_key(self):
        """Works with 1D query and single compressed key."""
        torch.manual_seed(42)
        d, bits = 128, 3

        est = make_pytorch_estimator(d, bits)
        tri = make_triton_estimator(d, bits)

        x = torch.randn(d, device=DEVICE)
        q = torch.randn(d, device=DEVICE)

        c_pt = est.quantize(x)
        c_tri = tri.quantize(x)

        ip_pt = est.inner_product(q, c_pt)
        ip_tri = tri.inner_product(q, c_tri)

        assert ip_pt.shape == ip_tri.shape
        rel_err = (ip_pt - ip_tri).abs() / (ip_pt.abs() + 1e-8)
        assert rel_err < 1e-5

    def test_ip_multi_query_multi_key(self):
        """Works with batched queries and batched keys."""
        torch.manual_seed(42)
        d, bits = 128, 3
        n_q, n_k = 8, 200

        est = make_pytorch_estimator(d, bits)
        tri = make_triton_estimator(d, bits)

        x = torch.randn(n_k, d, device=DEVICE)
        q = torch.randn(n_q, d, device=DEVICE)

        c_pt = est.quantize(x)
        c_tri = tri.quantize(x)

        ip_pt = est.inner_product(q, c_pt)
        ip_tri = tri.inner_product(q, c_tri)

        assert ip_pt.shape == (n_q, n_k)
        assert ip_tri.shape == (n_q, n_k)
        rel_err = (ip_pt - ip_tri).abs().max() / (ip_pt.abs().max() + 1e-8)
        assert rel_err < 1e-5


class TestTritonDequantizeMatchesPyTorch:
    """Triton dequantize matches TurboQuantEstimator.dequantize_mse()."""

    @pytest.mark.parametrize("d", [64, 128, 256])
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_dequantize_relative_error(self, d, bits):
        """Dequantized vectors must have < 1e-5 relative error."""
        torch.manual_seed(42)
        x = torch.randn(100, d, device=DEVICE)

        est = make_pytorch_estimator(d, bits)
        tri = make_triton_estimator(d, bits)

        c_pt = est.quantize(x)
        c_tri = tri.quantize(x)

        deq_pt = est.dequantize_mse(c_pt)
        deq_tri = tri.dequantize_mse(c_tri)

        rel_err = (deq_pt - deq_tri).abs().max() / (deq_pt.abs().max() + 1e-8)
        assert rel_err < 1e-5, f"Dequantize relative error: {rel_err:.2e}"


class TestTritonDequantizeKernel:
    """Test the raw triton_dequantize kernel (not the PyTorch fallback)."""

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_triton_dequantize_kernel(self, d):
        """Raw Triton dequantize kernel matches PyTorch."""
        torch.manual_seed(42)
        bits = 3
        n = 50

        tri = make_triton_estimator(d, bits)
        est = make_pytorch_estimator(d, bits)

        x = torch.randn(n, d, device=DEVICE)
        c_tri = tri.quantize(x)

        # Use raw Triton kernel
        deq_triton = triton_dequantize(
            c_tri["mse_indices"],
            tri.centroids,
            tri.R,
            c_tri["vec_norm"].unsqueeze(0) if c_tri["vec_norm"].dim() == 0 else c_tri["vec_norm"],
        )

        # Compare with PyTorch dequantize
        c_pt = est.quantize(x)
        deq_pt = est.dequantize_mse(c_pt)

        rel_err = (deq_pt - deq_triton).abs().max() / (deq_pt.abs().max() + 1e-8)
        assert rel_err < 1e-4, f"Triton kernel dequantize error: {rel_err:.2e}"


# ---------------------------------------------------------------------------
# Various Dimensions
# ---------------------------------------------------------------------------


class TestTritonVariousDims:
    """Works correctly for d=64, 128, 256."""

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_roundtrip(self, d):
        torch.manual_seed(42)
        tri = make_triton_estimator(d, 3)
        x = torch.randn(50, d, device=DEVICE)
        c = tri.quantize(x)
        deq = tri.dequantize_mse(c)

        assert c["mse_indices"].shape == (50, d)
        assert c["qjl_signs"].shape == (50, d)  # m defaults to d
        assert c["residual_norm"].shape == (50,)
        assert deq.shape == (50, d)


# ---------------------------------------------------------------------------
# Various Bits
# ---------------------------------------------------------------------------


class TestTritonVariousBits:
    """Works correctly for bits=2, 3, 4."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_quantize_index_range(self, bits):
        """Indices must be in [0, 2^mse_bits) range."""
        torch.manual_seed(42)
        d = 128
        tri = make_triton_estimator(d, bits)
        x = torch.randn(100, d, device=DEVICE)
        c = tri.quantize(x)

        mse_bits = max(bits - 1, 1)
        max_idx = (1 << mse_bits) - 1
        assert c["mse_indices"].max() <= max_idx
        assert c["mse_indices"].min() >= 0


# ---------------------------------------------------------------------------
# Large Batch
# ---------------------------------------------------------------------------


class TestTritonLargeBatch:
    """Works with 10K+ vectors."""

    def test_quantize_10k(self):
        torch.manual_seed(42)
        d, bits = 128, 3
        n = 10000

        tri = make_triton_estimator(d, bits)
        est = make_pytorch_estimator(d, bits)

        x = torch.randn(n, d, device=DEVICE)

        c_tri = tri.quantize(x)
        c_pt = est.quantize(x)

        assert c_tri["mse_indices"].shape == (n, d)
        idx_match = (c_pt["mse_indices"] == c_tri["mse_indices"]).float().mean()
        assert idx_match == 1.0

    def test_inner_product_10k(self):
        torch.manual_seed(42)
        d, bits = 128, 3
        n = 10000

        tri = make_triton_estimator(d, bits)
        est = make_pytorch_estimator(d, bits)

        x = torch.randn(n, d, device=DEVICE)
        q = torch.randn(4, d, device=DEVICE)

        c_tri = tri.quantize(x)
        c_pt = est.quantize(x)

        ip_tri = tri.inner_product(q, c_tri)
        ip_pt = est.inner_product(q, c_pt)

        assert ip_tri.shape == (4, n)
        rel_err = (ip_pt - ip_tri).abs().max() / (ip_pt.abs().max() + 1e-8)
        assert rel_err < 1e-5

    def test_quantize_50k(self):
        """Stress test with 50K vectors."""
        torch.manual_seed(42)
        d, bits = 128, 3
        n = 50000

        tri = make_triton_estimator(d, bits)
        x = torch.randn(n, d, device=DEVICE)
        c = tri.quantize(x)

        assert c["mse_indices"].shape == (n, d)
        assert c["qjl_signs"].shape == (n, d)
        assert c["residual_norm"].shape == (n,)


# ---------------------------------------------------------------------------
# Quality Preserved
# ---------------------------------------------------------------------------


class TestTritonQualityPreserved:
    """Cosine similarity matches PyTorch version."""

    def test_cosine_similarity_quantize(self):
        """Cosine similarity of dequantized vectors matches PyTorch."""
        torch.manual_seed(42)
        d, bits = 128, 3
        n = 1000

        est = make_pytorch_estimator(d, bits)
        tri = make_triton_estimator(d, bits)

        x = torch.randn(n, d, device=DEVICE)

        c_pt = est.quantize(x)
        c_tri = tri.quantize(x)

        deq_pt = est.dequantize_mse(c_pt)
        deq_tri = tri.dequantize_mse(c_tri)

        # Cosine similarity between original and dequantized
        cos_pt = torch.nn.functional.cosine_similarity(x, deq_pt, dim=-1)
        cos_tri = torch.nn.functional.cosine_similarity(x, deq_tri, dim=-1)

        # Must be essentially identical
        cos_diff = (cos_pt - cos_tri).abs().max()
        assert cos_diff < 1e-5, f"Cosine similarity diff: {cos_diff:.2e}"

        # Both should be reasonable quality. For 3-bit total (2-bit MSE),
        # the paper reports ~0.94 cosine similarity.
        assert cos_pt.mean() > 0.90, f"PyTorch cosine sim: {cos_pt.mean():.4f}"
        assert cos_tri.mean() > 0.90, f"Triton cosine sim: {cos_tri.mean():.4f}"

    def test_inner_product_quality(self):
        """Estimated IPs should correlate strongly with true IPs."""
        torch.manual_seed(42)
        d, bits = 128, 3
        n = 2000  # Large enough for stable correlation estimates

        tri = make_triton_estimator(d, bits)

        keys = torch.randn(n, d, device=DEVICE)
        queries = torch.randn(10, d, device=DEVICE)

        # True inner products
        true_ip = queries @ keys.T  # (10, n)

        # Estimated inner products
        c = tri.quantize(keys)
        est_ip = tri.inner_product(queries, c)  # (10, n)

        # Correlation between true and estimated (per query).
        # For 3-bit total (2-bit MSE + 1-bit QJL), mean correlation ~0.93.
        # Individual queries can be lower due to variance; check mean.
        corrs = []
        for qi in range(10):
            corr = torch.corrcoef(torch.stack([true_ip[qi], est_ip[qi]]))[0, 1]
            corrs.append(corr.item())

        mean_corr = sum(corrs) / len(corrs)
        min_corr = min(corrs)
        assert mean_corr > 0.90, f"Mean correlation: {mean_corr:.4f}"
        assert min_corr > 0.85, f"Min correlation: {min_corr:.4f} (mean: {mean_corr:.4f})"


# ---------------------------------------------------------------------------
# Speed Tests
# ---------------------------------------------------------------------------


class TestTritonSpeedImprovement:
    """Triton should be faster than PyTorch for quantize."""

    def _cuda_benchmark(self, fn, warmup=20, iters=100):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def test_quantize_faster(self):
        """Triton quantize must be at least 1.2x faster."""
        d, bits = 128, 3
        n = 5000
        keys = torch.randn(n, d, device=DEVICE)

        est = make_pytorch_estimator(d, bits)
        tri = make_triton_estimator(d, bits)

        pt_ms = self._cuda_benchmark(lambda: est.quantize(keys))
        tri_ms = self._cuda_benchmark(lambda: tri.quantize(keys))

        speedup = pt_ms / tri_ms
        print(f"\nQuantize speedup ({n} vecs): {speedup:.1f}x (PT={pt_ms:.3f}ms, Tri={tri_ms:.3f}ms)")
        assert speedup > 1.2, f"Expected >= 1.2x speedup, got {speedup:.1f}x"

    def test_inner_product_faster(self):
        """Triton inner product must be at least 1.2x faster."""
        d, bits = 128, 3
        n = 5000
        keys = torch.randn(n, d, device=DEVICE)
        query = torch.randn(1, d, device=DEVICE)

        est = make_pytorch_estimator(d, bits)
        tri = make_triton_estimator(d, bits)

        c_pt = est.quantize(keys)
        c_tri = tri.quantize(keys)

        pt_ms = self._cuda_benchmark(lambda: est.inner_product(query, c_pt))
        tri_ms = self._cuda_benchmark(lambda: tri.inner_product(query, c_tri))

        speedup = pt_ms / tri_ms
        print(f"\nIP speedup (1q x {n}k): {speedup:.1f}x (PT={pt_ms:.3f}ms, Tri={tri_ms:.3f}ms)")
        assert speedup > 1.2, f"Expected >= 1.2x speedup, got {speedup:.1f}x"

    def test_end_to_end_faster(self):
        """Full pipeline (quantize + IP) must be faster."""
        d, bits = 128, 3
        n = 5000
        keys = torch.randn(n, d, device=DEVICE)
        query = torch.randn(1, d, device=DEVICE)

        est = make_pytorch_estimator(d, bits)
        tri = make_triton_estimator(d, bits)

        def pt_pipeline():
            c = est.quantize(keys)
            return est.inner_product(query, c)

        def tri_pipeline():
            c = tri.quantize(keys)
            return tri.inner_product(query, c)

        pt_ms = self._cuda_benchmark(pt_pipeline)
        tri_ms = self._cuda_benchmark(tri_pipeline)

        speedup = pt_ms / tri_ms
        print(f"\nE2E speedup ({n} keys): {speedup:.1f}x (PT={pt_ms:.3f}ms, Tri={tri_ms:.3f}ms)")
        assert speedup > 1.2, f"Expected >= 1.2x speedup, got {speedup:.1f}x"
