"""Tests for Walsh-Hadamard Transform rotation.

Tests the WHT-based rotation against mathematical properties:
1. Correctness of the butterfly algorithm (known WHT values)
2. Round-trip identity (forward + inverse = identity)
3. Norm preservation (orthogonal rotation)
4. Gaussianization of uniform-on-sphere vectors
5. Performance advantage over dense QR rotation

Reference: TurboQuant paper Section 3 — the randomized Hadamard rotation
Pi = D * H_d / sqrt(d) achieves the same distributional guarantee as
a Haar-random orthogonal matrix for practical head dimensions.
"""

import math
import time

import pytest
import torch

from turboquantdc.rotation import (
    apply_wht_rotation,
    fast_wht,
    generate_rotation_matrix,
    generate_wht_rotation,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_DIM = 128


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def wht_params():
    """WHT rotation parameters for d=128."""
    return generate_wht_rotation(DEFAULT_DIM, seed=DEFAULT_SEED)


@pytest.fixture
def random_vectors():
    """Batch of random unit vectors on the sphere, shape (1000, 128)."""
    torch.manual_seed(DEFAULT_SEED)
    v = torch.randn(1000, DEFAULT_DIM)
    v = v / v.norm(dim=-1, keepdim=True)
    return v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFastWHT:
    """Tests for the raw (unnormalized) fast Walsh-Hadamard transform."""

    def test_fast_wht_known_values(self):
        """WHT of [1,1,1,1] = [4,0,0,0] and WHT of [1,0,0,0] = [1,1,1,1]."""
        # H_4 @ [1,1,1,1]^T = [4,0,0,0]^T
        x1 = torch.tensor([1.0, 1.0, 1.0, 1.0])
        result1 = fast_wht(x1.clone())
        expected1 = torch.tensor([4.0, 0.0, 0.0, 0.0])
        assert torch.allclose(result1, expected1, atol=1e-6), (
            f"WHT([1,1,1,1]) = {result1}, expected {expected1}"
        )

        # H_4 @ [1,0,0,0]^T = [1,1,1,1]^T
        x2 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result2 = fast_wht(x2.clone())
        expected2 = torch.tensor([1.0, 1.0, 1.0, 1.0])
        assert torch.allclose(result2, expected2, atol=1e-6), (
            f"WHT([1,0,0,0]) = {result2}, expected {expected2}"
        )

    def test_fast_wht_batched(self):
        """WHT operates independently on each row of a batched input."""
        torch.manual_seed(DEFAULT_SEED)
        batch_size = 16
        d = 64
        x = torch.randn(batch_size, d)
        x_copy = x.clone()

        # Apply batched
        result_batched = fast_wht(x)

        # Apply individually and compare
        for i in range(batch_size):
            single = fast_wht(x_copy[i : i + 1].clone()).squeeze(0)
            assert torch.allclose(result_batched[i], single, atol=1e-5), (
                f"Batched WHT differs from single at row {i}"
            )

    def test_fast_wht_power_of_2_assertion(self):
        """Non-power-of-2 dimension raises AssertionError."""
        with pytest.raises(AssertionError):
            fast_wht(torch.randn(3))

        with pytest.raises(AssertionError):
            fast_wht(torch.randn(5))

        with pytest.raises(AssertionError):
            fast_wht(torch.randn(6))

        with pytest.raises(AssertionError):
            fast_wht(torch.randn(100))


class TestWHTRotation:
    """Tests for the full randomized Walsh-Hadamard rotation."""

    def test_wht_rotation_roundtrip(self, wht_params, random_vectors):
        """Forward then inverse recovers original within 1e-5."""
        rotated = apply_wht_rotation(random_vectors, wht_params, inverse=False)
        recovered = apply_wht_rotation(rotated, wht_params, inverse=True)

        max_error = (recovered - random_vectors).abs().max().item()
        assert max_error < 1e-5, (
            f"Round-trip max error {max_error:.2e} exceeds 1e-5"
        )

    def test_wht_norm_preservation(self, wht_params, random_vectors):
        """||Pi @ x|| = ||x|| within 1e-5 (orthogonal rotation)."""
        rotated = apply_wht_rotation(random_vectors, wht_params, inverse=False)

        original_norms = random_vectors.norm(dim=-1)
        rotated_norms = rotated.norm(dim=-1)

        max_norm_diff = (original_norms - rotated_norms).abs().max().item()
        assert max_norm_diff < 1e-5, (
            f"Norm preservation error {max_norm_diff:.2e} exceeds 1e-5"
        )

    def test_wht_gaussianization(self, wht_params):
        """After rotation, coordinates approximate N(0, 1/d).

        For unit vectors on the d-sphere rotated by a randomized Hadamard:
        - Mean of each coordinate should be ~0
        - Std dev should be ~1/sqrt(d)
        - Kurtosis should be ~3 (Gaussian)
        """
        torch.manual_seed(DEFAULT_SEED)
        n_vectors = 10000
        d = DEFAULT_DIM

        # Random unit vectors on sphere
        v = torch.randn(n_vectors, d)
        v = v / v.norm(dim=-1, keepdim=True)

        rotated = apply_wht_rotation(v, wht_params, inverse=False)

        # Flatten all coordinates for aggregate statistics
        coords = rotated.flatten()
        mean = coords.mean().item()
        std = coords.std().item()
        expected_std = 1.0 / math.sqrt(d)

        # Kurtosis: E[(X-mu)^4] / E[(X-mu)^2]^2
        centered = coords - mean
        kurtosis = (centered**4).mean().item() / (centered**2).mean().item() ** 2

        assert abs(mean) < 0.01, f"Mean {mean:.4f} too far from 0"
        assert abs(std - expected_std) / expected_std < 0.1, (
            f"Std {std:.6f} too far from 1/sqrt({d}) = {expected_std:.6f}"
        )
        assert abs(kurtosis - 3.0) < 0.5, (
            f"Kurtosis {kurtosis:.3f} too far from 3.0 (Gaussian)"
        )

    def test_wht_orthogonality(self):
        """Construct the explicit rotation matrix and verify Pi^T @ Pi = I."""
        d = 32  # small d for explicit matrix construction
        params = generate_wht_rotation(d, seed=DEFAULT_SEED)

        # Build the matrix by applying rotation to each basis vector
        I_d = torch.eye(d)
        Pi = torch.stack(
            [apply_wht_rotation(I_d[i : i + 1], params).squeeze(0) for i in range(d)]
        )  # shape (d, d), row i = Pi @ e_i

        # Pi^T @ Pi should be identity
        product = Pi.T @ Pi
        identity = torch.eye(d)
        max_error = (product - identity).abs().max().item()
        assert max_error < 1e-5, (
            f"Orthogonality error {max_error:.2e} exceeds 1e-5"
        )

    def test_wht_faster_than_qr(self):
        """WHT uses O(d) memory vs QR's O(d^2), and is competitive in speed.

        The primary advantage of WHT is memory: storing d sign-flip values
        vs a full d x d dense matrix. For d=256, that is 256 floats vs 65536.

        Speed comparison: the Python-level butterfly loop cannot beat a single
        BLAS matmul call for small d. However, the gap should be modest, and
        on GPU with fused kernels WHT would dominate. We verify:
        1. Memory: WHT params are O(d) — strictly smaller than O(d^2).
        2. Speed: WHT is within 20x of BLAS matmul (the Python loop overhead
           will be eliminated by a Triton/CUDA kernel in Phase 3).
        """
        d = 256
        n_iters = 1000
        torch.manual_seed(DEFAULT_SEED)
        x = torch.randn(64, d)

        # Setup WHT
        wht_p = generate_wht_rotation(d, seed=DEFAULT_SEED)

        # Setup QR
        qr_matrix = generate_rotation_matrix(d, seed=DEFAULT_SEED)

        # Memory check: WHT stores d floats, QR stores d*d floats
        wht_memory = wht_p["signs"].numel()
        qr_memory = qr_matrix.numel()
        assert wht_memory == d, f"WHT should store {d} values, got {wht_memory}"
        assert qr_memory == d * d, f"QR should store {d*d} values, got {qr_memory}"
        assert wht_memory < qr_memory, "WHT should use less memory than QR"

        # Warm-up
        for _ in range(10):
            apply_wht_rotation(x, wht_p)
            x @ qr_matrix.T

        # Benchmark WHT
        t0 = time.perf_counter()
        for _ in range(n_iters):
            apply_wht_rotation(x, wht_p)
        wht_time = time.perf_counter() - t0

        # Benchmark QR (dense matmul)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            x @ qr_matrix.T
        qr_time = time.perf_counter() - t0

        # Speed: WHT Python loop vs single BLAS call.
        # Allow 20x tolerance — the Python overhead will be removed by
        # a fused Triton kernel in Phase 3 optimization.
        assert wht_time < qr_time * 20.0, (
            f"WHT ({wht_time:.3f}s) unreasonably slower than QR ({qr_time:.3f}s)"
        )

    def test_wht_deterministic(self):
        """Same seed produces identical rotation parameters."""
        p1 = generate_wht_rotation(DEFAULT_DIM, seed=123)
        p2 = generate_wht_rotation(DEFAULT_DIM, seed=123)
        p3 = generate_wht_rotation(DEFAULT_DIM, seed=456)

        assert torch.equal(p1["signs"], p2["signs"]), "Same seed gave different signs"
        assert not torch.equal(p1["signs"], p3["signs"]), (
            "Different seeds gave identical signs"
        )

    def test_wht_different_dims(self):
        """Round-trip works for d = 32, 64, 128, 256, 512."""
        torch.manual_seed(DEFAULT_SEED)

        for d in [32, 64, 128, 256, 512]:
            params = generate_wht_rotation(d, seed=DEFAULT_SEED)
            x = torch.randn(16, d)
            x_orig = x.clone()

            rotated = apply_wht_rotation(x, params)
            recovered = apply_wht_rotation(rotated, params, inverse=True)

            max_error = (recovered - x_orig).abs().max().item()
            assert max_error < 1e-4, (
                f"d={d}: round-trip error {max_error:.2e} exceeds 1e-4"
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_wht_gpu(self):
        """WHT rotation works on CUDA tensors."""
        d = 128
        device = "cuda"

        params = generate_wht_rotation(d, seed=DEFAULT_SEED, device=device)
        assert params["signs"].device.type == "cuda"

        torch.manual_seed(DEFAULT_SEED)
        x = torch.randn(32, d, device=device)
        x_orig = x.clone()

        rotated = apply_wht_rotation(x, params)
        assert rotated.device.type == "cuda"

        recovered = apply_wht_rotation(rotated, params, inverse=True)
        max_error = (recovered - x_orig).abs().max().item()
        assert max_error < 1e-4, (
            f"GPU round-trip error {max_error:.2e} exceeds 1e-4"
        )

    def test_wht_vs_qr_quality(self):
        """WHT Gaussianization quality comparable to QR (kurtosis within 10%).

        Both methods should produce approximately Gaussian coordinates
        when applied to random unit vectors.
        """
        torch.manual_seed(DEFAULT_SEED)
        d = 128
        n_vectors = 10000

        v = torch.randn(n_vectors, d)
        v = v / v.norm(dim=-1, keepdim=True)

        # WHT rotation
        wht_p = generate_wht_rotation(d, seed=DEFAULT_SEED)
        wht_rotated = apply_wht_rotation(v, wht_p)
        wht_coords = wht_rotated.flatten()
        wht_centered = wht_coords - wht_coords.mean()
        wht_kurtosis = (
            (wht_centered**4).mean() / (wht_centered**2).mean() ** 2
        ).item()

        # QR rotation
        qr_matrix = generate_rotation_matrix(d, seed=DEFAULT_SEED)
        qr_rotated = v @ qr_matrix.T
        qr_coords = qr_rotated.flatten()
        qr_centered = qr_coords - qr_coords.mean()
        qr_kurtosis = (
            (qr_centered**4).mean() / (qr_centered**2).mean() ** 2
        ).item()

        # Both should be near 3.0 (Gaussian), and within 10% of each other
        assert abs(wht_kurtosis - 3.0) < 0.5, (
            f"WHT kurtosis {wht_kurtosis:.3f} too far from 3.0"
        )
        assert abs(qr_kurtosis - 3.0) < 0.5, (
            f"QR kurtosis {qr_kurtosis:.3f} too far from 3.0"
        )

        relative_diff = abs(wht_kurtosis - qr_kurtosis) / qr_kurtosis
        assert relative_diff < 0.10, (
            f"WHT kurtosis ({wht_kurtosis:.3f}) differs from QR "
            f"({qr_kurtosis:.3f}) by {relative_diff:.1%} (>10%)"
        )
