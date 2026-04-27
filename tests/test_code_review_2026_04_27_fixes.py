"""Tests for the CRITICAL bugs surfaced in the 2026-04-27 Opus 4.7 code review.

See `docs/code_review/2026-04-27/CODE_REVIEW_2026-04-27.md` for full context.

These tests describe each bug with a minimal reproducer, then verify the
fixed behavior. All tests should fail against pre-2026-04-27 code and pass
against the patched code.
"""

from __future__ import annotations

import pytest
import torch

from turboquantdc.e8_lattice import nearest_d8
from turboquantdc.estimator import TurboQuantEstimator
from turboquantdc.polarquant import PolarQuant


# ── Bug 1: e8_lattice.nearest_d8 picks argmin of margins (Reviewer #3 CRIT-1) ──
#
# Original code at e8_lattice.py:52 used `argmin` of |x - rounded| to choose
# which coord to flip when parity is odd. This is wrong: it picks the coord
# rounded most confidently (smallest margin), so flipping it adds maximum
# distortion. The correct choice is argmax (the most ambiguous coord). When
# input has zero coordinates, argmin can pick a zero-margin coord and flip
# it by ±1, producing a full unit of distortion.

def test_nearest_d8_bug_zero_margin_coord_should_not_be_flipped():
    """Reviewer #3: x=(0.7, 0.4, 0,...,0) under buggy code returns squared
    distance 1.25 (flips a zero-margin coord by ±1). Correct behavior should
    flip coord 1 (largest margin = 0.4) for distance 0.45."""
    x = torch.tensor([0.7, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = nearest_d8(x)
    distance = ((x - result) ** 2).sum().item()
    assert distance < 1.0, (
        f"nearest_d8 returned suboptimal flip (distance={distance:.3f}). "
        f"Expected ≤0.45 (flip the largest-margin coord). result={result.tolist()}"
    )


def test_nearest_d8_returns_even_sum_d8_lattice_point():
    """D8 lattice point invariant: sum of coordinates must be even."""
    torch.manual_seed(0)
    x = torch.randn(64, 8) * 0.6
    result = nearest_d8(x)
    sums = result.sum(dim=-1)
    is_even = ((sums.long() % 2) == 0).all()
    assert is_even, f"Some D8 results have odd sum: {sums[~((sums.long() % 2) == 0)][:5]}"


# ── Bug 2: estimator.py fp16 underflow on small norms (Reviewer #1 CRITICAL) ──
#
# Original code at estimator.py:99 used `vec_norm + 1e-8` to avoid div-by-zero.
# But `1e-8` rounds to 0 in fp16 (min normal is ~6e-5). For inputs with small
# norm in fp16 the divide can produce inf/nan. Fix: cast through fp32, or
# `vec_norm.clamp_min(1e-6)` (above fp16 min normal).

def test_estimator_quantize_fp16_small_norm_returns_finite():
    """Reviewer #1: fp16 input with small norm should not produce inf/nan.

    The bug: `x_normalized = x / (vec_norm + 1e-8)` in fp16, with vec_norm
    near 1e-5, produces values up to ~1e5 → fp16 overflow → inf → nan in
    the codebook quantize step.
    """
    torch.manual_seed(0)
    # Construct fp16 input where naive normalization would overflow
    x = torch.randn(4, 128, dtype=torch.float16) * 1e-3  # tiny vector
    estimator = TurboQuantEstimator(d=128, bits=3)
    result = estimator.quantize(x)
    # The MSE indices are integer; check the reconstructed vector is finite
    reconstructed = estimator.dequantize_mse(result)
    assert torch.isfinite(reconstructed).all(), (
        f"Reconstructed vector contains inf/nan: "
        f"{reconstructed.flatten()[:5].tolist()}"
    )
    # vec_norm should also be finite
    assert torch.isfinite(result["vec_norm"]).all()


def test_estimator_quantize_fp16_zero_input_is_safe():
    """Pathological input: an all-zero vector. Must not crash, must return
    a finite reconstruction (ideally zeros)."""
    x = torch.zeros(2, 128, dtype=torch.float16)
    estimator = TurboQuantEstimator(d=128, bits=3)
    result = estimator.quantize(x)
    reconstructed = estimator.dequantize_mse(result)
    assert torch.isfinite(reconstructed).all()


# ── Bug 4: polarquant.quantize doesn't enforce unit-norm contract (Reviewer #1) ──
#
# Original code at polarquant.py:144 silently saturates the codebook when
# input norm differs significantly from 1. The codebook centroids assume
# post-rotation Beta/Gaussian distribution with variance 1/d (from a unit
# vector). For non-unit vectors, every coordinate clips to one of the two
# outermost centroids — silent quality cliff.
#
# Fix: validate the unit-norm contract at runtime (either assertion or
# normalize internally and return the norm).

def test_polarquant_quantize_non_unit_input_does_not_saturate_silently():
    """Reviewer #1: Document and enforce the unit-vector contract on
    `PolarQuant.quantize`. Either reject non-unit input clearly, or
    normalize internally and document the change.

    Specifically, with a vector of norm 8 (8x unit), the bug version
    silently clips every coordinate to one of two extreme centroids —
    a uniform-distribution-of-clipped-values bug.
    """
    # Norm-8 vector (well outside unit-norm)
    torch.manual_seed(0)
    pq = PolarQuant(d=128, bits=3, rotation_type="wht")
    x_unit = torch.randn(1, 128)
    x_unit = x_unit / x_unit.norm(dim=-1, keepdim=True)
    x_big = x_unit * 8.0  # norm 8

    # Acceptable behaviors after fix:
    #   (a) Raise a clear error mentioning unit-norm contract.
    #   (b) Internally normalize and produce indices that are NOT all clipped
    #       (i.e., entropy of indices is > 1 bit, not 0/1 bit).
    try:
        idx = pq.quantize(x_big)
    except (ValueError, AssertionError, RuntimeError) as e:
        # Behavior (a): rejected. Acceptable.
        assert "norm" in str(e).lower() or "unit" in str(e).lower(), (
            f"Error raised but message doesn't mention unit/norm: {e}"
        )
        return

    # Behavior (b): didn't raise → must produce non-degenerate indices.
    # If every coordinate clips to one of 2 outermost values, we'd see
    # entropy ≈ 1 bit. Healthy 3-bit quantization gives entropy ≈ 3 bits.
    n_levels = 2 ** pq.bits
    counts = torch.bincount(idx.flatten(), minlength=n_levels).float()
    probs = counts / counts.sum()
    entropy_bits = -(probs * torch.log2(probs.clamp_min(1e-12))).sum().item()
    # Healthy: entropy > 1.5 bits (real spread). Saturated: entropy ≈ 1 bit.
    assert entropy_bits > 1.5, (
        f"Codebook saturated: entropy {entropy_bits:.2f} bits ≤ 1.5 means "
        f"PolarQuant.quantize silently clipped a non-unit-norm input. "
        f"Histogram: {counts.long().tolist()}"
    )
