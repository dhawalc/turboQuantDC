"""Temporal delta coding for within-layer KV cache compression.

EXPERIMENTAL STATUS: Marginal. Key-only, requires anchoring.

Empirical findings from Qwen2.5-3B-Instruct (815 tokens, 36 layers):

    Key temporal cosine similarity:   0.80 (strongly correlated)
    Value temporal cosine similarity: 0.37 (weakly correlated)
    Key delta variance ratio:         0.39 (below 0.5 threshold -- viable)
    Value delta variance ratio:       1.26 (above 0.5 -- NOT viable)

The approach: store the first token's KV at full precision (anchor), then
store only the delta (difference) for subsequent tokens. Deltas have lower
variance and can be quantized at fewer bits.

CRITICAL LIMITATION: Error accumulation. Reconstructing via cumulative sum
causes quantization noise to grow as O(sqrt(T)). At layer 12 with 4-bit
deltas, reconstruction error reaches 2.4x the initial error by position 815.
This requires periodic anchors (full-precision tokens inserted every W
positions) to bound error growth, which reduces the compression advantage.

The trilemma:
  1. Low delta bits -> good compression, but error explodes
  2. Small anchor window -> bounded error, but poor compression
  3. High delta bits -> good quality, but no compression advantage

At operating points where quality matches TurboQuant 3-bit, this approach
offers similar or worse compression ratios. Included for completeness and
potential future improvements (e.g., learned predictors, entropy coding).

Reference: benchmarks/results/temporal_delta_results.md for full experiment data.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Uniform quantization utilities for deltas
# ---------------------------------------------------------------------------

def quantize_delta_uniform(
    delta: torch.Tensor,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize delta vectors using per-vector symmetric uniform quantization.

    Each vector gets its own scale factor (max absolute value), which provides
    much better quality than a single global scale at the cost of 16 bits per
    vector for the scale.

    Args:
        delta: Input tensor of shape (..., d).
        bits: Bits per coordinate (1-8).

    Returns:
        Tuple of (indices as int8, scale tensor of shape (..., 1)).
    """
    qmax = 2 ** (bits - 1) - 1
    # Per-vector scale: max absolute value along last dimension
    scale = delta.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) / qmax
    indices = torch.round(delta / scale).clamp(-qmax - 1, qmax).to(torch.int8)
    return indices, scale


def dequantize_delta_uniform(
    indices: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize delta vectors.

    Args:
        indices: Quantized indices from quantize_delta_uniform.
        scale: Per-vector scale factors.

    Returns:
        Reconstructed delta tensor (float32).
    """
    return indices.float() * scale


# ---------------------------------------------------------------------------
# Temporal Delta Encoder
# ---------------------------------------------------------------------------

class TemporalDeltaEncoder:
    """Encodes a sequence of vectors using anchored temporal delta coding.

    Storage format for a sequence of T vectors with anchor window W:
        - Positions 0, W, 2W, ... : full-precision anchors (FP16)
        - All other positions: quantized deltas from the previous position

    The anchor window W controls the error/compression tradeoff:
        - W=inf: pure delta coding, maximum compression, unbounded error
        - W=1:   pure anchor coding, no compression, zero error
        - W=16-32: practical sweet spot

    Error bound: O(sqrt(W) * delta_quantization_noise)

    This encoder is designed for KEYS ONLY. Values have temporal variance
    ratio > 1.0 (deltas are LARGER than absolutes) and should NOT be
    delta-coded.

    Args:
        delta_bits: Bits per coordinate for delta quantization. Default 2.
        anchor_window: Number of positions between anchors. Default 32.
    """

    def __init__(
        self,
        delta_bits: int = 2,
        anchor_window: int = 32,
    ):
        if delta_bits < 1 or delta_bits > 8:
            raise ValueError(f"delta_bits must be 1-8, got {delta_bits}")
        if anchor_window < 2:
            raise ValueError(f"anchor_window must be >= 2, got {anchor_window}")

        self.delta_bits = delta_bits
        self.anchor_window = anchor_window

    def encode(
        self,
        vectors: torch.Tensor,
    ) -> Dict[str, Any]:
        """Encode a sequence of vectors using anchored delta coding.

        Args:
            vectors: Input tensor of shape (seq_len, d) or (n_heads, seq_len, d).

        Returns:
            Encoded representation dict with:
                - anchors: List of (position, vector) tuples
                - deltas: List of (position, indices, scale) tuples
                - shape: Original tensor shape
                - delta_bits: Bit-width used for deltas
                - anchor_window: Window size
        """
        if vectors.dim() == 2:
            vectors = vectors.unsqueeze(0)  # Add head dimension
            squeeze_output = True
        else:
            squeeze_output = False

        n_heads, seq_len, d = vectors.shape
        anchors = []
        deltas = []

        for t in range(seq_len):
            if t % self.anchor_window == 0:
                # Anchor: store at full precision (FP16)
                anchors.append((t, vectors[:, t, :].half()))
            else:
                # Delta: quantize difference from previous position
                delta = vectors[:, t, :] - vectors[:, t - 1, :]
                indices, scale = quantize_delta_uniform(delta, self.delta_bits)
                deltas.append((t, indices, scale))

        return {
            "anchors": anchors,
            "deltas": deltas,
            "shape": (n_heads, seq_len, d),
            "delta_bits": self.delta_bits,
            "anchor_window": self.anchor_window,
            "_squeeze": squeeze_output,
        }

    def decode(
        self,
        encoded: Dict[str, Any],
    ) -> torch.Tensor:
        """Decode anchored delta representation back to full vectors.

        Reconstruction proceeds within each anchor window:
          1. Load the anchor vector (FP16 -> FP32)
          2. Cumulatively add dequantized deltas

        Args:
            encoded: Output from encode().

        Returns:
            Reconstructed tensor matching original shape.
        """
        n_heads, seq_len, d = encoded["shape"]
        device = encoded["anchors"][0][1].device if encoded["anchors"] else "cpu"
        result = torch.zeros(n_heads, seq_len, d, device=device)

        # Build lookup for fast access
        anchor_map = {pos: vec.float() for pos, vec in encoded["anchors"]}
        delta_map = {pos: (idx, scale) for pos, idx, scale in encoded["deltas"]}

        for t in range(seq_len):
            if t in anchor_map:
                result[:, t, :] = anchor_map[t]
            elif t in delta_map:
                indices, scale = delta_map[t]
                dq_delta = dequantize_delta_uniform(indices, scale)
                result[:, t, :] = result[:, t - 1, :] + dq_delta
            else:
                raise ValueError(f"Position {t} not found in anchors or deltas")

        if encoded.get("_squeeze", False):
            result = result.squeeze(0)

        return result

    def size_bits(self, encoded: Dict[str, Any]) -> Dict[str, int]:
        """Compute storage size in bits.

        Returns:
            Dict with anchor_bits, delta_bits_total, scale_bits, total_bits,
            fp16_baseline_bits, and compression_ratio.
        """
        n_heads, seq_len, d = encoded["shape"]

        n_anchors = len(encoded["anchors"])
        n_deltas = len(encoded["deltas"])

        # Anchors: FP16 = 16 bits per coordinate
        anchor_bits = n_anchors * n_heads * d * 16
        # Deltas: delta_bits per coordinate + 16 bits per scale (per vector per head)
        delta_data_bits = n_deltas * n_heads * d * self.delta_bits
        scale_bits = n_deltas * n_heads * 16  # FP16 scale per vector per head
        total_bits = anchor_bits + delta_data_bits + scale_bits

        fp16_baseline = seq_len * n_heads * d * 16

        return {
            "anchor_bits": anchor_bits,
            "delta_data_bits": delta_data_bits,
            "scale_bits": scale_bits,
            "total_bits": total_bits,
            "fp16_baseline_bits": fp16_baseline,
            "compression_ratio": fp16_baseline / total_bits if total_bits > 0 else 0.0,
            "effective_bits_per_coord": total_bits / (seq_len * n_heads * d) if seq_len > 0 else 0.0,
            "n_anchors": n_anchors,
            "n_deltas": n_deltas,
        }

    def reconstruction_quality(
        self,
        original: torch.Tensor,
        encoded: Dict[str, Any],
    ) -> Dict[str, float]:
        """Measure reconstruction quality.

        Args:
            original: Original vectors (same shape as input to encode).
            encoded: Output from encode().

        Returns:
            Dict with cosine_similarity, relative_mse, max_relative_error.
        """
        if original.dim() == 2:
            original = original.unsqueeze(0)

        reconstructed = self.decode(encoded)
        if reconstructed.dim() == 2:
            reconstructed = reconstructed.unsqueeze(0)

        # Flatten to (N, d) for metrics
        orig_flat = original.reshape(-1, original.shape[-1])
        recon_flat = reconstructed.reshape(-1, reconstructed.shape[-1])

        cos_sim = torch.nn.functional.cosine_similarity(
            orig_flat, recon_flat, dim=-1
        ).mean().item()

        errors = (orig_flat - recon_flat).norm(dim=-1)
        norms = orig_flat.norm(dim=-1).clamp(min=1e-10)
        rel_errors = errors / norms

        return {
            "cosine_similarity": cos_sim,
            "relative_mse": (rel_errors ** 2).mean().item(),
            "max_relative_error": rel_errors.max().item(),
            "mean_relative_error": rel_errors.mean().item(),
        }


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------

def compute_temporal_stats(
    vectors: torch.Tensor,
) -> Dict[str, float]:
    """Compute temporal redundancy statistics for a sequence of vectors.

    Useful for deciding whether temporal delta coding is viable for a
    given set of vectors (keys or values from a specific layer).

    Args:
        vectors: Shape (seq_len, d) or (n_heads, seq_len, d).

    Returns:
        Dict with temporal_cosine, temporal_pearson, delta_variance_ratio,
        delta_l2_ratio, delta_sparsity_10pct.
    """
    if vectors.dim() == 2:
        vectors = vectors.unsqueeze(0)

    n_heads, seq_len, d = vectors.shape
    if seq_len < 2:
        return {
            "temporal_cosine": 0.0,
            "temporal_pearson": 0.0,
            "delta_variance_ratio": float("inf"),
            "delta_l2_ratio": float("inf"),
            "delta_sparsity_10pct": 0.0,
            "viable": False,
        }

    t_curr = vectors[:, :-1, :]   # [n_heads, seq-1, d]
    t_next = vectors[:, 1:, :]

    delta = t_next - t_curr

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        t_curr.reshape(-1, d), t_next.reshape(-1, d), dim=-1
    ).mean().item()

    # Pearson correlation
    x = t_curr.reshape(-1, d)
    y = t_next.reshape(-1, d)
    x_c = x - x.mean(dim=-1, keepdim=True)
    y_c = y - y.mean(dim=-1, keepdim=True)
    pearson = (
        (x_c * y_c).sum(dim=-1) / (x_c.norm(dim=-1) * y_c.norm(dim=-1) + 1e-10)
    ).mean().item()

    # Variance ratio
    var_ratio = delta.var().item() / (t_next.var().item() + 1e-10)

    # L2 ratio
    l2_ratio = delta.norm(dim=-1).mean().item() / (t_next.norm(dim=-1).mean().item() + 1e-10)

    # Sparsity at 10% threshold
    abs_std = t_next.std().item()
    sparsity = (delta.abs() < 0.10 * abs_std).float().mean().item()

    return {
        "temporal_cosine": cos_sim,
        "temporal_pearson": pearson,
        "delta_variance_ratio": var_ratio,
        "delta_l2_ratio": l2_ratio,
        "delta_sparsity_10pct": sparsity,
        "viable": var_ratio < 0.5,
    }


def recommend_config(
    stats: Dict[str, float],
    target_quality: float = 0.99,
) -> Dict[str, Any]:
    """Recommend delta coding configuration based on temporal statistics.

    Args:
        stats: Output from compute_temporal_stats.
        target_quality: Target cosine similarity threshold.

    Returns:
        Dict with recommended delta_bits, anchor_window, or
        a recommendation to NOT use delta coding.
    """
    if not stats.get("viable", False):
        return {
            "recommendation": "DO_NOT_USE",
            "reason": f"Delta variance ratio {stats['delta_variance_ratio']:.3f} >= 0.5. "
                      f"Deltas are not sufficiently smaller than absolute values. "
                      f"Use standard TurboQuant quantization instead.",
        }

    var_ratio = stats["delta_variance_ratio"]

    # Heuristic: lower variance ratio -> can use fewer bits and wider windows
    if var_ratio < 0.1:
        delta_bits = 2
        anchor_window = 64
    elif var_ratio < 0.25:
        delta_bits = 2
        anchor_window = 32
    elif var_ratio < 0.4:
        delta_bits = 3
        anchor_window = 16
    else:
        delta_bits = 3
        anchor_window = 8

    # Effective bits per coordinate
    W = anchor_window
    eff_bits = (16 + (W - 1) * delta_bits) / W

    return {
        "recommendation": "USE_DELTA_CODING",
        "delta_bits": delta_bits,
        "anchor_window": anchor_window,
        "effective_bits_per_coord": eff_bits,
        "estimated_compression_vs_fp16": 16.0 / eff_bits,
        "note": "Apply to KEYS ONLY. Values have var_ratio > 1.0 and "
                "should use standard quantization.",
    }
