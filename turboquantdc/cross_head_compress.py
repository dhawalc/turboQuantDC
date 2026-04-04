"""Cross-Head Delta Compression for KV caches.

Exploits inter-head redundancy: in GQA and MHA models, KV heads at the
same layer position tend to produce correlated vectors. Instead of
compressing each head independently, we compress one "anchor" head at
full fidelity (3-bit) and encode remaining heads as DELTAS from the
anchor at reduced bit-width (1-2 bits).

Theory:
    If inter-head correlation is high (>0.8), the delta vectors
    delta_h = kv_h - kv_anchor have much lower variance than the
    absolute vectors. This means fewer bits suffice for the delta.

    For N heads, anchor at b bits and deltas at d bits:
        effective_rate = (b + (N-1) * d) / N  bits/element

    Example: 8 heads, 3-bit anchor, 1-bit deltas:
        (3 + 7*1) / 8 = 1.25 bits/element = 12.8x compression

Approach:
    1. Measure inter-head correlation on real KV caches
    2. Pick head 0 as anchor, compress at 3-bit with ResidualQuant
    3. For remaining heads: delta_h = kv_h - kv_anchor
    4. Quantize deltas at 1-2 bits using PolarQuant (rotation + Lloyd-Max)
    5. Reconstruction: kv_h = dequant(anchor) + dequant(delta_h)

Key insight: Delta coding works for HEADS (same layer, different heads)
even though it fails for LAYERS (different layers, same head). This is
because heads at the same position attend to the same token with the
same positional encoding -- they extract different but correlated
features from the same input representation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .codebook import LloydMaxCodebook
from .polarquant import PolarQuant
from .residual_quant import ResidualQuantEstimator


# ---------------------------------------------------------------------------
# Inter-head correlation measurement
# ---------------------------------------------------------------------------


def measure_inter_head_correlation(
    kv_states: torch.Tensor,
    mode: str = "key",
) -> Dict[str, Any]:
    """Measure correlation between KV heads at the same layer position.

    For each pair of heads (i, j), compute:
    - Per-token cosine similarity (averaged over sequence)
    - Pearson correlation of flattened head matrices
    - Relative delta norm: ||h_j - h_i|| / ||h_i||

    Args:
        kv_states: Tensor of shape [batch, num_heads, seq_len, head_dim].
        mode: "key" or "value" (for labeling only).

    Returns:
        Dict with:
        - pairwise_cosine: [num_heads, num_heads] mean cosine similarity
        - pairwise_pearson: [num_heads, num_heads] Pearson correlation
        - pairwise_delta_norm: [num_heads, num_heads] relative delta norms
        - mean_cosine: scalar mean off-diagonal cosine
        - mean_pearson: scalar mean off-diagonal Pearson r
        - mean_delta_norm: scalar mean off-diagonal relative delta norm
        - anchor_delta_stats: per-head stats relative to head 0
    """
    batch, num_heads, seq_len, head_dim = kv_states.shape

    # Flatten to [num_heads, batch * seq_len * head_dim] for Pearson
    flat = kv_states.float().permute(1, 0, 2, 3).reshape(num_heads, -1)

    # Per-token vectors: [num_heads, batch * seq_len, head_dim]
    per_token = kv_states.float().permute(1, 0, 2, 3).reshape(num_heads, -1, head_dim)

    # Compute pairwise cosine similarity (per-token, then averaged)
    norms = per_token.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normed = per_token / norms
    # [num_heads, num_tokens, head_dim] x [num_heads, head_dim, num_tokens]
    # -> [num_heads, num_heads] mean cosine
    pairwise_cosine = torch.zeros(num_heads, num_heads)
    for i in range(num_heads):
        for j in range(num_heads):
            cos_sim = (normed[i] * normed[j]).sum(dim=-1).mean()
            pairwise_cosine[i, j] = cos_sim.item()

    # Pearson correlation between flattened head vectors
    flat_centered = flat - flat.mean(dim=-1, keepdim=True)
    flat_std = flat_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    flat_normed = flat_centered / flat_std
    pairwise_pearson = (flat_normed @ flat_normed.T).cpu()

    # Relative delta norms (||h_j - h_i|| / ||h_i||)
    pairwise_delta_norm = torch.zeros(num_heads, num_heads)
    head_norms = flat.norm(dim=-1)  # [num_heads]
    for i in range(num_heads):
        for j in range(num_heads):
            delta_norm = (flat[j] - flat[i]).norm().item()
            pairwise_delta_norm[i, j] = delta_norm / (head_norms[i].item() + 1e-8)

    # Off-diagonal statistics
    mask = ~torch.eye(num_heads, dtype=torch.bool)
    mean_cosine = pairwise_cosine[mask].mean().item()
    mean_pearson = pairwise_pearson[mask].mean().item()
    mean_delta_norm = pairwise_delta_norm[mask].mean().item()

    # Anchor (head 0) specific stats
    anchor_delta_stats = []
    for h in range(1, num_heads):
        delta = per_token[h] - per_token[0]  # [num_tokens, head_dim]
        orig = per_token[h]

        delta_var = delta.var().item()
        orig_var = orig.var().item()
        variance_ratio = delta_var / (orig_var + 1e-10)

        anchor_delta_stats.append({
            "head": h,
            "cosine_to_anchor": pairwise_cosine[0, h].item(),
            "pearson_to_anchor": pairwise_pearson[0, h].item(),
            "relative_delta_norm": pairwise_delta_norm[0, h].item(),
            "delta_variance": delta_var,
            "original_variance": orig_var,
            "variance_ratio": variance_ratio,
        })

    return {
        "pairwise_cosine": pairwise_cosine,
        "pairwise_pearson": pairwise_pearson,
        "pairwise_delta_norm": pairwise_delta_norm,
        "mean_cosine": mean_cosine,
        "mean_pearson": mean_pearson,
        "mean_delta_norm": mean_delta_norm,
        "anchor_delta_stats": anchor_delta_stats,
        "num_heads": num_heads,
        "mode": mode,
    }


def select_best_anchor(kv_states: torch.Tensor) -> int:
    """Select the head that minimizes total delta variance.

    For each candidate anchor, compute the sum of delta variances
    across all other heads. The best anchor is the one whose vectors
    are most "central" among all heads.

    Args:
        kv_states: Tensor of shape [batch, num_heads, seq_len, head_dim].

    Returns:
        Index of the best anchor head.
    """
    num_heads = kv_states.shape[1]
    per_token = kv_states.float().permute(1, 0, 2, 3).reshape(num_heads, -1)

    best_anchor = 0
    best_total_var = float("inf")

    for a in range(num_heads):
        total_var = 0.0
        for h in range(num_heads):
            if h == a:
                continue
            delta = per_token[h] - per_token[a]
            total_var += delta.var().item()
        if total_var < best_total_var:
            best_total_var = total_var
            best_anchor = a

    return best_anchor


# ---------------------------------------------------------------------------
# Cross-Head Delta Quantizer
# ---------------------------------------------------------------------------


class CrossHeadDeltaQuantizer(nn.Module):
    """Cross-head delta compression for a single layer's KV cache.

    Compresses one anchor head at full fidelity and remaining heads
    as deltas from the anchor at reduced bit-width.

    The anchor uses ResidualQuant (MSE + 1-bit residual signs) for
    maximum quality. Deltas use PolarQuant (rotation + Lloyd-Max) at
    the delta bit-width.

    Args:
        d: Head dimension.
        num_heads: Number of KV heads.
        anchor_bits: Bits for anchor head (default 3).
        delta_bits: Bits for delta heads (default 1).
        anchor_head: Which head is the anchor (default 0).
        seed: Random seed.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        num_heads: int,
        anchor_bits: int = 3,
        delta_bits: int = 1,
        anchor_head: int = 0,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.anchor_bits = anchor_bits
        self.delta_bits = delta_bits
        self.anchor_head = anchor_head

        # Anchor quantizer: ResidualQuant for maximum quality
        self.anchor_quant = ResidualQuantEstimator(
            d=d, bits=anchor_bits, seed=seed, device=device,
        )

        # Delta quantizer: PolarQuant at reduced bits
        # Use a different seed for the delta rotation matrix
        self.delta_quant = PolarQuant(
            d=d, bits=delta_bits, seed=seed + 1000, device=device,
        )

        # Delta codebook for custom dequantization
        self.delta_codebook = self.delta_quant.codebook

    def effective_bits_per_element(self) -> float:
        """Compute the effective bits per element across all heads."""
        # Anchor: anchor_bits * d + 16 bits norm + 16 bits scale
        anchor_storage = self.anchor_bits * self.d + 32
        # Each delta: delta_bits * d + 16 bits norm
        delta_storage = self.delta_bits * self.d + 16
        total_bits = anchor_storage + (self.num_heads - 1) * delta_storage
        total_elements = self.num_heads * self.d
        return total_bits / total_elements

    def compression_ratio(self) -> float:
        """Compute compression ratio vs FP16."""
        return 16.0 / self.effective_bits_per_element()

    def quantize(
        self, kv_states: torch.Tensor
    ) -> Dict[str, Any]:
        """Compress all heads using anchor + delta scheme.

        Args:
            kv_states: Shape [batch, num_heads, seq_len, head_dim].

        Returns:
            Dict with compressed representations.
        """
        batch, num_heads, seq_len, head_dim = kv_states.shape
        assert num_heads == self.num_heads
        assert head_dim == self.d

        # Extract anchor head: [batch, seq_len, head_dim]
        anchor = kv_states[:, self.anchor_head, :, :]
        anchor_flat = anchor.float().reshape(-1, head_dim)

        # Compress anchor with ResidualQuant
        anchor_compressed = self.anchor_quant.quantize(anchor_flat)

        # Dequantize anchor for delta computation
        anchor_recon = self.anchor_quant.dequantize(anchor_compressed)
        anchor_recon = anchor_recon.reshape(batch, seq_len, head_dim)

        # Compress deltas for non-anchor heads
        delta_indices_list = []
        delta_norms_list = []

        for h in range(num_heads):
            if h == self.anchor_head:
                continue

            head_vecs = kv_states[:, h, :, :].float()  # [batch, seq_len, d]

            # Compute delta from anchor reconstruction
            delta = head_vecs - anchor_recon  # [batch, seq_len, d]
            delta_flat = delta.reshape(-1, head_dim)  # [batch*seq, d]

            # Store delta norms, then normalize
            delta_norm = delta_flat.norm(dim=-1, keepdim=True)  # [batch*seq, 1]
            delta_normalized = delta_flat / (delta_norm + 1e-8)

            # Quantize normalized delta with PolarQuant
            delta_idx = self.delta_quant.quantize(delta_normalized)

            delta_indices_list.append(delta_idx.reshape(batch, seq_len, head_dim))
            delta_norms_list.append(
                delta_norm.squeeze(-1).reshape(batch, seq_len)
            )

        return {
            "anchor_compressed": anchor_compressed,
            "anchor_shape": (batch, seq_len, head_dim),
            "delta_indices": delta_indices_list,
            "delta_norms": delta_norms_list,
            "num_heads": num_heads,
            "anchor_head": self.anchor_head,
        }

    def dequantize(self, compressed: Dict[str, Any]) -> torch.Tensor:
        """Reconstruct all heads from compressed representation.

        Args:
            compressed: Output from quantize().

        Returns:
            Reconstructed KV states [batch, num_heads, seq_len, head_dim].
        """
        batch, seq_len, head_dim = compressed["anchor_shape"]
        num_heads = compressed["num_heads"]
        anchor_head = compressed["anchor_head"]

        # Reconstruct anchor
        anchor_recon = self.anchor_quant.dequantize(compressed["anchor_compressed"])
        anchor_recon = anchor_recon.reshape(batch, seq_len, head_dim)

        # Reconstruct all heads
        result = torch.zeros(
            batch, num_heads, seq_len, head_dim,
            device=anchor_recon.device, dtype=anchor_recon.dtype,
        )
        result[:, anchor_head, :, :] = anchor_recon

        delta_idx = 0
        for h in range(num_heads):
            if h == anchor_head:
                continue

            delta_indices = compressed["delta_indices"][delta_idx]
            delta_norms = compressed["delta_norms"][delta_idx]

            # Dequantize delta
            delta_flat = delta_indices.reshape(-1, head_dim)
            delta_recon = self.delta_quant.dequantize(delta_flat)
            delta_recon = delta_recon.reshape(batch, seq_len, head_dim)

            # Rescale by norm
            delta_recon = delta_recon * delta_norms.unsqueeze(-1)

            # Reconstruct: anchor + delta
            result[:, h, :, :] = anchor_recon + delta_recon
            delta_idx += 1

        return result

    def quantize_dequantize(
        self, kv_states: torch.Tensor
    ) -> torch.Tensor:
        """Convenience: compress then immediately reconstruct."""
        compressed = self.quantize(kv_states)
        return self.dequantize(compressed)


# ---------------------------------------------------------------------------
# Uniform baseline for fair comparison
# ---------------------------------------------------------------------------


class UniformQuantizer(nn.Module):
    """Uniform per-head quantizer using ResidualQuant.

    Compresses every head identically at the same bit-width.
    Baseline for comparison with CrossHeadDeltaQuantizer.

    Args:
        d: Head dimension.
        num_heads: Number of KV heads.
        bits: Bits per coordinate.
        seed: Random seed.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        num_heads: int,
        bits: int = 3,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.bits = bits

        self.quant = ResidualQuantEstimator(
            d=d, bits=bits, seed=seed, device=device,
        )

    def effective_bits_per_element(self) -> float:
        return float(self.bits)

    def compression_ratio(self) -> float:
        return 16.0 / self.effective_bits_per_element()

    def quantize_dequantize(self, kv_states: torch.Tensor) -> torch.Tensor:
        """Compress and reconstruct all heads uniformly."""
        batch, num_heads, seq_len, head_dim = kv_states.shape
        flat = kv_states.float().reshape(-1, head_dim)
        compressed = self.quant.quantize(flat)
        recon = self.quant.dequantize(compressed)
        return recon.reshape(batch, num_heads, seq_len, head_dim)


# ---------------------------------------------------------------------------
# Quality evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_reconstruction_quality(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> Dict[str, float]:
    """Measure reconstruction quality across all heads.

    Args:
        original: [batch, num_heads, seq_len, head_dim]
        reconstructed: same shape

    Returns:
        Dict with quality metrics.
    """
    orig = original.float()
    recon = reconstructed.float()

    # Per-vector cosine similarity
    flat_orig = orig.reshape(-1, orig.shape[-1])
    flat_recon = recon.reshape(-1, recon.shape[-1])
    cos_sim = torch.nn.functional.cosine_similarity(flat_orig, flat_recon, dim=-1)

    # MSE
    mse = ((orig - recon) ** 2).mean().item()

    # Per-head cosine similarity
    num_heads = orig.shape[1]
    per_head_cos = []
    for h in range(num_heads):
        h_orig = orig[:, h].reshape(-1, orig.shape[-1])
        h_recon = recon[:, h].reshape(-1, recon.shape[-1])
        h_cos = torch.nn.functional.cosine_similarity(h_orig, h_recon, dim=-1)
        per_head_cos.append(h_cos.mean().item())

    return {
        "mean_cosine_sim": cos_sim.mean().item(),
        "min_cosine_sim": cos_sim.min().item(),
        "p5_cosine_sim": cos_sim.quantile(0.05).item(),
        "mse": mse,
        "per_head_cosine": per_head_cos,
    }


def evaluate_attention_quality(
    queries: torch.Tensor,
    keys_original: torch.Tensor,
    keys_reconstructed: torch.Tensor,
    top_k: int = 5,
) -> Dict[str, float]:
    """Measure attention quality: how well compressed keys preserve attention.

    Computes softmax attention scores with both original and compressed keys,
    then measures top-k overlap.

    Args:
        queries: [batch, num_heads, num_queries, head_dim]
        keys_original: [batch, num_heads, seq_len, head_dim]
        keys_reconstructed: same shape
        top_k: Number of top positions to compare.

    Returns:
        Dict with attention quality metrics.
    """
    batch, num_heads, seq_len, head_dim = keys_original.shape
    scale = 1.0 / math.sqrt(head_dim)

    # Compute attention scores
    # queries: [B, H, Q, D] @ keys.T: [B, H, D, S] -> [B, H, Q, S]
    scores_orig = (queries.float() @ keys_original.float().transpose(-2, -1)) * scale
    scores_recon = (queries.float() @ keys_reconstructed.float().transpose(-2, -1)) * scale

    # Top-k overlap
    _, top_orig = scores_orig.topk(min(top_k, seq_len), dim=-1)
    _, top_recon = scores_recon.topk(min(top_k, seq_len), dim=-1)

    # Compute overlap per query position
    overlaps = []
    for b in range(batch):
        for h in range(num_heads):
            for q in range(queries.shape[2]):
                orig_set = set(top_orig[b, h, q].tolist())
                recon_set = set(top_recon[b, h, q].tolist())
                overlap = len(orig_set & recon_set) / len(orig_set)
                overlaps.append(overlap)

    # Score correlation (Pearson r of attention logits)
    scores_flat_orig = scores_orig.reshape(-1)
    scores_flat_recon = scores_recon.reshape(-1)
    orig_c = scores_flat_orig - scores_flat_orig.mean()
    recon_c = scores_flat_recon - scores_flat_recon.mean()
    pearson_r = (
        (orig_c * recon_c).sum()
        / (orig_c.norm() * recon_c.norm() + 1e-8)
    ).item()

    return {
        f"top{top_k}_attention_match": sum(overlaps) / len(overlaps),
        "attention_score_pearson_r": pearson_r,
        "mean_score_error": (scores_orig - scores_recon).abs().mean().item(),
    }
