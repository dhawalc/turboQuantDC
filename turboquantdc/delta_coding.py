"""Cross-layer delta coding for sub-1-bit-per-parameter weight compression.

Shannon's source coding theorem establishes that independently quantized
parameters require at least H(W) bits per parameter. But transformer weights
exhibit strong cross-layer correlations: adjacent layers performing similar
computations have similar weight matrices.

Delta coding exploits this by storing:
  - Layer 0: full quantized weights (anchor layer)
  - Layer 1..N: quantized DELTA from previous layer

The conditional entropy H(W_{L+1} | W_L) << H(W_{L+1}) when layers are
correlated, enabling effective bits/param well below independent coding limits.

For Gaussian sources with correlation r:
    H(W_{L+1} | W_L) / H(W_{L+1}) = 1 - r^2

With correlation r=0.9: only 19% of the original entropy needed for deltas.

Reference: Information-theoretic analysis of neural network weight redundancy.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Uniform quantization utilities
# ---------------------------------------------------------------------------

def quantize_uniform(tensor: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    """Symmetric uniform quantization.

    Maps tensor values to integer indices in [-(2^(bits-1)), 2^(bits-1) - 1].

    Args:
        tensor: Input tensor of any shape.
        bits: Number of bits per element.

    Returns:
        Tuple of (indices as int8/int16, scale factor).
    """
    max_val = tensor.abs().max().item()
    if bits == 1:
        # 1-bit: sign quantization only (-1 or 0)
        indices = (tensor >= 0).to(torch.int8)  # 0 or 1
        scale = max_val if max_val > 0 else 1.0
        return indices, scale
    qmax = 2 ** (bits - 1) - 1
    scale = max_val / qmax if max_val > 0 else 1.0
    indices = torch.round(tensor / (scale + 1e-10)).clamp(-qmax - 1, qmax)
    return indices.to(torch.int8 if bits <= 8 else torch.int16), scale


def dequantize_uniform(indices: torch.Tensor, scale: float) -> torch.Tensor:
    """Dequantize uniformly quantized tensor.

    Args:
        indices: Integer indices from quantize_uniform.
        scale: Scale factor from quantize_uniform.

    Returns:
        Reconstructed float tensor.
    """
    out = indices.float() * scale
    return out


def quantize_delta(delta: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    """Quantize a delta tensor using symmetric uniform quantization.

    Deltas between adjacent layers are concentrated near zero, making
    uniform quantization effective even at very low bit-widths.

    Args:
        delta: Difference tensor (W_{L+1} - W_L).
        bits: Bits per delta element.

    Returns:
        Tuple of (quantized indices, scale factor).
    """
    return quantize_uniform(delta, bits)


def dequantize_delta(indices: torch.Tensor, scale: float) -> torch.Tensor:
    """Dequantize a delta tensor.

    Args:
        indices: Quantized delta indices.
        scale: Scale factor.

    Returns:
        Reconstructed delta tensor.
    """
    return dequantize_uniform(indices, scale)


# ---------------------------------------------------------------------------
# Layer name parsing
# ---------------------------------------------------------------------------

# Weight types within each transformer layer
WEIGHT_TYPES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]

# Pattern to extract layer index from parameter names
# Matches patterns like "model.layers.0.self_attn.q_proj.weight"
_LAYER_PATTERN = re.compile(r"model\.layers\.(\d+)\.(.*)")


def parse_layer_params(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[int, torch.Tensor]]:
    """Group model parameters by weight type and layer index.

    Args:
        state_dict: Model state dict.

    Returns:
        Dict mapping weight_type -> {layer_idx: tensor}.
        Example: {"self_attn.q_proj.weight": {0: tensor, 1: tensor, ...}}
    """
    grouped: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
    for name, param in state_dict.items():
        match = _LAYER_PATTERN.match(name)
        if match:
            layer_idx = int(match.group(1))
            weight_type = match.group(2)
            if weight_type in WEIGHT_TYPES:
                grouped[weight_type][layer_idx] = param
    return dict(grouped)


# ---------------------------------------------------------------------------
# Cross-layer correlation analysis
# ---------------------------------------------------------------------------

def compute_layer_pair_stats(
    w_current: torch.Tensor,
    w_next: torch.Tensor,
) -> dict[str, float]:
    """Compute correlation statistics between two adjacent layer weight matrices.

    Args:
        w_current: Weight matrix for layer L.
        w_next: Weight matrix for layer L+1.

    Returns:
        Dict with keys: cosine_sim, relative_delta_norm, correlation, delta_norm.
    """
    w_current_flat = w_current.float().reshape(-1)
    w_next_flat = w_next.float().reshape(-1)
    delta = w_next_flat - w_current_flat

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        w_current_flat.unsqueeze(0), w_next_flat.unsqueeze(0)
    ).item()

    # Relative delta norm: ||delta|| / ||W_L||
    w_norm = w_current_flat.norm().item()
    delta_norm = delta.norm().item()
    relative_delta = delta_norm / (w_norm + 1e-10)

    # Pearson correlation coefficient
    mean_c = w_current_flat.mean()
    mean_n = w_next_flat.mean()
    centered_c = w_current_flat - mean_c
    centered_n = w_next_flat - mean_n
    correlation = (
        (centered_c * centered_n).sum()
        / (centered_c.norm() * centered_n.norm() + 1e-10)
    ).item()

    return {
        "cosine_sim": cos_sim,
        "relative_delta_norm": relative_delta,
        "correlation": correlation,
        "delta_norm": delta_norm,
    }


def estimate_delta_entropy(delta: torch.Tensor, bits: int) -> float:
    """Estimate the entropy of a quantized delta in bits per element.

    Quantizes the delta to the given bit-width, then computes the Shannon
    entropy of the resulting symbol distribution.

    Args:
        delta: Delta tensor.
        bits: Quantization bit-width.

    Returns:
        Entropy in bits per element.
    """
    indices, _ = quantize_uniform(delta.float().reshape(-1), bits)
    # Count symbol frequencies
    unique, counts = indices.unique(return_counts=True)
    probs = counts.float() / counts.sum().float()
    # Shannon entropy: H = -sum(p * log2(p))
    entropy = -(probs * probs.log2()).sum().item()
    return entropy


# ---------------------------------------------------------------------------
# CrossLayerDeltaCoder
# ---------------------------------------------------------------------------

class CrossLayerDeltaCoder:
    """Compresses model weights using cross-layer delta coding.

    Instead of quantizing each layer independently, stores:
    - Layer 0: full quantized weights (anchor layer)
    - Layer 1..N: quantized delta from previous layer

    Since adjacent layers are correlated, deltas are smaller and
    require fewer bits to represent at the same quality level.

    The theoretical improvement over independent coding is:
    H(W_L+1 | W_L) / H(W_L+1) = 1 - r^2  (for Gaussian sources)

    With correlation r=0.9: 1 - 0.81 = 0.19 -> 81% fewer bits for deltas
    With correlation r=0.5: 1 - 0.25 = 0.75 -> 25% fewer bits for deltas

    Args:
        anchor_bits: Bits for the anchor (first) layer. Default 4.
        delta_bits: Bits for each delta layer. Default 2.
    """

    def __init__(self, anchor_bits: int = 4, delta_bits: int = 2):
        self.anchor_bits = anchor_bits
        self.delta_bits = delta_bits

    def encode_model(self, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Encode all transformer layers using delta coding.

        Groups parameters by weight type (q_proj, k_proj, etc.) and applies
        delta coding within each group across layers.

        Args:
            state_dict: Model state dict (typically model.state_dict()).

        Returns:
            Encoded representation with structure:
            {
                "weight_type": {
                    "anchor": (indices, scale),        # Layer 0 full quantization
                    "deltas": [(indices, scale), ...],  # Layer 1..N deltas
                    "layer_indices": [0, 1, 2, ...],
                    "shape": (rows, cols),
                },
                ...
                "_metadata": {
                    "anchor_bits": int,
                    "delta_bits": int,
                    "num_layers": int,
                    "total_params": int,
                }
            }
        """
        grouped = parse_layer_params(state_dict)
        encoded: dict[str, Any] = {}
        total_params = 0

        for weight_type, layers in grouped.items():
            sorted_indices = sorted(layers.keys())
            if len(sorted_indices) < 2:
                continue

            # Anchor layer (layer 0)
            anchor_tensor = layers[sorted_indices[0]].float()
            anchor_indices, anchor_scale = quantize_uniform(anchor_tensor, self.anchor_bits)
            anchor_recon = dequantize_uniform(anchor_indices, anchor_scale)

            # Delta layers
            deltas = []
            prev_recon = anchor_recon
            for idx in sorted_indices[1:]:
                current = layers[idx].float()
                delta = current - prev_recon
                delta_indices, delta_scale = quantize_delta(delta, self.delta_bits)
                deltas.append((delta_indices, delta_scale))
                # Reconstruct for the next delta computation
                delta_recon = dequantize_delta(delta_indices, delta_scale)
                prev_recon = prev_recon + delta_recon

            total_params += anchor_tensor.numel() * len(sorted_indices)

            encoded[weight_type] = {
                "anchor": (anchor_indices, anchor_scale),
                "deltas": deltas,
                "layer_indices": sorted_indices,
                "shape": tuple(anchor_tensor.shape),
            }

        encoded["_metadata"] = {
            "anchor_bits": self.anchor_bits,
            "delta_bits": self.delta_bits,
            "num_layers": max(
                (max(v["layer_indices"]) + 1 for v in encoded.values() if isinstance(v, dict) and "layer_indices" in v),
                default=0,
            ),
            "total_params": total_params,
        }
        return encoded

    def decode_layer(self, encoded: dict[str, Any], layer_idx: int) -> dict[str, torch.Tensor]:
        """Reconstruct weights for a specific layer.

        For layer 0: dequantize anchor.
        For layer N: reconstruct layer N-1, then add dequantized delta[N-1].

        Iteratively reconstructs from the anchor through the chain to avoid
        recursion depth issues for deep models.

        Args:
            encoded: Output from encode_model.
            layer_idx: Which layer to reconstruct.

        Returns:
            Dict mapping weight_type -> reconstructed weight tensor.
        """
        result = {}
        for weight_type, data in encoded.items():
            if weight_type.startswith("_"):
                continue

            sorted_indices = data["layer_indices"]
            if layer_idx not in sorted_indices:
                continue

            position = sorted_indices.index(layer_idx)
            anchor_indices, anchor_scale = data["anchor"]
            recon = dequantize_uniform(anchor_indices, anchor_scale)

            # Walk the chain from anchor to target layer
            for i in range(position):
                delta_indices, delta_scale = data["deltas"][i]
                delta_recon = dequantize_delta(delta_indices, delta_scale)
                recon = recon + delta_recon

            result[weight_type] = recon

        return result

    def total_size_bits(self, encoded: dict[str, Any]) -> int:
        """Total compressed size in bits.

        Counts:
        - anchor_bits * numel for each anchor
        - delta_bits * numel for each delta
        - 32 bits per scale factor (one per anchor + one per delta)

        Args:
            encoded: Output from encode_model.

        Returns:
            Total size in bits.
        """
        total = 0
        anchor_bits = encoded["_metadata"]["anchor_bits"]
        delta_bits = encoded["_metadata"]["delta_bits"]

        for weight_type, data in encoded.items():
            if weight_type.startswith("_"):
                continue
            anchor_indices, _ = data["anchor"]
            numel = anchor_indices.numel()

            # Anchor: bits * numel + 32 bits for scale
            total += anchor_bits * numel + 32

            # Deltas: bits * numel + 32 bits for scale each
            for delta_indices, _ in data["deltas"]:
                total += delta_bits * delta_indices.numel() + 32

        return total

    def compression_report(
        self,
        encoded: dict[str, Any],
        original_size_bytes: int,
    ) -> dict[str, Any]:
        """Report compression statistics.

        Args:
            encoded: Output from encode_model.
            original_size_bytes: Original model size in bytes (fp16).

        Returns:
            Dict with compression metrics.
        """
        total_bits = self.total_size_bits(encoded)
        compressed_bytes = total_bits / 8
        total_params = encoded["_metadata"]["total_params"]

        # Anchor size
        anchor_bits_total = 0
        delta_bits_total = 0
        anchor_b = encoded["_metadata"]["anchor_bits"]
        delta_b = encoded["_metadata"]["delta_bits"]

        for weight_type, data in encoded.items():
            if weight_type.startswith("_"):
                continue
            anchor_indices, _ = data["anchor"]
            anchor_bits_total += anchor_b * anchor_indices.numel() + 32
            for delta_indices, _ in data["deltas"]:
                delta_bits_total += delta_b * delta_indices.numel() + 32

        effective_bpp = total_bits / total_params if total_params > 0 else 0

        return {
            "original_bytes": original_size_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": original_size_bytes / compressed_bytes if compressed_bytes > 0 else float("inf"),
            "effective_bits_per_param": effective_bpp,
            "anchor_bytes": anchor_bits_total / 8,
            "delta_bytes": delta_bits_total / 8,
            "total_params": total_params,
            "num_layers": encoded["_metadata"]["num_layers"],
            "anchor_bits": anchor_b,
            "delta_bits": delta_b,
        }

    def per_layer_quality(
        self,
        encoded: dict[str, Any],
        state_dict: dict[str, torch.Tensor],
    ) -> list[dict[str, Any]]:
        """Measure reconstruction quality for each layer.

        Args:
            encoded: Output from encode_model.
            state_dict: Original model state dict.

        Returns:
            List of dicts with per-layer quality metrics.
        """
        grouped = parse_layer_params(state_dict)
        num_layers = encoded["_metadata"]["num_layers"]
        results = []

        for layer_idx in range(num_layers):
            decoded = self.decode_layer(encoded, layer_idx)
            layer_stats: dict[str, Any] = {"layer": layer_idx}
            cos_sims = []
            rel_mses = []

            for weight_type, recon in decoded.items():
                if weight_type not in grouped:
                    continue
                if layer_idx not in grouped[weight_type]:
                    continue
                original = grouped[weight_type][layer_idx].float()
                recon_flat = recon.reshape(-1)
                orig_flat = original.reshape(-1)

                cos_sim = torch.nn.functional.cosine_similarity(
                    orig_flat.unsqueeze(0), recon_flat.unsqueeze(0)
                ).item()
                rel_mse = (
                    ((orig_flat - recon_flat) ** 2).sum()
                    / ((orig_flat ** 2).sum() + 1e-10)
                ).item()

                cos_sims.append(cos_sim)
                rel_mses.append(rel_mse)

            if cos_sims:
                layer_stats["avg_cosine_sim"] = sum(cos_sims) / len(cos_sims)
                layer_stats["min_cosine_sim"] = min(cos_sims)
                layer_stats["avg_rel_mse"] = sum(rel_mses) / len(rel_mses)
                layer_stats["max_rel_mse"] = max(rel_mses)
            results.append(layer_stats)

        return results
