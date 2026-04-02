"""Cross-layer KV cache sharing for additional memory savings.

Hypothesis: Adjacent transformer layers have correlated KV representations,
enabling delta coding for ~2x additional compression.

Finding: **KV vector delta coding is disproved** (Pearson r=0.001 between
adjacent layers). Individual KV vectors are effectively independent random
vectors -- there is no exploitable cross-layer correlation in the actual
vector values.

However, the STATISTICAL DISTRIBUTION is identical across layers. After
random orthogonal rotation, each coordinate follows the same concentrated
Beta distribution (approximately N(0, 1/d) for d >= 64) regardless of
which layer produced the vector. This means:

1. **Codebook sharing**: The Lloyd-Max codebook for (d, bits) is identical
   for all layers. Sharing one codebook per group of layers instead of one
   per layer saves codebook storage with zero quality loss.

2. **Rotation matrix sharing**: The random orthogonal rotation matrix only
   needs to ensure Haar-uniformity. Sharing one rotation matrix across a
   group of layers is mathematically valid and saves O(d^2) per shared
   layer (or O(d) with WHT).

The CrossLayerKVCache implements this resource-sharing approach:
- Groups of `group_size` adjacent layers share one codebook + rotation
- Each layer still quantizes its own KV vectors independently
- Quality is identical to per-layer quantization (sharing is lossless)
- Memory savings come from reduced codebook/rotation storage overhead

For a 36-layer model with d=128, group_size=4:
- Per-layer rotation: 36 * 128^2 * 4 bytes = 2.36 MB
- Shared rotation: 9 * 128^2 * 4 bytes = 0.59 MB (4x saving on rotations)
- Codebook sharing: negligible size but cleaner memory layout

This module also includes the diagnostic tools that measured the cross-layer
correlation and confirmed the negative result for delta coding.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from .codebook import LloydMaxCodebook
from .generation_cache import (
    GenerationCache,
    _CompressedLayer,
    _FP16Layer,
    compute_anchor_schedule,
    ANCHOR_STRATEGIES,
)
from .rotation import (
    apply_wht_rotation,
    generate_rotation_matrix,
    generate_wht_rotation,
)


# ---------------------------------------------------------------------------
# Cross-layer KV correlation diagnostics
# ---------------------------------------------------------------------------


def measure_cross_layer_kv_correlation(
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> List[Dict[str, float]]:
    """Measure KV similarity between adjacent layers.

    This is the diagnostic that confirmed delta coding is not viable for KV
    caches: adjacent-layer KV vectors have near-zero Pearson correlation
    (r ~ 0.001), unlike model weights which are highly correlated (r ~ 0.9).

    Args:
        kv_by_layer: Dict mapping layer_idx -> (key_states, value_states).
            Each tensor has shape [batch, num_heads, seq_len, head_dim].

    Returns:
        List of dicts with per-pair statistics:
        - layer_pair: (L, L+1) tuple
        - key_cosine_sim: mean cosine similarity of key vectors
        - value_cosine_sim: mean cosine similarity of value vectors
        - key_pearson_r: Pearson correlation of flattened key matrices
        - value_pearson_r: Pearson correlation of flattened value matrices
        - key_relative_delta_norm: ||k_{L+1} - k_L|| / ||k_L||
        - value_relative_delta_norm: ||v_{L+1} - v_L|| / ||v_L||
    """
    sorted_layers = sorted(kv_by_layer.keys())
    results = []

    for i in range(len(sorted_layers) - 1):
        l_curr = sorted_layers[i]
        l_next = sorted_layers[i + 1]
        k_curr, v_curr = kv_by_layer[l_curr]
        k_next, v_next = kv_by_layer[l_next]

        stats = {"layer_pair": (l_curr, l_next)}

        for prefix, t_curr, t_next in [
            ("key", k_curr, k_next),
            ("value", v_curr, v_next),
        ]:
            flat_curr = t_curr.float().reshape(-1)
            flat_next = t_next.float().reshape(-1)

            # Cosine similarity (per-vector, then averaged)
            vec_curr = t_curr.float().reshape(-1, t_curr.shape[-1])
            vec_next = t_next.float().reshape(-1, t_next.shape[-1])
            cos_sims = torch.nn.functional.cosine_similarity(
                vec_curr, vec_next, dim=-1
            )
            stats[f"{prefix}_cosine_sim"] = cos_sims.mean().item()

            # Pearson correlation
            mean_c = flat_curr.mean()
            mean_n = flat_next.mean()
            centered_c = flat_curr - mean_c
            centered_n = flat_next - mean_n
            pearson_r = (
                (centered_c * centered_n).sum()
                / (centered_c.norm() * centered_n.norm() + 1e-10)
            ).item()
            stats[f"{prefix}_pearson_r"] = pearson_r

            # Relative delta norm
            delta_norm = (flat_next - flat_curr).norm().item()
            curr_norm = flat_curr.norm().item()
            stats[f"{prefix}_relative_delta_norm"] = (
                delta_norm / (curr_norm + 1e-10)
            )

        results.append(stats)

    return results


def measure_distribution_similarity(
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    num_bins: int = 100,
) -> List[Dict[str, float]]:
    """Measure how similar the KV VALUE DISTRIBUTIONS are across layers.

    Even though individual KV vectors differ (r=0.001), the distributions
    of coordinate values may be nearly identical -- which is what enables
    codebook sharing.

    Uses histogram-based KL divergence and Kolmogorov-Smirnov statistic.

    Args:
        kv_by_layer: Dict mapping layer_idx -> (key_states, value_states).
        num_bins: Number of histogram bins for distribution comparison.

    Returns:
        List of dicts with per-pair distribution statistics:
        - layer_pair: (L, L+1) tuple
        - key_kl_divergence: KL(P_L || P_{L+1}) for key coordinate values
        - value_kl_divergence: KL(P_L || P_{L+1}) for value coordinate values
        - key_mean_diff: |mean(k_L) - mean(k_{L+1})| / std(k_L)
        - value_mean_diff: |mean(v_L) - mean(v_{L+1})| / std(v_L)
        - key_std_ratio: std(k_{L+1}) / std(k_L)
        - value_std_ratio: std(v_{L+1}) / std(v_L)
    """
    sorted_layers = sorted(kv_by_layer.keys())
    results = []

    for i in range(len(sorted_layers) - 1):
        l_curr = sorted_layers[i]
        l_next = sorted_layers[i + 1]
        k_curr, v_curr = kv_by_layer[l_curr]
        k_next, v_next = kv_by_layer[l_next]

        stats = {"layer_pair": (l_curr, l_next)}

        for prefix, t_curr, t_next in [
            ("key", k_curr, k_next),
            ("value", v_curr, v_next),
        ]:
            flat_curr = t_curr.float().reshape(-1)
            flat_next = t_next.float().reshape(-1)

            mean_c = flat_curr.mean().item()
            mean_n = flat_next.mean().item()
            std_c = flat_curr.std().item()
            std_n = flat_next.std().item()

            stats[f"{prefix}_mean_diff"] = abs(mean_n - mean_c) / (std_c + 1e-10)
            stats[f"{prefix}_std_ratio"] = std_n / (std_c + 1e-10)

            # Histogram-based KL divergence
            combined = torch.cat([flat_curr, flat_next])
            bin_min = combined.min().item()
            bin_max = combined.max().item()
            bins = torch.linspace(bin_min, bin_max, num_bins + 1)

            hist_curr = torch.histc(flat_curr, bins=num_bins, min=bin_min, max=bin_max)
            hist_next = torch.histc(flat_next, bins=num_bins, min=bin_min, max=bin_max)

            # Normalize to probabilities with Laplace smoothing
            eps = 1e-8
            p = (hist_curr + eps) / (hist_curr.sum() + eps * num_bins)
            q = (hist_next + eps) / (hist_next.sum() + eps * num_bins)

            kl = (p * (p / q).log()).sum().item()
            stats[f"{prefix}_kl_divergence"] = kl

        results.append(stats)

    return results


def correlation_report(
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> str:
    """Generate a human-readable correlation analysis report.

    Combines both vector correlation and distribution similarity analysis.

    Args:
        kv_by_layer: Dict mapping layer_idx -> (key_states, value_states).

    Returns:
        Multi-line report string.
    """
    vec_stats = measure_cross_layer_kv_correlation(kv_by_layer)
    dist_stats = measure_distribution_similarity(kv_by_layer)

    lines = ["=" * 70]
    lines.append("Cross-Layer KV Cache Correlation Analysis")
    lines.append("=" * 70)
    lines.append("")

    # Vector correlation summary
    if vec_stats:
        avg_k_cos = sum(s["key_cosine_sim"] for s in vec_stats) / len(vec_stats)
        avg_v_cos = sum(s["value_cosine_sim"] for s in vec_stats) / len(vec_stats)
        avg_k_r = sum(s["key_pearson_r"] for s in vec_stats) / len(vec_stats)
        avg_v_r = sum(s["value_pearson_r"] for s in vec_stats) / len(vec_stats)
        avg_k_delta = sum(s["key_relative_delta_norm"] for s in vec_stats) / len(vec_stats)
        avg_v_delta = sum(s["value_relative_delta_norm"] for s in vec_stats) / len(vec_stats)

        lines.append("VECTOR CORRELATION (delta coding viability):")
        lines.append(f"  Avg key cosine sim:    {avg_k_cos:.4f}")
        lines.append(f"  Avg value cosine sim:  {avg_v_cos:.4f}")
        lines.append(f"  Avg key Pearson r:     {avg_k_r:.4f}")
        lines.append(f"  Avg value Pearson r:   {avg_v_r:.4f}")
        lines.append(f"  Avg key delta norm:    {avg_k_delta:.4f}")
        lines.append(f"  Avg value delta norm:  {avg_v_delta:.4f}")
        lines.append("")

        delta_viable = avg_k_r > 0.5 or avg_v_r > 0.5
        if delta_viable:
            lines.append("  => Delta coding MAY be viable (r > 0.5)")
        else:
            lines.append("  => Delta coding NOT viable (r << 0.5)")
            lines.append("     Adjacent KV vectors are effectively independent.")
        lines.append("")

    # Distribution similarity summary
    if dist_stats:
        avg_k_kl = sum(s["key_kl_divergence"] for s in dist_stats) / len(dist_stats)
        avg_v_kl = sum(s["value_kl_divergence"] for s in dist_stats) / len(dist_stats)
        avg_k_std = sum(s["key_std_ratio"] for s in dist_stats) / len(dist_stats)
        avg_v_std = sum(s["value_std_ratio"] for s in dist_stats) / len(dist_stats)

        lines.append("DISTRIBUTION SIMILARITY (codebook sharing viability):")
        lines.append(f"  Avg key KL divergence:   {avg_k_kl:.6f}")
        lines.append(f"  Avg value KL divergence: {avg_v_kl:.6f}")
        lines.append(f"  Avg key std ratio:       {avg_k_std:.4f}")
        lines.append(f"  Avg value std ratio:     {avg_v_std:.4f}")
        lines.append("")

        codebook_viable = avg_k_kl < 0.01 and avg_v_kl < 0.01
        if codebook_viable:
            lines.append("  => Codebook sharing IS viable (KL << 0.01)")
            lines.append("     Distributions are nearly identical across layers.")
        else:
            lines.append("  => Codebook sharing may lose quality (KL > 0.01)")
        lines.append("")

    lines.append("RECOMMENDATION:")
    if vec_stats and dist_stats:
        if delta_viable:
            lines.append("  Use CrossLayerKVCache with delta_mode='delta'")
        elif codebook_viable:
            lines.append("  Use CrossLayerKVCache with delta_mode='shared_resources'")
            lines.append("  (Share codebook + rotation across layer groups)")
        else:
            lines.append("  Use standard GenerationCache (no cross-layer benefit)")

    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared-resource compressed layer
# ---------------------------------------------------------------------------


class _SharedResourceLayer:
    """Compressed layer that uses externally-provided codebook and rotation.

    Instead of each layer creating its own LloydMaxCodebook and rotation
    matrix, this layer receives them from the group anchor. Since the
    Lloyd-Max codebook depends only on (d, bits) and the rotation matrix
    only needs to be Haar-uniform, sharing is mathematically lossless.

    Memory savings per shared layer:
    - QR rotation: d^2 * 4 bytes (e.g., 128^2 * 4 = 64 KB)
    - WHT rotation: d * 4 bytes (e.g., 128 * 4 = 512 bytes)
    - Codebook: n_levels * 4 bytes (negligible but cleaner)

    Args:
        key_codebook: Shared Lloyd-Max codebook for keys.
        val_codebook: Shared Lloyd-Max codebook for values.
        rotation: Shared QR rotation matrix (d x d), or None for WHT.
        wht_params: Shared WHT parameters, or None for QR.
        rotation_type: "wht" or "qr".
        key_bits: Bits for key quantization.
        val_bits: Bits for value quantization.
        fp16_window: Number of recent tokens at FP16.
        use_norm_correction: Apply norm correction.
        use_residual_quant: Apply 1-bit residual sign correction.
    """

    def __init__(
        self,
        key_codebook: LloydMaxCodebook,
        val_codebook: LloydMaxCodebook,
        rotation: Optional[torch.Tensor],
        wht_params: Optional[dict],
        rotation_type: str,
        key_bits: int = 3,
        val_bits: int = 2,
        fp16_window: int = 128,
        use_norm_correction: bool = True,
        use_residual_quant: bool = True,
    ):
        self.key_bits = key_bits
        self.val_bits = val_bits
        self.fp16_window = fp16_window
        self.use_norm_correction = use_norm_correction
        self.use_residual_quant = use_residual_quant

        # Shared resources (references, not copies)
        self._key_codebook = key_codebook
        self._val_codebook = val_codebook
        self._rotation = rotation
        self._wht_params = wht_params
        self._rotation_type = rotation_type

        self._seq_len: int = 0
        self._head_dim: Optional[int] = None
        self._num_heads: Optional[int] = None
        self._batch_size: Optional[int] = None
        self._dtype: Optional[torch.dtype] = None
        self._device: Optional[torch.device] = None

        # Compressed storage
        self._key_indices: List[torch.Tensor] = []
        self._key_norms: List[torch.Tensor] = []
        self._key_res_signs: List[torch.Tensor] = []
        self._key_res_scales: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._val_norms: List[torch.Tensor] = []

        # Raw FP16 copies for precision window
        self._raw_keys: List[torch.Tensor] = []
        self._raw_vals: List[torch.Tensor] = []

        # Incremental dequantization cache
        self._dequant_key_cache: Optional[torch.Tensor] = None
        self._dequant_val_cache: Optional[torch.Tensor] = None
        self._dequant_len: int = 0

    def _lazy_init(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Initialize shape metadata from first observed tensors."""
        self._batch_size = key_states.shape[0]
        self._num_heads = key_states.shape[1]
        self._head_dim = key_states.shape[3]
        self._dtype = key_states.dtype
        self._device = key_states.device

    def _quantize_vectors(
        self,
        vectors: torch.Tensor,
        codebook: LloydMaxCodebook,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize [batch, heads, seq, d] vectors with norm + residual correction."""
        batch, heads, seq, d = vectors.shape
        flat = vectors.float().reshape(-1, d)

        # Normalize
        norms = flat.norm(dim=-1, keepdim=True)
        normalized = flat / (norms + 1e-8)

        # Rotate
        if self._rotation_type == "wht":
            rotated = apply_wht_rotation(normalized, self._wht_params)
        else:
            rotated = normalized @ self._rotation

        # Quantize per coordinate
        indices = torch.bucketize(rotated, codebook.boundaries)
        indices = indices.clamp(0, codebook.centroids.shape[0] - 1)

        # Norm correction
        recon_rotated = codebook.centroids[indices]
        if self._rotation_type == "wht":
            recon_unrotated = apply_wht_rotation(
                recon_rotated, self._wht_params, inverse=True
            )
        else:
            recon_unrotated = recon_rotated @ self._rotation.T
        recon_norm = recon_unrotated.norm(dim=-1, keepdim=True)
        corrected_norms = norms * (
            norms / (recon_norm * norms.abs().clamp(min=1e-8) + 1e-8)
        ).clamp(0.5, 2.0)

        # Residual signs
        residual = rotated - recon_rotated
        res_signs = torch.sign(residual)
        res_scale = residual.abs().mean(dim=-1, keepdim=True)

        return (
            indices.reshape(batch, heads, seq, d),
            corrected_norms.reshape(batch, heads, seq),
            res_signs.reshape(batch, heads, seq, d),
            res_scale.reshape(batch, heads, seq),
        )

    def _dequantize_vectors(
        self,
        indices: torch.Tensor,
        norms: torch.Tensor,
        codebook: LloydMaxCodebook,
        res_signs: Optional[torch.Tensor] = None,
        res_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruct vectors from compressed representation."""
        batch, heads, seq, d = indices.shape
        flat_idx = indices.reshape(-1, d)
        flat_norms = norms.reshape(-1)

        reconstructed = codebook.centroids.float()[flat_idx.long()]

        # Residual correction
        if res_signs is not None and res_scale is not None:
            reconstructed = (
                reconstructed
                + res_signs.float().reshape(-1, d) * res_scale.float().reshape(-1, 1)
            )

        # Unrotate and rescale
        if self._rotation_type == "wht":
            reconstructed = apply_wht_rotation(
                reconstructed.float(), self._wht_params, inverse=True
            )
        else:
            reconstructed = torch.matmul(reconstructed, self._rotation.float().T)

        reconstructed = reconstructed * flat_norms.float().unsqueeze(-1)
        return reconstructed.reshape(batch, heads, seq, d)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress and store new KV states, return full dequantized cache."""
        if self._head_dim is None:
            self._lazy_init(key_states, value_states)

        new_seq = key_states.shape[2]

        # Compress keys
        k_idx, k_norms, k_rsigns, k_rscale = self._quantize_vectors(
            key_states, self._key_codebook,
        )
        self._key_indices.append(k_idx)
        self._key_norms.append(k_norms)
        self._key_res_signs.append(k_rsigns)
        self._key_res_scales.append(k_rscale)

        # Compress values (no residual signs)
        v_idx, v_norms, _, _ = self._quantize_vectors(
            value_states, self._val_codebook,
        )
        self._val_indices.append(v_idx)
        self._val_norms.append(v_norms)

        # Store raw FP16 for precision window
        self._raw_keys.append(key_states.detach())
        self._raw_vals.append(value_states.detach())

        # Trim raw FP16 storage
        if self.fp16_window > 0 and self._seq_len > self.fp16_window * 2:
            all_rk = torch.cat(self._raw_keys, dim=2)
            all_rv = torch.cat(self._raw_vals, dim=2)
            self._raw_keys = [all_rk[:, :, -self.fp16_window:, :]]
            self._raw_vals = [all_rv[:, :, -self.fp16_window:, :]]

        self._seq_len += new_seq
        return self._dequantize_all()

    def _dequantize_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct all keys and values with FP16 window splice."""
        if self._seq_len == 0:
            d = self._head_dim or 1
            empty = torch.zeros(
                self._batch_size or 1,
                self._num_heads or 1,
                0, d,
                dtype=self._dtype,
                device=self._device,
            )
            return empty, empty

        # Incremental dequantization
        if self._dequant_len < self._seq_len:
            new_k_idx_parts = []
            new_k_norms_parts = []
            new_k_rsigns_parts = []
            new_k_rscales_parts = []
            new_v_idx_parts = []
            new_v_norms_parts = []
            seen = 0
            for i, k_idx in enumerate(self._key_indices):
                chunk_len = k_idx.shape[2]
                chunk_end = seen + chunk_len
                if chunk_end <= self._dequant_len:
                    seen = chunk_end
                    continue
                start_in_chunk = max(0, self._dequant_len - seen)
                new_k_idx_parts.append(k_idx[:, :, start_in_chunk:, :])
                new_k_norms_parts.append(self._key_norms[i][:, :, start_in_chunk:])
                new_k_rsigns_parts.append(
                    self._key_res_signs[i][:, :, start_in_chunk:, :]
                )
                new_k_rscales_parts.append(
                    self._key_res_scales[i][:, :, start_in_chunk:]
                )
                new_v_idx_parts.append(self._val_indices[i][:, :, start_in_chunk:, :])
                new_v_norms_parts.append(self._val_norms[i][:, :, start_in_chunk:])
                seen = chunk_end

            if new_k_idx_parts:
                new_k_idx = torch.cat(new_k_idx_parts, dim=2)
                new_k_norms = torch.cat(new_k_norms_parts, dim=2)
                new_k_rsigns = torch.cat(new_k_rsigns_parts, dim=2)
                new_k_rscales = torch.cat(new_k_rscales_parts, dim=2)
                new_v_idx = torch.cat(new_v_idx_parts, dim=2)
                new_v_norms = torch.cat(new_v_norms_parts, dim=2)

                new_keys = self._dequantize_vectors(
                    new_k_idx, new_k_norms, self._key_codebook,
                    new_k_rsigns if self.use_residual_quant else None,
                    new_k_rscales if self.use_residual_quant else None,
                )
                new_values = self._dequantize_vectors(
                    new_v_idx, new_v_norms, self._val_codebook,
                )

                if self._dequant_key_cache is not None:
                    self._dequant_key_cache = torch.cat(
                        [self._dequant_key_cache, new_keys], dim=2,
                    )
                    self._dequant_val_cache = torch.cat(
                        [self._dequant_val_cache, new_values], dim=2,
                    )
                else:
                    self._dequant_key_cache = new_keys
                    self._dequant_val_cache = new_values

            self._dequant_len = self._seq_len

        keys = self._dequant_key_cache.clone()
        values = self._dequant_val_cache.clone()

        # FP16 window splice
        if self._raw_keys:
            raw_keys = torch.cat(self._raw_keys, dim=2)
            raw_vals = torch.cat(self._raw_vals, dim=2)
            win = min(self.fp16_window, raw_keys.shape[2])
            if win > 0 and keys.shape[2] >= win:
                keys[:, :, -win:, :] = raw_keys[:, :, -win:, :].to(keys.dtype)
                values[:, :, -win:, :] = raw_vals[:, :, -win:, :].to(values.dtype)

        return keys.to(self._dtype), values.to(self._dtype)

    def get_seq_length(self) -> int:
        return self._seq_len

    def clear(self) -> None:
        self._key_indices.clear()
        self._key_norms.clear()
        self._key_res_signs.clear()
        self._key_res_scales.clear()
        self._val_indices.clear()
        self._val_norms.clear()
        self._raw_keys.clear()
        self._raw_vals.clear()
        self._seq_len = 0
        self._dequant_key_cache = None
        self._dequant_val_cache = None
        self._dequant_len = 0

    def reorder(self, beam_idx: torch.LongTensor) -> None:
        self._key_indices = [t.index_select(0, beam_idx) for t in self._key_indices]
        self._key_norms = [t.index_select(0, beam_idx) for t in self._key_norms]
        self._key_res_signs = [
            t.index_select(0, beam_idx) for t in self._key_res_signs
        ]
        self._key_res_scales = [
            t.index_select(0, beam_idx) for t in self._key_res_scales
        ]
        self._val_indices = [t.index_select(0, beam_idx) for t in self._val_indices]
        self._val_norms = [t.index_select(0, beam_idx) for t in self._val_norms]
        self._raw_keys = [t.index_select(0, beam_idx) for t in self._raw_keys]
        self._raw_vals = [t.index_select(0, beam_idx) for t in self._raw_vals]
        self._dequant_key_cache = None
        self._dequant_val_cache = None
        self._dequant_len = 0

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self._seq_len + max_length
        if self._seq_len <= max_length:
            return
        if self._key_indices:
            all_k_idx = torch.cat(self._key_indices, dim=2)[:, :, :max_length]
            all_k_norms = torch.cat(self._key_norms, dim=2)[:, :, :max_length]
            all_k_rsigns = torch.cat(self._key_res_signs, dim=2)[:, :, :max_length]
            all_k_rscales = torch.cat(self._key_res_scales, dim=2)[:, :, :max_length]
            all_v_idx = torch.cat(self._val_indices, dim=2)[:, :, :max_length]
            all_v_norms = torch.cat(self._val_norms, dim=2)[:, :, :max_length]
            raw_keys = torch.cat(self._raw_keys, dim=2)[:, :, :max_length]
            raw_vals = torch.cat(self._raw_vals, dim=2)[:, :, :max_length]
            self._key_indices = [all_k_idx]
            self._key_norms = [all_k_norms]
            self._key_res_signs = [all_k_rsigns]
            self._key_res_scales = [all_k_rscales]
            self._val_indices = [all_v_idx]
            self._val_norms = [all_v_norms]
            self._raw_keys = [raw_keys]
            self._raw_vals = [raw_vals]
        if self._dequant_key_cache is not None and max_length < self._dequant_len:
            self._dequant_key_cache = self._dequant_key_cache[:, :, :max_length, :]
            self._dequant_val_cache = self._dequant_val_cache[:, :, :max_length, :]
            self._dequant_len = max_length
        self._seq_len = max_length

    def memory_usage_bits(self) -> Dict[str, Any]:
        if self._seq_len == 0 or self._head_dim is None:
            return {
                "key_bits": 0,
                "value_bits": 0,
                "total_bits": 0,
                "fp16_baseline_bits": 0,
                "compression_ratio": 1.0,
            }
        d = self._head_dim
        n_heads = self._num_heads
        batch = self._batch_size
        total_tokens = self._seq_len * n_heads * batch
        fp16_tokens = min(self.fp16_window, self._seq_len) * n_heads * batch
        compressed_tokens = total_tokens - fp16_tokens

        key_bits_compressed = compressed_tokens * (self.key_bits * d + d + 32)
        key_bits_fp16 = fp16_tokens * d * 16
        val_bits_compressed = compressed_tokens * (self.val_bits * d + 16)
        val_bits_fp16 = fp16_tokens * d * 16

        total_key = key_bits_compressed + key_bits_fp16
        total_val = val_bits_compressed + val_bits_fp16
        total = total_key + total_val
        fp16_baseline = total_tokens * d * 16 * 2

        return {
            "key_bits": total_key,
            "value_bits": total_val,
            "total_bits": total,
            "fp16_baseline_bits": fp16_baseline,
            "compression_ratio": fp16_baseline / total if total > 0 else 1.0,
        }


# ---------------------------------------------------------------------------
# CrossLayerKVCache
# ---------------------------------------------------------------------------


class CrossLayerKVCache:
    """KV cache with cross-layer resource sharing for additional memory savings.

    Groups adjacent layers into clusters of ``group_size``. Within each group:
    - All layers share a single LloydMaxCodebook and rotation matrix
    - Each layer still quantizes its own KV vectors independently
    - Quality is identical to per-layer quantization (sharing is lossless
      because codebooks depend only on (d, bits), not on layer identity)

    Memory savings come from reduced per-layer overhead:
    - QR rotation: saves (group_size - 1) * d^2 * 4 bytes per group
    - WHT rotation: saves (group_size - 1) * d * 4 bytes per group
    - Codebook: saves (group_size - 1) * n_levels * 4 bytes per group

    For a 36-layer model, d=128, group_size=4, QR rotation:
    - Standard: 36 rotations * 64 KB = 2.25 MB
    - Shared: 9 rotations * 64 KB = 0.56 MB
    - Saving: 1.69 MB on rotation storage alone

    This stacks with the existing quantization compression (5.0x for 3-bit).

    Note: This does NOT implement delta coding (storing KV deltas between
    layers) because adjacent-layer KV vectors have near-zero correlation
    (r=0.001). The sharing is on the quantization infrastructure, not on
    the KV values themselves.

    Duck-types the HuggingFace Cache protocol.

    Args:
        group_size: Number of adjacent layers sharing resources (default: 2).
        key_bits: Bits for key quantization (default: 3).
        val_bits: Bits for value quantization (default: 3).
        fp16_window: Recent tokens at FP16 (default: 64).
        anchor_strategy: Anchor placement strategy (default: "boundary").
        anchor_interval: Interval for "fixed" strategy (default: 12).
        num_layers: Total transformer layers. Required for non-fixed strategies.
        seed: Random seed (default: 42).
        use_norm_correction: Apply norm correction (default: True).
        use_residual_quant: Apply 1-bit residual signs (default: True).
    """

    is_compileable = False

    def __init__(
        self,
        group_size: int = 2,
        key_bits: int = 3,
        val_bits: int = 3,
        fp16_window: int = 64,
        anchor_strategy: str = "boundary",
        anchor_interval: int = 12,
        num_layers: Optional[int] = None,
        seed: int = 42,
        use_norm_correction: bool = True,
        use_residual_quant: bool = True,
    ):
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        if not (1 <= key_bits <= 8):
            raise ValueError(f"key_bits must be 1-8, got {key_bits}")
        if not (1 <= val_bits <= 8):
            raise ValueError(f"val_bits must be 1-8, got {val_bits}")
        if fp16_window < 0:
            raise ValueError(f"fp16_window must be >= 0, got {fp16_window}")
        if anchor_strategy not in ANCHOR_STRATEGIES:
            raise ValueError(
                f"Unknown anchor_strategy: '{anchor_strategy}'. "
                f"Must be one of {ANCHOR_STRATEGIES}"
            )
        if anchor_strategy in ("boundary", "gradient") and num_layers is None:
            raise ValueError(
                f"num_layers is required for anchor_strategy='{anchor_strategy}'"
            )

        self.group_size = group_size
        self.key_bits = key_bits
        self.val_bits = val_bits
        self.fp16_window = fp16_window
        self.anchor_strategy = anchor_strategy
        self.anchor_interval = anchor_interval
        self.num_layers = num_layers
        self.seed = seed
        self.use_norm_correction = use_norm_correction
        self.use_residual_quant = use_residual_quant

        # Pre-compute anchor schedule
        self._anchor_schedule: Optional[List[Tuple[bool, int]]] = None
        if num_layers is not None:
            self._anchor_schedule = compute_anchor_schedule(
                num_layers=num_layers,
                anchor_strategy=anchor_strategy,
                anchor_interval=anchor_interval,
                base_key_bits=key_bits,
            )

        # Shared resources per group (lazily initialized)
        # group_idx -> (key_codebook, val_codebook, rotation, wht_params, rotation_type)
        self._group_resources: Dict[int, Tuple] = {}

        self._layers: List[_SharedResourceLayer | _FP16Layer] = []

    def _is_anchor_layer(self, idx: int) -> bool:
        """Return True if layer idx should be stored at FP16."""
        if self._anchor_schedule is not None and idx < len(self._anchor_schedule):
            return self._anchor_schedule[idx][0]
        return self.anchor_interval > 0 and idx % self.anchor_interval == 0

    def _layer_key_bits(self, idx: int) -> int:
        """Return key bit-width for layer idx."""
        if self._anchor_schedule is not None and idx < len(self._anchor_schedule):
            return self._anchor_schedule[idx][1]
        return self.key_bits

    def _get_group_resources(
        self,
        group_idx: int,
        head_dim: int,
        device: str | torch.device,
    ) -> Tuple[LloydMaxCodebook, LloydMaxCodebook, Optional[torch.Tensor], Optional[dict], str]:
        """Get or create shared resources for a group.

        Returns:
            Tuple of (key_codebook, val_codebook, rotation, wht_params, rotation_type).
        """
        if group_idx in self._group_resources:
            return self._group_resources[group_idx]

        d = head_dim
        is_pow2 = d > 0 and (d & (d - 1)) == 0
        rotation_type = "wht" if is_pow2 else "qr"

        # One seed per group (not per layer)
        group_seed = self.seed + group_idx * 1000

        if rotation_type == "wht":
            wht_params = generate_wht_rotation(d, seed=group_seed, device=str(device))
            rotation = None
        else:
            rotation = generate_rotation_matrix(d, seed=group_seed, device=str(device))
            wht_params = None

        key_codebook = LloydMaxCodebook(d=d, bits=self.key_bits).to(str(device))
        val_codebook = LloydMaxCodebook(d=d, bits=self.val_bits).to(str(device))

        resources = (key_codebook, val_codebook, rotation, wht_params, rotation_type)
        self._group_resources[group_idx] = resources
        return resources

    def _make_layer(
        self,
        idx: int,
        head_dim: int,
        device: str | torch.device,
    ) -> _SharedResourceLayer | _FP16Layer:
        """Create the appropriate layer type for index idx."""
        if self._is_anchor_layer(idx):
            return _FP16Layer()

        group_idx = idx // self.group_size
        key_cb, val_cb, rotation, wht_params, rot_type = self._get_group_resources(
            group_idx, head_dim, device,
        )

        return _SharedResourceLayer(
            key_codebook=key_cb,
            val_codebook=val_cb,
            rotation=rotation,
            wht_params=wht_params,
            rotation_type=rot_type,
            key_bits=self._layer_key_bits(idx),
            val_bits=self.val_bits,
            fp16_window=self.fp16_window,
            use_norm_correction=self.use_norm_correction,
            use_residual_quant=self.use_residual_quant,
        )

    # ---- HF Cache protocol ----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress/store new KV pairs for a layer, return full cache.

        Args:
            key_states: [batch, num_heads, new_seq, head_dim]
            value_states: [batch, num_heads, new_seq, head_dim]
            layer_idx: Which transformer layer this belongs to.
            cache_kwargs: Additional HF cache kwargs (ignored).

        Returns:
            Tuple of (all_keys, all_values) tensors.
        """
        head_dim = key_states.shape[3]
        device = key_states.device
        while len(self._layers) <= layer_idx:
            self._layers.append(
                self._make_layer(len(self._layers), head_dim, device)
            )
        return self._layers[layer_idx].update(key_states, value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._layers):
            return 0
        return self._layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_mask_sizes(
        self,
        cache_position: torch.Tensor,
        layer_idx: int = 0,
    ) -> Tuple[int, int]:
        if layer_idx >= len(self._layers):
            return cache_position.shape[0], 0
        query_length = cache_position.shape[0]
        kv_length = self._layers[layer_idx].get_seq_length() + query_length
        return kv_length, 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for layer in self._layers:
            layer.reorder(beam_idx)

    def crop(self, max_length: int) -> None:
        for layer in self._layers:
            layer.crop(max_length)

    def reset(self) -> None:
        for layer in self._layers:
            layer.clear()

    @property
    def seen_tokens(self) -> int:
        return self._layers[0].get_seq_length() if self._layers else 0

    @property
    def is_initialized(self) -> bool:
        return len(self._layers) > 0

    @property
    def is_sliding(self) -> list[bool]:
        return [False] * max(len(self._layers), 1)

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):
        for layer in self._layers:
            keys, values = layer._dequantize_all()
            yield keys, values, None

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self._layers):
            raise IndexError(
                f"Layer {layer_idx} not in cache (have {len(self._layers)} layers)"
            )
        return self._layers[layer_idx]._dequantize_all()

    def __contains__(self, idx: int) -> bool:
        return 0 <= idx < len(self._layers)

    # ---- Reporting ----

    def resource_sharing_report(self) -> Dict[str, Any]:
        """Report resource sharing statistics.

        Returns:
            Dict with sharing metrics including memory savings from sharing.
        """
        n_layers = len(self._layers)
        n_groups = len(self._group_resources)
        n_anchor = sum(1 for i in range(n_layers) if self._is_anchor_layer(i))
        n_compressed = n_layers - n_anchor

        # Count how many layers actually share each group's resources
        layers_per_group: Dict[int, int] = {}
        for i in range(n_layers):
            if not self._is_anchor_layer(i):
                g = i // self.group_size
                layers_per_group[g] = layers_per_group.get(g, 0) + 1

        # Estimate memory savings
        # Without sharing: each compressed layer has its own codebook + rotation
        # With sharing: one per group
        # Rotation: d^2 * 4 bytes (QR) or d * 4 bytes (WHT)
        head_dim = None
        rot_type = None
        for layer in self._layers:
            if isinstance(layer, _SharedResourceLayer) and layer._head_dim is not None:
                head_dim = layer._head_dim
                rot_type = layer._rotation_type
                break

        rotation_bytes_per = 0
        codebook_bytes_per = 0
        if head_dim is not None:
            if rot_type == "wht":
                rotation_bytes_per = head_dim * 4  # sign vector
            else:
                rotation_bytes_per = head_dim * head_dim * 4  # d x d matrix
            # Key + val codebooks
            codebook_bytes_per = (
                (1 << self.key_bits) * 4  # key centroids
                + (1 << self.key_bits - 1) * 4  # key boundaries
                + (1 << self.val_bits) * 4  # val centroids
                + (1 << self.val_bits - 1) * 4  # val boundaries
            ) if self.key_bits > 0 and self.val_bits > 0 else 0

        without_sharing = n_compressed * (rotation_bytes_per + codebook_bytes_per)
        with_sharing = n_groups * (rotation_bytes_per + codebook_bytes_per)

        return {
            "num_layers": n_layers,
            "num_groups": n_groups,
            "group_size": self.group_size,
            "num_anchor_layers": n_anchor,
            "num_compressed_layers": n_compressed,
            "layers_per_group": layers_per_group,
            "rotation_type": rot_type,
            "head_dim": head_dim,
            "rotation_bytes_without_sharing": without_sharing,
            "rotation_bytes_with_sharing": with_sharing,
            "rotation_bytes_saved": without_sharing - with_sharing,
            "sharing_ratio": without_sharing / with_sharing if with_sharing > 0 else 1.0,
        }

    def memory_savings(self) -> Dict[str, Any]:
        """Report memory usage and savings across all layers."""
        per_layer = []
        total_compressed = 0
        total_fp16 = 0

        for i, layer in enumerate(self._layers):
            stats = layer.memory_usage_bits()
            layer_key_bits = self._layer_key_bits(i)
            per_layer.append({
                "layer": i,
                "is_anchor": self._is_anchor_layer(i),
                "key_bits": layer_key_bits,
                "group": i // self.group_size,
                **stats,
            })
            total_compressed += stats["total_bits"]
            total_fp16 += stats["fp16_baseline_bits"]

        return {
            "per_layer": per_layer,
            "total_compressed_bits": total_compressed,
            "total_fp16_bits": total_fp16,
            "overall_compression_ratio": (
                total_fp16 / total_compressed if total_compressed > 0 else 1.0
            ),
            "config": {
                "group_size": self.group_size,
                "key_bits": self.key_bits,
                "val_bits": self.val_bits,
                "fp16_window": self.fp16_window,
                "anchor_strategy": self.anchor_strategy,
                "use_residual_quant": self.use_residual_quant,
            },
            "num_layers": len(self._layers),
        }

    def config_summary(self) -> str:
        """Return a human-readable configuration summary."""
        n_layers = len(self._layers)
        n_anchor = sum(1 for i in range(n_layers) if self._is_anchor_layer(i))
        n_groups = len(self._group_resources)
        rq_desc = "+ 1b residual signs" if self.use_residual_quant else "(no residual signs)"
        return (
            f"CrossLayerKVCache: {self.key_bits}b keys {rq_desc}, "
            f"{self.val_bits}b values, group_size={self.group_size}, "
            f"{n_groups} shared groups, "
            f"{n_anchor}/{n_layers} anchor layers ({self.anchor_strategy})"
        )
