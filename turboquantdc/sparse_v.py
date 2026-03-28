"""Attention-gated sparse value dequantization for TurboQuant KV cache.

When computing attention-weighted value sums at long context, most softmax
weights are negligible (90%+ below 1e-6 at 32K+). This module skips the
dequantization of those value vectors, saving compute with zero quality loss.

This is NOT compression --- the values are still stored. We simply skip
decoding values that contribute negligibly to the output.

From TheTom/turboquant_plus: +22.8% decode speed at 32K,
zero measurable quality loss (PPL delta = 0.0000).

Usage:
    cache = TurboQuantKVCache(d_key=128, d_value=128, bits=3)
    # ... append keys and values ...
    sv = SparseVAttention(cache, threshold=1e-6)
    output = sv.attend(queries)
    print(sv.last_stats)  # {'sparsity_ratio': 0.92, ...}

    # Or use the functional API:
    output = sparse_attention(cache, queries, threshold=1e-6)
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from .kv_cache import TurboQuantKVCache


class SparseVAttention:
    """Attention-gated value dequantization for TurboQuant KV cache.

    Instead of dequantizing ALL value vectors and then computing the
    weighted sum, this class:
    1. Computes attention scores (already done with compressed keys)
    2. Applies softmax
    3. Identifies which positions have weight >= threshold
    4. Only dequantizes values at those positions
    5. Computes the weighted sum using only the significant values

    At long context (32K+), 90%+ of softmax weights are < 1e-6,
    so this skips 90% of value dequantization work.

    Args:
        cache: A TurboQuantKVCache instance with stored key-value pairs.
        threshold: Minimum softmax weight to trigger dequantization.
            Positions with weight below this are skipped. Default 1e-6.
    """

    def __init__(self, cache: TurboQuantKVCache, threshold: float = 1e-6):
        self.cache = cache
        self.threshold = threshold
        self.last_stats: Dict[str, object] = {
            "sparsity_ratio": 0.0,
            "positions_decoded": 0,
            "total_positions": 0,
        }

    def attend(
        self, queries: torch.Tensor, scale: Optional[float] = None
    ) -> torch.Tensor:
        """Full attention computation with sparse V dequantization.

        Args:
            queries: Query vectors, shape (n_queries, d_key) or (d_key,).
            scale: Attention scale factor. Default: 1/sqrt(d_key).

        Returns:
            Attention-weighted value sum, shape (n_queries, d_value) or
            (d_value,) when input is 1-D.
        """
        squeeze = queries.dim() == 1
        if squeeze:
            queries = queries.unsqueeze(0)

        n_total = self.cache.seq_len

        # Handle empty cache
        if n_total == 0:
            self.last_stats = {
                "sparsity_ratio": 0.0,
                "positions_decoded": 0,
                "total_positions": 0,
            }
            out = torch.zeros(
                queries.shape[0], self.cache.d_value, device=queries.device
            )
            if squeeze:
                out = out.squeeze(0)
            return out

        # 1. Get attention scores from cache (uses TurboQuant compressed keys)
        scores = self.cache.attention_scores(queries)  # (n_queries, seq_len)

        # 2. Apply softmax with scaling
        if scale is None:
            scale = 1.0 / math.sqrt(self.cache.d_key)
        weights = torch.softmax(scores * scale, dim=-1)  # (n_queries, seq_len)

        # 3. Find significant positions (weight >= threshold)
        mask = weights >= self.threshold  # (n_queries, seq_len)

        # Get unique significant positions across all queries
        significant_positions = mask.any(dim=0)  # (seq_len,)

        # 4. Selective value dequantization
        values = self._selective_dequant(significant_positions)  # (seq_len, d_value)

        # 5. Compute weighted sum (zero out insignificant weights)
        weights = weights * mask.float()
        # Re-normalize after zeroing to maintain probability distribution
        weight_sums = weights.sum(dim=-1, keepdim=True)
        weights = weights / (weight_sums + 1e-10)

        output = weights @ values  # (n_queries, d_value)

        # Store stats
        n_sig = significant_positions.sum().item()
        self.last_stats = {
            "sparsity_ratio": 1.0 - (n_sig / max(n_total, 1)),
            "positions_decoded": n_sig,
            "total_positions": n_total,
        }

        if squeeze:
            output = output.squeeze(0)
        return output

    def _selective_dequant(self, mask: torch.Tensor) -> torch.Tensor:
        """Dequantize only the value positions indicated by mask.

        For positions where mask is False, return zero vectors.
        This avoids the expensive dequantize() call for ~90% of positions
        at long context.

        Args:
            mask: Boolean tensor of shape (seq_len,). True = dequantize.

        Returns:
            Values tensor of shape (seq_len, d_value). Unmasked positions
            are zero vectors.
        """
        seq_len = mask.shape[0]
        device = mask.device

        # Pre-allocate output with zeros
        values = torch.zeros(seq_len, self.cache.d_value, device=device)

        n_sig = mask.sum().item()
        if n_sig == 0:
            return values

        # Gather all value indices and norms from the cache's internal storage
        # (mirrors the logic in kv_cache.get_values())
        all_indices = []
        all_norms = []
        for idx_tensor, norm_tensor in zip(
            self.cache._value_indices, self.cache._value_norms
        ):
            if idx_tensor.dim() == 1:
                all_indices.append(idx_tensor.unsqueeze(0))
                all_norms.append(norm_tensor.unsqueeze(0))
            else:
                all_indices.append(idx_tensor)
                all_norms.append(norm_tensor)

        indices_cat = torch.cat(all_indices, dim=0)  # (seq_len, d_value)
        norms_cat = torch.cat(all_norms, dim=0)  # (seq_len,)

        # Extract only the significant subset
        sig_indices = indices_cat[mask]  # (n_sig, d_value)
        sig_norms = norms_cat[mask]  # (n_sig,)

        # Dequantize only the significant positions
        sig_values_normalized = self.cache.value_quantizer.dequantize(sig_indices)

        # Rescale by original norms
        if sig_norms.dim() == 1:
            sig_norms = sig_norms.unsqueeze(-1)
        sig_values = sig_values_normalized * sig_norms  # (n_sig, d_value)

        # Place back into the full-size output
        values[mask] = sig_values

        return values


def sparse_attention(
    cache: TurboQuantKVCache,
    queries: torch.Tensor,
    threshold: float = 1e-6,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Functional API for sparse V attention.

    Convenience wrapper that creates a SparseVAttention instance and runs
    a single attend() call.

    Args:
        cache: A TurboQuantKVCache instance.
        queries: Query vectors, shape (n_queries, d_key) or (d_key,).
        threshold: Minimum softmax weight for dequantization. Default 1e-6.
        scale: Attention scale factor. Default: 1/sqrt(d_key).

    Returns:
        Attention-weighted value sum, same shape semantics as attend().
    """
    sv = SparseVAttention(cache, threshold)
    return sv.attend(queries, scale)
