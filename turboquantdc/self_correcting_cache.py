"""Self-correcting KV cache with periodic refresh to prevent error accumulation.

The fundamental problem with KV cache compression isn't per-vector error --
it's that errors ACCUMULATE during autoregressive generation. Each
slightly-wrong attention score produces slightly-wrong output, which becomes
the next step's input, leading to drift and eventually repetition loops.

This module implements periodic cache refresh, analogous to I-frames in
video codecs. Every ``refresh_interval`` tokens:

1. Identify the ``refresh_count`` most-attended cached tokens (via key norms).
2. Re-quantize those positions from the stored compressed representation
   with fresh norm correction to reset drift.
3. Continue generation.

The refresh uses **norm correction**: instead of recomputing KV from scratch
(which would require a model forward pass), it re-normalizes cached vectors
to correct for quantization-induced norm drift. This is cheap (no model
required) and effective because:

- Quantization error compounds primarily through norm drift (norms grow or
  shrink slightly per quantize/dequantize cycle).
- Correcting the norms of high-importance tokens (those receiving the most
  attention weight) has outsized quality benefit.
- Overhead is negligible: just norm recomputation on a handful of vectors.

Usage::

    from turboquantdc import GenerationCache
    from turboquantdc.self_correcting_cache import SelfCorrectingCache

    inner = GenerationCache()
    cache = SelfCorrectingCache(inner, refresh_interval=50, refresh_count=5)
    output = model.generate(inputs, past_key_values=cache, max_new_tokens=200)

Duck-types the HuggingFace Cache protocol by delegating all methods to the
inner cache, so it works as a drop-in wrapper around any cache type
(GenerationCache, EvictionCache, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .generation_cache import GenerationCache, _CompressedLayer, _FP16Layer


class SelfCorrectingCache:
    """KV cache wrapper with periodic norm-correction refresh.

    Wraps any inner cache (GenerationCache, EvictionCache, etc.) and
    periodically refreshes the most-attended tokens' norms to bound
    error accumulation during autoregressive generation.

    The importance of each token is tracked via key norms -- a cheap
    proxy for attention weight that doesn't require explicit attention
    score logging. Tokens with larger key norms act as "attention
    magnets" and are the highest-priority refresh targets.

    Args:
        inner_cache: The underlying KV cache (GenerationCache, EvictionCache,
            or any object following the HF Cache protocol).
        refresh_interval: Refresh every N tokens generated at layer 0
            (default: 50). Must be >= 1.
        refresh_count: Number of top-attended tokens to refresh each
            cycle (default: 5). Must be >= 1. Clamped to available
            token count if larger.
    """

    is_compileable = False

    def __init__(
        self,
        inner_cache,
        refresh_interval: int = 50,
        refresh_count: int = 5,
    ):
        if refresh_interval < 1:
            raise ValueError(
                f"refresh_interval must be >= 1, got {refresh_interval}"
            )
        if refresh_count < 1:
            raise ValueError(
                f"refresh_count must be >= 1, got {refresh_count}"
            )

        self.inner = inner_cache
        self.refresh_interval = refresh_interval
        self.refresh_count = refresh_count

        self.tokens_since_refresh: int = 0
        self.total_refreshes: int = 0
        self._total_tokens_refreshed: int = 0

        # Per-layer importance tracking: layer_idx -> list of key norms
        # Each entry is the mean key norm (across heads and batch) for
        # that token position. Appended on each update.
        self._importance_tracker: Dict[int, List[float]] = {}

    # ---- Core update with refresh trigger ----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress/store new KV pairs, track importance, maybe refresh.

        Args:
            key_states: ``[batch, num_heads, new_seq, head_dim]``
            value_states: ``[batch, num_heads, new_seq, head_dim]``
            layer_idx: Which transformer layer this belongs to.
            cache_kwargs: Additional HF cache kwargs (passed to inner).

        Returns:
            Tuple of ``(all_keys, all_values)`` tensors from inner cache.
        """
        result = self.inner.update(
            key_states, value_states, layer_idx, cache_kwargs,
        )

        # Track importance via key norms
        self._track_importance(key_states, layer_idx)

        # Count tokens only at layer 0 to avoid double-counting
        if layer_idx == 0:
            new_seq = key_states.shape[2]
            self.tokens_since_refresh += new_seq

            if self.tokens_since_refresh >= self.refresh_interval:
                self._perform_refresh()
                self.tokens_since_refresh = 0

        return result

    # ---- Importance tracking ----

    def _track_importance(
        self, key_states: torch.Tensor, layer_idx: int,
    ) -> None:
        """Track cumulative token importance via key norms.

        Key norm is a cheap, model-free proxy for attention weight.
        Tokens with large key norms dominate softmax attention via the
        dot product mechanism, making them the highest-priority refresh
        targets.

        Args:
            key_states: ``[batch, num_heads, new_seq, head_dim]``
            layer_idx: Layer index for this update.
        """
        with torch.no_grad():
            # [batch, heads, seq] -> mean across batch and heads -> [seq]
            per_token_norms = key_states.float().norm(dim=-1)
            mean_norms = per_token_norms.mean(dim=(0, 1))

            if layer_idx not in self._importance_tracker:
                self._importance_tracker[layer_idx] = []

            for i in range(mean_norms.shape[0]):
                self._importance_tracker[layer_idx].append(
                    mean_norms[i].item()
                )

    def _get_top_positions(
        self, layer_idx: int, count: int,
    ) -> List[int]:
        """Return the indices of the top-``count`` most important tokens.

        Importance is measured by cumulative key norm. Positions are
        returned in descending importance order.

        Args:
            layer_idx: Layer to query.
            count: Number of top positions to return.

        Returns:
            List of token position indices, length = min(count, available).
        """
        if layer_idx not in self._importance_tracker:
            return []

        norms = self._importance_tracker[layer_idx]
        if not norms:
            return []

        count = min(count, len(norms))
        norm_tensor = torch.tensor(norms, dtype=torch.float32)
        _, top_indices = torch.topk(norm_tensor, k=count)
        return top_indices.tolist()

    # ---- Norm correction refresh ----

    def _perform_refresh(self) -> None:
        """Refresh the most-attended tokens across all layers.

        For each layer in the inner cache, identifies the top-N tokens
        by key norm and re-applies norm correction to their compressed
        representation. This resets quantization-induced norm drift
        without requiring a model forward pass.

        No-op if the cache is empty or has no compressed layers.
        """
        if not hasattr(self.inner, '_layers') or not self.inner._layers:
            return

        any_refreshed = False

        for layer_idx, layer in enumerate(self.inner._layers):
            # Only refresh compressed layers, not FP16 anchors
            if isinstance(layer, _FP16Layer):
                continue
            if not isinstance(layer, _CompressedLayer):
                continue
            if layer._seq_len == 0:
                continue

            top_positions = self._get_top_positions(
                layer_idx, self.refresh_count,
            )
            if not top_positions:
                continue

            n_refreshed = self._refresh_layer_norms(layer, top_positions)
            self._total_tokens_refreshed += n_refreshed
            if n_refreshed > 0:
                any_refreshed = True

        if any_refreshed:
            self.total_refreshes += 1

    def _refresh_layer_norms(
        self,
        layer: _CompressedLayer,
        positions: List[int],
    ) -> int:
        """Re-apply norm correction for specific token positions in a layer.

        For each position, recomputes the reconstruction-to-original norm
        ratio and updates the stored corrected norms. This compensates
        for drift that accumulates when norms are stored as scalar
        approximations.

        Also invalidates the layer's dequantization cache so the next
        access uses the refreshed norms.

        Args:
            layer: The compressed layer to refresh.
            positions: Token position indices to refresh.

        Returns:
            Number of tokens actually refreshed.
        """
        if not layer._key_indices or layer._key_codebook is None:
            return 0

        # Concatenate compressed storage to access by position
        all_k_idx = torch.cat(layer._key_indices, dim=2)
        all_k_norms = torch.cat(layer._key_norms, dim=2)
        all_v_idx = torch.cat(layer._val_indices, dim=2)
        all_v_norms = torch.cat(layer._val_norms, dim=2)

        seq_len = all_k_idx.shape[2]
        valid_positions = [p for p in positions if 0 <= p < seq_len]
        if not valid_positions:
            return 0

        # Re-derive correct norms from the stored indices
        # by reconstructing in the rotated domain and measuring
        # the reconstruction norm vs stored norm.
        d = layer._head_dim
        k_codebook = layer._key_codebook
        v_codebook = layer._val_codebook

        for pos in valid_positions:
            # Key norm refresh
            k_idx_slice = all_k_idx[:, :, pos, :]  # [batch, heads, d]
            flat_k = k_idx_slice.reshape(-1, d)
            recon_rotated = k_codebook.centroids[flat_k.long()]

            # Apply residual correction if available
            if layer.use_residual_quant and layer._key_res_signs:
                all_k_rsigns = torch.cat(layer._key_res_signs, dim=2)
                all_k_rscales = torch.cat(layer._key_res_scales, dim=2)
                rsigns = all_k_rsigns[:, :, pos, :].reshape(-1, d)
                rscales = all_k_rscales[:, :, pos].reshape(-1, 1)
                recon_rotated = recon_rotated + rsigns * rscales

            # Unrotate
            if layer._rotation_type == "wht":
                from .rotation import apply_wht_rotation
                recon = apply_wht_rotation(
                    recon_rotated.float(), layer._wht_params, inverse=True,
                )
            else:
                recon = torch.matmul(
                    recon_rotated.float(), layer._rotation.float().T,
                )

            # The stored norm should produce correct magnitude when
            # applied to the unit-norm reconstruction. Re-derive it.
            recon_norm = recon.norm(dim=-1, keepdim=True)

            # Current stored norm
            current_norm = all_k_norms[:, :, pos]  # [batch, heads]
            current_flat = current_norm.reshape(-1, 1)

            # Apply a mild correction: bring the norm closer to
            # self-consistency. The original norm is lost, but we can
            # check that stored_norm * unit_recon has the right magnitude.
            target_magnitude = current_flat.abs()
            actual_magnitude = (recon_norm * current_flat.abs()).clamp(min=1e-8)
            correction = (target_magnitude / actual_magnitude).clamp(0.8, 1.2)
            refreshed_norm = current_flat * correction

            all_k_norms[:, :, pos] = refreshed_norm.reshape(
                current_norm.shape
            )

            # Value norm refresh (same logic, simpler -- no residual)
            v_idx_slice = all_v_idx[:, :, pos, :]
            flat_v = v_idx_slice.reshape(-1, d)
            v_recon_rotated = v_codebook.centroids[flat_v.long()]
            if layer._rotation_type == "wht":
                v_recon = apply_wht_rotation(
                    v_recon_rotated.float(), layer._wht_params, inverse=True,
                )
            else:
                v_recon = torch.matmul(
                    v_recon_rotated.float(), layer._rotation.float().T,
                )
            v_recon_norm = v_recon.norm(dim=-1, keepdim=True)
            current_v_norm = all_v_norms[:, :, pos].reshape(-1, 1)
            v_target = current_v_norm.abs()
            v_actual = (v_recon_norm * current_v_norm.abs()).clamp(min=1e-8)
            v_correction = (v_target / v_actual).clamp(0.8, 1.2)
            all_v_norms[:, :, pos] = (
                (current_v_norm * v_correction)
                .reshape(all_v_norms[:, :, pos].shape)
            )

        # Write back consolidated norms
        layer._key_norms = [all_k_norms]
        layer._val_norms = [all_v_norms]

        # Also consolidate indices to match (single chunk)
        layer._key_indices = [all_k_idx]
        layer._val_indices = [all_v_idx]
        if layer._key_res_signs:
            layer._key_res_signs = [torch.cat(layer._key_res_signs, dim=2)]
            layer._key_res_scales = [torch.cat(layer._key_res_scales, dim=2)]
        if layer._key_means:
            layer._key_means = [torch.cat(layer._key_means, dim=2)]

        # Invalidate dequantization cache so next access uses refreshed norms
        layer._dequant_key_cache = None
        layer._dequant_val_cache = None
        layer._dequant_len = 0

        return len(valid_positions)

    # ---- Statistics ----

    def refresh_stats(self) -> Dict[str, Any]:
        """Return refresh statistics.

        Returns:
            Dict with total_refreshes, total_tokens_refreshed,
            tokens_since_refresh, and config.
        """
        return {
            "total_refreshes": self.total_refreshes,
            "total_tokens_refreshed": self._total_tokens_refreshed,
            "tokens_since_refresh": self.tokens_since_refresh,
            "config": {
                "refresh_interval": self.refresh_interval,
                "refresh_count": self.refresh_count,
            },
        }

    # ---- HF Cache protocol delegation ----

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return number of cached tokens for a layer."""
        return self.inner.get_seq_length(layer_idx)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Return max cache shape (delegates to inner)."""
        return self.inner.get_max_cache_shape(layer_idx)

    def get_mask_sizes(
        self,
        cache_position: torch.Tensor,
        layer_idx: int = 0,
    ) -> tuple[int, int]:
        """Return (kv_length, kv_offset) for attention mask generation."""
        return self.inner.get_mask_sizes(cache_position, layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder all layers for beam search."""
        self.inner.reorder_cache(beam_idx)

    def crop(self, max_length: int) -> None:
        """Truncate all layers to max_length tokens."""
        self.inner.crop(max_length)

    def reset(self) -> None:
        """Clear all cached data and reset refresh state."""
        self.inner.reset()
        self.tokens_since_refresh = 0
        self._importance_tracker.clear()

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat cache entries for beam search expansion."""
        self.inner.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select specific batch indices from the cache."""
        self.inner.batch_select_indices(indices)

    def memory_savings(self) -> Dict[str, Any]:
        """Report memory usage and savings (delegates to inner)."""
        return self.inner.memory_savings()

    @property
    def seen_tokens(self) -> int:
        """Number of tokens seen by the first layer."""
        return self.inner.seen_tokens

    @property
    def is_initialized(self) -> bool:
        """Return whether the cache has been populated."""
        return self.inner.is_initialized

    @property
    def is_sliding(self) -> list[bool]:
        """Return sliding window status per layer."""
        return self.inner.is_sliding

    def __len__(self) -> int:
        return len(self.inner)

    def __iter__(self):
        return iter(self.inner)

    def __getitem__(self, layer_idx: int):
        return self.inner[layer_idx]

    def __contains__(self, idx: int) -> bool:
        return idx in self.inner
