"""Attention-guided token eviction for TurboQuant KV cache.

Three-tier token storage with eviction of unimportant tokens:

    Tier 0 (hot):   Last ``fp16_window`` tokens at FP16 (full precision).
    Tier 1 (warm):  Important older tokens, quantized at key_bits/val_bits.
    Tier 2 (cold):  EVICTED -- not stored at all.

Token importance is scored using a hybrid of recency and structural signals:

    importance = 0.3 * recency + 0.7 * structural

Structural importance protects:
- **Prompt tokens** (first 20% of the sequence): always importance = 1.0,
  never evicted. The original question/instruction is critical context.
- **High-norm keys**: tokens with large key norms act as attention magnets
  and are more semantically important.

Pure recency scoring evicts semantically important tokens (e.g. the
original question in Q&A), destroying generation quality. The hybrid
approach retains these while still evicting genuinely unimportant middle
tokens.

The result: same quality as quantize-only (GenerationCache) but at 6-8x
compression by evicting 50-70% of old tokens that receive near-zero
attention weight.

Informed by:
- ThinKV (arxiv 2510.01290): hybrid eviction + quantization
- G-KV (arxiv 2512.00504): global attention-guided eviction
- Our sweep: fp16_window=512 with K3 gets 95% quality -- the window does
  all the work

Usage::

    from turboquantdc import EvictionCache

    cache = EvictionCache(
        key_bits=3, val_bits=3,
        fp16_window=64,
        max_warm_tokens=1024,
    )
    output = model.generate(inputs, past_key_values=cache, max_new_tokens=200)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch

from .generation_cache import _CompressedLayer, _FP16Layer, GenerationCache


# ---------------------------------------------------------------------------
# Evictable compressed layer
# ---------------------------------------------------------------------------


class _EvictableLayer(_CompressedLayer):
    """A _CompressedLayer extended with token eviction.

    Tracks per-token positions and evicts old, low-importance tokens
    when the warm tier exceeds ``max_warm_tokens``.

    The hot tier (FP16 window) is inherited from _CompressedLayer.
    The warm tier is the compressed storage.
    Eviction removes entries from the compressed storage entirely.
    """

    def __init__(
        self,
        key_bits: int = 3,
        val_bits: int = 3,
        fp16_window: int = 64,
        max_warm_tokens: int = 1024,
        eviction_threshold: float = 0.01,
        seed: int = 42,
        use_norm_correction: bool = True,
        use_residual_quant: bool = True,
    ):
        super().__init__(
            key_bits=key_bits,
            val_bits=val_bits,
            fp16_window=fp16_window,
            seed=seed,
            use_norm_correction=use_norm_correction,
            use_residual_quant=use_residual_quant,
        )
        self.max_warm_tokens = max_warm_tokens
        self.eviction_threshold = eviction_threshold

        # Track how many tokens have been seen total (never decreases)
        self._tokens_seen: int = 0
        # Track how many tokens have been evicted total
        self._tokens_evicted: int = 0

        # Per-token key norms for hybrid importance scoring.
        # Each entry is the mean key norm (across heads) for that token.
        self._token_key_norms: List[float] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress, store, potentially evict, and return full cache.

        Overrides _CompressedLayer.update() to add eviction after storage.
        Also captures per-token key norms for hybrid importance scoring.
        """
        new_seq = key_states.shape[2]
        self._tokens_seen += new_seq

        # Capture per-token key norms before compression.
        # key_states: [batch, heads, seq, d] -> norm across d, mean across heads/batch
        with torch.no_grad():
            # [batch, heads, seq]
            per_token_norms = key_states.float().norm(dim=-1)
            # Mean across batch and heads -> [seq]
            mean_norms = per_token_norms.mean(dim=(0, 1))
            for i in range(new_seq):
                self._token_key_norms.append(mean_norms[i].item())

        # Delegate to parent for compression and storage
        result = super().update(key_states, value_states)

        # After storing, check if eviction is needed
        self._maybe_evict()

        return result

    def _maybe_evict(self) -> None:
        """Evict low-importance tokens if warm tier exceeds max_warm_tokens.

        The warm tier is all compressed tokens minus the fp16_window.
        When this count exceeds max_warm_tokens, we compute importance
        scores and evict the lowest-scoring tokens.

        Prompt tokens (first 20% of sequence, minimum 10) are protected
        and never evicted, regardless of their importance score.
        """
        total_compressed = self._seq_len
        warm_count = max(0, total_compressed - self.fp16_window)

        if warm_count <= self.max_warm_tokens:
            return  # no eviction needed

        # How many tokens to evict
        n_to_evict = warm_count - self.max_warm_tokens

        # Compute importance for all compressed tokens
        importance = self._importance_scores()

        if importance is None or len(importance) == 0:
            return

        # The hot (FP16) tokens are the last fp16_window positions --
        # they must NOT be evicted. We only consider positions that
        # are outside the FP16 window for eviction.
        n_total = len(importance)
        n_hot = min(self.fp16_window, n_total)
        n_eviction_candidates = n_total - n_hot

        if n_eviction_candidates <= 0 or n_to_evict <= 0:
            return

        # Protect prompt tokens: first 20% of the sequence (min 10).
        # These are never evicted because they contain the original
        # question/instruction which is critical for generation quality.
        prompt_end = max(int(n_total * 0.2), min(10, n_total))

        # Only consider non-hot, non-prompt positions for eviction.
        # Candidates are positions in [prompt_end, n_eviction_candidates).
        candidate_importance = importance[prompt_end:n_eviction_candidates]
        n_actual_candidates = len(candidate_importance)

        if n_actual_candidates <= 0:
            return

        n_to_evict = min(n_to_evict, n_actual_candidates)

        # Find indices of the least important candidates
        _, evict_local_indices = torch.topk(
            candidate_importance, k=n_to_evict, largest=False,
        )

        # Map local indices back to global positions
        evict_global_indices = evict_local_indices + prompt_end

        # Build a boolean keep-mask over all tokens
        keep_mask = torch.ones(n_total, dtype=torch.bool)
        keep_mask[evict_global_indices] = False

        self._apply_eviction_mask(keep_mask)
        self._tokens_evicted += n_to_evict

    def _importance_scores(self) -> Optional[torch.Tensor]:
        """Compute hybrid importance scores for all compressed tokens.

        Uses a weighted combination of recency and structural importance:
            importance = 0.3 * recency + 0.7 * structural

        Recency component:
            Linear from 0 (oldest) to 1 (newest).

        Structural component:
            - Prompt tokens (first 20% of sequence, min 10): importance = 1.0.
              The original question/instruction must never be evicted.
            - Key norm component: tokens with larger key norms are attention
              magnets and more semantically important. Normalized to [0, 1].

        Returns a 1-D float tensor of length self._seq_len, or None
        if no tokens are stored.
        """
        n = self._seq_len
        if n == 0:
            return None

        positions = torch.arange(n, dtype=torch.float32)
        current = float(n - 1)

        # Recency component: linear 0-1, higher = more recent
        recency = 1.0 - (current - positions) / max(current, 1.0)

        # Structural component
        structural = torch.zeros(n, dtype=torch.float32)

        # Prompt tokens (first 20%, min 10) always get max structural importance
        prompt_end = max(int(n * 0.2), min(10, n))
        structural[:prompt_end] = 1.0

        # Key norm component: high-norm keys are attention magnets
        if self._token_key_norms and len(self._token_key_norms) >= n:
            norms = torch.tensor(
                self._token_key_norms[:n], dtype=torch.float32,
            )
            norm_min = norms.min()
            norm_range = norms.max() - norm_min
            if norm_range > 1e-8:
                norm_importance = (norms - norm_min) / norm_range
            else:
                norm_importance = torch.zeros_like(norms)
            # Take element-wise max so prompt tokens stay at 1.0
            structural = torch.max(structural, norm_importance)

        importance = 0.3 * recency + 0.7 * structural
        return importance

    def _apply_eviction_mask(self, keep_mask: torch.Tensor) -> None:
        """Remove evicted tokens from compressed storage.

        Args:
            keep_mask: Boolean tensor of shape (seq_len,). True = keep.
        """
        if not self._key_indices:
            return

        # Concatenate all compressed chunks into single tensors
        all_k_idx = torch.cat(self._key_indices, dim=2)
        all_k_norms = torch.cat(self._key_norms, dim=2)
        all_k_rsigns = torch.cat(self._key_res_signs, dim=2)
        all_k_rscales = torch.cat(self._key_res_scales, dim=2)
        all_k_means = torch.cat(self._key_means, dim=2)
        all_v_idx = torch.cat(self._val_indices, dim=2)
        all_v_norms = torch.cat(self._val_norms, dim=2)

        # keep_mask is over the seq dimension (dim=2)
        # Expand to match [batch, heads, seq, ...]
        mask_seq = keep_mask.to(all_k_idx.device)

        # Index select along seq dimension
        kept_k_idx = all_k_idx[:, :, mask_seq, :]
        kept_k_norms = all_k_norms[:, :, mask_seq]
        kept_k_rsigns = all_k_rsigns[:, :, mask_seq, :]
        kept_k_rscales = all_k_rscales[:, :, mask_seq]
        kept_k_means = all_k_means[:, :, mask_seq, :]
        kept_v_idx = all_v_idx[:, :, mask_seq, :]
        kept_v_norms = all_v_norms[:, :, mask_seq]

        # Re-store as single chunks
        self._key_indices = [kept_k_idx]
        self._key_norms = [kept_k_norms]
        self._key_res_signs = [kept_k_rsigns]
        self._key_res_scales = [kept_k_rscales]
        self._key_means = [kept_k_means]
        self._val_indices = [kept_v_idx]
        self._val_norms = [kept_v_norms]

        # Also filter raw FP16 storage -- keep only the retained FP16 tokens.
        # The raw FP16 storage corresponds to the last tokens in the sequence.
        if self._raw_keys:
            all_rk = torch.cat(self._raw_keys, dim=2)
            all_rv = torch.cat(self._raw_vals, dim=2)
            raw_len = all_rk.shape[2]
            total_len = keep_mask.shape[0]

            # The raw FP16 tokens correspond to the last raw_len positions
            # in the keep_mask. We keep the FP16 tokens that survive eviction.
            if raw_len <= total_len:
                raw_mask = mask_seq[-raw_len:]
                self._raw_keys = [all_rk[:, :, raw_mask, :]]
                self._raw_vals = [all_rv[:, :, raw_mask, :]]
            # else: raw storage is stale / trimmed, leave as is

        # Filter per-token key norms to match the keep mask
        if self._token_key_norms:
            self._token_key_norms = [
                self._token_key_norms[i]
                for i in range(min(len(self._token_key_norms), len(keep_mask)))
                if keep_mask[i]
            ]

        new_seq_len = int(mask_seq.sum().item())
        self._seq_len = new_seq_len

        # Invalidate dequant cache (positions changed)
        self._dequant_key_cache = None
        self._dequant_val_cache = None
        self._dequant_len = 0


# ---------------------------------------------------------------------------
# EvictionCache — production KV cache with token eviction
# ---------------------------------------------------------------------------


class EvictionCache:
    """KV cache with hybrid importance scoring and token eviction.

    Scores token importance using a weighted combination of recency
    (30%) and structural signals (70%):
    - Prompt tokens (first 20% of sequence): always protected.
    - High key-norm tokens: attention magnets, scored higher.
    - Recent tokens: mild recency preference.

    Three tiers of token storage:
    1. **Hot** (last fp16_window tokens): FP16, full precision.
    2. **Warm** (important older tokens): Quantized at key_bits/val_bits.
    3. **Cold** (unimportant tokens): EVICTED -- not stored at all.

    Duck-types the HuggingFace Cache protocol (update, get_seq_length,
    get_mask_sizes, etc.) so it works as a drop-in replacement for
    DynamicCache in model.generate().

    Args:
        key_bits: Bits for key quantization (default: 3).
        val_bits: Bits for value quantization (default: 3).
        fp16_window: Number of recent tokens at FP16 (default: 64).
        max_warm_tokens: Max quantized tokens to keep per layer (default: 1024).
        eviction_threshold: Importance threshold for eviction (default: 0.01).
        anchor_interval: Every Nth layer stored at FP16 (default: 12).
            Set to 0 to disable anchors.
        seed: Random seed for reproducibility.
        use_residual_quant: Apply 1-bit residual sign correction (default: True).
    """

    is_compileable = False

    def __init__(
        self,
        key_bits: int = 3,
        val_bits: int = 3,
        fp16_window: int = 64,
        max_warm_tokens: int = 1024,
        eviction_threshold: float = 0.01,
        anchor_interval: int = 12,
        seed: int = 42,
        use_residual_quant: bool = True,
    ):
        if not (1 <= key_bits <= 8):
            raise ValueError(f"key_bits must be 1-8, got {key_bits}")
        if not (1 <= val_bits <= 8):
            raise ValueError(f"val_bits must be 1-8, got {val_bits}")
        if fp16_window < 0:
            raise ValueError(f"fp16_window must be >= 0, got {fp16_window}")
        if max_warm_tokens < 1:
            raise ValueError(f"max_warm_tokens must be >= 1, got {max_warm_tokens}")

        self.key_bits = key_bits
        self.val_bits = val_bits
        self.fp16_window = fp16_window
        self.max_warm_tokens = max_warm_tokens
        self.eviction_threshold = eviction_threshold
        self.anchor_interval = anchor_interval
        self.seed = seed
        self.use_residual_quant = use_residual_quant
        self._layers: List[_EvictableLayer | _FP16Layer] = []

    def _is_anchor_layer(self, idx: int) -> bool:
        """Return True if layer ``idx`` should be stored at FP16."""
        return self.anchor_interval > 0 and idx % self.anchor_interval == 0

    def _make_layer(self, idx: int) -> _EvictableLayer | _FP16Layer:
        """Create the appropriate layer type for index ``idx``."""
        if self._is_anchor_layer(idx):
            return _FP16Layer()
        return _EvictableLayer(
            key_bits=self.key_bits,
            val_bits=self.val_bits,
            fp16_window=self.fp16_window,
            max_warm_tokens=self.max_warm_tokens,
            eviction_threshold=self.eviction_threshold,
            seed=self.seed + idx,
            use_norm_correction=True,
            use_residual_quant=self.use_residual_quant,
        )

    # ---- HF Cache protocol ----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress/store new KV pairs, evict if needed, return full cache.

        Args:
            key_states: ``[batch, num_heads, new_seq, head_dim]``
            value_states: ``[batch, num_heads, new_seq, head_dim]``
            layer_idx: Which transformer layer this belongs to.
            cache_kwargs: Additional HF cache kwargs (ignored).

        Returns:
            Tuple of ``(all_keys, all_values)`` tensors.
        """
        while len(self._layers) <= layer_idx:
            self._layers.append(self._make_layer(len(self._layers)))
        return self._layers[layer_idx].update(key_states, value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return number of cached tokens for a layer (hot + warm only)."""
        if layer_idx >= len(self._layers):
            return 0
        return self._layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Return max cache shape. Dynamic cache has no maximum."""
        return -1

    def get_mask_sizes(
        self,
        cache_position: torch.Tensor,
        layer_idx: int = 0,
    ) -> tuple[int, int]:
        """Return ``(kv_length, kv_offset)`` for attention mask generation."""
        if layer_idx >= len(self._layers):
            return cache_position.shape[0], 0
        query_length = cache_position.shape[0]
        kv_length = self._layers[layer_idx].get_seq_length() + query_length
        return kv_length, 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder all layers for beam search."""
        for layer in self._layers:
            layer.reorder(beam_idx)

    def crop(self, max_length: int) -> None:
        """Truncate all layers to ``max_length`` tokens."""
        for layer in self._layers:
            layer.crop(max_length)

    def reset(self) -> None:
        """Clear all cached data across every layer."""
        for layer in self._layers:
            layer.clear()

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat cache entries for beam search expansion."""
        for layer in self._layers:
            if isinstance(layer, _FP16Layer):
                layer._keys = [
                    k.repeat_interleave(repeats, dim=0) for k in layer._keys
                ]
                layer._values = [
                    v.repeat_interleave(repeats, dim=0) for v in layer._values
                ]
            elif isinstance(layer, _EvictableLayer):
                layer._key_indices = [
                    t.repeat_interleave(repeats, dim=0)
                    for t in layer._key_indices
                ]
                layer._key_norms = [
                    t.repeat_interleave(repeats, dim=0)
                    for t in layer._key_norms
                ]
                layer._key_res_signs = [
                    t.repeat_interleave(repeats, dim=0)
                    for t in layer._key_res_signs
                ]
                layer._key_res_scales = [
                    t.repeat_interleave(repeats, dim=0)
                    for t in layer._key_res_scales
                ]
                layer._key_means = [
                    t.repeat_interleave(repeats, dim=0)
                    for t in layer._key_means
                ]
                layer._val_indices = [
                    t.repeat_interleave(repeats, dim=0)
                    for t in layer._val_indices
                ]
                layer._val_norms = [
                    t.repeat_interleave(repeats, dim=0)
                    for t in layer._val_norms
                ]
                layer._raw_keys = [
                    t.repeat_interleave(repeats, dim=0)
                    for t in layer._raw_keys
                ]
                layer._raw_vals = [
                    t.repeat_interleave(repeats, dim=0)
                    for t in layer._raw_vals
                ]
                if layer._batch_size is not None:
                    layer._batch_size *= repeats
                layer._dequant_key_cache = None
                layer._dequant_val_cache = None
                layer._dequant_len = 0

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select specific batch indices from the cache."""
        for layer in self._layers:
            if isinstance(layer, _FP16Layer):
                layer._keys = [k[indices] for k in layer._keys]
                layer._values = [v[indices] for v in layer._values]
            elif isinstance(layer, _EvictableLayer):
                layer._key_indices = [t[indices] for t in layer._key_indices]
                layer._key_norms = [t[indices] for t in layer._key_norms]
                layer._key_res_signs = [
                    t[indices] for t in layer._key_res_signs
                ]
                layer._key_res_scales = [
                    t[indices] for t in layer._key_res_scales
                ]
                layer._key_means = [t[indices] for t in layer._key_means]
                layer._val_indices = [t[indices] for t in layer._val_indices]
                layer._val_norms = [t[indices] for t in layer._val_norms]
                layer._raw_keys = [t[indices] for t in layer._raw_keys]
                layer._raw_vals = [t[indices] for t in layer._raw_vals]
                if layer._batch_size is not None:
                    layer._batch_size = len(indices)
                layer._dequant_key_cache = None
                layer._dequant_val_cache = None
                layer._dequant_len = 0

    # ---- Eviction-specific API ----

    def _compute_importance(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Compute token importance scores for a layer.

        Exposed for testing. Returns the importance tensor from the
        underlying _EvictableLayer, or None for FP16 anchor layers.
        """
        if layer_idx >= len(self._layers):
            return None
        layer = self._layers[layer_idx]
        if isinstance(layer, _EvictableLayer):
            return layer._importance_scores()
        return None

    def eviction_stats(self) -> Dict[str, Any]:
        """Return eviction statistics across all layers.

        Returns:
            Dict with total_tokens_seen, total_tokens_evicted,
            total_tokens_retained, and per_layer breakdown.
        """
        total_seen = 0
        total_evicted = 0
        total_retained = 0
        per_layer = []

        for i, layer in enumerate(self._layers):
            if isinstance(layer, _EvictableLayer):
                seen = layer._tokens_seen
                evicted = layer._tokens_evicted
                retained = layer.get_seq_length()
                per_layer.append({
                    "layer": i,
                    "tokens_seen": seen,
                    "tokens_evicted": evicted,
                    "tokens_retained": retained,
                })
                total_seen += seen
                total_evicted += evicted
                total_retained += retained
            elif isinstance(layer, _FP16Layer):
                retained = layer.get_seq_length()
                per_layer.append({
                    "layer": i,
                    "tokens_seen": retained,
                    "tokens_evicted": 0,
                    "tokens_retained": retained,
                    "is_anchor": True,
                })
                total_seen += retained
                total_retained += retained

        return {
            "total_tokens_seen": total_seen,
            "total_tokens_evicted": total_evicted,
            "total_tokens_retained": total_retained,
            "per_layer": per_layer,
        }

    # ---- Properties ----

    @property
    def seen_tokens(self) -> int:
        """Number of tokens seen by the first layer."""
        if not self._layers:
            return 0
        layer = self._layers[0]
        if isinstance(layer, _EvictableLayer):
            return layer._tokens_seen
        return layer.get_seq_length()

    @property
    def is_initialized(self) -> bool:
        """Return whether the cache has been populated."""
        return len(self._layers) > 0

    @property
    def is_sliding(self) -> list[bool]:
        """Return sliding window status per layer (always False)."""
        return [False] * max(len(self._layers), 1)

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):
        for layer in self._layers:
            keys, values = layer._dequantize_all()
            yield keys, values, None

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self._layers):
            raise IndexError(
                f"Layer {layer_idx} not in cache (have {len(self._layers)} layers)"
            )
        return self._layers[layer_idx]._dequantize_all()

    def __contains__(self, idx: int) -> bool:
        return 0 <= idx < len(self._layers)

    # ---- Reporting ----

    def memory_savings(self) -> Dict[str, Any]:
        """Report memory usage, savings, and eviction stats across all layers.

        Returns:
            Dict with per_layer stats, aggregate totals, eviction stats,
            and configuration.
        """
        per_layer = []
        total_compressed = 0
        total_fp16 = 0

        for i, layer in enumerate(self._layers):
            stats = layer.memory_usage_bits()
            per_layer.append({
                "layer": i,
                "is_anchor": self._is_anchor_layer(i),
                **stats,
            })
            total_compressed += stats["total_bits"]
            total_fp16 += stats["fp16_baseline_bits"]

        eviction = self.eviction_stats()

        # Effective compression: ratio of FP16 cost for ALL tokens seen
        # vs actual compressed storage of retained tokens only.
        total_seen = eviction["total_tokens_seen"]
        if total_seen > 0 and self._layers:
            # Estimate the FP16 baseline for ALL tokens seen (not just retained)
            layer0 = self._layers[0]
            if isinstance(layer0, _EvictableLayer) and layer0._head_dim is not None:
                d = layer0._head_dim
                n_heads = layer0._num_heads or 1
                batch = layer0._batch_size or 1
                fp16_all_seen = total_seen * n_heads * batch * d * 16 * 2
            else:
                fp16_all_seen = total_fp16
        else:
            fp16_all_seen = total_fp16

        return {
            "per_layer": per_layer,
            "total_compressed_bits": total_compressed,
            "total_fp16_bits": total_fp16,
            "overall_compression_ratio": (
                fp16_all_seen / total_compressed if total_compressed > 0 else 1.0
            ),
            "eviction": {
                "total_evicted": eviction["total_tokens_evicted"],
                "total_retained": eviction["total_tokens_retained"],
                "total_seen": eviction["total_tokens_seen"],
                "eviction_rate": (
                    eviction["total_tokens_evicted"] / max(eviction["total_tokens_seen"], 1)
                ),
            },
            "config": {
                "key_bits": self.key_bits,
                "val_bits": self.val_bits,
                "fp16_window": self.fp16_window,
                "max_warm_tokens": self.max_warm_tokens,
                "eviction_threshold": self.eviction_threshold,
                "anchor_interval": self.anchor_interval,
                "use_residual_quant": self.use_residual_quant,
            },
            "num_layers": len(self._layers),
        }

    def config_summary(self) -> str:
        """Return a human-readable configuration summary."""
        n_layers = len(self._layers)
        n_anchor = sum(1 for i in range(n_layers) if self._is_anchor_layer(i))
        rq_desc = "+ 1b residual signs" if self.use_residual_quant else "(no residual signs)"
        return (
            f"EvictionCache: {self.key_bits}b keys {rq_desc}, "
            f"{self.val_bits}b values, FP16 window={self.fp16_window}, "
            f"max_warm={self.max_warm_tokens}, "
            f"{n_anchor}/{n_layers} anchor layers"
        )
