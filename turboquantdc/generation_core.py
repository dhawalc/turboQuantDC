from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch

from .codebook import LloydMaxCodebook
from .rotation import apply_wht_rotation, generate_qjl_matrix, generate_rotation_matrix, generate_wht_rotation

_TRITON_AVAILABLE = False
try:
    from .triton_kernels import triton_quantize as _triton_quantize
    _TRITON_AVAILABLE = True
except (ImportError, RuntimeError):
    pass

try:
    from .triton_kernels import triton_dequantize, triton_dequantize_residual
except (ImportError, RuntimeError):
    pass

from .generation_strategy import compute_anchor_schedule, ANCHOR_STRATEGIES
from .generation_layers import _CompressedLayer, _FP16Layer

# ---------------------------------------------------------------------------
# GenerationCache — production KV cache
# ---------------------------------------------------------------------------


class GenerationCache:
    """Production KV cache for compressed autoregressive generation.

    The first KV cache compression that matches FP16 generation quality.

    Uses a novel combination discovered through autoresearch:
    - Lloyd-Max quantization for keys and values
    - 1-bit direct residual sign correction (NOT QJL random projection)
    - Norm correction (original/reconstruction ratio)
    - FP16 window: last N tokens stored at full precision
    - FP16 anchor layers every N layers to break error accumulation

    Defaults tuned from a 246-configuration autoresearch sweep:
    - K4/V3 anchor=12 win=64 RQ=True is a safe middle ground (default)
    - K8/V3 anchor=12 is the quality champion (96.4% of FP16, 2.5x compression)
    - K3/V3 anchor=36 win=64 RQ=True is the best tradeoff (95.1%, 3.3x, Gen=97%)
    - K3/V2 anchor=6 win=512 RQ=True is the aggressive option (94.9%, higher compression)
    - RQ=True outperforms RQ=False by 13% on average (0.801 vs 0.707)
    - Anchors are essential: anchor=0 avg 0.408, anchor=12 avg 0.872

    Anchor strategies control which layers store KV at full FP16 precision
    to break error accumulation:

    - ``"fixed"`` (default): Every ``anchor_interval``-th layer is FP16.
      Simple, proven baseline. E.g., layers 0, 12, 24, 36.
    - ``"boundary"``: First 2 + last 2 layers always FP16, rest compressed.
      Based on finding that boundary layers (embedding proximity, output
      head proximity) are most sensitive to quantization error.
    - ``"gradient"``: Boundary layers FP16 + gradient bit allocation for
      middle layers. Layers near boundaries get higher key_bits (4-bit),
      middle layers get base key_bits (3-bit). Allocates bits where they
      matter most for the same total budget.

    Usage::

        from turboquantdc import GenerationCache

        # Default (safe middle ground)
        cache = GenerationCache()

        # From a named preset
        cache = GenerationCache.from_preset("balanced")
        cache = GenerationCache.from_preset("lossless")
        cache = GenerationCache.from_preset("aggressive")

        # Preset with overrides
        cache = GenerationCache.from_preset("balanced", fp16_window=128)

        # Boundary anchoring: first 2 + last 2 FP16
        cache = GenerationCache(anchor_strategy="boundary", num_layers=36)

        # Gradient: boundary FP16 + per-layer bit allocation
        cache = GenerationCache(anchor_strategy="gradient", num_layers=36)

        output = model.generate(inputs, past_key_values=cache, max_new_tokens=100)

    Args:
        key_bits: Bits for key quantization (default: 4).
        val_bits: Bits for value quantization (default: 3).
        fp16_window: Number of recent tokens at FP16 (default: 64).
        anchor_interval: Every Nth layer is stored at FP16 to break error
            accumulation. Set to 0 to disable anchors. Only used when
            ``anchor_strategy="fixed"`` (default: 12).
        anchor_strategy: Anchor placement strategy. One of ``"fixed"``,
            ``"boundary"``, ``"gradient"`` (default: ``"fixed"``).
        num_layers: Total number of transformer layers. Required for
            ``"boundary"`` and ``"gradient"`` strategies. When None,
            layers are created lazily on first ``update()`` call (only
            valid for ``"fixed"`` strategy).
        seed: Random seed for reproducibility.
        use_norm_correction: Apply norm correction (original/reconstruction
            ratio) for improved perplexity. Default True per fused_attention
            finding of -1.17% perplexity improvement.
        use_residual_quant: Whether to apply 1-bit residual sign correction
            to keys during dequantization. When False, only MSE centroid
            reconstruction is used. Default: True.
        use_triton: Use Triton fused kernel for quantization when available.
            Default ``True`` if Triton is importable and CUDA is present.
            Falls back to Python for WHT rotation or non-CUDA devices.
    """

    # Quality presets from 246-config autoresearch sweep.
    # Each maps to GenerationCache __init__ kwargs.
    PRESETS = {
        "lossless": {
            "key_bits": 8,
            "val_bits": 3,
            "anchor_interval": 12,
            "fp16_window": 0,
            "use_residual_quant": False,
        },
        "balanced": {
            "key_bits": 3,
            "val_bits": 3,
            "anchor_interval": 36,
            "fp16_window": 64,
            "use_residual_quant": True,
        },
        "aggressive": {
            "key_bits": 3,
            "val_bits": 2,
            "anchor_interval": 6,
            "fp16_window": 512,
            "use_residual_quant": True,
        },
        # Hybrid presets: combine boundary anchoring + gradient bits +
        # FP16 window + ResidualQuant + norm correction.
        "hybrid_max_quality": {
            "key_bits": 3,
            "val_bits": 3,
            "anchor_interval": 0,  # not used -- boundary handles it
            "anchor_strategy": "boundary",
            "fp16_window": 64,
            "use_residual_quant": True,
            "use_norm_correction": True,
        },
        "hybrid_max_compression": {
            "key_bits": 3,
            "val_bits": 2,
            "anchor_interval": 0,
            "anchor_strategy": "gradient",
            "fp16_window": 64,
            "use_residual_quant": True,
            "use_norm_correction": True,
        },
    }

    is_compileable = False

    def __init__(
        self,
        key_bits: int = 4,
        val_bits: int = 3,
        fp16_window: int = 64,
        anchor_interval: int = 12,
        anchor_strategy: str = "fixed",
        num_layers: Optional[int] = None,
        seed: int = 42,
        use_norm_correction: bool = True,
        use_residual_quant: bool = True,
        rotation_type: str | None = None,
        use_triton: bool = _TRITON_AVAILABLE,
        center_before_quantize: bool = True,
        quantizer_type: str = "lloyd_max",
    ):
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

        self.key_bits = key_bits
        self.val_bits = val_bits
        self.fp16_window = fp16_window
        self.anchor_interval = anchor_interval
        self.anchor_strategy = anchor_strategy
        self.num_layers = num_layers
        self.seed = seed
        self.use_norm_correction = use_norm_correction
        self.use_residual_quant = use_residual_quant
        self.rotation_type = rotation_type  # None = auto (WHT for power-of-2 d)
        self.use_triton = use_triton
        self.quantizer_type = quantizer_type  # "lloyd_max" or "e8"
        self.center_before_quantize = center_before_quantize

        # Pre-compute anchor schedule when num_layers is known
        self._anchor_schedule: Optional[List[Tuple[bool, int]]] = None
        if num_layers is not None:
            self._anchor_schedule = compute_anchor_schedule(
                num_layers=num_layers,
                anchor_strategy=anchor_strategy,
                anchor_interval=anchor_interval,
                base_key_bits=key_bits,
            )

        self._layers: List[_CompressedLayer | _FP16Layer] = []

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> "GenerationCache":
        """Create a GenerationCache from a named quality preset.

        Available presets (from 246-config autoresearch sweep):
        - "lossless": K8/V3 anchor=12 (96.4% of FP16, 2.5x compression)
        - "balanced": K3/V3 anchor=36 win=64 RQ=True (95.1%, 3.3x, Gen=97%)
        - "aggressive": K3/V2 anchor=6 win=512 RQ=True (94.9%, higher compression)

        Args:
            preset: One of "lossless", "balanced", "aggressive".
            **overrides: Any GenerationCache kwarg to override the preset value.

        Returns:
            A new GenerationCache configured from the preset.

        Raises:
            KeyError: If preset is not a recognized preset name.
        """
        if preset not in cls.PRESETS:
            raise KeyError(
                f"Unknown preset '{preset}'. "
                f"Available presets: {list(cls.PRESETS.keys())}"
            )
        config = cls.PRESETS[preset].copy()
        config.update(overrides)
        return cls(**config)

    def _is_anchor_layer(self, idx: int) -> bool:
        """Return True if layer ``idx`` should be stored at FP16."""
        if self._anchor_schedule is not None and idx < len(self._anchor_schedule):
            return self._anchor_schedule[idx][0]
        # Fallback for fixed strategy when num_layers is unknown (lazy growth)
        return self.anchor_interval > 0 and idx % self.anchor_interval == 0

    def _layer_key_bits(self, idx: int) -> int:
        """Return key bit-width for layer ``idx``."""
        if self._anchor_schedule is not None and idx < len(self._anchor_schedule):
            return self._anchor_schedule[idx][1]
        return self.key_bits

    def _make_layer(self, idx: int) -> _CompressedLayer | _FP16Layer:
        """Create the appropriate layer type for index ``idx``."""
        if self._is_anchor_layer(idx):
            return _FP16Layer()
        return _CompressedLayer(
            key_bits=self._layer_key_bits(idx),
            val_bits=self.val_bits,
            fp16_window=self.fp16_window,
            seed=self.seed + idx,
            use_norm_correction=self.use_norm_correction,
            use_residual_quant=self.use_residual_quant,
            rotation_type=self.rotation_type,
            use_triton=self.use_triton,
            center_before_quantize=self.center_before_quantize,
            quantizer_type=getattr(self, 'quantizer_type', 'lloyd_max'),
        )

    # ---- HF Cache protocol ----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress/store new KV pairs for a layer, return full cache.

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
        """Return number of cached tokens for a layer."""
        if layer_idx >= len(self._layers):
            return 0
        return self._layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Return max cache shape. Dynamic cache has no maximum."""
        return -1

    def get_mask_sizes(
        self,
        cache_position,
        layer_idx: int = 0,
    ) -> tuple[int, int]:
        """Return ``(kv_length, kv_offset)`` for attention mask generation.

        This is the critical method that must return the correct total KV
        length including both cached tokens and new query tokens.  Getting
        this wrong produces misaligned attention masks and garbled output.

        ``cache_position`` may be a ``torch.Tensor`` (transformers <=5.3)
        or an ``int`` query_length (transformers >=5.5).
        """
        if isinstance(cache_position, int):
            query_length = cache_position
        else:
            query_length = cache_position.shape[0]
        if layer_idx >= len(self._layers):
            return query_length, 0
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
            else:
                layer._key_indices = [
                    t.repeat_interleave(repeats, dim=0) for t in layer._key_indices
                ]
                layer._key_norms = [
                    t.repeat_interleave(repeats, dim=0) for t in layer._key_norms
                ]
                layer._key_res_signs = [
                    t.repeat_interleave(repeats, dim=0) for t in layer._key_res_signs
                ]
                layer._key_res_scales = [
                    t.repeat_interleave(repeats, dim=0) for t in layer._key_res_scales
                ]
                layer._val_indices = [
                    t.repeat_interleave(repeats, dim=0) for t in layer._val_indices
                ]
                layer._val_norms = [
                    t.repeat_interleave(repeats, dim=0) for t in layer._val_norms
                ]
                layer._raw_keys = [
                    t.repeat_interleave(repeats, dim=0) for t in layer._raw_keys
                ]
                layer._raw_vals = [
                    t.repeat_interleave(repeats, dim=0) for t in layer._raw_vals
                ]
                if layer._batch_size is not None:
                    layer._batch_size *= repeats
                # Invalidate dequant cache (batch dimension changed)
                layer._dequant_key_cache = None
                layer._dequant_val_cache = None
                layer._dequant_len = 0

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select specific batch indices from the cache."""
        for layer in self._layers:
            if isinstance(layer, _FP16Layer):
                layer._keys = [k[indices] for k in layer._keys]
                layer._values = [v[indices] for v in layer._values]
            else:
                layer._key_indices = [t[indices] for t in layer._key_indices]
                layer._key_norms = [t[indices] for t in layer._key_norms]
                layer._key_res_signs = [t[indices] for t in layer._key_res_signs]
                layer._key_res_scales = [t[indices] for t in layer._key_res_scales]
                layer._val_indices = [t[indices] for t in layer._val_indices]
                layer._val_norms = [t[indices] for t in layer._val_norms]
                layer._raw_keys = [t[indices] for t in layer._raw_keys]
                layer._raw_vals = [t[indices] for t in layer._raw_vals]
                if layer._batch_size is not None:
                    layer._batch_size = len(indices)
                # Invalidate dequant cache (batch dimension changed)
                layer._dequant_key_cache = None
                layer._dequant_val_cache = None
                layer._dequant_len = 0

    @property
    def seen_tokens(self) -> int:
        """Number of tokens seen by the first layer."""
        return self._layers[0].get_seq_length() if self._layers else 0

    @property
    def is_initialized(self) -> bool:
        """Return whether the cache has been populated."""
        return len(self._layers) > 0

    @property
    def is_sliding(self) -> list[bool]:
        """Return sliding window status per layer (always False)."""
        return [False] * max(len(self._layers), 1)

    def __len__(self) -> int:
        """Number of layers in the cache."""
        return len(self._layers)

    def __iter__(self):
        """Iterate over layers, yielding ``(keys, values, None)`` tuples."""
        for layer in self._layers:
            keys, values = layer._dequantize_all()
            yield keys, values, None

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dequantized ``(keys, values)`` for a specific layer."""
        if layer_idx >= len(self._layers):
            raise IndexError(
                f"Layer {layer_idx} not in cache (have {len(self._layers)} layers)"
            )
        return self._layers[layer_idx]._dequantize_all()

    def __contains__(self, idx: int) -> bool:
        """Check whether a layer index exists in the cache."""
        return 0 <= idx < len(self._layers)

    # ---- Reporting ----

    def memory_savings(self) -> Dict[str, Any]:
        """Report memory usage and savings across all layers.

        Returns:
            Dict with per_layer stats, aggregate totals, and configuration.
        """
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
                "key_bits": self.key_bits,
                "val_bits": self.val_bits,
                "fp16_window": self.fp16_window,
                "anchor_interval": self.anchor_interval,
                "anchor_strategy": self.anchor_strategy,
                "use_residual_quant": self.use_residual_quant,
            },
            "num_layers": len(self._layers),
        }

    def anchor_summary(self) -> Dict[str, Any]:
        """Return a summary of the anchor schedule for all layers.

        Useful for inspecting exactly which layers are FP16 anchors and
        the per-layer key bit-widths when using gradient strategy.

        Returns:
            Dict with:
                - strategy: Anchor strategy name.
                - num_layers: Total layer count (from schedule or actual).
                - fp16_layers: List of layer indices that are FP16 anchors.
                - per_layer_key_bits: List of key bit-widths per layer
                  (8 for FP16 anchor layers).
                - fp16_count: Number of FP16 anchor layers.
                - compressed_count: Number of compressed layers.
                - avg_key_bits: Average effective key bits across all layers.
        """
        n = self.num_layers if self.num_layers is not None else len(self._layers)
        if n == 0:
            return {
                "strategy": self.anchor_strategy,
                "num_layers": 0,
                "fp16_layers": [],
                "per_layer_key_bits": [],
                "fp16_count": 0,
                "compressed_count": 0,
                "avg_key_bits": 0.0,
            }

        fp16_layers = []
        per_layer_key_bits = []
        for i in range(n):
            is_fp16 = self._is_anchor_layer(i)
            kb = 16 if is_fp16 else self._layer_key_bits(i)
            per_layer_key_bits.append(kb)
            if is_fp16:
                fp16_layers.append(i)

        fp16_count = len(fp16_layers)
        avg_bits = sum(per_layer_key_bits) / n if n > 0 else 0.0

        return {
            "strategy": self.anchor_strategy,
            "num_layers": n,
            "fp16_layers": fp16_layers,
            "per_layer_key_bits": per_layer_key_bits,
            "fp16_count": fp16_count,
            "compressed_count": n - fp16_count,
            "avg_key_bits": avg_bits,
        }

    def config_summary(self) -> str:
        """Return a human-readable configuration summary."""
        n_layers = len(self._layers)
        n_anchor = sum(1 for i in range(n_layers) if self._is_anchor_layer(i))
        rq_desc = "+ 1b residual signs" if self.use_residual_quant else "(no residual signs)"
        strategy_desc = f"anchor={self.anchor_strategy}"
        if self.anchor_strategy == "fixed":
            strategy_desc += f" interval={self.anchor_interval}"
        return (
            f"GenerationCache: {self.key_bits}b keys {rq_desc}, "
            f"{self.val_bits}b values, FP16 window={self.fp16_window}, "
            f"{n_anchor}/{n_layers} anchor layers ({strategy_desc})"
        )


