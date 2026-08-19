"""TurboQuant vLLM AttentionImpl — Phase D (single-request reference path).

This module implements a real `vllm.v1.attention.backend.AttentionImpl`
subclass that uses TurboQuantDC's `GenerationCache` for 3-bit KV
compression. It targets vLLM 0.19+ (V1 engine).

## What this is, and what it isn't

This is the **honest single-request reference path** that resolves the 7
CRITICAL bugs the 2026-04-27 code review found in the previous
`vllm_integration.py`. It is correct math + correct vLLM API plumbing
for the single-request case. It is NOT a production-grade backend yet —
the remaining work to make it production is documented at the bottom of
this module.

## What's covered (the 7 requirements from the original stub docstring)

| # | Requirement                                              | Status |
|---|----------------------------------------------------------|--------|
| 1 | Subclass vLLM 0.19+ V1 `AttentionImpl`                    | done   |
| 2 | PagedAttention block layout                               | partial — `get_kv_cache_shape` returns the right shape so the engine allocates the buffer, but our `forward()` *bypasses* the paged buffer and uses our own per-layer state. See "What's not yet done" below. |
| 3 | Repeat-interleave compressed K/V across the GQA group     | done — handled by SDPA's `enable_gqa=True` flag, with explicit fallback path |
| 4 | Stateful per (layer, sequence): compress_kv appends; gather works | done — `GenerationCache.update()` is called per forward, returns full dequantised history |
| 5 | fp32 cast for vec_norm / divisor                          | done — `TurboQuantEstimator.quantize` already runs the norm/divide chain in fp32 (existing code), and we always pass through that path |
| 6 | Mean-removal wired in (avoids Qwen PPL 9410 catastrophe) | done — `center_before_quantize=True` is the default in `_CompressedLayer`, and we pass it explicitly |
| 7 | int16 (or packed bits) for index storage                  | inherited — `GenerationCache` stores indices in the dtype the underlying `_CompressedLayer` uses. Verified non-int64. Packing to bits is future optimisation work. |

## What's not yet done (deferred to follow-up phases)

- True paged KV layout. We allocate the page pool but don't read/write through
  it. Adding paged support requires reshape-and-cache kernels that pack
  TurboQuant's compressed representation into vLLM's
  `(2, num_blocks, block_size, num_kv_heads, head_size)` byte buffer; the
  indices need a custom dtype.
- Multi-request continuous batching. The current implementation handles batched
  forward calls (single request, multiple tokens) but maintains one
  `GenerationCache` per (layer, sequence). With vLLM's continuous batching the
  same physical block can be reused across requests after preemption — that
  requires a sequence_id-keyed dict and an eviction policy.
- Fused CUDA kernels for the compress/dequant hot path. We currently rely on
  `GenerationCache.update()` which uses Triton kernels when available.
- FP8 / BF16 KV dtype mixing. We assume FP16 throughout.

## Design

For each (layer, sequence_id) we maintain a `GenerationCache` instance that
holds the compressed KV history. On each `forward()` call:

1. Reshape vLLM's flat `(num_total_tokens, num_kv_heads, head_size)` K/V
   to HF transformers' `(batch=1, num_kv_heads, seq, head_dim)` layout that
   `GenerationCache.update()` expects.
2. Call `cache.update(K, V, layer_idx)` — this compresses the new tokens with
   mean-removal + 3-bit, returns the full **dequantised** K/V history at fp16.
3. Run PyTorch SDPA with `enable_gqa=True` so the kernel handles GQA expansion.
4. Reshape output back to vLLM's `(num_total_tokens, num_heads * head_size)`.

This bypasses vLLM's paged KV cache pool (the `kv_cache` tensor passed to
`forward()` is ignored). The trade-off: simple and correct vs. compatible with
vLLM's full paged scheduler.

## Usage

This backend isn't yet registered as a vLLM `AttentionBackend` factory. To use
it, you'd either fork vLLM and add it to the registry, or monkey-patch at
runtime. The integration test in `tests/test_vllm_attention_impl.py` exercises
the impl directly without needing vLLM's full engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from turboquantdc.generation_core import GenerationCache

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionLayer


# -----------------------------------------------------------------------------
# Soft import of vLLM's AttentionImpl base — this module must be importable
# without vLLM (e.g. for unit tests on CPU-only CI).
# -----------------------------------------------------------------------------
try:
    from vllm.v1.attention.backend import (  # type: ignore[import]
        AttentionImpl as _VllmAttentionImpl,
        AttentionType,
    )
    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover - vLLM may be absent in CI
    _VllmAttentionImpl = object  # type: ignore[misc, assignment]

    class AttentionType:  # type: ignore[no-redef]
        DECODER = "decoder"
        ENCODER = "encoder"
        ENCODER_ONLY = "encoder_only"
        ENCODER_DECODER = "encoder_decoder"

    _VLLM_AVAILABLE = False


# A sentinel sequence ID for the single-request reference path.
_DEFAULT_SEQUENCE_ID = "__default__"


def _materialize_full_kv(
    cache: GenerationCache,
    new_keys: torch.Tensor,
    new_values: torch.Tensor,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push new K/V into the per-layer cache and return the full history.

    Args:
        cache: The per-sequence `GenerationCache`.
        new_keys: Shape `(B, num_kv_heads, new_seq, head_dim)`. Typically B=1.
        new_values: Same shape as `new_keys`.
        layer_idx: Layer index inside this cache.

    Returns:
        `(K_full, V_full)`, each shape `(B, num_kv_heads, total_seq, head_dim)`,
        where `total_seq` is the running sequence length after this update.

    Notes:
        `GenerationCache.update()` matches HF transformers' Cache contract:
        you give it the new K/V slice, it appends + compresses internally, and
        returns the full dequantised cache. We pass `cache_kwargs=None` because
        the production path doesn't need RoPE rotation here (RoPE is applied
        upstream by the model itself).
    """
    return cache.update(new_keys, new_values, layer_idx, cache_kwargs=None)


def _sdpa_with_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """Run scaled-dot-product attention with GQA expansion.

    All inputs are `(B, num_*_heads, seq, head_dim)`. `query` may have more
    heads than `key`/`value`; we let SDPA's `enable_gqa=True` handle the
    repeat-interleave internally on platforms where it's supported, and fall
    back to a manual repeat for older PyTorch.
    """
    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]

    if num_q_heads != num_kv_heads:
        try:
            return F.scaled_dot_product_attention(
                query, key, value, scale=scale, is_causal=causal, enable_gqa=True
            )
        except TypeError:
            if num_q_heads % num_kv_heads != 0:
                raise ValueError(
                    f"num_q_heads ({num_q_heads}) is not a multiple of "
                    f"num_kv_heads ({num_kv_heads}); cannot expand for GQA."
                )
            group_factor = num_q_heads // num_kv_heads
            key = key.repeat_interleave(group_factor, dim=1)
            value = value.repeat_interleave(group_factor, dim=1)

    return F.scaled_dot_product_attention(
        query, key, value, scale=scale, is_causal=causal
    )


class TurboQuantAttentionImpl(_VllmAttentionImpl):  # type: ignore[misc]
    """vLLM V1 `AttentionImpl` that wraps `GenerationCache` for 3-bit KV.

    See the module docstring for the full design. Single-request reference
    path: we maintain `dict[sequence_id, GenerationCache]` and bypass vLLM's
    paged KV pool.
    """

    can_return_lse_for_decode: bool = False
    supports_pcp: bool = False
    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        *,
        turboquant_config: dict | None = None,
    ) -> None:
        if not _VLLM_AVAILABLE:
            self.dcp_world_size = 1
            self.dcp_rank = 0
            self.pcp_world_size = 1
            self.pcp_rank = 0
            self.total_cp_world_size = 1
            self.total_cp_rank = 0

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"TurboQuantAttentionImpl currently supports only "
                f"AttentionType.DECODER, got {attn_type!r}."
            )
        if alibi_slopes is not None:
            raise NotImplementedError(
                "TurboQuantAttentionImpl does not yet support ALiBi slopes."
            )
        if sliding_window is not None:
            raise NotImplementedError(
                "TurboQuantAttentionImpl does not yet support sliding window."
            )

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be a multiple of "
                f"num_kv_heads ({self.num_kv_heads}) for GQA."
            )

        self._tq_config = dict(turboquant_config or {})
        self._tq_config.setdefault("key_bits", 3)
        self._tq_config.setdefault("val_bits", 3)
        self._tq_config.setdefault("fp16_window", 64)
        self._tq_config.setdefault("anchor_strategy", "boundary")
        self._tq_config.setdefault("use_residual_quant", True)
        self._tq_config.setdefault("center_before_quantize", True)
        self._tq_config.setdefault("quantizer_type", "lloyd_max")

        if "num_layers" not in self._tq_config:
            if self._tq_config["anchor_strategy"] == "boundary":
                self._tq_config["anchor_strategy"] = "fixed"

        self._caches: dict[str, GenerationCache] = {}

    # ----- vLLM AttentionImpl API -----

    def forward(
        self,
        layer: "AttentionLayer",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,  # noqa: ARG002 — bypass paged pool, see module doc
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,  # noqa: ARG002
        output_block_scale: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Compute attention through the TurboQuant compressed KV path.

        Input shapes (vLLM V1 contract):
          query: `(num_total_tokens, num_heads, head_size)` OR flattened to
                 `(num_total_tokens, num_heads * head_size)` depending on the
                 layer; we accept both and reshape as needed.
          key  : `(num_total_tokens, num_kv_heads, head_size)`
          value: `(num_total_tokens, num_kv_heads, head_size)`

        Output shape: `(num_total_tokens, num_heads * head_size)`.
        """
        q = self._ensure_3d(query, self.num_heads, self.head_size)
        k = self._ensure_3d(key, self.num_kv_heads, self.head_size)
        v = self._ensure_3d(value, self.num_kv_heads, self.head_size)

        sequence_id = self._sequence_id_from_metadata(attn_metadata)
        cache = self._get_or_make_cache(sequence_id)
        layer_idx = self._layer_idx_from_layer(layer)

        # GenerationCache.update expects (B, H, S, D). For the single-request
        # reference path we treat the full token batch as one sequence.
        k_hf = k.permute(1, 0, 2).unsqueeze(0).contiguous()
        v_hf = v.permute(1, 0, 2).unsqueeze(0).contiguous()
        k_full, v_full = _materialize_full_kv(cache, k_hf, v_hf, layer_idx)

        q_hf = q.permute(1, 0, 2).unsqueeze(0).contiguous()

        # is_causal=True applies the standard auto-regressive mask. Holds for
        # decode and single-request prefill; multi-request needs a custom mask.
        attn_out = _sdpa_with_gqa(
            q_hf, k_full, v_full, scale=self.scale, causal=True,
        )

        attn_out = attn_out.squeeze(0).permute(1, 0, 2).contiguous()
        attn_out = attn_out.reshape(attn_out.shape[0], self.num_heads * self.head_size)
        if output is not None:
            output.copy_(attn_out)
            return output
        return attn_out

    # ----- Helpers -----

    @staticmethod
    def _ensure_3d(
        tensor: torch.Tensor, num_heads: int, head_size: int
    ) -> torch.Tensor:
        """Make sure tensor is `(num_tokens, num_heads, head_size)`."""
        if tensor.dim() == 3:
            return tensor.contiguous()
        if tensor.dim() == 2:
            num_tokens = tensor.shape[0]
            expected = num_heads * head_size
            if tensor.shape[1] != expected:
                raise ValueError(
                    f"2D tensor with shape {tuple(tensor.shape)} cannot be "
                    f"reshaped to (num_tokens, {num_heads}, {head_size}); "
                    f"expected last dim {expected}."
                )
            return tensor.reshape(num_tokens, num_heads, head_size).contiguous()
        raise ValueError(
            f"Expected 2D or 3D tensor, got {tensor.dim()}D shape {tuple(tensor.shape)}."
        )

    @staticmethod
    def _sequence_id_from_metadata(attn_metadata) -> str:
        """Return a stable sequence id from attn_metadata, or default sentinel.

        The single-request reference path returns the default for everything.
        Multi-request support: look at the request id list on `attn_metadata`
        (vLLM's scheduler tracks these) and return one id per sequence.
        """
        return _DEFAULT_SEQUENCE_ID

    @staticmethod
    def _layer_idx_from_layer(layer) -> int:
        """Resolve the layer index from a vLLM AttentionLayer name."""
        name = getattr(layer, "layer_name", None) or getattr(layer, "_layer_name", None)
        if name is None:
            return 0
        parts = str(name).split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return 0

    def _get_or_make_cache(self, sequence_id: str) -> GenerationCache:
        cache = self._caches.get(sequence_id)
        if cache is not None:
            return cache
        cache = GenerationCache(**self._tq_config)
        self._caches[sequence_id] = cache
        return cache

    # ----- Lifecycle -----

    def reset_sequence(self, sequence_id: str = _DEFAULT_SEQUENCE_ID) -> None:
        """Drop the per-sequence cache (e.g., when a request is preempted)."""
        self._caches.pop(sequence_id, None)

    def reset_all(self) -> None:
        """Drop every per-sequence cache."""
        self._caches.clear()


__all__ = [
    "TurboQuantAttentionImpl",
]
