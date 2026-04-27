"""TurboQuantDC vLLM integration — STUB (2026-04-27 honest replacement).

> Status: STUB. The previous 935-line implementation in this file was a
> docstring-only sketch that did **not** import vLLM, targeted vLLM's V0 engine
> layout (removed in vLLM 0.8+, default since vLLM 0.19), and contained 7
> CRITICAL bugs (stateless backend, broken GQA, fp16 underflow, PagedAttention
> layout mismatch, missing mean-removal, V0 monkey-patch, int64 index blowup).
> See `docs/code_review/2026-04-27/CODE_REVIEW_2026-04-27.md` for the full
> findings. Replaced with this stub on 2026-04-27 so the module no longer
> claims a working integration that does not exist.

What still works:
  - `get_turboquant_config(model_name_or_path)` — model-architecture
    auto-detection helper, used by callers that need to pre-allocate a
    `GenerationCache`. Independent of any vLLM API.

What is intentionally not implemented:
  - `TurboQuantAttentionBackend` and `TurboQuantCacheManager` raise
    `NotImplementedError` on instantiation. They were not viable for vLLM in
    the previous form; a real implementation requires subclassing vLLM's
    `AttentionImpl` with PagedAttention block layout, GQA repeat-interleave,
    mean-removal wired in, and integration with vLLM's KV cache engine. That
    work is tracked at the top of CODE_REVIEW_2026-04-27.md.

Working production paths today (verified end-to-end):

  HF transformers + GenerationCache (single-sequence, autoregressive):

      from transformers import AutoModelForCausalLM, AutoTokenizer
      from turboquantdc import GenerationCache
      import torch

      model = AutoModelForCausalLM.from_pretrained(
          "Qwen/Qwen2.5-7B-Instruct",
          device_map="auto", torch_dtype=torch.float16,
      )
      cache = GenerationCache(
          key_bits=3, val_bits=3, fp16_window=128,
          anchor_strategy="boundary",
          num_layers=model.config.num_hidden_layers,
          use_residual_quant=True, center_before_quantize=True,
          quantizer_type="e8",
      )
      tok = AutoTokenizer.from_pretrained(model.name_or_path)
      out = model.generate(**tok("Hi", return_tensors="pt").to(model.device),
                           past_key_values=cache, max_new_tokens=128)

  vLLM with native FP8 KV (no TurboQuant compression, but well-tested for
  Qwen3.6-27B-class models on a 24 GB+ GPU):

      vllm serve <model> \\
          --kv-cache-dtype fp8_e4m3 \\
          --gpu-memory-utilization 0.85 \\
          --max-model-len 8192

  See `scripts/serve_qwen36_flawless.sh` for the working RTX 4090 recipe
  produced during the 2026-04-27 session.
"""

from __future__ import annotations

from typing import Optional


class TurboQuantAttentionBackend:
    """STUB. The previous implementation was not vLLM-compatible.

    The real implementation must:
      1. Subclass vLLM's `AttentionImpl` (`vllm.attention.backends.abstract`).
      2. Accept and use vLLM's PagedAttention block layout
         `(num_blocks, block_size, num_kv_heads, head_dim)`, not a flat
         per-request layout.
      3. Repeat-interleave compressed K/V across the GQA group factor before
         the score matmul (Qwen3.6-27B has num_heads=48, num_kv_heads=4).
      4. Be **stateful** per (layer, sequence): `compress_kv` must append to
         the per-layer compressed store; `compute_attention` must gather all
         past compressed K/V plus the current step.
      5. Cast through fp32 for vec_norm / divisor computations to avoid fp16
         underflow on outlier-heavy value vectors.
      6. Wire in mean-removal (`_CompressedLayer.center_before_quantize=True`)
         to avoid the Qwen PPL-9410 catastrophic-failure mode.
      7. Use int16 (or packed bits for QJL signs) for index storage, not
         int64 — the previous version's `mse_indices.long()` was an 8x
         memory blowup vs the int16 stored on disk.

    Until that work is done, attempting to construct this class raises
    NotImplementedError so callers get a clear pointer to the issue.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "TurboQuantAttentionBackend is a stub as of 2026-04-27. The "
            "previous implementation had 7 CRITICAL bugs documented in "
            "docs/code_review/2026-04-27/CODE_REVIEW_2026-04-27.md (review "
            "#1, hot-path correctness). For production single-sequence "
            "inference today use `turboquantdc.GenerationCache` with HF "
            "transformers; the recipe is in this module's docstring. For "
            "vLLM with vanilla FP8 KV (no TurboQuant), see "
            "scripts/serve_qwen36_flawless.sh."
        )


class TurboQuantCacheManager:
    """STUB. The previous implementation used a flat `(max_seq_len, H, D)`
    layout that is not compatible with vLLM's PagedAttention block layout
    `(num_blocks, block_size, num_kv_heads, head_dim)` and provided no
    block_table awareness. Replacing it requires the same scope of work as
    `TurboQuantAttentionBackend` above."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "TurboQuantCacheManager is a stub as of 2026-04-27. See the "
            "TurboQuantAttentionBackend docstring or the code review at "
            "docs/code_review/2026-04-27/CODE_REVIEW_2026-04-27.md for "
            "the rebuild plan."
        )


# ---------------------------------------------------------------------------
# Config helper — kept and corrected on 2026-04-27.
# Reviewer #8 found the `qwen3.5-27` entry had wrong constants (62/8/256;
# correct is 64/4/256). Adding `qwen3.6-27` for the new dense Qwen3.6 family.
# ---------------------------------------------------------------------------

_MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "qwen2.5-0.5": {"num_layers": 24, "num_kv_heads": 2,  "head_dim": 64},
    "qwen2.5-1.5": {"num_layers": 28, "num_kv_heads": 2,  "head_dim": 128},
    "qwen2.5-3":   {"num_layers": 36, "num_kv_heads": 2,  "head_dim": 128},
    "qwen2.5-7":   {"num_layers": 28, "num_kv_heads": 4,  "head_dim": 128},
    "qwen2.5-14":  {"num_layers": 48, "num_kv_heads": 8,  "head_dim": 128},
    "qwen2.5-32":  {"num_layers": 64, "num_kv_heads": 8,  "head_dim": 128},
    "qwen2.5-72":  {"num_layers": 80, "num_kv_heads": 8,  "head_dim": 128},
    "qwen3.5-7":   {"num_layers": 28, "num_kv_heads": 4,  "head_dim": 128},
    # Corrected 2026-04-27 (was 62 / 8): Qwen3.5-27B has 64 layers / 4 KV heads.
    "qwen3.5-27":  {"num_layers": 64, "num_kv_heads": 4,  "head_dim": 256},
    # Added 2026-04-27. Qwen3.6-27B is a hybrid GDN + standard-attention
    # multimodal architecture (Qwen3_5ForConditionalGeneration). Native 262K
    # context. Verify dims at runtime via _try_load_hf_config.
    "qwen3.6-27":  {"num_layers": 64, "num_kv_heads": 4,  "head_dim": 256},
    "llama-3.1-8":  {"num_layers": 32, "num_kv_heads": 8,  "head_dim": 128},
    "llama-3.1-70": {"num_layers": 80, "num_kv_heads": 8,  "head_dim": 128},
    "llama-3.2-1":  {"num_layers": 16, "num_kv_heads": 8,  "head_dim": 64},
    "llama-3.2-3":  {"num_layers": 28, "num_kv_heads": 8,  "head_dim": 64},
    "mistral-7":   {"num_layers": 32, "num_kv_heads": 8,  "head_dim": 128},
    "mixtral-8x7": {"num_layers": 32, "num_kv_heads": 8,  "head_dim": 128},
    "phi-4":       {"num_layers": 40, "num_kv_heads": 10, "head_dim": 96},
    "gemma-2-9":   {"num_layers": 46, "num_kv_heads": 4,  "head_dim": 256},
    "gemma-2-27":  {"num_layers": 62, "num_kv_heads": 16, "head_dim": 128},
    "minimax-m2":  {"num_layers": 62, "num_kv_heads": 4,  "head_dim": 128},
}

_DEFAULT_CONFIG = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128}


def _try_load_hf_config(model_name_or_path: str) -> Optional[dict[str, int]]:
    """Best-effort load of `{num_layers, num_kv_heads, head_dim}` from HF
    transformers `AutoConfig`. Returns None if transformers is missing or the
    config can't be loaded.
    """
    try:
        from transformers import AutoConfig  # type: ignore[import]

        hf_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=False)
    except Exception:
        return None

    try:
        num_heads = getattr(hf_cfg, "num_attention_heads", None)
        hidden_size = getattr(hf_cfg, "hidden_size", None)
        head_dim = getattr(hf_cfg, "head_dim", None)
        if head_dim is None and num_heads and hidden_size:
            head_dim = hidden_size // num_heads

        num_kv_heads = getattr(hf_cfg, "num_key_value_heads", num_heads)
        num_layers = getattr(
            hf_cfg, "num_hidden_layers", getattr(hf_cfg, "n_layer", None)
        )

        if head_dim is None or num_kv_heads is None or num_layers is None:
            return None

        return {
            "num_layers": int(num_layers),
            "num_kv_heads": int(num_kv_heads),
            "head_dim": int(head_dim),
        }
    except Exception:
        return None


def get_turboquant_config(
    model_name_or_path: str,
    bits: int = 3,
    vram_gb: float = 24.0,
) -> dict[str, object]:
    """Auto-detect model architecture and return a TurboQuant config dict.

    Resolution order: static lookup table → HF AutoConfig → default 7B layout.

    Returns:
        Dict with keys: model_name, head_dim, num_kv_heads, num_layers, bits,
        estimated_compression, estimated_max_context, config_source.

        `estimated_max_context` is a back-of-envelope estimate that assumes
        weights fit in 60% of VRAM and KV cache uses 40%. Validate against
        actual model size before relying on it (see docs/code_review/
        2026-04-27/01_hotpath_correctness.md, MEDIUM finding on this point).
    """
    lower_name = model_name_or_path.lower().replace("_", "-").replace("/", "-")
    found_cfg: Optional[dict[str, int]] = None
    config_source = "default"

    for key, cfg in _MODEL_CONFIGS.items():
        if key in lower_name:
            found_cfg = cfg
            config_source = "lookup"
            break

    if found_cfg is None:
        found_cfg = _try_load_hf_config(model_name_or_path)
        if found_cfg is not None:
            config_source = "hf_config"

    if found_cfg is None:
        found_cfg = _DEFAULT_CONFIG.copy()

    num_layers = found_cfg["num_layers"]
    num_kv_heads = found_cfg["num_kv_heads"]
    head_dim = found_cfg["head_dim"]

    # Theoretical compression ratio per kv_cache.py logic.
    mse_bits_key = max(bits - 1, 1)
    qjl_bits = head_dim
    key_norm_bits = 32
    val_bits = bits * head_dim + 16
    bits_per_token = (mse_bits_key * head_dim + qjl_bits + key_norm_bits) + val_bits
    fp16_bits_per_token = (head_dim + head_dim) * 16
    estimated_compression = fp16_bits_per_token / bits_per_token

    bytes_per_token = bits_per_token * num_kv_heads / 8 * num_layers
    vram_bytes = vram_gb * 1024 ** 3
    cache_budget_bytes = vram_bytes * 0.40
    estimated_max_context = int(cache_budget_bytes / bytes_per_token)

    return {
        "model_name": model_name_or_path,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "num_layers": num_layers,
        "bits": bits,
        "estimated_compression": round(estimated_compression, 2),
        "estimated_max_context": estimated_max_context,
        "config_source": config_source,
    }
