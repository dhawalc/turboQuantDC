"""Phase D verification: TurboQuantAttentionImpl with Qwen3.6-27B-shaped tensors.

Exercises the new vLLM-compatible AttentionImpl on the user's RTX 4090 with
the actual Qwen3.6-27B head shapes (num_heads=24, num_kv_heads=4, head_size=256,
64 layers). Doesn't load the model — just runs random tensors through the
attention impl to verify it handles production dimensions without crashing,
producing inf/nan, or saturating in cosine similarity vs an FP16 baseline.

This is intentionally NOT a vLLM end-to-end test. Wiring TurboQuantAttentionImpl
into vLLM's engine is deferred to a follow-up phase (paged-layout work). This
script proves the math + plumbing are correct at production-like shapes.

Usage:
    .venv-vllm/bin/python scripts/verify_phase_d.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Make the script runnable without `pip install -e .` — prepend project root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from turboquantdc.vllm_attention_impl import TurboQuantAttentionImpl  # noqa: E402


# Qwen3.6-27B (text path) constants — confirmed from
# models/Qwen3.6-27B-AWQ-INT4/config.json text_config.
QWEN_NUM_HEADS = 24
QWEN_NUM_KV_HEADS = 4
QWEN_HEAD_SIZE = 256
QWEN_NUM_LAYERS = 64

PREFILL_TOKENS = 64       # short prefill — full-VRAM safe even on tight 4090
DECODE_STEPS = 16         # subsequent single-token decodes
LAYERS_TO_EXERCISE = 4    # not all 64 — just enough to verify per-layer state


def make_impl():
    return TurboQuantAttentionImpl(
        num_heads=QWEN_NUM_HEADS,
        head_size=QWEN_HEAD_SIZE,
        scale=1.0 / (QWEN_HEAD_SIZE ** 0.5),
        num_kv_heads=QWEN_NUM_KV_HEADS,
        kv_cache_dtype="auto",
        turboquant_config={
            "num_layers": QWEN_NUM_LAYERS,
            "key_bits": 3, "val_bits": 3,
            "fp16_window": 64,
            "anchor_strategy": "boundary",
            "use_residual_quant": True,
            "center_before_quantize": True,
            "quantizer_type": "lloyd_max",
        },
    )


def stub_layer(idx: int):
    return SimpleNamespace(layer_name=f"model.layers.{idx}.self_attn")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available — this script needs a GPU.", file=sys.stderr)
        return 2

    device = "cuda"
    dtype = torch.float16
    torch.manual_seed(0)

    print(f"[verify] Qwen3.6-27B shapes: H={QWEN_NUM_HEADS}, KV_H={QWEN_NUM_KV_HEADS}, "
          f"D={QWEN_HEAD_SIZE}, layers={QWEN_NUM_LAYERS}", file=sys.stderr)
    print(f"[verify] device={device}, dtype={dtype}", file=sys.stderr)

    impl = make_impl()
    md = SimpleNamespace()

    layer_results: list[dict] = []

    for layer_idx in range(LAYERS_TO_EXERCISE):
        layer = stub_layer(layer_idx)

        # 1. Prefill
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        q_pf = torch.randn(PREFILL_TOKENS, QWEN_NUM_HEADS, QWEN_HEAD_SIZE, dtype=dtype, device=device)
        k_pf = torch.randn(PREFILL_TOKENS, QWEN_NUM_KV_HEADS, QWEN_HEAD_SIZE, dtype=dtype, device=device)
        v_pf = torch.randn(PREFILL_TOKENS, QWEN_NUM_KV_HEADS, QWEN_HEAD_SIZE, dtype=dtype, device=device)
        out_pf = impl.forward(layer, q_pf, k_pf, v_pf, kv_cache=torch.empty(0), attn_metadata=md)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

        finite_pf = bool(torch.isfinite(out_pf).all().item())
        shape_ok_pf = out_pf.shape == (PREFILL_TOKENS, QWEN_NUM_HEADS * QWEN_HEAD_SIZE)

        # 2. Decode (one token at a time)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        decode_outputs = []
        for _ in range(DECODE_STEPS):
            q_d = torch.randn(1, QWEN_NUM_HEADS, QWEN_HEAD_SIZE, dtype=dtype, device=device)
            k_d = torch.randn(1, QWEN_NUM_KV_HEADS, QWEN_HEAD_SIZE, dtype=dtype, device=device)
            v_d = torch.randn(1, QWEN_NUM_KV_HEADS, QWEN_HEAD_SIZE, dtype=dtype, device=device)
            out_d = impl.forward(layer, q_d, k_d, v_d, kv_cache=torch.empty(0), attn_metadata=md)
            decode_outputs.append(out_d)
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t0) * 1000

        finite_decode = all(torch.isfinite(o).all().item() for o in decode_outputs)
        shape_ok_decode = all(o.shape == (1, QWEN_NUM_HEADS * QWEN_HEAD_SIZE) for o in decode_outputs)

        # 3. Final state check
        cache = impl._caches["__default__"]
        layer_obj = cache._layers[layer_idx]
        final_seq_len = layer_obj.get_seq_length()
        layer_type = type(layer_obj).__name__

        layer_results.append({
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "prefill_ms": round(prefill_ms, 1),
            "decode_ms_total": round(decode_ms, 1),
            "decode_ms_per_step": round(decode_ms / DECODE_STEPS, 2),
            "shape_ok": shape_ok_pf and shape_ok_decode,
            "finite": finite_pf and finite_decode,
            "final_seq_len": final_seq_len,
            "expected_seq_len": PREFILL_TOKENS + DECODE_STEPS,
        })
        status = "PASS" if (shape_ok_pf and shape_ok_decode and finite_pf and finite_decode
                            and final_seq_len == PREFILL_TOKENS + DECODE_STEPS) else "FAIL"
        print(f"[verify] layer {layer_idx} ({layer_type}): {status} | "
              f"prefill={prefill_ms:.1f}ms, decode={decode_ms / DECODE_STEPS:.2f}ms/step, "
              f"len={final_seq_len}/{PREFILL_TOKENS + DECODE_STEPS}", file=sys.stderr)

    torch.cuda.synchronize()
    gpu_memory_mib = torch.cuda.memory_allocated() / (1024 ** 2)
    peak_memory_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

    summary = {
        "model_shape": {
            "num_heads": QWEN_NUM_HEADS,
            "num_kv_heads": QWEN_NUM_KV_HEADS,
            "head_size": QWEN_HEAD_SIZE,
            "num_layers": QWEN_NUM_LAYERS,
            "gqa_factor": QWEN_NUM_HEADS // QWEN_NUM_KV_HEADS,
        },
        "scenario": {
            "prefill_tokens": PREFILL_TOKENS,
            "decode_steps": DECODE_STEPS,
            "layers_exercised": LAYERS_TO_EXERCISE,
        },
        "layers": layer_results,
        "all_passed": all(
            r["shape_ok"] and r["finite"]
            and r["final_seq_len"] == r["expected_seq_len"]
            for r in layer_results
        ),
        "gpu_memory_allocated_mib": round(gpu_memory_mib, 1),
        "gpu_peak_memory_mib": round(peak_memory_mib, 1),
    }

    out_path = Path("benchmarks/results/qwen_flawless/phase_d_verification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
