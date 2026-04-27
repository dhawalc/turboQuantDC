"""TurboQuant single-sequence quality demo on Qwen3.6-27B-AWQ-INT4.

Loads the local Qwen3.6-27B-AWQ-INT4 via HF transformers, generates a short
completion with (a) the default HF cache and (b) `turboquantdc.GenerationCache`
3-bit, then compares text + decode tok/s. Single-sequence; this is the
research-demo path while a real vLLM custom backend is being built.

Caveat: Qwen3.6 uses a hybrid GDN + standard-attention architecture
(`full_attention_interval=4`). Only the 16-of-64 standard-attention layers
have a traditional KV cache; the GDN layers use recurrent state. We pass
`num_layers=64` to GenerationCache and let it handle whichever layers the
model actually populates.

Usage:
    .venv-vllm/bin/python scripts/turboquant_qwen36_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "./models/Qwen3.6-27B-AWQ-INT4"
PROMPT = "Q: In one sentence, why does TurboQuant compress KV cache?\nA:"
MAX_NEW = 64

OUTPUT_PATH = Path("benchmarks/results/qwen_flawless/turboquant_qwen36_demo.json")


def gen_with(model, tokenizer, prompt: str, past_key_values, label: str) -> dict:
    """Generate and time. Returns text + tok/s + elapsed."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW,
        do_sample=False,
        temperature=0.0,
        past_key_values=past_key_values,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_new = out.shape[-1] - inputs["input_ids"].shape[-1]
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return {
        "label": label,
        "text": text.strip()[:300],
        "n_new_tokens": int(n_new),
        "elapsed_s": round(elapsed, 3),
        "decode_tps": round(n_new / max(elapsed, 1e-6), 1),
    }


def main() -> int:
    print(f"[demo] loading model: {MODEL_PATH}", file=sys.stderr)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print(f"[demo] loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    # The loaded `model.config` is already the text-only sub-config when the
    # AutoModelForCausalLM wrapper picks Qwen3_5TextConfig.
    cfg_text = getattr(model.config, "text_config", model.config)
    n_layers = cfg_text.num_hidden_layers
    print(f"[demo] num_hidden_layers = {n_layers}", file=sys.stderr)

    baseline = gen_with(model, tokenizer, PROMPT, None, "baseline_fp16_kv")
    print(f"[baseline] {baseline['decode_tps']} tok/s "
          f"({baseline['n_new_tokens']} tokens in {baseline['elapsed_s']}s)",
          file=sys.stderr)
    print(f"[baseline] text: {baseline['text'][:120]!r}", file=sys.stderr)

    turbo_result = None
    try:
        from turboquantdc import GenerationCache

        cache = GenerationCache(
            key_bits=3,
            val_bits=3,
            fp16_window=128,
            anchor_strategy="boundary",
            num_layers=n_layers,
            use_residual_quant=True,
            center_before_quantize=True,
            quantizer_type="e8",
        )
        turbo_result = gen_with(model, tokenizer, PROMPT, cache, "turboquant_3bit_kv")
        print(f"[turbo] {turbo_result['decode_tps']} tok/s "
              f"({turbo_result['n_new_tokens']} tokens in {turbo_result['elapsed_s']}s)",
              file=sys.stderr)
        print(f"[turbo] text: {turbo_result['text'][:120]!r}", file=sys.stderr)
    except Exception as exc:
        turbo_result = {
            "label": "turboquant_3bit_kv",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"[turbo] FAILED: {exc}", file=sys.stderr)

    summary = {
        "model_path": MODEL_PATH,
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW,
        "baseline": baseline,
        "turboquant": turbo_result,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if "error" not in turbo_result else 1


if __name__ == "__main__":
    sys.exit(main())
