"""Large model validation for TurboQuantDC.

Tests ResidualQuant KV cache compression on:
  1. Qwen2.5-32B-Instruct (BnB NF4, ~18GB VRAM)
  2. Qwen2.5-72B-Instruct-GPTQ-Int4 (GPTQ, multi-device)

Both models have head_dim=128 (our most-tested configuration).

Usage:
    python benchmarks/large_model_validation.py
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BIT_WIDTHS = [3, 4]
RESULTS_PATH = os.path.join(REPO_ROOT, "benchmarks", "results", "large_model_results.md")

# HF cache on Beast
HF_CACHE = "/media/dhawal/Beast/cache/hub"
os.environ["HF_HOME"] = "/media/dhawal/Beast/cache"
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

TEST_PROMPT = (
    "You are a helpful assistant. Explain the concept of KV cache compression "
    "in large language models in exactly three sentences."
)
GENERATION_PROMPT = "The three most important inventions of the 20th century are"
MAX_NEW_TOKENS = 50

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class CompressionResult:
    layer: int
    head: int
    bits: int
    method: str
    cosine_sim: float
    top1_match: bool
    top5_match: bool

@dataclass
class AggregateResult:
    bits: int
    method: str
    mean_cosine: float
    top1_rate: float
    top5_rate: float
    num_heads_measured: int
    compression_ratio: float

@dataclass
class GenerationResult:
    method: str
    text: str
    tokens_generated: int
    time_sec: float

@dataclass
class ModelTestResult:
    model_name: str
    model_info: Dict[str, Any]
    aggregate_results: List[AggregateResult]
    generation_results: List[GenerationResult]
    findings: List[str]
    timings: Dict[str, float]
    success: bool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_attention_scores(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    d = queries.shape[-1]
    scores = queries @ keys.T / math.sqrt(d)
    return F.softmax(scores, dim=-1)


def gpu_mem_mb() -> str:
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return f"{used:.0f}MB alloc / {reserved:.0f}MB reserved"
    return "N/A"


def cleanup_gpu():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def extract_kv_layers(past_kv) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Extract (keys, values) from various HF cache formats."""
    if hasattr(past_kv, "layers") and len(past_kv.layers) > 0:
        return [
            (past_kv.layers[i].keys, past_kv.layers[i].values)
            for i in range(len(past_kv.layers))
        ]
    elif hasattr(past_kv, "key_cache"):
        return [
            (past_kv.key_cache[i], past_kv.value_cache[i])
            for i in range(len(past_kv.key_cache))
        ]
    elif isinstance(past_kv, (list, tuple)) and len(past_kv) > 0:
        if isinstance(past_kv[0], (list, tuple)):
            return [(layer[0], layer[1]) for layer in past_kv]
    raise ValueError(f"Unexpected past_kv type: {type(past_kv)}")


# ---------------------------------------------------------------------------
# Test a single model
# ---------------------------------------------------------------------------

def test_model(
    model_name: str,
    load_kwargs: Dict[str, Any],
    do_generation: bool = True,
) -> ModelTestResult:
    """Run full validation on a model: load, extract KV, compress, generate."""

    print("=" * 70)
    print(f"Testing: {model_name}")
    print("=" * 70)

    timings = {}
    findings = []
    model_info = {}
    aggregate_results = []
    generation_results = []

    # ---- Step 1: Load model ----
    print(f"\n[1/4] Loading {model_name}...")
    print(f"  GPU before load: {gpu_mem_mb()}")
    t0 = time.time()

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=HF_CACHE,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=HF_CACHE,
            **load_kwargs,
        )

        timings["model_load"] = time.time() - t0
        print(f"  Loaded in {timings['model_load']:.1f}s")
        print(f"  GPU after load: {gpu_mem_mb()}")

        # Extract architecture info
        config = model.config
        text_config = getattr(config, "text_config", config)
        num_layers = getattr(text_config, "num_hidden_layers", "?")
        num_kv_heads = getattr(text_config, "num_key_value_heads", "?")
        num_attn_heads = getattr(text_config, "num_attention_heads", "?")
        hidden_size = getattr(text_config, "hidden_size", "?")
        vocab_size = getattr(text_config, "vocab_size", "?")

        # head_dim: some configs have it explicitly, otherwise compute
        head_dim = getattr(text_config, "head_dim", None)
        if head_dim is None and isinstance(hidden_size, int) and isinstance(num_attn_heads, int):
            head_dim = hidden_size // num_attn_heads

        model_info = {
            "Model": model_name,
            "Num layers": num_layers,
            "Num attention heads": num_attn_heads,
            "Num KV heads": num_kv_heads,
            "Head dimension": head_dim,
            "Hidden size": hidden_size,
            "Vocab size": vocab_size,
            "Load kwargs": str({k: str(v) for k, v in load_kwargs.items()}),
        }

        print(f"  Architecture: {num_layers}L / {num_kv_heads} KV heads / d={head_dim}")
        findings.append(f"head_dim={head_dim} (d=128 path, our most-tested config)")

        # Check device map
        if hasattr(model, "hf_device_map"):
            devices = set(str(v) for v in model.hf_device_map.values())
            findings.append(f"Device map: {len(model.hf_device_map)} modules across devices {devices}")
            print(f"  Devices: {devices}")

    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        traceback.print_exc()
        findings.append(f"MODEL LOAD FAILED: {e}")
        return ModelTestResult(
            model_name=model_name,
            model_info=model_info,
            aggregate_results=aggregate_results,
            generation_results=generation_results,
            findings=findings,
            timings=timings,
            success=False,
        )

    # ---- Step 2: Extract KV caches ----
    print(f"\n[2/4] Extracting KV caches...")
    t0 = time.time()

    kv_layers = None
    try:
        inputs = tokenizer(TEST_PROMPT, return_tensors="pt")
        # Move to correct device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]
        print(f"  Prompt tokens: {seq_len}")

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=False,
                use_cache=True,
            )

        past_kv = outputs.past_key_values
        kv_layers = extract_kv_layers(past_kv)
        timings["kv_extraction"] = time.time() - t0

        num_cached_layers = len(kv_layers)
        sample_k = kv_layers[0][0]
        actual_head_dim = sample_k.shape[-1]
        actual_num_kv_heads = sample_k.shape[1]
        actual_seq = sample_k.shape[2]

        print(f"  Extracted {num_cached_layers} layers in {timings['kv_extraction']:.1f}s")
        print(f"  Key shape: {sample_k.shape} (batch, heads, seq, d)")
        print(f"  Key dtype: {sample_k.dtype}, device: {sample_k.device}")

        findings.append(
            f"KV cache: {num_cached_layers} layers, {actual_num_kv_heads} KV heads, "
            f"seq={actual_seq}, d={actual_head_dim}"
        )

        # Check for multi-device layers
        layer_devices = set()
        for i, (k, v) in enumerate(kv_layers):
            layer_devices.add(str(k.device))
        if len(layer_devices) > 1:
            findings.append(f"MULTI-DEVICE KV cache: layers spread across {layer_devices}")
            print(f"  Multi-device KV: {layer_devices}")
        else:
            findings.append(f"All KV layers on device: {layer_devices.pop()}")

    except Exception as e:
        print(f"  KV EXTRACTION FAILED: {e}")
        traceback.print_exc()
        findings.append(f"KV EXTRACTION FAILED: {e}")
        # Clean up model
        del model
        cleanup_gpu()
        return ModelTestResult(
            model_name=model_name,
            model_info=model_info,
            aggregate_results=aggregate_results,
            generation_results=generation_results,
            findings=findings,
            timings=timings,
            success=False,
        )

    # ---- Step 3: Compression quality test ----
    print(f"\n[3/4] Testing compression quality...")
    t0 = time.time()

    from turboquantdc.residual_quant import ResidualQuantEstimator
    from turboquantdc.polarquant import PolarQuant

    d = actual_head_dim

    # Sample layers evenly: first, 1/4, 1/2, 3/4, last
    total_layers = len(kv_layers)
    sample_indices = sorted(set([
        0,
        total_layers // 4,
        total_layers // 2,
        3 * total_layers // 4,
        total_layers - 1,
    ]))
    print(f"  Sampling layers: {sample_indices} (of {total_layers})")

    all_results: List[CompressionResult] = []

    # Cache estimators per (bits, head_dim)
    rq_cache = {}
    pq_cache = {}

    for bits in BIT_WIDTHS:
        print(f"\n  --- {bits}-bit compression ---")

        rq_key = (bits, d)
        if rq_key not in rq_cache:
            try:
                rq_cache[rq_key] = ResidualQuantEstimator(d=d, bits=bits, seed=42, device=DEVICE)
                pq_cache[rq_key] = PolarQuant(d=d, bits=max(bits - 1, 1), seed=42, device=DEVICE)
                print(f"    Estimators created for d={d}, {bits}-bit")
            except Exception as e:
                findings.append(f"Estimator creation FAILED ({bits}b, d={d}): {e}")
                traceback.print_exc()
                rq_cache[rq_key] = None
                pq_cache[rq_key] = None

        rq = rq_cache[rq_key]
        pq = pq_cache[rq_key]

        for layer_idx in sample_indices:
            keys_fp16, vals_fp16 = kv_layers[layer_idx]
            batch, num_h, seq, hd = keys_fp16.shape

            for head_idx in range(num_h):
                k = keys_fp16[0, head_idx].float().to(DEVICE)
                q = k[-1:, :]

                fp16_scores = compute_attention_scores(q, k)

                if rq is None or pq is None:
                    continue

                # --- ResidualQuant ---
                try:
                    compressed_rq = rq.quantize(k)
                    k_rq = rq.dequantize(compressed_rq)

                    rq_scores = compute_attention_scores(q, k_rq)
                    cos = F.cosine_similarity(
                        fp16_scores.flatten().unsqueeze(0),
                        rq_scores.flatten().unsqueeze(0),
                    ).item()

                    fp16_top1 = fp16_scores.argmax(dim=-1)
                    rq_top1 = rq_scores.argmax(dim=-1)
                    t1 = (fp16_top1 == rq_top1).all().item()

                    fp16_top5 = fp16_scores.topk(min(5, seq), dim=-1).indices
                    rq_top5 = rq_scores.topk(min(5, seq), dim=-1).indices
                    t5 = bool(set(fp16_top5[0].tolist()) & set(rq_top5[0].tolist()))

                    all_results.append(CompressionResult(
                        layer=layer_idx, head=head_idx, bits=bits,
                        method="ResidualQuant", cosine_sim=cos,
                        top1_match=t1, top5_match=t5,
                    ))
                except Exception as e:
                    findings.append(f"RQ {bits}b L{layer_idx}H{head_idx}: {e}")
                    traceback.print_exc()

                # --- PolarQuant (MSE only) ---
                try:
                    k_norm = k.norm(dim=-1, keepdim=True)
                    k_unit = k / (k_norm + 1e-8)
                    indices = pq.quantize(k_unit)
                    k_pq = pq.dequantize(indices) * k_norm

                    pq_scores = compute_attention_scores(q, k_pq)
                    cos = F.cosine_similarity(
                        fp16_scores.flatten().unsqueeze(0),
                        pq_scores.flatten().unsqueeze(0),
                    ).item()

                    fp16_top1 = fp16_scores.argmax(dim=-1)
                    pq_top1 = pq_scores.argmax(dim=-1)
                    t1 = (fp16_top1 == pq_top1).all().item()

                    fp16_top5 = fp16_scores.topk(min(5, seq), dim=-1).indices
                    pq_top5 = pq_scores.topk(min(5, seq), dim=-1).indices
                    t5 = bool(set(fp16_top5[0].tolist()) & set(pq_top5[0].tolist()))

                    all_results.append(CompressionResult(
                        layer=layer_idx, head=head_idx, bits=bits,
                        method="PolarQuant", cosine_sim=cos,
                        top1_match=t1, top5_match=t5,
                    ))
                except Exception as e:
                    findings.append(f"PQ {bits}b L{layer_idx}H{head_idx}: {e}")
                    traceback.print_exc()

        # Aggregate for this bit-width
        for method in ["ResidualQuant", "PolarQuant"]:
            method_results = [
                r for r in all_results if r.bits == bits and r.method == method
            ]
            if method_results:
                cos_mean = sum(r.cosine_sim for r in method_results) / len(method_results)
                t1_rate = sum(1 for r in method_results if r.top1_match) / len(method_results)
                t5_rate = sum(1 for r in method_results if r.top5_match) / len(method_results)
                print(f"    {method} {bits}b: cos={cos_mean:.6f} top1={t1_rate:.1%} top5={t5_rate:.1%}")

                if method == "ResidualQuant":
                    compressed_bpv = bits * d + 32
                else:
                    mse_bits_used = max(bits - 1, 1)
                    compressed_bpv = mse_bits_used * d + 16
                fp16_bpv = d * 16
                ratio = fp16_bpv / compressed_bpv

                aggregate_results.append(AggregateResult(
                    bits=bits, method=method,
                    mean_cosine=cos_mean, top1_rate=t1_rate, top5_rate=t5_rate,
                    num_heads_measured=len(method_results),
                    compression_ratio=ratio,
                ))

    timings["compression_test"] = time.time() - t0

    # ---- Step 4: Generation test ----
    if do_generation:
        print(f"\n[4/4] Generation comparison test...")
        t0 = time.time()

        try:
            from turboquantdc.generation_cache import GenerationCache

            gen_inputs = tokenizer(GENERATION_PROMPT, return_tensors="pt")
            device = next(model.parameters()).device
            gen_inputs = {k: v.to(device) for k, v in gen_inputs.items()}

            # FP16 baseline
            print("  Generating with FP16 KV cache...")
            t_gen = time.time()
            with torch.no_grad():
                fp16_output = model.generate(
                    **gen_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=1.0,
                )
            fp16_time = time.time() - t_gen
            fp16_text = tokenizer.decode(fp16_output[0], skip_special_tokens=True)
            fp16_tokens = fp16_output.shape[1] - gen_inputs["input_ids"].shape[1]
            print(f"  FP16: {fp16_tokens} tokens in {fp16_time:.2f}s")
            print(f"  FP16 text: {fp16_text[:200]}")

            generation_results.append(GenerationResult(
                method="FP16 Baseline",
                text=fp16_text,
                tokens_generated=fp16_tokens,
                time_sec=fp16_time,
            ))

            # 3-bit ResidualQuant generation
            print("\n  Generating with 3-bit ResidualQuant cache...")
            t_gen = time.time()
            try:
                rq_gen_cache = GenerationCache(
                    key_bits=3,
                    val_bits=2,
                    fp16_window=64,
                    anchor_interval=6,
                    use_residual_quant=True,
                )

                with torch.no_grad():
                    rq_output = model.generate(
                        **gen_inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        temperature=1.0,
                        past_key_values=rq_gen_cache,
                    )
                rq_time = time.time() - t_gen
                rq_text = tokenizer.decode(rq_output[0], skip_special_tokens=True)
                rq_tokens = rq_output.shape[1] - gen_inputs["input_ids"].shape[1]
                print(f"  RQ3: {rq_tokens} tokens in {rq_time:.2f}s")
                print(f"  RQ3 text: {rq_text[:200]}")

                generation_results.append(GenerationResult(
                    method="3-bit ResidualQuant (K3/V2, anchor=6, win=64)",
                    text=rq_text,
                    tokens_generated=rq_tokens,
                    time_sec=rq_time,
                ))

                if fp16_text.strip() == rq_text.strip():
                    findings.append("GENERATION MATCH: 3-bit output IDENTICAL to FP16")
                else:
                    fp16_toks = tokenizer.encode(fp16_text)
                    rq_toks = tokenizer.encode(rq_text)
                    common = sum(1 for a, b in zip(fp16_toks, rq_toks) if a == b)
                    match_pct = common / max(len(fp16_toks), 1) * 100
                    findings.append(f"GENERATION: {match_pct:.0f}% token match between FP16 and 3-bit RQ")

                # Memory stats
                if hasattr(rq_gen_cache, "memory_usage_bits"):
                    mem = rq_gen_cache.memory_usage_bits()
                    findings.append(
                        f"GenerationCache memory: {mem.get('compression_ratio', '?')}x compression"
                    )

            except Exception as e:
                print(f"  RQ generation FAILED: {e}")
                traceback.print_exc()
                findings.append(f"GenerationCache generation FAILED: {e}")

            # 4-bit ResidualQuant generation
            print("\n  Generating with 4-bit ResidualQuant cache...")
            t_gen = time.time()
            try:
                rq4_gen_cache = GenerationCache(
                    key_bits=4,
                    val_bits=3,
                    fp16_window=64,
                    anchor_interval=6,
                    use_residual_quant=True,
                )

                with torch.no_grad():
                    rq4_output = model.generate(
                        **gen_inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        temperature=1.0,
                        past_key_values=rq4_gen_cache,
                    )
                rq4_time = time.time() - t_gen
                rq4_text = tokenizer.decode(rq4_output[0], skip_special_tokens=True)
                rq4_tokens = rq4_output.shape[1] - gen_inputs["input_ids"].shape[1]
                print(f"  RQ4: {rq4_tokens} tokens in {rq4_time:.2f}s")
                print(f"  RQ4 text: {rq4_text[:200]}")

                generation_results.append(GenerationResult(
                    method="4-bit ResidualQuant (K4/V3, anchor=6, win=64)",
                    text=rq4_text,
                    tokens_generated=rq4_tokens,
                    time_sec=rq4_time,
                ))

                if fp16_text.strip() == rq4_text.strip():
                    findings.append("GENERATION 4-bit: output IDENTICAL to FP16")
                else:
                    fp16_toks = tokenizer.encode(fp16_text)
                    rq4_toks = tokenizer.encode(rq4_text)
                    common = sum(1 for a, b in zip(fp16_toks, rq4_toks) if a == b)
                    match_pct = common / max(len(fp16_toks), 1) * 100
                    findings.append(f"GENERATION 4-bit: {match_pct:.0f}% token match with FP16")

            except Exception as e:
                print(f"  RQ4 generation FAILED: {e}")
                traceback.print_exc()
                findings.append(f"GenerationCache 4-bit generation FAILED: {e}")

        except Exception as e:
            print(f"  Generation test FAILED: {e}")
            traceback.print_exc()
            findings.append(f"Generation test FAILED: {e}")

        timings["generation_test"] = time.time() - t0
    else:
        print(f"\n[4/4] Skipping generation test for this model")

    # ---- Cleanup ----
    print(f"\n  Cleaning up {model_name}...")
    del model
    if kv_layers is not None:
        del kv_layers
    del outputs
    del past_kv
    cleanup_gpu()
    print(f"  GPU after cleanup: {gpu_mem_mb()}")

    return ModelTestResult(
        model_name=model_name,
        model_info=model_info,
        aggregate_results=aggregate_results,
        generation_results=generation_results,
        findings=findings,
        timings=timings,
        success=True,
    )


# ---------------------------------------------------------------------------
# Format results as markdown
# ---------------------------------------------------------------------------

def format_all_results(results: List[ModelTestResult]) -> str:
    lines = []
    lines.append("# Large Model Validation Results")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Hardware:** RTX 4090 24GB, Ryzen 9 5900X, 62GB RAM")
    lines.append(f"**Library:** TurboQuantDC (ResidualQuant + PolarQuant)")
    lines.append("")

    # Summary table across all models
    lines.append("## Summary Across All Models")
    lines.append("")
    lines.append("| Model | Bits | Method | Cosine Sim | Top-1 | Top-5 | Ratio | Heads |")
    lines.append("|-------|------|--------|-----------|-------|-------|-------|-------|")
    for r in results:
        short_name = r.model_name.split("/")[-1]
        for ar in r.aggregate_results:
            lines.append(
                f"| {short_name} | {ar.bits} | {ar.method} | {ar.mean_cosine:.6f} | "
                f"{ar.top1_rate:.1%} | {ar.top5_rate:.1%} | {ar.compression_ratio:.2f}x | "
                f"{ar.num_heads_measured} |"
            )
    lines.append("")

    # Comparison with previously validated models
    lines.append("## Scaling Trend (All Validated Models)")
    lines.append("")
    lines.append("| Model | Params | d | 3-bit Cosine | 3-bit Top-5 | Compression |")
    lines.append("|-------|--------|---|-------------|-------------|-------------|")
    lines.append("| Qwen2.5-3B | 3B | 128 | 0.9969 | 94.4% | 5.0x |")
    lines.append("| Qwen2.5-14B | 14B | 128 | 0.9964 | 95.3% | 5.0x |")
    lines.append("| Qwen3.5-27B | 27B | 256 | 0.9932 | 100% | 5.2x |")
    lines.append("| Gemma 4 E4B | 4B | 256/512 | 0.999994 | 100% | - |")
    for r in results:
        rq3 = next((ar for ar in r.aggregate_results if ar.bits == 3 and ar.method == "ResidualQuant"), None)
        if rq3:
            short_name = r.model_name.split("/")[-1]
            params = "32B" if "32B" in short_name else "72B" if "72B" in short_name else "?"
            d = r.model_info.get("Head dimension", "?")
            lines.append(
                f"| {short_name} | {params} | {d} | {rq3.mean_cosine:.4f} | "
                f"{rq3.top5_rate:.0%} | {rq3.compression_ratio:.1f}x |"
            )
    lines.append("")

    # Paper targets
    lines.append("## Paper Targets")
    lines.append("")
    lines.append("| Metric | Target | Status |")
    lines.append("|--------|--------|--------|")
    for r in results:
        rq3 = next((ar for ar in r.aggregate_results if ar.bits == 3 and ar.method == "ResidualQuant"), None)
        if rq3:
            short = r.model_name.split("/")[-1]
            cos_ok = "PASS" if rq3.mean_cosine > 0.995 else "FAIL"
            t5_ok = "PASS" if rq3.top5_rate > 0.9 else "FAIL"
            ratio_ok = "PASS" if rq3.compression_ratio > 4.5 else "FAIL"
            lines.append(f"| {short} cos > 0.995 | {rq3.mean_cosine:.4f} | {cos_ok} |")
            lines.append(f"| {short} top-5 > 90% | {rq3.top5_rate:.0%} | {t5_ok} |")
            lines.append(f"| {short} ratio > 4.5x | {rq3.compression_ratio:.1f}x | {ratio_ok} |")
    lines.append("")

    # Per-model details
    for r in results:
        lines.append(f"## {r.model_name}")
        lines.append("")

        if not r.success:
            lines.append("**TEST FAILED** - see findings below.")
            lines.append("")

        # Model info
        lines.append("### Architecture")
        lines.append("")
        for k, v in r.model_info.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

        # Compression results
        if r.aggregate_results:
            lines.append("### Compression Quality")
            lines.append("")
            lines.append("| Bits | Method | Cosine Sim | Top-1 | Top-5 | Ratio | Heads |")
            lines.append("|------|--------|-----------|-------|-------|-------|-------|")
            for ar in r.aggregate_results:
                lines.append(
                    f"| {ar.bits} | {ar.method} | {ar.mean_cosine:.6f} | "
                    f"{ar.top1_rate:.1%} | {ar.top5_rate:.1%} | "
                    f"{ar.compression_ratio:.2f}x | {ar.num_heads_measured} |"
                )
            lines.append("")

        # Generation
        if r.generation_results:
            lines.append("### Generation Comparison")
            lines.append("")
            for gr in r.generation_results:
                lines.append(f"**{gr.method}** ({gr.tokens_generated} tokens, {gr.time_sec:.2f}s)")
                lines.append(f"> {gr.text}")
                lines.append("")

        # Timings
        if r.timings:
            lines.append("### Timings")
            lines.append("")
            for k, v in r.timings.items():
                lines.append(f"- **{k}:** {v:.2f}s")
            lines.append("")

        # Findings
        lines.append("### Findings")
        lines.append("")
        for f in r.findings:
            lines.append(f"- {f}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from transformers import BitsAndBytesConfig

    print("=" * 70)
    print("TurboQuantDC Large Model Validation")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {gpu_mem_mb()}")
    print("=" * 70)

    all_results: List[ModelTestResult] = []

    # ---- Test 1: Qwen2.5-32B-Instruct (BnB NF4) ----
    # 32B NF4: ~18GB for weights.  With embeddings/lm_head in FP16 it can
    # exceed 24GB, so cap GPU at 22GiB and let the rest spill to CPU.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    result_32b = test_model(
        model_name="Qwen/Qwen2.5-32B-Instruct",
        load_kwargs={
            "quantization_config": bnb_config,
            "device_map": "auto",
            "dtype": torch.float16,
            "max_memory": {0: "22GiB", "cpu": "40GiB"},
        },
        do_generation=True,
    )
    all_results.append(result_32b)

    # Force full cleanup before 72B
    cleanup_gpu()
    print(f"\n  GPU after 32B cleanup: {gpu_mem_mb()}")

    # ---- Test 2: Qwen2.5-72B-Instruct-GPTQ-Int4 ----
    # GPTQ-Int4 72B: ~39GB on disk.  device_map="auto" will place what fits
    # on GPU and offload the rest to CPU.  This exercises our multi-device
    # KV cache support.  Generation is skipped (too slow with CPU offload).
    print("\n" + "=" * 70)
    print("Attempting 72B model...")
    print("=" * 70)

    try:
        result_72b = test_model(
            model_name="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            load_kwargs={
                "device_map": "auto",
                "dtype": torch.float16,
                "max_memory": {0: "22GiB", "cpu": "50GiB"},
            },
            do_generation=False,  # Likely too slow with CPU offload; just test compression
        )
        all_results.append(result_72b)
    except Exception as e:
        print(f"\n72B FAILED at top level: {e}")
        traceback.print_exc()
        all_results.append(ModelTestResult(
            model_name="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            model_info={"Error": str(e)},
            aggregate_results=[],
            generation_results=[],
            findings=[f"TOP-LEVEL FAILURE: {e}"],
            timings={},
            success=False,
        ))

    # ---- Save all results ----
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for r in all_results:
        print(f"\n{r.model_name} ({'SUCCESS' if r.success else 'FAILED'}):")
        for ar in r.aggregate_results:
            print(f"  {ar.bits}b {ar.method}: cos={ar.mean_cosine:.6f} "
                  f"top1={ar.top1_rate:.1%} top5={ar.top5_rate:.1%} "
                  f"ratio={ar.compression_ratio:.2f}x")
        for f in r.findings:
            print(f"  - {f}")

    md = format_all_results(all_results)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.write(md)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
