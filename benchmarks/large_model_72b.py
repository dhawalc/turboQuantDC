"""Standalone 72B validation for TurboQuantDC.

Runs in a separate process to avoid OOM from 32B residue.
Tests Qwen2.5-72B-Instruct-GPTQ-Int4 with CPU offloading.

Usage:
    python benchmarks/large_model_72b.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

HF_CACHE = "/media/dhawal/Beast/cache/hub"
os.environ["HF_HOME"] = "/media/dhawal/Beast/cache"
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda"
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
BIT_WIDTHS = [3, 4]
TEST_PROMPT = (
    "You are a helpful assistant. Explain the concept of KV cache compression "
    "in large language models in exactly three sentences."
)


def gpu_mem_mb() -> str:
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        return f"{used:.0f}MB / {reserved:.0f}MB reserved / {total:.0f}MB total"
    return "N/A"


def compute_attention_scores(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    d = queries.shape[-1]
    scores = queries @ keys.T / math.sqrt(d)
    return F.softmax(scores, dim=-1)


def extract_kv_layers(past_kv) -> List[Tuple[torch.Tensor, torch.Tensor]]:
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


def main():
    print("=" * 70)
    print(f"Qwen2.5-72B-Instruct-GPTQ-Int4 Validation")
    print(f"GPU: {gpu_mem_mb()}")
    print("=" * 70)

    findings = []
    timings = {}

    # ---- Load model ----
    print("\n[1/3] Loading 72B GPTQ model...")
    t0 = time.time()

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)

        # GPTQ-Int4 72B: ~39GB.  GPU has ~23.5GB, so most layers must
        # offload to CPU.  offload_buffers=True is needed so the dispatch
        # mechanism doesn't pre-allocate GPU-side buffers for all CPU layers.
        # We limit GPU to 20GiB to leave room for KV cache + activation.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=HF_CACHE,
            device_map="auto",
            dtype=torch.float16,
            max_memory={0: "20GiB", "cpu": "50GiB"},
            offload_buffers=True,
        )

        timings["model_load"] = time.time() - t0
        print(f"  Loaded in {timings['model_load']:.1f}s")
        print(f"  GPU: {gpu_mem_mb()}")

        config = model.config
        text_config = getattr(config, "text_config", config)
        num_layers = getattr(text_config, "num_hidden_layers", "?")
        num_kv_heads = getattr(text_config, "num_key_value_heads", "?")
        num_attn_heads = getattr(text_config, "num_attention_heads", "?")
        hidden_size = getattr(text_config, "hidden_size", "?")
        head_dim = getattr(text_config, "head_dim", None)
        if head_dim is None and isinstance(hidden_size, int) and isinstance(num_attn_heads, int):
            head_dim = hidden_size // num_attn_heads

        print(f"  Architecture: {num_layers}L / {num_kv_heads} KV heads / d={head_dim}")

        if hasattr(model, "hf_device_map"):
            devices = {}
            for name, dev in model.hf_device_map.items():
                dev_str = str(dev)
                devices[dev_str] = devices.get(dev_str, 0) + 1
            print(f"  Device map: {devices}")
            findings.append(f"Device map: {devices}")

    except Exception as e:
        print(f"\n  LOAD FAILED: {e}")
        traceback.print_exc()
        findings.append(f"LOAD FAILED: {e}")
        print(f"\nFindings:")
        for f in findings:
            print(f"  - {f}")
        return

    # ---- Extract KV caches ----
    print("\n[2/3] Extracting KV caches...")
    t0 = time.time()

    try:
        inputs = tokenizer(TEST_PROMPT, return_tensors="pt")
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
        print(f"  Key shape: {sample_k.shape}")
        print(f"  Key dtype: {sample_k.dtype}, device: {sample_k.device}")

        findings.append(f"KV cache: {num_cached_layers}L, {actual_num_kv_heads} KV heads, seq={actual_seq}, d={actual_head_dim}")

        # Check multi-device
        layer_devices = {}
        for i, (k, v) in enumerate(kv_layers):
            dev = str(k.device)
            layer_devices[dev] = layer_devices.get(dev, 0) + 1
        print(f"  KV layer devices: {layer_devices}")
        findings.append(f"KV layer devices: {layer_devices}")

    except Exception as e:
        print(f"\n  KV EXTRACTION FAILED: {e}")
        traceback.print_exc()
        findings.append(f"KV EXTRACTION FAILED: {e}")
        print(f"\nFindings:")
        for f in findings:
            print(f"  - {f}")
        return

    # ---- Compression quality test ----
    print("\n[3/3] Testing compression quality...")
    t0 = time.time()

    from turboquantdc.residual_quant import ResidualQuantEstimator
    from turboquantdc.polarquant import PolarQuant

    d = actual_head_dim
    total_layers = len(kv_layers)

    # Sample layers
    sample_indices = sorted(set([
        0,
        total_layers // 4,
        total_layers // 2,
        3 * total_layers // 4,
        total_layers - 1,
    ]))
    print(f"  Sampling layers: {sample_indices} (of {total_layers})")

    for bits in BIT_WIDTHS:
        print(f"\n  --- {bits}-bit compression ---")

        try:
            rq = ResidualQuantEstimator(d=d, bits=bits, seed=42, device=DEVICE)
            pq = PolarQuant(d=d, bits=max(bits - 1, 1), seed=42, device=DEVICE)
        except Exception as e:
            findings.append(f"Estimator creation FAILED ({bits}b, d={d}): {e}")
            traceback.print_exc()
            continue

        rq_cosines = []
        rq_top1s = []
        rq_top5s = []
        pq_cosines = []
        pq_top1s = []
        pq_top5s = []
        debug_printed = False

        for layer_idx in sample_indices:
            keys_fp16, vals_fp16 = kv_layers[layer_idx]
            batch, num_h, seq, hd = keys_fp16.shape

            for head_idx in range(num_h):
                k = keys_fp16[0, head_idx].float().to(DEVICE)
                q = k[-1:, :]
                fp16_scores = compute_attention_scores(q, k)

                # ResidualQuant
                try:
                    compressed_rq = rq.quantize(k)
                    k_rq = rq.dequantize(compressed_rq)
                    rq_scores = compute_attention_scores(q, k_rq)

                    fp_flat = fp16_scores.flatten()
                    rq_flat = rq_scores.flatten()

                    # Debug: print first head's scores to understand nan
                    if not debug_printed:
                        print(f"      DEBUG L{layer_idx}H{head_idx}: fp16 scores range "
                              f"[{fp_flat.min():.8f}, {fp_flat.max():.8f}], "
                              f"sum={fp_flat.sum():.6f}, "
                              f"has_nan={fp_flat.isnan().any()}, has_inf={fp_flat.isinf().any()}")
                        print(f"      DEBUG: rq scores range "
                              f"[{rq_flat.min():.8f}, {rq_flat.max():.8f}], "
                              f"sum={rq_flat.sum():.6f}, "
                              f"has_nan={rq_flat.isnan().any()}, has_inf={rq_flat.isinf().any()}")
                        print(f"      DEBUG: fp_norm={fp_flat.norm():.8f}, rq_norm={rq_flat.norm():.8f}")
                        cos_debug = F.cosine_similarity(fp_flat.unsqueeze(0), rq_flat.unsqueeze(0))
                        print(f"      DEBUG: raw cosine_similarity = {cos_debug}")
                        debug_printed = True

                    # Compute cosine sim with nan guard
                    fp_n = fp_flat.norm()
                    rq_n = rq_flat.norm()
                    if fp_n > 1e-12 and rq_n > 1e-12:
                        cos = F.cosine_similarity(
                            fp_flat.unsqueeze(0), rq_flat.unsqueeze(0)
                        ).item()
                    else:
                        cos = 0.0
                    if math.isnan(cos):
                        cos = 0.0
                    rq_cosines.append(cos)

                    t1 = (fp16_scores.argmax(dim=-1) == rq_scores.argmax(dim=-1)).all().item()
                    rq_top1s.append(t1)

                    fp16_t5 = set(fp16_scores.topk(min(5, seq), dim=-1).indices[0].tolist())
                    rq_t5 = set(rq_scores.topk(min(5, seq), dim=-1).indices[0].tolist())
                    rq_top5s.append(bool(fp16_t5 & rq_t5))
                except Exception as e:
                    findings.append(f"RQ {bits}b L{layer_idx}H{head_idx}: {e}")

                # PolarQuant
                try:
                    k_norm = k.norm(dim=-1, keepdim=True)
                    k_unit = k / (k_norm + 1e-8)
                    indices = pq.quantize(k_unit)
                    k_pq = pq.dequantize(indices) * k_norm
                    pq_scores = compute_attention_scores(q, k_pq)

                    fp_flat = fp16_scores.flatten().unsqueeze(0)
                    pq_flat = pq_scores.flatten().unsqueeze(0)
                    if fp_flat.norm() > 1e-12 and pq_flat.norm() > 1e-12:
                        cos = F.cosine_similarity(fp_flat, pq_flat).item()
                    else:
                        cos = 0.0
                    if math.isnan(cos):
                        cos = 0.0
                    pq_cosines.append(cos)

                    t1 = (fp16_scores.argmax(dim=-1) == pq_scores.argmax(dim=-1)).all().item()
                    pq_top1s.append(t1)

                    fp16_t5 = set(fp16_scores.topk(min(5, seq), dim=-1).indices[0].tolist())
                    pq_t5 = set(pq_scores.topk(min(5, seq), dim=-1).indices[0].tolist())
                    pq_top5s.append(bool(fp16_t5 & pq_t5))
                except Exception as e:
                    findings.append(f"PQ {bits}b L{layer_idx}H{head_idx}: {e}")

        # Print results
        if rq_cosines:
            rq_cos = sum(rq_cosines) / len(rq_cosines)
            rq_t1 = sum(rq_top1s) / len(rq_top1s)
            rq_t5 = sum(rq_top5s) / len(rq_top5s)
            compressed_bpv = bits * d + 32
            fp16_bpv = d * 16
            ratio = fp16_bpv / compressed_bpv
            print(f"    RQ {bits}b: cos={rq_cos:.6f} top1={rq_t1:.1%} top5={rq_t5:.1%} ratio={ratio:.2f}x ({len(rq_cosines)} heads)")
            findings.append(f"RQ {bits}b: cos={rq_cos:.6f} top1={rq_t1:.1%} top5={rq_t5:.1%} ratio={ratio:.2f}x")

        if pq_cosines:
            pq_cos = sum(pq_cosines) / len(pq_cosines)
            pq_t1 = sum(pq_top1s) / len(pq_top1s)
            pq_t5 = sum(pq_top5s) / len(pq_top5s)
            mse_bits = max(bits - 1, 1)
            compressed_bpv = mse_bits * d + 16
            fp16_bpv = d * 16
            ratio = fp16_bpv / compressed_bpv
            print(f"    PQ {bits}b: cos={pq_cos:.6f} top1={pq_t1:.1%} top5={pq_t5:.1%} ratio={ratio:.2f}x ({len(pq_cosines)} heads)")
            findings.append(f"PQ {bits}b: cos={pq_cos:.6f} top1={pq_t1:.1%} top5={pq_t5:.1%} ratio={ratio:.2f}x")

    timings["compression_test"] = time.time() - t0

    # ---- Bonus: Try generation on 72B ----
    print("\n[BONUS] Attempting generation on 72B...")
    t0 = time.time()

    try:
        from turboquantdc.generation_cache import GenerationCache

        gen_prompt = "The three most important inventions of the 20th century are"
        gen_inputs = tokenizer(gen_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        gen_inputs = {k: v.to(device) for k, v in gen_inputs.items()}

        # FP16 baseline
        print("  FP16 baseline generation (50 tokens)...")
        t_gen = time.time()
        with torch.no_grad():
            fp16_out = model.generate(
                **gen_inputs, max_new_tokens=50, do_sample=False, temperature=1.0,
            )
        fp16_time = time.time() - t_gen
        fp16_text = tokenizer.decode(fp16_out[0], skip_special_tokens=True)
        fp16_tokens = fp16_out.shape[1] - gen_inputs["input_ids"].shape[1]
        print(f"  FP16: {fp16_tokens} tokens in {fp16_time:.1f}s")
        print(f"  FP16: {fp16_text[:200]}")
        findings.append(f"FP16 generation: {fp16_tokens} tokens in {fp16_time:.1f}s")

        # 3-bit ResidualQuant
        print("  3-bit RQ generation (50 tokens)...")
        t_gen = time.time()
        rq_cache = GenerationCache(
            key_bits=3, val_bits=2, fp16_window=64,
            anchor_interval=6, use_residual_quant=True,
        )
        with torch.no_grad():
            rq_out = model.generate(
                **gen_inputs, max_new_tokens=50, do_sample=False, temperature=1.0,
                past_key_values=rq_cache,
            )
        rq_time = time.time() - t_gen
        rq_text = tokenizer.decode(rq_out[0], skip_special_tokens=True)
        rq_tokens = rq_out.shape[1] - gen_inputs["input_ids"].shape[1]
        print(f"  RQ3: {rq_tokens} tokens in {rq_time:.1f}s")
        print(f"  RQ3: {rq_text[:200]}")
        findings.append(f"RQ3 generation: {rq_tokens} tokens in {rq_time:.1f}s")

        if fp16_text.strip() == rq_text.strip():
            findings.append("72B GENERATION MATCH: 3-bit output IDENTICAL to FP16!")
            print("  >>> IDENTICAL OUTPUT <<<")
        else:
            fp16_toks = tokenizer.encode(fp16_text)
            rq_toks = tokenizer.encode(rq_text)
            common = sum(1 for a, b in zip(fp16_toks, rq_toks) if a == b)
            pct = common / max(len(fp16_toks), 1) * 100
            findings.append(f"72B generation: {pct:.0f}% token match")
            print(f"  Token match: {pct:.0f}%")

    except Exception as e:
        print(f"  Generation FAILED: {e}")
        traceback.print_exc()
        findings.append(f"72B generation FAILED: {e}")

    timings["generation_test"] = time.time() - t0

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Architecture: {num_layers}L / {num_kv_heads} KV heads / d={head_dim}")
    for f in findings:
        print(f"  - {f}")
    for k, v in timings.items():
        print(f"  {k}: {v:.1f}s")


if __name__ == "__main__":
    main()
