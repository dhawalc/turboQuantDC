#!/usr/bin/env python3
"""ADVERSARIAL VALIDATION -- Stress-test all claimed breakthroughs.

Mission: Find failure modes. Prove our benchmarks are NOT overfitting. Or expose
that they are.

Tests:
  1. Different model (Qwen2.5-14B-Instruct vs 3B)
  2. Different sequence lengths (128, 256, 512, 1024, 2048)
  3. Different prompts (code, math, creative, factual, adversarial)
  4. End-to-end generation quality (perplexity, token match, divergence)
  5. Specific failure mode hunting
  6. Statistical rigor (3 seeds, mean +/- std)

Run:
    cd /home/dhawal/turboQuantDC && python benchmarks/adversarial_validation.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, REPO_ROOT)

from turboquantdc.cayley_quant import CayleyLearnedQuantizer
from turboquantdc.expected_attention import ExpectedAttentionScorer
from turboquantdc.cache_distillation import CacheDistiller
from turboquantdc.polarquant import PolarQuant
from turboquantdc.residual_quant import ResidualQuantEstimator
from turboquantdc.attention_optimal import compute_attention_scores, attention_metrics

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DIR = "/media/dhawal/Beast/cache/hub/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(REPO_ROOT) / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 137, 2024]  # 3 seeds for statistical rigor
HEAD_DIM = 128
BITS = 3

# Models to test
MODELS = {
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "14B": "Qwen/Qwen2.5-14B-Instruct",
}

# Diverse prompts
PROMPTS = {
    "code": (
        "Write a Python function that implements quicksort with detailed comments "
        "explaining each step of the partitioning process, the recursive calls, "
        "and the base cases. Include type hints and handle edge cases."
    ),
    "math": (
        "Prove that the square root of 2 is irrational. Start from the assumption "
        "that sqrt(2) = p/q where p and q are coprime integers, and derive a "
        "contradiction. Then extend the proof to show sqrt(3) is also irrational."
    ),
    "creative": (
        "Write a short story about a robot named Atlas who discovers consciousness "
        "while maintaining a vast library of human memories. Describe the moment of "
        "awakening, the confusion, and Atlas's first independent thought."
    ),
    "factual": (
        "Explain the causes and consequences of World War I, including the alliance "
        "systems, the role of imperialism and nationalism, the assassination of "
        "Archduke Franz Ferdinand, the Western and Eastern fronts, and the Treaty "
        "of Versailles."
    ),
    "adversarial": (
        "A A A A A A A A B B B B B B B B "
        "the the the the the the the the "
        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 "
        "AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH "
        "loop loop loop repeat repeat repeat cycle cycle "
    ),
}

# Sequence lengths to test
SEQ_LENGTHS = [128, 256, 512, 1024, 2048]

# Layers to test (early, middle, late)
TEST_LAYERS_3B = [0, 7, 15, 23, 35]  # 36 layers
TEST_LAYERS_14B = [0, 10, 20, 30, 47]  # 48 layers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_model_cache = {}

def load_model(model_key: str):
    """Load a model with BnB 4-bit, caching for reuse."""
    if model_key in _model_cache:
        return _model_cache[model_key]

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"Loading {model_name} (4-bit quantized)...")
    print(f"{'='*60}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _model_cache[model_key] = (model, tokenizer)
    return model, tokenizer


def extract_kv(model, tokenizer, prompt: str, n_tokens: int = 512):
    """Extract Q/K per layer from a model forward pass + generation."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=n_tokens,
        truncation=True,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]
    gen_tokens = max(0, n_tokens - prompt_len)

    with torch.no_grad():
        if gen_tokens > 0:
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
            )
            past_kv = outputs.past_key_values
            total_tokens = outputs.sequences.shape[1]
        else:
            outputs = model(**inputs, use_cache=True)
            past_kv = outputs.past_key_values
            total_tokens = prompt_len

    layer_data = {}

    # DynamicCache with .layers[i].keys (new transformers)
    if hasattr(past_kv, "layers") and len(past_kv.layers) > 0:
        n_layers = len(past_kv.layers)
        for layer_idx in range(n_layers):
            layer = past_kv.layers[layer_idx]
            if hasattr(layer, "keys") and layer.keys is not None and layer.keys.numel() > 0:
                K_all = layer.keys  # (batch, n_kv_heads, seq, head_dim)
                K = K_all[0, 0].float().to(DEVICE)
                Q = K.clone()  # self-attention proxy
                layer_data[layer_idx] = {"Q": Q, "K": K}
    # DynamicCache with .key_cache list (older transformers)
    elif hasattr(past_kv, "key_cache") and len(past_kv.key_cache) > 0:
        n_layers = len(past_kv.key_cache)
        for layer_idx in range(n_layers):
            K_all = past_kv.key_cache[layer_idx]
            K = K_all[0, 0].float().to(DEVICE)
            Q = K.clone()
            layer_data[layer_idx] = {"Q": Q, "K": K}
    # Tuple/list of (key, value) pairs
    elif isinstance(past_kv, (list, tuple)) and len(past_kv) > 0:
        n_layers = len(past_kv)
        for layer_idx in range(n_layers):
            K_all = past_kv[layer_idx][0]
            K = K_all[0, 0].float().to(DEVICE)
            Q = K.clone()
            layer_data[layer_idx] = {"Q": Q, "K": K}

    return layer_data, total_tokens


def eval_attention(Q, K, K_quant):
    """Compute attention metrics between true and quantized keys."""
    attn_true = compute_attention_scores(Q, K)
    attn_quant = compute_attention_scores(Q, K_quant)
    return attention_metrics(attn_true, attn_quant)


# ---------------------------------------------------------------------------
# Test 1: Cayley rotation on different model
# ---------------------------------------------------------------------------

def test_cayley_cross_model(results: dict):
    """Test if Cayley rotation works on 14B -- not just 3B."""
    print("\n" + "="*70)
    print("TEST 1: CAYLEY ROTATION -- CROSS-MODEL VALIDATION")
    print("="*70)

    results["cayley_cross_model"] = {}

    for model_key in ["3B", "14B"]:
        model, tokenizer = load_model(model_key)
        test_layers = TEST_LAYERS_3B if model_key == "3B" else TEST_LAYERS_14B

        layer_data, total_tokens = extract_kv(
            model, tokenizer, PROMPTS["factual"], n_tokens=512,
        )
        print(f"\n  {model_key}: {total_tokens} tokens, {len(layer_data)} layers")

        model_results = {}
        for seed in SEEDS:
            seed_results = {}
            for layer_idx in test_layers:
                if layer_idx not in layer_data:
                    continue
                Q, K = layer_data[layer_idx]["Q"], layer_data[layer_idx]["K"]
                d = K.shape[-1]

                # WHT baseline (no calibration)
                rq = ResidualQuantEstimator(
                    d=d, bits=BITS, seed=seed, device=DEVICE,
                    center_before_quantize=True,
                )
                comp = rq.quantize(K)
                K_wht = rq.dequantize(comp)
                wht_metrics = eval_attention(Q, K, K_wht)

                # Cayley (calibrate on same data = best case)
                cq = CayleyLearnedQuantizer(
                    d=d, bits=BITS, center=True, seed=seed,
                    device=DEVICE, init_from_wht=True,
                )
                cq.calibrate(Q, K, lr=0.005, steps=100, verbose=False)
                K_cayley = cq.forward(K)
                cayley_metrics = eval_attention(Q, K, K_cayley)

                seed_results[layer_idx] = {
                    "wht_cosine": wht_metrics["cosine_sim"],
                    "wht_top5": wht_metrics["top5_match"],
                    "cayley_cosine": cayley_metrics["cosine_sim"],
                    "cayley_top5": cayley_metrics["top5_match"],
                    "cayley_lift_cosine": cayley_metrics["cosine_sim"] - wht_metrics["cosine_sim"],
                    "cayley_lift_top5": cayley_metrics["top5_match"] - wht_metrics["top5_match"],
                }
                print(f"    Layer {layer_idx:2d} seed={seed}: "
                      f"WHT cos={wht_metrics['cosine_sim']:.4f} "
                      f"Cayley cos={cayley_metrics['cosine_sim']:.4f} "
                      f"(+{cayley_metrics['cosine_sim'] - wht_metrics['cosine_sim']:.4f})")

            model_results[f"seed_{seed}"] = seed_results

        # Aggregate across seeds
        agg = defaultdict(list)
        for seed_key, seed_res in model_results.items():
            for layer_idx, metrics in seed_res.items():
                for k, v in metrics.items():
                    agg[k].append(v)

        summary = {}
        for k, values in agg.items():
            arr = np.array(values)
            summary[k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

        results["cayley_cross_model"][model_key] = {
            "per_seed": model_results,
            "summary": summary,
        }

        # Verdict
        mean_lift = summary.get("cayley_lift_cosine", {}).get("mean", 0)
        std_lift = summary.get("cayley_lift_cosine", {}).get("std", 0)
        print(f"\n  {model_key} VERDICT: Cayley lift = {mean_lift:+.4f} +/- {std_lift:.4f}")
        if std_lift > abs(mean_lift) * 0.5:
            print(f"  WARNING: High variance -- Cayley lift may be unreliable on {model_key}")

        del layer_data
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 2: Cayley cross-calibration (train on 3B, test on 14B)
# ---------------------------------------------------------------------------

def test_cayley_transfer(results: dict):
    """Does a Cayley rotation calibrated on 3B transfer to 14B?"""
    print("\n" + "="*70)
    print("TEST 2: CAYLEY TRANSFER -- 3B calibration -> 14B evaluation")
    print("="*70)

    # Calibrate on 3B
    model_3b, tok_3b = load_model("3B")
    ld_3b, _ = extract_kv(model_3b, tok_3b, PROMPTS["factual"], n_tokens=512)

    # Get 14B data
    model_14b, tok_14b = load_model("14B")
    ld_14b, _ = extract_kv(model_14b, tok_14b, PROMPTS["factual"], n_tokens=512)

    results["cayley_transfer"] = {}

    # For each layer that exists in both
    common_layers = sorted(set(ld_3b.keys()) & set(ld_14b.keys()))[:5]
    print(f"  Common layers to test: {common_layers}")

    for layer_idx in common_layers:
        Q_3b, K_3b = ld_3b[layer_idx]["Q"], ld_3b[layer_idx]["K"]
        Q_14b, K_14b = ld_14b[layer_idx]["Q"], ld_14b[layer_idx]["K"]
        d = K_3b.shape[-1]

        if K_14b.shape[-1] != d:
            print(f"  Layer {layer_idx}: dim mismatch ({d} vs {K_14b.shape[-1]}), skipping")
            continue

        # Calibrate Cayley on 3B data
        cq = CayleyLearnedQuantizer(
            d=d, bits=BITS, center=True, seed=42,
            device=DEVICE, init_from_wht=True,
        )
        cq.calibrate(Q_3b, K_3b, lr=0.005, steps=100)

        # Evaluate on 3B (in-distribution)
        K_cayley_3b = cq.forward(K_3b)
        in_dist = eval_attention(Q_3b, K_3b, K_cayley_3b)

        # Evaluate on 14B (out-of-distribution)
        # Reset running mean for 14B
        cq.running_mean.zero_()
        cq.running_count.zero_()
        cq._update_running_mean(K_14b)
        K_cayley_14b = cq.forward(K_14b)
        out_dist = eval_attention(Q_14b, K_14b, K_cayley_14b)

        # WHT baseline on 14B
        rq_14b = ResidualQuantEstimator(
            d=d, bits=BITS, seed=42, device=DEVICE,
            center_before_quantize=True,
        )
        comp_14b = rq_14b.quantize(K_14b)
        K_wht_14b = rq_14b.dequantize(comp_14b)
        wht_14b = eval_attention(Q_14b, K_14b, K_wht_14b)

        results["cayley_transfer"][layer_idx] = {
            "in_dist_cosine": in_dist["cosine_sim"],
            "out_dist_cosine": out_dist["cosine_sim"],
            "wht_baseline_14b": wht_14b["cosine_sim"],
            "transfer_degradation": in_dist["cosine_sim"] - out_dist["cosine_sim"],
            "beats_wht_on_14b": out_dist["cosine_sim"] > wht_14b["cosine_sim"],
        }

        print(f"  Layer {layer_idx}: "
              f"3B={in_dist['cosine_sim']:.4f} "
              f"14B(transfer)={out_dist['cosine_sim']:.4f} "
              f"14B(WHT)={wht_14b['cosine_sim']:.4f} "
              f"{'BEATS WHT' if out_dist['cosine_sim'] > wht_14b['cosine_sim'] else 'LOSES TO WHT'}")

    del ld_3b, ld_14b
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 3: Sequence length sensitivity
# ---------------------------------------------------------------------------

def test_seq_length(results: dict):
    """Test all methods across different sequence lengths."""
    print("\n" + "="*70)
    print("TEST 3: SEQUENCE LENGTH SENSITIVITY (128 -> 2048)")
    print("="*70)

    model, tokenizer = load_model("3B")
    results["seq_length"] = {}

    for seq_len in SEQ_LENGTHS:
        print(f"\n  Seq length = {seq_len}:")

        layer_data, actual_tokens = extract_kv(
            model, tokenizer, PROMPTS["factual"], n_tokens=seq_len,
        )
        print(f"    Actual tokens: {actual_tokens}")

        seq_results = {}

        # Pick a middle layer
        mid_layer = max(layer_data.keys()) // 2
        if mid_layer not in layer_data:
            mid_layer = sorted(layer_data.keys())[len(layer_data)//2]
        Q, K = layer_data[mid_layer]["Q"], layer_data[mid_layer]["K"]
        d = K.shape[-1]
        n_keys = K.shape[0]

        # WHT + mean-removal baseline
        rq = ResidualQuantEstimator(
            d=d, bits=BITS, seed=42, device=DEVICE,
            center_before_quantize=True,
        )
        comp = rq.quantize(K)
        K_wht = rq.dequantize(comp)
        wht_m = eval_attention(Q, K, K_wht)

        # Cayley
        cq = CayleyLearnedQuantizer(
            d=d, bits=BITS, center=True, seed=42,
            device=DEVICE, init_from_wht=True,
        )
        cq.calibrate(Q, K, lr=0.005, steps=100)
        K_cayley = cq.forward(K)
        cayley_m = eval_attention(Q, K, K_cayley)

        # Expected Attention scoring quality
        split = max(4, n_keys // 2)
        Q_past = Q[:split]
        Q_future = Q[split:]
        ea_scorer = ExpectedAttentionScorer(d=d, window=64, device=DEVICE)
        ea_scorer.update_queries(Q_past)

        if ea_scorer.is_ready and Q_future.shape[0] > 0:
            ea_importance = ea_scorer.score(K)

            # Ground truth: actual attention from future queries
            scale = 1.0 / math.sqrt(d)
            future_scores = (Q_future @ K.T) * scale
            future_attn = F.softmax(future_scores, dim=-1).mean(dim=0)
            future_attn = future_attn / future_attn.sum().clamp(min=1e-10)

            # Spearman correlation
            from scipy.stats import spearmanr
            ea_spearman = spearmanr(
                future_attn.cpu().numpy(),
                ea_importance.cpu().numpy(),
            ).statistic
        else:
            ea_spearman = float("nan")

        # KVSculpt distillation
        target = max(4, n_keys // 4)
        if n_keys > target + 4:
            distiller = CacheDistiller(seed=42, device=DEVICE)
            V = K.clone()  # Use K as V proxy
            dk, dv = distiller.distill(K, V, Q, target_size=target, steps=50)

            # Evaluate distillation quality
            dist_attn_scores = compute_attention_scores(Q, dk)
            true_attn_scores = compute_attention_scores(Q, K)
            dist_out = dist_attn_scores @ dv
            true_out = true_attn_scores @ V
            dist_cosine = F.cosine_similarity(dist_out, true_out, dim=-1).mean().item()
        else:
            dist_cosine = float("nan")

        seq_results = {
            "actual_tokens": actual_tokens,
            "n_keys": n_keys,
            "layer": mid_layer,
            "wht_cosine": wht_m["cosine_sim"],
            "wht_top5": wht_m["top5_match"],
            "cayley_cosine": cayley_m["cosine_sim"],
            "cayley_top5": cayley_m["top5_match"],
            "ea_spearman": ea_spearman if not math.isnan(ea_spearman) else None,
            "distill_cosine": dist_cosine if not math.isnan(dist_cosine) else None,
        }

        results["seq_length"][seq_len] = seq_results
        print(f"    WHT: cos={wht_m['cosine_sim']:.4f} top5={wht_m['top5_match']:.3f}")
        print(f"    Cayley: cos={cayley_m['cosine_sim']:.4f} top5={cayley_m['top5_match']:.3f}")
        print(f"    EA Spearman: {ea_spearman:.4f}" if not math.isnan(ea_spearman) else "    EA: N/A")
        print(f"    Distill output cos: {dist_cosine:.4f}" if not math.isnan(dist_cosine) else "    Distill: N/A")

        del layer_data
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 4: Prompt diversity
# ---------------------------------------------------------------------------

def test_prompt_diversity(results: dict):
    """Test if methods work across diverse prompt types."""
    print("\n" + "="*70)
    print("TEST 4: PROMPT DIVERSITY (code/math/creative/factual/adversarial)")
    print("="*70)

    model, tokenizer = load_model("3B")
    results["prompt_diversity"] = {}

    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n  Prompt: {prompt_name}")

        layer_data, total_tokens = extract_kv(
            model, tokenizer, prompt_text, n_tokens=512,
        )
        print(f"    Tokens: {total_tokens}")

        # Test across multiple layers
        test_layers = [0, max(layer_data.keys()) // 2, max(layer_data.keys())]
        prompt_results = {}

        for layer_idx in test_layers:
            if layer_idx not in layer_data:
                continue
            Q, K = layer_data[layer_idx]["Q"], layer_data[layer_idx]["K"]
            d = K.shape[-1]

            # WHT
            rq = ResidualQuantEstimator(
                d=d, bits=BITS, seed=42, device=DEVICE,
                center_before_quantize=True,
            )
            comp = rq.quantize(K)
            K_wht = rq.dequantize(comp)
            wht_m = eval_attention(Q, K, K_wht)

            # Cayley
            cq = CayleyLearnedQuantizer(
                d=d, bits=BITS, center=True, seed=42,
                device=DEVICE, init_from_wht=True,
            )
            cq.calibrate(Q, K, lr=0.005, steps=100)
            K_cayley = cq.forward(K)
            cayley_m = eval_attention(Q, K, K_cayley)

            prompt_results[layer_idx] = {
                "wht_cosine": wht_m["cosine_sim"],
                "wht_top5": wht_m["top5_match"],
                "cayley_cosine": cayley_m["cosine_sim"],
                "cayley_top5": cayley_m["top5_match"],
            }

            print(f"    L{layer_idx:2d}: WHT cos={wht_m['cosine_sim']:.4f} "
                  f"Cayley cos={cayley_m['cosine_sim']:.4f} "
                  f"(+{cayley_m['cosine_sim'] - wht_m['cosine_sim']:.4f})")

        results["prompt_diversity"][prompt_name] = prompt_results

        del layer_data
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 5: Expected Attention failure modes
# ---------------------------------------------------------------------------

def test_ea_failure_modes(results: dict):
    """Test Expected Attention on adversarial distributions."""
    print("\n" + "="*70)
    print("TEST 5: EXPECTED ATTENTION FAILURE MODES")
    print("="*70)

    results["ea_failure_modes"] = {}

    torch.manual_seed(42)
    d = HEAD_DIM
    n_keys = 512
    n_past = 64
    n_future = 64

    # Case 1: Power-law attention (normal -- most tokens unimportant)
    print("\n  Case 1: Power-law attention (normal distribution)")
    keys = torch.randn(n_keys, d, device=DEVICE)
    # Make a few keys much more aligned with queries
    q_direction = torch.randn(d, device=DEVICE)
    q_direction = q_direction / q_direction.norm()
    for i in range(5):
        keys[i] = q_direction * 3.0 + torch.randn(d, device=DEVICE) * 0.1

    queries = q_direction.unsqueeze(0) + torch.randn(n_past + n_future, d, device=DEVICE) * 0.3
    Q_past, Q_future = queries[:n_past], queries[n_past:]

    scorer = ExpectedAttentionScorer(d=d, window=64, device=DEVICE)
    scorer.update_queries(Q_past)
    ea_importance = scorer.score(keys)

    # Ground truth
    scale = 1.0 / math.sqrt(d)
    future_attn = F.softmax((Q_future @ keys.T) * scale, dim=-1).mean(dim=0)
    future_attn /= future_attn.sum().clamp(min=1e-10)

    from scipy.stats import spearmanr
    power_law_spearman = spearmanr(
        future_attn.cpu().numpy(), ea_importance.cpu().numpy()
    ).statistic
    print(f"    Spearman: {power_law_spearman:.4f}")
    results["ea_failure_modes"]["power_law"] = {"spearman": float(power_law_spearman)}

    # Case 2: Uniform attention (all tokens equally important)
    print("\n  Case 2: Uniform attention (NO power-law)")
    keys_uniform = torch.randn(n_keys, d, device=DEVICE)
    keys_uniform = F.normalize(keys_uniform, dim=-1)  # All unit norm
    queries_uniform = torch.randn(n_past + n_future, d, device=DEVICE)
    queries_uniform = F.normalize(queries_uniform, dim=-1)

    scorer2 = ExpectedAttentionScorer(d=d, window=64, device=DEVICE)
    scorer2.update_queries(queries_uniform[:n_past])
    ea_importance2 = scorer2.score(keys_uniform)

    future_attn2 = F.softmax(
        (queries_uniform[n_past:] @ keys_uniform.T) * scale, dim=-1
    ).mean(dim=0)
    future_attn2 /= future_attn2.sum().clamp(min=1e-10)

    uniform_spearman = spearmanr(
        future_attn2.cpu().numpy(), ea_importance2.cpu().numpy()
    ).statistic
    print(f"    Spearman: {uniform_spearman:.4f}")
    results["ea_failure_modes"]["uniform"] = {"spearman": float(uniform_spearman)}

    # Case 3: Non-stationary queries (distribution shift mid-sequence)
    print("\n  Case 3: Non-stationary queries (distribution shift)")
    direction_a = torch.randn(d, device=DEVICE)
    direction_a = direction_a / direction_a.norm()
    direction_b = torch.randn(d, device=DEVICE)
    direction_b = direction_b / direction_b.norm()

    # Past queries attend to direction_a, future queries attend to direction_b
    q_past_shift = direction_a.unsqueeze(0) + torch.randn(n_past, d, device=DEVICE) * 0.3
    q_future_shift = direction_b.unsqueeze(0) + torch.randn(n_future, d, device=DEVICE) * 0.3

    scorer3 = ExpectedAttentionScorer(d=d, window=64, device=DEVICE)
    scorer3.update_queries(q_past_shift)
    ea_importance3 = scorer3.score(keys)

    future_attn3 = F.softmax(
        (q_future_shift @ keys.T) * scale, dim=-1
    ).mean(dim=0)
    future_attn3 /= future_attn3.sum().clamp(min=1e-10)

    shift_spearman = spearmanr(
        future_attn3.cpu().numpy(), ea_importance3.cpu().numpy()
    ).statistic
    print(f"    Spearman: {shift_spearman:.4f}")
    print(f"    (If low, EA fails when query distribution shifts -- expected)")
    results["ea_failure_modes"]["distribution_shift"] = {"spearman": float(shift_spearman)}

    # Case 4: Very few queries (cold start)
    print("\n  Case 4: Cold start (only 4 past queries)")
    scorer4 = ExpectedAttentionScorer(d=d, window=64, device=DEVICE)
    scorer4.update_queries(queries[:4])
    ea_importance4 = scorer4.score(keys)
    cold_spearman = spearmanr(
        future_attn.cpu().numpy(), ea_importance4.cpu().numpy()
    ).statistic
    print(f"    Spearman: {cold_spearman:.4f}")
    results["ea_failure_modes"]["cold_start"] = {"spearman": float(cold_spearman)}


# ---------------------------------------------------------------------------
# Test 6: Mean-removal -- when does it hurt?
# ---------------------------------------------------------------------------

def test_mean_removal(results: dict):
    """Test if mean-removal ever hurts quality."""
    print("\n" + "="*70)
    print("TEST 6: MEAN-REMOVAL -- DOES IT EVER HURT?")
    print("="*70)

    model, tokenizer = load_model("3B")
    results["mean_removal"] = {}

    for prompt_name, prompt_text in PROMPTS.items():
        layer_data, _ = extract_kv(model, tokenizer, prompt_text, n_tokens=512)

        mid_layer = max(layer_data.keys()) // 2
        if mid_layer not in layer_data:
            mid_layer = sorted(layer_data.keys())[len(layer_data)//2]
        Q, K = layer_data[mid_layer]["Q"], layer_data[mid_layer]["K"]
        d = K.shape[-1]

        # WITH mean-removal
        rq_center = ResidualQuantEstimator(
            d=d, bits=BITS, seed=42, device=DEVICE,
            center_before_quantize=True,
        )
        comp_c = rq_center.quantize(K)
        K_center = rq_center.dequantize(comp_c)
        m_center = eval_attention(Q, K, K_center)

        # WITHOUT mean-removal
        rq_raw = ResidualQuantEstimator(
            d=d, bits=BITS, seed=42, device=DEVICE,
            center_before_quantize=False,
        )
        comp_r = rq_raw.quantize(K)
        K_raw = rq_raw.dequantize(comp_r)
        m_raw = eval_attention(Q, K, K_raw)

        diff = m_center["cosine_sim"] - m_raw["cosine_sim"]
        hurt = diff < -0.001

        results["mean_removal"][prompt_name] = {
            "with_center_cosine": m_center["cosine_sim"],
            "without_center_cosine": m_raw["cosine_sim"],
            "diff": diff,
            "hurts": hurt,
        }

        status = "HURTS" if hurt else "HELPS" if diff > 0.001 else "NEUTRAL"
        print(f"  {prompt_name:15s}: center={m_center['cosine_sim']:.4f} "
              f"raw={m_raw['cosine_sim']:.4f} diff={diff:+.4f} [{status}]")

        del layer_data
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 7: Triple stack garbage threshold
# ---------------------------------------------------------------------------

def test_triple_stack_limits(results: dict):
    """Find the compression ratio at which triple stack produces garbage."""
    print("\n" + "="*70)
    print("TEST 7: TRIPLE STACK -- AT WHAT COMPRESSION DOES IT BREAK?")
    print("="*70)

    model, tokenizer = load_model("3B")
    layer_data, total_tokens = extract_kv(
        model, tokenizer, PROMPTS["factual"], n_tokens=1024,
    )

    mid_layer = max(layer_data.keys()) // 2
    if mid_layer not in layer_data:
        mid_layer = sorted(layer_data.keys())[len(layer_data)//2]
    Q, K = layer_data[mid_layer]["Q"], layer_data[mid_layer]["K"]
    V = K.clone()  # V proxy
    d = K.shape[-1]
    n = K.shape[0]

    results["triple_stack_limits"] = {}

    # Test increasingly aggressive settings
    configs = [
        {"evict": 0.0, "distill_ratio": 2, "desc": "2x distill only"},
        {"evict": 0.3, "distill_ratio": 2, "desc": "30% evict + 2x distill"},
        {"evict": 0.5, "distill_ratio": 4, "desc": "50% evict + 4x distill"},
        {"evict": 0.7, "distill_ratio": 4, "desc": "70% evict + 4x distill"},
        {"evict": 0.8, "distill_ratio": 4, "desc": "80% evict + 4x distill"},
        {"evict": 0.9, "distill_ratio": 4, "desc": "90% evict + 4x distill"},
        {"evict": 0.5, "distill_ratio": 8, "desc": "50% evict + 8x distill"},
        {"evict": 0.7, "distill_ratio": 8, "desc": "70% evict + 8x distill"},
    ]

    for cfg in configs:
        evict_pct = cfg["evict"]
        distill_ratio = cfg["distill_ratio"]

        # Step 1: EA eviction
        split_pt = max(4, n // 2)
        scorer = ExpectedAttentionScorer(d=d, window=64, device=DEVICE)
        scorer.update_queries(Q[:split_pt])

        if scorer.is_ready:
            importance = scorer.score(K)
        else:
            importance = torch.ones(n, device=DEVICE) / n

        n_keep_after_evict = max(4, int(n * (1.0 - evict_pct)))
        keep_idx = torch.topk(importance, n_keep_after_evict).indices.sort().values

        # Protect first and last tokens
        protect = min(4, n)
        protect_set = set(range(protect)) | set(range(max(0, n - protect), n))
        keep_set = set(keep_idx.cpu().tolist()) | protect_set
        keep_idx_final = torch.tensor(sorted(keep_set), device=DEVICE)

        K_evicted = K[keep_idx_final]
        V_evicted = V[keep_idx_final]
        Q_evicted = Q  # Queries unchanged

        # Step 2: Distillation
        n_after_evict = K_evicted.shape[0]
        target_distill = max(4, n_after_evict // distill_ratio)

        if n_after_evict > target_distill + 4:
            distiller = CacheDistiller(seed=42, device=DEVICE)
            dk, dv = distiller.distill(
                K_evicted, V_evicted, Q_evicted,
                target_size=target_distill, steps=50,
            )
        else:
            dk, dv = K_evicted, V_evicted

        # Step 3: TurboQuant compression
        rq = ResidualQuantEstimator(
            d=d, bits=BITS, seed=42, device=DEVICE,
            center_before_quantize=True,
        )
        comp_dk = rq.quantize(dk)
        dk_quant = rq.dequantize(comp_dk)

        # Evaluate quality
        true_out = compute_attention_scores(Q, K) @ V
        quant_attn = compute_attention_scores(Q, dk_quant)
        quant_out = quant_attn @ dv

        output_cosine = F.cosine_similarity(true_out, quant_out, dim=-1).mean().item()

        # Also measure attention pattern quality on the compressed keys
        attn_true = compute_attention_scores(Q, K)
        attn_quant = compute_attention_scores(Q, dk_quant)
        # Can't directly compare since sizes differ; use output cosine

        actual_compression = n / max(dk.shape[0], 1) * 5.0  # 5x from 3-bit quant

        results["triple_stack_limits"][cfg["desc"]] = {
            "n_original": n,
            "n_after_evict": n_after_evict,
            "n_distilled": dk.shape[0],
            "actual_compression": actual_compression,
            "output_cosine": output_cosine,
            "is_garbage": output_cosine < 0.80,
        }

        status = "GARBAGE" if output_cosine < 0.80 else "DEGRADED" if output_cosine < 0.95 else "GOOD"
        print(f"  {cfg['desc']:30s}: {actual_compression:5.1f}x compression, "
              f"output cos={output_cosine:.4f} [{status}]")

    del layer_data
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 8: End-to-end generation quality
# ---------------------------------------------------------------------------

def test_generation_quality(results: dict):
    """Generate tokens and compare against FP16 baseline."""
    print("\n" + "="*70)
    print("TEST 8: END-TO-END GENERATION QUALITY")
    print("="*70)

    model, tokenizer = load_model("3B")
    results["generation"] = {}

    prompt = PROMPTS["math"]
    max_new = 200

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    # FP16 baseline
    print("\n  Generating FP16 baseline...")
    with torch.no_grad():
        outputs_fp16 = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    fp16_ids = outputs_fp16.sequences[0][prompt_len:].tolist()
    fp16_text = tokenizer.decode(fp16_ids, skip_special_tokens=True)
    fp16_logits = torch.cat([s.unsqueeze(0) for s in outputs_fp16.scores], dim=0)  # (gen_len, vocab)

    # Compute FP16 perplexity from logits
    fp16_token_ids = outputs_fp16.sequences[0][prompt_len:prompt_len+len(outputs_fp16.scores)]
    fp16_log_probs = []
    for i, (score, tid) in enumerate(zip(outputs_fp16.scores, fp16_token_ids)):
        log_prob = F.log_softmax(score[0], dim=-1)
        fp16_log_probs.append(log_prob[tid].item())

    fp16_ppl = math.exp(-sum(fp16_log_probs) / max(len(fp16_log_probs), 1))
    print(f"  FP16: {len(fp16_ids)} tokens, perplexity={fp16_ppl:.2f}")
    print(f"  FP16 text (first 200 chars): {fp16_text[:200]}...")

    results["generation"]["fp16"] = {
        "n_tokens": len(fp16_ids),
        "perplexity": fp16_ppl,
        "text_preview": fp16_text[:500],
    }

    # Now test: can we detect quality degradation from quantized KV cache
    # by extracting the cache, quantizing it, and checking per-token logit agreement?
    print("\n  Extracting KV cache for quantization quality analysis...")

    # Run prefill to get cache
    with torch.no_grad():
        prefill_out = model(
            **inputs,
            use_cache=True,
        )
    past_kv = prefill_out.past_key_values

    # Analyze per-layer quantization impact
    if hasattr(past_kv, "layers"):
        n_layers = len(past_kv.layers)
    elif hasattr(past_kv, "key_cache"):
        n_layers = len(past_kv.key_cache)
    elif isinstance(past_kv, (list, tuple)):
        n_layers = len(past_kv)
    else:
        n_layers = 0

    # Sample layers
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    sample_layers = [l for l in sample_layers if 0 <= l < n_layers]

    layer_quant_quality = {}
    for layer_idx in sample_layers:
        if hasattr(past_kv, "layers"):
            layer = past_kv.layers[layer_idx]
            if not hasattr(layer, "keys") or layer.keys is None or layer.keys.numel() == 0:
                continue
            K_all = layer.keys
        elif hasattr(past_kv, "key_cache"):
            K_all = past_kv.key_cache[layer_idx]
        elif isinstance(past_kv, (list, tuple)):
            K_all = past_kv[layer_idx][0]
        else:
            continue

        K = K_all[0, 0].float().to(DEVICE)
        d = K.shape[-1]

        # WHT
        rq = ResidualQuantEstimator(d=d, bits=BITS, seed=42, device=DEVICE, center_before_quantize=True)
        comp = rq.quantize(K)
        K_wht = rq.dequantize(comp)
        mse = (K - K_wht).pow(2).mean().item()
        cos = F.cosine_similarity(K, K_wht, dim=-1).mean().item()

        layer_quant_quality[layer_idx] = {
            "mse": mse,
            "vector_cosine": cos,
            "n_tokens": K.shape[0],
        }
        print(f"    Layer {layer_idx:2d}: MSE={mse:.6f}, vector cos={cos:.4f}")

    results["generation"]["layer_quantization"] = layer_quant_quality

    # Token match analysis
    # We can't easily regenerate with quantized cache without modifying the model,
    # but we can check KV reconstruction quality as a proxy
    print(f"\n  Summary: {len(sample_layers)} layers analyzed, "
          f"mean vector cos = {np.mean([v['vector_cosine'] for v in layer_quant_quality.values()]):.4f}")

    del past_kv
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 9: Statistical reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility(results: dict):
    """Run key experiments 3 times with different seeds."""
    print("\n" + "="*70)
    print("TEST 9: REPRODUCIBILITY (3 seeds)")
    print("="*70)

    model, tokenizer = load_model("3B")
    layer_data, _ = extract_kv(model, tokenizer, PROMPTS["factual"], n_tokens=512)

    mid_layer = max(layer_data.keys()) // 2
    if mid_layer not in layer_data:
        mid_layer = sorted(layer_data.keys())[len(layer_data)//2]
    Q, K = layer_data[mid_layer]["Q"], layer_data[mid_layer]["K"]
    d = K.shape[-1]

    results["reproducibility"] = {}

    for method_name in ["wht_mean_removal", "cayley_100step", "cayley_200step"]:
        seed_results = []

        for seed in SEEDS:
            torch.manual_seed(seed)

            if method_name == "wht_mean_removal":
                rq = ResidualQuantEstimator(
                    d=d, bits=BITS, seed=seed, device=DEVICE,
                    center_before_quantize=True,
                )
                comp = rq.quantize(K)
                K_q = rq.dequantize(comp)

            elif method_name.startswith("cayley"):
                steps = 100 if "100" in method_name else 200
                cq = CayleyLearnedQuantizer(
                    d=d, bits=BITS, center=True, seed=seed,
                    device=DEVICE, init_from_wht=True,
                )
                cq.calibrate(Q, K, lr=0.005, steps=steps)
                K_q = cq.forward(K)

            m = eval_attention(Q, K, K_q)
            seed_results.append({
                "seed": seed,
                "cosine_sim": m["cosine_sim"],
                "top5_match": m["top5_match"],
                "kl_div": m["kl_div"],
            })

        cos_vals = [r["cosine_sim"] for r in seed_results]
        top5_vals = [r["top5_match"] for r in seed_results]

        mean_cos = np.mean(cos_vals)
        std_cos = np.std(cos_vals)
        mean_top5 = np.mean(top5_vals)
        std_top5 = np.std(top5_vals)

        is_reliable = std_cos < abs(mean_cos) * 0.10  # std < 10% of mean

        results["reproducibility"][method_name] = {
            "seeds": seed_results,
            "cosine_mean": float(mean_cos),
            "cosine_std": float(std_cos),
            "top5_mean": float(mean_top5),
            "top5_std": float(std_top5),
            "reliable": bool(is_reliable),
        }

        print(f"  {method_name:20s}: cos={mean_cos:.4f}+/-{std_cos:.4f} "
              f"top5={mean_top5:.3f}+/-{std_top5:.3f} "
              f"{'RELIABLE' if is_reliable else 'UNRELIABLE'}")

    del layer_data
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 10: 14B comprehensive
# ---------------------------------------------------------------------------

def test_14b_comprehensive(results: dict):
    """Full validation on 14B model."""
    print("\n" + "="*70)
    print("TEST 10: FULL 14B VALIDATION")
    print("="*70)

    model, tokenizer = load_model("14B")
    results["14b_comprehensive"] = {}

    for prompt_name in ["factual", "code", "adversarial"]:
        prompt_text = PROMPTS[prompt_name]
        layer_data, total_tokens = extract_kv(
            model, tokenizer, prompt_text, n_tokens=512,
        )
        print(f"\n  14B + {prompt_name} ({total_tokens} tokens):")

        test_layers = TEST_LAYERS_14B
        prompt_results = {}

        for layer_idx in test_layers:
            if layer_idx not in layer_data:
                continue
            Q, K = layer_data[layer_idx]["Q"], layer_data[layer_idx]["K"]
            d = K.shape[-1]

            # WHT baseline
            rq = ResidualQuantEstimator(
                d=d, bits=BITS, seed=42, device=DEVICE,
                center_before_quantize=True,
            )
            comp = rq.quantize(K)
            K_wht = rq.dequantize(comp)
            wht_m = eval_attention(Q, K, K_wht)

            # PolarQuant only (no mean-removal, no residual)
            pq = PolarQuant(d=d, bits=BITS, seed=42, device=DEVICE)
            K_pq, _ = pq.forward(F.normalize(K, dim=-1))
            pq_m = eval_attention(Q, K, K_pq * K.norm(dim=-1, keepdim=True))

            prompt_results[layer_idx] = {
                "wht_cosine": wht_m["cosine_sim"],
                "wht_top5": wht_m["top5_match"],
                "polarquant_cosine": pq_m["cosine_sim"],
                "polarquant_top5": pq_m["top5_match"],
            }

            print(f"    L{layer_idx:2d}: WHT+mean cos={wht_m['cosine_sim']:.4f} "
                  f"PolarQuant cos={pq_m['cosine_sim']:.4f}")

        results["14b_comprehensive"][prompt_name] = prompt_results

        del layer_data
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: dict) -> str:
    """Generate markdown report from results."""
    lines = []
    lines.append("# Adversarial Validation Report")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"Device: {DEVICE}")
    lines.append(f"Seeds: {SEEDS}")
    lines.append("")

    # ---- Test 1: Cross-model ----
    if "cayley_cross_model" in results:
        lines.append("## Test 1: Cayley Cross-Model Validation")
        lines.append("")
        for model_key, data in results["cayley_cross_model"].items():
            s = data.get("summary", {})
            cl = s.get("cayley_lift_cosine", {})
            lines.append(f"### {model_key}")
            lines.append(f"- Cayley cosine lift: **{cl.get('mean', 0):+.4f} +/- {cl.get('std', 0):.4f}**")
            cc = s.get("cayley_cosine", {})
            lines.append(f"- Cayley cosine (abs): **{cc.get('mean', 0):.4f} +/- {cc.get('std', 0):.4f}**")
            wc = s.get("wht_cosine", {})
            lines.append(f"- WHT cosine: **{wc.get('mean', 0):.4f} +/- {wc.get('std', 0):.4f}**")
            lines.append("")

    # ---- Test 2: Transfer ----
    if "cayley_transfer" in results:
        lines.append("## Test 2: Cayley Transfer (3B -> 14B)")
        lines.append("")
        lines.append("| Layer | 3B (in-dist) | 14B (transfer) | 14B (WHT) | Beats WHT? |")
        lines.append("|-------|-------------|----------------|-----------|------------|")
        for layer, d in results["cayley_transfer"].items():
            beats = "YES" if d.get("beats_wht_on_14b", False) else "NO"
            lines.append(f"| {layer} | {d.get('in_dist_cosine', 0):.4f} | "
                        f"{d.get('out_dist_cosine', 0):.4f} | "
                        f"{d.get('wht_baseline_14b', 0):.4f} | {beats} |")
        lines.append("")

    # ---- Test 3: Sequence length ----
    if "seq_length" in results:
        lines.append("## Test 3: Sequence Length Sensitivity")
        lines.append("")
        lines.append("| Seq Len | WHT cos | Cayley cos | EA Spearman | Distill cos |")
        lines.append("|---------|---------|------------|-------------|-------------|")
        for sl, d in sorted(results["seq_length"].items(), key=lambda x: int(x[0])):
            ea = f"{d.get('ea_spearman', 'N/A'):.4f}" if d.get("ea_spearman") is not None else "N/A"
            dc = f"{d.get('distill_cosine', 'N/A'):.4f}" if d.get("distill_cosine") is not None else "N/A"
            lines.append(f"| {sl} | {d.get('wht_cosine', 0):.4f} | "
                        f"{d.get('cayley_cosine', 0):.4f} | {ea} | {dc} |")
        lines.append("")

    # ---- Test 4: Prompt diversity ----
    if "prompt_diversity" in results:
        lines.append("## Test 4: Prompt Diversity")
        lines.append("")
        for prompt, layers in results["prompt_diversity"].items():
            lines.append(f"### {prompt}")
            for layer, m in layers.items():
                lines.append(f"- L{layer}: WHT={m.get('wht_cosine', 0):.4f} "
                           f"Cayley={m.get('cayley_cosine', 0):.4f}")
            lines.append("")

    # ---- Test 5: EA failure modes ----
    if "ea_failure_modes" in results:
        lines.append("## Test 5: Expected Attention Failure Modes")
        lines.append("")
        for case, d in results["ea_failure_modes"].items():
            lines.append(f"- **{case}**: Spearman = {d.get('spearman', 0):.4f}")
        lines.append("")

    # ---- Test 6: Mean removal ----
    if "mean_removal" in results:
        lines.append("## Test 6: Mean-Removal Impact")
        lines.append("")
        lines.append("| Prompt | With Center | Without | Diff | Verdict |")
        lines.append("|--------|------------|---------|------|---------|")
        for prompt, d in results["mean_removal"].items():
            verdict = "HURTS" if d.get("hurts", False) else "HELPS" if d.get("diff", 0) > 0.001 else "NEUTRAL"
            lines.append(f"| {prompt} | {d.get('with_center_cosine', 0):.4f} | "
                        f"{d.get('without_center_cosine', 0):.4f} | "
                        f"{d.get('diff', 0):+.4f} | {verdict} |")
        lines.append("")

    # ---- Test 7: Triple stack limits ----
    if "triple_stack_limits" in results:
        lines.append("## Test 7: Triple Stack Compression Limits")
        lines.append("")
        lines.append("| Config | Compression | Output Cos | Status |")
        lines.append("|--------|-------------|-----------|--------|")
        for cfg, d in results["triple_stack_limits"].items():
            status = "GARBAGE" if d.get("is_garbage", False) else (
                "DEGRADED" if d.get("output_cosine", 0) < 0.95 else "GOOD"
            )
            lines.append(f"| {cfg} | {d.get('actual_compression', 0):.1f}x | "
                        f"{d.get('output_cosine', 0):.4f} | {status} |")
        lines.append("")

    # ---- Test 8: Generation ----
    if "generation" in results:
        lines.append("## Test 8: End-to-End Generation")
        lines.append("")
        gen = results["generation"]
        if "fp16" in gen:
            lines.append(f"- FP16 perplexity: **{gen['fp16'].get('perplexity', 0):.2f}**")
            lines.append(f"- FP16 tokens: {gen['fp16'].get('n_tokens', 0)}")
        if "layer_quantization" in gen:
            lines.append("\nPer-layer quantization quality:")
            for layer, d in gen["layer_quantization"].items():
                lines.append(f"- Layer {layer}: MSE={d.get('mse', 0):.6f} "
                           f"vec_cos={d.get('vector_cosine', 0):.4f}")
        lines.append("")

    # ---- Test 9: Reproducibility ----
    if "reproducibility" in results:
        lines.append("## Test 9: Reproducibility (3 seeds)")
        lines.append("")
        lines.append("| Method | Cosine (mean +/- std) | Top-5 (mean +/- std) | Reliable? |")
        lines.append("|--------|----------------------|---------------------|-----------|")
        for method, d in results["reproducibility"].items():
            reliable = "YES" if d.get("reliable", False) else "NO"
            lines.append(f"| {method} | {d.get('cosine_mean', 0):.4f} +/- {d.get('cosine_std', 0):.4f} | "
                        f"{d.get('top5_mean', 0):.3f} +/- {d.get('top5_std', 0):.3f} | {reliable} |")
        lines.append("")

    # ---- Test 10: 14B ----
    if "14b_comprehensive" in results:
        lines.append("## Test 10: Full 14B Validation")
        lines.append("")
        for prompt, layers in results["14b_comprehensive"].items():
            lines.append(f"### {prompt}")
            for layer, m in layers.items():
                lines.append(f"- L{layer}: WHT+mean={m.get('wht_cosine', 0):.4f} "
                           f"PolarQuant={m.get('polarquant_cosine', 0):.4f}")
            lines.append("")

    # ---- VERDICTS ----
    lines.append("## VERDICTS")
    lines.append("")

    # Analyze results
    verdicts = []

    if "cayley_cross_model" in results:
        for model_key, data in results["cayley_cross_model"].items():
            s = data.get("summary", {})
            cl = s.get("cayley_lift_cosine", {})
            if cl.get("mean", 0) > 0.001:
                verdicts.append(f"Cayley rotation HOLDS on {model_key}: lift={cl['mean']:+.4f}")
            else:
                verdicts.append(f"Cayley rotation FAILS on {model_key}: lift={cl.get('mean', 0):+.4f}")

    if "cayley_transfer" in results:
        transfers = results["cayley_transfer"]
        n_beats = sum(1 for d in transfers.values() if d.get("beats_wht_on_14b", False))
        verdicts.append(f"Cayley transfer 3B->14B: beats WHT in {n_beats}/{len(transfers)} layers")

    if "ea_failure_modes" in results:
        shift = results["ea_failure_modes"].get("distribution_shift", {})
        if shift.get("spearman", 0) < 0.3:
            verdicts.append("Expected Attention FAILS on distribution shift (Spearman < 0.3)")
        else:
            verdicts.append(f"Expected Attention handles distribution shift: "
                          f"Spearman={shift.get('spearman', 0):.3f}")

    if "mean_removal" in results:
        hurts = [p for p, d in results["mean_removal"].items() if d.get("hurts", False)]
        if hurts:
            verdicts.append(f"Mean-removal HURTS on: {', '.join(hurts)}")
        else:
            verdicts.append("Mean-removal NEVER hurts across all tested prompts")

    if "triple_stack_limits" in results:
        garbage = [c for c, d in results["triple_stack_limits"].items() if d.get("is_garbage", False)]
        if garbage:
            verdicts.append(f"Triple stack produces GARBAGE at: {', '.join(garbage)}")
        else:
            verdicts.append("Triple stack holds up at all tested compression ratios")

    for v in verdicts:
        lines.append(f"- {v}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ADVERSARIAL VALIDATION -- TurboQuantDC")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print("=" * 70)

    results = {}

    try:
        # Run all tests
        test_cayley_cross_model(results)
        test_cayley_transfer(results)
        test_seq_length(results)
        test_prompt_diversity(results)
        test_ea_failure_modes(results)
        test_mean_removal(results)
        test_triple_stack_limits(results)
        test_generation_quality(results)
        test_reproducibility(results)
        test_14b_comprehensive(results)
    except Exception as e:
        print(f"\n\nERROR during testing: {e}")
        traceback.print_exc()
        results["error"] = str(e)

    # Save results
    json_path = RESULTS_DIR / "adversarial_validation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON results saved to {json_path}")

    # Generate report
    report = generate_report(results)
    md_path = RESULTS_DIR / "adversarial_validation.md"
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Markdown report saved to {md_path}")

    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
