#!/usr/bin/env python3
"""Unified adaptive generation cache benchmark.

Tests AdaptiveGenerationCache against:
  - FP16 baseline
  - Uniform 3-bit GenerationCache (production)
  - Attention-gated only (ultra_compress)
  - Adaptive-bits only (adaptive_bits)

Measures:
  - Effective bits per coordinate
  - Compression ratio vs FP16
  - Generation quality (200 tokens, greedy)
  - Cosine similarity of KV reconstructions
  - Top-5 attention match

Run:
    python benchmarks/unified_adaptive_benchmark.py
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Allow running from repo root
REPO_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("UNIFIED_MODEL", "Qwen/Qwen2.5-3B-Instruct")
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MAX_NEW_TOKENS = 200

GENERATION_PROMPT = (
    "Explain the mathematical foundations of KV cache compression in "
    "transformer-based language models. Start from the attention mechanism "
    "and derive why quantization of key vectors requires careful treatment "
    "of inner product preservation:"
)

GENERATION_PROMPTS = [
    "What is the capital of Australia? Answer in one sentence:",
    "A neural network is",
    "Write a Python function to calculate the factorial of a number:",
]

# Long context for attention analysis
CONTEXT_PROMPT = """You are an expert research assistant analyzing quantum computing papers.

Note 1: Quantum Error Correction
Quantum error correction (QEC) is essential for building fault-tolerant quantum computers. The surface code achieves error thresholds of approximately 1%. Recent experimental demonstrations by Google's Sycamore processor have shown logical error rates below threshold for distance-3 and distance-5 surface codes.

Note 2: Quantum Advantage
The quantum approximate optimization algorithm (QAOA) is designed for combinatorial optimization problems. Despite progress, definitive quantum advantage for optimization remains elusive. Classical algorithms continue to compete effectively.

Note 3: Quantum Machine Learning
Quantum machine learning has seen explosive growth, with variational quantum eigensolvers being the most studied. The barren plateau phenomenon poses a fundamental challenge: gradients vanish exponentially with system size.

Note 4: Superconducting Qubits
Superconducting qubits based on Josephson junctions dominate the current landscape. Transmon qubits achieve coherence times exceeding 100 microseconds. Key challenges include improving gate fidelities and reducing crosstalk.

Note 5: Trapped Ion Computing
Trapped ion quantum computers offer higher two-qubit gate fidelities exceeding 99.9% and longer coherence times. Challenges include scaling beyond several dozen qubits in a single trap.

Based on all these research notes, explain the current state of quantum computing and its most promising near-term applications:"""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    """Load model with 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME} (4-bit quantized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def extract_attention_and_kv(model, tokenizer, prompt: str):
    """Run forward pass, extract attention weights and KV states."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=True,
        )

    attentions = outputs.attentions  # tuple of (batch, heads, q, kv) per layer
    past_kv = outputs.past_key_values  # DynamicCache

    # Extract KV tensors -- transformers >= 5.x uses .layers[i].keys/.values
    kv_pairs = []
    if hasattr(past_kv, 'layers'):
        for layer in past_kv.layers:
            kv_pairs.append((layer.keys, layer.values))
    else:
        # Fallback for older transformers
        for layer_idx in range(len(past_kv)):
            keys, values = past_kv[layer_idx]
            kv_pairs.append((keys, values))

    return attentions, kv_pairs, inputs


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    original_keys: torch.Tensor,
    reconstructed_keys: torch.Tensor,
    original_values: torch.Tensor,
    reconstructed_values: torch.Tensor,
    attention_weights: torch.Tensor,
    query_window: int = 64,
) -> Dict[str, float]:
    """Compute quality metrics comparing reconstructed vs original."""
    # Flatten to 2D for metrics
    ok = original_keys.float().reshape(-1, original_keys.shape[-1])
    rk = reconstructed_keys.float().reshape(-1, reconstructed_keys.shape[-1])

    # Cosine similarity
    cos_sim = F.cosine_similarity(ok, rk, dim=-1).mean().item()

    # MSE
    mse = F.mse_loss(rk, ok).item()

    # Top-5 attention match
    # Use last query_window queries
    attn = attention_weights.float()
    if attn.dim() == 4:
        attn = attn.mean(dim=0)  # average across batch
    n_q = min(query_window, attn.shape[1])
    attn_slice = attn[:, -n_q:, :]  # (heads, n_q, kv_len)

    # Compute attention with original keys
    q_len = original_keys.shape[2]
    query_states = original_keys[:, :, -n_q:, :]  # Use last queries

    # Attention scores: q @ k^T / sqrt(d)
    d = original_keys.shape[-1]
    scale = 1.0 / math.sqrt(d)

    orig_scores = torch.matmul(
        query_states.float(), original_keys.float().transpose(-2, -1)
    ) * scale
    recon_scores = torch.matmul(
        query_states.float(), reconstructed_keys.float().transpose(-2, -1)
    ) * scale

    # Apply causal mask
    kv_len = original_keys.shape[2]
    causal_mask = torch.triu(
        torch.ones(n_q, kv_len, device=orig_scores.device), diagonal=kv_len - n_q + 1
    ).bool()
    orig_scores = orig_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    recon_scores = recon_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    orig_probs = F.softmax(orig_scores, dim=-1)
    recon_probs = F.softmax(recon_scores, dim=-1)

    # Top-5 match
    orig_top5 = orig_probs.topk(5, dim=-1).indices
    recon_top5 = recon_probs.topk(5, dim=-1).indices

    matches = 0
    total = 0
    for i in range(orig_top5.shape[-2]):
        for b in range(orig_top5.shape[0]):
            for h in range(orig_top5.shape[1]):
                orig_set = set(orig_top5[b, h, i].tolist())
                recon_set = set(recon_top5[b, h, i].tolist())
                matches += len(orig_set & recon_set)
                total += 5
    top5_match = matches / max(total, 1)

    # Value reconstruction quality
    ov = original_values.float().reshape(-1, original_values.shape[-1])
    rv = reconstructed_values.float().reshape(-1, reconstructed_values.shape[-1])
    val_cos_sim = F.cosine_similarity(ov, rv, dim=-1).mean().item()

    return {
        "key_cosine_sim": cos_sim,
        "key_mse": mse,
        "top5_match": top5_match,
        "val_cosine_sim": val_cos_sim,
    }


# ---------------------------------------------------------------------------
# Generation comparison
# ---------------------------------------------------------------------------

def generate_with_cache(model, tokenizer, prompt, cache=None, max_new_tokens=MAX_NEW_TOKENS):
    """Generate text with a given cache."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            past_key_values=cache,
            use_cache=True,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated


def is_coherent(text: str) -> bool:
    """Basic coherence check."""
    if len(text.strip()) < 10:
        return False
    words = text.split()
    if len(words) < 3:
        return False
    # Check for excessive repetition
    if len(set(words)) < len(words) * 0.15:
        return False
    return True


# ---------------------------------------------------------------------------
# KV cache simulation
# ---------------------------------------------------------------------------

def simulate_adaptive_cache(
    kv_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    attentions: tuple,
    tier_thresholds: List[float],
    tier_bits: List[int],
    fp16_buffer_size: int = 128,
    boundary_layers: int = 2,
    ema_decay: float = 0.9,
) -> Dict[str, Any]:
    """Simulate AdaptiveGenerationCache on extracted KV pairs.

    Since we can't hook into the model's attention loop directly for
    per-step updates, we simulate the adaptive allocation using the
    full attention matrix from a single forward pass.

    Returns metrics and reconstructed KV pairs.
    """
    from turboquantdc.adaptive_generation_cache import AdaptiveGenerationCache

    num_layers = len(kv_pairs)
    cache = AdaptiveGenerationCache(
        hot_window=64,
        fp16_buffer_size=fp16_buffer_size,
        tier_thresholds=tier_thresholds,
        tier_bits=tier_bits,
        boundary_layers=boundary_layers,
        ema_decay=ema_decay,
        reclassify_interval=16,
        num_layers=num_layers,
        seed=SEED,
        use_residual_quant=True,
    )

    # Feed KV pairs into the cache and update importance
    all_recon_keys = []
    all_recon_vals = []

    for layer_idx, (keys, values) in enumerate(kv_pairs):
        # Update importance from attention weights
        if layer_idx < len(attentions) and attentions[layer_idx] is not None:
            cache.update_importance(attentions[layer_idx], layer_idx)

        # Store KV
        recon_k, recon_v = cache.update(keys, values, layer_idx)
        all_recon_keys.append(recon_k)
        all_recon_vals.append(recon_v)

    return {
        "cache": cache,
        "recon_keys": all_recon_keys,
        "recon_vals": all_recon_vals,
        "effective_bits": cache.effective_bits(),
        "compression_ratio": cache.compression_ratio(),
        "tier_summary": cache.tier_summary(),
    }


def simulate_uniform_cache(
    kv_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    key_bits: int = 3,
    val_bits: int = 2,
    fp16_window: int = 64,
    boundary_layers: int = 2,
) -> Dict[str, Any]:
    """Simulate uniform compression using GenerationCache."""
    from turboquantdc.generation_cache import GenerationCache

    num_layers = len(kv_pairs)
    cache = GenerationCache(
        key_bits=key_bits,
        val_bits=val_bits,
        fp16_window=fp16_window,
        anchor_strategy="boundary",
        num_layers=num_layers,
        seed=SEED,
        use_residual_quant=True,
        use_norm_correction=True,
    )

    all_recon_keys = []
    all_recon_vals = []

    for layer_idx, (keys, values) in enumerate(kv_pairs):
        recon_k, recon_v = cache.update(keys, values, layer_idx)
        all_recon_keys.append(recon_k)
        all_recon_vals.append(recon_v)

    savings = cache.memory_savings()
    return {
        "cache": cache,
        "recon_keys": all_recon_keys,
        "recon_vals": all_recon_vals,
        "compression_ratio": savings["overall_compression_ratio"],
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    """Run the full unified adaptive benchmark."""
    print("=" * 70)
    print("UNIFIED ADAPTIVE GENERATION CACHE BENCHMARK")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    model, tokenizer = load_model_and_tokenizer()
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"Layers: {num_layers}, KV heads: {num_heads}, Head dim: {head_dim}")
    print()

    # ---- Phase 1: Extract attention patterns and KV states ----
    print("Phase 1: Extracting attention patterns and KV states...")
    t0 = time.time()
    attentions, kv_pairs, inputs = extract_attention_and_kv(model, tokenizer, CONTEXT_PROMPT)
    seq_len = kv_pairs[0][0].shape[2]
    print(f"  Context length: {seq_len} tokens ({time.time() - t0:.1f}s)")
    print()

    # ---- Phase 2: Simulate different compression strategies ----
    print("Phase 2: Simulating compression strategies...")

    results = {}

    # 2a. Uniform 3-bit ResidualQuant (production baseline)
    print("  [1/5] Uniform 3-bit ResidualQuant (production)...")
    t0 = time.time()
    uniform_3bit = simulate_uniform_cache(kv_pairs, key_bits=3, val_bits=2, fp16_window=64)
    print(f"    Compression: {uniform_3bit['compression_ratio']:.1f}x ({time.time() - t0:.1f}s)")
    results["uniform_3bit"] = uniform_3bit

    # 2b. Uniform 1-bit (lower bound)
    print("  [2/5] Uniform 1-bit (lower bound)...")
    t0 = time.time()
    uniform_1bit = simulate_uniform_cache(kv_pairs, key_bits=1, val_bits=1, fp16_window=64)
    print(f"    Compression: {uniform_1bit['compression_ratio']:.1f}x ({time.time() - t0:.1f}s)")
    results["uniform_1bit"] = uniform_1bit

    # 2c. Adaptive (our unified system) -- target config
    # Use fp16_buffer = min(64, seq_len//4) so compression actually kicks in
    adaptive_buf = min(64, max(seq_len // 4, 16))
    print(f"  [3/5] Adaptive unified (T0=FP16, T1=4b, T2=3b, T3=1b, buf={adaptive_buf})...")
    t0 = time.time()
    adaptive_target = simulate_adaptive_cache(
        kv_pairs, attentions,
        tier_thresholds=[0.05, 0.20, 0.80],
        tier_bits=[16, 4, 3, 1],
        fp16_buffer_size=adaptive_buf,
        boundary_layers=2,
    )
    print(f"    Effective bits: {adaptive_target['effective_bits']:.2f}, "
          f"Compression: {adaptive_target['compression_ratio']:.1f}x ({time.time() - t0:.1f}s)")
    results["adaptive_target"] = adaptive_target

    # 2d. Adaptive aggressive (push for max compression)
    print(f"  [4/5] Adaptive aggressive (T0=4b, T1=2b, T2=1b, buf={adaptive_buf})...")
    t0 = time.time()
    adaptive_aggressive = simulate_adaptive_cache(
        kv_pairs, attentions,
        tier_thresholds=[0.05, 0.20, 0.80],
        tier_bits=[4, 2, 1, 1],
        fp16_buffer_size=adaptive_buf,
        boundary_layers=2,
    )
    print(f"    Effective bits: {adaptive_aggressive['effective_bits']:.2f}, "
          f"Compression: {adaptive_aggressive['compression_ratio']:.1f}x ({time.time() - t0:.1f}s)")
    results["adaptive_aggressive"] = adaptive_aggressive

    # 2e. Adaptive conservative (maximize quality)
    print(f"  [5/5] Adaptive conservative (T0=FP16, T1=4b, T2=3b, buf={adaptive_buf})...")
    t0 = time.time()
    adaptive_conservative = simulate_adaptive_cache(
        kv_pairs, attentions,
        tier_thresholds=[0.10, 0.30, 0.70],
        tier_bits=[16, 4, 3, 2],
        fp16_buffer_size=adaptive_buf,
        boundary_layers=2,
    )
    print(f"    Effective bits: {adaptive_conservative['effective_bits']:.2f}, "
          f"Compression: {adaptive_conservative['compression_ratio']:.1f}x ({time.time() - t0:.1f}s)")
    results["adaptive_conservative"] = adaptive_conservative

    # ---- Phase 3: Quality metrics ----
    print()
    print("Phase 3: Computing quality metrics...")

    quality = {}
    for name, result in results.items():
        metrics_per_layer = []
        for layer_idx in range(num_layers):
            orig_k, orig_v = kv_pairs[layer_idx]
            recon_k = result["recon_keys"][layer_idx]
            recon_v = result["recon_vals"][layer_idx]

            # Ensure shapes match
            min_seq = min(orig_k.shape[2], recon_k.shape[2])
            orig_k = orig_k[:, :, :min_seq, :]
            recon_k = recon_k[:, :, :min_seq, :]
            orig_v = orig_v[:, :, :min_seq, :]
            recon_v = recon_v[:, :, :min_seq, :]

            if layer_idx < len(attentions) and attentions[layer_idx] is not None:
                m = compute_metrics(orig_k, recon_k, orig_v, recon_v, attentions[layer_idx])
                metrics_per_layer.append(m)

        if metrics_per_layer:
            avg_metrics = {
                k: sum(m[k] for m in metrics_per_layer) / len(metrics_per_layer)
                for k in metrics_per_layer[0]
            }
        else:
            avg_metrics = {"key_cosine_sim": 0, "key_mse": 0, "top5_match": 0, "val_cosine_sim": 0}

        quality[name] = avg_metrics
        print(f"  {name}: cos={avg_metrics['key_cosine_sim']:.4f}, "
              f"top5={avg_metrics['top5_match']:.1%}, "
              f"val_cos={avg_metrics['val_cosine_sim']:.4f}")

    # ---- Phase 4: Generation quality ----
    print()
    print("Phase 4: Generation quality comparison (200 tokens)...")

    generation_results = {}

    # FP16 baseline
    print("  Generating FP16 baseline...")
    fp16_texts = []
    for prompt in GENERATION_PROMPTS:
        text = generate_with_cache(model, tokenizer, prompt, cache=None, max_new_tokens=50)
        fp16_texts.append(text)
        coherent = is_coherent(text)
        print(f"    {'PASS' if coherent else 'FAIL'}: {text[:80]}...")
    generation_results["fp16"] = fp16_texts

    # Uniform 3-bit GenerationCache
    print("  Generating with uniform 3-bit...")
    from turboquantdc.generation_cache import GenerationCache
    gen_texts_3bit = []
    for prompt in GENERATION_PROMPTS:
        cache = GenerationCache(
            key_bits=3, val_bits=2, fp16_window=64,
            anchor_strategy="boundary", num_layers=num_layers,
            use_residual_quant=True, use_norm_correction=True,
        )
        text = generate_with_cache(model, tokenizer, prompt, cache=cache, max_new_tokens=50)
        gen_texts_3bit.append(text)
        coherent = is_coherent(text)
        print(f"    {'PASS' if coherent else 'FAIL'}: {text[:80]}...")
    generation_results["uniform_3bit"] = gen_texts_3bit

    # Adaptive unified cache (generation)
    print("  Generating with adaptive unified...")
    from turboquantdc.adaptive_generation_cache import AdaptiveGenerationCache
    gen_texts_adaptive = []
    for prompt in GENERATION_PROMPTS:
        cache = AdaptiveGenerationCache(
            hot_window=64, fp16_buffer_size=128,
            tier_thresholds=[0.05, 0.20, 0.80],
            tier_bits=[16, 4, 3, 1],
            boundary_layers=2, num_layers=num_layers,
            use_residual_quant=True,
        )
        text = generate_with_cache(model, tokenizer, prompt, cache=cache, max_new_tokens=50)
        gen_texts_adaptive.append(text)
        coherent = is_coherent(text)
        print(f"    {'PASS' if coherent else 'FAIL'}: {text[:80]}...")
    generation_results["adaptive_unified"] = gen_texts_adaptive

    # Long generation (200 tokens)
    print()
    print("  Long generation (200 tokens) comparison...")
    long_gen_results = {}

    for method_name, cache_factory in [
        ("fp16", lambda: None),
        ("uniform_3bit", lambda: GenerationCache(
            key_bits=3, val_bits=2, fp16_window=64,
            anchor_strategy="boundary", num_layers=num_layers,
            use_residual_quant=True, use_norm_correction=True,
        )),
        ("adaptive_unified", lambda: AdaptiveGenerationCache(
            hot_window=64, fp16_buffer_size=128,
            tier_thresholds=[0.05, 0.20, 0.80], tier_bits=[16, 4, 3, 1],
            boundary_layers=2, num_layers=num_layers,
            use_residual_quant=True,
        )),
        ("adaptive_aggressive", lambda: AdaptiveGenerationCache(
            hot_window=64, fp16_buffer_size=128,
            tier_thresholds=[0.05, 0.20, 0.80], tier_bits=[4, 2, 1, 1],
            boundary_layers=2, num_layers=num_layers,
            use_residual_quant=True,
        )),
    ]:
        print(f"    {method_name}...")
        cache = cache_factory()
        t0 = time.time()
        text = generate_with_cache(model, tokenizer, GENERATION_PROMPT, cache=cache, max_new_tokens=200)
        elapsed = time.time() - t0
        coherent = is_coherent(text)
        long_gen_results[method_name] = {
            "text": text,
            "coherent": coherent,
            "time": elapsed,
            "tokens_per_sec": 200 / elapsed,
        }
        print(f"      {'PASS' if coherent else 'FAIL'} ({elapsed:.1f}s, {200/elapsed:.1f} tok/s)")
        print(f"      Preview: {text[:120]}...")

    # ---- Phase 5: Token match rates ----
    print()
    print("Phase 5: Token match rates vs FP16...")
    fp16_long = long_gen_results["fp16"]["text"]
    fp16_tokens = tokenizer.encode(fp16_long)

    for method in ["uniform_3bit", "adaptive_unified", "adaptive_aggressive"]:
        method_tokens = tokenizer.encode(long_gen_results[method]["text"])
        min_len = min(len(fp16_tokens), len(method_tokens))
        matches = sum(1 for a, b in zip(fp16_tokens[:min_len], method_tokens[:min_len]) if a == b)
        match_rate = matches / max(min_len, 1)
        print(f"  {method}: {match_rate:.1%} token match ({matches}/{min_len})")

    # ---- Write results ----
    print()
    print("Writing results...")
    write_results(
        results, quality, generation_results, long_gen_results,
        num_layers, num_heads, head_dim, seq_len,
        fp16_tokens, tokenizer,
    )

    print()
    print("Done! Results saved to benchmarks/results/unified_adaptive_results.md")


def write_results(
    results, quality, generation_results, long_gen_results,
    num_layers, num_heads, head_dim, seq_len,
    fp16_tokens, tokenizer,
):
    """Write benchmark results to markdown."""
    out_dir = Path(REPO_ROOT) / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "unified_adaptive_results.md"

    lines = []
    lines.append("# Unified Adaptive Generation Cache Results")
    lines.append("")
    lines.append(f"**Model:** {MODEL_NAME}")
    lines.append(f"**Device:** {DEVICE}")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Context:** {seq_len} tokens")
    lines.append(f"**Layers:** {num_layers}, **KV Heads:** {num_heads}, **Head Dim:** {head_dim}")
    lines.append("")

    # Compression comparison
    lines.append("## 1. Compression Comparison")
    lines.append("")
    lines.append("| Method | Eff. Bits | Compression | Key CosSim | Top-5 Match | Val CosSim |")
    lines.append("|--------|-----------|-------------|------------|-------------|------------|")

    method_labels = {
        "uniform_3bit": "Uniform 3-bit (production)",
        "uniform_1bit": "Uniform 1-bit (lower bound)",
        "adaptive_target": "Adaptive unified (T0=FP16,T1=4b,T2=3b,T3=1b)",
        "adaptive_aggressive": "Adaptive aggressive (T0=4b,T1=2b,T2=1b)",
        "adaptive_conservative": "Adaptive conservative (T0=FP16,T1=4b,T2=3b,T3=2b)",
    }

    for name, label in method_labels.items():
        r = results.get(name, {})
        q = quality.get(name, {})

        if name.startswith("adaptive"):
            eff_bits = r.get("effective_bits", 16)
            compression = r.get("compression_ratio", 1.0)
        else:
            compression = r.get("compression_ratio", 1.0)
            if name == "uniform_3bit":
                eff_bits = 3.0
            elif name == "uniform_1bit":
                eff_bits = 1.0
            else:
                eff_bits = 16.0 / max(compression, 0.01)

        lines.append(
            f"| {label} | {eff_bits:.2f} | {compression:.1f}x | "
            f"{q.get('key_cosine_sim', 0):.4f} | {q.get('top5_match', 0):.1%} | "
            f"{q.get('val_cosine_sim', 0):.4f} |"
        )

    lines.append("")

    # Tier distribution for adaptive methods
    lines.append("## 2. Tier Distribution (Adaptive Methods)")
    lines.append("")

    for name in ["adaptive_target", "adaptive_aggressive", "adaptive_conservative"]:
        if name not in results:
            continue
        r = results[name]
        ts = r.get("tier_summary", {})
        lines.append(f"### {method_labels[name]}")
        lines.append("")
        if "tier_counts" in ts:
            lines.append("| Tier | Count | Percentage |")
            lines.append("|------|-------|------------|")
            total = ts.get("total_tokens", 1)
            for tier_name, count in ts["tier_counts"].items():
                pct = count / max(total, 1)
                lines.append(f"| {tier_name} | {count} | {pct:.1%} |")
        lines.append(f"\n**Effective bits:** {ts.get('effective_bits', 'N/A')}")
        lines.append(f"**Compression:** {ts.get('compression_ratio', 'N/A'):.1f}x")
        lines.append("")

    # Generation quality
    lines.append("## 3. Generation Quality (Short: 50 tokens)")
    lines.append("")

    for method_name, method_label in [
        ("fp16", "FP16 Baseline"),
        ("uniform_3bit", "Uniform 3-bit"),
        ("adaptive_unified", "Adaptive Unified"),
    ]:
        texts = generation_results.get(method_name, [])
        lines.append(f"### {method_label}")
        for i, text in enumerate(texts):
            coherent = is_coherent(text)
            status = "PASS" if coherent else "FAIL"
            lines.append(f"- [{status}] Q{i}: `{text[:100]}...`")
        lines.append("")

    # Long generation
    lines.append("## 4. Long Generation (200 tokens)")
    lines.append("")
    lines.append("| Method | Coherent | Tok/s | Token Match vs FP16 |")
    lines.append("|--------|----------|-------|---------------------|")

    fp16_long_text = long_gen_results.get("fp16", {}).get("text", "")
    fp16_long_tokens = tokenizer.encode(fp16_long_text) if fp16_long_text else []

    for method in ["fp16", "uniform_3bit", "adaptive_unified", "adaptive_aggressive"]:
        r = long_gen_results.get(method, {})
        coherent = "Yes" if r.get("coherent", False) else "No"
        tps = r.get("tokens_per_sec", 0)

        if method == "fp16":
            match_str = "100.0%"
        else:
            method_tokens = tokenizer.encode(r.get("text", ""))
            min_len = min(len(fp16_long_tokens), len(method_tokens))
            if min_len > 0:
                matches = sum(1 for a, b in zip(fp16_long_tokens[:min_len], method_tokens[:min_len]) if a == b)
                match_str = f"{matches / min_len:.1%}"
            else:
                match_str = "N/A"

        lines.append(f"| {method} | {coherent} | {tps:.1f} | {match_str} |")

    lines.append("")

    # Generated text samples
    lines.append("### Generated Text Samples (first 200 chars)")
    lines.append("")
    for method in ["fp16", "uniform_3bit", "adaptive_unified", "adaptive_aggressive"]:
        r = long_gen_results.get(method, {})
        text = r.get("text", "")[:200]
        lines.append(f"**{method}:** `{text}`")
        lines.append("")

    # Key findings
    lines.append("## 5. Key Findings")
    lines.append("")

    # Compute key comparisons
    adaptive_eff = results.get("adaptive_target", {}).get("effective_bits", 16)
    adaptive_comp = results.get("adaptive_target", {}).get("compression_ratio", 1)
    adaptive_top5 = quality.get("adaptive_target", {}).get("top5_match", 0)
    uniform_top5 = quality.get("uniform_3bit", {}).get("top5_match", 0)
    uniform_comp = results.get("uniform_3bit", {}).get("compression_ratio", 1)

    lines.append(f"1. **Adaptive unified system:** {adaptive_eff:.2f} effective bits, "
                 f"{adaptive_comp:.1f}x compression, {adaptive_top5:.1%} top-5 match")
    lines.append(f"2. **vs Uniform 3-bit:** {uniform_comp:.1f}x compression, "
                 f"{uniform_top5:.1%} top-5 match")
    lines.append(f"3. **Compression improvement:** {adaptive_comp/max(uniform_comp, 0.01):.1f}x "
                 f"better compression than production 3-bit")

    aggressive_eff = results.get("adaptive_aggressive", {}).get("effective_bits", 16)
    aggressive_comp = results.get("adaptive_aggressive", {}).get("compression_ratio", 1)
    aggressive_top5 = quality.get("adaptive_aggressive", {}).get("top5_match", 0)
    lines.append(f"4. **Aggressive config:** {aggressive_eff:.2f} bits, "
                 f"{aggressive_comp:.1f}x compression, {aggressive_top5:.1%} top-5")

    lines.append("")
    lines.append("## 6. Conclusion")
    lines.append("")

    if adaptive_top5 > 0.90 and adaptive_comp > 5.0:
        lines.append(f"**TARGET MET:** The unified adaptive system achieves >{adaptive_top5:.0%} top-5 "
                     f"attention match at {adaptive_comp:.1f}x compression ({adaptive_eff:.2f} effective bits).")
    elif adaptive_comp > 5.0:
        lines.append(f"**High compression achieved:** {adaptive_comp:.1f}x ({adaptive_eff:.2f} bits) "
                     f"with {adaptive_top5:.1%} quality. Tier allocation is working.")
    else:
        lines.append(f"**Baseline results:** {adaptive_comp:.1f}x compression. "
                     f"Further tuning of tier thresholds needed.")

    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Results written to {out_path}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    run_benchmark()
