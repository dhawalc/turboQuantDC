"""Asymptotic analysis: Does optimal KV compression improve with context length?

Hypothesis: As context grows, attention follows a power-law where older tokens
get exponentially less attention. This means the OPTIMAL bit allocation gives
fewer bits to most tokens at long context. If true, the effective bits/token
DECREASES with context length -- compression improves with scale.

This would mean: at 1K context you need ~3 bits/token, but at 100K context
you might only need ~1.5 bits/token because 90% of tokens are old and unattended.

Measurements at each context length:
    1. Attention concentration: what % of tokens get >1% of total attention?
    2. Gini coefficient: inequality in attention distribution
    3. Optimal bit allocation using adaptive tiering
    4. Effective bits/token needed for 95% top-5 attention match
    5. Token age vs attention received (power-law exponent)
    6. Theoretical minimum bits/token

Usage:
    python benchmarks/asymptotic_analysis.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from turboquantdc.adaptive_bits import (
    ImportanceScorer,
    analyze_attention_distribution,
)
from turboquantdc.codebook import LloydMaxCodebook
from turboquantdc.rotation import generate_rotation_matrix

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"

# Context lengths to test (tokens)
CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048]

# For top-5 quality sweep -- bit-widths to test at each length
BIT_WIDTHS = [1, 2, 3, 4]

# Tier configs for optimal bit allocation analysis
ADAPTIVE_TIERS = {
    "aggressive": {
        "thresholds": [0.05, 0.20, 0.50],
        "bits": [16, 4, 3, 2],
    },
    "ultra_aggressive": {
        "thresholds": [0.03, 0.10, 0.30],
        "bits": [16, 4, 3, 1],
    },
    "temporal_decay": {
        # Simulating hot/warm/cold with 10%/20%/70% split
        "thresholds": [0.10, 0.30],
        "bits": [4, 3, 2],
    },
}

# Query window: last N queries for decode-relevant analysis
QUERY_WINDOW = 64

# Filler text for building long contexts
FILLER = (
    "The quarterly financial review meeting covered several topics including "
    "budget allocations for the upcoming fiscal year, departmental spending reports, "
    "and projected revenue streams from various business units. The committee discussed "
    "infrastructure upgrades planned for the western regional offices and noted that "
    "maintenance schedules should be coordinated with the facilities management team. "
    "Several action items were assigned to team leads for follow-up before the next "
    "meeting cycle.\n\n"
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    """Load Qwen2.5-3B-Instruct with BnB 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        attn_implementation="eager",  # need explicit attention weights
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    dt = time.time() - t0
    print(f"  Loaded in {dt:.1f}s")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt construction at various lengths
# ---------------------------------------------------------------------------
def build_prompt(tokenizer, target_tokens: int) -> str:
    """Build a prompt padded with filler to reach target_tokens length."""
    filler_len = len(tokenizer.encode(FILLER, add_special_tokens=False))
    n_reps = max(1, target_tokens // filler_len)

    parts = []
    for i in range(n_reps):
        parts.append(FILLER)

    # Add a question at the end to create decode-relevant attention
    parts.append(
        "\nBased on the meeting notes above, what were the key action items "
        "discussed regarding infrastructure upgrades and facilities management?"
    )

    text = "".join(parts)

    # Apply chat template
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Check actual token count and trim if needed
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) > target_tokens + 50:
        # Trim to approximate target
        tokens = tokens[:target_tokens]
        prompt = tokenizer.decode(tokens)

    return prompt


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------
def extract_attentions(model, tokenizer, prompt: str) -> Dict[str, Any]:
    """Run forward pass and extract per-layer attention weights."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    seq_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=False,
        )

    attentions = []
    for layer_attn in outputs.attentions:
        attentions.append(layer_attn.cpu().float())

    # Also extract KV cache for quality testing
    del outputs
    gc.collect()
    torch.cuda.empty_cache()

    # Re-run with use_cache=True for KV extraction
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=True,
        )

    kv_cache = outputs.past_key_values
    keys_per_layer = []
    values_per_layer = []
    # New HF API: DynamicCache uses .layers[i].keys / .values
    if hasattr(kv_cache, 'layers'):
        n_layers = len(kv_cache.layers)
        for layer_idx in range(n_layers):
            layer = kv_cache.layers[layer_idx]
            keys_per_layer.append(layer.keys.cpu().float())
            values_per_layer.append(layer.values.cpu().float())
    else:
        # Fallback: older tuple-based API
        n_layers = len(kv_cache)
        for layer_idx in range(n_layers):
            k, v = kv_cache[layer_idx]
            keys_per_layer.append(k.cpu().float())
            values_per_layer.append(v.cpu().float())

    del outputs
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "attentions": attentions,
        "keys": keys_per_layer,
        "values": values_per_layer,
        "seq_len": seq_len,
        "n_layers": len(attentions),
        "n_heads": attentions[0].shape[1],
        "head_dim": keys_per_layer[0].shape[-1],
    }


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------
def compute_attention_concentration(attentions: List[torch.Tensor], query_window: int = 64) -> Dict[str, Any]:
    """Measure how concentrated attention is across tokens.

    Returns metrics that characterize the power-law distribution:
    - pct_tokens_above_1pct: fraction of tokens receiving >1% attention
    - gini: Gini coefficient (inequality measure)
    - top_k concentration at various thresholds
    - entropy and normalized entropy
    """
    all_ginis = []
    all_entropies = []
    all_concentrations = {pct: [] for pct in [0.01, 0.05, 0.10, 0.20, 0.50]}
    all_pct_above_1pct = []
    all_pct_above_threshold = []

    for layer_attn in attentions:
        # layer_attn: (batch, heads, q_len, kv_len)
        if layer_attn.dim() == 3:
            layer_attn = layer_attn.unsqueeze(0)

        batch, heads, q_len, kv_len = layer_attn.shape
        n_queries = min(query_window, q_len)

        # Use last query_window queries (decode-relevant)
        decode_attn = layer_attn[:, :, -n_queries:, :]

        # Average across queries, heads, batch -> per-token attention
        per_token = decode_attn.mean(dim=2).mean(dim=(0, 1))  # (kv_len,)
        per_token = per_token / per_token.sum().clamp(min=1e-10)

        # Fraction above 1/kv_len (above uniform)
        uniform_level = 1.0 / kv_len
        pct_above = (per_token > uniform_level).float().mean().item()
        all_pct_above_threshold.append(pct_above)

        # Fraction above 1% of total attention
        pct_above_1pct = (per_token > 0.01).float().mean().item()
        all_pct_above_1pct.append(pct_above_1pct)

        # Gini coefficient
        sorted_vals = torch.sort(per_token)[0]
        n = kv_len
        if n > 1:
            index = torch.arange(1, n + 1, dtype=torch.float32)
            gini = (2 * (index * sorted_vals).sum() / (n * sorted_vals.sum().clamp(min=1e-10)) - (n + 1) / n).item()
            gini = max(0.0, gini)
        else:
            gini = 0.0
        all_ginis.append(gini)

        # Entropy
        entropy = -(per_token * torch.log(per_token + 1e-10)).sum().item()
        max_entropy = math.log(max(kv_len, 1))
        all_entropies.append(entropy / max(max_entropy, 1e-10))

        # Concentration at percentile thresholds
        sorted_desc = torch.sort(per_token, descending=True)[0]
        cumsum = torch.cumsum(sorted_desc, dim=0)
        total = cumsum[-1].item()

        for pct in [0.01, 0.05, 0.10, 0.20, 0.50]:
            k = max(1, int(pct * kv_len))
            captured = cumsum[min(k - 1, kv_len - 1)].item() / max(total, 1e-10)
            all_concentrations[pct].append(captured)

    return {
        "gini": float(np.mean(all_ginis)),
        "gini_std": float(np.std(all_ginis)),
        "normalized_entropy": float(np.mean(all_entropies)),
        "pct_tokens_above_1pct": float(np.mean(all_pct_above_1pct)),
        "pct_tokens_above_uniform": float(np.mean(all_pct_above_threshold)),
        "concentration": {
            f"top_{int(pct*100)}pct": float(np.mean(vals))
            for pct, vals in all_concentrations.items()
        },
        "per_layer_gini": [float(g) for g in all_ginis],
    }


def compute_age_vs_attention(attentions: List[torch.Tensor], query_window: int = 64) -> Dict[str, Any]:
    """Measure how attention scales with token age (position from end).

    Fits a power-law: attention(age) ~ age^(-alpha)
    Higher alpha means older tokens decay faster -- more compressible.
    """
    age_attention_curves = []

    for layer_attn in attentions:
        if layer_attn.dim() == 3:
            layer_attn = layer_attn.unsqueeze(0)

        batch, heads, q_len, kv_len = layer_attn.shape
        n_queries = min(query_window, q_len)

        decode_attn = layer_attn[:, :, -n_queries:, :]
        per_token = decode_attn.mean(dim=2).mean(dim=(0, 1))  # (kv_len,)
        per_token = per_token / per_token.sum().clamp(min=1e-10)

        # Token "age" = distance from end of sequence
        # age[i] = kv_len - i (most recent token has age 1)
        ages = torch.arange(kv_len, 0, -1, dtype=torch.float32)

        age_attention_curves.append((ages.numpy(), per_token.numpy()))

    # Average across layers -- bin by age
    avg_attn = np.mean([curve[1] for curve in age_attention_curves], axis=0)
    ages = age_attention_curves[0][0]  # same for all layers

    # Fit power law on log-log scale: log(attn) = -alpha * log(age) + c
    # Exclude very recent tokens (age < 5) to avoid edge effects
    kv_len = len(ages)
    mask = ages > 5
    if mask.sum() > 10:
        log_ages = np.log(ages[mask])
        log_attn = np.log(avg_attn[mask] + 1e-15)

        # Linear regression: log_attn = slope * log_ages + intercept
        valid = np.isfinite(log_attn)
        if valid.sum() > 5:
            coeffs = np.polyfit(log_ages[valid], log_attn[valid], 1)
            alpha = -coeffs[0]  # power-law exponent (positive means decay)
        else:
            alpha = 0.0
    else:
        alpha = 0.0

    # Compute attention decay percentiles
    # What fraction of total attention do the most recent 10%, 20%, 50% of tokens get?
    sorted_by_recency = avg_attn[::-1]  # most recent first
    cumsum = np.cumsum(sorted_by_recency)
    total = cumsum[-1]

    recency_concentration = {}
    for pct in [0.10, 0.20, 0.50]:
        k = max(1, int(pct * kv_len))
        recency_concentration[f"recent_{int(pct*100)}pct"] = float(cumsum[min(k-1, kv_len-1)] / max(total, 1e-10))

    return {
        "power_law_exponent": float(alpha),
        "recency_concentration": recency_concentration,
        "avg_attention_curve": avg_attn.tolist(),
    }


def compute_optimal_bits_for_quality(
    keys_per_layer: List[torch.Tensor],
    attentions: List[torch.Tensor],
    head_dim: int,
    target_top5: float = 0.95,
    query_window: int = 64,
) -> Dict[str, Any]:
    """Find the minimum effective bits/token for target attention quality.

    For each layer/head, compresses keys at various bit-widths using adaptive
    allocation, then checks if top-5 attention match >= target.

    Returns the minimum effective bits/token that achieves the target.
    """
    results_per_bits = {}

    for bits in BIT_WIDTHS:
        all_top5_matches = []

        # Pre-compute codebook and rotation for this bit-width
        cb = LloydMaxCodebook(d=head_dim, bits=bits)
        rot = generate_rotation_matrix(head_dim, seed=42, device="cpu")

        for layer_idx in range(len(keys_per_layer)):
            layer_keys = keys_per_layer[layer_idx]  # (batch, n_heads, seq_len, d)
            layer_attn = attentions[layer_idx]  # (batch, heads, q_len, kv_len)

            if layer_keys.dim() == 4:
                n_heads = layer_keys.shape[1]
                seq_len = layer_keys.shape[2]
            else:
                continue

            n_queries = min(query_window, layer_attn.shape[2])

            for head_idx in range(n_heads):
                # Extract keys for this head: (seq_len, d)
                head_keys = layer_keys[0, head_idx]

                # Quantize and dequantize
                norms = head_keys.norm(dim=-1, keepdim=True)
                normalized = head_keys / (norms + 1e-8)
                rotated = normalized @ rot
                indices = torch.bucketize(rotated, cb.boundaries)
                indices = indices.clamp(0, cb.centroids.shape[0] - 1)
                reconstructed = cb.centroids[indices]
                unrotated = reconstructed @ rot.T
                compressed_keys = unrotated * norms

                # Use last few queries as test queries
                # Generate queries from the attention pattern
                query_keys = head_keys[-n_queries:]  # use last N keys as proxy queries

                # Compute attention scores
                scale = 1.0 / math.sqrt(head_dim)
                fp16_scores = (query_keys @ head_keys.T) * scale
                comp_scores = (query_keys @ compressed_keys.T) * scale

                fp16_attn = F.softmax(fp16_scores, dim=-1)
                comp_attn = F.softmax(comp_scores, dim=-1)

                # Top-5 match
                for q_idx in range(n_queries):
                    fp16_top5 = set(torch.topk(fp16_attn[q_idx], k=min(5, seq_len)).indices.tolist())
                    comp_top5 = set(torch.topk(comp_attn[q_idx], k=min(5, seq_len)).indices.tolist())
                    match = len(fp16_top5 & comp_top5) / 5.0
                    all_top5_matches.append(match)

        avg_top5 = float(np.mean(all_top5_matches)) if all_top5_matches else 0.0
        results_per_bits[bits] = {
            "avg_top5_overlap": avg_top5,
            "n_samples": len(all_top5_matches),
        }

    # Find minimum bits for target quality
    min_bits_for_target = None
    for bits in sorted(BIT_WIDTHS):
        if results_per_bits[bits]["avg_top5_overlap"] >= target_top5:
            min_bits_for_target = bits
            break

    return {
        "per_bits": results_per_bits,
        "min_bits_for_95pct_top5": min_bits_for_target,
    }


def compute_adaptive_effective_bits(
    attentions: List[torch.Tensor],
    query_window: int = 64,
) -> Dict[str, Dict[str, float]]:
    """Compute effective bits/token under various adaptive tiering schemes.

    Uses actual attention patterns to classify tokens into importance tiers,
    then computes the weighted average bit-width.
    """
    results = {}

    for config_name, config in ADAPTIVE_TIERS.items():
        scorer = ImportanceScorer(ema_decay=0.9)

        # Feed all layers' attention patterns to build importance scores
        for layer_attn in attentions:
            scorer.update(layer_attn, query_window=query_window)

        # Classify into tiers
        tiers = scorer.classify_tiers(config["thresholds"])
        tier_bits = config["bits"]

        # Compute effective bits
        total_bits = 0.0
        tier_counts = {}
        for tier_id, b in enumerate(tier_bits):
            count = (tiers == tier_id).sum().item()
            total_bits += count * b
            tier_counts[f"tier_{tier_id}_bits_{b}"] = count

        n_tokens = tiers.shape[0]
        effective = total_bits / max(n_tokens, 1)

        results[config_name] = {
            "effective_bits": effective,
            "tier_distribution": tier_counts,
            "n_tokens": n_tokens,
            "compression_ratio_vs_fp16": 16.0 / effective if effective > 0 else 0,
        }

    return results


def compute_theoretical_minimum_bits(
    attentions: List[torch.Tensor],
    query_window: int = 64,
) -> Dict[str, float]:
    """Estimate the theoretical minimum bits/token from attention entropy.

    The idea: if a token receives 0 attention, it could be 0 bits.
    The information content of the attention distribution bounds
    the minimum bits needed to preserve it.

    We compute: for each token, what's the minimum bits needed given
    how much attention it receives? Tokens with near-zero attention
    need near-zero bits. The weighted average gives a lower bound.
    """
    all_min_bits = []

    for layer_attn in attentions:
        if layer_attn.dim() == 3:
            layer_attn = layer_attn.unsqueeze(0)

        batch, heads, q_len, kv_len = layer_attn.shape
        n_queries = min(query_window, q_len)
        decode_attn = layer_attn[:, :, -n_queries:, :]
        per_token = decode_attn.mean(dim=2).mean(dim=(0, 1))
        per_token = per_token / per_token.sum().clamp(min=1e-10)

        # Minimum bits per token: proportional to attention received
        # Tokens with attention < 1/n need < 1 bit (they barely matter)
        # Tokens at top need ~4 bits to preserve attention ranking
        # Scale: 4 bits for max-attention token, linear down to 0
        max_attn = per_token.max().item()
        if max_attn > 0:
            relative_importance = per_token / max_attn  # 0 to 1
            # Minimum bits: 4 * relative_importance (top token needs 4 bits, zero-attention needs 0)
            min_bits_per_token = 4.0 * relative_importance
            # Average weighted by nothing (all tokens equal size)
            avg_min = min_bits_per_token.mean().item()
        else:
            avg_min = 0.0

        all_min_bits.append(avg_min)

    return {
        "avg_theoretical_min_bits": float(np.mean(all_min_bits)),
        "per_layer": [float(x) for x in all_min_bits],
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis():
    """Run the full asymptotic analysis across context lengths."""
    print("=" * 72)
    print("ASYMPTOTIC ANALYSIS: Does optimal compression improve with context?")
    print("=" * 72)
    print(f"Model: {MODEL_NAME}")
    print(f"Context lengths: {CONTEXT_LENGTHS}")
    print(f"Query window: {QUERY_WINDOW}")
    print()

    model, tokenizer = load_model()

    all_results = {}

    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n{'='*60}")
        print(f"CONTEXT LENGTH: {ctx_len} tokens")
        print(f"{'='*60}")

        # Build prompt
        prompt = build_prompt(tokenizer, target_tokens=ctx_len)
        actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        print(f"  Prompt built: {actual_tokens} tokens (target: {ctx_len})")

        # Extract attentions and KV cache
        print("  Running forward pass...")
        t0 = time.time()
        data = extract_attentions(model, tokenizer, prompt)
        dt = time.time() - t0
        actual_seq_len = data["seq_len"]
        print(f"  Forward pass: {dt:.1f}s, seq_len={actual_seq_len}, "
              f"layers={data['n_layers']}, heads={data['n_heads']}, d={data['head_dim']}")

        # 1. Attention concentration
        print("  Analyzing attention concentration...")
        concentration = compute_attention_concentration(data["attentions"], QUERY_WINDOW)
        print(f"    Gini coefficient: {concentration['gini']:.4f}")
        print(f"    Tokens above 1% attention: {concentration['pct_tokens_above_1pct']:.2%}")
        print(f"    Normalized entropy: {concentration['normalized_entropy']:.4f}")
        for k, v in concentration["concentration"].items():
            print(f"    {k} captures: {v:.2%} of attention")

        # 2. Age vs attention (power-law exponent)
        print("  Fitting power-law decay...")
        age_analysis = compute_age_vs_attention(data["attentions"], QUERY_WINDOW)
        print(f"    Power-law exponent (alpha): {age_analysis['power_law_exponent']:.4f}")
        for k, v in age_analysis["recency_concentration"].items():
            print(f"    {k}: {v:.2%} of attention")

        # 3. Adaptive bit allocation
        print("  Computing adaptive bit allocation...")
        adaptive = compute_adaptive_effective_bits(data["attentions"], QUERY_WINDOW)
        for config_name, config_result in adaptive.items():
            eff_bits = config_result["effective_bits"]
            cr = config_result["compression_ratio_vs_fp16"]
            print(f"    {config_name}: {eff_bits:.2f} eff bits, {cr:.1f}x compression")

        # 4. Quality at each bit-width
        print("  Measuring quality at each bit-width...")
        quality = compute_optimal_bits_for_quality(
            data["keys"], data["attentions"], data["head_dim"],
            target_top5=0.95, query_window=QUERY_WINDOW,
        )
        for bits, q in quality["per_bits"].items():
            print(f"    {bits}-bit: top-5 overlap = {q['avg_top5_overlap']:.2%}")
        if quality["min_bits_for_95pct_top5"] is not None:
            print(f"    => Min bits for 95% top-5: {quality['min_bits_for_95pct_top5']}")
        else:
            print(f"    => No bit-width achieved 95% top-5 (max: {max(q['avg_top5_overlap'] for q in quality['per_bits'].values()):.2%})")

        # 5. Theoretical minimum bits
        print("  Computing theoretical minimum bits...")
        theory = compute_theoretical_minimum_bits(data["attentions"], QUERY_WINDOW)
        print(f"    Theoretical min bits/token: {theory['avg_theoretical_min_bits']:.4f}")

        # Store results
        all_results[ctx_len] = {
            "actual_seq_len": actual_seq_len,
            "concentration": concentration,
            "age_analysis": {
                "power_law_exponent": age_analysis["power_law_exponent"],
                "recency_concentration": age_analysis["recency_concentration"],
            },
            "adaptive_bits": adaptive,
            "quality": quality,
            "theoretical_min": theory,
        }

        # Cleanup
        del data
        gc.collect()
        torch.cuda.empty_cache()

    # Unload model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return all_results


def format_results(results: Dict[int, Dict]) -> str:
    """Format results into a markdown report."""
    lines = []
    lines.append("# Asymptotic Analysis: Compression vs Context Length")
    lines.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Model:** {MODEL_NAME}")
    lines.append(f"**Query window:** {QUERY_WINDOW}")
    lines.append("")

    # ---- The Key Result ----
    lines.append("## The Key Question: Does compression improve with scale?")
    lines.append("")
    lines.append("If effective bits/token **decreases** with context length, then")
    lines.append("compression improves with scale -- the exact opposite of the usual")
    lines.append("\"longer context = more memory\" problem.")
    lines.append("")

    # Summary table
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Context | Gini | Entropy | Top-1% Captures | Tokens >1% | Power-Law alpha | Aggressive eff_bits | Ultra eff_bits | Temporal eff_bits | Theoretical min |")
    lines.append("|---------|------|---------|-----------------|------------|-----------------|--------------------|--------------------|-------------------|-----------------|")

    for ctx_len in sorted(results.keys()):
        r = results[ctx_len]
        c = r["concentration"]
        a = r["age_analysis"]
        ad = r["adaptive_bits"]
        th = r["theoretical_min"]

        top1_captures = c["concentration"].get("top_1pct", 0)
        aggressive_bits = ad.get("aggressive", {}).get("effective_bits", 0)
        ultra_bits = ad.get("ultra_aggressive", {}).get("effective_bits", 0)
        temporal_bits = ad.get("temporal_decay", {}).get("effective_bits", 0)

        lines.append(
            f"| {r['actual_seq_len']:,} | {c['gini']:.4f} | {c['normalized_entropy']:.4f} | "
            f"{top1_captures:.2%} | {c['pct_tokens_above_1pct']:.2%} | "
            f"{a['power_law_exponent']:.4f} | {aggressive_bits:.2f} | {ultra_bits:.2f} | "
            f"{temporal_bits:.2f} | {th['avg_theoretical_min_bits']:.4f} |"
        )

    # Quality table
    lines.append("")
    lines.append("## Quality at Each Bit-Width (Top-5 Attention Overlap)")
    lines.append("")
    header = "| Context |"
    sep = "|---------|"
    for bits in BIT_WIDTHS:
        header += f" {bits}-bit |"
        sep += "-------|"
    header += " Min bits for 95% |"
    sep += "-----------------|"
    lines.append(header)
    lines.append(sep)

    for ctx_len in sorted(results.keys()):
        r = results[ctx_len]
        q = r["quality"]
        row = f"| {r['actual_seq_len']:,} |"
        for bits in BIT_WIDTHS:
            if bits in q["per_bits"]:
                row += f" {q['per_bits'][bits]['avg_top5_overlap']:.2%} |"
            else:
                row += " -- |"
        min_bits = q.get("min_bits_for_95pct_top5")
        row += f" {min_bits if min_bits else 'N/A'} |"
        lines.append(row)

    # Adaptive allocation detail
    lines.append("")
    lines.append("## Adaptive Bit Allocation Detail")
    lines.append("")

    for config_name, config in ADAPTIVE_TIERS.items():
        lines.append(f"### {config_name}")
        lines.append(f"Tiers: {config['bits']}, Thresholds: {config['thresholds']}")
        lines.append("")
        lines.append("| Context | Eff Bits | Compression vs FP16 | Tier Distribution |")
        lines.append("|---------|----------|---------------------|-------------------|")

        for ctx_len in sorted(results.keys()):
            r = results[ctx_len]
            ad = r["adaptive_bits"].get(config_name, {})
            eff = ad.get("effective_bits", 0)
            cr = ad.get("compression_ratio_vs_fp16", 0)
            dist = ad.get("tier_distribution", {})
            dist_str = ", ".join(f"{k}:{v}" for k, v in dist.items())
            lines.append(f"| {r['actual_seq_len']:,} | {eff:.2f} | {cr:.1f}x | {dist_str} |")

        lines.append("")

    # Power-law analysis
    lines.append("## Power-Law Analysis")
    lines.append("")
    lines.append("The power-law exponent alpha characterizes how fast attention decays")
    lines.append("with token age. Higher alpha = faster decay = more compressible.")
    lines.append("")
    lines.append("| Context | Power-Law alpha | Recent 10% gets | Recent 20% gets | Recent 50% gets |")
    lines.append("|---------|----------------|-----------------|-----------------|-----------------|")

    for ctx_len in sorted(results.keys()):
        r = results[ctx_len]
        a = r["age_analysis"]
        rc = a["recency_concentration"]
        lines.append(
            f"| {r['actual_seq_len']:,} | {a['power_law_exponent']:.4f} | "
            f"{rc.get('recent_10pct', 0):.2%} | "
            f"{rc.get('recent_20pct', 0):.2%} | "
            f"{rc.get('recent_50pct', 0):.2%} |"
        )

    # Theoretical minimum
    lines.append("")
    lines.append("## Theoretical Minimum Bits/Token")
    lines.append("")
    lines.append("Lower bound on bits/token based on attention distribution.")
    lines.append("Tokens receiving near-zero attention could theoretically be stored at near-zero bits.")
    lines.append("")
    lines.append("| Context | Avg Min Bits | Layer Range |")
    lines.append("|---------|-------------|-------------|")

    for ctx_len in sorted(results.keys()):
        r = results[ctx_len]
        th = r["theoretical_min"]
        min_layer = min(th["per_layer"])
        max_layer = max(th["per_layer"])
        lines.append(
            f"| {r['actual_seq_len']:,} | {th['avg_theoretical_min_bits']:.4f} | "
            f"{min_layer:.4f} - {max_layer:.4f} |"
        )

    # Conclusions
    lines.append("")
    lines.append("## Analysis")
    lines.append("")

    # Check if the trend is decreasing
    ctx_lens = sorted(results.keys())
    if len(ctx_lens) >= 2:
        first = results[ctx_lens[0]]
        last = results[ctx_lens[-1]]

        gini_trend = last["concentration"]["gini"] - first["concentration"]["gini"]
        alpha_trend = last["age_analysis"]["power_law_exponent"] - first["age_analysis"]["power_law_exponent"]
        theory_trend = last["theoretical_min"]["avg_theoretical_min_bits"] - first["theoretical_min"]["avg_theoretical_min_bits"]

        lines.append(f"**Gini trend ({ctx_lens[0]} -> {ctx_lens[-1]} tokens):** {gini_trend:+.4f}")
        if gini_trend > 0:
            lines.append("  => Attention becomes MORE concentrated at longer context (good for compression)")
        else:
            lines.append("  => Attention becomes less concentrated at longer context")

        lines.append(f"\n**Power-law exponent trend:** {alpha_trend:+.4f}")
        if alpha_trend > 0:
            lines.append("  => Attention decays FASTER with age at longer context (good for compression)")
        else:
            lines.append("  => Attention decay rate stable or decreasing")

        lines.append(f"\n**Theoretical min bits trend:** {theory_trend:+.4f}")
        if theory_trend < 0:
            lines.append("  => Theoretical minimum bits DECREASES with context (COMPRESSION IMPROVES WITH SCALE)")
        else:
            lines.append("  => Theoretical minimum bits stable or increasing")

        # Adaptive bits trend
        for config_name in ADAPTIVE_TIERS:
            first_eff = first["adaptive_bits"].get(config_name, {}).get("effective_bits", 0)
            last_eff = last["adaptive_bits"].get(config_name, {}).get("effective_bits", 0)
            trend = last_eff - first_eff
            lines.append(f"\n**{config_name} effective bits trend:** {trend:+.4f}")

    lines.append("")
    return "\n".join(lines)


def main():
    results = run_analysis()

    # Save structured results
    results_dir = os.path.join(REPO_ROOT, "benchmarks", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save JSON (with non-serializable items handled)
    json_path = os.path.join(results_dir, "asymptotic_results.json")
    json_safe = {}
    for ctx_len, r in results.items():
        # Remove large attention curves from JSON
        r_copy = {k: v for k, v in r.items()}
        if "age_analysis" in r_copy:
            r_copy["age_analysis"] = {
                k: v for k, v in r_copy["age_analysis"].items()
                if k != "avg_attention_curve"
            }
        json_safe[str(ctx_len)] = r_copy

    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2, default=str)
    print(f"\nJSON results saved to {json_path}")

    # Save markdown report
    md_path = os.path.join(results_dir, "asymptotic_results.md")
    report = format_results(results)
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Markdown report saved to {md_path}")

    # Print the key curve
    print("\n" + "=" * 72)
    print("THE CURVE: Effective bits/token vs context length")
    print("=" * 72)
    print(f"{'Context':>10} | {'Gini':>8} | {'Alpha':>8} | {'Aggressive':>12} | {'Ultra':>12} | {'Theory Min':>12}")
    print("-" * 72)
    for ctx_len in sorted(results.keys()):
        r = results[ctx_len]
        gini = r["concentration"]["gini"]
        alpha = r["age_analysis"]["power_law_exponent"]
        agg = r["adaptive_bits"].get("aggressive", {}).get("effective_bits", 0)
        ultra = r["adaptive_bits"].get("ultra_aggressive", {}).get("effective_bits", 0)
        theory = r["theoretical_min"]["avg_theoretical_min_bits"]
        print(f"{r['actual_seq_len']:>10,} | {gini:>8.4f} | {alpha:>8.4f} | {agg:>12.2f} | {ultra:>12.2f} | {theory:>12.4f}")

    print()
    print("If Gini INCREASES and Theory Min DECREASES with context => COMPRESSION IMPROVES WITH SCALE")


if __name__ == "__main__":
    main()
