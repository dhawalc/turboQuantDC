"""Cross-Head Delta Compression Experiment.

Measures inter-head correlation on real KV caches from Qwen2.5-3B,
then compares cross-head delta compression vs uniform baselines.

Configurations tested:
1. FP16 baseline (no compression)
2. Uniform 3-bit ResidualQuant (5.1x compression)
3. Uniform 2-bit ResidualQuant (8x compression)
4. Cross-head: 3-bit anchor + 1-bit deltas (12.8x compression for 8 heads)
5. Cross-head: 3-bit anchor + 2-bit deltas (6.9x for 8 heads)
6. Cross-head: 2-bit anchor + 1-bit deltas (16x for 8 heads)

Quality metrics:
- Vector cosine similarity (per-head and overall)
- Top-5 attention position match
- Attention score Pearson correlation
- Generation quality (coherence + keyword match)
"""

import gc
import os
import sys
import time
from typing import Dict, List

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from turboquantdc.cross_head_compress import (
    CrossHeadDeltaQuantizer,
    UniformQuantizer,
    evaluate_attention_quality,
    evaluate_reconstruction_quality,
    measure_inter_head_correlation,
    select_best_anchor,
)

# ---- Configuration ----
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 60
DO_SAMPLE = False

PROMPTS = [
    "What is the capital of Australia? Answer briefly:",
    "What is 15 + 27?",
    "Who wrote the novel 1984? Answer briefly:",
    "Explain what a neural network is in two sentences:",
    "Write a Python function that returns the factorial of n:",
]

EXPECTED_KEYWORDS = [
    ["canberra"],
    ["42"],
    ["george", "orwell"],
    ["layer", "neuron", "learn", "network", "input", "output", "weight"],
    ["def", "factorial", "return"],
]


def load_model():
    """Load model once for all experiments."""
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


def extract_kv_cache(model, tokenizer, prompt: str):
    """Extract KV cache from a forward pass.

    Returns dict mapping layer_idx -> (keys, values).
    Each has shape [batch=1, num_heads, seq_len, head_dim].
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past = outputs.past_key_values

    # DynamicCache iteration yields (keys, values, None) tuples
    kv_by_layer = {}
    for layer_idx, item in enumerate(past):
        k, v = item[0], item[1]
        kv_by_layer[layer_idx] = (
            k.detach().float().cuda(),
            v.detach().float().cuda(),
        )

    print(f"  Captured KV from {len(kv_by_layer)} layers")
    if kv_by_layer:
        sample_k = next(iter(kv_by_layer.values()))[0]
        print(f"  Shape: {sample_k.shape} "
              f"(batch={sample_k.shape[0]}, heads={sample_k.shape[1]}, "
              f"seq={sample_k.shape[2]}, dim={sample_k.shape[3]})")
    return kv_by_layer


def measure_all_correlations(kv_by_layer):
    """Measure inter-head correlation across all layers."""
    print("\n" + "=" * 70)
    print("  INTER-HEAD CORRELATION ANALYSIS")
    print("=" * 70)

    all_key_stats = []
    all_value_stats = []

    # Sample a few representative layers
    num_layers = len(kv_by_layer)
    sample_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
    sample_layers = sorted(set(sample_layers))

    for layer_idx in sample_layers:
        keys, values = kv_by_layer[layer_idx]
        print(f"\n  Layer {layer_idx}: keys shape = {keys.shape}")

        key_stats = measure_inter_head_correlation(keys, mode="key")
        val_stats = measure_inter_head_correlation(values, mode="value")

        print(f"    Keys:   mean cosine = {key_stats['mean_cosine']:.4f}, "
              f"mean Pearson = {key_stats['mean_pearson']:.4f}, "
              f"mean delta norm = {key_stats['mean_delta_norm']:.4f}")
        print(f"    Values: mean cosine = {val_stats['mean_cosine']:.4f}, "
              f"mean Pearson = {val_stats['mean_pearson']:.4f}, "
              f"mean delta norm = {val_stats['mean_delta_norm']:.4f}")

        # Anchor delta stats
        print(f"    Anchor (head 0) delta stats for keys:")
        for s in key_stats["anchor_delta_stats"][:3]:
            print(f"      Head {s['head']}: cosine={s['cosine_to_anchor']:.4f}, "
                  f"var_ratio={s['variance_ratio']:.4f}, "
                  f"delta_norm={s['relative_delta_norm']:.4f}")

        all_key_stats.append(key_stats)
        all_value_stats.append(val_stats)

    # Summary statistics
    avg_key_cosine = sum(s["mean_cosine"] for s in all_key_stats) / len(all_key_stats)
    avg_key_pearson = sum(s["mean_pearson"] for s in all_key_stats) / len(all_key_stats)
    avg_val_cosine = sum(s["mean_cosine"] for s in all_value_stats) / len(all_value_stats)
    avg_val_pearson = sum(s["mean_pearson"] for s in all_value_stats) / len(all_value_stats)

    avg_var_ratio_keys = sum(
        sum(s["variance_ratio"] for s in stats["anchor_delta_stats"]) / len(stats["anchor_delta_stats"])
        for stats in all_key_stats
    ) / len(all_key_stats)

    avg_var_ratio_vals = sum(
        sum(s["variance_ratio"] for s in stats["anchor_delta_stats"]) / len(stats["anchor_delta_stats"])
        for stats in all_value_stats
    ) / len(all_value_stats)

    # Select best anchor for a middle layer
    mid_layer = num_layers // 2
    keys_mid, _ = kv_by_layer[mid_layer]
    best_anchor = select_best_anchor(keys_mid)

    print(f"\n  CORRELATION SUMMARY:")
    print(f"    Keys:   avg cosine = {avg_key_cosine:.4f}, avg Pearson = {avg_key_pearson:.4f}")
    print(f"    Values: avg cosine = {avg_val_cosine:.4f}, avg Pearson = {avg_val_pearson:.4f}")
    print(f"    Key delta variance ratio (vs original): {avg_var_ratio_keys:.4f}")
    print(f"    Value delta variance ratio (vs original): {avg_var_ratio_vals:.4f}")
    print(f"    Best anchor head (layer {mid_layer}): head {best_anchor}")

    return {
        "avg_key_cosine": avg_key_cosine,
        "avg_key_pearson": avg_key_pearson,
        "avg_val_cosine": avg_val_cosine,
        "avg_val_pearson": avg_val_pearson,
        "avg_var_ratio_keys": avg_var_ratio_keys,
        "avg_var_ratio_vals": avg_var_ratio_vals,
        "best_anchor": best_anchor,
        "per_layer_key_stats": all_key_stats,
        "per_layer_val_stats": all_value_stats,
    }


def run_compression_comparison(kv_by_layer):
    """Compare compression methods on real KV cache data."""
    print("\n" + "=" * 70)
    print("  COMPRESSION QUALITY COMPARISON")
    print("=" * 70)

    # Pick a representative middle layer
    num_layers = len(kv_by_layer)
    test_layers = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]

    all_results = {}

    for layer_idx in test_layers:
        keys, values = kv_by_layer[layer_idx]
        batch, num_heads, seq_len, head_dim = keys.shape

        print(f"\n  Layer {layer_idx}: [{batch}, {num_heads} heads, {seq_len} tokens, d={head_dim}]")

        # Generate synthetic queries for attention quality measurement
        torch.manual_seed(42)
        queries = torch.randn(batch, num_heads, min(4, seq_len), head_dim, device=keys.device)

        configs = [
            ("Uniform 3-bit RQ",    "uniform", {"bits": 3}),
            ("Uniform 2-bit RQ",    "uniform", {"bits": 2}),
            ("Uniform 1-bit RQ",    "uniform", {"bits": 1}),
            ("Cross-head 3+1",      "cross",   {"anchor_bits": 3, "delta_bits": 1}),
            ("Cross-head 3+2",      "cross",   {"anchor_bits": 3, "delta_bits": 2}),
            ("Cross-head 2+1",      "cross",   {"anchor_bits": 2, "delta_bits": 1}),
        ]

        layer_results = {}

        for name, method, params in configs:
            if method == "uniform":
                quant = UniformQuantizer(
                    d=head_dim, num_heads=num_heads,
                    bits=params["bits"], seed=42, device="cuda",
                )
            else:
                quant = CrossHeadDeltaQuantizer(
                    d=head_dim, num_heads=num_heads,
                    anchor_bits=params["anchor_bits"],
                    delta_bits=params["delta_bits"],
                    anchor_head=0, seed=42, device="cuda",
                )

            t0 = time.time()
            recon_keys = quant.quantize_dequantize(keys)
            compress_time = time.time() - t0

            # Reconstruction quality
            recon_qual = evaluate_reconstruction_quality(keys, recon_keys)

            # Attention quality
            attn_qual = evaluate_attention_quality(queries, keys, recon_keys, top_k=5)

            eff_bits = quant.effective_bits_per_element()
            comp_ratio = quant.compression_ratio()

            print(f"\n    {name} ({eff_bits:.2f} bits, {comp_ratio:.1f}x):")
            print(f"      Cosine sim:     mean={recon_qual['mean_cosine_sim']:.4f}, "
                  f"min={recon_qual['min_cosine_sim']:.4f}, "
                  f"p5={recon_qual['p5_cosine_sim']:.4f}")
            print(f"      Top-5 attn:     {attn_qual['top5_attention_match']:.4f}")
            print(f"      Attn Pearson r: {attn_qual['attention_score_pearson_r']:.4f}")
            print(f"      MSE:            {recon_qual['mse']:.6f}")
            print(f"      Time:           {compress_time*1000:.1f}ms")

            # Per-head cosine sim
            head_cos = recon_qual["per_head_cosine"]
            print(f"      Per-head cos:   {' '.join(f'{c:.3f}' for c in head_cos[:8])}")

            layer_results[name] = {
                **recon_qual,
                **attn_qual,
                "effective_bits": eff_bits,
                "compression_ratio": comp_ratio,
                "time_ms": compress_time * 1000,
            }

            del quant, recon_keys
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[layer_idx] = layer_results

    return all_results


def check_coherence(response, keywords):
    """Check if response is coherent and contains expected keywords."""
    response_lower = response.lower()
    words = response.split()
    if len(words) < 2:
        return False, "too_short"
    if len(words) >= 6:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        for tg in set(trigrams):
            if trigrams.count(tg) > 3:
                return False, "repetitive"
    has_keyword = any(kw in response_lower for kw in keywords)
    return has_keyword, "correct" if has_keyword else "wrong_content"


def run_generation_test(model, tokenizer):
    """Test generation quality with cross-head delta compression cache.

    Since CrossHeadDeltaQuantizer works on raw tensors, we extract
    KV cache from the model, compress/decompress offline, and compare
    attention quality. For generation, we compare against the cached
    ResidualQuantCache baseline.
    """
    print("\n" + "=" * 70)
    print("  GENERATION QUALITY TEST")
    print("=" * 70)

    from turboquantdc.residual_quant import ResidualQuantCache

    gen_results = {}

    # FP16 baseline
    print("\n  FP16 Baseline:")
    correct = 0
    for i, (prompt, keywords) in enumerate(zip(PROMPTS, EXPECTED_KEYWORDS)):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE)
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        is_correct, status = check_coherence(response, keywords)
        if is_correct:
            correct += 1
        icon = "PASS" if is_correct else "FAIL"
        print(f"    [{icon}] {prompt[:50]}... -> {response[:100]}")
        gc.collect()
        torch.cuda.empty_cache()
    gen_results["FP16"] = f"{correct}/{len(PROMPTS)}"
    print(f"    SCORE: {correct}/{len(PROMPTS)}")

    # ResidualQuant 3-bit baseline
    print("\n  ResidualQuant 3-bit:")
    correct = 0
    try:
        for i, (prompt, keywords) in enumerate(zip(PROMPTS, EXPECTED_KEYWORDS)):
            cache = ResidualQuantCache(bits=3, seed=42)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE, past_key_values=cache,
                )
            response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            is_correct, status = check_coherence(response, keywords)
            if is_correct:
                correct += 1
            icon = "PASS" if is_correct else "FAIL"
            print(f"    [{icon}] {prompt[:50]}... -> {response[:100]}")
            del cache
            gc.collect()
            torch.cuda.empty_cache()
        gen_results["RQ-3bit"] = f"{correct}/{len(PROMPTS)}"
        print(f"    SCORE: {correct}/{len(PROMPTS)}")
    except Exception as e:
        print(f"    ERROR: {e}")
        gen_results["RQ-3bit"] = "error"

    return gen_results


def format_results_markdown(
    correlation_stats: Dict,
    compression_results: Dict,
    generation_results: Dict,
    synthetic_results: Dict = None,
) -> str:
    """Format all results into a markdown report."""
    lines = [
        "# Cross-Head Delta Compression Results",
        "",
        f"Model: {MODEL_NAME}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Hypothesis",
        "",
        "KV heads at the same layer position are correlated. By compressing",
        "one anchor head at 3-bit and remaining heads as 1-bit deltas,",
        "we can achieve 12.8x compression (1.25 bits/element) for 8 KV heads.",
        "",
        "## Inter-Head Correlation",
        "",
        "| Metric | Keys | Values |",
        "|--------|------|--------|",
        f"| Mean cosine similarity | {correlation_stats['avg_key_cosine']:.4f} | {correlation_stats['avg_val_cosine']:.4f} |",
        f"| Mean Pearson r | {correlation_stats['avg_key_pearson']:.4f} | {correlation_stats['avg_val_pearson']:.4f} |",
        f"| Delta variance ratio | {correlation_stats['avg_var_ratio_keys']:.4f} | {correlation_stats['avg_var_ratio_vals']:.4f} |",
        "",
        "(Variance ratio < 1.0 means deltas have lower variance than originals,",
        "indicating cross-head compression is beneficial.)",
        "",
    ]

    # Per-layer anchor delta stats
    lines.extend([
        "### Per-Layer Key Delta Statistics (vs Head 0 Anchor)",
        "",
        "| Layer | Head | Cosine to Anchor | Variance Ratio | Relative Delta Norm |",
        "|-------|------|------------------|----------------|---------------------|",
    ])
    for i, stats in enumerate(correlation_stats["per_layer_key_stats"]):
        for s in stats["anchor_delta_stats"][:4]:
            lines.append(
                f"| {i} | {s['head']} | {s['cosine_to_anchor']:.4f} | "
                f"{s['variance_ratio']:.4f} | {s['relative_delta_norm']:.4f} |"
            )
    lines.append("")

    # Compression quality comparison
    lines.extend([
        "## Compression Quality Comparison",
        "",
    ])

    for layer_idx, layer_results in compression_results.items():
        lines.extend([
            f"### Layer {layer_idx}",
            "",
            "| Config | Eff. Bits | Comp. Ratio | Mean Cos | Min Cos | Top-5 Attn | Attn r | MSE |",
            "|--------|-----------|-------------|----------|---------|------------|--------|-----|",
        ])
        for name, metrics in layer_results.items():
            lines.append(
                f"| {name} | {metrics['effective_bits']:.2f} | "
                f"{metrics['compression_ratio']:.1f}x | "
                f"{metrics['mean_cosine_sim']:.4f} | "
                f"{metrics['min_cosine_sim']:.4f} | "
                f"{metrics['top5_attention_match']:.4f} | "
                f"{metrics['attention_score_pearson_r']:.4f} | "
                f"{metrics['mse']:.6f} |"
            )
        lines.append("")

    # Generation quality
    lines.extend([
        "## Generation Quality",
        "",
        "| Config | Score |",
        "|--------|-------|",
    ])
    for name, score in generation_results.items():
        lines.append(f"| {name} | {score} |")
    lines.append("")

    # Analysis
    lines.extend([
        "## Analysis",
        "",
        "### Key Finding: Inter-Head Correlation",
        "",
    ])

    if correlation_stats["avg_key_cosine"] > 0.5:
        lines.append(
            f"Inter-head KEY cosine similarity is {correlation_stats['avg_key_cosine']:.4f} "
            f"(>0.5), suggesting meaningful cross-head redundancy exists."
        )
    elif correlation_stats["avg_key_cosine"] > 0.1:
        lines.append(
            f"Inter-head KEY cosine similarity is {correlation_stats['avg_key_cosine']:.4f} "
            f"(moderate), suggesting partial cross-head redundancy."
        )
    else:
        lines.append(
            f"Inter-head KEY cosine similarity is {correlation_stats['avg_key_cosine']:.4f} "
            f"(low), suggesting heads are largely independent."
        )

    lines.append("")

    if correlation_stats["avg_var_ratio_keys"] < 0.8:
        lines.append(
            f"Delta variance ratio is {correlation_stats['avg_var_ratio_keys']:.4f} (<0.8), "
            f"confirming deltas need fewer bits than absolute values."
        )
    else:
        lines.append(
            f"Delta variance ratio is {correlation_stats['avg_var_ratio_keys']:.4f} (>=0.8), "
            f"suggesting deltas are nearly as complex as original vectors."
        )

    lines.append("")

    # Effective rates table
    lines.extend([
        "### Effective Bit Rates",
        "",
        "| Config | Formula | Bits/elem | Compression |",
        "|--------|---------|-----------|-------------|",
        "| Uniform 3-bit | 3 | 3.00 | 5.3x |",
        "| Uniform 2-bit | 2 | 2.00 | 8.0x |",
        "| Cross-head 3+1 (8 heads) | (3+7*1)/8 | 1.25 | 12.8x |",
        "| Cross-head 3+2 (8 heads) | (3+7*2)/8 | 2.13 | 7.5x |",
        "| Cross-head 2+1 (8 heads) | (2+7*1)/8 | 1.13 | 14.2x |",
        "",
    ])

    # Synthetic 8-head results
    if synthetic_results:
        lines.extend([
            "## Synthetic 8-Head Experiment",
            "",
            "Simulates MHA with 8 heads at controlled inter-head correlation levels.",
            "Base vectors from real Qwen2.5-3B KV cache, with added noise to control correlation.",
            "",
        ])
        for level_name, level_results in synthetic_results.items():
            lines.extend([
                f"### {level_name}",
                "",
                "| Config | Eff. Bits | Comp. Ratio | Mean Cos | Top-5 Attn | Attn r |",
                "|--------|-----------|-------------|----------|------------|--------|",
            ])
            for name, metrics in level_results.items():
                lines.append(
                    f"| {name} | {metrics['effective_bits']:.2f} | "
                    f"{metrics['compression_ratio']:.1f}x | "
                    f"{metrics['mean_cosine_sim']:.4f} | "
                    f"{metrics['top5_attention_match']:.4f} | "
                    f"{metrics['attention_score_pearson_r']:.4f} |"
                )
            lines.append("")

    # Conclusion
    lines.extend([
        "## Conclusion",
        "",
    ])

    # Determine if cross-head delta compression is viable
    best_cross = None
    for layer_results in compression_results.values():
        for name, metrics in layer_results.items():
            if "Cross" in name:
                if best_cross is None or metrics["top5_attention_match"] > best_cross[1]["top5_attention_match"]:
                    best_cross = (name, metrics)

    if best_cross:
        name, metrics = best_cross
        if metrics["top5_attention_match"] > 0.90:
            lines.append(
                f"**BREAKTHROUGH**: {name} achieves {metrics['top5_attention_match']:.1%} "
                f"top-5 attention match at {metrics['compression_ratio']:.1f}x compression "
                f"({metrics['effective_bits']:.2f} bits/element)."
            )
        elif metrics["top5_attention_match"] > 0.70:
            lines.append(
                f"**PROMISING**: {name} achieves {metrics['top5_attention_match']:.1%} "
                f"top-5 attention match at {metrics['compression_ratio']:.1f}x compression. "
                f"Quality is usable but not yet matching uniform 3-bit."
            )
        else:
            lines.append(
                f"**NEGATIVE**: {name} achieves only {metrics['top5_attention_match']:.1%} "
                f"top-5 attention match. Cross-head delta compression degrades quality "
                f"beyond acceptable levels at this delta bit-width."
            )

    return "\n".join(lines)


def main():
    model, tokenizer = load_model()

    print("\n" + "=" * 70)
    print("  CROSS-HEAD DELTA COMPRESSION EXPERIMENT")
    print("  Hypothesis: inter-head correlation enables 10x+ compression")
    print("=" * 70)

    # Step 1: Extract KV caches from a longer prompt
    long_prompt = (
        "The history of artificial intelligence began in the 1950s when pioneers "
        "like Alan Turing, John McCarthy, and Marvin Minsky laid the theoretical "
        "foundations. Turing proposed the famous Turing test in 1950, while McCarthy "
        "coined the term 'artificial intelligence' at the Dartmouth Conference in 1956. "
        "Early AI research focused on symbolic reasoning and problem solving. "
        "The field experienced several AI winters where funding dried up due to "
        "unmet expectations. The resurgence came with machine learning approaches, "
        "particularly deep learning and neural networks in the 2010s."
    )
    print(f"\n  Extracting KV cache from prompt ({len(long_prompt.split())} words)...")
    kv_by_layer = extract_kv_cache(model, tokenizer, long_prompt)
    print(f"  Extracted {len(kv_by_layer)} layers")

    # Step 2: Measure inter-head correlation
    correlation_stats = measure_all_correlations(kv_by_layer)

    # Step 3: Run compression quality comparison
    compression_results = run_compression_comparison(kv_by_layer)

    # Free KV cache memory
    del kv_by_layer
    gc.collect()
    torch.cuda.empty_cache()

    # Step 4: Synthetic multi-head experiment
    # Qwen2.5-3B uses GQA with only 2 KV heads.
    # Simulate an 8-head MHA scenario by replicating + perturbing real KV vectors
    # to test the approach at scale.
    print("\n" + "=" * 70)
    print("  SYNTHETIC 8-HEAD EXPERIMENT")
    print("  (Simulating MHA from real KV data with controlled correlation)")
    print("=" * 70)

    synthetic_results = {}
    for corr_level, noise_scale in [("high_corr_0.95", 0.05), ("medium_corr_0.7", 0.3), ("low_corr_0.3", 0.7)]:
        # Reload KV since we freed it
        kv_by_layer_synth = extract_kv_cache(model, tokenizer, long_prompt)
        mid = len(kv_by_layer_synth) // 2
        real_keys, _ = kv_by_layer_synth[mid]
        # real_keys: [1, 2, seq, 128] -> take head 0 as base, create 8 heads
        base = real_keys[:, 0:1, :, :]  # [1, 1, seq, 128]
        torch.manual_seed(42)
        noise = noise_scale * base.std() * torch.randn(1, 8, base.shape[2], 128, device="cuda")
        synth_keys = base.expand(-1, 8, -1, -1) + noise  # [1, 8, seq, 128]

        print(f"\n  {corr_level}: base + {noise_scale}*std noise")
        corr_stats = measure_inter_head_correlation(synth_keys)
        print(f"    Mean cosine: {corr_stats['mean_cosine']:.4f}")
        print(f"    Mean delta var ratio: "
              f"{sum(s['variance_ratio'] for s in corr_stats['anchor_delta_stats']) / len(corr_stats['anchor_delta_stats']):.4f}")

        queries = torch.randn(1, 8, 4, 128, device="cuda")
        level_results = {}
        for name, method, params in [
            ("Uniform 3-bit", "uniform", {"bits": 3}),
            ("Uniform 2-bit", "uniform", {"bits": 2}),
            ("Cross-head 3+1", "cross", {"anchor_bits": 3, "delta_bits": 1}),
            ("Cross-head 3+2", "cross", {"anchor_bits": 3, "delta_bits": 2}),
        ]:
            if method == "uniform":
                quant = UniformQuantizer(d=128, num_heads=8, bits=params["bits"], device="cuda")
            else:
                quant = CrossHeadDeltaQuantizer(
                    d=128, num_heads=8,
                    anchor_bits=params["anchor_bits"],
                    delta_bits=params["delta_bits"],
                    device="cuda",
                )
            recon = quant.quantize_dequantize(synth_keys)
            rq = evaluate_reconstruction_quality(synth_keys, recon)
            aq = evaluate_attention_quality(queries, synth_keys, recon)
            eff = quant.effective_bits_per_element()
            cr = quant.compression_ratio()
            print(f"    {name} ({eff:.2f}b, {cr:.1f}x): cos={rq['mean_cosine_sim']:.4f}, "
                  f"top5={aq['top5_attention_match']:.4f}, r={aq['attention_score_pearson_r']:.4f}")
            level_results[name] = {**rq, **aq, "effective_bits": eff, "compression_ratio": cr}
            del quant, recon

        synthetic_results[corr_level] = level_results
        del kv_by_layer_synth, synth_keys
        gc.collect()
        torch.cuda.empty_cache()

    # Step 5: Generation quality test
    generation_results = run_generation_test(model, tokenizer)

    # Step 6: Write results
    results_dir = os.path.join(REPO_ROOT, "benchmarks", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "cross_head_results.md")

    report = format_results_markdown(
        correlation_stats, compression_results, generation_results,
        synthetic_results=synthetic_results,
    )
    with open(results_path, "w") as f:
        f.write(report)

    print(f"\n  Results saved to {results_path}")
    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
