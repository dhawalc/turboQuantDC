"""
Cayley-Parameterized Full Rotation Benchmark
==============================================

Can a learned FULL d x d rotation (8128 DOF) beat WHT (896 DOF)?

The Givens block-diagonal approach (64 DOF) already proved that block-diagonal
rotations cannot beat WHT. The Cayley map parameterizes full SO(d) rotations
via a skew-symmetric matrix, giving d*(d-1)/2 = 8128 free parameters for d=128.

Configurations tested at 3-bit on Qwen2.5-3B-Instruct:
  1. WHT + mean-removal (current best baseline)
  2. Learned Givens + mean-removal (block-diagonal, 64 DOF)
  3. Cayley from identity + mean-removal (cold start, 8128 DOF)
  4. Cayley from WHT + mean-removal (warm start, 8128 DOF)

Sweeps:
  - Steps: 25, 50, 100, 200
  - Transfer: calibrate on prompt A, evaluate on prompt B
  - Warm start (WHT init) vs cold start (identity init)

Usage:
    python benchmarks/cayley_quant_benchmark.py
"""

import sys
import os
import time
import math
import json
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/dhawal/turboQuantDC")

from turboquantdc.cayley_quant import CayleyLearnedQuantizer, CayleyRotation
from turboquantdc.learned_quant import LearnedQuantizer
from turboquantdc.residual_quant import ResidualQuantEstimator
from turboquantdc.attention_optimal import compute_attention_scores, attention_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"
SEED = 42
HEAD_DIM = 128
BITS = 3

RESULTS_DIR = Path("/home/dhawal/turboQuantDC/benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Prompts
# ============================================================================

PROMPT_A = (
    "You are a world-class computer scientist giving a comprehensive lecture on "
    "the history of computing, starting from Charles Babbage's Analytical Engine "
    "through Alan Turing's theoretical foundations, the development of ENIAC, "
    "the transistor revolution, the birth of the internet at ARPANET, the rise "
    "of personal computing with Apple and IBM, the open source movement with "
    "Linux, the mobile revolution with iPhone, cloud computing with AWS, and "
    "finally the current AI revolution with large language models. Cover the "
    "key innovations, the people behind them, and the societal impact of each "
    "era in detail."
)

PROMPT_B = (
    "Explain quantum computing from first principles, including superposition, "
    "entanglement, quantum gates, error correction, and the current landscape "
    "of quantum hardware from superconducting qubits to trapped ions. Compare "
    "quantum supremacy claims and practical near-term applications in chemistry "
    "simulation, optimization, and cryptography."
)


# ============================================================================
# Model loading and KV extraction
# ============================================================================

def load_model_and_extract_kv(prompt_text: str, n_tokens: int = 256):
    """Load Qwen2.5-3B in BnB 4-bit, extract Q/K per layer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=CACHE_DIR,
        dtype=torch.float16,
    )
    model.eval()

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        max_length=n_tokens,
        truncation=True,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]
    gen_target = max(n_tokens, prompt_len + 32)
    gen_tokens = gen_target - prompt_len

    print(f"  Prompt tokens: {prompt_len}, generating {gen_tokens} more...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )

    past_kv = outputs.past_key_values
    total_tokens = outputs.sequences.shape[1]
    print(f"  Total tokens in cache: {total_tokens}")

    layer_data = {}

    if hasattr(past_kv, "layers"):
        # New DynamicCache with .layers[i].keys / .values
        n_layers = len(past_kv.layers)
        for layer_idx in range(n_layers):
            layer = past_kv.layers[layer_idx]
            if not hasattr(layer, "keys") or layer.keys.numel() == 0:
                continue
            K_all = layer.keys  # (batch, n_kv_heads, seq, head_dim)
            K = K_all[0, 0].float().to(DEVICE)
            Q = K.clone()
            layer_data[layer_idx] = {"Q": Q, "K": K}
    elif hasattr(past_kv, "key_cache"):
        n_layers = len(past_kv.key_cache)
        for layer_idx in range(n_layers):
            K_all = past_kv.key_cache[layer_idx]
            K = K_all[0, 0].float().to(DEVICE)
            Q = K.clone()
            layer_data[layer_idx] = {"Q": Q, "K": K}
    elif isinstance(past_kv, (list, tuple)):
        n_layers = len(past_kv)
        for layer_idx in range(n_layers):
            K_all = past_kv[layer_idx][0]
            K = K_all[0, 0].float().to(DEVICE)
            Q = K.clone()
            layer_data[layer_idx] = {"Q": Q, "K": K}
    else:
        raise RuntimeError(f"Unsupported cache type: {type(past_kv)}")

    return layer_data, model, tokenizer


# ============================================================================
# Evaluation helper
# ============================================================================

def evaluate_config(Q, K, quantizer_fn):
    """Evaluate a quantizer on real Q/K, return attention metrics."""
    attn_true = compute_attention_scores(Q, K)

    t0 = time.perf_counter()
    K_quant = quantizer_fn(K)
    quant_time = time.perf_counter() - t0

    attn_quant = compute_attention_scores(Q, K_quant)
    metrics = attention_metrics(attn_true, attn_quant)
    metrics["quant_time_ms"] = quant_time * 1000
    return metrics


# ============================================================================
# Baseline: WHT + mean-removal
# ============================================================================

def run_wht_baseline(Q, K):
    """WHT + mean-removal (current best fixed approach)."""
    rq = ResidualQuantEstimator(
        d=HEAD_DIM, bits=BITS, seed=SEED, device=DEVICE,
        center_before_quantize=True,
    )
    def quant_fn(K):
        comp = rq.quantize(K)
        return rq.dequantize(comp)
    return evaluate_config(Q, K, quant_fn)


# ============================================================================
# Learned Givens baseline
# ============================================================================

def run_learned_givens(Q, K, steps=100, lr=0.01):
    """Learned Givens rotation + mean-removal (64 DOF)."""
    lq = LearnedQuantizer(
        d=HEAD_DIM, bits=BITS, center=True, seed=SEED, device=DEVICE,
    )
    t0 = time.perf_counter()
    losses = lq.calibrate(Q, K, lr=lr, steps=steps)
    cal_time = time.perf_counter() - t0

    def quant_fn(K):
        return lq.forward(K).detach()
    metrics = evaluate_config(Q, K, quant_fn)
    metrics["calibration_time_ms"] = cal_time * 1000
    metrics["initial_kl"] = losses[0]
    metrics["final_kl"] = losses[-1]
    metrics["best_kl"] = min(losses)
    metrics["dof"] = HEAD_DIM // 2
    return metrics


# ============================================================================
# Cayley learned rotation
# ============================================================================

def run_cayley(Q, K, steps=100, lr=0.005, init_from_wht=False, verbose=False):
    """Cayley full rotation + mean-removal (8128 DOF)."""
    lq = CayleyLearnedQuantizer(
        d=HEAD_DIM, bits=BITS, center=True, seed=SEED, device=DEVICE,
        init_from_wht=init_from_wht,
    )
    t0 = time.perf_counter()
    losses = lq.calibrate(Q, K, lr=lr, steps=steps, verbose=verbose)
    cal_time = time.perf_counter() - t0

    def quant_fn(K):
        return lq.forward(K).detach()
    metrics = evaluate_config(Q, K, quant_fn)
    metrics["calibration_time_ms"] = cal_time * 1000
    metrics["initial_kl"] = losses[0]
    metrics["final_kl"] = losses[-1]
    metrics["best_kl"] = min(losses)
    metrics["dof"] = HEAD_DIM * (HEAD_DIM - 1) // 2
    return metrics


# ============================================================================
# Step sweep
# ============================================================================

def sweep_steps(Q, K, step_counts=[25, 50, 100, 200], init_from_wht=False):
    """Test different calibration step counts."""
    results = {}
    for steps in step_counts:
        label = f"cayley_{'wht' if init_from_wht else 'identity'}_{steps}steps"
        print(f"    {label}...")
        metrics = run_cayley(Q, K, steps=steps, init_from_wht=init_from_wht)
        results[label] = metrics
    return results


# ============================================================================
# Transfer test
# ============================================================================

def extract_kv_from_model(model, tokenizer, prompt_text, n_tokens=256):
    """Extract KV cache from an already-loaded model."""
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        max_length=n_tokens,
        truncation=True,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]
    gen_target = max(n_tokens, prompt_len + 32)
    gen_tokens = gen_target - prompt_len

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )

    past_kv = outputs.past_key_values
    layer_data = {}

    if hasattr(past_kv, "layers"):
        n_layers = len(past_kv.layers)
        for layer_idx in range(n_layers):
            layer = past_kv.layers[layer_idx]
            if not hasattr(layer, "keys") or layer.keys.numel() == 0:
                continue
            K_all = layer.keys
            K = K_all[0, 0].float().to(DEVICE)
            Q = K.clone()
            layer_data[layer_idx] = {"Q": Q, "K": K}
    elif hasattr(past_kv, "key_cache"):
        n_layers = len(past_kv.key_cache)
        for layer_idx in range(n_layers):
            K_all = past_kv.key_cache[layer_idx]
            K = K_all[0, 0].float().to(DEVICE)
            Q = K.clone()
            layer_data[layer_idx] = {"Q": Q, "K": K}

    return layer_data


def run_transfer_test(model, tokenizer):
    """Calibrate on prompt A, test on prompt B."""
    print("\n  Extracting KV from prompt A (calibration)...")
    layer_data_a = extract_kv_from_model(model, tokenizer, PROMPT_A, n_tokens=256)

    print("  Extracting KV from prompt B (evaluation)...")
    layer_data_b = extract_kv_from_model(model, tokenizer, PROMPT_B, n_tokens=256)

    test_layers = [0, 8, 16, max(layer_data_a.keys())]
    test_layers = sorted(set(l for l in test_layers if l in layer_data_a and l in layer_data_b))

    results = {}
    for layer_idx in test_layers:
        Q_a, K_a = layer_data_a[layer_idx]["Q"], layer_data_a[layer_idx]["K"]
        Q_b, K_b = layer_data_b[layer_idx]["Q"], layer_data_b[layer_idx]["K"]

        # Cayley calibrated on A, tested on B
        lq_cayley = CayleyLearnedQuantizer(
            d=HEAD_DIM, bits=BITS, center=True, seed=SEED, device=DEVICE,
            init_from_wht=True,
        )
        lq_cayley.calibrate(Q_a, K_a, lr=0.005, steps=100)

        # Reset running mean for evaluation on B
        lq_cayley.running_mean.zero_()
        lq_cayley.running_count.zero_()
        lq_cayley._update_running_mean(K_b)

        attn_true_b = compute_attention_scores(Q_b, K_b)
        K_b_quant = lq_cayley.forward(K_b).detach()
        attn_quant_b = compute_attention_scores(Q_b, K_b_quant)
        transfer_metrics = attention_metrics(attn_true_b, attn_quant_b)

        # WHT baseline on B (no learning needed)
        rq_wht = ResidualQuantEstimator(
            d=HEAD_DIM, bits=BITS, seed=SEED, device=DEVICE,
            center_before_quantize=True,
        )
        comp_wht = rq_wht.quantize(K_b)
        K_b_wht = rq_wht.dequantize(comp_wht)
        attn_wht_b = compute_attention_scores(Q_b, K_b_wht)
        wht_metrics = attention_metrics(attn_true_b, attn_wht_b)

        # Cayley calibrated directly on B (upper bound)
        lq_direct = CayleyLearnedQuantizer(
            d=HEAD_DIM, bits=BITS, center=True, seed=SEED, device=DEVICE,
            init_from_wht=True,
        )
        lq_direct.calibrate(Q_b, K_b, lr=0.005, steps=100)
        K_b_direct = lq_direct.forward(K_b).detach()
        attn_direct_b = compute_attention_scores(Q_b, K_b_direct)
        direct_metrics = attention_metrics(attn_true_b, attn_direct_b)

        results[f"layer_{layer_idx}"] = {
            "wht_baseline": wht_metrics,
            "cayley_transfer_from_A": transfer_metrics,
            "cayley_calibrated_on_B": direct_metrics,
        }

    return results


# ============================================================================
# Main
# ============================================================================

def format_m(m, keys=None):
    """Format metrics dict as a string."""
    if keys is None:
        keys = ["cosine_sim", "top1_match", "top5_match", "kl_div"]
    parts = []
    for k in keys:
        if k in m:
            if "kl" in k:
                parts.append(f"{k}={m[k]:.6f}")
            elif "time" in k:
                parts.append(f"{k}={m[k]:.1f}ms")
            else:
                parts.append(f"{k}={m[k]:.4f}")
    return ", ".join(parts)


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print("=" * 70)
    print("CAYLEY FULL ROTATION BENCHMARK")
    print("Can learned full d x d rotation (8128 DOF) beat WHT (896 DOF)?")
    print("=" * 70)

    # ---- Step 1: Load model and extract KV caches ----
    print("\nStep 1: Loading model and extracting KV caches...")
    layer_data, model, tokenizer = load_model_and_extract_kv(PROMPT_A, n_tokens=256)
    n_layers = len(layer_data)
    print(f"  Extracted {n_layers} layers, head_dim={HEAD_DIM}")

    # Test on representative layers
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    test_layers = sorted(set(l for l in test_layers if l in layer_data))

    all_results = {
        "metadata": {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "bits": BITS,
            "head_dim": HEAD_DIM,
            "n_layers": n_layers,
            "test_layers": test_layers,
            "cayley_dof": HEAD_DIM * (HEAD_DIM - 1) // 2,
            "givens_dof": HEAD_DIM // 2,
            "wht_effective_dof": HEAD_DIM * int(math.log2(HEAD_DIM)),
        },
        "per_layer": {},
        "aggregated": {},
        "step_sweep": {},
        "transfer": {},
    }

    # ---- Step 2: Per-layer comparison ----
    print(f"\nStep 2: Per-layer comparison on {len(test_layers)} layers...")
    print(f"  {'Layer':>5} | {'Method':>25} | {'Cosine':>8} {'Top-1':>7} {'Top-5':>7} {'KL':>10}")
    print("  " + "-" * 75)

    agg = {
        "wht": {"cos": [], "t1": [], "t5": [], "kl": []},
        "givens": {"cos": [], "t1": [], "t5": [], "kl": []},
        "cayley_cold": {"cos": [], "t1": [], "t5": [], "kl": []},
        "cayley_warm": {"cos": [], "t1": [], "t5": [], "kl": []},
    }

    for li in test_layers:
        Q = layer_data[li]["Q"]
        K = layer_data[li]["K"]
        layer_results = {}

        # 1. WHT + mean-removal baseline
        print(f"  {li:>5} | {'WHT + mean-removal':>25}", end="", flush=True)
        m = run_wht_baseline(Q, K)
        layer_results["wht_mean"] = m
        print(f" | {m['cosine_sim']:>8.4f} {m['top1_match']:>7.1%} {m['top5_match']:>7.1%} {m['kl_div']:>10.6f}")
        agg["wht"]["cos"].append(m["cosine_sim"])
        agg["wht"]["t1"].append(m["top1_match"])
        agg["wht"]["t5"].append(m["top5_match"])
        agg["wht"]["kl"].append(m["kl_div"])

        # 2. Learned Givens + mean-removal
        print(f"  {'':>5} | {'Givens learned (64 DOF)':>25}", end="", flush=True)
        m = run_learned_givens(Q, K, steps=100)
        layer_results["givens_learned"] = m
        print(f" | {m['cosine_sim']:>8.4f} {m['top1_match']:>7.1%} {m['top5_match']:>7.1%} {m['kl_div']:>10.6f}  cal={m['calibration_time_ms']:.0f}ms")
        agg["givens"]["cos"].append(m["cosine_sim"])
        agg["givens"]["t1"].append(m["top1_match"])
        agg["givens"]["t5"].append(m["top5_match"])
        agg["givens"]["kl"].append(m["kl_div"])

        # 3. Cayley from identity (cold start)
        print(f"  {'':>5} | {'Cayley cold (8128 DOF)':>25}", end="", flush=True)
        m = run_cayley(Q, K, steps=100, init_from_wht=False)
        layer_results["cayley_cold"] = m
        print(f" | {m['cosine_sim']:>8.4f} {m['top1_match']:>7.1%} {m['top5_match']:>7.1%} {m['kl_div']:>10.6f}  cal={m['calibration_time_ms']:.0f}ms")
        agg["cayley_cold"]["cos"].append(m["cosine_sim"])
        agg["cayley_cold"]["t1"].append(m["top1_match"])
        agg["cayley_cold"]["t5"].append(m["top5_match"])
        agg["cayley_cold"]["kl"].append(m["kl_div"])

        # 4. Cayley from WHT (warm start)
        print(f"  {'':>5} | {'Cayley WHT-warm (8128 DOF)':>25}", end="", flush=True)
        m = run_cayley(Q, K, steps=100, init_from_wht=True)
        layer_results["cayley_warm"] = m
        print(f" | {m['cosine_sim']:>8.4f} {m['top1_match']:>7.1%} {m['top5_match']:>7.1%} {m['kl_div']:>10.6f}  cal={m['calibration_time_ms']:.0f}ms")
        agg["cayley_warm"]["cos"].append(m["cosine_sim"])
        agg["cayley_warm"]["t1"].append(m["top1_match"])
        agg["cayley_warm"]["t5"].append(m["top5_match"])
        agg["cayley_warm"]["kl"].append(m["kl_div"])

        print()
        all_results["per_layer"][str(li)] = layer_results

    # ---- Step 3: Aggregated results ----
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS (averaged across test layers)")
    print("=" * 70)
    print(f"  {'Method':>28} | {'Cosine':>8} {'Top-1':>7} {'Top-5':>7} {'KL':>10}")
    print("  " + "-" * 65)

    for name, label in [
        ("wht", "WHT + mean-removal"),
        ("givens", "Givens learned (64 DOF)"),
        ("cayley_cold", "Cayley cold (8128 DOF)"),
        ("cayley_warm", "Cayley WHT-warm (8128 DOF)"),
    ]:
        d = agg[name]
        avg_cos = sum(d["cos"]) / len(d["cos"])
        avg_t1 = sum(d["t1"]) / len(d["t1"])
        avg_t5 = sum(d["t5"]) / len(d["t5"])
        avg_kl = sum(d["kl"]) / len(d["kl"])
        print(f"  {label:>28} | {avg_cos:>8.4f} {avg_t1:>7.1%} {avg_t5:>7.1%} {avg_kl:>10.6f}")
        all_results["aggregated"][name] = {
            "cosine_sim": avg_cos,
            "top1_match": avg_t1,
            "top5_match": avg_t5,
            "kl_div": avg_kl,
        }

    # ---- Step 4: Step sweep ----
    print("\n" + "=" * 70)
    print("STEP SWEEP (Cayley WHT-warm, layer 8)")
    print("=" * 70)

    test_layer = min(8, max(layer_data.keys()))
    Q_sweep = layer_data[test_layer]["Q"]
    K_sweep = layer_data[test_layer]["K"]

    step_counts = [25, 50, 100, 200]
    print(f"  {'Steps':>6} | {'Cosine':>8} {'Top-1':>7} {'Top-5':>7} {'KL':>10} {'Cal.Time':>10}")
    print("  " + "-" * 55)

    for steps in step_counts:
        m = run_cayley(Q_sweep, K_sweep, steps=steps, init_from_wht=True)
        print(f"  {steps:>6} | {m['cosine_sim']:>8.4f} {m['top1_match']:>7.1%} {m['top5_match']:>7.1%} {m['kl_div']:>10.6f} {m['calibration_time_ms']:>8.0f}ms")
        all_results["step_sweep"][f"{steps}_steps"] = m

    # Also test cold-start step sweep for comparison
    print(f"\n  Cold-start comparison:")
    print(f"  {'Steps':>6} | {'Cosine':>8} {'Top-1':>7} {'Top-5':>7} {'KL':>10} {'Cal.Time':>10}")
    print("  " + "-" * 55)
    for steps in step_counts:
        m = run_cayley(Q_sweep, K_sweep, steps=steps, init_from_wht=False)
        print(f"  {steps:>6} | {m['cosine_sim']:>8.4f} {m['top1_match']:>7.1%} {m['top5_match']:>7.1%} {m['kl_div']:>10.6f} {m['calibration_time_ms']:>8.0f}ms")
        all_results["step_sweep"][f"cold_{steps}_steps"] = m

    # ---- Step 5: Transfer test ----
    print("\n" + "=" * 70)
    print("TRANSFER TEST: calibrate on A, evaluate on B")
    print("=" * 70)

    transfer_results = run_transfer_test(model, tokenizer)
    all_results["transfer"] = transfer_results

    for layer_key, layer_res in transfer_results.items():
        print(f"\n  {layer_key}:")
        for method, m in layer_res.items():
            print(f"    {method:>28}: cos={m['cosine_sim']:.4f} t1={m['top1_match']:.1%} t5={m['top5_match']:.1%} kl={m['kl_div']:.6f}")

    # ---- Save results ----
    json_path = RESULTS_DIR / "cayley_quant_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON results saved to {json_path}")

    # ---- Generate markdown report ----
    md_path = RESULTS_DIR / "cayley_quant_results.md"
    write_markdown_report(all_results, md_path)
    print(f"Markdown report saved to {md_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


def write_markdown_report(results, path):
    """Write results as markdown."""
    meta = results["metadata"]
    agg = results["aggregated"]

    lines = [
        "# Cayley Full Rotation Benchmark Results",
        "",
        f"**Model:** {meta['model']}",
        f"**Bits:** {meta['bits']}",
        f"**Head dim:** {meta['head_dim']}",
        f"**Layers tested:** {meta['test_layers']}",
        "",
        "## DOF Comparison",
        "",
        f"| Rotation | DOF | % of max |",
        f"|----------|-----|----------|",
        f"| Givens block-diagonal | {meta['givens_dof']} | {meta['givens_dof']*100/meta['cayley_dof']:.1f}% |",
        f"| WHT (butterfly) | {meta['wht_effective_dof']} | {meta['wht_effective_dof']*100/meta['cayley_dof']:.1f}% |",
        f"| Cayley full SO(d) | {meta['cayley_dof']} | 100% |",
        "",
        "## Aggregated Results (3-bit, averaged across test layers)",
        "",
        "| Method | Cosine | Top-1 | Top-5 | KL |",
        "|--------|--------|-------|-------|-----|",
    ]

    for name, label in [
        ("wht", "WHT + mean-removal"),
        ("givens", "Givens learned (64 DOF)"),
        ("cayley_cold", "Cayley cold (8128 DOF)"),
        ("cayley_warm", "Cayley WHT-warm (8128 DOF)"),
    ]:
        if name in agg:
            a = agg[name]
            lines.append(
                f"| {label} | {a['cosine_sim']:.4f} | {a['top1_match']:.1%} | "
                f"{a['top5_match']:.1%} | {a['kl_div']:.6f} |"
            )

    lines.extend([
        "",
        "## Step Sweep (Cayley WHT-warm, single layer)",
        "",
        "| Steps | Cosine | Top-1 | Top-5 | KL | Cal.Time |",
        "|-------|--------|-------|-------|-----|----------|",
    ])

    for key in ["25_steps", "50_steps", "100_steps", "200_steps"]:
        if key in results.get("step_sweep", {}):
            m = results["step_sweep"][key]
            lines.append(
                f"| {key.replace('_steps','')} | {m['cosine_sim']:.4f} | "
                f"{m['top1_match']:.1%} | {m['top5_match']:.1%} | "
                f"{m['kl_div']:.6f} | {m.get('calibration_time_ms', 0):.0f}ms |"
            )

    lines.extend([
        "",
        "## Transfer Test (calibrate on A, evaluate on B)",
        "",
    ])

    for layer_key, layer_res in results.get("transfer", {}).items():
        lines.append(f"### {layer_key}")
        lines.append("")
        lines.append("| Method | Cosine | Top-1 | Top-5 | KL |")
        lines.append("|--------|--------|-------|-------|-----|")
        for method, m in layer_res.items():
            lines.append(
                f"| {method} | {m['cosine_sim']:.4f} | {m['top1_match']:.1%} | "
                f"{m['top5_match']:.1%} | {m['kl_div']:.6f} |"
            )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
