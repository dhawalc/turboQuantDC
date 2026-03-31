"""Combined Stack Experiment -- finding maximum compression with clean generation.

Tests every combination of TurboQuantDC breakthroughs:
1. FP16 anchor layers (every 12th layer)
2. ResidualQuant (direct residual signs instead of QJL)
3. Asymmetric K/V (keys at higher bits, values at lower)
4. Residual windowing (last N tokens at FP16)

Goal: find the MAXIMUM compression ratio that still produces coherent,
factually correct generation on Qwen2.5-3B-Instruct (4-bit weights).

Usage:
    cd /home/dhawal/turboQuantDC && python benchmarks/combined_stack_experiment.py
"""

from __future__ import annotations

import gc
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from turboquantdc.ultimate_cache import UltimateCache
from turboquantdc.adaptive_hf_cache import AdaptiveHFCache
from turboquantdc.hf_integration import TurboQuantCache

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
NUM_LAYERS = 36
MAX_NEW_TOKENS = 80
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
    ["layer", "neuron", "learn", "network", "input", "output", "weight", "node", "connect"],
    ["def", "factorial", "return"],
]


# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------

@dataclass
class Config:
    name: str
    short: str
    key_bits: int
    val_bits: int
    anchor_interval: int  # 0 = no anchors
    fp16_window: int
    use_residual_quant: bool
    is_fp16_baseline: bool = False
    expected_ratio: str = ""


CONFIGS = [
    Config(
        name="FP16 baseline",
        short="fp16",
        key_bits=16, val_bits=16,
        anchor_interval=0, fp16_window=0,
        use_residual_quant=False,
        is_fp16_baseline=True,
        expected_ratio="1.0x",
    ),
    Config(
        name="Anchor-12 + MSE-4 keys + MSE-4 values",
        short="a12_k4_v4",
        key_bits=4, val_bits=4,
        anchor_interval=12, fp16_window=0,
        use_residual_quant=False,
        expected_ratio="~3.1x",
    ),
    Config(
        name="Anchor-12 + MSE-4 keys + MSE-2 values",
        short="a12_k4_v2",
        key_bits=4, val_bits=2,
        anchor_interval=12, fp16_window=0,
        use_residual_quant=False,
        expected_ratio="~4.0x",
    ),
    Config(
        name="Anchor-12 + ResQ-4 keys + MSE-2 values",
        short="a12_rq4_v2",
        key_bits=4, val_bits=2,
        anchor_interval=12, fp16_window=0,
        use_residual_quant=True,
        expected_ratio="~4.0x",
    ),
    Config(
        name="Anchor-12 + MSE-5 keys + MSE-2 values",
        short="a12_k5_v2",
        key_bits=5, val_bits=2,
        anchor_interval=12, fp16_window=0,
        use_residual_quant=False,
        expected_ratio="~3.5x",
    ),
    Config(
        name="Anchor-12 + ResQ-4 keys + MSE-2 vals + win128",
        short="a12_rq4_v2_w128",
        key_bits=4, val_bits=2,
        anchor_interval=12, fp16_window=128,
        use_residual_quant=True,
        expected_ratio="~3.8x",
    ),
    Config(
        name="No anchors + ResQ-4 keys + MSE-2 values",
        short="rq4_v2",
        key_bits=4, val_bits=2,
        anchor_interval=0, fp16_window=0,
        use_residual_quant=True,
        expected_ratio="~4.5x",
    ),
    Config(
        name="No anchors + MSE-5 keys + MSE-2 values",
        short="k5_v2",
        key_bits=5, val_bits=2,
        anchor_interval=0, fp16_window=0,
        use_residual_quant=False,
        expected_ratio="~3.8x",
    ),
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load model once for all experiments."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation + scoring
# ---------------------------------------------------------------------------

def generate_with_cache(model, tokenizer, prompt, cache=None):
    """Generate text with optional KV cache."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        kwargs = dict(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
        )
        if cache is not None:
            kwargs["past_key_values"] = cache
        out = model.generate(**kwargs)

    response = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response


def check_coherence(response: str, keywords: List[str]) -> Tuple[bool, str]:
    """Check if response is coherent and contains expected keywords."""
    response_lower = response.lower()
    words = response.split()

    # Check 1: Not too short
    if len(words) < 2:
        return False, "too_short"

    # Check 2: Not repetitive (same 3-gram repeated > 3 times)
    if len(words) >= 6:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        for tg in set(trigrams):
            if trigrams.count(tg) > 3:
                return False, "repetitive"

    # Check 3: Contains at least one expected keyword
    has_keyword = any(kw in response_lower for kw in keywords)

    return has_keyword, "correct" if has_keyword else "wrong_content"


# ---------------------------------------------------------------------------
# Create cache from config
# ---------------------------------------------------------------------------

def make_cache(cfg: Config):
    """Create the appropriate cache object for a configuration."""
    if cfg.is_fp16_baseline:
        return None  # FP16 baseline = no cache compression

    return UltimateCache(
        num_layers=NUM_LAYERS,
        key_bits=cfg.key_bits,
        val_bits=cfg.val_bits,
        anchor_interval=cfg.anchor_interval,
        fp16_window=cfg.fp16_window,
        use_residual_quant=cfg.use_residual_quant,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Run single config
# ---------------------------------------------------------------------------

def run_config(model, tokenizer, cfg: Config) -> Dict:
    """Run a single configuration across all prompts."""
    print(f"\n{'='*72}")
    print(f"  {cfg.name}")
    if not cfg.is_fp16_baseline:
        cache_sample = make_cache(cfg)
        print(f"  Config: {cache_sample.config_summary()}")
        print(f"  Theoretical compression: {cache_sample.theoretical_compression_ratio():.2f}x")
        del cache_sample
    else:
        print(f"  Config: FP16 baseline (no compression)")
    print(f"{'='*72}")

    correct = 0
    total = len(PROMPTS)
    results = []
    total_time = 0.0

    for i, (prompt, keywords) in enumerate(zip(PROMPTS, EXPECTED_KEYWORDS)):
        cache = make_cache(cfg)

        t0 = time.time()
        response = generate_with_cache(model, tokenizer, prompt, cache=cache)
        elapsed = time.time() - t0
        total_time += elapsed

        is_correct, status = check_coherence(response, keywords)
        if is_correct:
            correct += 1

        # Get actual compression ratio if cache was used
        actual_ratio = 1.0
        if cache is not None:
            try:
                savings = cache.memory_savings()
                actual_ratio = savings["overall_compression_ratio"]
            except Exception:
                pass

        results.append({
            "prompt": prompt,
            "response": response[:200],
            "correct": is_correct,
            "status": status,
            "time": elapsed,
            "compression_ratio": actual_ratio,
        })

        status_icon = "PASS" if is_correct else "FAIL"
        print(f"\n  [{status_icon}] Q: {prompt}")
        print(f"       A: {response[:200]}")
        if len(response) > 200:
            print(f"       ... ({len(response)} chars total)")
        print(f"       Status: {status}, Time: {elapsed:.2f}s, Ratio: {actual_ratio:.2f}x")

        del cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_ratio = sum(r["compression_ratio"] for r in results) / len(results)
    print(f"\n  SCORE: {correct}/{total} correct | Avg compression: {avg_ratio:.2f}x | Time: {total_time:.1f}s")

    return {
        "config": cfg,
        "score": correct,
        "total": total,
        "avg_ratio": avg_ratio,
        "results": results,
        "total_time": total_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model, tokenizer = load_model()

    print("\n" + "=" * 72)
    print("  COMBINED STACK EXPERIMENT: Maximum Compression with Clean Generation")
    print("  Testing all TurboQuantDC breakthroughs in combination")
    print("  Model: Qwen2.5-3B-Instruct (4-bit weights)")
    print("=" * 72)

    all_results = {}

    for cfg in CONFIGS:
        result = run_config(model, tokenizer, cfg)
        all_results[cfg.short] = result

    # ---- Summary table ----
    print("\n" + "=" * 72)
    print("  COMBINED STACK RESULTS")
    print("=" * 72)

    # Header
    print(f"\n  {'Config':<50} {'Score':>6} {'Ratio':>8} {'Quality':>8}")
    print(f"  {'-'*50} {'-'*6} {'-'*8} {'-'*8}")

    for cfg in CONFIGS:
        r = all_results[cfg.short]
        score = r["score"]
        total = r["total"]
        ratio = r["avg_ratio"]
        quality = "CLEAN" if score >= 4 else ("OK" if score >= 3 else "GARBLED")
        print(f"  {cfg.name:<50} {score}/{total:>3} {ratio:>7.2f}x {quality:>8}")

    # ---- Find best config ----
    print(f"\n  {'='*72}")
    print(f"  ANALYSIS")
    print(f"  {'='*72}")

    # Separate clean from garbled
    clean_configs = [
        (cfg.short, all_results[cfg.short])
        for cfg in CONFIGS
        if all_results[cfg.short]["score"] >= 4
    ]
    ok_configs = [
        (cfg.short, all_results[cfg.short])
        for cfg in CONFIGS
        if all_results[cfg.short]["score"] == 3
    ]
    garbled_configs = [
        (cfg.short, all_results[cfg.short])
        for cfg in CONFIGS
        if all_results[cfg.short]["score"] < 3
    ]

    if clean_configs:
        # Sort by compression ratio (highest first)
        clean_configs.sort(key=lambda x: x[1]["avg_ratio"], reverse=True)
        best_name, best_result = clean_configs[0]
        best_cfg = best_result["config"]
        print(f"\n  BEST CLEAN CONFIG (score >= 4/5):")
        print(f"    {best_cfg.name}")
        print(f"    Score: {best_result['score']}/{best_result['total']}")
        print(f"    Compression: {best_result['avg_ratio']:.2f}x")
        print(f"    Key bits: {best_cfg.key_bits}, Val bits: {best_cfg.val_bits}")
        print(f"    Anchors: every {best_cfg.anchor_interval} layers" if best_cfg.anchor_interval > 0 else "    Anchors: none")
        print(f"    ResidualQuant: {best_cfg.use_residual_quant}")
        print(f"    FP16 window: {best_cfg.fp16_window}")
    else:
        print("\n  No configs achieved clean generation (4/5+)")

    if ok_configs:
        ok_configs.sort(key=lambda x: x[1]["avg_ratio"], reverse=True)
        print(f"\n  HIGHEST COMPRESSION WITH OK QUALITY (3/5):")
        for name, r in ok_configs:
            cfg = r["config"]
            print(f"    {cfg.name}: {r['avg_ratio']:.2f}x ({r['score']}/{r['total']})")

    if garbled_configs:
        print(f"\n  GARBLED CONFIGS (< 3/5):")
        for name, r in garbled_configs:
            cfg = r["config"]
            print(f"    {cfg.name}: {r['avg_ratio']:.2f}x ({r['score']}/{r['total']})")

    # ---- Key insights ----
    print(f"\n  {'='*72}")
    print(f"  KEY INSIGHTS")
    print(f"  {'='*72}")

    fp16_score = all_results["fp16"]["score"]

    # Does asymmetric K/V help?
    if "a12_k4_v4" in all_results and "a12_k4_v2" in all_results:
        s44 = all_results["a12_k4_v4"]["score"]
        s42 = all_results["a12_k4_v2"]["score"]
        r44 = all_results["a12_k4_v4"]["avg_ratio"]
        r42 = all_results["a12_k4_v2"]["avg_ratio"]
        if s42 >= s44:
            print(f"\n  ASYMMETRIC K/V: 2-bit values match 4-bit values!")
            print(f"    K4/V4: {s44}/5, {r44:.2f}x  vs  K4/V2: {s42}/5, {r42:.2f}x")
            print(f"    -> Free compression by dropping V to 2-bit")
        else:
            print(f"\n  ASYMMETRIC K/V: 2-bit values degrade quality")
            print(f"    K4/V4: {s44}/5  vs  K4/V2: {s42}/5")

    # Does ResidualQuant help over MSE-only?
    if "a12_k4_v2" in all_results and "a12_rq4_v2" in all_results:
        s_mse = all_results["a12_k4_v2"]["score"]
        s_rq = all_results["a12_rq4_v2"]["score"]
        if s_rq > s_mse:
            print(f"\n  RESIDUALQUANT: Improves over MSE-only at same bits")
            print(f"    MSE-4: {s_mse}/5  vs  ResQ-4: {s_rq}/5")
        elif s_rq == s_mse:
            print(f"\n  RESIDUALQUANT: Matches MSE-only (no gain at 4-bit)")
            print(f"    MSE-4: {s_mse}/5  vs  ResQ-4: {s_rq}/5")
        else:
            print(f"\n  RESIDUALQUANT: Hurts at 4-bit (unexpected)")
            print(f"    MSE-4: {s_mse}/5  vs  ResQ-4: {s_rq}/5")

    # Can we remove anchors?
    if "a12_rq4_v2" in all_results and "rq4_v2" in all_results:
        s_anchor = all_results["a12_rq4_v2"]["score"]
        s_noanchor = all_results["rq4_v2"]["score"]
        r_anchor = all_results["a12_rq4_v2"]["avg_ratio"]
        r_noanchor = all_results["rq4_v2"]["avg_ratio"]
        if s_noanchor >= s_anchor:
            print(f"\n  ANCHORS: Removing anchors works! Higher compression, same quality")
            print(f"    With anchors: {s_anchor}/5, {r_anchor:.2f}x")
            print(f"    No anchors:   {s_noanchor}/5, {r_noanchor:.2f}x")
        else:
            print(f"\n  ANCHORS: Still needed for quality")
            print(f"    With anchors: {s_anchor}/5, {r_anchor:.2f}x")
            print(f"    No anchors:   {s_noanchor}/5, {r_noanchor:.2f}x")

    # Does windowing help?
    if "a12_rq4_v2" in all_results and "a12_rq4_v2_w128" in all_results:
        s_nowin = all_results["a12_rq4_v2"]["score"]
        s_win = all_results["a12_rq4_v2_w128"]["score"]
        if s_win > s_nowin:
            print(f"\n  FP16 WINDOW: Windowing improves quality!")
            print(f"    No window: {s_nowin}/5  vs  128-token window: {s_win}/5")
        elif s_win == s_nowin:
            print(f"\n  FP16 WINDOW: No quality impact (already clean)")
        else:
            print(f"\n  FP16 WINDOW: Unexpectedly hurts quality")

    # Does 5-bit keys help?
    if "a12_k5_v2" in all_results and "a12_k4_v2" in all_results:
        s5 = all_results["a12_k5_v2"]["score"]
        s4 = all_results["a12_k4_v2"]["score"]
        r5 = all_results["a12_k5_v2"]["avg_ratio"]
        r4 = all_results["a12_k4_v2"]["avg_ratio"]
        if s5 > s4:
            print(f"\n  5-BIT KEYS: More key bits improve quality")
            print(f"    K4/V2: {s4}/5, {r4:.2f}x  vs  K5/V2: {s5}/5, {r5:.2f}x")
        else:
            print(f"\n  5-BIT KEYS: No improvement over 4-bit keys")
            print(f"    K4/V2: {s4}/5, {r4:.2f}x  vs  K5/V2: {s5}/5, {r5:.2f}x")

    print(f"\n  {'='*72}")
    print(f"  CONCLUSION")
    print(f"  {'='*72}")

    if clean_configs:
        best_name, best_result = clean_configs[0]
        best_cfg = best_result["config"]
        print(f"\n  Maximum clean compression: {best_result['avg_ratio']:.2f}x")
        print(f"  Config: {best_cfg.name}")
        print(f"  Score: {best_result['score']}/{best_result['total']} on standard prompts")
    else:
        print("\n  No configuration achieved clean generation.")
        print("  The baseline anchor-12 + MSE-4 approach remains the best option.")

    print()


if __name__ == "__main__":
    main()
