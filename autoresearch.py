"""Autoresearch -- autonomous overnight configuration sweep for TurboQuantDC.

Inspired by Karpathy's autoresearch concept: one script, loads model ONCE,
sweeps hundreds of KV cache compression configurations, auto-scores each,
saves results after every round, and reports the Pareto frontier of
compression vs generation quality.

The search space covers:
    key_bits: [3, 4, 5, 6, 8]
    val_bits: [2, 3, 4]
    anchor_interval: [0, 6, 12, 18, 36]
    fp16_window: [0, 64, 128, 256]
    use_residual_quant: [True, False]
    mse_only: [True]  (QJL is dead; always MSE-only)
= 600 total configurations

Scoring uses 8 test prompts covering factual recall, math, code, and
reasoning.  Each prompt is scored 0-1 on factual accuracy (0.6 weight),
coherence/repetition (0.3 weight), and length sanity (0.1 weight).

Usage:
    cd /home/dhawal/turboQuantDC
    python autoresearch.py                          # 600 rounds
    python autoresearch.py --max-rounds 200         # first 200 (most promising)
    python autoresearch.py --resume                 # skip already-tested configs
    nohup python autoresearch.py > autoresearch.log 2>&1 &
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import torch

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
NUM_LAYERS = 36
MAX_NEW_TOKENS = 80
DO_SAMPLE = False

SEARCH_SPACE = {
    "key_bits": [3, 4, 5, 6, 8],
    "val_bits": [2, 3, 4],
    "anchor_interval": [0, 6, 12, 18, 36],
    "fp16_window": [0, 64, 128, 256],
    "use_residual_quant": [True, False],
    "mse_only": [True],
}

TEST_PROMPTS = [
    {
        "prompt": "What is the capital of Australia? Answer with just the city name:",
        "expected": ["Canberra"],
        "type": "factual",
    },
    {
        "prompt": "What is 15 + 27? Answer with just the number:",
        "expected": ["42"],
        "type": "math",
    },
    {
        "prompt": "Who wrote the novel 1984? Answer briefly:",
        "expected": ["George Orwell", "Orwell"],
        "type": "factual",
    },
    {
        "prompt": "What is the largest planet in our solar system? Answer briefly:",
        "expected": ["Jupiter"],
        "type": "factual",
    },
    {
        "prompt": "Write a Python function that returns the factorial of n:",
        "expected": ["def ", "factorial", "return"],
        "type": "code",
    },
    {
        "prompt": "Explain photosynthesis in one sentence:",
        "expected": ["light", "energy", "plant"],
        "type": "reasoning",
    },
    {
        "prompt": "What is the chemical formula for water?",
        "expected": ["H2O"],
        "type": "factual",
    },
    {
        "prompt": "List three primary colors:",
        "expected": ["red", "blue"],
        "type": "factual",
    },
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_response(prompt_config: Dict, response: str) -> float:
    """Score a single response from 0 to 1.

    Checks:
    1. Factual accuracy (0.6 weight): does the response contain expected keywords?
    2. Coherence (0.3 weight): is it free of degenerate repetition?
    3. Length (0.1 weight): is it a reasonable length?
    """
    score = 0.0

    # Factual accuracy (0.6 weight)
    expected = prompt_config["expected"]
    found = sum(1 for e in expected if e.lower() in response.lower())
    accuracy = found / len(expected)
    score += 0.6 * accuracy

    # Coherence (0.3 weight) -- penalize repetition
    words = response.split()
    if len(words) > 5:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        coherence = min(unique_ratio / 0.5, 1.0)  # 50%+ unique words = full marks
    else:
        coherence = 0.5  # Short responses are OK but not great
    score += 0.3 * coherence

    # Length (0.1 weight) -- penalize empty or extremely long
    if 3 < len(words) < 200:
        score += 0.1

    return round(score, 4)


def compute_compression_ratio(config: Dict) -> float:
    """Compute theoretical compression ratio for a configuration.

    Uses the UltimateCache theoretical formula without storing any data.
    """
    key_bits = config["key_bits"]
    val_bits = config["val_bits"]
    anchor_interval = config["anchor_interval"]
    fp16_window = config["fp16_window"]
    use_residual_quant = config["use_residual_quant"]

    if anchor_interval > 0 and anchor_interval < NUM_LAYERS:
        n_fp16 = len([i for i in range(NUM_LAYERS) if i % anchor_interval == 0])
    else:
        n_fp16 = 0
    n_comp = NUM_LAYERS - n_fp16

    # FP16 layers: 32 bits per coordinate (K+V each at 16-bit)
    fp16_cost = n_fp16 * 32.0

    # Compressed layers
    if use_residual_quant:
        comp_key = key_bits + 32 / 128.0  # MSE + signs + scale + norm overhead
    else:
        comp_key = key_bits + 16 / 128.0  # MSE + norm overhead

    comp_val = val_bits + 16 / 128.0
    comp_cost = n_comp * (comp_key + comp_val)

    # FP16 window adjustment (approximate for ~512 tokens)
    if fp16_window > 0 and n_comp > 0:
        window_frac = min(fp16_window / 512.0, 1.0)
        comp_cost_windowed = n_comp * (
            (1 - window_frac) * (comp_key + comp_val)
            + window_frac * 32
        )
        comp_cost = comp_cost_windowed

    total_cost = fp16_cost + comp_cost
    baseline = NUM_LAYERS * 32.0

    return round(baseline / total_cost, 2) if total_cost > 0 else 1.0


# ---------------------------------------------------------------------------
# Cache building
# ---------------------------------------------------------------------------

def build_cache(config: Dict):
    """Build the appropriate cache for a configuration.

    Uses UltimateCache which handles all combinations:
    anchor layers, residual quant, asymmetric K/V, FP16 windowing.
    """
    from turboquantdc.ultimate_cache import UltimateCache

    anchor_interval = config["anchor_interval"]
    key_bits = config["key_bits"]
    val_bits = config["val_bits"]
    fp16_window = config["fp16_window"]
    use_residual_quant = config["use_residual_quant"]

    cache = UltimateCache(
        num_layers=NUM_LAYERS,
        key_bits=key_bits,
        val_bits=val_bits,
        anchor_interval=anchor_interval,
        fp16_window=fp16_window,
        use_residual_quant=use_residual_quant,
        seed=42,
    )
    return cache


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_with_cache(model, tokenizer, prompt: str, cache=None) -> str:
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


# ---------------------------------------------------------------------------
# Score a full configuration
# ---------------------------------------------------------------------------

def score_config(
    model, tokenizer, config: Dict, cache=None, is_baseline: bool = False,
) -> Tuple[float, List[Dict], float]:
    """Score a full configuration across all test prompts.

    Returns:
        (total_score, per_prompt_details, compression_ratio)
    """
    per_prompt = []
    total_score = 0.0

    for prompt_cfg in TEST_PROMPTS:
        # Build a fresh cache for each prompt (autoregressive generation
        # accumulates state, so each prompt needs its own cache instance)
        if is_baseline:
            prompt_cache = None
        else:
            prompt_cache = build_cache(config)

        try:
            response = generate_with_cache(
                model, tokenizer, prompt_cfg["prompt"], cache=prompt_cache,
            )
        except Exception as e:
            response = f"[ERROR: {e}]"

        s = score_response(prompt_cfg, response)
        total_score += s

        per_prompt.append({
            "prompt": prompt_cfg["prompt"],
            "type": prompt_cfg["type"],
            "response": response[:300],
            "score": s,
        })

        del prompt_cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Normalize to 0-1
    total_score = round(total_score / len(TEST_PROMPTS), 4)

    compression = compute_compression_ratio(config) if not is_baseline else 1.0

    return total_score, per_prompt, compression


# ---------------------------------------------------------------------------
# Config generation -- priority-ordered
# ---------------------------------------------------------------------------

def config_key(config: Dict) -> str:
    """Stable string key for a configuration (for dedup/resume)."""
    return (
        f"k{config['key_bits']}_v{config['val_bits']}"
        f"_a{config['anchor_interval']}"
        f"_w{config['fp16_window']}"
        f"_rq{int(config['use_residual_quant'])}"
    )


def generate_config_list() -> List[Dict]:
    """Generate all configurations, priority-ordered.

    Strategy: most promising first, then systematic sweep.
    Priority 1: known breakthroughs (from prior experiments)
    Priority 2: combined stacks at moderate compression
    Priority 3: full systematic sweep of remaining space
    """
    seen = set()
    configs = []

    def add(cfg: Dict):
        key = config_key(cfg)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    # --- Priority 1: known good configs from prior experiments ---
    # Anchor-12 + 4-bit keys + 4-bit values (known clean)
    add({"key_bits": 4, "val_bits": 4, "anchor_interval": 12, "fp16_window": 0, "use_residual_quant": False, "mse_only": True})
    # Anchor-12 + 4-bit keys + 2-bit values (known clean, higher compression)
    add({"key_bits": 4, "val_bits": 2, "anchor_interval": 12, "fp16_window": 0, "use_residual_quant": False, "mse_only": True})
    # Anchor-12 + ResQ-4 keys + 2-bit values
    add({"key_bits": 4, "val_bits": 2, "anchor_interval": 12, "fp16_window": 0, "use_residual_quant": True, "mse_only": True})
    # No anchors + ResQ-4 keys + 2-bit values (highest compression that may work)
    add({"key_bits": 4, "val_bits": 2, "anchor_interval": 0, "fp16_window": 0, "use_residual_quant": True, "mse_only": True})
    # Anchor-12 + 5-bit keys + 2-bit values
    add({"key_bits": 5, "val_bits": 2, "anchor_interval": 12, "fp16_window": 0, "use_residual_quant": False, "mse_only": True})
    # Windowed variants
    add({"key_bits": 4, "val_bits": 2, "anchor_interval": 12, "fp16_window": 128, "use_residual_quant": True, "mse_only": True})
    add({"key_bits": 4, "val_bits": 2, "anchor_interval": 12, "fp16_window": 64, "use_residual_quant": False, "mse_only": True})
    # 3-bit keys (aggressive)
    add({"key_bits": 3, "val_bits": 2, "anchor_interval": 12, "fp16_window": 0, "use_residual_quant": False, "mse_only": True})
    add({"key_bits": 3, "val_bits": 2, "anchor_interval": 6, "fp16_window": 0, "use_residual_quant": False, "mse_only": True})
    add({"key_bits": 3, "val_bits": 3, "anchor_interval": 6, "fp16_window": 0, "use_residual_quant": False, "mse_only": True})
    add({"key_bits": 3, "val_bits": 2, "anchor_interval": 12, "fp16_window": 128, "use_residual_quant": True, "mse_only": True})

    # --- Priority 2: combined stacks at moderate compression ---
    for kb in [4, 5, 6]:
        for vb in [2, 3]:
            for ai in [6, 12, 18]:
                for rq in [True, False]:
                    add({"key_bits": kb, "val_bits": vb, "anchor_interval": ai, "fp16_window": 0, "use_residual_quant": rq, "mse_only": True})

    # --- Priority 3: full systematic sweep ---
    for kb, vb, ai, fw, rq, mo in product(
        SEARCH_SPACE["key_bits"],
        SEARCH_SPACE["val_bits"],
        SEARCH_SPACE["anchor_interval"],
        SEARCH_SPACE["fp16_window"],
        SEARCH_SPACE["use_residual_quant"],
        SEARCH_SPACE["mse_only"],
    ):
        add({"key_bits": kb, "val_bits": vb, "anchor_interval": ai, "fp16_window": fw, "use_residual_quant": rq, "mse_only": mo})

    return configs


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed_keys(results_file: str) -> set:
    """Load config keys already tested from results file."""
    completed = set()
    if not os.path.exists(results_file):
        return completed
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cfg = entry.get("config", {})
                if cfg:
                    completed.add(config_key(cfg))
            except json.JSONDecodeError:
                continue
    return completed


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def update_pareto(
    frontier: List[Tuple[float, float]],
    compression: float,
    score: float,
) -> List[Tuple[float, float]]:
    """Update the Pareto frontier (compression, score) and return pruned list.

    A point (c, s) is Pareto-dominated if there exists another point
    (c2, s2) with c2 >= c AND s2 >= s AND (c2, s2) != (c, s).
    """
    candidate = (compression, score)
    # Check if new point is dominated
    dominated = any(
        c2 >= compression and s2 >= score and (c2, s2) != candidate
        for c2, s2 in frontier
    )
    if dominated:
        return frontier

    # Add new point and remove points it dominates
    frontier.append(candidate)
    frontier = [
        (c, s) for c, s in frontier
        if not any(
            c2 >= c and s2 >= s and (c2, s2) != (c, s)
            for c2, s2 in frontier
        )
    ]
    # Sort by compression descending
    frontier.sort(key=lambda x: x[0], reverse=True)
    return frontier


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def print_final_summary(results_file: str):
    """Read results file and print a ranked summary."""
    if not os.path.exists(results_file):
        print("No results file found.")
        return

    results = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not results:
        print("No results to summarize.")
        return

    # Separate baseline
    baseline = [r for r in results if r.get("config", {}).get("baseline")]
    configs = [r for r in results if not r.get("config", {}).get("baseline")]

    # Sort by score descending, then compression descending
    configs.sort(key=lambda r: (r["total_score"], r["compression"]), reverse=True)

    print("\n" + "=" * 80)
    print("AUTORESEARCH FINAL SUMMARY")
    print("=" * 80)

    if baseline:
        b = baseline[0]
        print(f"\nFP16 Baseline score: {b['total_score']:.4f}")

    print(f"\nTotal configurations tested: {len(configs)}")

    # Top 20 by score
    print(f"\n{'='*80}")
    print("TOP 20 BY QUALITY SCORE")
    print(f"{'='*80}")
    print(f"{'#':>3}  {'Score':>6}  {'Comp':>6}  {'K':>2}b  {'V':>2}b  {'Anchor':>6}  {'Window':>6}  {'ResQ':>4}  {'Time':>6}")
    print("-" * 70)
    for i, r in enumerate(configs[:20]):
        c = r["config"]
        print(
            f"{i+1:>3}  {r['total_score']:>6.4f}  {r['compression']:>5.2f}x  "
            f"{c['key_bits']:>2}   {c['val_bits']:>2}   {c['anchor_interval']:>6}  "
            f"{c['fp16_window']:>6}  {str(c['use_residual_quant']):>5}  "
            f"{r['elapsed_s']:>5.1f}s"
        )

    # Top 20 by compression (among configs scoring >= 0.7)
    good_configs = [r for r in configs if r["total_score"] >= 0.7]
    good_configs.sort(key=lambda r: r["compression"], reverse=True)

    print(f"\n{'='*80}")
    print("TOP 20 BY COMPRESSION (score >= 0.7)")
    print(f"{'='*80}")
    print(f"{'#':>3}  {'Score':>6}  {'Comp':>6}  {'K':>2}b  {'V':>2}b  {'Anchor':>6}  {'Window':>6}  {'ResQ':>4}")
    print("-" * 65)
    for i, r in enumerate(good_configs[:20]):
        c = r["config"]
        print(
            f"{i+1:>3}  {r['total_score']:>6.4f}  {r['compression']:>5.2f}x  "
            f"{c['key_bits']:>2}   {c['val_bits']:>2}   {c['anchor_interval']:>6}  "
            f"{c['fp16_window']:>6}  {str(c['use_residual_quant']):>5}"
        )

    # Pareto frontier
    frontier = []
    for r in configs:
        frontier = update_pareto(frontier, r["compression"], r["total_score"])

    print(f"\n{'='*80}")
    print("PARETO FRONTIER (compression vs quality)")
    print(f"{'='*80}")
    for comp, score in frontier:
        # Find the config
        match = [r for r in configs if abs(r["compression"] - comp) < 0.01 and abs(r["total_score"] - score) < 0.0001]
        if match:
            c = match[0]["config"]
            desc = f"K{c['key_bits']} V{c['val_bits']} A{c['anchor_interval']} W{c['fp16_window']} RQ={c['use_residual_quant']}"
        else:
            desc = "?"
        print(f"  {comp:>5.2f}x compression, {score:.4f} score -- {desc}")

    # Quality tiers
    tiers = {
        "Excellent (>= 0.9)": [r for r in configs if r["total_score"] >= 0.9],
        "Good (>= 0.8)": [r for r in configs if 0.8 <= r["total_score"] < 0.9],
        "OK (>= 0.7)": [r for r in configs if 0.7 <= r["total_score"] < 0.8],
        "Poor (>= 0.5)": [r for r in configs if 0.5 <= r["total_score"] < 0.7],
        "Garbled (< 0.5)": [r for r in configs if r["total_score"] < 0.5],
    }

    print(f"\n{'='*80}")
    print("QUALITY TIER DISTRIBUTION")
    print(f"{'='*80}")
    for tier_name, tier_configs in tiers.items():
        if tier_configs:
            max_comp = max(r["compression"] for r in tier_configs)
            avg_comp = sum(r["compression"] for r in tier_configs) / len(tier_configs)
            print(f"  {tier_name}: {len(tier_configs)} configs, max compression {max_comp:.2f}x, avg {avg_comp:.2f}x")
        else:
            print(f"  {tier_name}: 0 configs")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_autoresearch(
    model,
    tokenizer,
    max_rounds: int = 600,
    results_file: str = "autoresearch_results.jsonl",
    resume: bool = False,
):
    """Run the autoresearch loop.

    Sweeps configurations in priority order, scores each, and saves results
    incrementally to JSONL file.
    """
    all_configs = generate_config_list()

    # Resume support: skip already-tested configs
    if resume:
        completed = load_completed_keys(results_file)
        original_count = len(all_configs)
        all_configs = [c for c in all_configs if config_key(c) not in completed]
        print(f"Resume: {len(completed)} already tested, {len(all_configs)} remaining "
              f"(out of {original_count} total)")
    else:
        completed = set()

    # Apply max_rounds cap
    configs = all_configs[:max_rounds]
    total_configs = len(configs)

    if total_configs == 0:
        print("No configurations to test. All done!")
        return

    print(f"\nAutoresearch: testing {total_configs} configurations")
    print(f"Results file: {results_file}")
    print(f"Estimated time: {total_configs * 30 / 60:.0f}-{total_configs * 60 / 60:.0f} minutes")
    print()

    best_score = 0.0
    best_config = None
    pareto_frontier: List[Tuple[float, float]] = []

    # Load existing Pareto frontier from results if resuming
    if resume and os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if not entry.get("config", {}).get("baseline"):
                        pareto_frontier = update_pareto(
                            pareto_frontier,
                            entry["compression"],
                            entry["total_score"],
                        )
                        if entry["total_score"] > best_score:
                            best_score = entry["total_score"]
                            best_config = entry["config"]
                except (json.JSONDecodeError, KeyError):
                    continue

    cumulative_time = 0.0

    for round_num, config in enumerate(configs):
        start_time = time.time()

        try:
            total_score, per_prompt, compression = score_config(
                model, tokenizer, config, is_baseline=False,
            )
        except Exception as e:
            # Don't let one bad config kill the loop
            elapsed = time.time() - start_time
            error_result = {
                "round": round_num,
                "config": config,
                "config_key": config_key(config),
                "total_score": 0.0,
                "per_prompt": [],
                "compression": compute_compression_ratio(config),
                "elapsed_s": round(elapsed, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": f"{type(e).__name__}: {e}",
            }
            with open(results_file, "a") as f:
                f.write(json.dumps(error_result) + "\n")

            print(
                f"[{round_num+1}/{total_configs}] ERROR "
                f"k={config['key_bits']}b v={config['val_bits']}b "
                f"anchor={config['anchor_interval']} "
                f"window={config['fp16_window']} "
                f"resq={config['use_residual_quant']} "
                f"-- {type(e).__name__}: {e}"
            )
            traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        elapsed = time.time() - start_time
        cumulative_time += elapsed

        # Log result
        result = {
            "round": round_num,
            "config": config,
            "config_key": config_key(config),
            "total_score": total_score,
            "per_prompt": per_prompt,
            "compression": compression,
            "elapsed_s": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Track best
        is_new_best = total_score > best_score
        if is_new_best:
            best_score = total_score
            best_config = config

        # Update Pareto frontier
        pareto_frontier = update_pareto(pareto_frontier, compression, total_score)

        # Progress indicator
        if total_score >= 0.8:
            stars = "**"
        elif total_score >= 0.5:
            stars = "* "
        else:
            stars = ". "

        # Estimated time remaining
        avg_time = cumulative_time / (round_num + 1)
        remaining = (total_configs - round_num - 1) * avg_time
        eta_min = remaining / 60.0

        print(
            f"[{round_num+1}/{total_configs}] {stars} "
            f"score={total_score:.4f}  "
            f"comp={compression:.2f}x  "
            f"k={config['key_bits']}b v={config['val_bits']}b  "
            f"anchor={config['anchor_interval']}  "
            f"win={config['fp16_window']}  "
            f"resq={config['use_residual_quant']}  "
            f"({elapsed:.1f}s, ETA {eta_min:.0f}m)"
        )

        if is_new_best:
            print(f"  >>> NEW BEST: {total_score:.4f} at {compression:.2f}x compression")

        # Periodic GC
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final report
    print("\n" + "=" * 80)
    print("AUTORESEARCH COMPLETE")
    print("=" * 80)
    print(f"Rounds tested:  {total_configs}")
    print(f"Total time:     {cumulative_time / 60:.1f} minutes")
    print(f"Avg per round:  {cumulative_time / max(total_configs, 1):.1f}s")
    print(f"Best score:     {best_score:.4f}")
    print(f"Best config:    {best_config}")
    print(f"Pareto points:  {len(pareto_frontier)}")
    for comp, score in pareto_frontier:
        print(f"  {comp:.2f}x compression, {score:.4f} score")
    print("=" * 80)

    # Full summary from file
    print_final_summary(results_file)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch: overnight TurboQuantDC configuration sweep",
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help=f"Model name (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=600,
        help="Max configurations to test (default: 600 = full sweep)",
    )
    parser.add_argument(
        "--results-file", default="autoresearch_results.jsonl",
        help="JSONL file to save results (default: autoresearch_results.jsonl)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-tested configurations (read from results file)",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Just print the summary from an existing results file, don't run",
    )
    args = parser.parse_args()

    results_path = os.path.join(REPO_ROOT, args.results_file)

    if args.summary_only:
        print_final_summary(results_path)
        return

    # Load model ONCE
    print(f"Loading model: {args.model}")
    print(f"This takes ~30-60 seconds...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Model loaded on {next(model.parameters()).device}")

    # Run FP16 baseline for comparison
    print("\n--- FP16 BASELINE ---")
    baseline_config = {"baseline": True, "key_bits": 16, "val_bits": 16, "anchor_interval": 0, "fp16_window": 0, "use_residual_quant": False, "mse_only": True}
    baseline_score, baseline_prompts, _ = score_config(
        model, tokenizer, baseline_config, is_baseline=True,
    )
    print(f"FP16 baseline score: {baseline_score:.4f}")
    for p in baseline_prompts:
        print(f"  [{p['type']:>10}] {p['score']:.2f}  {p['response'][:80]}")

    # Save baseline
    baseline_result = {
        "round": -1,
        "config": baseline_config,
        "config_key": "fp16_baseline",
        "total_score": baseline_score,
        "per_prompt": baseline_prompts,
        "compression": 1.0,
        "elapsed_s": 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Only write baseline if not resuming or if file doesn't exist
    if not args.resume or not os.path.exists(results_path):
        with open(results_path, "a") as f:
            f.write(json.dumps(baseline_result) + "\n")

    print()

    # Run the loop
    run_autoresearch(
        model, tokenizer,
        max_rounds=args.max_rounds,
        results_file=results_path,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
