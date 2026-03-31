"""Anchor Layer Experiment -- FP16 anchor layers to break error accumulation.

Tests whether keeping every Nth layer at FP16 (no compression) prevents
catastrophic drift during autoregressive generation with TurboQuant KV cache.

Hypothesis: Error accumulates across layers during generation. If every Nth
layer resets to perfect FP16 KV, the error chain breaks and output stays
coherent even when most layers are aggressively compressed.

Configurations tested:
    1. All TQ-4 mse_only (baseline -- expected garbled)
    2. Every 3rd layer FP16, rest TQ-4 (33% uncompressed)
    3. Every 6th layer FP16, rest TQ-4 (17% uncompressed)
    4. Every 12th layer FP16, rest TQ-4 (8% uncompressed)
    5. First 6 layers FP16, rest TQ-4 (17% uncompressed)
    6. First 12 layers FP16, rest TQ-4 (33% uncompressed)
    7. Last 6 layers FP16, rest TQ-4 (17% uncompressed)
    8. All FP16 (sanity check -- should be perfect)

Usage:
    cd /home/dhawal/turboQuantDC && python benchmarks/anchor_layer_experiment.py
"""

from __future__ import annotations

import gc
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from turboquantdc.hf_integration import TurboQuantCache
from turboquantdc.adaptive_hf_cache import AdaptiveHFCache


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
NUM_LAYERS = 36  # Qwen2.5-3B has 36 layers
MAX_NEW_TOKENS = 100

PROMPTS = [
    "What is the capital of Australia? Answer briefly:",
    "What is 15 + 27?",
    "Who wrote 1984? Answer briefly:",
    "Explain what a neural network is in two sentences:",
    "Write a Python function that returns the factorial of n:",
]

# Experiment configurations: (name, cache_factory)
# Each factory returns a cache object (or None for FP16 baseline)


def make_configs():
    """Build list of (name, description, cache_factory) tuples."""
    configs = []

    # 1. All TQ-4 mse_only (baseline)
    configs.append((
        "All TQ-4 (mse_only)",
        "All 36 layers compressed at 4-bit MSE-only",
        lambda: TurboQuantCache(bits=4, mse_only=True),
    ))

    # 2. Every 3rd layer FP16
    configs.append((
        "Interval-3 (33% FP16)",
        "Every 3rd layer FP16 (12/36), rest TQ-4",
        lambda: AdaptiveHFCache(
            num_layers=NUM_LAYERS, compressed_bits=4,
            anchor_interval=3, anchor_mode="interval", mse_only=True,
        ),
    ))

    # 3. Every 6th layer FP16
    configs.append((
        "Interval-6 (17% FP16)",
        "Every 6th layer FP16 (6/36), rest TQ-4",
        lambda: AdaptiveHFCache(
            num_layers=NUM_LAYERS, compressed_bits=4,
            anchor_interval=6, anchor_mode="interval", mse_only=True,
        ),
    ))

    # 4. Every 12th layer FP16
    configs.append((
        "Interval-12 (8% FP16)",
        "Every 12th layer FP16 (3/36), rest TQ-4",
        lambda: AdaptiveHFCache(
            num_layers=NUM_LAYERS, compressed_bits=4,
            anchor_interval=12, anchor_mode="interval", mse_only=True,
        ),
    ))

    # 5. First 6 layers FP16
    configs.append((
        "Early-6 (17% FP16)",
        "First 6 layers FP16, rest TQ-4",
        lambda: AdaptiveHFCache(
            num_layers=NUM_LAYERS, compressed_bits=4,
            anchor_mode="early", n_early_fp16=6, mse_only=True,
        ),
    ))

    # 6. First 12 layers FP16
    configs.append((
        "Early-12 (33% FP16)",
        "First 12 layers FP16, rest TQ-4",
        lambda: AdaptiveHFCache(
            num_layers=NUM_LAYERS, compressed_bits=4,
            anchor_mode="early", n_early_fp16=12, mse_only=True,
        ),
    ))

    # 7. Last 6 layers FP16
    configs.append((
        "Tail-6 (17% FP16)",
        "Last 6 layers FP16, rest TQ-4",
        lambda: AdaptiveHFCache(
            num_layers=NUM_LAYERS, compressed_bits=4,
            anchor_mode="tail", n_early_fp16=6, mse_only=True,
        ),
    ))

    # 8. All FP16 (sanity check)
    configs.append((
        "All FP16 (baseline)",
        "No compression -- sanity check",
        lambda: None,  # None = use HF default DynamicCache
    ))

    return configs


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ConfigResult:
    name: str
    description: str
    outputs: List[str] = field(default_factory=list)
    generation_time_s: float = 0.0
    compression_ratio: float = 1.0
    fp16_layer_count: int = 0
    compressed_layer_count: int = 0
    peak_vram_mb: float = 0.0
    is_coherent: List[bool] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Coherence heuristic
# ---------------------------------------------------------------------------

def is_output_coherent(prompt: str, output: str) -> bool:
    """Simple heuristic: check if output looks like natural language.

    Detects garbled output by checking for:
    - Excessive repetition of the same character/word
    - Very short output (< 5 chars after stripping)
    - High ratio of non-ASCII characters
    - Known garbled patterns
    """
    text = output.strip()
    if len(text) < 3:
        return False

    # Check for excessive single-character repetition
    if len(text) > 10:
        char_counts = {}
        for c in text:
            char_counts[c] = char_counts.get(c, 0) + 1
        most_common = max(char_counts.values())
        if most_common / len(text) > 0.5:
            return False

    # Check for excessive word repetition
    words = text.split()
    if len(words) > 5:
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        most_common_word = max(word_counts.values())
        if most_common_word / len(words) > 0.5:
            return False

    # Check for non-ASCII ratio (garbled output often has lots of strange chars)
    non_ascii = sum(1 for c in text if ord(c) > 127 and not c.isalpha())
    if len(text) > 10 and non_ascii / len(text) > 0.3:
        return False

    return True


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def load_model():
    """Load Qwen2.5-3B-Instruct in 4-bit."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME} (4-bit NF4)...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    load_time = time.time() - t0
    gpu_mb = torch.cuda.memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0
    print(f"  Loaded in {load_time:.1f}s | GPU: {gpu_mb} MB")
    print(f"  Config: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_attention_heads} heads, "
          f"{model.config.hidden_size // model.config.num_attention_heads} head_dim")
    print()
    return model, tokenizer


def run_config(
    model,
    tokenizer,
    name: str,
    description: str,
    cache_factory,
) -> ConfigResult:
    """Run all prompts with a single cache configuration."""
    result = ConfigResult(name=name, description=description)

    total_time = 0.0
    peak_vram = 0.0

    for prompt_idx, prompt in enumerate(PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        cache = cache_factory()
        kwargs = {}
        if cache is not None:
            kwargs["past_key_values"] = cache

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                **kwargs,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        total_time += elapsed

        if torch.cuda.is_available():
            current_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            peak_vram = max(peak_vram, current_peak)

        text_out = tokenizer.decode(
            output[0, input_len:], skip_special_tokens=True,
        )
        result.outputs.append(text_out)
        result.is_coherent.append(is_output_coherent(prompt, text_out))

        # Get compression stats from last prompt
        if prompt_idx == len(PROMPTS) - 1 and cache is not None:
            if isinstance(cache, AdaptiveHFCache):
                savings = cache.memory_savings()
                result.compression_ratio = savings["overall_compression_ratio"]
                result.fp16_layer_count = savings["fp16_anchor_count"]
                result.compressed_layer_count = savings["compressed_layer_count"]
            elif isinstance(cache, TurboQuantCache):
                savings = cache.memory_savings()
                result.compression_ratio = savings["overall_compression_ratio"]
                result.fp16_layer_count = 0
                result.compressed_layer_count = savings["num_layers"]
            else:
                result.compression_ratio = 1.0

        del cache, output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result.generation_time_s = total_time
    result.peak_vram_mb = peak_vram

    if result.compression_ratio == 0.0:
        result.compression_ratio = 1.0  # FP16 baseline

    return result


def print_results(results: List[ConfigResult]):
    """Print formatted results table and per-prompt outputs."""

    print()
    print("=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)

    # Summary table
    print()
    print(f"{'Config':<25} | {'FP16/Total':<10} | {'Ratio':<7} | {'Coherent':<10} | {'Time':<7} | {'VRAM':<8}")
    print(f"{'-'*25}-+-{'-'*10}-+-{'-'*7}-+-{'-'*10}-+-{'-'*7}-+-{'-'*8}")

    for r in results:
        n_coherent = sum(r.is_coherent)
        coherent_str = f"{n_coherent}/{len(r.is_coherent)}"
        if n_coherent == len(r.is_coherent):
            coherent_str += " OK"
        elif n_coherent == 0:
            coherent_str += " BAD"
        else:
            coherent_str += " MIX"

        if r.fp16_layer_count > 0 or r.compressed_layer_count > 0:
            layers_str = f"{r.fp16_layer_count}/{r.fp16_layer_count + r.compressed_layer_count}"
        else:
            layers_str = "all/all" if r.name.startswith("All FP16") else "0/36"

        print(
            f"{r.name:<25} | {layers_str:<10} | {r.compression_ratio:<7.2f}x | "
            f"{coherent_str:<10} | {r.generation_time_s:<7.1f}s | {r.peak_vram_mb:<7.0f}M"
        )

    # Detailed per-prompt outputs
    print()
    print("=" * 90)
    print("DETAILED OUTPUT")
    print("=" * 90)

    for prompt_idx, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {prompt_idx + 1}: {prompt}")
        for r in results:
            if prompt_idx < len(r.outputs):
                text = r.outputs[prompt_idx]
                coherent_tag = "OK" if r.is_coherent[prompt_idx] else "GARBLED"
                # Truncate long outputs for display
                display = text[:200]
                if len(text) > 200:
                    display += "..."
                print(f"  [{r.name:<25}] [{coherent_tag:>7}] {display}")

    # Analysis
    print()
    print("=" * 90)
    print("ANALYSIS")
    print("=" * 90)

    coherent_configs = [r for r in results if all(r.is_coherent)]
    partial_configs = [r for r in results if any(r.is_coherent) and not all(r.is_coherent)]
    garbled_configs = [r for r in results if not any(r.is_coherent)]

    if coherent_configs:
        print(f"\nFully coherent ({len(coherent_configs)} configs):")
        for r in coherent_configs:
            print(f"  - {r.name}: {r.compression_ratio:.2f}x compression")

    if partial_configs:
        print(f"\nPartially coherent ({len(partial_configs)} configs):")
        for r in partial_configs:
            n = sum(r.is_coherent)
            print(f"  - {r.name}: {n}/{len(r.is_coherent)} prompts coherent, {r.compression_ratio:.2f}x")

    if garbled_configs:
        print(f"\nGarbled ({len(garbled_configs)} configs):")
        for r in garbled_configs:
            print(f"  - {r.name}: {r.compression_ratio:.2f}x compression")

    # Best coherent config with highest compression
    if coherent_configs:
        # Filter out the all-FP16 baseline
        non_baseline = [r for r in coherent_configs if not r.name.startswith("All FP16")]
        if non_baseline:
            best = max(non_baseline, key=lambda r: r.compression_ratio)
            print(f"\nBest coherent compressed config: {best.name}")
            print(f"  Compression: {best.compression_ratio:.2f}x")
            print(f"  FP16 layers: {best.fp16_layer_count}/{best.fp16_layer_count + best.compressed_layer_count}")

    print()


def main():
    print()
    print("=" * 90)
    print("  ANCHOR LAYER EXPERIMENT")
    print("  Hypothesis: FP16 anchor layers break error accumulation chain")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print(f"  Prompts: {len(PROMPTS)}")
    print("=" * 90)
    print()

    model, tokenizer = load_model()
    configs = make_configs()
    results: List[ConfigResult] = []

    for i, (name, desc, factory) in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] {name}: {desc}")
        result = run_config(model, tokenizer, name, desc, factory)
        n_coherent = sum(result.is_coherent)
        print(f"  -> {n_coherent}/{len(PROMPTS)} coherent, "
              f"{result.compression_ratio:.2f}x compression, "
              f"{result.generation_time_s:.1f}s total")
        results.append(result)
        print()

    print_results(results)

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    main()
