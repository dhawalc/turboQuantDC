"""Benchmark: TurboQuant KV cache vs FP16 baseline.

Measures memory usage, generation speed, and output quality for
HuggingFace model.generate() with TurboQuantCache.

Produces a markdown table suitable for inclusion in a PR description.

Usage:
    python benchmarks/hf_benchmark.py

Reference: TurboQuant paper (arxiv 2504.19874), ICLR 2026.
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 100
WARMUP_RUNS = 1
BENCHMARK_RUNS = 3
BIT_WIDTHS = [4, 3, 2]

# Prompts for quality evaluation
QUALITY_PROMPTS = [
    "Explain the Pythagorean theorem in simple terms.",
    "What causes the seasons on Earth?",
    "Write a haiku about programming.",
    "List the first five prime numbers and explain why they are prime.",
    "What is the difference between a stack and a queue in computer science?",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Results from a single benchmark configuration."""
    config: str
    bits: Optional[int]
    peak_vram_mb: float = 0.0
    avg_tokens_per_sec: float = 0.0
    compression_ratio: float = 1.0
    kv_cache_mb: float = 0.0
    fp16_equiv_mb: float = 0.0
    outputs: List[str] = field(default_factory=list)
    avg_generation_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_peak_vram_mb() -> float:
    """Return peak VRAM allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_vram_stats():
    """Reset CUDA memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def compute_token_overlap(output_a: str, output_b: str) -> float:
    """Compute token-level overlap (Jaccard similarity) between two outputs.

    This is a simple proxy for output quality -- it measures how many
    unique whitespace-separated tokens are shared between the two outputs.
    """
    tokens_a = set(output_a.lower().split())
    tokens_b = set(output_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def compute_ngram_overlap(output_a: str, output_b: str, n: int = 4) -> float:
    """Compute n-gram overlap (BLEU-like) between two outputs."""
    words_a = output_a.lower().split()
    words_b = output_b.lower().split()
    if len(words_a) < n or len(words_b) < n:
        return compute_token_overlap(output_a, output_b)

    ngrams_a = set(tuple(words_a[i:i+n]) for i in range(len(words_a) - n + 1))
    ngrams_b = set(tuple(words_b[i:i+n]) for i in range(len(words_b) - n + 1))
    if not ngrams_a or not ngrams_b:
        return 0.0
    return len(ngrams_a & ngrams_b) / max(len(ngrams_a), len(ngrams_b))


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_generation_benchmark(
    model,
    tokenizer,
    prompts: List[str],
    cache_factory,
    config_name: str,
    bits: Optional[int],
    max_new_tokens: int = MAX_NEW_TOKENS,
    num_runs: int = BENCHMARK_RUNS,
    warmup_runs: int = WARMUP_RUNS,
) -> BenchmarkResult:
    """Run generation benchmark for a single configuration.

    Args:
        model: HF model.
        tokenizer: HF tokenizer.
        prompts: List of prompts to generate from.
        cache_factory: Callable returning a cache object (or None for FP16).
        config_name: Name of this configuration.
        bits: Bit-width (or None for FP16 baseline).
        max_new_tokens: Max tokens to generate per prompt.
        num_runs: Number of timed runs (results are averaged).
        warmup_runs: Number of warmup runs before timing.

    Returns:
        BenchmarkResult with aggregated metrics.
    """
    result = BenchmarkResult(config=config_name, bits=bits)
    all_times = []
    all_tokens = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        # Warmup
        for _ in range(warmup_runs):
            cache = cache_factory()
            kwargs = {"past_key_values": cache} if cache is not None else {}
            with torch.no_grad():
                _ = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, **kwargs,
                )
            del cache
            gc.collect()
            torch.cuda.empty_cache()

        # Timed runs
        for run_idx in range(num_runs):
            cache = cache_factory()
            kwargs = {"past_key_values": cache} if cache is not None else {}

            reset_vram_stats()

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()

            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, **kwargs,
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()

            elapsed = t1 - t0
            new_tokens = output.shape[1] - input_len
            all_times.append(elapsed)
            all_tokens.append(new_tokens)

            # Record peak VRAM from last run
            result.peak_vram_mb = max(result.peak_vram_mb, get_peak_vram_mb())

            # Record output text (from last run only)
            if run_idx == num_runs - 1:
                text_out = tokenizer.decode(
                    output[0, input_len:], skip_special_tokens=True,
                )
                result.outputs.append(text_out)

            # Record KV cache stats (from last run only)
            if run_idx == num_runs - 1 and cache is not None:
                savings = cache.memory_savings()
                result.kv_cache_mb = savings["total_compressed_bits"] / 8 / 1024 / 1024
                result.fp16_equiv_mb = savings["total_fp16_bits"] / 8 / 1024 / 1024
                result.compression_ratio = savings["overall_compression_ratio"]

            del cache, output
            gc.collect()
            torch.cuda.empty_cache()

    # Aggregate timing
    total_tokens = sum(all_tokens)
    total_time = sum(all_times)
    result.avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    result.avg_generation_time_s = total_time / len(all_times) if all_times else 0

    return result


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_full_benchmark():
    """Run complete FP16 vs TurboQuant benchmark suite."""

    print("=" * 70)
    print("TurboQuantDC HuggingFace Benchmark")
    print("=" * 70)
    print(f"Model:          {MODEL_NAME}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Benchmark runs: {BENCHMARK_RUNS}")
    print(f"Warmup runs:    {WARMUP_RUNS}")
    print(f"Prompts:        {len(QUALITY_PROMPTS)}")
    if torch.cuda.is_available():
        print(f"GPU:            {torch.cuda.get_device_name()}")
        print(f"VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turboquantdc import TurboQuantCache

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("Model loaded.\n")

    results: List[BenchmarkResult] = []

    # --- FP16 baseline ---
    print("Running FP16 baseline...")
    fp16_result = run_generation_benchmark(
        model, tokenizer, QUALITY_PROMPTS,
        cache_factory=lambda: None,
        config_name="FP16 (baseline)",
        bits=None,
    )
    results.append(fp16_result)
    print(f"  {fp16_result.avg_tokens_per_sec:.1f} tok/s, peak VRAM {fp16_result.peak_vram_mb:.0f} MB")

    # --- TurboQuant at each bit-width ---
    for bits in BIT_WIDTHS:
        print(f"Running TQ-{bits} ({bits}-bit)...")
        tq_result = run_generation_benchmark(
            model, tokenizer, QUALITY_PROMPTS,
            cache_factory=lambda b=bits: TurboQuantCache(bits=b),
            config_name=f"TQ-{bits} ({bits}-bit)",
            bits=bits,
        )
        results.append(tq_result)
        print(f"  {tq_result.avg_tokens_per_sec:.1f} tok/s, {tq_result.compression_ratio:.1f}x, peak VRAM {tq_result.peak_vram_mb:.0f} MB")

    # --- Quality comparison ---
    print("\n" + "=" * 70)
    print("Quality Comparison (vs FP16 baseline)")
    print("=" * 70)

    fp16_outputs = fp16_result.outputs
    for tq_result in results[1:]:
        overlaps = []
        ngram_overlaps = []
        for fp16_out, tq_out in zip(fp16_outputs, tq_result.outputs):
            overlaps.append(compute_token_overlap(fp16_out, tq_out))
            ngram_overlaps.append(compute_ngram_overlap(fp16_out, tq_out))

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        avg_ngram = sum(ngram_overlaps) / len(ngram_overlaps) if ngram_overlaps else 0

        print(f"\n{tq_result.config}:")
        print(f"  Token overlap (Jaccard):  {avg_overlap:.3f}")
        print(f"  4-gram overlap:           {avg_ngram:.3f}")

        # Show first prompt comparison
        if fp16_outputs and tq_result.outputs:
            print(f"  Sample output (first prompt):")
            print(f"    FP16: {fp16_outputs[0][:150]}...")
            print(f"    TQ:   {tq_result.outputs[0][:150]}...")

    # --- Markdown table ---
    print("\n" + "=" * 70)
    print("Markdown Results Table (for PR description)")
    print("=" * 70)
    print()
    print_markdown_table(results, fp16_result)

    # --- Per-prompt output comparison ---
    print("\n" + "=" * 70)
    print("Detailed Output Comparison")
    print("=" * 70)
    for i, prompt in enumerate(QUALITY_PROMPTS):
        print(f"\nPrompt {i+1}: {prompt}")
        for r in results:
            if i < len(r.outputs):
                print(f"  [{r.config}]: {r.outputs[i][:200]}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


def print_markdown_table(results: List[BenchmarkResult], fp16_result: BenchmarkResult):
    """Print results as a markdown table."""

    print("| Config | Bits | Compression | KV Cache | Tok/s | Peak VRAM | Token Overlap |")
    print("|--------|------|-------------|----------|-------|-----------|---------------|")

    for r in results:
        # Compute overlap with FP16
        if r.bits is None:
            overlap_str = "1.000 (ref)"
        else:
            overlaps = [
                compute_token_overlap(fp16_out, tq_out)
                for fp16_out, tq_out in zip(fp16_result.outputs, r.outputs)
            ]
            avg = sum(overlaps) / len(overlaps) if overlaps else 0
            overlap_str = f"{avg:.3f}"

        bits_str = str(r.bits) if r.bits else "16"
        ratio_str = f"{r.compression_ratio:.1f}x" if r.compression_ratio > 1.0 else "1.0x (baseline)"
        cache_str = f"{r.kv_cache_mb:.2f} MB" if r.kv_cache_mb > 0 else "FP16"

        print(
            f"| {r.config:<22} | {bits_str:<4} | {ratio_str:<11} | {cache_str:<8} "
            f"| {r.avg_tokens_per_sec:<5.1f} | {r.peak_vram_mb:<9.0f} MB | {overlap_str:<13} |"
        )

    print()
    print("*Token overlap: Jaccard similarity of whitespace-tokenized output vs FP16 baseline.*")
    print(f"*Benchmark: {MODEL_NAME}, {MAX_NEW_TOKENS} max new tokens, {BENCHMARK_RUNS} runs averaged.*")


# ---------------------------------------------------------------------------
# Standalone attention quality benchmark (does not require generate)
# ---------------------------------------------------------------------------

def run_attention_quality_benchmark():
    """Measure attention score quality with TurboQuant compression.

    This extracts real KV cache tensors from the model and compares
    attention scores computed with FP16 vs TurboQuant-compressed KV pairs.
    """
    print("\n" + "=" * 70)
    print("Attention Score Quality Benchmark")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turboquantdc import TurboQuantEstimator

    model_name = MODEL_NAME

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Generate KV cache from a prompt
    prompt = (
        "The research team discovered that quantum entanglement could be "
        "maintained over distances exceeding one hundred kilometers using "
        "specialized fiber optic cables cooled to near absolute zero. "
        "This breakthrough in quantum communication technology paves the way "
        "for unhackable long-distance networks. "
    ) * 3
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")

    # Extract KV cache
    with torch.no_grad():
        output = model(**inputs, use_cache=True)
        kv_cache = output.past_key_values

    # Get KV tensors from a few layers
    # DynamicCache in transformers 5.x uses .layers[i].keys / .layers[i].values
    num_layers = len(kv_cache.layers)
    head_dim = kv_cache.layers[0].keys.shape[-1]
    sample_layers = [0, num_layers // 2, num_layers - 1]

    print(f"Head dim: {head_dim}, Layers: {num_layers}, Sampling layers: {sample_layers}")
    print()

    print(f"{'Bits':<6} {'Layer':<7} {'Cosine Sim':<12} {'Top-1':<8} {'Top-5':<8}")
    print("-" * 43)

    for bits in BIT_WIDTHS:
        for layer_idx in sample_layers:
            keys = kv_cache.layers[layer_idx].keys.float()   # [1, num_heads, seq, d]
            values = kv_cache.layers[layer_idx].values.float()

            # Pick a random query head
            num_heads = keys.shape[1]
            seq_len = keys.shape[2]

            # Use last token as query
            query = keys[:, :, -1:, :]  # [1, num_heads, 1, d]

            # FP16 attention scores
            scores_fp16 = torch.matmul(query, keys.transpose(-2, -1)).squeeze(2)  # [1, num_heads, seq]
            scores_fp16 = scores_fp16 / math.sqrt(head_dim)

            # Compress keys with TurboQuant, then compute attention
            device = str(keys.device)
            est = TurboQuantEstimator(d=head_dim, bits=bits, seed=42, device=device)

            # Flatten keys and compress
            keys_flat = keys.reshape(-1, head_dim)
            norms = keys_flat.norm(dim=-1, keepdim=True)
            keys_norm = keys_flat / (norms + 1e-8)
            indices = est.polar.quantize(keys_norm)
            keys_recon = est.polar.dequantize(indices) * norms
            keys_recon = keys_recon.reshape(keys.shape)

            # Compressed attention scores
            scores_tq = torch.matmul(query, keys_recon.transpose(-2, -1)).squeeze(2)
            scores_tq = scores_tq / math.sqrt(head_dim)

            # Metrics
            cos_sim = F.cosine_similarity(
                scores_fp16.reshape(-1, seq_len),
                scores_tq.reshape(-1, seq_len),
                dim=-1,
            ).mean().item()

            # Top-k match
            top1_fp16 = scores_fp16.reshape(-1, seq_len).argmax(dim=-1)
            top1_tq = scores_tq.reshape(-1, seq_len).argmax(dim=-1)
            top1_match = (top1_fp16 == top1_tq).float().mean().item()

            top5_fp16 = scores_fp16.reshape(-1, seq_len).topk(5, dim=-1).indices
            top5_tq = scores_tq.reshape(-1, seq_len).topk(5, dim=-1).indices
            top5_match_count = 0
            for h in range(top5_fp16.shape[0]):
                overlap = len(set(top5_fp16[h].tolist()) & set(top5_tq[h].tolist()))
                top5_match_count += overlap / 5
            top5_match = top5_match_count / top5_fp16.shape[0]

            print(f"{bits:<6} {layer_idx:<7} {cos_sim:<12.4f} {top1_match:<8.1%} {top5_match:<8.1%}")

            del est
            gc.collect()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_full_benchmark()
    run_attention_quality_benchmark()

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)
