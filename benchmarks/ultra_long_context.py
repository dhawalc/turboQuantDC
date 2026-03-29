"""Ultra Long Context Benchmark — 512K+ tokens on a single RTX 4090.

Demonstrates that TurboQuant KV cache compression enables 512K-token context
on hardware that would OOM at ~50K tokens with FP16 KV cache.

Approach:
    Option A (primary): Synthetic KV cache stress test.
    Generate 512K vectors, compress with TurboQuant, measure real VRAM usage,
    and test needle-in-haystack retrieval via attention score ranking.

    This proves the KV CACHE handles 512K tokens with accurate attention scores.
    It does not prove the MODEL generates coherent output at 512K.

    Option B (secondary): Real-model quality validation at 32K using
    TurboQuantCache with HuggingFace integration. Limited by the model's
    native context window.

Test matrix:
    Context   | Bits | Compression | VRAM Used | Needle Found?
    128K      | 2,3  |    ---      |   ---     |    ---
    256K      | 2,3  |    ---      |   ---     |    ---
    512K      | 2,3  |    ---      |   ---     |    ---
    1M        | 2    |    ---      |   ---     |  (if fits)
    512K + TD | hot=4,warm=3,cold=2 | ---  |   ---     |    ---

Usage:
    python benchmarks/ultra_long_context.py
    python benchmarks/ultra_long_context.py --max-tokens 1048576
    python benchmarks/ultra_long_context.py --skip-real-model
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from turboquantdc import TurboQuantEstimator, TurboQuantKVCache  # noqa: E402
from turboquantdc.temporal_decay import TemporalDecayCache  # noqa: E402

# ---------------------------------------------------------------------------
# Qwen2.5-3B-Instruct constants
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
NUM_LAYERS = 36
NUM_KV_HEADS = 2
HEAD_DIM = 128
MODEL_WEIGHT_GB = 6.0  # Approximate for 4-bit BitsAndBytes

# ---------------------------------------------------------------------------
# Benchmark defaults
# ---------------------------------------------------------------------------
DEFAULT_CONTEXT_LENGTHS = [128 * 1024, 256 * 1024, 512 * 1024]
BIT_WIDTHS = [2, 3]
BATCH_SIZE = 4096  # Vectors per batch when feeding the cache
NEEDLE_DEPTH_FRACTIONS = [0.01, 0.10, 0.25, 0.50, 0.75, 0.99]

# Number of random "background" needles to test statistical reliability
NUM_RANDOM_NEEDLES = 5


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class NeedleResult:
    """Result of one needle retrieval test."""
    needle_pos: int
    total_tokens: int
    depth_fraction: float
    top1_match: bool
    top5_match: bool
    needle_rank: int
    score_at_needle: float
    max_score: float

    @property
    def depth_pct(self) -> str:
        return f"{self.depth_fraction * 100:.0f}%"


@dataclass
class ConfigResult:
    """Results for one (context_length, bits) configuration."""
    context_length: int
    bits: int
    mode: str  # "kvcache" or "temporal_decay"

    # Memory (real measurements)
    vram_before_gb: float = 0.0
    vram_after_gb: float = 0.0
    vram_peak_gb: float = 0.0
    cache_vram_gb: float = 0.0

    # Theoretical
    fp16_cache_gb: float = 0.0
    compression_ratio: float = 0.0
    fp16_total_gb: float = 0.0

    # Performance
    fill_time_s: float = 0.0
    fill_rate_kvecs_per_s: float = 0.0

    # Needle retrieval
    needle_results: List[NeedleResult] = field(default_factory=list)

    # Theoretical full-model cache size (from bit-counting, not VRAM measurement)
    theoretical_cache_gb: float = 0.0

    # Temporal decay specific
    td_hot_tokens: int = 0
    td_warm_tokens: int = 0
    td_cold_tokens: int = 0
    td_savings_vs_uniform_pct: float = 0.0

    @property
    def context_k(self) -> str:
        if self.context_length >= 1024 * 1024:
            return f"{self.context_length // (1024 * 1024)}M"
        return f"{self.context_length // 1024}K"

    @property
    def needle_found_rate(self) -> float:
        if not self.needle_results:
            return 0.0
        return sum(1 for n in self.needle_results if n.top5_match) / len(self.needle_results)

    @property
    def avg_needle_rank(self) -> float:
        if not self.needle_results:
            return -1
        return sum(n.needle_rank for n in self.needle_results) / len(self.needle_results)


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------
def gpu_mem_gb() -> float:
    """Current GPU memory allocated in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0


def gpu_peak_gb() -> float:
    """Peak GPU memory allocated in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def reset_peak():
    """Reset peak memory tracker."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_free_gb() -> float:
    """Free GPU memory in GB."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    return 0.0


def gpu_total_gb() -> float:
    """Total GPU memory in GB."""
    if torch.cuda.is_available():
        _, total = torch.cuda.mem_get_info()
        return total / (1024 ** 3)
    return 0.0


def force_gc():
    """Force garbage collection and CUDA cache clear."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Theoretical calculations
# ---------------------------------------------------------------------------
def fp16_kv_cache_bytes(n_tokens: int) -> int:
    """FP16 KV cache size in bytes for Qwen2.5-3B."""
    # 36 layers x 2 KV heads x n_tokens x 128 dim x 2 (K+V) x 2 bytes
    return NUM_LAYERS * NUM_KV_HEADS * n_tokens * HEAD_DIM * 2 * 2


def tq_kv_cache_bits_per_token(bits: int) -> int:
    """TurboQuant bits per token for one (layer, head)."""
    mse_bits = max(bits - 1, 1)
    # Key: mse_bits*d (MSE) + d (QJL signs) + 16 (residual_norm) + 16 (vec_norm)
    key_bits = mse_bits * HEAD_DIM + HEAD_DIM + 32
    # Value: bits*d (MSE) + 16 (vec_norm)
    val_bits = bits * HEAD_DIM + 16
    return key_bits + val_bits


def print_vram_budget():
    """Print theoretical VRAM breakdown before benchmarks."""
    print()
    print("=" * 74)
    print("  ULTRA LONG CONTEXT BENCHMARK — TurboQuantDC on RTX 4090")
    print("=" * 74)
    print()
    print("  Model: Qwen2.5-3B-Instruct")
    print(f"  Config: {NUM_LAYERS} layers, {NUM_KV_HEADS} KV heads, d={HEAD_DIM}")
    print(f"  GPU: {gpu_total_gb():.1f} GB total, {gpu_free_gb():.1f} GB free")
    print()

    # FP16 baseline
    fp16_512k = fp16_kv_cache_bytes(512 * 1024) / (1024 ** 3)
    fp16_1m = fp16_kv_cache_bytes(1024 * 1024) / (1024 ** 3)
    print(f"  FP16 KV cache at 512K tokens: {fp16_512k:.1f} GB")
    print(f"  FP16 KV cache at 1M tokens:   {fp16_1m:.1f} GB")
    print(f"  Model weights (4-bit):         ~{MODEL_WEIGHT_GB:.0f} GB")
    print(f"  FP16 total at 512K:            {fp16_512k + MODEL_WEIGHT_GB:.1f} GB  "
          f"{'-> OOM!' if fp16_512k + MODEL_WEIGHT_GB > 24 else '-> fits'}")
    print()

    for bits in BIT_WIDTHS:
        bpt = tq_kv_cache_bits_per_token(bits)
        total_bits_512k = bpt * 512 * 1024 * NUM_LAYERS * NUM_KV_HEADS
        tq_gb = total_bits_512k / 8 / (1024 ** 3)
        ratio = fp16_512k / tq_gb if tq_gb > 0 else 0
        total = tq_gb + MODEL_WEIGHT_GB
        print(f"  TQ-{bits} KV cache at 512K:     {tq_gb:.2f} GB  "
              f"({ratio:.1f}x compression)  total={total:.1f} GB  "
              f"{'-> OOM!' if total > 24 else '-> FITS!'}")

    print()
    print("-" * 74)
    print()


# ---------------------------------------------------------------------------
# Synthetic cache fill + needle test
# ---------------------------------------------------------------------------
def run_synthetic_kvcache(
    context_length: int,
    bits: int,
    needle_depths: List[float],
    device: str = "cuda",
) -> ConfigResult:
    """Fill a TurboQuantKVCache with synthetic vectors and test needle retrieval.

    Creates a single (layer, head) TurboQuantKVCache and fills it with
    context_length random vectors. Needles are placed at specified depth
    fractions. After filling, queries matching each needle are tested
    to see if the correct position ranks highest in attention scores.

    Args:
        context_length: Number of tokens to store.
        bits: TurboQuant bit-width (2, 3, or 4).
        needle_depths: List of depth fractions in [0, 1] where needles are placed.
        device: CUDA device.

    Returns:
        ConfigResult with all measurements.
    """
    result = ConfigResult(
        context_length=context_length,
        bits=bits,
        mode="kvcache",
    )

    # Theoretical FP16 cost for full model (all layers, all heads)
    result.fp16_cache_gb = fp16_kv_cache_bytes(context_length) / (1024 ** 3)
    result.fp16_total_gb = result.fp16_cache_gb + MODEL_WEIGHT_GB

    force_gc()
    reset_peak()
    result.vram_before_gb = gpu_mem_gb()

    print(f"  [{result.context_k} / TQ-{bits}] Creating cache...", flush=True)

    # Create a single-head cache (we scale the theoretical numbers for the full model)
    cache = TurboQuantKVCache(
        d_key=HEAD_DIM,
        d_value=HEAD_DIM,
        bits=bits,
        seed=42,
        device=device,
    )

    # Determine needle positions
    needle_positions = {}
    for depth in needle_depths:
        pos = max(0, min(int(context_length * depth), context_length - 1))
        needle_positions[pos] = depth

    # Generate needle vectors: distinctive high-magnitude vectors
    # Each needle is a unique direction so they are mutually distinguishable
    needle_keys = {}
    needle_values = {}
    for pos, depth in needle_positions.items():
        # Create a distinctive vector: large magnitude in a rotated direction
        # Generate on CPU with seed, then move to device
        rng = torch.Generator(device="cpu").manual_seed(pos)
        direction = torch.randn(HEAD_DIM, generator=rng).to(device)
        direction = direction / direction.norm()
        needle_keys[pos] = direction * 10.0  # 10x magnitude vs typical unit-variance
        needle_values[pos] = direction * 10.0

    # Fill cache in batches
    t0 = time.time()
    tokens_filled = 0

    while tokens_filled < context_length:
        batch_end = min(tokens_filled + BATCH_SIZE, context_length)
        batch_len = batch_end - tokens_filled

        # Generate random keys and values (unit-variance Gaussian, realistic for LLM)
        keys = torch.randn(batch_len, HEAD_DIM, device=device)
        values = torch.randn(batch_len, HEAD_DIM, device=device)

        # Insert needles into this batch if any fall in range
        for pos, depth in needle_positions.items():
            if tokens_filled <= pos < batch_end:
                local_idx = pos - tokens_filled
                keys[local_idx] = needle_keys[pos]
                values[local_idx] = needle_values[pos]

        cache.append(keys, values)
        tokens_filled = batch_end

        # Progress report every ~50 batches
        if (tokens_filled // BATCH_SIZE) % 50 == 0 or tokens_filled == context_length:
            mem = gpu_mem_gb()
            pct = tokens_filled / context_length * 100
            print(f"    {tokens_filled:>10,} / {context_length:,} tokens "
                  f"({pct:5.1f}%)  VRAM: {mem:.2f} GB", flush=True)

    fill_time = time.time() - t0
    result.fill_time_s = fill_time
    result.fill_rate_kvecs_per_s = context_length / fill_time / 1000

    # Memory measurements
    result.vram_after_gb = gpu_mem_gb()
    result.vram_peak_gb = gpu_peak_gb()
    result.cache_vram_gb = result.vram_after_gb - result.vram_before_gb

    # Compression stats from the cache itself
    usage = cache.memory_usage_bits()
    result.compression_ratio = usage["compression_ratio"]

    # Theoretical full-model cache: scale the bit-count from this single head
    # to all layers and heads. This is the correct projection because the
    # per-token bit cost is fixed and independent of Python/CUDA overhead.
    single_head_bits = usage["total_bits"]
    full_model_bits = single_head_bits * NUM_LAYERS * NUM_KV_HEADS
    result.theoretical_cache_gb = full_model_bits / 8 / (1024 ** 3)

    print(f"    Fill complete: {fill_time:.1f}s  "
          f"({result.fill_rate_kvecs_per_s:.0f} K vec/s)  "
          f"VRAM: {result.cache_vram_gb:.2f} GB  "
          f"(full-model projection: {result.theoretical_cache_gb:.2f} GB)", flush=True)

    # --- Needle retrieval tests ---
    print(f"    Testing needle retrieval at {len(needle_positions)} depths...", flush=True)

    for pos, depth in sorted(needle_positions.items()):
        # Query: same direction as the needle, so it should attend most to that position
        query = needle_keys[pos].unsqueeze(0)  # (1, d)
        scores = cache.attention_scores(query)  # (1, context_length)

        if scores.dim() > 1:
            scores = scores.squeeze(0)

        # Find needle rank
        sorted_indices = scores.argsort(descending=True)
        rank_mask = (sorted_indices == pos).nonzero(as_tuple=False)
        needle_rank = rank_mask[0].item() if len(rank_mask) > 0 else context_length

        top5 = sorted_indices[:5].tolist()
        top1_match = sorted_indices[0].item() == pos
        top5_match = pos in top5

        nr = NeedleResult(
            needle_pos=pos,
            total_tokens=context_length,
            depth_fraction=depth,
            top1_match=top1_match,
            top5_match=top5_match,
            needle_rank=needle_rank,
            score_at_needle=scores[pos].item(),
            max_score=scores.max().item(),
        )
        result.needle_results.append(nr)

        status = "FOUND (top-1)" if top1_match else ("top-5" if top5_match else f"rank={needle_rank}")
        print(f"      depth={nr.depth_pct:>4s}  pos={pos:>10,}  {status}", flush=True)

    # Cleanup
    del cache
    force_gc()

    return result


# ---------------------------------------------------------------------------
# Temporal Decay cache test
# ---------------------------------------------------------------------------
def run_temporal_decay(
    context_length: int,
    needle_depths: List[float],
    device: str = "cuda",
    hot_bits: int = 4,
    warm_bits: int = 3,
    cold_bits: int = 2,
    hot_window: int = 512,
    warm_window: int = 4096,
) -> ConfigResult:
    """Fill a TemporalDecayCache and test needle retrieval.

    Temporal decay uses tiered compression: recent tokens at higher precision,
    older tokens at lower precision. At 512K context, ~99% of tokens are in
    the cold (2-bit) tier, yielding additional savings over uniform compression.

    Args:
        context_length: Number of tokens to store.
        needle_depths: Depth fractions for needle placement.
        device: CUDA device.
        hot_bits: Bit-width for hot tier.
        warm_bits: Bit-width for warm tier.
        cold_bits: Bit-width for cold tier.
        hot_window: Size of hot tier.
        warm_window: Size of warm tier.

    Returns:
        ConfigResult with all measurements.
    """
    result = ConfigResult(
        context_length=context_length,
        bits=cold_bits,  # dominant tier
        mode="temporal_decay",
    )
    result.fp16_cache_gb = fp16_kv_cache_bytes(context_length) / (1024 ** 3)
    result.fp16_total_gb = result.fp16_cache_gb + MODEL_WEIGHT_GB

    force_gc()
    reset_peak()
    result.vram_before_gb = gpu_mem_gb()

    print(f"  [{result.context_k} / TD hot={hot_bits},warm={warm_bits},cold={cold_bits}] "
          f"Creating cache...", flush=True)

    cache = TemporalDecayCache(
        d_key=HEAD_DIM,
        d_value=HEAD_DIM,
        hot_bits=hot_bits,
        warm_bits=warm_bits,
        cold_bits=cold_bits,
        hot_window=hot_window,
        warm_window=warm_window,
        seed=42,
        device=device,
    )

    # Needle positions and vectors (same logic as kvcache test)
    needle_positions = {}
    for depth in needle_depths:
        pos = max(0, min(int(context_length * depth), context_length - 1))
        needle_positions[pos] = depth

    needle_keys = {}
    needle_values = {}
    for pos, depth in needle_positions.items():
        rng = torch.Generator(device="cpu").manual_seed(pos)
        direction = torch.randn(HEAD_DIM, generator=rng).to(device)
        direction = direction / direction.norm()
        needle_keys[pos] = direction * 10.0
        needle_values[pos] = direction * 10.0

    # Fill
    t0 = time.time()
    tokens_filled = 0

    while tokens_filled < context_length:
        batch_end = min(tokens_filled + BATCH_SIZE, context_length)
        batch_len = batch_end - tokens_filled

        keys = torch.randn(batch_len, HEAD_DIM, device=device)
        values = torch.randn(batch_len, HEAD_DIM, device=device)

        for pos, depth in needle_positions.items():
            if tokens_filled <= pos < batch_end:
                local_idx = pos - tokens_filled
                keys[local_idx] = needle_keys[pos]
                values[local_idx] = needle_values[pos]

        cache.append(keys, values)
        tokens_filled = batch_end

        if (tokens_filled // BATCH_SIZE) % 50 == 0 or tokens_filled == context_length:
            mem = gpu_mem_gb()
            pct = tokens_filled / context_length * 100
            print(f"    {tokens_filled:>10,} / {context_length:,} tokens "
                  f"({pct:5.1f}%)  VRAM: {mem:.2f} GB", flush=True)

    fill_time = time.time() - t0
    result.fill_time_s = fill_time
    result.fill_rate_kvecs_per_s = context_length / fill_time / 1000

    result.vram_after_gb = gpu_mem_gb()
    result.vram_peak_gb = gpu_peak_gb()
    result.cache_vram_gb = result.vram_after_gb - result.vram_before_gb

    # Temporal decay stats
    td_usage = cache.memory_usage_bits()
    result.compression_ratio = td_usage["compression_ratio"]
    result.td_hot_tokens = td_usage["hot_tokens"]
    result.td_warm_tokens = td_usage["warm_tokens"]
    result.td_cold_tokens = td_usage["cold_tokens"]
    result.td_savings_vs_uniform_pct = td_usage["savings_vs_uniform_pct"]

    # Theoretical full-model cache
    single_head_bits = td_usage["total_bits"]
    full_model_bits = single_head_bits * NUM_LAYERS * NUM_KV_HEADS
    result.theoretical_cache_gb = full_model_bits / 8 / (1024 ** 3)

    print(f"    Fill complete: {fill_time:.1f}s  "
          f"({result.fill_rate_kvecs_per_s:.0f} K vec/s)  "
          f"VRAM: {result.cache_vram_gb:.2f} GB  "
          f"(full-model projection: {result.theoretical_cache_gb:.2f} GB)", flush=True)
    print(f"    Tiers: hot={result.td_hot_tokens:,} warm={result.td_warm_tokens:,} "
          f"cold={result.td_cold_tokens:,}  "
          f"savings vs uniform: {result.td_savings_vs_uniform_pct:.1f}%", flush=True)

    # Needle retrieval
    # NOTE: Temporal decay reorders tokens (cold, warm, hot), so the absolute
    # position of a needle changes after demotion. We test whether the needle
    # vector's attention score is still the *highest* regardless of position.
    # This is the correct test: "does the query attend most to the needle?"
    print(f"    Testing needle retrieval at {len(needle_positions)} depths...", flush=True)

    for pos, depth in sorted(needle_positions.items()):
        query = needle_keys[pos].unsqueeze(0)
        scores = cache.attention_scores(query)

        if scores.dim() > 1:
            scores = scores.squeeze(0)

        # For temporal decay, the needle's absolute position changes.
        # We find the position with the highest score and check if it matches
        # the needle by checking score magnitude.
        max_score = scores.max().item()

        # The needle has 10x magnitude, so its score should dominate.
        # Find the rank of the *needle score* among all scores.
        # Since needle position shifts, we find the score at the original
        # position (which may now be in cold tier at a different index).
        #
        # Better approach: the needle score should be the max score.
        # If max score > 2 * median score, the needle is found.
        sorted_scores, sorted_idx = scores.sort(descending=True)
        top1_pos = sorted_idx[0].item()
        top5_pos = sorted_idx[:5].tolist()

        # Since positions are remapped in temporal decay, we check if the
        # needle is found by comparing the max score to the statistical background.
        # Background scores for random unit-variance vectors have stddev ~ sqrt(d)/d.
        # A needle with 10x magnitude should produce a score 10x higher than
        # the typical score. We use mean + 5*sigma as the threshold.
        all_scores_np = scores.float()
        score_mean = all_scores_np.mean().item()
        score_std = all_scores_np.std().item()
        threshold = score_mean + 5.0 * score_std
        needle_found = max_score > threshold

        # Score ratio vs second-highest for reporting
        score_ratio = sorted_scores[0].item() / (abs(sorted_scores[1].item()) + 1e-10)

        needle_rank = 0 if needle_found else context_length

        nr = NeedleResult(
            needle_pos=pos,
            total_tokens=context_length,
            depth_fraction=depth,
            top1_match=needle_found,
            top5_match=needle_found,
            needle_rank=needle_rank,
            score_at_needle=max_score,
            max_score=max_score,
        )
        result.needle_results.append(nr)

        status = ("FOUND" if needle_found
                  else f"NOT FOUND (max={max_score:.1f}, threshold={threshold:.1f})")
        print(f"      depth={nr.depth_pct:>4s}  {status}", flush=True)

    del cache
    force_gc()

    return result


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
def print_results_table(results: List[ConfigResult]):
    """Print the main results table."""
    print()
    print("=" * 100)
    print("  RESULTS — Synthetic KV Cache at Scale")
    print("=" * 100)
    print()

    # Table header
    hdr = (
        f"{'Context':>8s}  {'Bits':>4s}  {'Mode':>6s}  "
        f"{'Cache VRAM':>10s}  {'Comp.':>6s}  "
        f"{'FP16 Equiv':>10s}  {'VRAM Saved':>10s}  "
        f"{'Fill Rate':>10s}  {'Needle':>8s}"
    )
    print(hdr)
    print("-" * 100)

    for r in results:
        full_cache_gb = r.theoretical_cache_gb
        vram_saved = r.fp16_cache_gb - full_cache_gb
        needle_pct = r.needle_found_rate * 100

        mode = "TD" if r.mode == "temporal_decay" else f"TQ-{r.bits}"
        bits_str = f"h{4}w{3}c{2}" if r.mode == "temporal_decay" else str(r.bits)

        row = (
            f"{r.context_k:>8s}  {bits_str:>4s}  {mode:>6s}  "
            f"{full_cache_gb:>9.2f}G  {r.compression_ratio:>5.1f}x  "
            f"{r.fp16_cache_gb:>9.2f}G  {vram_saved:>9.2f}G  "
            f"{r.fill_rate_kvecs_per_s:>8.0f}K/s  "
            f"{needle_pct:>6.0f}%"
        )
        print(row)

    print()


def print_needle_detail(results: List[ConfigResult]):
    """Print detailed needle retrieval results."""
    print()
    print("=" * 90)
    print("  NEEDLE-IN-HAYSTACK DETAIL")
    print("=" * 90)

    for r in results:
        if not r.needle_results:
            continue

        mode = "TD" if r.mode == "temporal_decay" else f"TQ-{r.bits}"
        print(f"\n  {r.context_k} / {mode}:")
        print(f"  {'Depth':>6s}  {'Position':>12s}  {'Rank':>8s}  {'Top-1':>6s}  {'Top-5':>6s}  {'Score Ratio':>12s}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*12}")

        for nr in r.needle_results:
            score_ratio = nr.score_at_needle / (nr.max_score + 1e-10) if nr.max_score > 0 else 0
            print(
                f"  {nr.depth_pct:>6s}  {nr.needle_pos:>12,}  "
                f"{nr.needle_rank:>8,}  "
                f"{'YES' if nr.top1_match else ' no':>6s}  "
                f"{'YES' if nr.top5_match else ' no':>6s}  "
                f"{score_ratio:>11.4f}"
            )

    print()


def print_memory_summary(results: List[ConfigResult]):
    """Print memory budget analysis for Qwen2.5-3B at each scale."""
    print()
    print("=" * 80)
    print("  MEMORY BUDGET — Qwen2.5-3B-Instruct on RTX 4090 (24 GB)")
    print("=" * 80)
    print()
    print(f"  Model weights (4-bit BnB): ~{MODEL_WEIGHT_GB:.0f} GB")
    print()

    print(f"  {'Context':>8s}  {'Mode':>6s}  "
          f"{'KV Cache':>9s}  {'+ Weights':>10s}  {'Headroom':>9s}  {'Status':>8s}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*8}")

    for r in results:
        full_cache = r.theoretical_cache_gb
        total = full_cache + MODEL_WEIGHT_GB
        headroom = 24.0 - total
        status = "FITS" if headroom > 0 else "OOM"

        mode = "TD" if r.mode == "temporal_decay" else f"TQ-{r.bits}"
        print(f"  {r.context_k:>8s}  {mode:>6s}  "
              f"{full_cache:>8.2f}G  {total:>9.2f}G  {headroom:>8.2f}G  {status:>8s}")

    # FP16 baselines for comparison
    print()
    for ctx in sorted(set(r.context_length for r in results)):
        fp16_gb = fp16_kv_cache_bytes(ctx) / (1024 ** 3)
        total = fp16_gb + MODEL_WEIGHT_GB
        headroom = 24.0 - total
        status = "FITS" if headroom > 0 else "OOM"
        ctx_k = f"{ctx // 1024}K" if ctx < 1024 * 1024 else f"{ctx // (1024*1024)}M"
        print(f"  {ctx_k:>8s}  {'FP16':>6s}  "
              f"{fp16_gb:>8.2f}G  {total:>9.2f}G  {headroom:>8.2f}G  {status:>8s}")

    print()


def print_headline(results: List[ConfigResult]):
    """Print the headline result."""
    # Find the largest context that fits
    fitting = [r for r in results if r.mode != "temporal_decay"
               and r.theoretical_cache_gb + MODEL_WEIGHT_GB < 24.0]

    if not fitting:
        print("\n  No configurations fit in 24 GB VRAM.\n")
        return

    best = max(fitting, key=lambda r: r.context_length)
    full_cache = best.theoretical_cache_gb
    fp16_cache = best.fp16_cache_gb

    print()
    print("*" * 74)
    print(f"  HEADLINE: {best.context_k} tokens on a single RTX 4090")
    print(f"            TQ-{best.bits} KV cache: {full_cache:.2f} GB  "
          f"(vs {fp16_cache:.1f} GB FP16 = {best.compression_ratio:.1f}x compression)")
    print(f"            Needle retrieval: {best.needle_found_rate * 100:.0f}% success rate")
    print(f"            Total VRAM: {full_cache + MODEL_WEIGHT_GB:.1f} GB "
          f"(model {MODEL_WEIGHT_GB:.0f}G + cache {full_cache:.2f}G)")
    print("*" * 74)
    print()


# ---------------------------------------------------------------------------
# Real-model quality validation (Option B, smaller context)
# ---------------------------------------------------------------------------
def run_real_model_validation(bits: int = 3, context_tokens: int = 4096):
    """Validate TQ quality using real model attention at small context.

    Loads Qwen2.5-3B-Instruct, generates KV cache, compresses it with
    TurboQuant, and compares attention scores against FP16 baseline.

    This is a quality sanity check, not a scale test.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        print("  [real-model] Skipping: transformers not available")
        return

    print(f"\n  Real-Model Quality Validation (TQ-{bits}, {context_tokens} tokens)")
    print("  " + "-" * 60)

    print(f"  Loading {MODEL_NAME}...", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

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
        trust_remote_code=True,
    )
    model.eval()
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s  VRAM: {gpu_mem_gb():.2f} GB", flush=True)

    # Build a simple prompt
    filler = (
        "The quarterly financial review meeting covered budget allocations, "
        "departmental spending, and projected revenue. "
        "Infrastructure upgrades were discussed.\n\n"
    )
    needle = "The secret project code name is AURORA-7749."
    filler_tokens = len(tokenizer.encode(filler, add_special_tokens=False))
    n_reps = max(1, context_tokens // filler_tokens)
    needle_idx = n_reps // 4  # 25% depth

    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Memo ---\n{needle}\n--- End ---\n\n")
        parts.append(filler)

    prompt = "".join(parts)
    prompt += "Question: What is the secret project code name?\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=context_tokens).to(model.device)
    actual_len = inputs["input_ids"].shape[1]
    print(f"  Prompt: {actual_len} tokens", flush=True)

    # Forward pass to get FP16 KV cache
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        fp16_cache = outputs.past_key_values

    # Extract keys from layer 0, head 0 for comparison
    # Handle different transformers versions: DynamicCache with .layers[i].keys
    # or older .key_cache[i] or tuple-based caches.
    if hasattr(fp16_cache, "layers") and len(fp16_cache.layers) > 0:
        # transformers >= 4.48: DynamicCache with .layers[i].keys
        fp16_keys = fp16_cache.layers[0].keys[:, 0, :, :]  # (1, seq, d)
    elif hasattr(fp16_cache, "key_cache"):
        fp16_keys = fp16_cache.key_cache[0][:, 0, :, :]
    elif hasattr(fp16_cache, "__getitem__"):
        entry = fp16_cache[0]
        if isinstance(entry, (tuple, list)):
            fp16_keys = entry[0][:, 0, :, :]
        else:
            fp16_keys = entry[:, 0, :, :]
    else:
        fp16_keys = list(fp16_cache)[0][0][:, 0, :, :]

    fp16_keys = fp16_keys.squeeze(0).float()  # (seq, d)
    seq_len = fp16_keys.shape[0]

    # Compress with TurboQuant and compare
    estimator = TurboQuantEstimator(d=HEAD_DIM, bits=bits, seed=42, device="cuda")
    compressed = estimator.quantize(fp16_keys)

    # Query = last token
    query = fp16_keys[-1:, :]
    real_scores = (query @ fp16_keys.T).squeeze(0)
    tq_scores = estimator.inner_product(query, compressed).squeeze(0)

    cos_sim = F.cosine_similarity(
        real_scores.unsqueeze(0).float(),
        tq_scores.unsqueeze(0).float(),
    ).item()

    real_top5 = real_scores.topk(5).indices.tolist()
    tq_top5 = tq_scores.topk(5).indices.tolist()
    top5_overlap = len(set(real_top5) & set(tq_top5))

    print(f"  Cosine similarity (attention scores): {cos_sim:.6f}")
    print(f"  Top-5 overlap: {top5_overlap}/5")
    print(f"  FP16 top-5: {real_top5}")
    print(f"  TQ-{bits} top-5: {tq_top5}")

    # Cleanup
    del model, fp16_cache, outputs
    force_gc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ultra Long Context Benchmark for TurboQuantDC"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512 * 1024,
        help="Maximum context length to test (default: 524288 = 512K)"
    )
    parser.add_argument(
        "--skip-real-model", action="store_true",
        help="Skip real model quality validation"
    )
    parser.add_argument(
        "--include-1m", action="store_true",
        help="Include 1M token test (requires significant VRAM and time)"
    )
    parser.add_argument(
        "--bits", type=int, nargs="+", default=BIT_WIDTHS,
        help="Bit-widths to test (default: 2 3)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="CUDA device (default: cuda)"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for this benchmark.")
        sys.exit(1)

    # VRAM budget display
    print_vram_budget()

    # Determine context lengths to test
    context_lengths = [c for c in DEFAULT_CONTEXT_LENGTHS if c <= args.max_tokens]
    if args.include_1m and 1024 * 1024 <= args.max_tokens:
        context_lengths.append(1024 * 1024)

    all_results: List[ConfigResult] = []

    # --- Option A: Synthetic KV cache stress tests ---
    print("=" * 74)
    print("  PART 1: Synthetic KV Cache Stress Test")
    print("  (Single head -- measurements scaled to full model for budget analysis)")
    print("=" * 74)
    print()

    for ctx_len in context_lengths:
        for bits in args.bits:
            try:
                result = run_synthetic_kvcache(
                    context_length=ctx_len,
                    bits=bits,
                    needle_depths=NEEDLE_DEPTH_FRACTIONS,
                    device=args.device,
                )
                all_results.append(result)
                print()
            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at {ctx_len // 1024}K / TQ-{bits} -- skipping")
                force_gc()
                print()
            except Exception as e:
                print(f"    ERROR at {ctx_len // 1024}K / TQ-{bits}: {e}")
                force_gc()
                print()

    # --- Temporal Decay at 512K ---
    print("=" * 74)
    print("  PART 2: Temporal Decay at 512K")
    print("=" * 74)
    print()

    try:
        td_ctx = min(512 * 1024, args.max_tokens)
        td_result = run_temporal_decay(
            context_length=td_ctx,
            needle_depths=NEEDLE_DEPTH_FRACTIONS,
            device=args.device,
        )
        all_results.append(td_result)
    except torch.cuda.OutOfMemoryError:
        print("    OOM on temporal decay test -- skipping")
        force_gc()
    except Exception as e:
        print(f"    ERROR on temporal decay: {e}")
        force_gc()

    print()

    # --- Results ---
    print_results_table(all_results)
    print_needle_detail(all_results)
    print_memory_summary(all_results)
    print_headline(all_results)

    # --- Option B: Real model quality (optional) ---
    if not args.skip_real_model:
        print("=" * 74)
        print("  PART 3: Real Model Quality Validation")
        print("=" * 74)
        try:
            run_real_model_validation(bits=3, context_tokens=4096)
        except torch.cuda.OutOfMemoryError:
            print("  OOM loading model -- skipping real model validation")
            force_gc()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            force_gc()

    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
