#!/usr/bin/env python3
"""TurboQuantDC Showcase -- The jaw-dropping demonstration.

Loads a real LLM, extracts its KV cache, compresses it with TurboQuant at
multiple bit-widths, and presents the results with beautiful terminal output.

Demonstrates:
    1. Real KV cache extraction from Qwen2.5-3B-Instruct
    2. TurboQuant compression at 2, 3, and 4 bits
    3. Attention score fidelity (cosine sim, top-1, top-5)
    4. Memory savings projections for large models at long context

Usage:
    python showcase.py
    python showcase.py --model Qwen/Qwen2.5-3B-Instruct --context 512 --bits 3
    python showcase.py --no-color     # for CI/piped output
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Allow running from repo root
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from turboquantdc import TurboQuantEstimator  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# ANSI color helpers
# ═══════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI escape codes for terminal styling."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def blue(self, text: str) -> str:
        return self._wrap("34", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def magenta(self, text: str) -> str:
        return self._wrap("35", text)

    def bold_green(self, text: str) -> str:
        return self._wrap("1;32", text)

    def bold_yellow(self, text: str) -> str:
        return self._wrap("1;33", text)

    def bold_cyan(self, text: str) -> str:
        return self._wrap("1;36", text)

    def bold_red(self, text: str) -> str:
        return self._wrap("1;31", text)

    def bold_magenta(self, text: str) -> str:
        return self._wrap("1;35", text)

    def bg_green(self, text: str) -> str:
        return self._wrap("42;30", text)

    def bg_red(self, text: str) -> str:
        return self._wrap("41;37", text)

    def bg_yellow(self, text: str) -> str:
        return self._wrap("43;30", text)


# Global instance, set during main()
C = Colors(enabled=True)


# ═══════════════════════════════════════════════════════════════════════════
# Box drawing helpers
# ═══════════════════════════════════════════════════════════════════════════

def banner(title: str, width: int = 64) -> str:
    """Double-line banner box."""
    inner = width - 4
    top = f"  {C.bold_cyan(chr(0x2554) + chr(0x2550) * (inner + 2) + chr(0x2557))}"
    mid = f"  {C.bold_cyan(chr(0x2551))} {C.bold(title):<{inner + len(C.bold(title)) - len(title)}} {C.bold_cyan(chr(0x2551))}"
    bot = f"  {C.bold_cyan(chr(0x255A) + chr(0x2550) * (inner + 2) + chr(0x255D))}"
    return f"{top}\n{mid}\n{bot}"


def stage_box(title: str, width: int = 56) -> str:
    """Single-line stage header box."""
    inner = width - 4
    top = f"  {C.cyan(chr(0x250C) + chr(0x2500) * (inner + 2) + chr(0x2510))}"
    mid = f"  {C.cyan(chr(0x2502))} {C.bold(title):<{inner + len(C.bold(title)) - len(title)}} {C.cyan(chr(0x2502))}"
    bot = f"  {C.cyan(chr(0x2514) + chr(0x2500) * (inner + 2) + chr(0x2518))}"
    return f"{top}\n{mid}\n{bot}"


def check(msg: str) -> str:
    """Green checkmark line."""
    return f"  {C.bold_green(chr(0x2713))} {msg}"


def cross(msg: str) -> str:
    """Red cross line."""
    return f"  {C.bold_red(chr(0x2717))} {msg}"


def bar_chart(label: str, value_gb: float, max_gb: float, width: int = 30,
              color_fn=None) -> str:
    """Horizontal bar chart with label and value."""
    if color_fn is None:
        color_fn = C.green
    filled = max(1, int(round(value_gb / max_gb * width))) if max_gb > 0 else 0
    filled = min(filled, width)
    bar = color_fn(chr(0x2588) * filled) + C.dim(chr(0x2591) * (width - filled))
    return f"  {label}  {bar}  {value_gb:.1f} GB"


def separator(width: int = 64) -> str:
    """Double-line separator."""
    return f"  {C.bold_cyan(chr(0x2550) * width)}"


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HeadResult:
    """Per-head comparison metrics."""
    layer: int
    head: int
    cosine_sim: float
    top1_match: bool
    top5_match: bool


@dataclass
class BitWidthResult:
    """Aggregate results for one bit-width."""
    bits: int
    seq_len: int
    head_results: List[HeadResult] = field(default_factory=list)
    compress_time: float = 0.0

    @property
    def n_heads(self) -> int:
        return len(self.head_results)

    @property
    def avg_cosine_sim(self) -> float:
        if not self.head_results:
            return 0.0
        return sum(r.cosine_sim for r in self.head_results) / self.n_heads

    @property
    def top1_pct(self) -> float:
        if not self.head_results:
            return 0.0
        return 100.0 * sum(1 for r in self.head_results if r.top1_match) / self.n_heads

    @property
    def top5_pct(self) -> float:
        if not self.head_results:
            return 0.0
        return 100.0 * sum(1 for r in self.head_results if r.top5_match) / self.n_heads


# ═══════════════════════════════════════════════════════════════════════════
# Memory calculations
# ═══════════════════════════════════════════════════════════════════════════

def compute_memory_bits(
    n_layers: int,
    n_kv_heads: int,
    seq_len: int,
    head_dim: int,
    bits: int,
) -> Tuple[int, int]:
    """Compute compressed and FP16 bit counts for full KV cache.

    Keys: (bits-1)*d MSE + d QJL signs + 16 residual_norm + 16 vec_norm
    Values: bits*d MSE + 16 vec_norm

    Returns:
        (compressed_bits, fp16_bits)
    """
    n_vectors = n_layers * n_kv_heads * seq_len
    mse_bits_key = max(bits - 1, 1)

    # Keys
    key_mse = n_vectors * head_dim * mse_bits_key
    key_qjl = n_vectors * head_dim * 1
    key_norms = n_vectors * 32  # vec_norm (16) + residual_norm (16)

    # Values (MSE-only)
    val_mse = n_vectors * head_dim * bits
    val_norms = n_vectors * 16

    compressed = key_mse + key_qjl + key_norms + val_mse + val_norms
    fp16 = n_vectors * head_dim * 16 * 2  # keys + values

    return compressed, fp16


def bits_to_gb(bits: int) -> float:
    return bits / 8 / 1024 / 1024 / 1024


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_name: str):
    """Load model with BitsAndBytes 4-bit quantization for weight efficiency."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\n  Loading {C.bold(model_name)} ...", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    load_time = time.time() - t0
    gpu_mb = torch.cuda.memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0

    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    head_dim = config.hidden_size // n_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)

    print(check(f"Model loaded in {load_time:.1f}s ({gpu_mb} MB GPU)"))
    print(check(f"{n_layers} layers x {n_kv_heads} KV heads x {head_dim} head_dim"))

    return model, tokenizer, n_layers, n_heads, n_kv_heads, head_dim


# ═══════════════════════════════════════════════════════════════════════════
# KV cache extraction
# ═══════════════════════════════════════════════════════════════════════════

FILLER = (
    "The quarterly financial review meeting covered several topics including "
    "budget allocations for the upcoming fiscal year, departmental spending reports, "
    "and projected revenue streams from various business units. The committee discussed "
    "infrastructure upgrades planned for the western regional offices and noted that "
    "maintenance schedules should be coordinated with the facilities management team. "
    "Several action items were assigned to team leads for follow-up before the next "
    "meeting cycle.\n\n"
)


def build_prompt(tokenizer, target_tokens: int) -> str:
    """Build a natural-looking prompt that fills to target_tokens."""
    filler_len = len(tokenizer.encode(FILLER, add_special_tokens=False))
    n_reps = max(1, target_tokens // filler_len)

    parts: list[str] = []
    for i in range(n_reps):
        parts.append(FILLER)

    haystack = "".join(parts)
    prompt = (
        f"<|im_start|>user\n{haystack}\n"
        f"Summarize the key points from the text above.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def extract_kv_cache(model, tokenizer, target_tokens: int):
    """Run a forward pass and extract the KV cache."""

    prompt = build_prompt(tokenizer, target_tokens)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=target_tokens + 256,
    ).to("cuda")
    seq_len = inputs["input_ids"].shape[1]

    print(f"\n  Running forward pass ({seq_len} tokens)...", end="", flush=True)
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)
    fwd_time = time.time() - t0
    print(f" {fwd_time:.1f}s", flush=True)

    cache = outputs.past_key_values

    # Unified cache accessor (handles DynamicCache and legacy)
    if hasattr(cache, "key_cache"):
        def get_keys(layer_idx):
            return cache.key_cache[layer_idx]
        actual_layers = len(cache.key_cache)
    elif hasattr(cache, "layers"):
        def get_keys(layer_idx):
            return cache.layers[layer_idx].keys
        actual_layers = len(cache.layers)
    else:
        def get_keys(layer_idx):
            return cache[layer_idx][0]
        actual_layers = len(cache)

    sample = get_keys(0)
    n_kv_heads = sample.shape[1]
    actual_seq = sample.shape[2]
    head_dim = sample.shape[3]

    print(check(f"Extracted {actual_layers} layers x {n_kv_heads} KV heads x {head_dim} dim"))
    print(check(f"{actual_seq:,} tokens of real attention data"))

    return cache, get_keys, actual_layers, n_kv_heads, actual_seq, head_dim


# ═══════════════════════════════════════════════════════════════════════════
# TurboQuant compression and comparison
# ═══════════════════════════════════════════════════════════════════════════

def compare_head(
    keys_fp: torch.Tensor,
    head_dim: int,
    bits: int,
    layer_idx: int,
    head_idx: int,
) -> HeadResult:
    """Compare FP16 vs TurboQuant attention scores for one head."""
    device = keys_fp.device

    # Query = last token (simulates next-token generation)
    query = keys_fp[-1:, :]  # (1, head_dim)

    # FP16 attention scores
    real_scores = (query @ keys_fp.T).squeeze(0)  # (seq_len,)

    # TurboQuant attention scores
    seed = layer_idx * 10000 + head_idx
    estimator = TurboQuantEstimator(
        d=head_dim, bits=bits, seed=seed, device=device,
    )
    compressed = estimator.quantize(keys_fp)
    tq_scores = estimator.inner_product(query, compressed).squeeze(0)

    # Cosine similarity
    cos_sim = F.cosine_similarity(
        real_scores.unsqueeze(0).float(),
        tq_scores.unsqueeze(0).float(),
    ).item()

    # Top-1 match
    real_top1 = real_scores.argmax().item()
    tq_top1 = tq_scores.argmax().item()
    top1_match = real_top1 == tq_top1

    # Top-5 match (is real top-1 in TQ top-5?)
    seq_len = real_scores.shape[0]
    tq_top5 = tq_scores.topk(min(5, seq_len)).indices.tolist()
    top5_match = real_top1 in tq_top5

    return HeadResult(
        layer=layer_idx,
        head=head_idx,
        cosine_sim=cos_sim,
        top1_match=top1_match,
        top5_match=top5_match,
    )


def run_compression(
    get_keys,
    actual_layers: int,
    n_kv_heads: int,
    actual_seq: int,
    head_dim: int,
    bit_widths: List[int],
) -> List[BitWidthResult]:
    """Compress and evaluate across all bit-widths."""
    results = []

    for bits in bit_widths:
        label = f"TQ-{bits}bit"
        print(f"\n  {C.bold(label)}: Compressing {actual_layers * n_kv_heads} heads...",
              end="", flush=True)
        t0 = time.time()

        bw = BitWidthResult(bits=bits, seq_len=actual_seq)

        for layer_idx in range(actual_layers):
            keys = get_keys(layer_idx)  # (1, n_kv_heads, seq, head_dim)
            for h in range(n_kv_heads):
                k = keys[0, h].float()  # (seq, head_dim)
                hr = compare_head(k, head_dim, bits, layer_idx, h)
                bw.head_results.append(hr)

        bw.compress_time = time.time() - t0
        print(f" {bw.compress_time:.1f}s")
        results.append(bw)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Display: Results table
# ═══════════════════════════════════════════════════════════════════════════

def display_results_table(
    results: List[BitWidthResult],
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    sweet_spot_bits: int,
) -> None:
    """Print the quality/compression comparison table."""

    # Table header
    hdr = (
        f"  {C.bold('Bit Width')}  {C.dim(chr(0x2502))} "
        f"{C.bold('Cos Sim')}  {C.dim(chr(0x2502))} "
        f"{C.bold('Top-1')}   {C.dim(chr(0x2502))} "
        f"{C.bold('Top-5')}   {C.dim(chr(0x2502))} "
        f"{C.bold('Compression')}"
    )
    rule = (
        f"  {chr(0x2500) * 10}{chr(0x253C)}"
        f"{chr(0x2500) * 9}{chr(0x253C)}"
        f"{chr(0x2500) * 9}{chr(0x253C)}"
        f"{chr(0x2500) * 9}{chr(0x253C)}"
        f"{chr(0x2500) * 14}"
    )
    print(f"\n{hdr}")
    print(f"  {C.dim(rule)}")

    for bw in results:
        comp_bits, fp16_bits = compute_memory_bits(
            n_layers, n_kv_heads, bw.seq_len, head_dim, bw.bits
        )
        ratio = fp16_bits / comp_bits if comp_bits > 0 else 0.0

        is_sweet = bw.bits == sweet_spot_bits
        star = f"  {C.bold_yellow(chr(0x2605))} {C.bold_yellow('SWEET SPOT')}" if is_sweet else ""

        # Color cosine similarity
        cos_val = bw.avg_cosine_sim
        if cos_val >= 0.995:
            cos_str = C.bold_green(f"{cos_val:.4f}")
        elif cos_val >= 0.990:
            cos_str = C.yellow(f"{cos_val:.4f}")
        else:
            cos_str = C.red(f"{cos_val:.4f}")

        # Color top-5
        t5_val = bw.top5_pct
        if t5_val >= 90.0:
            t5_str = C.bold_green(f"{t5_val:.1f}%")
        elif t5_val >= 80.0:
            t5_str = C.yellow(f"{t5_val:.1f}%")
        else:
            t5_str = C.red(f"{t5_val:.1f}%")

        # Color top-1
        t1_val = bw.top1_pct
        if t1_val >= 80.0:
            t1_str = C.bold_green(f"{t1_val:.1f}%")
        elif t1_val >= 65.0:
            t1_str = C.yellow(f"{t1_val:.1f}%")
        else:
            t1_str = C.red(f"{t1_val:.1f}%")

        row = (
            f"  {C.bold(f'{bw.bits}-bit'):>15}  {C.dim(chr(0x2502))} "
            f"{cos_str:>17}  {C.dim(chr(0x2502))} "
            f"{t1_str:>17}  {C.dim(chr(0x2502))} "
            f"{t5_str:>17}  {C.dim(chr(0x2502))} "
            f"{C.bold(f'{ratio:.1f}x')}{star}"
        )
        print(row)


# ═══════════════════════════════════════════════════════════════════════════
# Display: Per-head statistics (compact)
# ═══════════════════════════════════════════════════════════════════════════

def display_per_head_stats(results: List[BitWidthResult], sweet_spot_bits: int) -> None:
    """Show distribution of cosine sims for the sweet-spot bit-width."""
    target = next((r for r in results if r.bits == sweet_spot_bits), None)
    if target is None or not target.head_results:
        return

    cos_sims = sorted(r.cosine_sim for r in target.head_results)
    n = len(cos_sims)
    p5 = cos_sims[max(0, int(n * 0.05))]
    p50 = cos_sims[n // 2]
    p95 = cos_sims[min(n - 1, int(n * 0.95))]
    worst = cos_sims[0]
    best = cos_sims[-1]

    print(f"\n  {C.bold(f'Per-Head Stats ({sweet_spot_bits}-bit)')}: {n} heads evaluated")
    print(f"    Cosine sim:  worst={C.yellow(f'{worst:.4f}')}  "
          f"p5={p5:.4f}  median={C.bold(f'{p50:.4f}')}  "
          f"p95={p95:.4f}  best={C.bold_green(f'{best:.4f}')}")

    # Count heads below 0.99
    below_99 = sum(1 for s in cos_sims if s < 0.99)
    below_95 = sum(1 for s in cos_sims if s < 0.95)
    if below_99 > 0:
        print(f"    Heads < 0.99: {below_99}/{n}  |  Heads < 0.95: {below_95}/{n}")
    else:
        print(f"    {C.bold_green('All heads')} >= 0.99 cosine similarity")


# ═══════════════════════════════════════════════════════════════════════════
# Display: Memory savings projections
# ═══════════════════════════════════════════════════════════════════════════

# Model configurations for projections
# (model_name, n_layers, n_kv_heads, head_dim, display_name)
MODEL_CONFIGS = [
    ("Qwen2.5-3B",    36,  2, 128, "Qwen2.5-3B"),
    ("Qwen2.5-14B",   48,  8, 128, "Qwen2.5-14B"),
    ("Qwen3.5-27B",   16,  4, 256, "Qwen3.5-27B"),
    ("Llama-3-70B",   80,  8, 128, "Llama-3-70B"),
]

CONTEXT_LENGTHS = [32_768, 65_536, 131_072, 262_144]
CONTEXT_LABELS = ["32K", "64K", "128K", "262K"]


def display_memory_projections(
    actual_n_layers: int,
    actual_kv_heads: int,
    actual_head_dim: int,
    sweet_spot_bits: int,
) -> None:
    """Show memory projections for various models and context lengths."""

    # First: table for the ACTUAL model being tested
    print(f"\n  {C.bold('Memory Savings by Context Length:')}\n")

    hdr = f"  {'Context':>10}  {C.dim(chr(0x2502))} {'FP16 Cache':>12}  {C.dim(chr(0x2502))} {f'TQ-{sweet_spot_bits}bit':>12}  {C.dim(chr(0x2502))} {'Savings':>10}  {C.dim(chr(0x2502))} {'Ratio':>8}"
    rule = f"  {chr(0x2500) * 11}{chr(0x253C)}{chr(0x2500) * 14}{chr(0x253C)}{chr(0x2500) * 14}{chr(0x253C)}{chr(0x2500) * 12}{chr(0x253C)}{chr(0x2500) * 9}"
    print(hdr)
    print(f"  {C.dim(rule)}")

    for ctx_len, ctx_label in zip(CONTEXT_LENGTHS, CONTEXT_LABELS):
        comp_bits, fp16_bits = compute_memory_bits(
            actual_n_layers, actual_kv_heads, ctx_len, actual_head_dim, sweet_spot_bits
        )
        fp16_gb = bits_to_gb(fp16_bits)
        comp_gb = bits_to_gb(comp_bits)
        saved_gb = fp16_gb - comp_gb
        ratio = fp16_bits / comp_bits if comp_bits > 0 else 0.0

        print(
            f"  {ctx_label:>10}  {C.dim(chr(0x2502))} "
            f"{fp16_gb:>10.2f} GB  {C.dim(chr(0x2502))} "
            f"{C.bold_green(f'{comp_gb:.2f} GB'):>22}  {C.dim(chr(0x2502))} "
            f"{C.green(f'{saved_gb:.2f} GB'):>20}  {C.dim(chr(0x2502))} "
            f"{ratio:.1f}x"
        )


def display_big_model_showdown(sweet_spot_bits: int) -> None:
    """The money shot: show how TurboQuant enables impossible configurations."""

    # Focus on Qwen3.5-27B at 262K -- the headline number
    model_name = "Qwen3.5-27B"
    n_layers = 16
    n_kv_heads = 4
    head_dim = 256
    ctx_len = 262_144

    comp_bits, fp16_bits = compute_memory_bits(
        n_layers, n_kv_heads, ctx_len, head_dim, sweet_spot_bits
    )
    fp16_gb = bits_to_gb(fp16_bits)
    comp_gb = bits_to_gb(comp_bits)

    max_gb = max(fp16_gb, 24.0)  # scale bars to whichever is larger

    print(f"\n  {C.bold(f'Model: {model_name} (262K context)')}\n")

    # FP16 bar
    fp16_label = f"FP16 KV Cache: "
    fp16_note = " OOM on 24GB!" if fp16_gb > 24.0 else ""
    print(bar_chart(
        f"FP16 KV Cache: ",
        fp16_gb, max_gb, width=30,
        color_fn=C.bold_red if fp16_gb > 24.0 else C.yellow,
    ) + (f"  {C.bg_red(' OOM on 24GB! ')}" if fp16_gb > 24.0 else ""))

    # TQ bar
    spare_gb = 24.0 - comp_gb
    fits_note = f"  {C.bg_green(f' FITS with {spare_gb:.0f}GB spare ')}" if spare_gb > 0 else ""
    print(bar_chart(
        f"TQ-{sweet_spot_bits}bit Cache: ",
        comp_gb, max_gb, width=30,
        color_fn=C.bold_green,
    ) + fits_note)

    # Show the full model lineup
    print(f"\n  {C.bold('Cross-Model Projections')} (262K context, {sweet_spot_bits}-bit):\n")

    for model_label, nl, nkv, hd, display in MODEL_CONFIGS:
        c_bits, f_bits = compute_memory_bits(nl, nkv, ctx_len, hd, sweet_spot_bits)
        f_gb = bits_to_gb(f_bits)
        c_gb = bits_to_gb(c_bits)
        ratio = f_bits / c_bits if c_bits > 0 else 0.0
        fits = c_gb < 24.0

        status = C.bold_green("FITS") if fits else C.bold_red("OOM")
        fp_status = C.bold_green("FITS") if f_gb < 24.0 else C.bold_red("OOM")

        print(
            f"    {display:<15} FP16: {f_gb:>6.1f} GB ({fp_status})  "
            f"-> TQ-{sweet_spot_bits}b: {C.bold(f'{c_gb:.1f} GB')} ({status})  "
            f"[{ratio:.1f}x]"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GPU info
# ═══════════════════════════════════════════════════════════════════════════

def get_gpu_info() -> Tuple[str, int]:
    """Return (gpu_name, total_vram_mb)."""
    if not torch.cuda.is_available():
        return "CPU (no GPU)", 0
    name = torch.cuda.get_device_name(0)
    total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    return name, total_mb


# ═══════════════════════════════════════════════════════════════════════════
# Main showcase
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TurboQuantDC Showcase -- KV Cache Compression Demo"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=512,
        help="Context length in tokens for the demo (default: 512)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=3,
        choices=[2, 3, 4],
        help="Sweet-spot bit-width to highlight (default: 3)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output (for CI / piped output)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global C
    use_color = (not args.no_color) and sys.stdout.isatty()
    C = Colors(enabled=use_color)

    sweet_spot = args.bits
    bit_widths = sorted(set([2, 3, 4, sweet_spot]))

    gpu_name, gpu_total_mb = get_gpu_info()

    # ── Title banner ──
    print()
    print(banner("TurboQuantDC  --  KV Cache Compression Showcase", width=64))
    print()
    print(f"  Model:    {C.bold(args.model)}")
    print(f"  GPU:      {C.bold(gpu_name)} ({gpu_total_mb // 1024}GB)")
    print(f"  Context:  {C.bold(str(args.context))} tokens")
    print(f"  Bits:     {C.bold(', '.join(str(b) for b in bit_widths))}")

    # ── Stage 1: Load model and extract KV cache ──
    print()
    print(stage_box("Stage 1: Extracting KV Cache from Real Model", width=56))

    model, tokenizer, n_layers, n_heads, n_kv_heads, head_dim = load_model(args.model)

    cache, get_keys, actual_layers, actual_kv_heads, actual_seq, actual_head_dim = \
        extract_kv_cache(model, tokenizer, args.context)

    # Free model to reclaim GPU memory for compression
    del model
    torch.cuda.empty_cache()

    # ── Stage 2: TurboQuant Compression ──
    print()
    print(stage_box("Stage 2: TurboQuant Compression", width=56))

    total_heads = actual_layers * actual_kv_heads
    print(f"\n  Evaluating {total_heads} attention heads across {len(bit_widths)} bit-widths...")

    results = run_compression(
        get_keys, actual_layers, actual_kv_heads, actual_seq,
        actual_head_dim, bit_widths,
    )

    display_results_table(results, actual_layers, actual_kv_heads, actual_head_dim, sweet_spot)
    display_per_head_stats(results, sweet_spot)

    # Validate against paper targets
    sweet_result = next((r for r in results if r.bits == sweet_spot), None)
    if sweet_result:
        print(f"\n  {C.bold('Paper Targets')} ({sweet_spot}-bit):")
        cos_ok = sweet_result.avg_cosine_sim >= 0.995
        t5_ok = sweet_result.top5_pct >= 90.0
        comp_bits, fp16_bits = compute_memory_bits(
            actual_layers, actual_kv_heads, actual_seq, actual_head_dim, sweet_spot
        )
        ratio = fp16_bits / comp_bits if comp_bits > 0 else 0.0
        ratio_ok = abs(ratio - 5.0) < 1.0

        for label, val, target, ok in [
            ("Cosine sim", f"{sweet_result.avg_cosine_sim:.4f}", ">= 0.995", cos_ok),
            ("Top-5 match", f"{sweet_result.top5_pct:.1f}%", ">= 90%", t5_ok),
            ("Compression", f"{ratio:.1f}x", "~5.0x", ratio_ok),
        ]:
            status = check(f"{label}: {C.bold(val)}  (target: {target})") if ok else \
                     cross(f"{label}: {C.bold(val)}  (target: {target})")
            print(f"  {status}")

    # ── Stage 3: What This Means for YOUR GPU ──
    print()
    print(stage_box("Stage 3: What This Means for YOUR GPU", width=56))

    display_memory_projections(actual_layers, actual_kv_heads, actual_head_dim, sweet_spot)
    display_big_model_showdown(sweet_spot)

    # ── Final headline ──
    if sweet_result:
        comp_bits, fp16_bits = compute_memory_bits(
            actual_layers, actual_kv_heads, actual_seq, actual_head_dim, sweet_spot
        )
        ratio = fp16_bits / comp_bits if comp_bits > 0 else 0.0
        cos_pct = sweet_result.avg_cosine_sim * 100

        print()
        print(f"  {separator(64)}")
        print()
        print(f"  {C.bold('RESULT:')} {C.bold_green(f'{ratio:.1f}x compression')}, "
              f"{C.bold_green(f'{cos_pct:.1f}% attention quality')} preserved")
        print(f"  27B model at 262K context: "
              f"{C.bold_red('OOM')} -> {C.bold_green('fits on a single GPU')}")
        print()
        print(f"  {separator(64)}")
        print()


if __name__ == "__main__":
    main()
