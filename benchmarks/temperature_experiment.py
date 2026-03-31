"""TEMPCAL: Temperature calibration for TurboQuant KV cache compression.

Hypothesis: Compressed KV cache changes the distribution of attention scores
(wider/narrower variance). Adjusting the softmax temperature (scaling factor)
might compensate, producing correct attention patterns even with noisy scores.

This experiment:
  1. Measures how TQ-4 compression affects attention score distributions
     (mean, std, max, range, softmax entropy) per layer.
  2. Tests temperature compensation by scaling dequantized keys, effectively
     changing the softmax temperature without modifying the model.
  3. Tries adaptive per-layer temperature based on variance ratios.
  4. Evaluates generation quality with the best temperature.

Usage:
    cd /home/dhawal/turboQuantDC && python benchmarks/temperature_experiment.py
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from turboquantdc import TurboQuantEstimator  # noqa: E402
from turboquantdc.hf_integration import TurboQuantCache, TurboQuantLayer  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BITS = 4  # TQ-4 as specified

# Temperature multipliers applied to 1/sqrt(d)
TEMPERATURE_SCALES = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]

# Prompts for generation quality test
QUALITY_PROMPTS = [
    "Explain the Pythagorean theorem in simple terms.",
    "What causes the seasons on Earth?",
    "Write a haiku about programming.",
    "List the first five prime numbers and explain why they are prime.",
    "What is the difference between a stack and a queue in computer science?",
]

MAX_NEW_TOKENS = 100


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class LayerScoreStats:
    """Attention score statistics for one layer."""
    layer: int
    # FP16 stats (averaged across heads)
    fp16_mean: float = 0.0
    fp16_std: float = 0.0
    fp16_max: float = 0.0
    fp16_range: float = 0.0
    fp16_entropy: float = 0.0
    # TQ stats
    tq_mean: float = 0.0
    tq_std: float = 0.0
    tq_max: float = 0.0
    tq_range: float = 0.0
    tq_entropy: float = 0.0
    # Derived
    std_ratio: float = 0.0  # fp16_std / tq_std
    suggested_temp: float = 1.0  # sqrt(var_fp16 / var_tq)


@dataclass
class TemperatureResult:
    """Generation quality at a specific temperature."""
    temperature: float
    outputs: List[str] = field(default_factory=list)
    avg_token_overlap: float = 0.0
    avg_ngram_overlap: float = 0.0
    coherent_count: int = 0  # Number of outputs judged coherent


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def softmax_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute entropy of softmax distribution: H = -sum(p * log(p))."""
    probs = F.softmax(logits, dim=dim)
    log_probs = F.log_softmax(logits, dim=dim)
    return -(probs * log_probs).sum(dim=dim)


def compute_token_overlap(a: str, b: str) -> float:
    """Jaccard similarity of whitespace-tokenized outputs."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def compute_ngram_overlap(a: str, b: str, n: int = 4) -> float:
    """N-gram overlap between two outputs."""
    words_a = a.lower().split()
    words_b = b.lower().split()
    if len(words_a) < n or len(words_b) < n:
        return compute_token_overlap(a, b)
    ngrams_a = set(tuple(words_a[i:i+n]) for i in range(len(words_a) - n + 1))
    ngrams_b = set(tuple(words_b[i:i+n]) for i in range(len(words_b) - n + 1))
    if not ngrams_a or not ngrams_b:
        return 0.0
    return len(ngrams_a & ngrams_b) / max(len(ngrams_a), len(ngrams_b))


def is_coherent(text: str) -> bool:
    """Simple heuristic: output is coherent if it has reasonable word diversity
    and doesn't repeat the same few tokens excessively."""
    words = text.lower().split()
    if len(words) < 5:
        return False
    unique = set(words)
    # If less than 20% of words are unique, likely degenerate repetition
    if len(unique) / len(words) < 0.2:
        return False
    # Check for excessive single-word repetition
    from collections import Counter
    counts = Counter(words)
    most_common_freq = counts.most_common(1)[0][1]
    if most_common_freq > len(words) * 0.5:
        return False
    return True


# ---------------------------------------------------------------------------
# Step 1: Measure score distributions
# ---------------------------------------------------------------------------

def measure_score_distributions(
    model, tokenizer
) -> Tuple[List[LayerScoreStats], Dict]:
    """Compare FP16 vs TQ-4 attention score statistics per layer.

    Runs a forward pass, extracts the KV cache, then for each layer:
    - Compute Q @ K^T attention scores with FP16 keys
    - Compress keys with TQ-4, dequantize, compute scores again
    - Compare statistics (mean, std, max, range, entropy)

    Returns:
        (per_layer_stats, metadata_dict)
    """
    print("=" * 70)
    print("Step 1: Measuring FP16 vs TQ-4 attention score distributions")
    print("=" * 70)

    # Build a moderate-length prompt
    prompt_text = (
        "The research team discovered that quantum entanglement could be "
        "maintained over distances exceeding one hundred kilometers using "
        "specialized fiber optic cables cooled to near absolute zero. "
        "This breakthrough in quantum communication technology paves the way "
        "for unhackable long-distance networks. "
    ) * 4
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]
    print(f"  Prompt tokens: {seq_len}")

    # Forward pass to extract KV cache
    print("  Running forward pass...", end="", flush=True)
    t0 = time.time()
    with torch.no_grad():
        output = model(**inputs, use_cache=True)
    print(f" {time.time() - t0:.1f}s")

    kv_cache = output.past_key_values

    # Extract cache info
    if hasattr(kv_cache, "key_cache"):
        num_layers = len(kv_cache.key_cache)
        get_keys = lambda i: kv_cache.key_cache[i]
    elif hasattr(kv_cache, "layers"):
        num_layers = len(kv_cache.layers)
        get_keys = lambda i: kv_cache.layers[i].keys
    else:
        num_layers = len(kv_cache)
        get_keys = lambda i: kv_cache[i][0]

    sample_keys = get_keys(0)
    head_dim = sample_keys.shape[-1]
    num_kv_heads = sample_keys.shape[1]
    actual_seq = sample_keys.shape[2]

    print(f"  Cache: {num_layers} layers, {num_kv_heads} KV heads, "
          f"seq={actual_seq}, head_dim={head_dim}")

    scale = 1.0 / math.sqrt(head_dim)

    layer_stats: List[LayerScoreStats] = []

    print(f"\n  {'Layer':>5} | {'FP16 std':>9} | {'TQ-4 std':>9} | "
          f"{'Ratio':>6} | {'FP16 H':>7} | {'TQ-4 H':>7} | {'Sug. T':>6}")
    print(f"  {'-'*5}-+-{'-'*9}-+-{'-'*9}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}")

    for layer_idx in range(num_layers):
        keys_4d = get_keys(layer_idx).float()  # (1, n_kv_heads, seq, d)
        # Use last token as query for each head
        query = keys_4d[:, :, -1:, :]  # (1, n_kv_heads, 1, d)

        # FP16 scores: (1, n_kv_heads, 1, seq)
        scores_fp16 = torch.matmul(query, keys_4d.transpose(-1, -2)).squeeze(2)
        scores_fp16_scaled = scores_fp16 * scale  # (1, n_kv_heads, seq)

        # TQ-4 compressed scores
        device = str(keys_4d.device)
        est = TurboQuantEstimator(d=head_dim, bits=BITS, seed=42, device=device)

        # Compress all keys (flatten across batch and heads)
        keys_flat = keys_4d.reshape(-1, head_dim)  # (n_kv_heads * seq, d)
        key_norms = keys_flat.norm(dim=-1, keepdim=True)
        keys_norm = keys_flat / (key_norms + 1e-8)
        indices = est.polar.quantize(keys_norm)
        keys_recon = est.polar.dequantize(indices) * key_norms
        keys_recon_4d = keys_recon.reshape(keys_4d.shape)

        scores_tq = torch.matmul(query, keys_recon_4d.transpose(-1, -2)).squeeze(2)
        scores_tq_scaled = scores_tq * scale

        # Compute statistics (averaged across heads)
        fp16_mean = scores_fp16_scaled.mean().item()
        fp16_std = scores_fp16_scaled.std().item()
        fp16_max = scores_fp16_scaled.max().item()
        fp16_range = (scores_fp16_scaled.max() - scores_fp16_scaled.min()).item()
        fp16_ent = softmax_entropy(scores_fp16_scaled, dim=-1).mean().item()

        tq_mean = scores_tq_scaled.mean().item()
        tq_std = scores_tq_scaled.std().item()
        tq_max = scores_tq_scaled.max().item()
        tq_range = (scores_tq_scaled.max() - scores_tq_scaled.min()).item()
        tq_ent = softmax_entropy(scores_tq_scaled, dim=-1).mean().item()

        # Variance ratio and suggested temperature
        std_ratio = fp16_std / tq_std if tq_std > 1e-8 else 1.0
        suggested_temp = math.sqrt(fp16_std**2 / tq_std**2) if tq_std > 1e-8 else 1.0

        stat = LayerScoreStats(
            layer=layer_idx,
            fp16_mean=fp16_mean, fp16_std=fp16_std, fp16_max=fp16_max,
            fp16_range=fp16_range, fp16_entropy=fp16_ent,
            tq_mean=tq_mean, tq_std=tq_std, tq_max=tq_max,
            tq_range=tq_range, tq_entropy=tq_ent,
            std_ratio=std_ratio, suggested_temp=suggested_temp,
        )
        layer_stats.append(stat)

        print(f"  {layer_idx:>5} | {fp16_std:>9.4f} | {tq_std:>9.4f} | "
              f"{std_ratio:>6.3f} | {fp16_ent:>7.2f} | {tq_ent:>7.2f} | "
              f"{suggested_temp:>6.3f}")

        del est
        gc.collect()

    # Summary
    avg_std_ratio = sum(s.std_ratio for s in layer_stats) / len(layer_stats)
    avg_suggested = sum(s.suggested_temp for s in layer_stats) / len(layer_stats)
    print(f"\n  Average std ratio (FP16/TQ): {avg_std_ratio:.4f}")
    print(f"  Average suggested temperature: {avg_suggested:.4f}")

    metadata = {
        "seq_len": actual_seq,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "bits": BITS,
        "avg_std_ratio": avg_std_ratio,
        "avg_suggested_temp": avg_suggested,
    }

    del output, kv_cache
    gc.collect()
    torch.cuda.empty_cache()

    return layer_stats, metadata


# ---------------------------------------------------------------------------
# Step 2: Temperature-compensated TurboQuantCache
# ---------------------------------------------------------------------------

class TemperatureScaledCache(TurboQuantCache):
    """TurboQuantCache with temperature scaling on dequantized keys.

    After dequantizing keys, scales them by temperature_scale. This
    effectively changes the softmax temperature without modifying the model.

    Scaling keys by T changes scores from <q, k>/sqrt(d) to T*<q, k>/sqrt(d),
    which is equivalent to dividing the softmax temperature by T.

    Args:
        bits: Bit-width for compression.
        temperature: Global temperature scale applied to dequantized keys.
        per_layer_temps: Optional dict of {layer_idx: temperature} for
            adaptive per-layer temperature.
        seed: Random seed.
        mse_only: If True, skip QJL for keys.
    """

    def __init__(
        self,
        bits: int = 3,
        temperature: float = 1.0,
        per_layer_temps: Optional[Dict[int, float]] = None,
        seed: int = 42,
        mse_only: bool = True,
    ):
        super().__init__(bits=bits, seed=seed, mse_only=mse_only)
        self.temperature = temperature
        self.per_layer_temps = per_layer_temps or {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override: compress, dequantize, then scale keys by temperature."""
        # Lazily create layers as needed
        while len(self._layers) <= layer_idx:
            self._layers.append(self._make_layer(len(self._layers)))

        tq_layer = self._layers[layer_idx]

        # Do the normal TurboQuantLayer.update (compress + dequantize)
        keys_out, vals_out = tq_layer.update(key_states, value_states)

        # Apply temperature scaling to keys
        temp = self.per_layer_temps.get(layer_idx, self.temperature)
        if abs(temp - 1.0) > 1e-6:
            keys_out = keys_out * temp

        return keys_out, vals_out


# ---------------------------------------------------------------------------
# Step 2: Test temperature compensation
# ---------------------------------------------------------------------------

def test_temperature_scales(
    model, tokenizer, fp16_outputs: List[str]
) -> List[TemperatureResult]:
    """Test different temperature scales with TQ-4 mse_only cache.

    For each scale in TEMPERATURE_SCALES, creates a TemperatureScaledCache,
    generates with the same 5 prompts, and compares output quality to FP16.

    Args:
        model: HF model.
        tokenizer: HF tokenizer.
        fp16_outputs: Reference FP16 outputs for quality comparison.

    Returns:
        List of TemperatureResult, one per scale.
    """
    print("\n" + "=" * 70)
    print("Step 2: Testing temperature compensation")
    print("=" * 70)
    print(f"  Bit-width: TQ-{BITS} (mse_only)")
    print(f"  Scales: {TEMPERATURE_SCALES}")
    print()

    results: List[TemperatureResult] = []

    for temp in TEMPERATURE_SCALES:
        print(f"  Temperature = {temp:.2f} ...", end="", flush=True)
        t0 = time.time()

        tr = TemperatureResult(temperature=temp)

        for i, prompt in enumerate(QUALITY_PROMPTS):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]

            cache = TemperatureScaledCache(
                bits=BITS, temperature=temp, mse_only=True,
            )

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    past_key_values=cache,
                )

            text_out = tokenizer.decode(
                output[0, input_len:], skip_special_tokens=True,
            )
            tr.outputs.append(text_out)

            del cache, output
            gc.collect()
            torch.cuda.empty_cache()

        # Compute quality metrics
        overlaps = []
        ngrams = []
        coherent = 0
        for fp16_out, tq_out in zip(fp16_outputs, tr.outputs):
            overlaps.append(compute_token_overlap(fp16_out, tq_out))
            ngrams.append(compute_ngram_overlap(fp16_out, tq_out))
            if is_coherent(tq_out):
                coherent += 1

        tr.avg_token_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        tr.avg_ngram_overlap = sum(ngrams) / len(ngrams) if ngrams else 0
        tr.coherent_count = coherent

        elapsed = time.time() - t0
        print(f" {elapsed:.1f}s | overlap={tr.avg_token_overlap:.3f} "
              f"| 4-gram={tr.avg_ngram_overlap:.3f} "
              f"| coherent={tr.coherent_count}/{len(QUALITY_PROMPTS)}")

        results.append(tr)

    return results


# ---------------------------------------------------------------------------
# Step 3: Adaptive per-layer temperature
# ---------------------------------------------------------------------------

def test_adaptive_temperature(
    model, tokenizer, fp16_outputs: List[str],
    layer_stats: List[LayerScoreStats],
) -> TemperatureResult:
    """Test per-layer temperature based on variance ratios.

    Sets temperature_scale[layer] = sqrt(var_fp16 / var_tq) for each layer,
    so that compressed scores have the same variance as FP16 scores.

    Args:
        model: HF model.
        tokenizer: HF tokenizer.
        fp16_outputs: Reference FP16 outputs.
        layer_stats: Per-layer stats from Step 1.

    Returns:
        TemperatureResult for the adaptive configuration.
    """
    print("\n" + "=" * 70)
    print("Step 3: Adaptive per-layer temperature")
    print("=" * 70)

    # Build per-layer temperature dict
    per_layer_temps = {}
    for stat in layer_stats:
        per_layer_temps[stat.layer] = stat.suggested_temp

    # Clip extreme values to avoid instability
    clipped = {}
    for layer_idx, temp in per_layer_temps.items():
        clipped[layer_idx] = max(0.5, min(2.0, temp))

    avg_temp = sum(clipped.values()) / len(clipped)
    min_temp = min(clipped.values())
    max_temp = max(clipped.values())
    print(f"  Per-layer temps: avg={avg_temp:.4f}, min={min_temp:.4f}, max={max_temp:.4f}")

    # Print a sample of per-layer temps
    sample_layers = list(range(0, len(layer_stats), max(1, len(layer_stats) // 8)))
    for l in sample_layers:
        print(f"    Layer {l:>3}: temp = {clipped[l]:.4f} "
              f"(std ratio = {layer_stats[l].std_ratio:.4f})")

    print(f"\n  Generating with adaptive temperatures...", end="", flush=True)
    t0 = time.time()

    tr = TemperatureResult(temperature=-1.0)  # -1 signals adaptive

    for i, prompt in enumerate(QUALITY_PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        cache = TemperatureScaledCache(
            bits=BITS,
            temperature=1.0,  # Default, overridden by per_layer_temps
            per_layer_temps=clipped,
            mse_only=True,
        )

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                past_key_values=cache,
            )

        text_out = tokenizer.decode(
            output[0, input_len:], skip_special_tokens=True,
        )
        tr.outputs.append(text_out)

        del cache, output
        gc.collect()
        torch.cuda.empty_cache()

    # Compute quality metrics
    overlaps = []
    ngrams = []
    coherent = 0
    for fp16_out, tq_out in zip(fp16_outputs, tr.outputs):
        overlaps.append(compute_token_overlap(fp16_out, tq_out))
        ngrams.append(compute_ngram_overlap(fp16_out, tq_out))
        if is_coherent(tq_out):
            coherent += 1

    tr.avg_token_overlap = sum(overlaps) / len(overlaps)
    tr.avg_ngram_overlap = sum(ngrams) / len(ngrams)
    tr.coherent_count = coherent

    elapsed = time.time() - t0
    print(f" {elapsed:.1f}s")
    print(f"  Results: overlap={tr.avg_token_overlap:.3f} "
          f"| 4-gram={tr.avg_ngram_overlap:.3f} "
          f"| coherent={tr.coherent_count}/{len(QUALITY_PROMPTS)}")

    return tr


# ---------------------------------------------------------------------------
# Generate FP16 baselines
# ---------------------------------------------------------------------------

def generate_fp16_baselines(model, tokenizer) -> List[str]:
    """Generate FP16 baseline outputs for quality comparison."""
    print("\n" + "=" * 70)
    print("Generating FP16 baseline outputs")
    print("=" * 70)

    outputs = []
    for i, prompt in enumerate(QUALITY_PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        text_out = tokenizer.decode(
            output[0, input_len:], skip_special_tokens=True,
        )
        outputs.append(text_out)
        print(f"  Prompt {i+1}: {text_out[:100]}...")

        del output
        gc.collect()
        torch.cuda.empty_cache()

    return outputs


# ---------------------------------------------------------------------------
# Generate TQ-4 baseline (no temperature adjustment, mse_only)
# ---------------------------------------------------------------------------

def generate_tq_baseline(model, tokenizer, fp16_outputs: List[str]) -> TemperatureResult:
    """Generate TQ-4 mse_only baseline (temperature=1.0) for comparison."""
    print("\n" + "=" * 70)
    print("Generating TQ-4 mse_only baseline (temperature=1.0)")
    print("=" * 70)

    tr = TemperatureResult(temperature=1.0)

    for i, prompt in enumerate(QUALITY_PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        cache = TurboQuantCache(bits=BITS, mse_only=True)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                past_key_values=cache,
            )

        text_out = tokenizer.decode(
            output[0, input_len:], skip_special_tokens=True,
        )
        tr.outputs.append(text_out)
        print(f"  Prompt {i+1}: {text_out[:100]}...")

        del cache, output
        gc.collect()
        torch.cuda.empty_cache()

    overlaps = []
    ngrams = []
    coherent = 0
    for fp16_out, tq_out in zip(fp16_outputs, tr.outputs):
        overlaps.append(compute_token_overlap(fp16_out, tq_out))
        ngrams.append(compute_ngram_overlap(fp16_out, tq_out))
        if is_coherent(tq_out):
            coherent += 1

    tr.avg_token_overlap = sum(overlaps) / len(overlaps)
    tr.avg_ngram_overlap = sum(ngrams) / len(ngrams)
    tr.coherent_count = coherent
    print(f"\n  Baseline: overlap={tr.avg_token_overlap:.3f} "
          f"| 4-gram={tr.avg_ngram_overlap:.3f} "
          f"| coherent={tr.coherent_count}/{len(QUALITY_PROMPTS)}")

    return tr


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def print_final_report(
    layer_stats: List[LayerScoreStats],
    metadata: Dict,
    tq_baseline: TemperatureResult,
    temp_results: List[TemperatureResult],
    adaptive_result: TemperatureResult,
    fp16_outputs: List[str],
):
    """Print the final experiment report."""
    print("\n" + "=" * 70)
    print("TEMPCAL: Temperature Calibration Experiment Results")
    print("=" * 70)

    # --- Score distribution summary ---
    print("\n--- Score Distribution Analysis ---")
    avg_fp16_std = sum(s.fp16_std for s in layer_stats) / len(layer_stats)
    avg_tq_std = sum(s.tq_std for s in layer_stats) / len(layer_stats)
    avg_ratio = sum(s.std_ratio for s in layer_stats) / len(layer_stats)
    print(f"  Avg FP16 score std: {avg_fp16_std:.6f}")
    print(f"  Avg TQ-{BITS} score std:  {avg_tq_std:.6f}")
    print(f"  Avg std ratio (FP16/TQ): {avg_ratio:.4f}")
    if avg_ratio > 1.0:
        print(f"  -> TQ-{BITS} scores have LOWER variance than FP16")
        print(f"     (compression smooths out score differences)")
    else:
        print(f"  -> TQ-{BITS} scores have HIGHER variance than FP16")
        print(f"     (quantization noise widens the score distribution)")

    # --- Temperature sweep results ---
    print("\n--- Temperature Sweep (fixed global temperature) ---")
    print(f"  {'Temp':>6} | {'Overlap':>8} | {'4-gram':>7} | {'Coherent':>8}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}")

    # Find best temperature by overlap
    best_temp_result = max(temp_results, key=lambda r: r.avg_token_overlap)
    best_ngram_result = max(temp_results, key=lambda r: r.avg_ngram_overlap)

    for tr in temp_results:
        marker = " <-- best" if tr.temperature == best_temp_result.temperature else ""
        print(f"  {tr.temperature:>6.2f} | {tr.avg_token_overlap:>8.3f} | "
              f"{tr.avg_ngram_overlap:>7.3f} | "
              f"{tr.coherent_count:>3}/{len(QUALITY_PROMPTS)}{marker}")

    # --- Adaptive result ---
    print(f"\n--- Adaptive Per-Layer Temperature ---")
    print(f"  Overlap: {adaptive_result.avg_token_overlap:.3f} | "
          f"4-gram: {adaptive_result.avg_ngram_overlap:.3f} | "
          f"Coherent: {adaptive_result.coherent_count}/{len(QUALITY_PROMPTS)}")

    # --- Comparison ---
    print(f"\n--- Summary Comparison ---")
    print(f"  {'Config':<25} | {'Overlap':>8} | {'4-gram':>7} | {'Coherent':>8}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}")
    print(f"  {'TQ-4 mse_only (T=1.0)':<25} | {tq_baseline.avg_token_overlap:>8.3f} | "
          f"{tq_baseline.avg_ngram_overlap:>7.3f} | "
          f"{tq_baseline.coherent_count:>3}/{len(QUALITY_PROMPTS)}")
    print(f"  {'Best fixed T=' + f'{best_temp_result.temperature:.2f}':<25} | "
          f"{best_temp_result.avg_token_overlap:>8.3f} | "
          f"{best_temp_result.avg_ngram_overlap:>7.3f} | "
          f"{best_temp_result.coherent_count:>3}/{len(QUALITY_PROMPTS)}")
    print(f"  {'Adaptive per-layer':<25} | {adaptive_result.avg_token_overlap:>8.3f} | "
          f"{adaptive_result.avg_ngram_overlap:>7.3f} | "
          f"{adaptive_result.coherent_count:>3}/{len(QUALITY_PROMPTS)}")

    # --- Conclusion ---
    print(f"\n--- Conclusion ---")
    baseline_overlap = tq_baseline.avg_token_overlap
    best_fixed_overlap = best_temp_result.avg_token_overlap
    adaptive_overlap = adaptive_result.avg_token_overlap

    improvement_fixed = best_fixed_overlap - baseline_overlap
    improvement_adaptive = adaptive_overlap - baseline_overlap

    if improvement_fixed > 0.01:
        print(f"  Temperature calibration HELPS.")
        print(f"  Best fixed temperature: {best_temp_result.temperature:.2f} "
              f"(+{improvement_fixed:.3f} overlap vs baseline)")
    elif improvement_fixed > -0.01:
        print(f"  Temperature calibration has MARGINAL effect.")
        print(f"  Best fixed temperature: {best_temp_result.temperature:.2f} "
              f"({improvement_fixed:+.3f} overlap vs baseline)")
    else:
        print(f"  Temperature calibration does NOT help (or hurts).")
        print(f"  Best fixed temperature: {best_temp_result.temperature:.2f} "
              f"({improvement_fixed:+.3f} overlap vs baseline)")

    if improvement_adaptive > improvement_fixed + 0.01:
        print(f"  Adaptive per-layer temperature is BETTER than fixed "
              f"(+{improvement_adaptive:.3f} vs baseline).")
    elif improvement_adaptive > improvement_fixed - 0.01:
        print(f"  Adaptive per-layer temperature is SIMILAR to fixed "
              f"({improvement_adaptive:+.3f} vs baseline).")
    else:
        print(f"  Adaptive per-layer temperature is WORSE than fixed "
              f"({improvement_adaptive:+.3f} vs baseline).")

    # --- Sample outputs for best config ---
    best_config = best_temp_result
    if adaptive_overlap > best_fixed_overlap:
        best_config = adaptive_result
        best_name = "Adaptive per-layer"
    else:
        best_name = f"Fixed T={best_temp_result.temperature:.2f}"

    print(f"\n--- Sample Outputs (best: {best_name}) ---")
    for i, prompt in enumerate(QUALITY_PROMPTS[:3]):
        print(f"\n  Prompt: {prompt}")
        print(f"  FP16:     {fp16_outputs[i][:150]}...")
        print(f"  Baseline: {tq_baseline.outputs[i][:150]}...")
        print(f"  Best:     {best_config.outputs[i][:150]}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 70)
    print("  TEMPCAL: Temperature Calibration for TurboQuant KV Cache")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Compression: TQ-{BITS} (mse_only for generation)")
    print(f"  Temperature scales: {TEMPERATURE_SCALES}")
    print("=" * 70)
    print()

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM used: {torch.cuda.memory_allocated() // 1024**2} MB")
    print()

    # Step 1: Measure score distributions
    layer_stats, metadata = measure_score_distributions(model, tokenizer)

    # Generate FP16 baselines
    fp16_outputs = generate_fp16_baselines(model, tokenizer)

    # Generate TQ-4 baseline (no temperature adjustment)
    tq_baseline = generate_tq_baseline(model, tokenizer, fp16_outputs)

    # Step 2: Test temperature scales
    temp_results = test_temperature_scales(model, tokenizer, fp16_outputs)

    # Step 3: Adaptive per-layer temperature
    adaptive_result = test_adaptive_temperature(
        model, tokenizer, fp16_outputs, layer_stats,
    )

    # Final report
    print_final_report(
        layer_stats, metadata, tq_baseline,
        temp_results, adaptive_result, fp16_outputs,
    )

    print(f"\n{'=' * 70}")
    print("TEMPCAL experiment complete.")
    print(f"{'=' * 70}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
