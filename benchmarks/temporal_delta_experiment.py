"""Temporal delta coding experiment for within-layer KV cache compression.

Cross-layer prediction failed (deltas 2.6x larger than absolutes).
This experiment tests the OPPOSITE hypothesis: consecutive tokens within
the SAME layer should have highly correlated KV vectors because:
  - Token N sees context [1..N], token N+1 sees [1..N+1]
  - The context differs by exactly one token
  - The KV projections should change smoothly

Experiment plan:
  1. Load Qwen2.5-3B-Instruct (BnB 4-bit)
  2. Extract KV caches for a 500-token prompt (prefill)
  3. Measure temporal correlation: corr(KV[t], KV[t+1]) per layer
  4. Compute delta variance ratio: var(delta) / var(absolute)
  5. If ratio < 0.5 -> delta coding is viable
  6. Test delta quantization at lower bit-widths
  7. Measure attention quality with delta-coded cache

Usage:
    cd /home/dhawal/turboQuantDC
    python benchmarks/temporal_delta_experiment.py
"""

import gc
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"

# Long prompt to get ~500 tokens for meaningful temporal statistics
PROMPT = """The following is a detailed technical analysis of machine learning optimization.

Gradient descent is the foundational optimization algorithm in deep learning. The basic idea
is simple: compute the gradient of the loss function with respect to each parameter, then
update each parameter in the direction that reduces the loss. The learning rate controls
the step size. Too large a learning rate causes divergence; too small causes slow convergence.

Stochastic gradient descent (SGD) approximates the full gradient using mini-batches. This
introduces noise but enables training on large datasets. The noise can actually help escape
local minima. Mini-batch sizes typically range from 32 to 4096, with larger batches providing
more stable gradients but less regularization.

Momentum adds a velocity term that accumulates past gradients. This helps traverse flat regions
and dampens oscillations in ravines. The momentum coefficient, typically 0.9, controls how much
history influences the current update. Nesterov momentum evaluates the gradient at the predicted
next position, providing better convergence.

Adam combines momentum with adaptive learning rates. It maintains running averages of both the
first moment (mean) and second moment (variance) of gradients. The bias correction terms ensure
stable early training. Adam's default hyperparameters (lr=0.001, beta1=0.9, beta2=0.999) work
well across many problems, making it the default optimizer for most practitioners.

Learning rate scheduling is critical for good performance. Common strategies include step decay,
cosine annealing, warmup followed by decay, and cyclical learning rates. The one-cycle policy
uses a warmup phase followed by cosine decay to the minimum learning rate. This has been shown
to enable super-convergence, reaching good accuracy much faster than constant learning rates.

Weight decay (L2 regularization) adds a penalty proportional to the squared magnitude of weights.
In Adam, decoupled weight decay (AdamW) is preferred because it separates the regularization
from the adaptive learning rate mechanism. This is important because L2 regularization in Adam
effectively scales the regularization by the inverse of the second moment estimate, which can
lead to under-regularization of parameters with large gradients.

Batch normalization normalizes activations within each mini-batch, stabilizing training and
allowing higher learning rates. Layer normalization, used in transformers, normalizes across
features instead of the batch dimension. RMSNorm, a simpler variant, normalizes by the root
mean square without centering, and has become popular in modern language models.

The transformer architecture relies on self-attention, which computes queries, keys, and values
from the input. The attention scores are computed as softmax(QK^T / sqrt(d)), where d is the
head dimension. Multi-head attention splits the representation into multiple heads, each
attending to different aspects of the input. The outputs are concatenated and projected.

Key-value caching is essential for efficient autoregressive generation. During the prefill
phase, all tokens are processed in parallel and their key-value pairs are stored. During
generation, only the new token's query is computed, and attention is computed against all
cached keys and values. This avoids redundant computation of previous tokens' representations.

Quantization reduces the precision of model weights and activations to save memory and compute.
Post-training quantization (PTQ) quantizes a pre-trained model without retraining. GPTQ and
AWQ are popular PTQ methods for large language models. Quantization-aware training (QAT)
fine-tunes the model with simulated quantization during forward passes.

The key-value cache grows linearly with sequence length and is often the memory bottleneck
for long-context inference. At 128K context with a 70B model, the KV cache alone can consume
over 40GB of memory. Compression techniques like quantization, eviction, and delta coding
can dramatically reduce this memory footprint while maintaining generation quality."""


# ---------------------------------------------------------------------------
# Model loading and KV extraction
# ---------------------------------------------------------------------------


def load_model():
    """Load Qwen2.5-3B in BnB 4-bit."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME} (BnB 4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True,
        quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def extract_kv_caches(model, tokenizer, prompt: str) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Run prefill and extract per-layer KV caches.

    Returns:
        kv_by_layer: {layer_idx: (keys, values)}
            keys shape:   [1, n_kv_heads, seq_len, head_dim]
            values shape: [1, n_kv_heads, seq_len, head_dim]
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]
    print(f"Prompt tokenized: {seq_len} tokens")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)

    cache = outputs.past_key_values
    kv_by_layer = {}

    if hasattr(cache, "key_cache"):
        for i in range(len(cache.key_cache)):
            k = cache.key_cache[i].float().cpu()
            v = cache.value_cache[i].float().cpu()
            kv_by_layer[i] = (k, v)
    elif hasattr(cache, "layers"):
        for i, layer in enumerate(cache.layers):
            k = layer.keys.float().cpu()
            v = layer.values.float().cpu()
            kv_by_layer[i] = (k, v)
    else:
        for i, entry in enumerate(cache):
            k = entry[0].float().cpu()
            v = entry[1].float().cpu()
            kv_by_layer[i] = (k, v)

    print(f"Extracted KV from {len(kv_by_layer)} layers")
    if 0 in kv_by_layer:
        k0 = kv_by_layer[0][0]
        print(f"  Shape: {list(k0.shape)} = [batch, n_kv_heads, seq, head_dim]")
    return kv_by_layer


# ---------------------------------------------------------------------------
# Experiment 1: Temporal Correlation Measurement
# ---------------------------------------------------------------------------


def measure_temporal_correlation(
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, Any]:
    """Measure correlation between consecutive tokens' KV vectors.

    For each layer and head, compute:
      - Pearson correlation between KV[t] and KV[t+1]
      - Cosine similarity between consecutive vectors
      - Mean absolute change per coordinate
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Temporal Correlation (consecutive tokens, same layer)")
    print("=" * 70)

    results = {
        "per_layer_key_corr": [],
        "per_layer_val_corr": [],
        "per_layer_key_cos": [],
        "per_layer_val_cos": [],
    }

    layers = sorted(kv_by_layer.keys())

    for layer_idx in layers:
        keys, values = kv_by_layer[layer_idx]
        # keys: [1, n_heads, seq, d] -> [n_heads, seq, d]
        keys = keys.squeeze(0)
        values = values.squeeze(0)
        n_heads, seq_len, d = keys.shape

        # Consecutive token pairs
        k_t = keys[:, :-1, :]    # [n_heads, seq-1, d]
        k_tp1 = keys[:, 1:, :]   # [n_heads, seq-1, d]
        v_t = values[:, :-1, :]
        v_tp1 = values[:, 1:, :]

        # Pearson correlation per (head, time pair), then average
        # Flatten to [n_heads * (seq-1), d]
        k_t_flat = k_t.reshape(-1, d)
        k_tp1_flat = k_tp1.reshape(-1, d)
        v_t_flat = v_t.reshape(-1, d)
        v_tp1_flat = v_tp1.reshape(-1, d)

        # Element-wise Pearson across the d dimension
        def batch_pearson(x, y):
            """Pearson correlation for each row of x and y."""
            x_c = x - x.mean(dim=-1, keepdim=True)
            y_c = y - y.mean(dim=-1, keepdim=True)
            num = (x_c * y_c).sum(dim=-1)
            den = x_c.norm(dim=-1) * y_c.norm(dim=-1) + 1e-10
            return (num / den).mean().item()

        # Cosine similarity
        def batch_cosine(x, y):
            return F.cosine_similarity(x, y, dim=-1).mean().item()

        key_corr = batch_pearson(k_t_flat, k_tp1_flat)
        val_corr = batch_pearson(v_t_flat, v_tp1_flat)
        key_cos = batch_cosine(k_t_flat, k_tp1_flat)
        val_cos = batch_cosine(v_t_flat, v_tp1_flat)

        results["per_layer_key_corr"].append(key_corr)
        results["per_layer_val_corr"].append(val_corr)
        results["per_layer_key_cos"].append(key_cos)
        results["per_layer_val_cos"].append(val_cos)

        if layer_idx % 6 == 0 or layer_idx == layers[-1]:
            print(f"  Layer {layer_idx:2d}: key_cos={key_cos:.4f} val_cos={val_cos:.4f} "
                  f"key_corr={key_corr:.4f} val_corr={val_corr:.4f}")

    # Summary
    avg_key_cos = np.mean(results["per_layer_key_cos"])
    avg_val_cos = np.mean(results["per_layer_val_cos"])
    avg_key_corr = np.mean(results["per_layer_key_corr"])
    avg_val_corr = np.mean(results["per_layer_val_corr"])

    print(f"\n  AVERAGE across layers:")
    print(f"    Key cosine sim:  {avg_key_cos:.4f}")
    print(f"    Val cosine sim:  {avg_val_cos:.4f}")
    print(f"    Key Pearson:     {avg_key_corr:.4f}")
    print(f"    Val Pearson:     {avg_val_corr:.4f}")

    results["avg_key_cos"] = avg_key_cos
    results["avg_val_cos"] = avg_val_cos
    results["avg_key_corr"] = avg_key_corr
    results["avg_val_corr"] = avg_val_corr

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Delta Variance Ratio
# ---------------------------------------------------------------------------


def measure_delta_variance(
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, Any]:
    """The critical test: is var(KV[t+1] - KV[t]) < 0.5 * var(KV[t])?

    If yes, delta coding wins -- deltas need fewer bits to quantize.
    If no (like cross-layer where ratio was 2.6), delta coding is worse.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Delta Variance Ratio (threshold: < 0.5 for viable)")
    print("=" * 70)

    layers = sorted(kv_by_layer.keys())
    results = {
        "per_layer_key_var_ratio": [],
        "per_layer_val_var_ratio": [],
        "per_layer_key_l2_ratio": [],
        "per_layer_val_l2_ratio": [],
        "per_layer_key_linf_ratio": [],
        "per_layer_val_linf_ratio": [],
    }

    for layer_idx in layers:
        keys, values = kv_by_layer[layer_idx]
        keys = keys.squeeze(0)    # [n_heads, seq, d]
        values = values.squeeze(0)

        # Deltas between consecutive tokens
        k_delta = keys[:, 1:, :] - keys[:, :-1, :]   # [n_heads, seq-1, d]
        v_delta = values[:, 1:, :] - values[:, :-1, :]

        # Absolute values (for comparison)
        k_abs = keys[:, 1:, :]
        v_abs = values[:, 1:, :]

        # Variance ratio: var(delta) / var(absolute)
        k_var_ratio = k_delta.var().item() / (k_abs.var().item() + 1e-10)
        v_var_ratio = v_delta.var().item() / (v_abs.var().item() + 1e-10)

        # L2 norm ratio: mean ||delta|| / mean ||abs||
        k_l2_ratio = k_delta.norm(dim=-1).mean().item() / (k_abs.norm(dim=-1).mean().item() + 1e-10)
        v_l2_ratio = v_delta.norm(dim=-1).mean().item() / (v_abs.norm(dim=-1).mean().item() + 1e-10)

        # L-inf ratio: mean max|delta| / mean max|abs|
        k_linf_ratio = k_delta.abs().max(dim=-1).values.mean().item() / (k_abs.abs().max(dim=-1).values.mean().item() + 1e-10)
        v_linf_ratio = v_delta.abs().max(dim=-1).values.mean().item() / (v_abs.abs().max(dim=-1).values.mean().item() + 1e-10)

        results["per_layer_key_var_ratio"].append(k_var_ratio)
        results["per_layer_val_var_ratio"].append(v_var_ratio)
        results["per_layer_key_l2_ratio"].append(k_l2_ratio)
        results["per_layer_val_l2_ratio"].append(v_l2_ratio)
        results["per_layer_key_linf_ratio"].append(k_linf_ratio)
        results["per_layer_val_linf_ratio"].append(v_linf_ratio)

        if layer_idx % 6 == 0 or layer_idx == layers[-1]:
            print(f"  Layer {layer_idx:2d}: key_var_ratio={k_var_ratio:.4f} val_var_ratio={v_var_ratio:.4f} "
                  f"key_l2_ratio={k_l2_ratio:.4f} val_l2_ratio={v_l2_ratio:.4f}")

    # Summary
    avg_key_var = np.mean(results["per_layer_key_var_ratio"])
    avg_val_var = np.mean(results["per_layer_val_var_ratio"])
    avg_key_l2 = np.mean(results["per_layer_key_l2_ratio"])
    avg_val_l2 = np.mean(results["per_layer_val_l2_ratio"])
    avg_key_linf = np.mean(results["per_layer_key_linf_ratio"])
    avg_val_linf = np.mean(results["per_layer_val_linf_ratio"])

    print(f"\n  AVERAGE across layers:")
    print(f"    Key var ratio:   {avg_key_var:.4f} ({'VIABLE' if avg_key_var < 0.5 else 'NOT VIABLE'} need < 0.5)")
    print(f"    Val var ratio:   {avg_val_var:.4f} ({'VIABLE' if avg_val_var < 0.5 else 'NOT VIABLE'} need < 0.5)")
    print(f"    Key L2 ratio:    {avg_key_l2:.4f}")
    print(f"    Val L2 ratio:    {avg_val_l2:.4f}")
    print(f"    Key Linf ratio:  {avg_key_linf:.4f}")
    print(f"    Val Linf ratio:  {avg_val_linf:.4f}")

    viable_key = avg_key_var < 0.5
    viable_val = avg_val_var < 0.5

    results["avg_key_var_ratio"] = avg_key_var
    results["avg_val_var_ratio"] = avg_val_var
    results["avg_key_l2_ratio"] = avg_key_l2
    results["avg_val_l2_ratio"] = avg_val_l2
    results["avg_key_linf_ratio"] = avg_key_linf
    results["avg_val_linf_ratio"] = avg_val_linf
    results["key_viable"] = viable_key
    results["val_viable"] = viable_val

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Per-Position Delta Analysis
# ---------------------------------------------------------------------------


def analyze_delta_by_position(
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, Any]:
    """Check if delta magnitude varies by position.

    Hypothesis: early positions have larger deltas (context changing rapidly),
    later positions have smaller deltas (one token in 400+ is marginal).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Delta Magnitude by Token Position")
    print("=" * 70)

    layers = sorted(kv_by_layer.keys())

    # Accumulate delta norms across layers (average over all layers)
    keys0, _ = kv_by_layer[0]
    seq_len = keys0.shape[2]
    n_positions = seq_len - 1

    key_delta_norms = torch.zeros(n_positions)
    val_delta_norms = torch.zeros(n_positions)
    key_abs_norms = torch.zeros(n_positions)
    val_abs_norms = torch.zeros(n_positions)

    for layer_idx in layers:
        keys, values = kv_by_layer[layer_idx]
        keys = keys.squeeze(0)    # [n_heads, seq, d]
        values = values.squeeze(0)

        k_delta = keys[:, 1:, :] - keys[:, :-1, :]
        v_delta = values[:, 1:, :] - values[:, :-1, :]

        # Average over heads
        key_delta_norms += k_delta.norm(dim=-1).mean(dim=0).cpu()
        val_delta_norms += v_delta.norm(dim=-1).mean(dim=0).cpu()
        key_abs_norms += keys[:, 1:, :].norm(dim=-1).mean(dim=0).cpu()
        val_abs_norms += values[:, 1:, :].norm(dim=-1).mean(dim=0).cpu()

    n_layers = len(layers)
    key_delta_norms /= n_layers
    val_delta_norms /= n_layers
    key_abs_norms /= n_layers
    val_abs_norms /= n_layers

    key_ratio_by_pos = (key_delta_norms / (key_abs_norms + 1e-10)).numpy()
    val_ratio_by_pos = (val_delta_norms / (val_abs_norms + 1e-10)).numpy()

    # Report by quartile
    q1 = n_positions // 4
    q2 = n_positions // 2
    q3 = 3 * n_positions // 4

    print(f"  Token positions: {n_positions}")
    print(f"  Key delta/abs ratio by quartile:")
    print(f"    Q1 (pos 0-{q1}):     {key_ratio_by_pos[:q1].mean():.4f}")
    print(f"    Q2 (pos {q1}-{q2}):   {key_ratio_by_pos[q1:q2].mean():.4f}")
    print(f"    Q3 (pos {q2}-{q3}):   {key_ratio_by_pos[q2:q3].mean():.4f}")
    print(f"    Q4 (pos {q3}-{n_positions}): {key_ratio_by_pos[q3:].mean():.4f}")

    print(f"  Value delta/abs ratio by quartile:")
    print(f"    Q1 (pos 0-{q1}):     {val_ratio_by_pos[:q1].mean():.4f}")
    print(f"    Q2 (pos {q1}-{q2}):   {val_ratio_by_pos[q1:q2].mean():.4f}")
    print(f"    Q3 (pos {q2}-{q3}):   {val_ratio_by_pos[q2:q3].mean():.4f}")
    print(f"    Q4 (pos {q3}-{n_positions}): {val_ratio_by_pos[q3:].mean():.4f}")

    # First 10 vs last 10
    print(f"\n  First 10 positions: key ratio={key_ratio_by_pos[:10].mean():.4f}, "
          f"val ratio={val_ratio_by_pos[:10].mean():.4f}")
    print(f"  Last 10 positions:  key ratio={key_ratio_by_pos[-10:].mean():.4f}, "
          f"val ratio={val_ratio_by_pos[-10:].mean():.4f}")

    return {
        "key_ratio_by_pos": key_ratio_by_pos.tolist(),
        "val_ratio_by_pos": val_ratio_by_pos.tolist(),
        "key_q1": float(key_ratio_by_pos[:q1].mean()),
        "key_q2": float(key_ratio_by_pos[q1:q2].mean()),
        "key_q3": float(key_ratio_by_pos[q2:q3].mean()),
        "key_q4": float(key_ratio_by_pos[q3:].mean()),
        "val_q1": float(val_ratio_by_pos[:q1].mean()),
        "val_q2": float(val_ratio_by_pos[q1:q2].mean()),
        "val_q3": float(val_ratio_by_pos[q2:q3].mean()),
        "val_q4": float(val_ratio_by_pos[q3:].mean()),
    }


# ---------------------------------------------------------------------------
# Experiment 4: Delta Sparsity
# ---------------------------------------------------------------------------


def measure_delta_sparsity(
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    thresholds: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.20),
) -> Dict[str, Any]:
    """Measure what fraction of delta coordinates are near-zero.

    If deltas are sparse (most coordinates barely change), we can use
    sparse encoding for additional compression.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Delta Sparsity Analysis")
    print("=" * 70)

    layers = sorted(kv_by_layer.keys())
    results = {f"key_sparse_{t}": [] for t in thresholds}
    results.update({f"val_sparse_{t}": [] for t in thresholds})

    for layer_idx in layers:
        keys, values = kv_by_layer[layer_idx]
        keys = keys.squeeze(0)
        values = values.squeeze(0)

        k_delta = keys[:, 1:, :] - keys[:, :-1, :]
        v_delta = values[:, 1:, :] - values[:, :-1, :]

        # Normalize deltas by the std of absolute values for scale-independent threshold
        k_std = keys[:, 1:, :].std().item()
        v_std = values[:, 1:, :].std().item()

        for t in thresholds:
            k_sparse = (k_delta.abs() < t * k_std).float().mean().item()
            v_sparse = (v_delta.abs() < t * v_std).float().mean().item()
            results[f"key_sparse_{t}"].append(k_sparse)
            results[f"val_sparse_{t}"].append(v_sparse)

    # Summary
    print(f"  Fraction of delta coordinates below threshold (relative to abs std):")
    print(f"  {'Threshold':<12} {'Key sparsity':<15} {'Val sparsity':<15}")
    print(f"  {'-' * 42}")
    for t in thresholds:
        k_mean = np.mean(results[f"key_sparse_{t}"])
        v_mean = np.mean(results[f"val_sparse_{t}"])
        print(f"  {t:<12.2f} {k_mean:<15.4f} {v_mean:<15.4f}")
        results[f"avg_key_sparse_{t}"] = k_mean
        results[f"avg_val_sparse_{t}"] = v_mean

    return results


# ---------------------------------------------------------------------------
# Experiment 5: Delta Entropy (bits needed)
# ---------------------------------------------------------------------------


def measure_delta_entropy(
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, Any]:
    """Estimate bits needed to encode deltas vs absolute values.

    Quantize both to uniform grids and measure Shannon entropy.
    If delta entropy < absolute entropy, delta coding saves bits.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Delta vs Absolute Entropy (bits needed)")
    print("=" * 70)

    layers = sorted(kv_by_layer.keys())
    bit_widths = [2, 3, 4]
    results = {}

    for bits in bit_widths:
        key_abs_entropies = []
        key_delta_entropies = []
        val_abs_entropies = []
        val_delta_entropies = []

        for layer_idx in layers:
            keys, values = kv_by_layer[layer_idx]
            keys = keys.squeeze(0)
            values = values.squeeze(0)

            k_abs = keys[:, 1:, :].reshape(-1)
            v_abs = values[:, 1:, :].reshape(-1)
            k_delta = (keys[:, 1:, :] - keys[:, :-1, :]).reshape(-1)
            v_delta = (values[:, 1:, :] - values[:, :-1, :]).reshape(-1)

            def entropy_of(tensor, nbits):
                """Quantize to nbits and compute Shannon entropy."""
                max_val = tensor.abs().max().item()
                if max_val == 0:
                    return 0.0
                qmax = 2 ** (nbits - 1) - 1
                scale = max_val / qmax
                indices = torch.round(tensor / (scale + 1e-10)).clamp(-qmax - 1, qmax).long()
                unique, counts = indices.unique(return_counts=True)
                probs = counts.float() / counts.sum().float()
                return -(probs * probs.log2()).sum().item()

            key_abs_entropies.append(entropy_of(k_abs, bits))
            key_delta_entropies.append(entropy_of(k_delta, bits))
            val_abs_entropies.append(entropy_of(v_abs, bits))
            val_delta_entropies.append(entropy_of(v_delta, bits))

        k_abs_h = np.mean(key_abs_entropies)
        k_delta_h = np.mean(key_delta_entropies)
        v_abs_h = np.mean(val_abs_entropies)
        v_delta_h = np.mean(val_delta_entropies)

        print(f"\n  {bits}-bit quantization grid:")
        print(f"    Keys:   abs entropy={k_abs_h:.3f} bits, delta entropy={k_delta_h:.3f} bits, "
              f"ratio={k_delta_h / (k_abs_h + 1e-10):.3f}")
        print(f"    Values: abs entropy={v_abs_h:.3f} bits, delta entropy={v_delta_h:.3f} bits, "
              f"ratio={v_delta_h / (v_abs_h + 1e-10):.3f}")

        results[f"{bits}bit_key_abs_entropy"] = k_abs_h
        results[f"{bits}bit_key_delta_entropy"] = k_delta_h
        results[f"{bits}bit_val_abs_entropy"] = v_abs_h
        results[f"{bits}bit_val_delta_entropy"] = v_delta_h

    return results


# ---------------------------------------------------------------------------
# Experiment 6: Attention Score Quality with Delta Coding
# ---------------------------------------------------------------------------


def test_attention_quality(
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    sample_layers: Tuple[int, ...] = (0, 6, 12, 18, 24, 30, 35),
) -> Dict[str, Any]:
    """Test whether delta-coded KV cache preserves attention scores.

    Approach:
      1. Use last token as query
      2. Compute FP16 attention scores (ground truth)
      3. Reconstruct keys from delta coding at various bit-widths
      4. Compare attention scores
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Attention Quality with Delta-Coded Keys")
    print("=" * 70)

    results = {}
    bit_configs = [
        ("delta-2bit", 2),
        ("delta-3bit", 3),
        ("delta-4bit", 4),
    ]

    for layer_idx in sample_layers:
        if layer_idx not in kv_by_layer:
            continue

        keys, values = kv_by_layer[layer_idx]
        keys = keys.squeeze(0)  # [n_heads, seq, d]
        n_heads, seq_len, d = keys.shape

        # Use last token as query
        query = keys[:, -1:, :]  # [n_heads, 1, d]

        # Ground truth attention
        gt_scores = torch.matmul(query, keys.transpose(-1, -2)).squeeze(1)  # [n_heads, seq]
        gt_scores = gt_scores / math.sqrt(d)

        for config_name, bits in bit_configs:
            # Encode: store first token at full precision, rest as quantized deltas
            reconstructed = torch.zeros_like(keys)
            reconstructed[:, 0, :] = keys[:, 0, :]  # Anchor: full precision

            deltas = keys[:, 1:, :] - keys[:, :-1, :]  # [n_heads, seq-1, d]

            # Uniform quantization of deltas
            qmax = 2 ** (bits - 1) - 1
            # Per-head, per-position scale for best quality
            delta_max = deltas.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
            scale = delta_max / qmax
            q_deltas = torch.round(deltas / scale).clamp(-qmax - 1, qmax)
            dq_deltas = q_deltas * scale

            # Reconstruct via cumulative sum of dequantized deltas
            reconstructed[:, 1:, :] = keys[:, 0:1, :] + torch.cumsum(dq_deltas, dim=1)

            # Compute attention with reconstructed keys
            recon_scores = torch.matmul(query, reconstructed.transpose(-1, -2)).squeeze(1)
            recon_scores = recon_scores / math.sqrt(d)

            # Quality metrics
            cos_sim = F.cosine_similarity(gt_scores, recon_scores, dim=-1).mean().item()

            # Top-5 overlap
            k = min(5, seq_len)
            gt_topk = gt_scores.topk(k, dim=-1).indices
            recon_topk = recon_scores.topk(k, dim=-1).indices
            overlap = 0
            for h in range(n_heads):
                gt_set = set(gt_topk[h].tolist())
                recon_set = set(recon_topk[h].tolist())
                overlap += len(gt_set & recon_set) / k
            top5_match = overlap / n_heads

            # Score MSE (normalized)
            score_mse = ((gt_scores - recon_scores) ** 2).mean().item()
            score_var = (gt_scores ** 2).mean().item()
            relative_mse = score_mse / (score_var + 1e-10)

            key = f"layer{layer_idx}_{config_name}"
            results[key] = {
                "cos_sim": cos_sim,
                "top5_match": top5_match,
                "relative_mse": relative_mse,
            }

            if layer_idx % 12 == 0 or layer_idx == sample_layers[-1]:
                print(f"  Layer {layer_idx:2d} {config_name}: "
                      f"cos={cos_sim:.6f} top5={top5_match:.2%} rel_mse={relative_mse:.6f}")

    # Average across layers
    for config_name, bits in bit_configs:
        cos_vals = [v["cos_sim"] for k, v in results.items() if config_name in k]
        top5_vals = [v["top5_match"] for k, v in results.items() if config_name in k]
        mse_vals = [v["relative_mse"] for k, v in results.items() if config_name in k]
        if cos_vals:
            print(f"\n  {config_name} AVERAGE: cos={np.mean(cos_vals):.6f} "
                  f"top5={np.mean(top5_vals):.2%} rel_mse={np.mean(mse_vals):.6f}")
            results[f"avg_{config_name}"] = {
                "cos_sim": float(np.mean(cos_vals)),
                "top5_match": float(np.mean(top5_vals)),
                "relative_mse": float(np.mean(mse_vals)),
            }

    return results


# ---------------------------------------------------------------------------
# Experiment 7: Compression Ratio Analysis
# ---------------------------------------------------------------------------


def analyze_compression(
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, Any]:
    """Calculate effective compression ratios for delta coding.

    Storage model for delta coding:
      - Token 0: d * 16 bits (FP16 anchor)
      - Token 1..N: d * delta_bits + 16 bits (scale per vector)
      - Total: d*16 + (N-1)*(d*delta_bits + 16) bits per head

    Compare against:
      - FP16 baseline: N * d * 16 bits
      - TurboQuant 3-bit: N * d * 3 bits (approximate)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Compression Ratio Analysis")
    print("=" * 70)

    keys0, _ = kv_by_layer[0]
    seq_len = keys0.shape[2]
    d = keys0.shape[3]

    fp16_bits = seq_len * d * 16
    tq3_bits = seq_len * d * 3

    print(f"  Seq length: {seq_len}, head dim: {d}")
    print(f"  FP16 baseline: {fp16_bits} bits ({fp16_bits / 8 / 1024:.1f} KB per head)")
    print(f"  TQ 3-bit:      {tq3_bits} bits ({tq3_bits / 8 / 1024:.1f} KB per head)")

    results = {}
    for delta_bits in [1, 2, 3]:
        # Anchor (first token) at FP16, rest as delta
        anchor_bits = d * 16
        delta_total = (seq_len - 1) * (d * delta_bits + 16)  # +16 for scale
        total_bits = anchor_bits + delta_total

        ratio_vs_fp16 = fp16_bits / total_bits
        ratio_vs_tq3 = tq3_bits / total_bits

        print(f"\n  Delta {delta_bits}-bit:")
        print(f"    Total: {total_bits} bits ({total_bits / 8 / 1024:.1f} KB per head)")
        print(f"    vs FP16: {ratio_vs_fp16:.2f}x compression")
        print(f"    vs TQ3:  {ratio_vs_tq3:.2f}x compression")
        print(f"    Effective bits/coord: {total_bits / (seq_len * d):.2f}")

        results[f"delta_{delta_bits}bit"] = {
            "total_bits": total_bits,
            "ratio_vs_fp16": ratio_vs_fp16,
            "ratio_vs_tq3": ratio_vs_tq3,
            "effective_bpc": total_bits / (seq_len * d),
        }

    return results


# ---------------------------------------------------------------------------
# Experiment 8: Error Accumulation Analysis
# ---------------------------------------------------------------------------


def measure_error_accumulation(
    kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    sample_layers: Tuple[int, ...] = (0, 12, 24, 35),
) -> Dict[str, Any]:
    """Measure how reconstruction error grows with position.

    Delta coding reconstructs via cumulative sum: x[t] = x[0] + sum(delta[1..t]).
    Quantization errors in early deltas propagate to all later positions.
    This is the main risk of delta coding.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Error Accumulation Over Sequence Length")
    print("=" * 70)

    results = {}

    for layer_idx in sample_layers:
        if layer_idx not in kv_by_layer:
            continue

        keys, values = kv_by_layer[layer_idx]
        keys = keys.squeeze(0)  # [n_heads, seq, d]
        n_heads, seq_len, d = keys.shape

        for bits in [2, 3, 4]:
            # Delta quantization
            deltas = keys[:, 1:, :] - keys[:, :-1, :]
            qmax = 2 ** (bits - 1) - 1
            delta_max = deltas.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
            scale = delta_max / qmax
            q_deltas = torch.round(deltas / scale).clamp(-qmax - 1, qmax)
            dq_deltas = q_deltas * scale

            # Reconstruct
            recon = torch.zeros_like(keys)
            recon[:, 0, :] = keys[:, 0, :]
            recon[:, 1:, :] = keys[:, 0:1, :] + torch.cumsum(dq_deltas, dim=1)

            # Per-position error
            errors = (keys - recon).norm(dim=-1)    # [n_heads, seq]
            norms = keys.norm(dim=-1)                # [n_heads, seq]
            rel_errors = errors / (norms + 1e-10)    # [n_heads, seq]

            avg_rel_error = rel_errors.mean(dim=0).numpy()  # [seq]

            # Report at quartiles
            q1, q2, q3, q4 = seq_len // 4, seq_len // 2, 3 * seq_len // 4, seq_len - 1
            e_q1 = avg_rel_error[:q1].mean()
            e_q2 = avg_rel_error[q1:q2].mean()
            e_q3 = avg_rel_error[q2:q3].mean()
            e_q4 = avg_rel_error[q3:].mean()

            if layer_idx % 12 == 0 or layer_idx == sample_layers[-1]:
                print(f"  Layer {layer_idx:2d}, {bits}-bit delta: "
                      f"Q1={e_q1:.4f} Q2={e_q2:.4f} Q3={e_q3:.4f} Q4={e_q4:.4f}")

            results[f"layer{layer_idx}_{bits}bit"] = {
                "q1_error": float(e_q1),
                "q2_error": float(e_q2),
                "q3_error": float(e_q3),
                "q4_error": float(e_q4),
                "max_error": float(avg_rel_error.max()),
                "mean_error": float(avg_rel_error.mean()),
            }

    return results


# ---------------------------------------------------------------------------
# Results writing
# ---------------------------------------------------------------------------


def write_results(
    all_results: Dict[str, Any],
    seq_len: int,
    n_layers: int,
    runtime: float,
) -> str:
    """Write results to markdown file."""
    output_path = os.path.join(REPO_ROOT, "benchmarks", "results", "temporal_delta_results.md")

    r_corr = all_results["temporal_correlation"]
    r_var = all_results["delta_variance"]
    r_pos = all_results["position_analysis"]
    r_sparse = all_results["sparsity"]
    r_entropy = all_results["entropy"]
    r_attention = all_results["attention_quality"]
    r_compress = all_results["compression"]
    r_error = all_results["error_accumulation"]

    lines = []
    lines.append("# Temporal Delta Coding Experiment Results")
    lines.append("")
    lines.append(f"**Model:** {MODEL_NAME}")
    lines.append(f"**Prompt tokens:** {seq_len}")
    lines.append(f"**Layers:** {n_layers}")
    lines.append(f"**Runtime:** {runtime:.1f}s")
    lines.append("")

    lines.append("## Hypothesis")
    lines.append("")
    lines.append("Consecutive tokens within the same layer share nearly identical context")
    lines.append("(token N sees [1..N], token N+1 sees [1..N+1]). The KV projections should")
    lines.append("be highly correlated temporally, making delta coding viable for within-layer")
    lines.append("compression. Cross-layer delta coding failed (ratio 2.6x), but temporal")
    lines.append("deltas may succeed because the mechanism is fundamentally different.")
    lines.append("")

    # Experiment 1: Temporal correlation
    lines.append("## Experiment 1: Temporal Correlation")
    lines.append("")
    lines.append("| Metric | Keys | Values |")
    lines.append("|--------|------|--------|")
    lines.append(f"| Cosine similarity | {r_corr['avg_key_cos']:.4f} | {r_corr['avg_val_cos']:.4f} |")
    lines.append(f"| Pearson correlation | {r_corr['avg_key_corr']:.4f} | {r_corr['avg_val_corr']:.4f} |")
    lines.append("")

    # Per-layer detail
    lines.append("### Per-Layer Cosine Similarity")
    lines.append("")
    lines.append("| Layer | Key cos | Val cos |")
    lines.append("|-------|---------|---------|")
    for i, (kc, vc) in enumerate(zip(r_corr["per_layer_key_cos"], r_corr["per_layer_val_cos"])):
        if i % 6 == 0 or i == n_layers - 1:
            lines.append(f"| {i} | {kc:.4f} | {vc:.4f} |")
    lines.append("")

    # Experiment 2: Delta variance
    lines.append("## Experiment 2: Delta Variance Ratio")
    lines.append("")
    lines.append("**Threshold:** ratio < 0.5 means delta coding is viable")
    lines.append("")
    lines.append("| Metric | Keys | Values | Viable? |")
    lines.append("|--------|------|--------|---------|")
    kv = r_var['avg_key_var_ratio']
    vv = r_var['avg_val_var_ratio']
    lines.append(f"| Variance ratio | {kv:.4f} | {vv:.4f} | {'YES' if kv < 0.5 and vv < 0.5 else 'NO'} |")
    lines.append(f"| L2 norm ratio | {r_var['avg_key_l2_ratio']:.4f} | {r_var['avg_val_l2_ratio']:.4f} | |")
    lines.append(f"| Linf ratio | {r_var['avg_key_linf_ratio']:.4f} | {r_var['avg_val_linf_ratio']:.4f} | |")
    lines.append("")
    lines.append(f"- Key delta viable: **{'YES' if r_var['key_viable'] else 'NO'}** (ratio={kv:.4f})")
    lines.append(f"- Value delta viable: **{'YES' if r_var['val_viable'] else 'NO'}** (ratio={vv:.4f})")
    lines.append("")

    # Per-layer variance ratios
    lines.append("### Per-Layer Variance Ratios")
    lines.append("")
    lines.append("| Layer | Key var ratio | Val var ratio |")
    lines.append("|-------|---------------|---------------|")
    for i, (kr, vr) in enumerate(zip(r_var["per_layer_key_var_ratio"], r_var["per_layer_val_var_ratio"])):
        if i % 6 == 0 or i == n_layers - 1:
            lines.append(f"| {i} | {kr:.4f} | {vr:.4f} |")
    lines.append("")

    # Experiment 3: Position analysis
    lines.append("## Experiment 3: Delta by Token Position")
    lines.append("")
    lines.append("| Quartile | Key delta/abs | Val delta/abs |")
    lines.append("|----------|---------------|---------------|")
    lines.append(f"| Q1 (early) | {r_pos['key_q1']:.4f} | {r_pos['val_q1']:.4f} |")
    lines.append(f"| Q2 | {r_pos['key_q2']:.4f} | {r_pos['val_q2']:.4f} |")
    lines.append(f"| Q3 | {r_pos['key_q3']:.4f} | {r_pos['val_q3']:.4f} |")
    lines.append(f"| Q4 (late) | {r_pos['key_q4']:.4f} | {r_pos['val_q4']:.4f} |")
    lines.append("")

    # Experiment 4: Sparsity
    lines.append("## Experiment 4: Delta Sparsity")
    lines.append("")
    lines.append("| Threshold | Key sparsity | Val sparsity |")
    lines.append("|-----------|-------------|-------------|")
    for t in [0.01, 0.05, 0.10, 0.20]:
        ks = r_sparse.get(f"avg_key_sparse_{t}", 0)
        vs = r_sparse.get(f"avg_val_sparse_{t}", 0)
        lines.append(f"| {t:.2f} | {ks:.4f} | {vs:.4f} |")
    lines.append("")

    # Experiment 5: Entropy
    lines.append("## Experiment 5: Entropy Analysis")
    lines.append("")
    lines.append("| Bit-width | Key abs H | Key delta H | Ratio | Val abs H | Val delta H | Ratio |")
    lines.append("|-----------|-----------|-------------|-------|-----------|-------------|-------|")
    for bits in [2, 3, 4]:
        kah = r_entropy.get(f"{bits}bit_key_abs_entropy", 0)
        kdh = r_entropy.get(f"{bits}bit_key_delta_entropy", 0)
        vah = r_entropy.get(f"{bits}bit_val_abs_entropy", 0)
        vdh = r_entropy.get(f"{bits}bit_val_delta_entropy", 0)
        kr = kdh / (kah + 1e-10)
        vr = vdh / (vah + 1e-10)
        lines.append(f"| {bits} | {kah:.3f} | {kdh:.3f} | {kr:.3f} | {vah:.3f} | {vdh:.3f} | {vr:.3f} |")
    lines.append("")

    # Experiment 6: Attention quality
    lines.append("## Experiment 6: Attention Quality")
    lines.append("")
    lines.append("| Config | Cosine Sim | Top-5 Match | Relative MSE |")
    lines.append("|--------|-----------|-------------|--------------|")
    for config in ["delta-2bit", "delta-3bit", "delta-4bit"]:
        avg = r_attention.get(f"avg_{config}", {})
        if avg:
            lines.append(f"| {config} | {avg['cos_sim']:.6f} | {avg['top5_match']:.2%} | {avg['relative_mse']:.6f} |")
    lines.append("")

    # Experiment 7: Compression
    lines.append("## Experiment 7: Compression Ratios")
    lines.append("")
    lines.append("| Config | vs FP16 | vs TQ3 | Eff bits/coord |")
    lines.append("|--------|---------|--------|----------------|")
    for bits in [1, 2, 3]:
        c = r_compress.get(f"delta_{bits}bit", {})
        if c:
            lines.append(f"| delta-{bits}bit | {c['ratio_vs_fp16']:.2f}x | {c['ratio_vs_tq3']:.2f}x | {c['effective_bpc']:.2f} |")
    lines.append("")

    # Experiment 8: Error accumulation
    lines.append("## Experiment 8: Error Accumulation")
    lines.append("")
    lines.append("| Layer | Bits | Q1 err | Q2 err | Q3 err | Q4 err | Max err |")
    lines.append("|-------|------|--------|--------|--------|--------|---------|")
    for key, val in sorted(r_error.items()):
        parts = key.split("_")
        layer = parts[0].replace("layer", "")
        bits = parts[1]
        lines.append(f"| {layer} | {bits} | {val['q1_error']:.4f} | {val['q2_error']:.4f} | "
                      f"{val['q3_error']:.4f} | {val['q4_error']:.4f} | {val['max_error']:.4f} |")
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")

    key_viable = r_var["key_viable"]
    val_viable = r_var["val_viable"]
    kv_ratio = r_var["avg_key_var_ratio"]
    vv_ratio = r_var["avg_val_var_ratio"]

    if key_viable and val_viable:
        verdict = "VIABLE"
        lines.append(f"**TEMPORAL DELTA CODING IS {verdict}**")
        lines.append("")
        lines.append(f"Both key and value delta variance ratios are below 0.5:")
        lines.append(f"- Keys: {kv_ratio:.4f}")
        lines.append(f"- Values: {vv_ratio:.4f}")
    elif key_viable or val_viable:
        verdict = "PARTIALLY VIABLE"
        lines.append(f"**TEMPORAL DELTA CODING IS {verdict}**")
        lines.append("")
        lines.append(f"- Keys: {kv_ratio:.4f} ({'VIABLE' if key_viable else 'NOT VIABLE'})")
        lines.append(f"- Values: {vv_ratio:.4f} ({'VIABLE' if val_viable else 'NOT VIABLE'})")
    else:
        verdict = "NOT VIABLE"
        lines.append(f"**TEMPORAL DELTA CODING IS {verdict}**")
        lines.append("")
        lines.append(f"Delta variance ratios are too high:")
        lines.append(f"- Keys: {kv_ratio:.4f} (need < 0.5)")
        lines.append(f"- Values: {vv_ratio:.4f} (need < 0.5)")
    lines.append("")

    # Analysis
    lines.append("## Analysis")
    lines.append("")
    if not (key_viable and val_viable):
        lines.append("### Why Temporal Delta Coding Fails (or Partially Fails)")
        lines.append("")
        lines.append("Despite the intuition that consecutive tokens share similar context,")
        lines.append("the KV projections change substantially between positions because:")
        lines.append("")
        lines.append("1. **Rotary Position Embeddings (RoPE):** Each position gets a unique")
        lines.append("   rotation applied to its key vector. This rotation changes the vector")
        lines.append("   direction at every position, even if the content representation is similar.")
        lines.append("   RoPE is specifically designed to make keys position-dependent.")
        lines.append("")
        lines.append("2. **Self-attention mechanism:** The key for position t is computed as")
        lines.append("   K[t] = W_K @ hidden[t], where hidden[t] incorporates attention-weighted")
        lines.append("   information from all previous positions. Adding one new token can")
        lines.append("   shift the residual stream representation non-trivially.")
        lines.append("")
        lines.append("3. **Non-linear accumulation:** Even if each token adds a small perturbation,")
        lines.append("   layer normalization and non-linear activations can amplify or reshape")
        lines.append("   these perturbations, making consecutive hidden states less similar than expected.")
        lines.append("")
    else:
        lines.append("### Why Temporal Delta Coding Works")
        lines.append("")
        lines.append("Consecutive tokens' KV vectors are highly correlated because:")
        lines.append("")
        lines.append("1. The context changes by exactly one token between positions")
        lines.append("2. The KV projection is a linear function of the hidden state")
        lines.append("3. The hidden state changes gradually as context grows")
        lines.append("")

    lines.append("### Comparison with Cross-Layer Delta Coding")
    lines.append("")
    lines.append("| Approach | Key var ratio | Val var ratio | Verdict |")
    lines.append("|----------|---------------|---------------|---------|")
    lines.append(f"| Cross-layer (layers) | 2.58 | 2.09 | NOT VIABLE |")
    lines.append(f"| Temporal (positions) | {kv_ratio:.4f} | {vv_ratio:.4f} | {verdict} |")
    lines.append("")

    lines.append(f"---")
    lines.append(f"*Generated on {time.strftime('%Y-%m-%d %H:%M:%S')} by temporal_delta_experiment.py*")

    content = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(content)
    print(f"\nResults written to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    start_time = time.time()

    # Load model and extract KV caches
    model, tokenizer = load_model()
    kv_by_layer = extract_kv_caches(model, tokenizer, PROMPT)

    seq_len = kv_by_layer[0][0].shape[2]
    n_layers = len(kv_by_layer)

    # Free model memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Run all experiments
    all_results = {}
    all_results["temporal_correlation"] = measure_temporal_correlation(kv_by_layer)
    all_results["delta_variance"] = measure_delta_variance(kv_by_layer)
    all_results["position_analysis"] = analyze_delta_by_position(kv_by_layer)
    all_results["sparsity"] = measure_delta_sparsity(kv_by_layer)
    all_results["entropy"] = measure_delta_entropy(kv_by_layer)
    all_results["attention_quality"] = test_attention_quality(kv_by_layer)
    all_results["compression"] = analyze_compression(kv_by_layer)
    all_results["error_accumulation"] = measure_error_accumulation(kv_by_layer)

    runtime = time.time() - start_time

    # Write results
    output_path = write_results(all_results, seq_len, n_layers, runtime)

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE in {runtime:.1f}s")
    print(f"Results: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
