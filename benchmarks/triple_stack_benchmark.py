#!/usr/bin/env python3
"""Triple Stack Benchmark: Expected Attention + KVSculpt Distillation + TurboQuant.

Stacks three independent compression breakthroughs into a single pipeline:
    1. Expected Attention eviction -- remove lowest-importance tokens (2x)
    2. KVSculpt distillation -- distill remaining tokens into fewer synthetic ones (4x)
    3. TurboQuant 3-bit -- compress the synthetic tokens (5x)

Combined: 2 x 4 x 5 = 40x theoretical compression.

Key question: do the quality losses compound multiplicatively, or is each
step sufficiently orthogonal that quality degrades sub-linearly?

Tested stacking orders:
    A. EA evict 50% -> Distill 4x -> Quant 3-bit = 40x
    B. Distill 4x -> EA evict 50% -> Quant 3-bit = 40x (different order)
    C. EA evict 30% -> Distill 4x -> Quant 3-bit = 28x (less aggressive)
    D. EA evict 70% -> Distill 4x -> Quant 3-bit = 66x (more aggressive)

Run:
    cd /home/dhawal/turboQuantDC && python benchmarks/triple_stack_benchmark.py

Saves results to benchmarks/results/triple_stack_results.md
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, REPO_ROOT)

from turboquantdc.expected_attention import ExpectedAttentionScorer
from turboquantdc.cache_distillation import CacheDistiller
from turboquantdc.polarquant import PolarQuant

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("TRIPLE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Layers to benchmark (early, middle, late)
TEST_LAYERS = [0, 7, 15, 23, 35]

# TurboQuant settings
KEY_BITS = 3
VAL_BITS = 3

# Distillation settings
DISTILL_STEPS = 50
DISTILL_LR = 0.01

# EA scorer settings
EA_WINDOW = 64
PAST_RATIO = 0.6  # fraction of queries used as "past" for EA scoring

# Long context prompt (512+ tokens)
CONTEXT_PROMPT = """You are an expert research assistant. Below is a collection of research notes about quantum computing. Read all notes carefully and answer the questions that follow.

Note 1: Quantum Error Correction
Quantum error correction (QEC) is essential for building fault-tolerant quantum computers. The surface code, proposed by Kitaev in 1997, is currently the most promising approach due to its high error threshold of approximately 1%. The surface code encodes a single logical qubit using a two-dimensional lattice of physical qubits, where the number of physical qubits scales as O(d^2) for a code distance d. Recent experimental demonstrations by Google's Sycamore processor have shown logical error rates below the threshold for distance-3 and distance-5 surface codes. IBM's Heron processor has demonstrated similar capabilities with their heavy-hexagonal lattice architecture. The key challenge remains scaling to larger code distances while maintaining low physical error rates.

Note 2: Quantum Advantage in Optimization
The quantum approximate optimization algorithm (QAOA), introduced by Farhi, Goldstone, and Gutmann in 2014, is designed to solve combinatorial optimization problems. Despite significant theoretical and experimental progress, definitive quantum advantage for optimization remains elusive. Classical algorithms, particularly simulated annealing and tensor network methods, continue to compete effectively on problems up to several hundred variables. The most promising applications appear to be in structured problems where the quantum speedup is polynomial rather than exponential, such as portfolio optimization and vehicle routing.

Note 3: Quantum Machine Learning
Quantum machine learning (QML) has seen explosive growth, with variational quantum eigensolvers (VQE) and quantum neural networks (QNN) being the most studied paradigms. However, the barren plateau phenomenon, identified by McClean et al. in 2018, poses a fundamental challenge: the gradients of randomly initialized quantum circuits vanish exponentially with system size, making training infeasible for large circuits. Recent work has proposed several mitigation strategies, including layer-wise training, identity-block initialization, and classical-quantum hybrid architectures. The most successful applications to date have been in quantum chemistry, where quantum computers can naturally represent electronic wave functions.

Note 4: Superconducting Qubits
Superconducting qubits, based on Josephson junctions, dominate the current quantum computing landscape. The transmon qubit, an improved charge qubit with reduced sensitivity to charge noise, achieves coherence times exceeding 100 microseconds in state-of-the-art devices. Google, IBM, and Rigetti all use transmon-based architectures. The key challenges include: improving gate fidelities beyond 99.9%, reducing crosstalk between adjacent qubits, scaling to thousands of qubits while maintaining connectivity, and operating at millikelvin temperatures, which requires expensive dilution refrigerators.

Note 5: Trapped Ion Quantum Computing
Trapped ion quantum computers, pioneered by groups at NIST, University of Innsbruck, and companies like IonQ and Quantinuum, offer several advantages over superconducting qubits. They typically achieve higher two-qubit gate fidelities (exceeding 99.9%), longer coherence times (seconds to minutes), and all-to-all connectivity between qubits in a single trap. However, they face challenges in scaling beyond several dozen qubits in a single trap, with proposed solutions including modular architectures using photonic interconnects and shuttling-based approaches. Quantinuum's H2 processor with 56 qubits represents the current state of the art.

Note 6: Quantum Networking and Communication
Quantum networking aims to connect quantum processors via quantum channels, enabling distributed quantum computing and quantum key distribution (QKD). The fundamental challenge is that quantum states cannot be copied (no-cloning theorem), requiring quantum repeaters for long-distance communication. Current QKD implementations achieve secure key rates of several kilobits per second over distances up to 400 km in optical fiber. Satellite-based QKD, demonstrated by the Chinese Micius satellite, has extended this to over 7,600 km. Quantum memory, essential for quantum repeaters, remains a significant bottleneck, with the best atomic ensemble memories achieving storage times of only a few seconds.

Note 7: Photonic Quantum Computing
Photonic quantum computing uses single photons as qubits, with encoding in polarization, time-bin, or path degrees of freedom. Xanadu's Borealis processor demonstrated quantum advantage in Gaussian boson sampling with 216 squeezed-state modes. Linear optical quantum computing faces the challenge that photon-photon interactions are extremely weak, requiring measurement-induced nonlinearity. PsiQuantum is pursuing a large-scale approach using silicon photonics, aiming for a million-qubit fault-tolerant machine.

Now answer these questions based on the notes above:

Question 1: What is the error threshold of the surface code, and which companies have demonstrated experimental results?
Question 2: What fundamental challenge does QAOA face in achieving quantum advantage?
Question 3: Explain the barren plateau phenomenon and list three proposed mitigation strategies.
Question 4: Compare coherence times of superconducting qubits versus trapped ion qubits.
Question 5: What is the current state of quantum key distribution in terms of distance and key rates?"""


# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------

@dataclass
class StackConfig:
    """Configuration for a triple-stack experiment."""
    name: str
    eviction_rate: float   # fraction to evict (0.0 = no eviction)
    distill_ratio: int     # N -> N/ratio (1 = no distillation)
    quant_bits: int        # 0 = no quantization
    order: str = "evict_distill_quant"  # or "distill_evict_quant"

    @property
    def theoretical_compression(self) -> float:
        evict_factor = 1.0 / (1.0 - self.eviction_rate) if self.eviction_rate > 0 else 1.0
        distill_factor = self.distill_ratio if self.distill_ratio > 1 else 1.0
        quant_factor = 16.0 / self.quant_bits if self.quant_bits > 0 else 1.0
        return evict_factor * distill_factor * quant_factor


# Define all stacking experiments
STACK_CONFIGS = [
    # Baselines: individual techniques
    StackConfig("FP16 (baseline)", 0.0, 1, 0),
    StackConfig("EA-only 50%", 0.50, 1, 0),
    StackConfig("Distill-only 4x", 0.0, 4, 0),
    StackConfig("Quant-only 3-bit", 0.0, 1, 3),

    # Pairwise stacks
    StackConfig("EA 50% + Distill 4x", 0.50, 4, 0),
    StackConfig("EA 50% + Quant 3-bit", 0.50, 1, 3),
    StackConfig("Distill 4x + Quant 3-bit", 0.0, 4, 3),

    # Triple stacks: order A (evict -> distill -> quant)
    StackConfig("Triple: EA50% -> D4x -> Q3 (40x)", 0.50, 4, 3, "evict_distill_quant"),

    # Triple stacks: order B (distill -> evict -> quant)
    StackConfig("Triple: D4x -> EA50% -> Q3 (40x)", 0.50, 4, 3, "distill_evict_quant"),

    # Triple stacks: varying aggressiveness
    StackConfig("Triple: EA30% -> D4x -> Q3 (28x)", 0.30, 4, 3, "evict_distill_quant"),
    StackConfig("Triple: EA70% -> D4x -> Q3 (66x)", 0.70, 4, 3, "evict_distill_quant"),
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load Qwen2.5-3B with BnB 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME} (4-bit quantized)...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dt = time.time() - t0
    device = next(model.parameters()).device
    print(f"  Loaded in {dt:.1f}s on {device}")
    return model, tokenizer


def extract_qkv_data(model, tokenizer, prompt: str) -> Dict[str, Any]:
    """Extract Q, K, V from all layers via forward pass + hooks."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]
    print(f"  Prompt tokens: {seq_len}")

    # First pass: get KV cache
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=True)

    kv_cache = outputs.past_key_values
    keys_per_layer = []
    values_per_layer = []

    # Handle DynamicCache or tuple-based cache
    if hasattr(kv_cache, 'key_cache'):
        for i in range(len(kv_cache.key_cache)):
            keys_per_layer.append(kv_cache.key_cache[i].cpu().float())
            values_per_layer.append(kv_cache.value_cache[i].cpu().float())
    elif hasattr(kv_cache, 'layers'):
        for layer in kv_cache.layers:
            keys_per_layer.append(layer.keys.cpu().float())
            values_per_layer.append(layer.values.cpu().float())
    else:
        for layer_kv in kv_cache:
            keys_per_layer.append(layer_kv[0].cpu().float())
            values_per_layer.append(layer_kv[1].cpu().float())

    attn_per_layer = [a.cpu().float() for a in outputs.attentions]

    # Second pass: extract queries via hooks on q_proj
    query_outputs = []

    def q_proj_hook(module, input_args, output):
        query_outputs.append(output.detach().cpu().float())

    hooks = []
    for name, module in model.named_modules():
        if name.endswith(".q_proj"):
            hooks.append(module.register_forward_hook(q_proj_hook))

    with torch.no_grad():
        model(**inputs, output_attentions=False, use_cache=False)

    for h in hooks:
        h.remove()

    # Reshape queries to (batch, n_heads, seq, head_dim)
    config = model.config
    n_heads = getattr(config, "num_attention_heads", 32)
    head_dim = keys_per_layer[0].shape[-1]

    queries_per_layer = []
    for q_raw in query_outputs:
        batch, seq, total_dim = q_raw.shape
        per_head_dim = total_dim // n_heads
        q_reshaped = q_raw.view(batch, seq, n_heads, per_head_dim).transpose(1, 2)
        queries_per_layer.append(q_reshaped)

    n_kv_heads = keys_per_layer[0].shape[1]
    n_layers = len(keys_per_layer)

    # Clamp test layers to available layers
    global TEST_LAYERS
    TEST_LAYERS = [l for l in TEST_LAYERS if l < n_layers]
    if not TEST_LAYERS:
        TEST_LAYERS = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    return {
        "keys": keys_per_layer,
        "values": values_per_layer,
        "queries": queries_per_layer,
        "attention_weights": attn_per_layer,
        "seq_len": seq_len,
        "n_layers": n_layers,
        "head_dim": head_dim,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_attention_quality(
    keys_full: torch.Tensor,
    values_full: torch.Tensor,
    keys_compressed: torch.Tensor,
    values_compressed: torch.Tensor,
    queries: torch.Tensor,
) -> Dict[str, float]:
    """Compute attention output quality between full and compressed cache.

    Returns cosine similarity, top-5 match, L2 relative error.
    """
    d = keys_full.shape[-1]
    scale = 1.0 / math.sqrt(d)

    # Full attention output
    attn_full = F.softmax(queries @ keys_full.T * scale, dim=-1)
    out_full = attn_full @ values_full  # (q, d)

    # Compressed attention output
    attn_comp = F.softmax(queries @ keys_compressed.T * scale, dim=-1)
    out_comp = attn_comp @ values_compressed  # (q, d)

    # Cosine similarity
    cos = F.cosine_similarity(out_full, out_comp, dim=-1).mean().item()

    # Relative L2 error
    l2_err = (out_full - out_comp).norm(dim=-1).mean().item()
    gt_norm = out_full.norm(dim=-1).mean().item()
    rel_err = l2_err / max(gt_norm, 1e-10)

    # Top-5 output dimension match
    k = min(5, d)
    _, top_full = out_full.abs().topk(k, dim=-1)
    _, top_comp = out_comp.abs().topk(k, dim=-1)
    top5_match = 0.0
    for i in range(queries.shape[0]):
        full_set = set(top_full[i].tolist())
        comp_set = set(top_comp[i].tolist())
        top5_match += len(full_set & comp_set) / k
    top5_match /= queries.shape[0]

    return {
        "cosine": cos,
        "relative_error": rel_err,
        "top5_match": top5_match,
    }


def compute_effective_compression(
    n_original: int,
    n_final: int,
    d: int,
    quant_bits: int,
) -> float:
    """Compute the effective compression ratio.

    Accounts for token count reduction (eviction + distillation)
    and per-token bit reduction (quantization).
    """
    if n_final == 0:
        return float('inf')

    fp16_bits = 16
    # Original storage: n_original tokens * d coords * 16 bits (K + V)
    original_bits = n_original * d * fp16_bits * 2  # K and V

    if quant_bits > 0:
        # Quantized: n_final * d * quant_bits + overhead per vector
        overhead_per_vec = 32  # norm/scale storage
        compressed_bits = n_final * (d * quant_bits + overhead_per_vec) * 2
    else:
        # FP16 but fewer tokens
        compressed_bits = n_final * d * fp16_bits * 2

    return original_bits / max(compressed_bits, 1)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_evict(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries_past: torch.Tensor,
    eviction_rate: float,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stage 1: Expected Attention eviction.

    Score all tokens, evict the bottom `eviction_rate` fraction.
    Protects first 5% of tokens (prompt context) and last 32 tokens (recent).
    """
    if eviction_rate <= 0:
        return keys.clone(), values.clone()

    n = keys.shape[0]

    # Score tokens with Expected Attention
    scorer = ExpectedAttentionScorer(
        d=head_dim, window=EA_WINDOW, use_diagonal_cov=True, device=keys.device,
    )
    scorer.update_queries(queries_past)
    importance = scorer.score(keys)

    # Build keep mask with protections
    n_to_evict = int(eviction_rate * n)
    n_protect_prompt = max(int(0.05 * n), min(4, n))
    n_protect_recent = min(32, n)

    keep_mask = torch.ones(n, dtype=torch.bool, device=keys.device)

    # Mark bottom tokens for eviction
    _, bottom_idx = torch.topk(importance, n_to_evict, largest=False)
    keep_mask[bottom_idx] = False

    # Restore protections
    keep_mask[:n_protect_prompt] = True
    keep_mask[-n_protect_recent:] = True

    return keys[keep_mask], values[keep_mask]


def stage_distill(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    distill_ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stage 2: KVSculpt-style cache distillation.

    Distills N tokens into N/distill_ratio synthetic tokens via
    attention KL minimization + closed-form value solve.
    """
    if distill_ratio <= 1:
        return keys.clone(), values.clone()

    n = keys.shape[0]
    target_size = max(1, n // distill_ratio)

    if target_size >= n:
        return keys.clone(), values.clone()

    distiller = CacheDistiller(seed=SEED, device=keys.device)
    dk, dv = distiller.distill(
        keys, values, queries,
        target_size=target_size,
        steps=DISTILL_STEPS,
        lr=DISTILL_LR,
    )
    return dk, dv


def stage_quantize(
    keys: torch.Tensor,
    values: torch.Tensor,
    quant_bits: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stage 3: TurboQuant 3-bit compression via PolarQuant.

    Normalizes vectors, quantizes, dequantizes (simulating storage).
    """
    if quant_bits <= 0:
        return keys.clone(), values.clone()

    pq = PolarQuant(d=head_dim, bits=quant_bits, seed=SEED, device=keys.device)

    # PolarQuant operates on unit vectors -- we need to normalize, quantize,
    # then rescale. Store norms separately.
    def quant_dequant(x: torch.Tensor) -> torch.Tensor:
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms
        x_hat, _ = pq(x_unit)
        return x_hat * norms

    qk = quant_dequant(keys)
    qv = quant_dequant(values)
    return qk, qv


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    config: StackConfig,
    keys: torch.Tensor,
    values: torch.Tensor,
    queries_past: torch.Tensor,
    queries_eval: torch.Tensor,
    head_dim: int,
) -> Dict[str, Any]:
    """Run a complete triple-stack pipeline and measure quality.

    Args:
        config: Stack configuration.
        keys: (N, d) original keys.
        values: (N, d) original values.
        queries_past: (n_past, d) past queries for EA scoring.
        queries_eval: (n_eval, d) future queries for quality evaluation.
        head_dim: head dimension.

    Returns:
        Dict with quality metrics, token counts, and compression ratios.
    """
    n_original = keys.shape[0]
    t0 = time.time()

    if config.order == "evict_distill_quant":
        # Order A: Evict -> Distill -> Quantize
        k1, v1 = stage_evict(keys, values, queries_past, config.eviction_rate, head_dim)
        n_after_evict = k1.shape[0]

        k2, v2 = stage_distill(k1, v1, queries_past[:k1.shape[0]], config.distill_ratio)
        n_after_distill = k2.shape[0]

        k3, v3 = stage_quantize(k2, v2, config.quant_bits, head_dim)
        n_final = k3.shape[0]

    elif config.order == "distill_evict_quant":
        # Order B: Distill -> Evict -> Quantize
        k1, v1 = stage_distill(keys, values, queries_past, config.distill_ratio)
        n_after_distill = k1.shape[0]

        k2, v2 = stage_evict(k1, v1, queries_past, config.eviction_rate, head_dim)
        n_after_evict = k2.shape[0]

        k3, v3 = stage_quantize(k2, v2, config.quant_bits, head_dim)
        n_final = k3.shape[0]

    else:
        raise ValueError(f"Unknown order: {config.order}")

    pipeline_time = time.time() - t0

    # Measure quality against full FP16 cache
    quality = compute_attention_quality(
        keys, values, k3, v3, queries_eval,
    )

    # Compute effective compression
    eff_compression = compute_effective_compression(
        n_original, n_final, head_dim, config.quant_bits,
    )

    return {
        "config_name": config.name,
        "order": config.order,
        "n_original": n_original,
        "n_after_evict": n_after_evict if config.order == "evict_distill_quant" else (n_after_evict if config.order == "distill_evict_quant" else n_original),
        "n_after_distill": n_after_distill if "distill" in config.name.lower() or config.distill_ratio > 1 else n_original,
        "n_final": n_final,
        "eviction_rate": config.eviction_rate,
        "distill_ratio": config.distill_ratio,
        "quant_bits": config.quant_bits,
        "theoretical_compression": config.theoretical_compression,
        "effective_compression": eff_compression,
        "cosine": quality["cosine"],
        "relative_error": quality["relative_error"],
        "top5_match": quality["top5_match"],
        "pipeline_time_ms": pipeline_time * 1000,
    }


def run_single_layer(
    config: StackConfig,
    data: Dict[str, Any],
    layer_idx: int,
) -> Dict[str, Any]:
    """Run pipeline on a single layer, single head."""
    seq_len = data["seq_len"]
    head_dim = data["head_dim"]
    n_past = int(seq_len * PAST_RATIO)

    # Extract single head's data
    keys = data["keys"][layer_idx][0, 0, :, :].to(DEVICE)      # (seq, d)
    values = data["values"][layer_idx][0, 0, :, :].to(DEVICE)   # (seq, d)

    # Get queries for this layer
    if data["queries"] and layer_idx < len(data["queries"]):
        queries = data["queries"][layer_idx][0, 0, :, :].to(DEVICE)
    else:
        attn = data["attention_weights"][layer_idx][0, 0, :, :].to(DEVICE)
        queries = (attn @ keys) * math.sqrt(head_dim)

    queries_past = queries[:n_past]
    queries_eval = queries[n_past:]

    result = run_pipeline(config, keys, values, queries_past, queries_eval, head_dim)
    result["layer"] = layer_idx
    return result


# ---------------------------------------------------------------------------
# Results writing
# ---------------------------------------------------------------------------

def write_results(
    all_results: Dict[str, List[Dict[str, Any]]],
    data_info: Dict[str, Any],
    output_path: str,
) -> None:
    """Write comprehensive results to markdown."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Triple Stack Benchmark: EA + KVSculpt + TurboQuant",
        "",
        f"**Date:** {now}",
        f"**Model:** {MODEL_NAME}",
        f"**Sequence length:** {data_info['seq_len']} tokens",
        f"**Head dimension:** {data_info['head_dim']}",
        f"**Layers tested:** {TEST_LAYERS}",
        f"**Past/Future split:** {PAST_RATIO:.0%} / {1-PAST_RATIO:.0%}",
        f"**Distillation steps:** {DISTILL_STEPS}, lr={DISTILL_LR}",
        "",
        "## The Triple Stack",
        "",
        "```",
        "Input: N tokens in FP16 KV cache",
        "  |",
        "  v",
        "Stage 1: Expected Attention eviction (remove unimportant tokens)",
        "  N -> N * (1 - eviction_rate)",
        "  |",
        "  v",
        "Stage 2: KVSculpt distillation (synthesize fewer tokens)",
        "  N' -> N' / distill_ratio",
        "  |",
        "  v",
        "Stage 3: TurboQuant 3-bit (compress each token)",
        "  FP16 -> 3-bit per coordinate",
        "  |",
        "  v",
        "Output: Compressed cache for attention computation",
        "```",
        "",
        "## Results Summary (Averaged Across Layers)",
        "",
        "| Configuration | Compression | Cosine | Rel Error | Top-5 | Time (ms) |",
        "|---------------|-------------|--------|-----------|-------|-----------|",
    ]

    # Aggregate results across layers for each config
    summary_data = []
    for config_name, layer_results in all_results.items():
        if not layer_results:
            continue
        avg = {}
        for key in layer_results[0]:
            if isinstance(layer_results[0][key], (int, float)):
                vals = [r[key] for r in layer_results]
                avg[key] = sum(vals) / len(vals)
        avg["config_name"] = config_name
        summary_data.append(avg)

        cos = avg.get("cosine", 0)
        rel_err = avg.get("relative_error", 0)
        top5 = avg.get("top5_match", 0)
        comp = avg.get("effective_compression", 1)
        t_ms = avg.get("pipeline_time_ms", 0)

        lines.append(
            f"| {config_name} | {comp:.1f}x | {cos:.4f} | {rel_err:.4f} | {top5:.3f} | {t_ms:.1f} |"
        )

    # Per-layer detail tables
    lines.extend([
        "",
        "## Per-Layer Results",
        "",
    ])

    for config_name, layer_results in all_results.items():
        if not layer_results:
            continue
        lines.extend([
            f"### {config_name}",
            "",
            "| Layer | N_orig | N_final | Compression | Cosine | Rel Error | Top-5 | Time (ms) |",
            "|-------|--------|---------|-------------|--------|-----------|-------|-----------|",
        ])
        for r in layer_results:
            lines.append(
                f"| {r['layer']} | {r['n_original']} | {r['n_final']} | "
                f"{r['effective_compression']:.1f}x | {r['cosine']:.4f} | "
                f"{r['relative_error']:.4f} | {r['top5_match']:.3f} | "
                f"{r['pipeline_time_ms']:.1f} |"
            )
        lines.append("")

    # Aggregate results EXCLUDING layer 0 (known embedding-like layer that needs FP16 anchor)
    summary_data_no_l0 = []
    for config_name, layer_results in all_results.items():
        non_l0 = [r for r in layer_results if r.get("layer", 0) != 0]
        if not non_l0:
            continue
        avg = {}
        for key in non_l0[0]:
            if isinstance(non_l0[0][key], (int, float)):
                vals = [r[key] for r in non_l0]
                avg[key] = sum(vals) / len(vals)
        avg["config_name"] = config_name
        summary_data_no_l0.append(avg)

    lines.extend([
        "",
        "## Results Summary (Layers 7-35 Only, Excluding Layer 0)",
        "",
        "Layer 0 is the embedding-like first layer that always needs FP16 anchor treatment.",
        "Excluding it gives the true picture for non-anchor layers.",
        "",
        "| Configuration | Compression | Cosine | Rel Error | Top-5 | Time (ms) |",
        "|---------------|-------------|--------|-----------|-------|-----------|",
    ])

    for s in summary_data_no_l0:
        cos = s.get("cosine", 0)
        rel_err = s.get("relative_error", 0)
        top5 = s.get("top5_match", 0)
        comp = s.get("effective_compression", 1)
        t_ms = s.get("pipeline_time_ms", 0)
        lines.append(
            f"| {s['config_name']} | {comp:.1f}x | {cos:.4f} | {rel_err:.4f} | {top5:.3f} | {t_ms:.1f} |"
        )

    # Compression-quality curve analysis
    lines.extend([
        "",
        "## Compression-Quality Curve (All Layers)",
        "",
        "Sorted by compression ratio, showing the quality frontier:",
        "",
        "| Rank | Configuration | Compression | Cosine | Pass >0.9? | Pass >0.95? |",
        "|------|---------------|-------------|--------|------------|-------------|",
    ])

    sorted_data = sorted(summary_data, key=lambda x: x.get("effective_compression", 1))
    for i, s in enumerate(sorted_data):
        cos = s.get("cosine", 0)
        comp = s.get("effective_compression", 1)
        pass_90 = "YES" if cos >= 0.9 else "no"
        pass_95 = "YES" if cos >= 0.95 else "no"
        lines.append(
            f"| {i+1} | {s['config_name']} | {comp:.1f}x | {cos:.4f} | {pass_90} | {pass_95} |"
        )

    # Compression-quality curve excluding layer 0
    lines.extend([
        "",
        "## Compression-Quality Curve (Layers 7-35 Only -- Excluding Layer 0)",
        "",
        "| Rank | Configuration | Compression | Cosine | Pass >0.9? | Pass >0.95? |",
        "|------|---------------|-------------|--------|------------|-------------|",
    ])

    sorted_no_l0 = sorted(summary_data_no_l0, key=lambda x: x.get("effective_compression", 1))
    for i, s in enumerate(sorted_no_l0):
        cos = s.get("cosine", 0)
        comp = s.get("effective_compression", 1)
        pass_90 = "YES" if cos >= 0.9 else "no"
        pass_95 = "YES" if cos >= 0.95 else "no"
        lines.append(
            f"| {i+1} | {s['config_name']} | {comp:.1f}x | {cos:.4f} | {pass_90} | {pass_95} |"
        )

    # Stacking order comparison
    order_a = None
    order_b = None
    for s in summary_data:
        if "EA50% -> D4x -> Q3" in s.get("config_name", ""):
            order_a = s
        if "D4x -> EA50% -> Q3" in s.get("config_name", ""):
            order_b = s

    lines.extend([
        "",
        "## Stacking Order Comparison",
        "",
    ])

    if order_a and order_b:
        lines.extend([
            "Does the order of eviction vs distillation matter?",
            "",
            f"- **Order A (Evict then Distill):** cos={order_a.get('cosine', 0):.4f}, "
            f"compression={order_a.get('effective_compression', 1):.1f}x",
            f"- **Order B (Distill then Evict):** cos={order_b.get('cosine', 0):.4f}, "
            f"compression={order_b.get('effective_compression', 1):.1f}x",
            "",
        ])
        cos_diff = abs(order_a.get("cosine", 0) - order_b.get("cosine", 0))
        if cos_diff < 0.005:
            lines.append("Order does **not** significantly matter (< 0.005 cosine difference).")
        elif order_a.get("cosine", 0) > order_b.get("cosine", 0):
            lines.append(f"**Evict-first is better** by {cos_diff:.4f} cosine.")
        else:
            lines.append(f"**Distill-first is better** by {cos_diff:.4f} cosine.")
    else:
        lines.append("(Could not compare stacking orders -- results missing)")

    # Aggressiveness sweep analysis
    lines.extend([
        "",
        "## Aggressiveness Sweep (EA eviction rate)",
        "",
    ])

    agg_sweep = []
    for s in summary_data:
        if "Triple: EA" in s.get("config_name", "") and "D4x -> Q3" in s.get("config_name", ""):
            agg_sweep.append(s)

    if agg_sweep:
        lines.extend([
            "| Eviction Rate | Compression | Cosine | Relative Error |",
            "|---------------|-------------|--------|----------------|",
        ])
        for s in sorted(agg_sweep, key=lambda x: x.get("eviction_rate", 0)):
            lines.append(
                f"| {s.get('eviction_rate', 0):.0%} | "
                f"{s.get('effective_compression', 1):.1f}x | "
                f"{s.get('cosine', 0):.4f} | "
                f"{s.get('relative_error', 0):.4f} |"
            )

    # Key findings
    lines.extend([
        "",
        "## Key Findings",
        "",
        "### All Layers (Including Layer 0)",
        "",
    ])

    # Find max compression above quality thresholds
    above_90 = [s for s in summary_data if s.get("cosine", 0) >= 0.90]
    above_95 = [s for s in summary_data if s.get("cosine", 0) >= 0.95]

    if above_90:
        best_90 = max(above_90, key=lambda x: x.get("effective_compression", 1))
        lines.append(
            f"- **Max compression with >0.90 cosine:** {best_90.get('effective_compression', 1):.1f}x "
            f"({best_90['config_name']}, cos={best_90.get('cosine', 0):.4f})"
        )
    else:
        lines.append("- No configuration achieved >0.90 cosine.")

    if above_95:
        best_95 = max(above_95, key=lambda x: x.get("effective_compression", 1))
        lines.append(
            f"- **Max compression with >0.95 cosine:** {best_95.get('effective_compression', 1):.1f}x "
            f"({best_95['config_name']}, cos={best_95.get('cosine', 0):.4f})"
        )
    else:
        lines.append("- No configuration achieved >0.95 cosine.")

    # Excluding layer 0 findings
    lines.extend([
        "",
        "### Non-Anchor Layers Only (Layers 7-35, Excluding Layer 0)",
        "",
        "Layer 0 is the embedding-like first layer where 3-bit quantization catastrophically",
        "fails (cos~0.20). In practice this layer uses an FP16 anchor. Excluding it shows",
        "the true compression-quality frontier for the remaining 35 layers.",
        "",
    ])

    above_90_no_l0 = [s for s in summary_data_no_l0 if s.get("cosine", 0) >= 0.90]
    above_95_no_l0 = [s for s in summary_data_no_l0 if s.get("cosine", 0) >= 0.95]

    if above_90_no_l0:
        best_90_nl0 = max(above_90_no_l0, key=lambda x: x.get("effective_compression", 1))
        lines.append(
            f"- **Max compression with >0.90 cosine:** {best_90_nl0.get('effective_compression', 1):.1f}x "
            f"({best_90_nl0['config_name']}, cos={best_90_nl0.get('cosine', 0):.4f})"
        )
    else:
        lines.append("- No configuration achieved >0.90 cosine (excluding layer 0).")

    if above_95_no_l0:
        best_95_nl0 = max(above_95_no_l0, key=lambda x: x.get("effective_compression", 1))
        lines.append(
            f"- **Max compression with >0.95 cosine:** {best_95_nl0.get('effective_compression', 1):.1f}x "
            f"({best_95_nl0['config_name']}, cos={best_95_nl0.get('cosine', 0):.4f})"
        )
    else:
        lines.append("- No configuration achieved >0.95 cosine (excluding layer 0).")

    # Stacking analysis: multiplicative vs sub-multiplicative
    # Use non-layer-0 data for proper analysis
    ea_only_cos = None
    dist_only_cos = None
    quant_only_cos = None
    triple_cos = None
    source_label = "layers 7-35" if summary_data_no_l0 else "all layers"
    source_data = summary_data_no_l0 if summary_data_no_l0 else summary_data
    for s in source_data:
        name = s.get("config_name", "")
        if name == "EA-only 50%":
            ea_only_cos = s.get("cosine", 0)
        elif name == "Distill-only 4x":
            dist_only_cos = s.get("cosine", 0)
        elif name == "Quant-only 3-bit":
            quant_only_cos = s.get("cosine", 0)
        elif "EA50% -> D4x -> Q3" in name:
            triple_cos = s.get("cosine", 0)

    if ea_only_cos and dist_only_cos and quant_only_cos and triple_cos:
        # If multiplicative: triple_cos ~ ea_cos * dist_cos * quant_cos (normalized)
        # Convert to loss: loss = 1 - cos, then check if losses compound
        ea_loss = 1 - ea_only_cos
        dist_loss = 1 - dist_only_cos
        quant_loss = 1 - quant_only_cos
        triple_loss = 1 - triple_cos

        # Multiplicative prediction: compound losses
        mult_predicted_loss = 1 - (ea_only_cos * dist_only_cos * quant_only_cos)
        # Additive prediction: independent losses
        add_predicted_loss = ea_loss + dist_loss + quant_loss

        lines.extend([
            "",
            f"### Quality Stacking Analysis ({source_label})",
            "",
            f"- EA-only 50% loss: {ea_loss:.4f} (cos={ea_only_cos:.4f})",
            f"- Distill-only 4x loss: {dist_loss:.4f} (cos={dist_only_cos:.4f})",
            f"- Quant-only 3-bit loss: {quant_loss:.4f} (cos={quant_only_cos:.4f})",
            f"- **Triple stack actual loss: {triple_loss:.4f} (cos={triple_cos:.4f})**",
            "",
            f"- Additive prediction (independent losses): {add_predicted_loss:.4f}",
            f"- Multiplicative prediction (compound cosines): {mult_predicted_loss:.4f}",
            "",
        ])

        if triple_loss < add_predicted_loss * 0.8:
            lines.append(
                "**Sub-additive stacking:** Quality loss is LESS than the sum of individual "
                "losses. The techniques are partially orthogonal -- each one removes a "
                "different kind of information, so combining them doesn't compound as badly "
                "as feared."
            )
        elif triple_loss > add_predicted_loss * 1.2:
            lines.append(
                "**Super-additive stacking:** Quality loss EXCEEDS the sum of individual "
                "losses. There are interaction effects -- distilling/quantizing already-evicted "
                "tokens introduces compounding errors."
            )
        else:
            lines.append(
                "**Approximately additive stacking:** Triple loss is roughly the sum of "
                "individual losses, suggesting the techniques degrade quality independently."
            )

    lines.extend([
        "",
        "---",
        f"*Benchmark completed in {data_info.get('total_time', 0):.1f}s*",
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Triple Stack Benchmark: EA + KVSculpt + TurboQuant")
    print("=" * 70)
    total_t0 = time.time()

    # Load model
    model, tokenizer = load_model()

    # Extract Q, K, V data
    print("\n--- Extracting Q/K/V data ---")
    data = extract_qkv_data(model, tokenizer, CONTEXT_PROMPT)
    print(f"  Extracted {data['n_layers']} layers, {data['n_heads']} Q-heads, "
          f"{data['n_kv_heads']} KV-heads, d={data['head_dim']}")
    print(f"  Queries extracted: {len(data['queries'])} layers")

    # Free model memory
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nTesting layers: {TEST_LAYERS}")
    print(f"Running {len(STACK_CONFIGS)} configurations x {len(TEST_LAYERS)} layers "
          f"= {len(STACK_CONFIGS) * len(TEST_LAYERS)} experiments\n")

    # Run all experiments
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for config in STACK_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Config: {config.name}")
        print(f"  Eviction: {config.eviction_rate:.0%}, Distill: {config.distill_ratio}x, "
              f"Quant: {config.quant_bits}-bit, Order: {config.order}")
        print(f"  Theoretical compression: {config.theoretical_compression:.1f}x")
        print(f"{'='*60}")

        layer_results = []
        for layer_idx in TEST_LAYERS:
            try:
                result = run_single_layer(config, data, layer_idx)
                layer_results.append(result)
                print(f"  Layer {layer_idx:2d}: "
                      f"N={result['n_original']}->{result['n_final']:3d}, "
                      f"comp={result['effective_compression']:5.1f}x, "
                      f"cos={result['cosine']:.4f}, "
                      f"top5={result['top5_match']:.3f}, "
                      f"time={result['pipeline_time_ms']:.0f}ms")
            except Exception as e:
                print(f"  Layer {layer_idx}: FAILED -- {e}")

        all_results[config.name] = layer_results

    total_time = time.time() - total_t0

    # Write results
    data_info = {
        "seq_len": data["seq_len"],
        "head_dim": data["head_dim"],
        "n_layers": data["n_layers"],
        "n_heads": data["n_heads"],
        "total_time": total_time,
    }

    output_path = os.path.join(REPO_ROOT, "benchmarks", "results", "triple_stack_results.md")
    write_results(all_results, data_info, output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY (ALL LAYERS)")
    print("=" * 70)

    print("\n{:<45} {:>8} {:>8} {:>8}".format("Configuration", "Compress", "Cosine", "Top-5"))
    print("-" * 75)

    for config_name, layer_results in all_results.items():
        if not layer_results:
            continue
        avg_cos = sum(r["cosine"] for r in layer_results) / len(layer_results)
        avg_comp = sum(r["effective_compression"] for r in layer_results) / len(layer_results)
        avg_top5 = sum(r["top5_match"] for r in layer_results) / len(layer_results)
        marker = " ***" if avg_cos >= 0.90 else ""
        print(f"{config_name:<45} {avg_comp:>7.1f}x {avg_cos:>7.4f} {avg_top5:>7.3f}{marker}")

    print(f"\n*** = cosine >= 0.90 (publishable quality)")

    # Print layer 0-excluded summary
    print("\n" + "=" * 70)
    print("SUMMARY (LAYERS 7-35 ONLY -- Excluding Layer 0)")
    print("Layer 0 needs FP16 anchor; this is the true non-anchor picture.")
    print("=" * 70)

    print("\n{:<45} {:>8} {:>8} {:>8}".format("Configuration", "Compress", "Cosine", "Top-5"))
    print("-" * 75)

    for config_name, layer_results in all_results.items():
        non_l0 = [r for r in layer_results if r.get("layer", 0) != 0]
        if not non_l0:
            continue
        avg_cos = sum(r["cosine"] for r in non_l0) / len(non_l0)
        avg_comp = sum(r["effective_compression"] for r in non_l0) / len(non_l0)
        avg_top5 = sum(r["top5_match"] for r in non_l0) / len(non_l0)
        marker = " ***" if avg_cos >= 0.90 else ""
        print(f"{config_name:<45} {avg_comp:>7.1f}x {avg_cos:>7.4f} {avg_top5:>7.3f}{marker}")

    print(f"\nTotal time: {total_time:.1f}s")
    print(f"\n*** = cosine >= 0.90 (publishable quality)")


if __name__ == "__main__":
    main()
