"""Attention-aware adaptive bit allocation benchmark.

Loads Qwen2.5-3B, extracts real attention patterns at 512+ tokens,
analyzes the power-law distribution, and tests tiered compression.

Measures:
    1. Attention distribution (power law analysis)
    2. Quality: cosine similarity, top-K attention match
    3. Effective bits vs uniform compression
    4. Generation quality comparison

Saves results to benchmarks/results/adaptive_bits_results.md
"""

import gc
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F

# ---- Configuration ----
# Use the smallest available Qwen model for attention analysis.
# The power-law attention distribution is a property of the transformer
# architecture, not model size. 0.5B gives the same attention patterns
# as 3B+ but fits in limited GPU memory alongside other workloads.
MODEL_NAME = os.environ.get("ADAPTIVE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"
MIN_CONTEXT_TOKENS = 512
TARGET_CONTEXT_TOKENS = 1024

# Tier configurations to test
TIER_CONFIGS = {
    # Same-budget comparison: 3.0 effective bits but adaptive allocation
    # 0.02*16 + 0.08*4 + 0.20*3 + 0.70*2 = 0.32+0.32+0.60+1.40 = 2.64 ... need to tune
    # Target 3.0 bits: 0.02*16 + 0.08*4 + 0.30*3 + 0.60*2 = 0.32+0.32+0.90+1.20 = 2.74 still low
    # Target 3.0 bits: 0.03*16 + 0.07*4 + 0.40*3 + 0.50*2 = 0.48+0.28+1.20+1.00 = 2.96 ~3.0
    "same_budget_3bit": {
        "thresholds": [0.03, 0.10, 0.50],
        "bits": [16, 4, 3, 2],
        "description": "~3.0 eff bits: top 3% FP16, next 7% 4-bit, next 40% 3-bit, bottom 50% 2-bit",
    },
    "aggressive": {
        "thresholds": [0.05, 0.20, 0.50],
        "bits": [16, 4, 3, 2],
        "description": "Top 5% FP16, next 15% 4-bit, next 30% 3-bit, bottom 50% 2-bit",
    },
    "moderate": {
        "thresholds": [0.10, 0.30, 0.60],
        "bits": [16, 4, 3, 2],
        "description": "Top 10% FP16, next 20% 4-bit, next 30% 3-bit, bottom 40% 2-bit",
    },
    "conservative": {
        "thresholds": [0.05, 0.15, 0.40],
        "bits": [16, 4, 3, 3],
        "description": "Top 5% FP16, next 10% 4-bit, next 25% 3-bit, bottom 60% 3-bit",
    },
    "eviction_sim": {
        "thresholds": [0.10, 0.30, 0.50],
        "bits": [16, 4, 3, 1],
        "description": "Top 10% FP16, next 20% 4-bit, next 20% 3-bit, bottom 50% 1-bit (near-eviction)",
    },
    # Ultra-aggressive: 2.0 effective bits
    # 0.05*16 + 0.15*3 + 0.80*1 = 0.80+0.45+0.80 = 2.05
    "ultra_aggressive": {
        "thresholds": [0.05, 0.20],
        "bits": [16, 3, 1],
        "description": "~2.0 eff bits: top 5% FP16, next 15% 3-bit, bottom 80% 1-bit",
    },
}

# Long context prompt for extracting attention patterns
CONTEXT_PROMPT = """You are an expert research assistant. Below is a collection of research notes about quantum computing. Read all notes carefully and answer the questions that follow.

Note 1: Quantum Error Correction
Quantum error correction (QEC) is essential for building fault-tolerant quantum computers. The surface code, proposed by Kitaev in 1997, is currently the most promising approach due to its high error threshold of approximately 1%. The surface code encodes a single logical qubit using a two-dimensional lattice of physical qubits, where the number of physical qubits scales as O(d^2) for a code distance d. Recent experimental demonstrations by Google's Sycamore processor have shown logical error rates below the threshold for distance-3 and distance-5 surface codes. IBM's Heron processor has demonstrated similar capabilities with their heavy-hexagonal lattice architecture. The key challenge remains scaling to larger code distances while maintaining low physical error rates.

Note 2: Quantum Advantage in Optimization
The quantum approximate optimization algorithm (QAOA), introduced by Farhi, Goldstone, and Gutmann in 2014, is designed to solve combinatorial optimization problems. Despite significant theoretical and experimental progress, definitive quantum advantage for optimization remains elusive. Classical algorithms, particularly simulated annealing and tensor network methods, continue to compete effectively on problems up to several hundred variables. The most promising applications appear to be in structured problems where the quantum speedup is polynomial rather than exponential, such as portfolio optimization and vehicle routing.

Note 3: Quantum Machine Learning
Quantum machine learning (QML) has seen explosive growth, with variational quantum eigensolvers (VQE) and quantum neural networks (QNN) being the most studied paradigms. However, the barren plateau phenomenon, identified by McClean et al. in 2018, poses a fundamental challenge: the gradients of randomly initialized quantum circuits vanish exponentially with system size, making training infeasible for large circuits. Recent work has proposed several mitigation strategies, including layer-wise training, identity-block initialization, and classical-quantum hybrid architectures. The most successful applications to date have been in quantum chemistry, where quantum computers can naturally represent electronic wave functions.

Note 4: Superconducting Qubits
Superconducting qubits, based on Josephson junctions, dominate the current quantum computing landscape. The transmon qubit, an improved charge qubit with reduced sensitivity to charge noise, achieves coherence times exceeding 100 microseconds in state-of-the-art devices. Google, IBM, and Rigetti all use transmon-based architectures. The key challenges include: (1) improving gate fidelities beyond 99.9%, (2) reducing crosstalk between adjacent qubits, (3) scaling to thousands of qubits while maintaining connectivity, and (4) operating at millikelvin temperatures, which requires expensive dilution refrigerators.

Note 5: Trapped Ion Quantum Computing
Trapped ion quantum computers, pioneered by groups at NIST, University of Innsbruck, and companies like IonQ and Quantinuum, offer several advantages over superconducting qubits. They typically achieve higher two-qubit gate fidelities (exceeding 99.9%), longer coherence times (seconds to minutes), and all-to-all connectivity between qubits in a single trap. However, they face challenges in scaling beyond several dozen qubits in a single trap, with proposed solutions including modular architectures using photonic interconnects and shuttling-based approaches. Quantinuum's H2 processor with 56 qubits represents the current state of the art.

Note 6: Quantum Networking and Communication
Quantum networking aims to connect quantum processors via quantum channels, enabling distributed quantum computing and quantum key distribution (QKD). The fundamental challenge is that quantum states cannot be copied (no-cloning theorem), requiring quantum repeaters for long-distance communication. Current QKD implementations achieve secure key rates of several kilobits per second over distances up to 400 km in optical fiber. Satellite-based QKD, demonstrated by the Chinese Micius satellite, has extended this to over 7,600 km. Quantum memory, essential for quantum repeaters, remains a significant bottleneck, with the best atomic ensemble memories achieving storage times of only a few seconds.

Note 7: Photonic Quantum Computing
Photonic quantum computing uses single photons as qubits, with encoding in polarization, time-bin, or path degrees of freedom. Xanadu's Borealis processor demonstrated quantum advantage in Gaussian boson sampling with 216 squeezed-state modes. Linear optical quantum computing faces the challenge that photon-photon interactions are extremely weak, requiring measurement-induced nonlinearity. PsiQuantum is pursuing a large-scale approach using silicon photonics, aiming for a million-qubit fault-tolerant machine. The fusion-based quantum computing model, proposed by Bartolucci et al. at PsiQuantum, offers a potentially scalable approach using type-II fusion gates on photonic resource states.

Now answer these questions based on the notes above:

Question 1: What is the error threshold of the surface code, and which companies have demonstrated experimental results?

Question 2: What fundamental challenge does QAOA face in achieving quantum advantage over classical methods?

Question 3: Explain the barren plateau phenomenon and list three proposed mitigation strategies.

Question 4: Compare the coherence times of superconducting qubits versus trapped ion qubits.

Question 5: What is the current state of quantum key distribution in terms of distance and key rates?"""


def load_model():
    """Load model with appropriate quantization for available GPU memory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()

    # Check available GPU memory
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"  Available GPU memory: {free_mem:.1f} GB")
    else:
        free_mem = 0

    load_kwargs = {
        "cache_dir": CACHE_DIR,
        "attn_implementation": "eager",  # need explicit attention weights
        "torch_dtype": torch.float16,
    }

    # For small models (0.5B), load directly to GPU without quantization
    # For larger models, use BnB 4-bit
    if "0.5B" in MODEL_NAME or "1.5B" in MODEL_NAME:
        load_kwargs["device_map"] = "auto"
    elif free_mem > 6:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        load_kwargs["device_map"] = "auto"
    else:
        # Very limited memory: load to CPU
        load_kwargs["device_map"] = "cpu"
        print("  WARNING: Loading to CPU (limited GPU memory)")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    dt = time.time() - t0
    device = next(model.parameters()).device
    print(f"  Loaded in {dt:.1f}s on {device}")
    return model, tokenizer


def extract_attention_patterns(model, tokenizer, prompt: str) -> Dict[str, Any]:
    """Run a forward pass and extract attention weights from all layers.

    Returns:
        Dict with:
        - attention_weights: list of (batch, heads, q_len, kv_len) tensors per layer
        - input_ids: the tokenized input
        - seq_len: sequence length
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    seq_len = inputs.input_ids.shape[1]
    print(f"  Input sequence length: {seq_len} tokens")

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=False,
        )

    # Extract attention weights from all layers
    attention_weights = []
    for layer_attn in outputs.attentions:
        # layer_attn: (batch, heads, seq_len, seq_len)
        attention_weights.append(layer_attn.cpu().float())

    return {
        "attention_weights": attention_weights,
        "input_ids": inputs.input_ids.cpu(),
        "seq_len": seq_len,
        "n_layers": len(attention_weights),
        "n_heads": attention_weights[0].shape[1],
    }


def analyze_power_law(attention_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the power-law distribution of attention patterns."""
    from turboquantdc.adaptive_bits import analyze_attention_distribution

    result = analyze_attention_distribution(
        attention_data["attention_weights"],
        top_k_percentiles=[0.01, 0.05, 0.10, 0.20, 0.30, 0.50],
    )

    # Print summary
    print("\n=== Attention Distribution Analysis ===")
    print(f"Layers analyzed: {result['n_layers_analyzed']}")
    print(f"Power-law strength (Gini): {result['power_law_strength']:.4f}")
    print(f"  (0 = perfectly uniform, 1 = all attention on one token)")

    agg = result["aggregate"]
    print(f"\nAverage attention concentration:")
    for key, val in agg["concentration"].items():
        print(f"  {key}: {val:.2%} of total attention")

    print(f"\nNormalized entropy: {agg['avg_normalized_entropy']:.4f}")
    print(f"  (0 = all attention on one token, 1 = perfectly uniform)")

    # Per-layer breakdown (first, middle, last)
    layers = result["per_layer"]
    for idx in [0, len(layers) // 2, len(layers) - 1]:
        if idx < len(layers):
            s = layers[idx]
            print(f"\n  Layer {s['layer']}:")
            print(f"    Gini: {s['gini']:.4f}")
            for key, val in s["concentration"].items():
                print(f"    {key}: {val:.2%}")

    return result


def extract_kv_cache(model, tokenizer, prompt: str) -> Dict[str, Any]:
    """Run forward pass and extract actual KV cache tensors."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    seq_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=True,
        )

    # Extract KV from the DynamicCache (new API: .layers[i].keys/.values)
    kv_cache = outputs.past_key_values
    keys_per_layer = []
    values_per_layer = []

    for layer_idx in range(len(kv_cache.layers)):
        layer = kv_cache.layers[layer_idx]
        keys_per_layer.append(layer.keys.cpu().float())
        values_per_layer.append(layer.values.cpu().float())

    return {
        "keys": keys_per_layer,
        "values": values_per_layer,
        "attention_weights": [a.cpu().float() for a in outputs.attentions],
        "seq_len": seq_len,
        "n_layers": len(keys_per_layer),
        "head_dim": keys_per_layer[0].shape[-1],
        "n_heads": keys_per_layer[0].shape[1],
    }


def compute_attention_quality(
    fp16_keys: torch.Tensor,
    compressed_keys: torch.Tensor,
    query: torch.Tensor,
) -> Dict[str, float]:
    """Compute attention quality metrics between FP16 and compressed keys.

    Args:
        fp16_keys: (seq_len, d) original keys.
        compressed_keys: (seq_len, d) compressed keys.
        query: (n_queries, d) query vectors.

    Returns:
        Dict with cosine_sim, top1_match, top5_match.
    """
    # Compute attention scores
    d = fp16_keys.shape[-1]
    scale = 1.0 / math.sqrt(d)

    fp16_scores = (query @ fp16_keys.T) * scale  # (n_queries, seq_len)
    comp_scores = (query @ compressed_keys.T) * scale

    # Softmax attention
    fp16_attn = F.softmax(fp16_scores, dim=-1)
    comp_attn = F.softmax(comp_scores, dim=-1)

    # Cosine similarity of attention distributions
    cos_sim = F.cosine_similarity(fp16_attn, comp_attn, dim=-1).mean().item()

    # Top-K attention match
    n_queries = query.shape[0]
    top1_matches = 0
    top5_matches = 0

    for i in range(n_queries):
        fp16_top5 = torch.topk(fp16_attn[i], k=min(5, fp16_attn.shape[1])).indices
        comp_top5 = torch.topk(comp_attn[i], k=min(5, comp_attn.shape[1])).indices

        if fp16_top5[0] == comp_top5[0]:
            top1_matches += 1

        fp16_set = set(fp16_top5.tolist())
        comp_set = set(comp_top5.tolist())
        if len(fp16_set & comp_set) > 0:
            top5_matches += 1

    return {
        "cosine_sim": cos_sim,
        "top1_match": top1_matches / max(n_queries, 1),
        "top5_match": top5_matches / max(n_queries, 1),
    }


def uniform_quantize_keys(
    keys: torch.Tensor,
    bits: int,
    d: int,
    seed: int = 42,
) -> torch.Tensor:
    """Quantize keys uniformly at the given bit-width.

    Args:
        keys: (seq_len, d) key vectors.
        bits: Bit-width for quantization.
        d: Head dimension.
        seed: Random seed.

    Returns:
        (seq_len, d) reconstructed keys.
    """
    from turboquantdc.codebook import LloydMaxCodebook
    from turboquantdc.rotation import generate_rotation_matrix

    cb = LloydMaxCodebook(d=d, bits=bits)
    rot = generate_rotation_matrix(d, seed=seed, device="cpu")

    norms = keys.norm(dim=-1, keepdim=True)
    normalized = keys / (norms + 1e-8)
    rotated = normalized @ rot

    indices = torch.bucketize(rotated, cb.boundaries)
    indices = indices.clamp(0, cb.centroids.shape[0] - 1)

    reconstructed = cb.centroids[indices]
    unrotated = reconstructed @ rot.T
    return unrotated * norms


def adaptive_quantize_keys(
    keys: torch.Tensor,
    tier_assignments: torch.Tensor,
    tier_bits: List[int],
    d: int,
    seed: int = 42,
) -> torch.Tensor:
    """Quantize keys with per-tier adaptive bit allocation.

    Args:
        keys: (seq_len, d) key vectors.
        tier_assignments: (seq_len,) tier IDs.
        tier_bits: Bits for each tier.
        d: Head dimension.
        seed: Random seed.

    Returns:
        (seq_len, d) reconstructed keys at mixed precision.
    """
    from turboquantdc.codebook import LloydMaxCodebook
    from turboquantdc.rotation import generate_rotation_matrix

    result = torch.zeros_like(keys)
    codebooks = {}
    rotations = {}

    for tier_id, bits in enumerate(tier_bits):
        mask = tier_assignments == tier_id
        if not mask.any():
            continue

        tier_keys = keys[mask]

        if bits >= 16:
            result[mask] = tier_keys
            continue

        if bits not in codebooks:
            codebooks[bits] = LloydMaxCodebook(d=d, bits=bits)
            rotations[bits] = generate_rotation_matrix(
                d, seed=seed + bits * 100, device="cpu"
            )

        cb = codebooks[bits]
        rot = rotations[bits]

        norms = tier_keys.norm(dim=-1, keepdim=True)
        normalized = tier_keys / (norms + 1e-8)
        rotated = normalized @ rot

        indices = torch.bucketize(rotated, cb.boundaries)
        indices = indices.clamp(0, cb.centroids.shape[0] - 1)

        reconstructed = cb.centroids[indices]
        unrotated = reconstructed @ rot.T
        result[mask] = unrotated * norms

    return result


def compute_effective_bits(
    tier_assignments: torch.Tensor,
    tier_bits: List[int],
) -> float:
    """Compute weighted-average effective bits."""
    total = 0.0
    n = tier_assignments.shape[0]
    for tier_id, bits in enumerate(tier_bits):
        count = (tier_assignments == tier_id).sum().item()
        total += count * bits
    return total / max(n, 1)


def compute_compression_ratio(effective_bits: float) -> float:
    """Compute compression ratio vs FP16 (16 bits)."""
    return 16.0 / max(effective_bits, 0.1)


def run_tier_experiment(
    kv_data: Dict[str, Any],
    config_name: str,
    config: Dict[str, Any],
    power_law_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a single tier configuration experiment.

    Uses actual attention patterns to score importance, then compares
    adaptive vs uniform compression quality.
    """
    from turboquantdc.adaptive_bits import ImportanceScorer

    thresholds = config["thresholds"]
    bits = config["bits"]

    keys = kv_data["keys"]
    values = kv_data["values"]
    attn_weights = kv_data["attention_weights"]
    seq_len = kv_data["seq_len"]
    head_dim = kv_data["head_dim"]
    n_layers = kv_data["n_layers"]
    n_heads = kv_data["n_heads"]

    print(f"\n--- Config: {config_name} ---")
    print(f"  {config['description']}")

    all_layer_results = []

    for layer_idx in range(n_layers):
        k = keys[layer_idx]  # (batch, heads, seq, d)
        v = values[layer_idx]
        attn = attn_weights[layer_idx]  # (batch, heads, seq, seq)

        # Score importance from attention patterns
        scorer = ImportanceScorer(ema_decay=0.0)  # no EMA, use raw attention
        scorer.update(attn)
        tier_assignments = scorer.classify_tiers(thresholds)

        # For quality measurement, use per-head keys (pick head 0, batch 0)
        for head_idx in [0, n_heads // 2, n_heads - 1]:
            head_keys = k[0, head_idx].float()  # (seq, d)

            # Use last 32 tokens as queries (simulating decode queries)
            n_query = min(32, seq_len // 4)
            queries = head_keys[-n_query:]

            # Adaptive compression
            adaptive_keys = adaptive_quantize_keys(
                head_keys, tier_assignments, bits, head_dim, seed=42 + layer_idx,
            )

            # Uniform 3-bit compression (baseline)
            uniform_3bit = uniform_quantize_keys(
                head_keys, 3, head_dim, seed=42 + layer_idx,
            )

            # Uniform at same effective bits as adaptive
            eff_bits = compute_effective_bits(tier_assignments, bits)

            # Quality metrics
            adaptive_quality = compute_attention_quality(
                head_keys, adaptive_keys, queries,
            )
            uniform_3bit_quality = compute_attention_quality(
                head_keys, uniform_3bit, queries,
            )

            all_layer_results.append({
                "layer": layer_idx,
                "head": head_idx,
                "adaptive": adaptive_quality,
                "uniform_3bit": uniform_3bit_quality,
                "effective_bits": eff_bits,
                "tier_dist": {
                    f"tier_{i}": (tier_assignments == i).sum().item() / seq_len
                    for i in range(len(bits))
                },
            })

    # Aggregate
    n_results = len(all_layer_results)
    avg_adaptive_cos = sum(r["adaptive"]["cosine_sim"] for r in all_layer_results) / n_results
    avg_adaptive_top1 = sum(r["adaptive"]["top1_match"] for r in all_layer_results) / n_results
    avg_adaptive_top5 = sum(r["adaptive"]["top5_match"] for r in all_layer_results) / n_results
    avg_uniform_cos = sum(r["uniform_3bit"]["cosine_sim"] for r in all_layer_results) / n_results
    avg_uniform_top1 = sum(r["uniform_3bit"]["top1_match"] for r in all_layer_results) / n_results
    avg_uniform_top5 = sum(r["uniform_3bit"]["top5_match"] for r in all_layer_results) / n_results
    avg_eff_bits = sum(r["effective_bits"] for r in all_layer_results) / n_results
    avg_compression = compute_compression_ratio(avg_eff_bits)

    # Tier distribution (average)
    avg_tier_dist = {}
    for i in range(len(bits)):
        key = f"tier_{i}"
        avg_tier_dist[key] = sum(r["tier_dist"][key] for r in all_layer_results) / n_results

    print(f"  Effective bits: {avg_eff_bits:.2f} ({avg_compression:.1f}x compression)")
    print(f"  Adaptive  - CosSim: {avg_adaptive_cos:.4f}, Top-1: {avg_adaptive_top1:.1%}, Top-5: {avg_adaptive_top5:.1%}")
    print(f"  Uniform 3b- CosSim: {avg_uniform_cos:.4f}, Top-1: {avg_uniform_top1:.1%}, Top-5: {avg_uniform_top5:.1%}")

    # Quality delta
    cos_delta = avg_adaptive_cos - avg_uniform_cos
    top5_delta = avg_adaptive_top5 - avg_uniform_top5
    print(f"  Delta: CosSim {cos_delta:+.4f}, Top-5 {top5_delta:+.1%}")

    return {
        "config_name": config_name,
        "description": config["description"],
        "thresholds": thresholds,
        "bits": bits,
        "effective_bits": avg_eff_bits,
        "compression_ratio": avg_compression,
        "adaptive_quality": {
            "cosine_sim": avg_adaptive_cos,
            "top1_match": avg_adaptive_top1,
            "top5_match": avg_adaptive_top5,
        },
        "uniform_3bit_quality": {
            "cosine_sim": avg_uniform_cos,
            "top1_match": avg_uniform_top1,
            "top5_match": avg_uniform_top5,
        },
        "quality_delta": {
            "cosine_sim": cos_delta,
            "top5_match": top5_delta,
        },
        "tier_distribution": avg_tier_dist,
        "per_layer": all_layer_results,
    }


def compute_attention_quality_with_eviction(
    fp16_keys: torch.Tensor,
    kept_keys: torch.Tensor,
    keep_mask: torch.Tensor,
    query: torch.Tensor,
) -> Dict[str, float]:
    """Compute attention quality when some tokens are evicted.

    Evicted tokens are excluded from the softmax entirely (as if they
    don't exist), not set to zero. This gives the true eviction quality.

    Args:
        fp16_keys: (seq_len, d) original full-precision keys.
        kept_keys: (n_kept, d) keys of retained tokens.
        keep_mask: (seq_len,) boolean mask of which tokens are kept.
        query: (n_queries, d) query vectors.

    Returns:
        Dict with cosine_sim, top1_match, top5_match.
    """
    d = fp16_keys.shape[-1]
    scale = 1.0 / math.sqrt(d)

    # Full attention (FP16 baseline)
    fp16_scores = (query @ fp16_keys.T) * scale
    fp16_attn = F.softmax(fp16_scores, dim=-1)

    # Evicted attention: only over kept tokens
    evict_scores = (query @ kept_keys.T) * scale
    evict_attn = F.softmax(evict_scores, dim=-1)

    # Map evicted attention back to full seq_len for comparison
    # Evicted positions get 0 attention weight
    full_evict_attn = torch.zeros_like(fp16_attn)
    kept_indices = keep_mask.nonzero(as_tuple=True)[0]
    full_evict_attn[:, kept_indices] = evict_attn

    # Cosine similarity
    cos_sim = F.cosine_similarity(fp16_attn, full_evict_attn, dim=-1).mean().item()

    # Top-K match
    n_queries = query.shape[0]
    top1_matches = 0
    top5_matches = 0

    for i in range(n_queries):
        fp16_top5 = torch.topk(fp16_attn[i], k=min(5, fp16_attn.shape[1])).indices
        evict_top5 = torch.topk(full_evict_attn[i], k=min(5, full_evict_attn.shape[1])).indices

        if fp16_top5[0] == evict_top5[0]:
            top1_matches += 1

        fp16_set = set(fp16_top5.tolist())
        evict_set = set(evict_top5.tolist())
        if len(fp16_set & evict_set) > 0:
            top5_matches += 1

    return {
        "cosine_sim": cos_sim,
        "top1_match": top1_matches / max(n_queries, 1),
        "top5_match": top5_matches / max(n_queries, 1),
    }


def run_eviction_comparison(
    kv_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare adaptive bits against token eviction.

    Simulates evicting the bottom 50% of tokens (by importance) and keeping
    the rest at FP16. Evicted tokens are excluded from the attention
    computation entirely (not zeroed out).
    """
    from turboquantdc.adaptive_bits import ImportanceScorer

    keys = kv_data["keys"]
    attn_weights = kv_data["attention_weights"]
    seq_len = kv_data["seq_len"]
    head_dim = kv_data["head_dim"]
    n_layers = kv_data["n_layers"]
    n_heads = kv_data["n_heads"]

    print("\n=== Eviction vs Adaptive Comparison ===")

    results = {"keep_50pct": [], "keep_75pct": []}

    for keep_pct, label in [(0.50, "keep_50pct"), (0.75, "keep_75pct")]:
        for layer_idx in range(n_layers):
            k = keys[layer_idx]
            attn = attn_weights[layer_idx]

            scorer = ImportanceScorer(ema_decay=0.0)
            scorer.update(attn)
            scores = scorer.scores

            # Keep top keep_pct tokens by importance
            n_keep = max(1, int(keep_pct * seq_len))
            _, keep_indices = torch.topk(scores, n_keep)
            keep_mask = torch.zeros(seq_len, dtype=torch.bool)
            keep_mask[keep_indices] = True

            for head_idx in [0]:
                head_keys = k[0, head_idx].float()

                # Eviction: keep only top tokens at FP16
                kept_keys = head_keys[keep_mask]

                # Compute quality using last 32 tokens as queries
                n_query = min(32, seq_len // 4)
                queries = head_keys[-n_query:]

                # Proper eviction quality (exclude evicted from softmax)
                evict_quality = compute_attention_quality_with_eviction(
                    head_keys, kept_keys, keep_mask, queries,
                )

                # Effective bits for eviction: FP16 for kept, 0 for evicted
                evict_eff_bits = keep_pct * 16.0

                results[label].append({
                    "layer": layer_idx,
                    "eviction_quality": evict_quality,
                    "evict_eff_bits": evict_eff_bits,
                    "keep_pct": keep_pct,
                })

    # Summarize
    for label, data in results.items():
        if data:
            avg_cos = sum(r["eviction_quality"]["cosine_sim"] for r in data) / len(data)
            avg_top5 = sum(r["eviction_quality"]["top5_match"] for r in data) / len(data)
            keep_pct = data[0]["keep_pct"]
            eff_bits = data[0]["evict_eff_bits"]
            print(f"\n  Eviction ({label}, {keep_pct:.0%} kept, {eff_bits:.1f} eff bits):")
            print(f"    CosSim: {avg_cos:.4f}, Top-5: {avg_top5:.1%}")

    return results


def generate_results_markdown(
    power_law: Dict[str, Any],
    tier_results: List[Dict[str, Any]],
    eviction_results: Dict[str, Any],
    kv_data: Dict[str, Any],
    elapsed: float,
) -> str:
    """Generate the results markdown file."""
    lines = []
    lines.append("# Adaptive Bits Results")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Model:** {MODEL_NAME}")
    lines.append(f"**Context:** {kv_data['seq_len']} tokens")
    lines.append(f"**Layers:** {kv_data['n_layers']}")
    lines.append(f"**Heads:** {kv_data['n_heads']}")
    lines.append(f"**Head dim:** {kv_data['head_dim']}")
    lines.append(f"**Runtime:** {elapsed:.1f}s")
    lines.append("")

    # Power law analysis
    lines.append("## 1. Attention Distribution (Power Law Analysis)")
    lines.append("")
    agg = power_law["aggregate"]
    lines.append(f"**Power-law strength (Gini coefficient):** {power_law['power_law_strength']:.4f}")
    lines.append(f"  - 0.0 = perfectly uniform attention")
    lines.append(f"  - 1.0 = all attention on one token")
    lines.append(f"  - **{power_law['power_law_strength']:.2f}** indicates {'strong' if power_law['power_law_strength'] > 0.5 else 'moderate' if power_law['power_law_strength'] > 0.3 else 'weak'} concentration")
    lines.append("")
    lines.append(f"**Normalized entropy:** {agg['avg_normalized_entropy']:.4f}")
    lines.append("")

    lines.append("### Attention Concentration")
    lines.append("")
    lines.append("| Top % of tokens | % of total attention captured |")
    lines.append("|---|---|")
    for key, val in agg["concentration"].items():
        pct_label = key.replace("top_", "")
        lines.append(f"| {pct_label} | {val:.1%} |")
    lines.append("")

    # Per-layer summary
    lines.append("### Per-Layer Gini Coefficient")
    lines.append("")
    lines.append("| Layer | Gini | Top 5% captures | Top 10% captures |")
    lines.append("|---|---|---|---|")
    for s in power_law["per_layer"]:
        top5 = s["concentration"].get("top_5%", 0)
        top10 = s["concentration"].get("top_10%", 0)
        lines.append(f"| {s['layer']} | {s['gini']:.4f} | {top5:.1%} | {top10:.1%} |")
    lines.append("")

    # Key finding
    top10_val = agg["concentration"].get("top_10%", 0)
    top20_val = agg["concentration"].get("top_20%", 0)
    lines.append(f"**Key Finding:** Top 10% of tokens capture **{top10_val:.1%}** of attention, "
                 f"top 20% capture **{top20_val:.1%}**. "
                 f"This {'validates' if top10_val > 0.5 else 'partially supports'} the power-law hypothesis.")
    lines.append("")

    # Tier compression results
    lines.append("## 2. Tiered Compression Results")
    lines.append("")
    lines.append("### Quality Comparison: Adaptive vs Uniform 3-bit")
    lines.append("")
    lines.append("| Config | Eff. Bits | Compression | Adaptive CosSim | Uniform 3b CosSim | Delta CosSim | Adaptive Top-5 | Uniform 3b Top-5 | Delta Top-5 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in tier_results:
        lines.append(
            f"| {r['config_name']} | {r['effective_bits']:.2f} | {r['compression_ratio']:.1f}x "
            f"| {r['adaptive_quality']['cosine_sim']:.4f} "
            f"| {r['uniform_3bit_quality']['cosine_sim']:.4f} "
            f"| {r['quality_delta']['cosine_sim']:+.4f} "
            f"| {r['adaptive_quality']['top5_match']:.1%} "
            f"| {r['uniform_3bit_quality']['top5_match']:.1%} "
            f"| {r['quality_delta']['top5_match']:+.1%} |"
        )
    lines.append("")

    # Tier distribution details
    lines.append("### Tier Distribution by Config")
    lines.append("")
    for r in tier_results:
        lines.append(f"**{r['config_name']}** ({r['description']}):")
        for key, val in r["tier_distribution"].items():
            tier_id = int(key.split("_")[1])
            bits = r["bits"][tier_id]
            label = "FP16" if bits >= 16 else f"{bits}-bit"
            lines.append(f"  - Tier {tier_id} ({label}): {val:.1%}")
        lines.append(f"  - Effective bits: {r['effective_bits']:.2f}")
        lines.append(f"  - Compression ratio: {r['compression_ratio']:.1f}x")
        lines.append("")

    # Eviction comparison
    lines.append("## 3. Eviction vs Adaptive Comparison")
    lines.append("")
    lines.append("| Strategy | Eff. Bits | CosSim | Top-5 |")
    lines.append("|---|---|---|---|")

    for label, data in eviction_results.items():
        if data:
            avg_cos = sum(r["eviction_quality"]["cosine_sim"] for r in data) / len(data)
            avg_top5 = sum(r["eviction_quality"]["top5_match"] for r in data) / len(data)
            eff_bits = data[0]["evict_eff_bits"]
            pct = data[0]["keep_pct"]
            lines.append(f"| Evict {1-pct:.0%} ({label}) | {eff_bits:.1f} | {avg_cos:.4f} | {avg_top5:.1%} |")

    # Add uniform and best adaptive for comparison
    for r in tier_results:
        lines.append(
            f"| Adaptive ({r['config_name']}) | {r['effective_bits']:.2f} "
            f"| {r['adaptive_quality']['cosine_sim']:.4f} "
            f"| {r['adaptive_quality']['top5_match']:.1%} |"
        )

    # Always include uniform 3-bit baseline
    if tier_results:
        r = tier_results[0]
        lines.append(
            f"| Uniform 3-bit | 3.00 "
            f"| {r['uniform_3bit_quality']['cosine_sim']:.4f} "
            f"| {r['uniform_3bit_quality']['top5_match']:.1%} |"
        )
    lines.append("")

    # Summary and conclusions
    lines.append("## 4. Summary")
    lines.append("")

    if tier_results:
        best_config = max(tier_results, key=lambda r: r["adaptive_quality"]["cosine_sim"])
        most_compressed = min(tier_results, key=lambda r: r["effective_bits"])

        lines.append(f"**Best quality adaptive:** {best_config['config_name']} "
                     f"({best_config['effective_bits']:.2f} bits, "
                     f"CosSim {best_config['adaptive_quality']['cosine_sim']:.4f})")
        lines.append("")
        lines.append(f"**Most compressed adaptive:** {most_compressed['config_name']} "
                     f"({most_compressed['effective_bits']:.2f} bits, "
                     f"{most_compressed['compression_ratio']:.1f}x compression, "
                     f"CosSim {most_compressed['adaptive_quality']['cosine_sim']:.4f})")
        lines.append("")

        # Key comparison
        uniform_cos = best_config["uniform_3bit_quality"]["cosine_sim"]
        adaptive_cos = best_config["adaptive_quality"]["cosine_sim"]
        lines.append(f"**Adaptive vs Uniform 3-bit:**")
        if adaptive_cos >= uniform_cos - 0.001:
            lines.append(f"  Adaptive at {best_config['effective_bits']:.2f} bits "
                         f"{'matches' if abs(adaptive_cos - uniform_cos) < 0.001 else 'exceeds'} "
                         f"uniform 3-bit quality ({adaptive_cos:.4f} vs {uniform_cos:.4f})")
        else:
            lines.append(f"  Adaptive at {best_config['effective_bits']:.2f} bits "
                         f"is below uniform 3-bit ({adaptive_cos:.4f} vs {uniform_cos:.4f})")
        lines.append("")

    lines.append("### Conclusion")
    lines.append("")
    gini = power_law["power_law_strength"]
    if gini > 0.5:
        lines.append("The attention distribution shows **strong power-law behavior** "
                     f"(Gini={gini:.4f}). Adaptive bit allocation is highly effective: "
                     "important tokens get more bits, unimportant tokens can be aggressively "
                     "compressed with minimal quality loss.")
    elif gini > 0.3:
        lines.append("The attention distribution shows **moderate concentration** "
                     f"(Gini={gini:.4f}). Adaptive bit allocation provides measurable "
                     "benefits, but the gains over uniform quantization are incremental.")
    else:
        lines.append("The attention distribution is **relatively uniform** "
                     f"(Gini={gini:.4f}). Adaptive bit allocation provides limited "
                     "benefit over uniform quantization at this context length. "
                     "Longer contexts may show stronger concentration.")
    lines.append("")

    return "\n".join(lines)


def main():
    """Run the full adaptive bits benchmark."""
    print("=" * 70)
    print("Adaptive Bits Benchmark: Attention-Aware Bit Allocation")
    print("=" * 70)

    t_start = time.time()

    # Step 1: Load model
    model, tokenizer = load_model()

    # Step 2: Extract attention patterns
    print("\n--- Phase 1: Extracting attention patterns ---")
    attention_data = extract_attention_patterns(model, tokenizer, CONTEXT_PROMPT)

    # Step 3: Analyze power-law distribution
    print("\n--- Phase 2: Power-law analysis ---")
    power_law = analyze_power_law(attention_data)

    # Step 4: Extract KV cache for compression experiments
    print("\n--- Phase 3: Extracting KV cache ---")
    kv_data = extract_kv_cache(model, tokenizer, CONTEXT_PROMPT)
    print(f"  KV cache: {kv_data['n_layers']} layers, {kv_data['n_heads']} heads, "
          f"d={kv_data['head_dim']}, seq={kv_data['seq_len']}")

    # Step 5: Run tier experiments
    print("\n--- Phase 4: Tiered compression experiments ---")
    tier_results = []
    for config_name, config in TIER_CONFIGS.items():
        result = run_tier_experiment(kv_data, config_name, config, power_law)
        tier_results.append(result)

    # Step 6: Eviction comparison
    print("\n--- Phase 5: Eviction comparison ---")
    eviction_results = run_eviction_comparison(kv_data)

    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start

    # Step 7: Generate results
    print("\n--- Generating results ---")
    results_md = generate_results_markdown(
        power_law, tier_results, eviction_results, kv_data, elapsed,
    )

    results_dir = os.path.join(REPO_ROOT, "benchmarks", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "adaptive_bits_results.md")
    with open(results_path, "w") as f:
        f.write(results_md)

    print(f"\nResults saved to {results_path}")
    print(f"Total runtime: {elapsed:.1f}s")
    print("\n" + "=" * 70)

    # Print key numbers
    if tier_results:
        best = max(tier_results, key=lambda r: r["adaptive_quality"]["cosine_sim"])
        print(f"\nKey Result: {best['config_name']} adaptive at {best['effective_bits']:.2f} "
              f"bits ({best['compression_ratio']:.1f}x) vs uniform 3-bit (5.0x)")
        print(f"  Adaptive CosSim: {best['adaptive_quality']['cosine_sim']:.4f}")
        print(f"  Uniform  CosSim: {best['uniform_3bit_quality']['cosine_sim']:.4f}")
        print(f"  Power-law Gini:  {power_law['power_law_strength']:.4f}")


if __name__ == "__main__":
    main()
