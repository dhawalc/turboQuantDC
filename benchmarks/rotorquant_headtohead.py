"""
Head-to-head benchmark: TurboQuantDC vs RotorQuant
====================================================

Loads Qwen2.5-3B-Instruct, extracts real KV caches (500+ tokens),
and benchmarks ALL methods on the SAME vectors at 2/3/4-bit.

Methods tested:
  RotorQuant:
    - PlanarQuant (2D Givens) -- their fastest
    - IsoQuant-Full (4D quaternion SO(4)) -- their recommended
    - IsoQuant-Fast (4D quaternion, isoclinic SO(3) subgroup)
    - Standard TurboQuant (QR rotation) -- their baseline

  TurboQuantDC (ours):
    - PolarQuant WHT (our baseline, same paper as theirs)
    - ResidualQuant WHT (our Stage 2 improvement)
    - ResidualQuant WHT + mean-removal (our production)
    - PCA + ResidualQuant + mean-removal (our full stack)

Metrics:
    - Cosine similarity to FP16
    - Top-1 attention match
    - Top-5 attention match
    - MSE (normalized)
    - Quantize latency (ms per 1000 vectors)
    - Dequantize latency (ms per 1000 vectors)
"""

import sys
import os
import time
import math
import json
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn.functional as F

# -- Ensure both packages are importable --
sys.path.insert(0, "/home/dhawal/turboQuantDC")
sys.path.insert(0, "/tmp/rotorquant")

# RotorQuant imports
from turboquant.planarquant import PlanarQuantMSE, PlanarQuantProd
from turboquant.isoquant import IsoQuantMSE, IsoQuantProd
from turboquant.turboquant import (
    TurboQuantMSE as RQ_TurboQuantMSE,
    TurboQuantProd as RQ_TurboQuantProd,
)

# TurboQuantDC imports
from turboquantdc.polarquant import PolarQuant
from turboquantdc.residual_quant import ResidualQuantEstimator
from turboquantdc.learned_rotation import PCARotatedQuantizer, compute_pca_rotation
from turboquantdc.attention_optimal import (
    compute_attention_scores,
    attention_metrics,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"
SEED = 42
N_WARMUP = 3
N_TIMING = 20
HEAD_DIM = 128  # Qwen2.5 head dim


# ============================================================================
# Step 1: Extract real KV caches from Qwen2.5-3B-Instruct
# ============================================================================

def load_model_and_extract_kv():
    """Load Qwen2.5-3B-Instruct (BnB 4-bit) and extract real KV caches."""
    print("=" * 70)
    print("Loading Qwen2.5-3B-Instruct (BnB 4-bit)...")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
    )
    model.eval()

    # Long prompt to get 500+ token KV cache -- use generate() with max_new_tokens
    prompt = """You are a world-class computer scientist giving a comprehensive lecture on
the history of computing, starting from Charles Babbage's Analytical Engine through
Alan Turing's theoretical foundations, the development of ENIAC, the transistor revolution,
the birth of the internet at ARPANET, the rise of personal computing with Apple and
IBM, the open source movement with Linux, the mobile revolution with iPhone, cloud
computing with AWS, and finally the current AI revolution with large language models
like GPT-4 and Claude. Cover the key innovations, the people behind them, and the
societal impact of each era. Be thorough and detailed, covering at least 20 major
milestones in computing history. Discuss the technical details of each innovation,
how it built on previous work, and what made it revolutionary for its time. Include
lesser-known figures who made critical contributions alongside the famous names.
Also discuss the evolution of programming languages from Assembly to Rust, the
development of databases from hierarchical to relational to NoSQL, the evolution of
networking from dial-up to fiber optics and 5G, and the progression of AI from
expert systems through machine learning to deep learning and transformers."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs['input_ids'].shape[1]
    print(f"Prompt tokens: {prompt_len}")

    # Generate tokens to get 500+ total in the KV cache
    target_total = max(512, prompt_len + 1)
    gen_tokens = target_total - prompt_len
    print(f"Generating {gen_tokens} tokens to reach {target_total} total...")

    with torch.no_grad():
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=gen_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True,
        )

    # Run a single forward pass on the full sequence to get the complete KV cache
    full_seq = gen_outputs.sequences  # (1, total_len)
    total_len = full_seq.shape[1]
    print(f"Total sequence length: {total_len}")

    with torch.no_grad():
        outputs = model(
            full_seq,
            use_cache=True,
        )

    past_kv = outputs.past_key_values
    # DynamicCache API: past_kv.layers is a list of DynamicLayer objects
    # Each layer has .keys and .values tensors of shape (B, H, S, D)
    n_layers = len(past_kv.layers)
    print(f"Extracted KV cache from {n_layers} layers")

    # Extract keys from several layers for diversity
    all_keys = []
    layers_to_use = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    for layer_idx in layers_to_use:
        k = past_kv.layers[layer_idx].keys  # (batch, n_heads, seq_len, head_dim)
        # Flatten across batch and heads
        B, H, S, D = k.shape
        keys_flat = k.float().reshape(-1, D)  # (B*H*S, D)
        all_keys.append(keys_flat)
        print(f"  Layer {layer_idx}: {k.shape} -> {keys_flat.shape[0]} vectors, head_dim={D}")

    keys = torch.cat(all_keys, dim=0).to(DEVICE)
    print(f"\nTotal key vectors: {keys.shape[0]}, dim={keys.shape[1]}")

    # Extract query vectors: sample multiple positions from each layer/head
    # Since Qwen2.5-3B uses GQA with only 2 KV heads, we need more queries
    queries_list = []
    for layer_idx in layers_to_use:
        k_layer = past_kv.layers[layer_idx].keys  # (B, H, S, D)
        B, H, S, D = k_layer.shape
        # Sample up to 20 evenly-spaced positions per layer
        n_sample = min(20, S)
        positions = torch.linspace(S // 4, S - 1, n_sample).long()
        q_sampled = k_layer[:, :, positions, :].float()  # (B, H, n_sample, D)
        queries_list.append(q_sampled.reshape(-1, HEAD_DIM))
    queries = torch.cat(queries_list, dim=0).to(DEVICE)
    print(f"Total query vectors: {queries.shape[0]}")

    # Clean up model to free VRAM
    del model, outputs, past_kv
    torch.cuda.empty_cache()

    return keys, queries


# ============================================================================
# Step 2: Define method wrappers with uniform interface
# ============================================================================

class MethodWrapper:
    """Uniform interface: quantize -> dequantize -> compute metrics."""

    def __init__(self, name, family):
        self.name = name
        self.family = family  # "rotorquant" or "turboquantdc"

    def quantize_dequantize(self, keys, bits):
        """Returns reconstructed keys (same shape as input)."""
        raise NotImplementedError

    def benchmark_latency(self, keys, bits, n_warmup=N_WARMUP, n_iter=N_TIMING):
        """Returns (quant_ms, dequant_ms) per 1000 vectors."""
        raise NotImplementedError


# -- RotorQuant Methods --

class PlanarQuantMethod(MethodWrapper):
    def __init__(self):
        super().__init__("PlanarQuant (2D Givens)", "rotorquant")

    def quantize_dequantize(self, keys, bits):
        d = keys.shape[-1]
        pq = PlanarQuantMSE(d, bits, seed=SEED, device=str(keys.device))
        norms = keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        keys_unit = keys / norms
        _, indices = pq.quantize(keys_unit)
        recon = pq.dequantize(indices) * norms
        return recon

    def benchmark_latency(self, keys, bits, n_warmup=N_WARMUP, n_iter=N_TIMING):
        d = keys.shape[-1]
        n = keys.shape[0]
        pq = PlanarQuantMSE(d, bits, seed=SEED, device=str(keys.device))
        norms = keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        keys_unit = keys / norms

        # Warmup
        for _ in range(n_warmup):
            _, idx = pq.quantize(keys_unit)
            _ = pq.dequantize(idx)
        torch.cuda.synchronize() if DEVICE == "cuda" else None

        # Time quantize
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _, idx = pq.quantize(keys_unit)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        q_ms = (time.perf_counter() - t0) / n_iter * 1000 / n * 1000

        # Time dequantize
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = pq.dequantize(idx)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        dq_ms = (time.perf_counter() - t0) / n_iter * 1000 / n * 1000

        return q_ms, dq_ms


class IsoQuantMethod(MethodWrapper):
    def __init__(self, mode="full"):
        name = f"IsoQuant-{'Full' if mode == 'full' else 'Fast'} (4D Quat)"
        super().__init__(name, "rotorquant")
        self.mode = mode

    def quantize_dequantize(self, keys, bits):
        d = keys.shape[-1]
        iq = IsoQuantMSE(d, bits, seed=SEED, mode=self.mode, device=str(keys.device))
        norms = keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        keys_unit = keys / norms
        _, indices = iq.quantize(keys_unit)
        recon = iq.dequantize(indices) * norms
        return recon

    def benchmark_latency(self, keys, bits, n_warmup=N_WARMUP, n_iter=N_TIMING):
        d = keys.shape[-1]
        n = keys.shape[0]
        iq = IsoQuantMSE(d, bits, seed=SEED, mode=self.mode, device=str(keys.device))
        norms = keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        keys_unit = keys / norms

        for _ in range(n_warmup):
            _, idx = iq.quantize(keys_unit)
            _ = iq.dequantize(idx)
        torch.cuda.synchronize() if DEVICE == "cuda" else None

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _, idx = iq.quantize(keys_unit)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        q_ms = (time.perf_counter() - t0) / n_iter * 1000 / n * 1000

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = iq.dequantize(idx)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        dq_ms = (time.perf_counter() - t0) / n_iter * 1000 / n * 1000

        return q_ms, dq_ms


class RQ_TurboQuantMethod(MethodWrapper):
    """RotorQuant's standard TurboQuant (QR rotation baseline).

    Note: Their TurboQuantMSE doesn't do norm separation internally,
    but their CompressorV2 does. We normalize like CompressorV2 for
    a fair comparison.
    """
    def __init__(self):
        super().__init__("TurboQuant-QR (RotorQuant)", "rotorquant")

    def quantize_dequantize(self, keys, bits):
        d = keys.shape[-1]
        tq = RQ_TurboQuantMSE(d, bits, seed=SEED, device=str(keys.device))
        norms = keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        keys_unit = keys / norms
        indices = tq.quantize(keys_unit)
        recon = tq.dequantize(indices) * norms
        return recon

    def benchmark_latency(self, keys, bits, n_warmup=N_WARMUP, n_iter=N_TIMING):
        d = keys.shape[-1]
        n = keys.shape[0]
        tq = RQ_TurboQuantMSE(d, bits, seed=SEED, device=str(keys.device))
        norms = keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        keys_unit = keys / norms

        for _ in range(n_warmup):
            idx = tq.quantize(keys_unit)
            _ = tq.dequantize(idx)
        torch.cuda.synchronize() if DEVICE == "cuda" else None

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n_iter):
            idx = tq.quantize(keys_unit)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        q_ms = (time.perf_counter() - t0) / n_iter * 1000 / n * 1000

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = tq.dequantize(idx)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        dq_ms = (time.perf_counter() - t0) / n_iter * 1000 / n * 1000

        return q_ms, dq_ms


# -- TurboQuantDC Methods (Ours) --

class PolarQuantWHTMethod(MethodWrapper):
    """Our WHT baseline (same paper, our implementation)."""
    def __init__(self):
        super().__init__("PolarQuant-WHT (Ours)", "turboquantdc")

    def quantize_dequantize(self, keys, bits):
        d = keys.shape[-1]
        pq = PolarQuant(d, bits, seed=SEED, device=str(keys.device), rotation_type="wht")
        norms = keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        keys_unit = keys / norms
        x_hat, _ = pq(keys_unit)
        return x_hat * norms

    def benchmark_latency(self, keys, bits, n_warmup=N_WARMUP, n_iter=N_TIMING):
        d = keys.shape[-1]
        n = keys.shape[0]
        pq = PolarQuant(d, bits, seed=SEED, device=str(keys.device), rotation_type="wht")
        norms = keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        keys_unit = keys / norms

        for _ in range(n_warmup):
            idx = pq.quantize(keys_unit)
            _ = pq.dequantize(idx)
        torch.cuda.synchronize() if DEVICE == "cuda" else None

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n_iter):
            idx = pq.quantize(keys_unit)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        q_ms = (time.perf_counter() - t0) / n_iter * 1000 / n * 1000

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = pq.dequantize(idx)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        dq_ms = (time.perf_counter() - t0) / n_iter * 1000 / n * 1000

        return q_ms, dq_ms


class ResidualQuantWHTMethod(MethodWrapper):
    """Our Stage 2: ResidualQuant (no mean removal)."""
    def __init__(self):
        super().__init__("ResidualQuant-WHT (Ours)", "turboquantdc")

    def quantize_dequantize(self, keys, bits):
        d = keys.shape[-1]
        rq = ResidualQuantEstimator(d, bits, seed=SEED, device=str(keys.device),
                                      center_before_quantize=False)
        comp = rq.quantize(keys)
        return rq.dequantize(comp)

    def benchmark_latency(self, keys, bits, n_warmup=N_WARMUP, n_iter=N_TIMING):
        d = keys.shape[-1]
        n = keys.shape[0]
        rq = ResidualQuantEstimator(d, bits, seed=SEED, device=str(keys.device),
                                      center_before_quantize=False)

        for _ in range(n_warmup):
            comp = rq.quantize(keys)
            _ = rq.dequantize(comp)
        torch.cuda.synchronize() if DEVICE == "cuda" else None

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n_iter):
            comp = rq.quantize(keys)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        q_ms = (time.perf_counter() - t0) / n_iter * 1000 / n * 1000

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = rq.dequantize(comp)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        dq_ms = (time.perf_counter() - t0) / n_iter * 1000 / n * 1000

        return q_ms, dq_ms


class ResidualQuantMeanMethod(MethodWrapper):
    """Our production config: ResidualQuant + mean-removal."""
    def __init__(self):
        super().__init__("ResidualQuant+Mean (Ours)", "turboquantdc")

    def quantize_dequantize(self, keys, bits):
        d = keys.shape[-1]
        rq = ResidualQuantEstimator(d, bits, seed=SEED, device=str(keys.device),
                                      center_before_quantize=True)
        comp = rq.quantize(keys)
        return rq.dequantize(comp)

    def benchmark_latency(self, keys, bits, n_warmup=N_WARMUP, n_iter=N_TIMING):
        d = keys.shape[-1]
        n = keys.shape[0]
        rq = ResidualQuantEstimator(d, bits, seed=SEED, device=str(keys.device),
                                      center_before_quantize=True)

        for _ in range(n_warmup):
            comp = rq.quantize(keys)
            _ = rq.dequantize(comp)
        torch.cuda.synchronize() if DEVICE == "cuda" else None

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n_iter):
            comp = rq.quantize(keys)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        q_ms = (time.perf_counter() - t0) / n_iter * 1000 / n * 1000

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = rq.dequantize(comp)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        dq_ms = (time.perf_counter() - t0) / n_iter * 1000 / n * 1000

        return q_ms, dq_ms


class PCAFullStackMethod(MethodWrapper):
    """Our full stack: PCA rotation (data-adapted) + residual signs + mean-removal.

    The PCA rotation whitens the data so each coordinate has variance ~1/d,
    matching the Lloyd-Max codebook's Gaussian assumption. This replaces
    the random WHT rotation with a data-optimal one.

    Pipeline:
        1. Calibrate PCA rotation on a subset of keys
        2. Mean-remove keys (softmax shift-invariance)
        3. Normalize to unit vectors, store norms
        4. Apply PCA rotation (rotate + whiten)
        5. Lloyd-Max quantize in whitened space (same codebook as WHT)
        6. Compute residual signs in whitened space
        7. Reconstruct: centroid + scale*signs -> un-whiten -> un-rotate -> rescale + mean
    """
    def __init__(self):
        super().__init__("PCA+RQ+Mean (Ours, Full)", "turboquantdc")

    def _build(self, keys, bits):
        """Build PCA quantizer from calibration data.

        Critical: PCA must be fit on UNIT vectors (post-normalization),
        not raw vectors, because PCARotatedQuantizer's whitening assumes
        the data is unit-norm. The eigenvalues of unit vectors are ~1/d,
        which matches the Lloyd-Max codebook's N(0, 1/d) assumption.
        """
        d = keys.shape[-1]
        device = str(keys.device)

        # Mean-remove first
        mean_k = keys.mean(dim=0, keepdim=True)
        keys_centered = keys - mean_k

        # Normalize to unit vectors
        norms_calib = keys_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        keys_unit_calib = keys_centered / norms_calib

        # Calibrate PCA on unit vectors (use up to 512)
        n_calib = min(512, keys.shape[0])
        calib = keys_unit_calib[:n_calib].float()
        pca_data = compute_pca_rotation(calib, center=False)

        # Build PCA quantizer with mse_bits
        mse_bits = max(bits - 1, 1)
        pca_q = PCARotatedQuantizer(d, mse_bits, pca_data,
                                      adaptive_bits=False, device=device)
        return pca_q, mean_k

    def quantize_dequantize(self, keys, bits):
        pca_q, mean_k = self._build(keys, bits)

        keys_centered = keys - mean_k
        norms = keys_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        keys_unit = keys_centered / norms

        # PCA rotate + whiten -> quantize in whitened space
        y = pca_q.rotate(keys_unit)  # whitened PCA coords ~N(0, 1/d)
        indices = pca_q.codebook.quantize(y)
        y_mse = pca_q.centroids[indices]

        # Residual in whitened space
        residual = y - y_mse
        signs = (residual >= 0).float() * 2.0 - 1.0
        scale = residual.abs().mean(dim=-1, keepdim=True)

        # Reconstruct in whitened space with residual correction
        y_corrected = y_mse + scale * signs

        # Unwhiten + un-rotate back to original space
        x_hat = pca_q.unrotate(y_corrected)

        # Rescale and add mean back
        return x_hat * norms + mean_k

    def benchmark_latency(self, keys, bits, n_warmup=N_WARMUP, n_iter=N_TIMING):
        n = keys.shape[0]
        pca_q, mean_k = self._build(keys, bits)
        keys_centered = keys - mean_k
        norms = keys_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        keys_unit = keys_centered / norms

        # Warmup
        for _ in range(n_warmup):
            y = pca_q.rotate(keys_unit)
            indices = pca_q.codebook.quantize(y)
            y_mse = pca_q.centroids[indices]
            r = y - y_mse
            signs = (r >= 0).float() * 2.0 - 1.0
            scale = r.abs().mean(dim=-1, keepdim=True)
            y_c = y_mse + scale * signs
            _ = pca_q.unrotate(y_c) * norms + mean_k
        torch.cuda.synchronize() if DEVICE == "cuda" else None

        # Time quantize
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n_iter):
            y = pca_q.rotate(keys_unit)
            indices = pca_q.codebook.quantize(y)
            y_mse = pca_q.centroids[indices]
            r = y - y_mse
            signs = (r >= 0).float() * 2.0 - 1.0
            scale = r.abs().mean(dim=-1, keepdim=True)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        q_ms = (time.perf_counter() - t0) / n_iter * 1000 / n * 1000

        # Time dequantize
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n_iter):
            y_c = y_mse + scale * signs
            _ = pca_q.unrotate(y_c) * norms + mean_k
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        dq_ms = (time.perf_counter() - t0) / n_iter * 1000 / n * 1000

        return q_ms, dq_ms


# ============================================================================
# Step 3: Compute metrics
# ============================================================================

def compute_all_metrics(keys_orig, keys_recon, queries, scale=None):
    """Compute cosine sim, MSE, and attention metrics."""
    d = keys_orig.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # Vector-level metrics
    cos_sim = F.cosine_similarity(keys_orig, keys_recon, dim=-1).mean().item()
    mse = ((keys_orig - keys_recon) ** 2).mean().item()
    nmse = mse / (keys_orig ** 2).mean().item()  # normalized MSE

    # Attention-level metrics: sample N_Q queries, use all keys as context
    # Use a subset for tractability
    n_keys_for_attn = min(keys_orig.shape[0], 2048)
    n_queries_for_attn = min(queries.shape[0], 64)

    k_sub = keys_orig[:n_keys_for_attn]
    k_recon_sub = keys_recon[:n_keys_for_attn]
    q_sub = queries[:n_queries_for_attn]

    attn_true = compute_attention_scores(q_sub, k_sub)
    attn_quant = compute_attention_scores(q_sub, k_recon_sub)

    metrics = attention_metrics(attn_true, attn_quant)
    metrics["vector_cosine_sim"] = cos_sim
    metrics["mse"] = mse
    metrics["nmse"] = nmse

    return metrics


# ============================================================================
# Step 4: Main benchmark loop
# ============================================================================

def run_benchmark():
    keys, queries = load_model_and_extract_kv()

    print(f"\n{'=' * 70}")
    print(f"Running benchmark on {DEVICE.upper()}")
    print(f"Key vectors: {keys.shape[0]}, dim={keys.shape[1]}")
    print(f"Query vectors: {queries.shape[0]}")
    print(f"{'=' * 70}\n")

    # Define all methods
    methods = [
        # RotorQuant
        PlanarQuantMethod(),
        IsoQuantMethod(mode="full"),
        IsoQuantMethod(mode="fast"),
        RQ_TurboQuantMethod(),
        # TurboQuantDC (ours)
        PolarQuantWHTMethod(),
        ResidualQuantWHTMethod(),
        ResidualQuantMeanMethod(),
        PCAFullStackMethod(),
    ]

    # Primary benchmark: 3-bit
    bits_primary = 3
    # Extended: 2-bit and 4-bit for best methods
    bits_extended = [2, 4]

    all_results = {}

    # ---- Primary 3-bit benchmark ----
    print(f"\n{'#' * 70}")
    print(f"# 3-BIT BENCHMARK (PRIMARY)")
    print(f"{'#' * 70}\n")

    for method in methods:
        print(f"\n--- {method.name} ({method.family}) @ {bits_primary}-bit ---")
        try:
            # Quality metrics
            with torch.no_grad():
                recon = method.quantize_dequantize(keys, bits_primary)
                metrics = compute_all_metrics(keys, recon, queries)

            # Latency
            with torch.no_grad():
                q_ms, dq_ms = method.benchmark_latency(keys, bits_primary)
            metrics["quant_ms_per_1k"] = q_ms
            metrics["dequant_ms_per_1k"] = dq_ms

            key = f"{method.name}_{bits_primary}bit"
            all_results[key] = {
                "method": method.name,
                "family": method.family,
                "bits": bits_primary,
                **metrics,
            }

            print(f"  Cosine Sim:  {metrics['vector_cosine_sim']:.6f}")
            print(f"  NMSE:        {metrics['nmse']:.6f}")
            print(f"  Attn Cos:    {metrics['cosine_sim']:.6f}")
            print(f"  Top-1 Match: {metrics['top1_match']:.4f}")
            print(f"  Top-5 Match: {metrics['top5_match']:.4f}")
            print(f"  Spearman:    {metrics['spearman_rho']:.4f}")
            print(f"  Quant:       {q_ms:.3f} ms/1k vec")
            print(f"  Dequant:     {dq_ms:.3f} ms/1k vec")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ---- Extended bit-width sweep for top methods ----
    # Run 2-bit and 4-bit for all methods
    for bits in bits_extended:
        print(f"\n{'#' * 70}")
        print(f"# {bits}-BIT BENCHMARK (EXTENDED)")
        print(f"{'#' * 70}\n")

        for method in methods:
            print(f"\n--- {method.name} @ {bits}-bit ---")
            try:
                with torch.no_grad():
                    recon = method.quantize_dequantize(keys, bits)
                    metrics = compute_all_metrics(keys, recon, queries)

                with torch.no_grad():
                    q_ms, dq_ms = method.benchmark_latency(keys, bits)
                metrics["quant_ms_per_1k"] = q_ms
                metrics["dequant_ms_per_1k"] = dq_ms

                key = f"{method.name}_{bits}bit"
                all_results[key] = {
                    "method": method.name,
                    "family": method.family,
                    "bits": bits,
                    **metrics,
                }

                print(f"  Cosine Sim:  {metrics['vector_cosine_sim']:.6f}")
                print(f"  Top-1 Match: {metrics['top1_match']:.4f}")
                print(f"  Top-5 Match: {metrics['top5_match']:.4f}")
                print(f"  Quant:       {q_ms:.3f} ms/1k vec")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    return all_results


# ============================================================================
# Step 5: Generate comparison report
# ============================================================================

def generate_report(results):
    """Generate markdown comparison report."""
    lines = []
    lines.append("# TurboQuantDC vs RotorQuant: Head-to-Head Benchmark")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Model:** Qwen2.5-3B-Instruct (BnB 4-bit)")
    lines.append(f"**Device:** {DEVICE.upper()}")
    lines.append(f"**Head dimension:** {HEAD_DIM}")
    lines.append("")

    # Group by bit-width
    for bits in [3, 2, 4]:
        bit_results = {k: v for k, v in results.items() if v["bits"] == bits}
        if not bit_results:
            continue

        lines.append(f"## {bits}-Bit Comparison")
        lines.append("")

        # Table header
        lines.append("| Method | Family | Vec CosSim | Attn CosSim | Top-1 | Top-5 | Spearman | NMSE | Q ms/1k | DQ ms/1k |")
        lines.append("|--------|--------|-----------|-------------|-------|-------|----------|------|---------|----------|")

        # Sort: our methods first, then theirs; within group, by attn cosine sim descending
        sorted_results = sorted(
            bit_results.values(),
            key=lambda r: (0 if r["family"] == "turboquantdc" else 1, -r.get("cosine_sim", 0)),
        )

        best_attn_cos = max(r.get("cosine_sim", 0) for r in sorted_results)
        best_top1 = max(r.get("top1_match", 0) for r in sorted_results)
        best_top5 = max(r.get("top5_match", 0) for r in sorted_results)

        for r in sorted_results:
            name = r["method"]
            fam = "Ours" if r["family"] == "turboquantdc" else "RQ"

            vec_cos = r.get("vector_cosine_sim", 0)
            attn_cos = r.get("cosine_sim", 0)
            top1 = r.get("top1_match", 0)
            top5 = r.get("top5_match", 0)
            spear = r.get("spearman_rho", 0)
            nmse = r.get("nmse", 0)
            q_ms = r.get("quant_ms_per_1k", 0)
            dq_ms = r.get("dequant_ms_per_1k", 0)

            # Bold the best values
            vc_str = f"**{vec_cos:.6f}**" if vec_cos >= max(x.get("vector_cosine_sim", 0) for x in sorted_results) - 1e-6 else f"{vec_cos:.6f}"
            ac_str = f"**{attn_cos:.6f}**" if attn_cos >= best_attn_cos - 1e-6 else f"{attn_cos:.6f}"
            t1_str = f"**{top1:.4f}**" if top1 >= best_top1 - 1e-4 else f"{top1:.4f}"
            t5_str = f"**{top5:.4f}**" if top5 >= best_top5 - 1e-4 else f"{top5:.4f}"

            lines.append(
                f"| {name} | {fam} | {vc_str} | {ac_str} | {t1_str} | {t5_str} | {spear:.4f} | {nmse:.6f} | {q_ms:.3f} | {dq_ms:.3f} |"
            )

        lines.append("")

    # ---- Winner analysis ----
    lines.append("## Winner Analysis")
    lines.append("")

    # 3-bit winners
    bit3 = {k: v for k, v in results.items() if v["bits"] == 3}
    if bit3:
        ours_3 = {k: v for k, v in bit3.items() if v["family"] == "turboquantdc"}
        theirs_3 = {k: v for k, v in bit3.items() if v["family"] == "rotorquant"}

        if ours_3 and theirs_3:
            our_best = max(ours_3.values(), key=lambda r: r.get("cosine_sim", 0))
            their_best = max(theirs_3.values(), key=lambda r: r.get("cosine_sim", 0))

            lines.append("### 3-Bit (Primary)")
            lines.append("")
            lines.append(f"- **Our best:** {our_best['method']} -- Attn CosSim={our_best.get('cosine_sim', 0):.6f}, Top-1={our_best.get('top1_match', 0):.4f}, Top-5={our_best.get('top5_match', 0):.4f}")
            lines.append(f"- **Their best:** {their_best['method']} -- Attn CosSim={their_best.get('cosine_sim', 0):.6f}, Top-1={their_best.get('top1_match', 0):.4f}, Top-5={their_best.get('top5_match', 0):.4f}")
            lines.append("")

            delta_cos = our_best.get("cosine_sim", 0) - their_best.get("cosine_sim", 0)
            delta_top1 = our_best.get("top1_match", 0) - their_best.get("top1_match", 0)
            delta_top5 = our_best.get("top5_match", 0) - their_best.get("top5_match", 0)

            if delta_cos > 0:
                lines.append(f"**TurboQuantDC wins on attention cosine similarity by {delta_cos:.6f}**")
            else:
                lines.append(f"**RotorQuant wins on attention cosine similarity by {-delta_cos:.6f}**")
            lines.append("")

            if delta_top1 > 0:
                lines.append(f"**TurboQuantDC wins on top-1 match by {delta_top1:.4f}**")
            elif delta_top1 < 0:
                lines.append(f"**RotorQuant wins on top-1 match by {-delta_top1:.4f}**")
            else:
                lines.append("**Tied on top-1 match**")
            lines.append("")

            # Speed comparison
            our_fastest = min(ours_3.values(), key=lambda r: r.get("quant_ms_per_1k", float("inf")))
            their_fastest = min(theirs_3.values(), key=lambda r: r.get("quant_ms_per_1k", float("inf")))
            speed_ratio = their_fastest.get("quant_ms_per_1k", 1) / max(our_fastest.get("quant_ms_per_1k", 1), 1e-6)

            lines.append("### Speed")
            lines.append("")
            lines.append(f"- **Our fastest:** {our_fastest['method']} -- {our_fastest.get('quant_ms_per_1k', 0):.3f} ms/1k vectors")
            lines.append(f"- **Their fastest:** {their_fastest['method']} -- {their_fastest.get('quant_ms_per_1k', 0):.3f} ms/1k vectors")
            if speed_ratio > 1:
                lines.append(f"- **We are {speed_ratio:.2f}x faster at quantization**")
            else:
                lines.append(f"- **They are {1/speed_ratio:.2f}x faster at quantization**")
            lines.append("")

    # ---- Key questions answered ----
    lines.append("## Key Questions")
    lines.append("")

    lines.append("### Q1: Does their block-diagonal rotation beat our WHT at same bit-width?")
    lines.append("")
    # Compare IsoQuant-Full vs PolarQuant-WHT at 3-bit
    iso_3 = next((v for v in results.values() if "IsoQuant-Full" in v["method"] and v["bits"] == 3), None)
    wht_3 = next((v for v in results.values() if "PolarQuant-WHT" in v["method"] and v["bits"] == 3), None)
    if iso_3 and wht_3:
        delta = iso_3.get("cosine_sim", 0) - wht_3.get("cosine_sim", 0)
        if delta > 0.001:
            lines.append(f"YES. IsoQuant-Full (4D quaternion) beats our WHT by {delta:.6f} on attention cosine similarity at 3-bit. Their block-diagonal SO(4) rotation provides better decorrelation than WHT.")
        elif delta < -0.001:
            lines.append(f"NO. Our WHT beats IsoQuant-Full by {-delta:.6f} on attention cosine similarity at 3-bit. WHT's global mixing is superior to block-diagonal rotation.")
        else:
            lines.append(f"TIED (delta={delta:.6f}). Both rotations achieve similar decorrelation at 3-bit. WHT is simpler and faster.")
    lines.append("")

    lines.append("### Q2: Does our PCA + mean-removal + ResidualQuant beat their best?")
    lines.append("")
    pca_3 = next((v for v in results.values() if "PCA+RQ+Mean" in v["method"] and v["bits"] == 3), None)
    their_best_3 = max(
        (v for v in results.values() if v["family"] == "rotorquant" and v["bits"] == 3),
        key=lambda r: r.get("cosine_sim", 0),
        default=None,
    )
    if pca_3 and their_best_3:
        delta = pca_3.get("cosine_sim", 0) - their_best_3.get("cosine_sim", 0)
        if delta > 0:
            lines.append(f"YES. Our full stack (PCA+RQ+Mean) beats their best ({their_best_3['method']}) by {delta:.6f} on attention cosine similarity.")
            lines.append(f"  - Our Attn CosSim: {pca_3.get('cosine_sim', 0):.6f}")
            lines.append(f"  - Their best:      {their_best_3.get('cosine_sim', 0):.6f}")
        else:
            lines.append(f"NO. Their best ({their_best_3['method']}) beats our full stack by {-delta:.6f}.")
    lines.append("")

    lines.append("### Q3: Where is the speed vs quality tradeoff?")
    lines.append("")
    if bit3:
        # Sort all 3-bit by quality
        sorted_3 = sorted(bit3.values(), key=lambda r: r.get("cosine_sim", 0), reverse=True)
        lines.append("Methods ranked by quality (3-bit):")
        lines.append("")
        for i, r in enumerate(sorted_3, 1):
            fam = "Ours" if r["family"] == "turboquantdc" else "RQ"
            lines.append(f"{i}. **{r['method']}** ({fam}) -- AttnCos={r.get('cosine_sim', 0):.6f}, Q={r.get('quant_ms_per_1k', 0):.3f}ms/1k")
        lines.append("")

    # ---- Strategic takeaway ----
    lines.append("## Strategic Takeaway")
    lines.append("")

    if bit3:
        our_best = max(
            (v for v in bit3.values() if v["family"] == "turboquantdc"),
            key=lambda r: r.get("cosine_sim", 0),
            default=None,
        )
        their_best = max(
            (v for v in bit3.values() if v["family"] == "rotorquant"),
            key=lambda r: r.get("cosine_sim", 0),
            default=None,
        )
        if our_best and their_best:
            our_cos = our_best.get("cosine_sim", 0)
            their_cos = their_best.get("cosine_sim", 0)
            if our_cos > their_cos:
                margin = (our_cos - their_cos)
                lines.append(f"TurboQuantDC's full stack ({our_best['method']}) achieves the highest attention quality, ")
                lines.append(f"beating RotorQuant's best ({their_best['method']}) by {margin:.6f} attention cosine similarity at 3-bit.")
                lines.append("")
                lines.append("The key differentiators in our stack:")
                lines.append("- **ResidualQuant** (direct residual signs vs QJL random projection): lower variance reconstruction")
                lines.append("- **Mean-removal**: exploits softmax shift-invariance for free precision gain")
                lines.append("- **PCA rotation**: data-adapted rotation outperforms random rotation (WHT/QR)")
                lines.append("")
                lines.append("RotorQuant's block-diagonal rotations (Givens, quaternion, Clifford) are an interesting")
                lines.append("approach but do not overcome the fundamental advantage of our algorithmic innovations.")
            else:
                lines.append(f"RotorQuant's best method ({their_best['method']}) outperforms our full stack.")
                lines.append("Their block-diagonal rotation approach provides superior decorrelation.")

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TurboQuantDC vs RotorQuant: Head-to-Head Benchmark")
    print("=" * 70)

    results = run_benchmark()

    # Save raw results as JSON
    results_path = Path("/home/dhawal/turboQuantDC/benchmarks/results")
    results_path.mkdir(parents=True, exist_ok=True)

    json_path = results_path / "rotorquant_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {json_path}")

    # Generate and save report
    report = generate_report(results)
    md_path = results_path / "rotorquant_comparison.md"
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Report saved to {md_path}")
    print("\n" + report)
