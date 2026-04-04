"""Gemma 4 E4B-it validation for TurboQuantDC.

Tests ResidualQuant KV cache compression on Gemma 4 (4B params, 42 layers,
2 KV heads, head_dim=256). This exercises the d=256 CUDA kernel path.

Usage:
    python benchmarks/gemma4_validation.py
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "google/gemma-4-E4B-it"
BIT_WIDTHS = [3, 4]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_PATH = os.path.join(REPO_ROOT, "benchmarks", "results", "gemma4_results.md")

# Test prompt
TEST_PROMPT = (
    "You are a helpful assistant. Explain the concept of KV cache compression "
    "in large language models in exactly three sentences."
)

GENERATION_PROMPT = "The three most important inventions of the 20th century are"
MAX_NEW_TOKENS = 50


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class CompressionResult:
    """Per-layer, per-head compression metrics."""
    layer: int
    head: int
    bits: int
    method: str  # "ResidualQuant" or "PolarQuant"
    cosine_sim: float
    top1_match: bool
    top5_match: bool


@dataclass
class AggregateResult:
    """Aggregate metrics for one configuration."""
    bits: int
    method: str
    mean_cosine: float
    top1_rate: float
    top5_rate: float
    num_heads_measured: int
    compression_ratio: float


@dataclass
class GenerationResult:
    """Result of generation comparison."""
    method: str
    text: str
    tokens_generated: int
    time_sec: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_attention_scores(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Compute scaled dot-product attention scores.

    Args:
        queries: (seq_q, d)
        keys: (seq_k, d)

    Returns:
        Attention weights (seq_q, seq_k) after softmax.
    """
    d = queries.shape[-1]
    scores = queries @ keys.T / math.sqrt(d)
    return F.softmax(scores, dim=-1)


def top_k_match(fp16_scores: torch.Tensor, quant_scores: torch.Tensor, k: int) -> float:
    """Fraction of queries where the top-k tokens match."""
    fp16_topk = fp16_scores.topk(k, dim=-1).indices
    quant_topk = quant_scores.topk(k, dim=-1).indices
    matches = 0
    total = fp16_scores.shape[0]
    for i in range(total):
        fp_set = set(fp16_topk[i].tolist())
        q_set = set(quant_topk[i].tolist())
        if fp_set & q_set:  # any overlap counts
            matches += 1
    return matches / max(total, 1)


def format_results_md(
    model_info: Dict[str, Any],
    aggregate_results: List[AggregateResult],
    generation_results: List[GenerationResult],
    findings: List[str],
    timings: Dict[str, float],
) -> str:
    """Format all results as markdown."""
    lines = []
    lines.append("# Gemma 4 E4B-it Validation Results")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Model:** {MODEL_NAME}")
    lines.append("")

    # Model info
    lines.append("## Model Architecture")
    lines.append("")
    for k, v in model_info.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")

    # Compression quality
    lines.append("## Compression Quality")
    lines.append("")
    lines.append("| Bits | Method | Cosine Sim | Top-1 Match | Top-5 Match | Compression Ratio | Heads |")
    lines.append("|------|--------|-----------|-------------|-------------|-------------------|-------|")
    for r in aggregate_results:
        lines.append(
            f"| {r.bits} | {r.method} | {r.mean_cosine:.6f} | {r.top1_rate:.1%} | "
            f"{r.top5_rate:.1%} | {r.compression_ratio:.2f}x | {r.num_heads_measured} |"
        )
    lines.append("")

    # Paper targets comparison
    lines.append("### Paper Targets Comparison")
    lines.append("")
    lines.append("| Metric | Target | Our 3-bit RQ | Our 4-bit RQ | Status |")
    lines.append("|--------|--------|-------------|-------------|--------|")
    rq3 = next((r for r in aggregate_results if r.bits == 3 and r.method == "ResidualQuant"), None)
    rq4 = next((r for r in aggregate_results if r.bits == 4 and r.method == "ResidualQuant"), None)
    if rq3 and rq4:
        cos3 = f"{rq3.mean_cosine:.4f}"
        cos4 = f"{rq4.mean_cosine:.4f}"
        t1_3 = f"{rq3.top1_rate:.0%}"
        t1_4 = f"{rq4.top1_rate:.0%}"
        t5_3 = f"{rq3.top5_rate:.0%}"
        t5_4 = f"{rq4.top5_rate:.0%}"
        lines.append(f"| Cosine Sim > 0.995 | 0.995 | {cos3} | {cos4} | {'PASS' if rq3.mean_cosine > 0.995 else 'FAIL'} / {'PASS' if rq4.mean_cosine > 0.995 else 'FAIL'} |")
        lines.append(f"| Compression > 4.5x | 4.5x | {rq3.compression_ratio:.1f}x | {rq4.compression_ratio:.1f}x | {'PASS' if rq3.compression_ratio > 4.5 else 'FAIL'} / {'PASS' if rq4.compression_ratio > 4.5 else 'FAIL'} |")
        lines.append(f"| Top-5 Match > 90% | 90% | {t5_3} | {t5_4} | {'PASS' if rq3.top5_rate > 0.9 else 'FAIL'} / {'PASS' if rq4.top5_rate > 0.9 else 'FAIL'} |")
    lines.append("")

    # Generation comparison
    if generation_results:
        lines.append("## Generation Comparison")
        lines.append("")
        for gr in generation_results:
            lines.append(f"### {gr.method}")
            lines.append(f"- Tokens: {gr.tokens_generated}")
            lines.append(f"- Time: {gr.time_sec:.2f}s")
            lines.append(f"- Text: `{gr.text}`")
            lines.append("")

    # Timings
    lines.append("## Timings")
    lines.append("")
    for k, v in timings.items():
        lines.append(f"- **{k}:** {v:.2f}s")
    lines.append("")

    # d=256 specific findings
    lines.append("## Key Findings (d=256 Path)")
    lines.append("")
    for f in findings:
        lines.append(f"- {f}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Gemma 4 E4B-it Validation for TurboQuantDC")
    print("=" * 70)
    print()

    timings = {}
    findings = []
    model_info = {}
    aggregate_results = []
    generation_results = []

    # ----- Step 1: Load the model -----
    print("[1/4] Loading model...")
    t0 = time.time()

    try:
        from transformers import AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Try AutoModelForImageTextToText first (Gemma 4 is multimodal)
        model = None
        load_class_name = None
        for cls_name in [
            "AutoModelForImageTextToText",
            "Gemma4ForConditionalGeneration",
            "AutoModelForCausalLM",
        ]:
            try:
                import transformers
                cls = getattr(transformers, cls_name, None)
                if cls is None:
                    print(f"  {cls_name}: not found in transformers")
                    continue
                print(f"  Trying {cls_name}...")
                model = cls.from_pretrained(
                    MODEL_NAME,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                load_class_name = cls_name
                print(f"  Loaded with {cls_name}")
                break
            except Exception as e:
                print(f"  {cls_name} failed: {e}")
                continue

        if model is None:
            raise RuntimeError("Could not load model with any known class")

        timings["model_load"] = time.time() - t0
        print(f"  Model loaded in {timings['model_load']:.1f}s via {load_class_name}")

        # Extract architecture info
        config = model.config
        # Gemma 4 might nest the text config under .text_config
        text_config = getattr(config, "text_config", config)
        num_layers = getattr(text_config, "num_hidden_layers", "?")
        num_kv_heads = getattr(text_config, "num_key_value_heads", "?")
        num_attn_heads = getattr(text_config, "num_attention_heads", "?")
        head_dim = getattr(text_config, "head_dim", "?")
        hidden_size = getattr(text_config, "hidden_size", "?")
        vocab_size = getattr(text_config, "vocab_size", "?")

        model_info = {
            "Load class": load_class_name,
            "Num layers": num_layers,
            "Num attention heads": num_attn_heads,
            "Num KV heads": num_kv_heads,
            "Head dimension": head_dim,
            "Hidden size": hidden_size,
            "Vocab size": vocab_size,
            "Device": str(model.device if hasattr(model, "device") else "auto"),
            "Quantization": "4-bit (bitsandbytes NF4)",
        }

        print(f"  Architecture: {num_layers}L / {num_kv_heads} KV heads / head_dim={head_dim}")

        # Validate d=256
        if head_dim == 256:
            findings.append("head_dim=256 CONFIRMED -- exercises the CUDA kernel d=256 path")
            findings.append("d=256 is a power of 2 -- WHT rotation will be used (O(d log d))")
        elif isinstance(head_dim, int):
            findings.append(f"UNEXPECTED head_dim={head_dim} (expected 256)")
        else:
            findings.append(f"Could not determine head_dim from config: {head_dim}")

    except Exception as e:
        print(f"  MODEL LOAD FAILED: {e}")
        traceback.print_exc()
        findings.append(f"MODEL LOAD FAILED: {e}")
        # Write partial results and exit
        md = format_results_md(model_info, aggregate_results, generation_results, findings, timings)
        with open(RESULTS_PATH, "w") as f:
            f.write(md)
        print(f"\nPartial results saved to {RESULTS_PATH}")
        return

    # ----- Step 2: Extract KV caches -----
    print("\n[2/4] Extracting KV caches...")
    t0 = time.time()

    try:
        inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
        seq_len = inputs["input_ids"].shape[1]
        print(f"  Prompt tokens: {seq_len}")

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=False,
                use_cache=True,
            )

        past_kv = outputs.past_key_values
        timings["kv_extraction"] = time.time() - t0
        print(f"  KV cache extracted in {timings['kv_extraction']:.1f}s")

        # Inspect KV cache structure
        # transformers 5.5+ DynamicCache uses .layers[i].keys/.values
        if hasattr(past_kv, "layers") and len(past_kv.layers) > 0:
            num_cached_layers = len(past_kv.layers)
            sample_layer = past_kv.layers[0]
            sample_k = sample_layer.keys
            print(f"  DynamicCache ({type(past_kv).__name__}): {num_cached_layers} layers")
            print(f"  Key shape: {sample_k.shape}")  # (batch, heads, seq, dim)
            print(f"  Key dtype: {sample_k.dtype}")

            kv_layers = [
                (past_kv.layers[i].keys, past_kv.layers[i].values)
                for i in range(num_cached_layers)
            ]
        elif hasattr(past_kv, "key_cache"):
            # Older transformers DynamicCache with key_cache/value_cache
            num_cached_layers = len(past_kv.key_cache)
            sample_k = past_kv.key_cache[0]
            print(f"  DynamicCache (legacy): {num_cached_layers} layers")
            print(f"  Key shape: {sample_k.shape}")
            kv_layers = [
                (past_kv.key_cache[i], past_kv.value_cache[i])
                for i in range(num_cached_layers)
            ]
        elif isinstance(past_kv, (list, tuple)) and len(past_kv) > 0:
            if isinstance(past_kv[0], (list, tuple)):
                num_cached_layers = len(past_kv)
                sample_k = past_kv[0][0]
                print(f"  Tuple cache: {num_cached_layers} layers")
                print(f"  Key shape: {sample_k.shape}")
                kv_layers = [(layer[0], layer[1]) for layer in past_kv]
            else:
                raise ValueError(f"Unexpected past_kv structure: {type(past_kv[0])}")
        else:
            raise ValueError(f"Unexpected past_kv type: {type(past_kv)}")

        actual_head_dim = kv_layers[0][0].shape[-1]
        actual_num_kv_heads = kv_layers[0][0].shape[1]
        actual_seq = kv_layers[0][0].shape[2]
        findings.append(
            f"KV cache shape: batch=1, kv_heads={actual_num_kv_heads}, "
            f"seq={actual_seq}, head_dim={actual_head_dim}"
        )
        print(f"  Actual: kv_heads={actual_num_kv_heads}, seq={actual_seq}, d={actual_head_dim}")

    except Exception as e:
        print(f"  KV EXTRACTION FAILED: {e}")
        traceback.print_exc()
        findings.append(f"KV EXTRACTION FAILED: {e}")
        md = format_results_md(model_info, aggregate_results, generation_results, findings, timings)
        with open(RESULTS_PATH, "w") as f:
            f.write(md)
        print(f"\nPartial results saved to {RESULTS_PATH}")
        return

    # ----- Step 3: Compression quality test -----
    print("\n[3/4] Testing compression quality...")
    t0 = time.time()

    from turboquantdc.residual_quant import ResidualQuantEstimator
    from turboquantdc.polarquant import PolarQuant

    d = actual_head_dim

    # Inspect all layer shapes -- Gemma 4 is multimodal, layers may differ
    print(f"  Inspecting layer shapes:")
    layer_shapes = {}
    for i, (k, v) in enumerate(kv_layers):
        shape_key = k.shape
        if shape_key not in layer_shapes:
            layer_shapes[shape_key] = []
        layer_shapes[shape_key].append(i)
    for shape, layers in layer_shapes.items():
        print(f"    Shape {shape}: layers {layers[:5]}{'...' if len(layers) > 5 else ''} ({len(layers)} total)")
        findings.append(f"KV shape {shape}: {len(layers)} layers")

    # Test codebook creation for d=256
    print(f"  Creating codebooks for d={d}...")
    try:
        from turboquantdc.codebook import LloydMaxCodebook
        for bits in BIT_WIDTHS:
            cb = LloydMaxCodebook(d=d, bits=bits)
            print(f"    {bits}-bit codebook: {cb.n_levels} levels, "
                  f"centroids range [{cb.centroids.min():.6f}, {cb.centroids.max():.6f}]")
        findings.append(f"LloydMaxCodebook for d={d} created successfully at all bit widths")
    except Exception as e:
        findings.append(f"CODEBOOK CREATION FAILED for d={d}: {e}")
        traceback.print_exc()

    # Test WHT rotation at d=256
    print(f"  Testing WHT rotation at d={d}...")
    try:
        from turboquantdc.rotation import generate_wht_rotation, apply_wht_rotation
        wht = generate_wht_rotation(d, seed=42, device=DEVICE)
        test_vec = torch.randn(4, d, device=DEVICE)
        rotated = apply_wht_rotation(test_vec, wht)
        # Check orthogonality: ||rotated|| should equal ||original||
        norm_ratio = rotated.norm(dim=-1) / test_vec.norm(dim=-1)
        print(f"    Norm preservation: {norm_ratio.mean():.6f} (should be 1.0)")
        findings.append(f"WHT rotation at d={d}: norm preservation = {norm_ratio.mean():.6f}")
    except Exception as e:
        findings.append(f"WHT ROTATION FAILED at d={d}: {e}")
        traceback.print_exc()

    # Test CUDA kernels at d=256
    print(f"  Testing CUDA kernels at d={d}...")
    try:
        from turboquantdc.cuda_kernels import _ensure_backend, _CUDA_DEQUANTIZE, _CUDA_WHT
        _ensure_backend()
        if _CUDA_DEQUANTIZE is not None:
            findings.append(f"CUDA dequantize kernel loaded for d={d}")
        else:
            findings.append(f"CUDA dequantize kernel NOT available -- using fallback")
        if _CUDA_WHT is not None:
            findings.append(f"CUDA WHT kernel loaded for d={d}")
        else:
            findings.append(f"CUDA WHT kernel NOT available -- using fallback")
    except Exception as e:
        findings.append(f"CUDA kernel check: {e}")

    # Run compression on each layer/head
    all_results: List[CompressionResult] = []

    # Sample layers evenly: first, 1/4, 1/2, 3/4, last
    total_layers = len(kv_layers)
    sample_indices = sorted(set([
        0,
        total_layers // 4,
        total_layers // 2,
        3 * total_layers // 4,
        total_layers - 1,
    ]))
    print(f"  Sampling layers: {sample_indices} (of {total_layers})")

    # Cache of estimators keyed by (bits, head_dim) -- handle variable head dims
    rq_cache = {}
    pq_cache = {}

    for bits in BIT_WIDTHS:
        print(f"\n  --- {bits}-bit compression ---")

        for layer_idx in sample_indices:
            keys_fp16, vals_fp16 = kv_layers[layer_idx]
            # keys shape: (batch, num_kv_heads, seq, head_dim)
            batch, num_h, seq, hd = keys_fp16.shape

            # Get or create estimators for this head_dim
            rq_key = (bits, hd)
            if rq_key not in rq_cache:
                try:
                    rq_cache[rq_key] = ResidualQuantEstimator(d=hd, bits=bits, seed=42, device=DEVICE)
                    pq_cache[rq_key] = PolarQuant(d=hd, bits=max(bits - 1, 1), seed=42, device=DEVICE)
                    print(f"    Created estimators for d={hd}, {bits}-bit")
                except Exception as e:
                    findings.append(f"{bits}-bit estimator creation FAILED for d={hd}: {e}")
                    traceback.print_exc()
                    rq_cache[rq_key] = None
                    pq_cache[rq_key] = None

            rq = rq_cache[rq_key]
            pq = pq_cache[rq_key]

            for head_idx in range(num_h):
                # Extract single head's keys: (seq, d)
                k = keys_fp16[0, head_idx].float().to(DEVICE)  # (seq, hd)
                print(f"    L{layer_idx} H{head_idx}: k.shape={k.shape}, dtype={k.dtype}, device={k.device}")
                # Use last token as query
                q = k[-1:, :]  # (1, hd)

                # FP16 baseline attention
                fp16_scores = compute_attention_scores(q, k)  # (1, seq)

                if rq is None or pq is None:
                    continue

                # --- ResidualQuant ---
                try:
                    compressed_rq = rq.quantize(k)
                    k_rq = rq.dequantize(compressed_rq)

                    rq_scores = compute_attention_scores(q, k_rq)
                    cos = F.cosine_similarity(
                        fp16_scores.flatten().unsqueeze(0),
                        rq_scores.flatten().unsqueeze(0),
                    ).item()

                    fp16_top1 = fp16_scores.argmax(dim=-1)
                    rq_top1 = rq_scores.argmax(dim=-1)
                    t1 = (fp16_top1 == rq_top1).all().item()

                    fp16_top5 = fp16_scores.topk(5, dim=-1).indices
                    rq_top5 = rq_scores.topk(5, dim=-1).indices
                    t5 = bool(set(fp16_top5[0].tolist()) & set(rq_top5[0].tolist()))

                    all_results.append(CompressionResult(
                        layer=layer_idx, head=head_idx, bits=bits,
                        method="ResidualQuant", cosine_sim=cos,
                        top1_match=t1, top5_match=t5,
                    ))
                except Exception as e:
                    findings.append(f"ResidualQuant {bits}b L{layer_idx}H{head_idx}: {e}")
                    traceback.print_exc()

                # --- PolarQuant (MSE only) ---
                try:
                    # Normalize for PolarQuant
                    k_norm = k.norm(dim=-1, keepdim=True)
                    k_unit = k / (k_norm + 1e-8)
                    indices = pq.quantize(k_unit)
                    k_pq = pq.dequantize(indices) * k_norm

                    pq_scores = compute_attention_scores(q, k_pq)
                    cos = F.cosine_similarity(
                        fp16_scores.flatten().unsqueeze(0),
                        pq_scores.flatten().unsqueeze(0),
                    ).item()

                    fp16_top1 = fp16_scores.argmax(dim=-1)
                    pq_top1 = pq_scores.argmax(dim=-1)
                    t1 = (fp16_top1 == pq_top1).all().item()

                    fp16_top5 = fp16_scores.topk(5, dim=-1).indices
                    pq_top5 = pq_scores.topk(5, dim=-1).indices
                    t5 = bool(set(fp16_top5[0].tolist()) & set(pq_top5[0].tolist()))

                    all_results.append(CompressionResult(
                        layer=layer_idx, head=head_idx, bits=bits,
                        method="PolarQuant", cosine_sim=cos,
                        top1_match=t1, top5_match=t5,
                    ))
                except Exception as e:
                    findings.append(f"PolarQuant {bits}b L{layer_idx}H{head_idx}: {e}")
                    traceback.print_exc()

        # Print summary for this bit-width
        for method in ["ResidualQuant", "PolarQuant"]:
            method_results = [
                r for r in all_results if r.bits == bits and r.method == method
            ]
            if method_results:
                cos_mean = sum(r.cosine_sim for r in method_results) / len(method_results)
                t1_rate = sum(1 for r in method_results if r.top1_match) / len(method_results)
                t5_rate = sum(1 for r in method_results if r.top5_match) / len(method_results)
                print(f"    {method} {bits}b: cos={cos_mean:.6f} top1={t1_rate:.1%} top5={t5_rate:.1%}")

                # Compute compression ratio
                if method == "ResidualQuant":
                    # bits = (mse_bits)*d + d (signs) + 16 (scale) + 16 (norm)
                    # = bits*d + 32 per vector
                    compressed_bpv = bits * d + 32
                else:
                    # PolarQuant only: (bits-1)*d + 16 (norm)
                    mse_bits_used = max(bits - 1, 1)
                    compressed_bpv = mse_bits_used * d + 16
                fp16_bpv = d * 16
                ratio = fp16_bpv / compressed_bpv

                aggregate_results.append(AggregateResult(
                    bits=bits, method=method,
                    mean_cosine=cos_mean, top1_rate=t1_rate, top5_rate=t5_rate,
                    num_heads_measured=len(method_results),
                    compression_ratio=ratio,
                ))

    timings["compression_test"] = time.time() - t0

    # ----- Step 4: Generation test -----
    print("\n[4/4] Generation comparison test...")
    t0 = time.time()

    try:
        from turboquantdc.generation_cache import GenerationCache

        gen_inputs = tokenizer(GENERATION_PROMPT, return_tensors="pt").to(model.device)

        # FP16 baseline generation
        print("  Generating with FP16 KV cache...")
        t_gen = time.time()
        with torch.no_grad():
            fp16_output = model.generate(
                **gen_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
            )
        fp16_time = time.time() - t_gen
        fp16_text = tokenizer.decode(fp16_output[0], skip_special_tokens=True)
        fp16_tokens = fp16_output.shape[1] - gen_inputs["input_ids"].shape[1]
        print(f"  FP16: {fp16_tokens} tokens in {fp16_time:.2f}s")
        print(f"  FP16 text: {fp16_text[:200]}")

        generation_results.append(GenerationResult(
            method="FP16 Baseline",
            text=fp16_text,
            tokens_generated=fp16_tokens,
            time_sec=fp16_time,
        ))

        # 3-bit ResidualQuant generation
        print("\n  Generating with 3-bit ResidualQuant cache...")
        t_gen = time.time()
        try:
            rq_cache = GenerationCache(
                key_bits=3,
                val_bits=2,
                fp16_window=64,
                anchor_interval=6,
                use_residual_quant=True,
            )

            # Monkey-patch get_mask_sizes to handle int argument from
            # transformers 5.5+ (Gemma 4 masking_utils passes int, not tensor)
            _original_gms = rq_cache.get_mask_sizes
            def _patched_gms(query_length_or_pos, layer_idx=0):
                if isinstance(query_length_or_pos, int):
                    # transformers 5.5+ passes query_length as int
                    seq = rq_cache.get_seq_length(layer_idx)
                    return seq + query_length_or_pos, 0
                return _original_gms(query_length_or_pos, layer_idx)
            rq_cache.get_mask_sizes = _patched_gms

            with torch.no_grad():
                rq_output = model.generate(
                    **gen_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=1.0,
                    past_key_values=rq_cache,
                )
            rq_time = time.time() - t_gen
            rq_text = tokenizer.decode(rq_output[0], skip_special_tokens=True)
            rq_tokens = rq_output.shape[1] - gen_inputs["input_ids"].shape[1]
            print(f"  RQ3: {rq_tokens} tokens in {rq_time:.2f}s")
            print(f"  RQ3 text: {rq_text[:200]}")

            generation_results.append(GenerationResult(
                method="3-bit ResidualQuant (K3/V2, anchor=6, win=64)",
                text=rq_text,
                tokens_generated=rq_tokens,
                time_sec=rq_time,
            ))

            # Compare outputs
            if fp16_text.strip() == rq_text.strip():
                findings.append("GENERATION MATCH: 3-bit ResidualQuant output IDENTICAL to FP16")
            else:
                # Check token-level similarity
                fp16_toks = tokenizer.encode(fp16_text)
                rq_toks = tokenizer.encode(rq_text)
                common = sum(1 for a, b in zip(fp16_toks, rq_toks) if a == b)
                match_pct = common / max(len(fp16_toks), 1) * 100
                findings.append(
                    f"GENERATION DIVERGENCE: {match_pct:.0f}% token match between FP16 and 3-bit RQ"
                )

            # Memory stats
            if hasattr(rq_cache, "memory_usage_bits"):
                mem = rq_cache.memory_usage_bits()
                findings.append(
                    f"GenerationCache memory: {mem.get('compression_ratio', '?')}x compression"
                )

        except Exception as e:
            print(f"  ResidualQuant generation FAILED: {e}")
            traceback.print_exc()
            findings.append(f"GenerationCache FAILED: {e}")

    except Exception as e:
        print(f"  Generation test FAILED: {e}")
        traceback.print_exc()
        findings.append(f"Generation test FAILED: {e}")

    timings["generation_test"] = time.time() - t0

    # ----- Save results -----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for r in aggregate_results:
        print(f"  {r.bits}b {r.method}: cos={r.mean_cosine:.6f} "
              f"top1={r.top1_rate:.1%} top5={r.top5_rate:.1%} "
              f"ratio={r.compression_ratio:.2f}x")

    print("\nFindings:")
    for f in findings:
        print(f"  - {f}")

    md = format_results_md(model_info, aggregate_results, generation_results, findings, timings)
    with open(RESULTS_PATH, "w") as f:
        f.write(md)
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
