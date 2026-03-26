#!/usr/bin/env python3
"""TurboQuant Bit-Width Comparison Benchmark.

Compares TurboQuant across bit-widths and measures inner product fidelity.

Run with:
    python benchmarks/compare.py

Sections:
    1. Bit-Width Sweep          (d=128, 2000 vectors, bits 1-4)
    2. Dimension Sweep          (d=64/128/256, bits=3)
    3. KV Cache Compression     (token count scaling)
    4. Attention Score Fidelity (seq_len scaling, bits=3)
    5. MSE-only vs Full TurboQuant (bias analysis)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch

# Allow running from repo root: python benchmarks/compare.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from turboquantdc import (
    LloydMaxCodebook,  # noqa: F401 — exported, available for downstream scripts
    PolarQuant,
    QJL,  # noqa: F401
    TurboQuantEstimator,
    TurboQuantKVCache,
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _rand_unit(n: int, d: int, seed: int = 0) -> torch.Tensor:
    """Return n random unit vectors of dimension d (on DEVICE)."""
    # torch.Generator is device-bound — generate on CPU then move to DEVICE.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn(n, d, generator=gen).to(DEVICE)
    return x / x.norm(dim=-1, keepdim=True)


def _section_header(title: str) -> None:
    width = 62
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _compute_bit_width_stats(
    d: int,
    bits: int,
    n_vectors: int = 2000,
    seed: int = SEED,
) -> Dict[str, float]:
    """Run a single (d, bits) configuration and return fidelity metrics.

    Returns a dict with keys:
        d_mse         -- mean squared reconstruction error (MSE stage only)
        ip_rmse       -- RMSE of inner product estimate vs ground truth
        ip_bias       -- mean signed error of inner product estimate
        cosine_sim    -- mean cosine similarity between x and MSE reconstruction
        comp_ratio    -- bits-16 compression ratio
        bytes_per_vec -- storage per vector in bytes
    """
    estimator = TurboQuantEstimator(d=d, bits=bits, seed=seed, device=DEVICE)
    x = _rand_unit(n_vectors, d, seed=seed)

    # Compress all vectors
    compressed = estimator.quantize(x)

    # MSE distortion (reconstruction via MSE stage only, rescaled by vec_norm)
    x_mse = estimator.dequantize_mse(compressed)   # (n, d)
    diff = x - x_mse
    d_mse = (diff * diff).sum(dim=-1).mean().item()

    # Cosine similarity between original and MSE reconstruction
    cos_sim = torch.nn.functional.cosine_similarity(x, x_mse, dim=-1).mean().item()

    # Inner product fidelity: pair first half as queries, second half as keys
    half = n_vectors // 2
    queries = x[:half]
    keys_compressed = {
        "mse_indices":   compressed["mse_indices"][half:],
        "qjl_signs":     compressed["qjl_signs"][half:],
        "residual_norm": compressed["residual_norm"][half:],
        "vec_norm":      compressed["vec_norm"][half:],
    }
    true_ip = (queries * x[half:]).sum(dim=-1)              # (half,)
    est_ip  = estimator.inner_product(queries, keys_compressed).diag()  # (half,)

    ip_error = est_ip - true_ip
    ip_rmse  = ip_error.pow(2).mean().sqrt().item()
    ip_bias  = ip_error.mean().item()

    # Compression ratio: FP16 (16*d) vs TurboQuant (mse_bits*d + qjl_m + 32)
    mse_bits     = estimator.mse_bits
    qjl_m        = estimator.qjl.m
    storage_bits = mse_bits * d + qjl_m * 1 + 16 + 16   # +16 r_norm, +16 v_norm
    fp16_bits    = 16 * d
    comp_ratio   = fp16_bits / storage_bits
    bytes_per_vec = storage_bits / 8.0

    return {
        "d_mse":         d_mse,
        "ip_rmse":       ip_rmse,
        "ip_bias":       ip_bias,
        "cosine_sim":    cos_sim,
        "comp_ratio":    comp_ratio,
        "bytes_per_vec": bytes_per_vec,
    }


# ---------------------------------------------------------------------------
# Section 1 — Bit-Width Sweep
# ---------------------------------------------------------------------------

def section_bit_width_sweep(d: int = 128, n_vectors: int = 2000) -> None:
    _section_header(f"1. Bit-Width Comparison (d={d}, {n_vectors} vectors)")

    print(
        f"\n{'Bits':>5}  {'D_mse':>9}  {'IP RMSE':>9}  {'Bias':>9}  "
        f"{'CosSim':>8}  {'Ratio':>6}  {'Bytes/vec':>10}"
    )
    sep = "-" * 5 + "  " + "  ".join(["-" * 9, "-" * 9, "-" * 9, "-" * 8, "-" * 6, "-" * 10])
    print(sep)

    for bits in [1, 2, 3, 4]:
        s = _compute_bit_width_stats(d=d, bits=bits, n_vectors=n_vectors)
        bias_sign = "+" if s["ip_bias"] >= 0 else ""
        print(
            f"{bits:>5}  "
            f"{s['d_mse']:>9.4f}  "
            f"{s['ip_rmse']:>9.4f}  "
            f"{bias_sign}{s['ip_bias']:>8.4f}  "
            f"{s['cosine_sim']:>8.4f}  "
            f"{s['comp_ratio']:>5.1f}x  "
            f"{s['bytes_per_vec']:>8.1f} B"
        )

    print()
    print("  Paper targets (3-bit): CosSim > 0.9945, comp ratio > 4.5x")


# ---------------------------------------------------------------------------
# Section 2 — Dimension Sweep
# ---------------------------------------------------------------------------

def section_dimension_sweep(bits: int = 3, n_vectors: int = 2000) -> None:
    _section_header(f"2. Dimension Sweep (bits={bits}, {n_vectors} vectors)")

    print(
        f"\n{'d':>5}  {'D_mse':>9}  {'IP RMSE':>9}  {'Bias':>9}  "
        f"{'CosSim':>8}  {'Ratio':>6}  {'Bytes/vec':>10}"
    )
    sep = "-" * 5 + "  " + "  ".join(["-" * 9, "-" * 9, "-" * 9, "-" * 8, "-" * 6, "-" * 10])
    print(sep)

    for d in [64, 128, 256]:
        s = _compute_bit_width_stats(d=d, bits=bits, n_vectors=n_vectors)
        bias_sign = "+" if s["ip_bias"] >= 0 else ""
        print(
            f"{d:>5}  "
            f"{s['d_mse']:>9.4f}  "
            f"{s['ip_rmse']:>9.4f}  "
            f"{bias_sign}{s['ip_bias']:>8.4f}  "
            f"{s['cosine_sim']:>8.4f}  "
            f"{s['comp_ratio']:>5.1f}x  "
            f"{s['bytes_per_vec']:>8.1f} B"
        )

    print()
    print("  Quality should stay consistent across dims — rotation spreads error uniformly.")


# ---------------------------------------------------------------------------
# Section 3 — KV Cache Compression Demo
# ---------------------------------------------------------------------------

def section_kv_cache_compression(d: int = 128, bits: int = 3) -> None:
    _section_header(f"3. KV Cache Compression Demo (d_key={d}, d_value={d}, bits={bits})")

    token_counts = [1024, 4096, 16384]

    print(
        f"\n{'Tokens':>8}  {'Total bits':>12}  {'FP16 bits':>12}  "
        f"{'Ratio':>8}  {'Key MSE':>10}  {'Key QJL':>10}  {'Val MSE':>10}"
    )
    print("-" * 90)

    for n_tokens in token_counts:
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=bits, seed=SEED, device=DEVICE)

        chunk_size = 256
        gen = torch.Generator(device="cpu")
        gen.manual_seed(SEED)

        for start in range(0, n_tokens, chunk_size):
            batch = min(chunk_size, n_tokens - start)
            keys   = torch.randn(batch, d, generator=gen).to(DEVICE)
            values = torch.randn(batch, d, generator=gen).to(DEVICE)
            cache.append(keys, values)

        mem   = cache.memory_usage_bits()
        total = mem["total_bits"]
        fp16  = mem["fp16_baseline_bits"]
        ratio = mem["compression_ratio"]

        print(
            f"{n_tokens:>8,}  "
            f"{total:>12,}  "
            f"{fp16:>12,}  "
            f"{ratio:>7.2f}x  "
            f"{mem['key_mse_bits']:>10,}  "
            f"{mem['key_qjl_bits']:>10,}  "
            f"{mem['value_mse_bits']:>10,}"
        )

    print()
    print("  Paper target: ~5x compression for d=128, 3-bit keys+values.")


# ---------------------------------------------------------------------------
# Section 4 — Attention Score Fidelity
# ---------------------------------------------------------------------------

def _topk_match_pct(
    scores_true: torch.Tensor,
    scores_est: torch.Tensor,
    k: int,
) -> float:
    """Fraction of queries for which estimated top-k intersects true top-k.

    Args:
        scores_true: (n_queries, seq_len)
        scores_est:  (n_queries, seq_len)
        k:           top-k size
    """
    _, true_topk = scores_true.topk(k, dim=-1)   # (n_queries, k)
    _, est_topk  = scores_est.topk(k, dim=-1)

    n = scores_true.shape[0]
    match = 0
    for i in range(n):
        t = set(true_topk[i].tolist())
        e = set(est_topk[i].tolist())
        if t & e:
            match += 1
    return match / n


def section_attention_fidelity(
    bits: int = 3,
    d: int = 128,
    n_queries: int = 32,
) -> None:
    _section_header(
        f"4. Attention Score Fidelity (bits={bits}, d={d}, {n_queries} queries)"
    )

    seq_lengths = [512, 2048, 8192]

    print(
        f"\n{'SeqLen':>8}  {'Score CosSim':>13}  {'Top-1 Match%':>13}  {'Top-5 Match%':>13}"
    )
    print("-" * 55)

    for seq_len in seq_lengths:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(SEED)

        keys    = torch.randn(seq_len, d, generator=gen).to(DEVICE)
        queries = torch.randn(n_queries, d, generator=gen).to(DEVICE)

        true_scores = queries @ keys.T   # (n_queries, seq_len)

        estimator  = TurboQuantEstimator(d=d, bits=bits, seed=SEED, device=DEVICE)
        compressed = estimator.quantize(keys)
        est_scores = estimator.inner_product(queries, compressed)   # (n_queries, seq_len)

        cos_sim = torch.nn.functional.cosine_similarity(
            true_scores, est_scores, dim=-1
        ).mean().item()

        top1 = _topk_match_pct(true_scores, est_scores, k=1) * 100.0
        top5 = _topk_match_pct(true_scores, est_scores, k=5) * 100.0

        print(
            f"{seq_len:>8,}  "
            f"{cos_sim:>13.4f}  "
            f"{top1:>12.1f}%  "
            f"{top5:>12.1f}%"
        )

    print()
    print("  Paper targets: top-5 match > 88-94%.")


# ---------------------------------------------------------------------------
# Section 5 — MSE-only vs Full TurboQuant
# ---------------------------------------------------------------------------

def section_mse_vs_full(
    d: int = 128,
    bits: int = 3,
    n_vectors: int = 2000,
) -> None:
    _section_header(
        f"5. PolarQuant (MSE-only) vs TurboQuant (MSE+QJL)  d={d}, bits={bits}"
    )

    mse_bits = max(bits - 1, 1)

    # Same rotation seed as TurboQuantEstimator uses internally
    polar = PolarQuant(d=d, bits=mse_bits, seed=SEED, device=DEVICE)
    full  = TurboQuantEstimator(d=d, bits=bits, seed=SEED, device=DEVICE)

    x = _rand_unit(n_vectors, d, seed=SEED)

    half    = n_vectors // 2
    queries = x[:half]
    keys    = x[half:]

    true_ip = (queries * keys).sum(dim=-1)   # (half,)

    # -- PolarQuant (MSE-only) inner products --
    key_norm      = keys.norm(dim=-1, keepdim=True)
    keys_unit     = keys / (key_norm + 1e-8)
    key_indices   = polar.quantize(keys_unit)           # (half, d)
    keys_mse_hat  = polar.dequantize(key_indices)       # (half, d) unit recon
    keys_mse_full = keys_mse_hat * key_norm              # rescale

    mse_ip    = (queries @ keys_mse_full.T).diag()
    mse_err   = mse_ip - true_ip
    mse_rmse  = mse_err.pow(2).mean().sqrt().item()
    mse_bias  = mse_err.mean().item()
    mse_var   = mse_err.var().item()

    # -- Full TurboQuant inner products --
    compressed    = full.quantize(keys)
    full_ip       = full.inner_product(queries, compressed).diag()
    full_err      = full_ip - true_ip
    full_rmse     = full_err.pow(2).mean().sqrt().item()
    full_bias     = full_err.mean().item()
    full_var      = full_err.var().item()

    col_w = [22, 9, 9, 10, 10]
    hdr = (
        f"{'Method':<{col_w[0]}}  {'RMSE':>{col_w[1]}}  {'Bias':>{col_w[2]}}  "
        f"{'Variance':>{col_w[3]}}  {'Unbiased?':>{col_w[4]}}"
    )
    print()
    print(hdr)
    print("-" * len(hdr))

    def _flag(bias: float) -> str:
        return "YES" if abs(bias) < 5e-3 else f"NO ({bias:+.4f})"

    print(
        f"{'PolarQuant (MSE)':<{col_w[0]}}  "
        f"{mse_rmse:>{col_w[1]}.4f}  "
        f"{mse_bias:>+{col_w[2]}.4f}  "
        f"{mse_var:>{col_w[3]}.6f}  "
        f"{_flag(mse_bias):>{col_w[4]}}"
    )
    print(
        f"{'TurboQuant (MSE+QJL)':<{col_w[0]}}  "
        f"{full_rmse:>{col_w[1]}.4f}  "
        f"{full_bias:>+{col_w[2]}.4f}  "
        f"{full_var:>{col_w[3]}.6f}  "
        f"{_flag(full_bias):>{col_w[4]}}"
    )

    bias_reduction  = abs(mse_bias) / (abs(full_bias) + 1e-10)
    variance_ratio  = full_var / (mse_var + 1e-10)

    print()
    print(f"  Bias reduction  (QJL): {bias_reduction:.1f}x  (ideally >> 1x)")
    print(f"  Variance overhead (QJL): {variance_ratio:.2f}x  (small cost for zero bias)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * 62)
    print("  TurboQuant Benchmark — Bit-Width Comparison")
    print(f"  Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    import turboquantdc
    print(f"  Package: turboquantdc v{turboquantdc.__version__}")
    print("=" * 62)

    section_bit_width_sweep(d=128, n_vectors=2000)
    section_dimension_sweep(bits=3, n_vectors=2000)
    section_kv_cache_compression(d=128, bits=3)
    section_attention_fidelity(bits=3, d=128, n_queries=32)
    section_mse_vs_full(d=128, bits=3, n_vectors=2000)

    print()
    print("=" * 62)
    print("  Benchmark complete.")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
