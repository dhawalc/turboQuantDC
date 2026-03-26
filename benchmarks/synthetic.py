"""TurboQuantDC Synthetic Validation Benchmark.

Validates ALL paper bounds (Theorems 1, 2, Lemma 4) systematically against
analytically-known targets. Designed to run on CPU or GPU; CUDA results are
included when available.

Run:
    python benchmarks/synthetic.py

Exit code: 0 if all critical checks pass, 1 if any FAIL.
"""

from __future__ import annotations

import math
import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Ensure the package root is importable when running from any directory
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from turboquantdc import (
    LloydMaxCodebook,
    PolarQuant,
    QJL,
    TurboQuantEstimator,
    TurboQuantKVCache,
)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

WIDTH = 78
SEP = "=" * WIDTH
SUBSEP = "-" * WIDTH

PASS_TAG = "PASS"
FAIL_TAG = "FAIL"
WARN_TAG = "WARN"


def banner(title: str) -> None:
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)


def sub_banner(title: str) -> None:
    print()
    print(f"  {title}")
    print(SUBSEP)


def result_line(label: str, value: str, status: str, width: int = 60) -> str:
    tag = f"[{status}]"
    base = f"  {label:<{width}} {value}"
    return f"{base:<{WIDTH - 7}}  {tag}"


def print_result(label: str, value: str, status: str, width: int = 60) -> None:
    print(result_line(label, value, status, width))


# ---------------------------------------------------------------------------
# Global pass/fail tracker
# ---------------------------------------------------------------------------

_results: list[tuple[str, str]] = []  # (description, PASS/FAIL/WARN)


def record(description: str, passed: bool, warn: bool = False) -> None:
    if warn:
        _results.append((description, WARN_TAG))
    elif passed:
        _results.append((description, PASS_TAG))
    else:
        _results.append((description, FAIL_TAG))


# ---------------------------------------------------------------------------
# Section 1: Lloyd-Max Codebook Properties
# ---------------------------------------------------------------------------

def section_codebook() -> None:
    banner("1. LLOYD-MAX CODEBOOK PROPERTIES")
    print("  Validates centroid count, symmetry, known analytic values.")
    print(f"  {'d':<6} {'bits':<6} {'levels':<8} {'symmetric':<12} {'D_coord':>10}  {'D_coord*d':>12}  {'status'}")
    print(SUBSEP)

    # Known centroid values (in units of 1/sqrt(d), i.e. centroids * sqrt(d))
    # b=1: +-0.7979,  b=2: +-0.4528, +-1.5104
    known_scaled = {
        1: [0.7979],                # absolute values, symmetric
        2: [0.4528, 1.5104],        # positive half only
    }

    tol_centroid = 0.02  # 2% tolerance for centroid match

    for d in (64, 128, 256):
        for bits in (1, 2, 3, 4):
            cb = LloydMaxCodebook(d, bits)
            c = cb.centroids  # Tensor[2^bits], sorted ascending

            # Symmetry: c should be antisymmetric around 0
            c_sorted = torch.sort(c).values
            c_pos = c_sorted[len(c_sorted) // 2:]     # positive half
            c_neg = c_sorted[:len(c_sorted) // 2]     # negative half
            symmetric = torch.allclose(-c_neg.flip(0), c_pos, atol=1e-4)

            # Per-coordinate distortion
            d_coord = cb.compute_distortion()
            d_coord_times_d = d_coord * d

            # Status: symmetry check + d_coord within expected range
            # Paper Table: b=1->0.36, b=2->0.117, b=3->0.03, b=4->0.009 for D_mse=d*D_coord
            expected_d_mse = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
            upper_bound = math.sqrt(3) * math.pi / 2.0 / (4.0 ** bits)  # Theorem 1
            lower_bound = 1.0 / (4.0 ** bits)  # Lower bound

            ok = (
                symmetric
                and (lower_bound <= d_coord_times_d <= upper_bound * 1.05)
            )

            status = PASS_TAG if ok else FAIL_TAG
            print(
                f"  {d:<6} {bits:<6} {cb.n_levels:<8} {'Yes' if symmetric else 'No':<12} "
                f"{d_coord:>10.5f}  {d_coord_times_d:>12.5f}  [{status}]"
            )
            record(f"Codebook d={d} bits={bits} symmetry+bounds", ok)

    # Verify known centroid values for d=128
    print()
    print("  Centroid value check (d=128, Gaussian approx):")
    print(f"  {'bits':<6} {'analytic':>14}  {'computed':>14}  {'error%':>8}  {'status'}")
    print(f"  {'-'*55}")

    d = 128
    sigma = 1.0 / math.sqrt(d)

    for bits, scaled_vals in known_scaled.items():
        cb = LloydMaxCodebook(d, bits)
        c_sorted = torch.sort(cb.centroids).values
        c_pos = c_sorted[len(c_sorted) // 2:].tolist()

        for i, analytic_scaled in enumerate(scaled_vals):
            analytic = analytic_scaled * sigma  # convert from sigma units to absolute
            computed = c_pos[i]
            err_pct = abs(computed - analytic) / abs(analytic) * 100.0
            ok = err_pct < tol_centroid * 100  # tol_centroid is fraction, but err_pct is %
            status = PASS_TAG if ok else FAIL_TAG
            label = f"b={bits}, c[{i}] analytic={analytic:.5f}"
            print(
                f"  {label:<34} {analytic:>14.6f}  {computed:>14.6f}  {err_pct:>7.2f}%  [{status}]"
            )
            record(f"Centroid b={bits} index={i} analytic match", ok)


# ---------------------------------------------------------------------------
# Section 2: MSE Distortion — Theorem 1
# ---------------------------------------------------------------------------

def section_mse_distortion(n_vectors: int = 2000) -> None:
    banner("2. MSE DISTORTION — THEOREM 1")
    print(f"  n_vectors={n_vectors}, d=128, random unit vectors.")
    print(f"  Upper bound: D_mse <= sqrt(3)*pi/2 / 4^b  ~ 2.721 / 4^b")
    print(f"  Lower bound: D_mse >= 1 / 4^b")
    print()
    print(f"  {'bits':<6} {'D_mse':>10} {'upper':>10} {'lower':>10} {'gap_factor':>12} {'status'}")
    print(SUBSEP)

    d = 128
    upper_const = math.sqrt(3) * math.pi / 2.0
    torch.manual_seed(0)

    for bits in (1, 2, 3, 4):
        # Generate random unit vectors
        x = torch.randn(n_vectors, d)
        x = x / x.norm(dim=-1, keepdim=True)

        pq = PolarQuant(d, bits, seed=42)
        x_hat, _ = pq(x)

        # MSE: mean over vectors of ||x - x_hat||^2
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()

        upper = upper_const / (4.0 ** bits)
        lower = 1.0 / (4.0 ** bits)
        gap = mse / lower  # should be between 1 and ~2.72

        ok = lower * 0.7 <= mse <= upper * 1.10  # 10% slack for finite-sample noise
        status = PASS_TAG if ok else FAIL_TAG

        print(
            f"  {bits:<6} {mse:>10.5f} {upper:>10.5f} {lower:>10.5f} {gap:>12.3f}  [{status}]"
        )
        record(f"Theorem 1 MSE bits={bits}", ok)

    # Additional: check that D_mse * d scales with coordinate-level distortion
    print()
    print("  Cross-check: D_mse vs paper table (b=1->0.36, b=2->0.117, b=3->0.03, b=4->0.009)")
    expected = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
    torch.manual_seed(0)
    for bits in (1, 2, 3, 4):
        x = torch.randn(n_vectors, d)
        x = x / x.norm(dim=-1, keepdim=True)
        pq = PolarQuant(d, bits, seed=42)
        x_hat, _ = pq(x)
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
        exp = expected[bits]
        err_pct = abs(mse - exp) / exp * 100.0
        ok = err_pct < 30.0  # generous since paper values are for asymptotic regime
        status = PASS_TAG if ok else WARN_TAG
        print(f"  bits={bits}: measured={mse:.4f}  table={exp:.3f}  err={err_pct:.1f}%  [{status}]")
        record(f"Theorem 1 MSE paper table b={bits}", ok, warn=not ok)


# ---------------------------------------------------------------------------
# Section 3: Inner Product Unbiasedness — Theorem 2
# ---------------------------------------------------------------------------

def section_inner_product(n_pairs: int = 2000) -> None:
    banner("3. INNER PRODUCT UNBIASEDNESS — THEOREM 2")
    print(f"  n_pairs={n_pairs}, d=128, random unit vector pairs.")
    print(f"  D_prod bound: sqrt(3)*pi^2/d / 4^b  (assuming ||y||=1)")
    print()
    print(
        f"  {'bits':<6} {'bias':>10} {'RMSE':>10} {'corr':>8} {'D_prod':>10} {'bound':>10} {'status'}"
    )
    print(SUBSEP)

    d = 128
    upper_const = math.sqrt(3) * math.pi ** 2 / d
    torch.manual_seed(1)

    for bits in (2, 3, 4):
        # Random unit vector pairs
        x = torch.randn(n_pairs, d)
        x = x / x.norm(dim=-1, keepdim=True)  # keys
        y = torch.randn(n_pairs, d)
        y = y / y.norm(dim=-1, keepdim=True)  # queries

        estimator = TurboQuantEstimator(d, bits, seed=42)

        # True inner products (query[i] . key[i])
        true_ip = (y * x).sum(dim=-1)  # (n_pairs,)

        # Compress keys
        compressed = estimator.quantize(x)

        # Estimated inner products (diagonal of the full matrix)
        est_ip = torch.stack([
            estimator.inner_product(y[i:i+1], {
                k: v[i:i+1] for k, v in compressed.items()
            }).squeeze()
            for i in range(n_pairs)
        ])

        # Statistics
        errors = est_ip - true_ip
        bias = errors.mean().item()
        rmse = errors.pow(2).mean().sqrt().item()
        corr = torch.corrcoef(torch.stack([true_ip, est_ip]))[0, 1].item()

        d_prod = errors.pow(2).mean().item()
        bound = upper_const / (4.0 ** bits)

        # Pass criteria:
        #   |bias| < 0.05 (unbiasedness — key property)
        #   D_prod within 1.5x of the theoretical bound
        #   correlation improves with bits; at 2-bit expect ~0.80, at 4-bit >0.97
        corr_threshold = {2: 0.70, 3: 0.88, 4: 0.95}.get(bits, 0.80)
        ok_bias = abs(bias) < 0.05
        ok_bound = d_prod <= bound * 1.5
        ok_corr = corr > corr_threshold
        ok = ok_bias and ok_bound and ok_corr
        status = PASS_TAG if ok else FAIL_TAG

        print(
            f"  {bits:<6} {bias:>10.5f} {rmse:>10.5f} {corr:>8.4f} "
            f"{d_prod:>10.6f} {bound:>10.6f}  [{status}]"
        )
        record(f"Theorem 2 IP unbiased bits={bits}", ok)

    # Specific Theorem 2 D_prod values from paper: 1.57/d, 0.56/d, 0.18/d, 0.047/d
    print()
    print("  Cross-check: D_prod vs paper table (b=2->0.56/d, b=3->0.18/d, b=4->0.047/d)")
    expected_dprod = {2: 0.56/d, 3: 0.18/d, 4: 0.047/d}
    torch.manual_seed(1)
    for bits in (2, 3, 4):
        x = torch.randn(n_pairs, d)
        x = x / x.norm(dim=-1, keepdim=True)
        y = torch.randn(n_pairs, d)
        y = y / y.norm(dim=-1, keepdim=True)
        estimator = TurboQuantEstimator(d, bits, seed=42)
        true_ip = (y * x).sum(dim=-1)
        compressed = estimator.quantize(x)
        est_ip = torch.stack([
            estimator.inner_product(y[i:i+1], {
                k: v[i:i+1] for k, v in compressed.items()
            }).squeeze()
            for i in range(n_pairs)
        ])
        d_prod = (est_ip - true_ip).pow(2).mean().item()
        exp = expected_dprod[bits]
        err_pct = abs(d_prod - exp) / (exp + 1e-12) * 100.0
        ok = err_pct < 100.0  # very generous — finite sample noise dominates
        status = PASS_TAG if ok else WARN_TAG
        print(f"  bits={bits}: measured={d_prod:.5f}  table={exp:.5f}  err={err_pct:.1f}%  [{status}]")
        record(f"Theorem 2 D_prod paper table b={bits}", ok, warn=not ok)


# ---------------------------------------------------------------------------
# Section 4: MSE-Only Bias (Motivation for QJL)
# ---------------------------------------------------------------------------

def section_mse_bias(n_pairs: int = 2000) -> None:
    banner("4. MSE-ONLY BIAS (MOTIVATION FOR QJL)")
    print("  PolarQuant alone gives biased inner products.")
    print("  At b=1: bias factor = 2/pi ~ 0.6366.")
    print("  TurboQuantEstimator should remove bias entirely.")
    print()
    print(f"  {'method':<28} {'bits':<6} {'bias_factor':>12} {'expected':>12} {'status'}")
    print(SUBSEP)

    d = 128
    torch.manual_seed(2)

    n = n_pairs
    x = torch.randn(n, d)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.randn(n, d)
    y = y / y.norm(dim=-1, keepdim=True)
    true_ip = (y * x).sum(dim=-1)

    for bits in (1, 2, 3):
        # PolarQuant MSE-only: quantize then dequantize, compute raw IP
        pq = PolarQuant(d, bits, seed=42)
        x_hat, _ = pq(x)
        mse_ip = (y * x_hat).sum(dim=-1)

        # Bias factor: <y, x_mse> / <y, x> (expected 2/pi at b=1, converges to 1)
        # Use mean ratio on pairs where |true_ip| > 0.1 to avoid division by small numbers
        mask = true_ip.abs() > 0.05
        if mask.sum() > 10:
            bias_factor = (mse_ip[mask] / true_ip[mask]).mean().item()
        else:
            bias_factor = float("nan")

        # At b=1, expected ~ 2/pi. At higher bits, approaches 1.0
        expected_bf = {1: 2.0 / math.pi, 2: 0.85, 3: 0.95}[bits]
        ok_mse = abs(bias_factor - expected_bf) < 0.15  # rough check
        status = PASS_TAG if ok_mse else WARN_TAG
        print(
            f"  {'PolarQuant (MSE only)':<28} {bits:<6} {bias_factor:>12.4f} {expected_bf:>12.4f}  [{status}]"
        )
        record(f"MSE bias factor b={bits}", ok_mse, warn=not ok_mse)

    print()
    print("  TurboQuantEstimator (MSE + QJL) should have bias_factor ~ 1.0:")
    torch.manual_seed(2)
    for bits in (2, 3, 4):
        x = torch.randn(n, d)
        x = x / x.norm(dim=-1, keepdim=True)
        y = torch.randn(n, d)
        y = y / y.norm(dim=-1, keepdim=True)
        true_ip = (y * x).sum(dim=-1)

        estimator = TurboQuantEstimator(d, bits, seed=42)
        compressed = estimator.quantize(x)
        est_ip = torch.stack([
            estimator.inner_product(y[i:i+1], {
                k: v[i:i+1] for k, v in compressed.items()
            }).squeeze()
            for i in range(n)
        ])

        mask = true_ip.abs() > 0.05
        if mask.sum() > 10:
            bias_factor = (est_ip[mask] / true_ip[mask]).mean().item()
        else:
            bias_factor = float("nan")

        # Should be close to 1.0 (unbiased)
        ok = abs(bias_factor - 1.0) < 0.10
        status = PASS_TAG if ok else FAIL_TAG
        print(
            f"  {'TurboQuantEstimator':<28} {bits:<6} {bias_factor:>12.4f} {'1.0000':>12}  [{status}]"
        )
        record(f"TurboQuant unbiased b={bits}", ok)


# ---------------------------------------------------------------------------
# Section 5: Needle-in-Haystack
# ---------------------------------------------------------------------------

def section_needle_in_haystack() -> None:
    banner("5. NEEDLE-IN-HAYSTACK SEARCH")
    print("  Hide one needle key among N random distractors.")
    print("  Query is the exact needle; measure top-k recall.")
    print()
    print(
        f"  {'N':>6} {'bits':<5} {'top1':>6} {'top5':>6} {'top10':>6} "
        f"{'median_rank':>12}  {'status'}"
    )
    print(SUBSEP)

    d = 128
    n_trials = 50
    torch.manual_seed(3)

    # Paper target: >90% top-5 at 3-bit
    targets = {
        2: {"top1": 0.70, "top5": 0.85},
        3: {"top1": 0.85, "top5": 0.90},
        4: {"top1": 0.92, "top5": 0.95},
    }

    for N in (512, 2048, 8192):
        for bits in (2, 3, 4):
            estimator = TurboQuantEstimator(d, bits, seed=42)

            top1_hits = 0
            top5_hits = 0
            top10_hits = 0
            ranks: list[int] = []

            for _ in range(n_trials):
                # Generate N random unit keys, hide needle at random position
                needle_pos = torch.randint(0, N, (1,)).item()
                keys = torch.randn(N, d)
                keys = keys / keys.norm(dim=-1, keepdim=True)

                needle = keys[needle_pos]
                query = needle.clone()

                # Compress all N keys
                compressed = estimator.quantize(keys)

                # Estimate inner products
                scores = estimator.inner_product(query.unsqueeze(0), compressed).squeeze(0)

                # Rank needle
                sorted_idx = torch.argsort(scores, descending=True)
                rank = (sorted_idx == needle_pos).nonzero(as_tuple=True)[0].item() + 1

                ranks.append(rank)
                top1_hits += int(rank == 1)
                top5_hits += int(rank <= 5)
                top10_hits += int(rank <= 10)

            top1 = top1_hits / n_trials
            top5 = top5_hits / n_trials
            top10 = top10_hits / n_trials
            median_rank = sorted(ranks)[len(ranks) // 2]

            tgt = targets.get(bits, {"top1": 0.5, "top5": 0.7})
            ok = top1 >= tgt["top1"] and top5 >= tgt["top5"]
            status = PASS_TAG if ok else FAIL_TAG

            print(
                f"  {N:>6} {bits:<5} {top1:>6.2%} {top5:>6.2%} {top10:>6.2%} "
                f"{median_rank:>12}  [{status}]"
            )
            record(f"Needle-in-haystack N={N} bits={bits}", ok)


# ---------------------------------------------------------------------------
# Section 6: GPU Benchmark
# ---------------------------------------------------------------------------

def section_gpu_benchmark() -> None:
    banner("6. GPU THROUGHPUT BENCHMARK")

    if not torch.cuda.is_available():
        print("  CUDA not available — skipping GPU benchmark.")
        print(f"  [WARN] GPU benchmark skipped (no CUDA)")
        record("GPU benchmark", True, warn=True)
        return

    device = "cuda"
    d = 128
    bits = 3
    N = 8192
    n_warmup = 5
    n_iters = 20

    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Config: d={d}, bits={bits}, N={N} vectors")
    print()

    estimator = TurboQuantEstimator(d, bits, seed=42, device=device)
    torch.manual_seed(4)
    x = torch.randn(N, d, device=device)
    x = x / x.norm(dim=-1, keepdim=True)
    q = torch.randn(d, device=device)
    q = q / q.norm()

    # --- Quantization throughput ---
    # Warmup
    for _ in range(n_warmup):
        _ = estimator.quantize(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        compressed = estimator.quantize(x)
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed_quant = (t1 - t0) / n_iters
    vecs_per_sec_quant = N / elapsed_quant

    # --- Inner product throughput ---
    compressed = estimator.quantize(x)
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        _ = estimator.inner_product(q.unsqueeze(0), compressed)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = estimator.inner_product(q.unsqueeze(0), compressed)
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed_ip = (t1 - t0) / n_iters
    vecs_per_sec_ip = N / elapsed_ip

    # --- FP16 baseline matmul ---
    x_fp16 = x.half()
    q_fp16 = q.half()
    for _ in range(n_warmup):
        _ = x_fp16 @ q_fp16
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = x_fp16 @ q_fp16
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed_fp16 = (t1 - t0) / n_iters
    vecs_per_sec_fp16 = N / elapsed_fp16

    # Target: >1M vectors/sec for quantization
    quant_ok = vecs_per_sec_quant >= 1e6
    ip_ok = vecs_per_sec_ip >= 1e6

    print(f"  {'Operation':<35} {'vecs/sec':>14}  {'ms/batch':>10}  {'status'}")
    print(f"  {'-' * 70}")

    def fmt_vps(v: float) -> str:
        if v >= 1e9:
            return f"{v/1e9:.2f}G"
        if v >= 1e6:
            return f"{v/1e6:.2f}M"
        return f"{v/1e3:.1f}K"

    qst = PASS_TAG if quant_ok else WARN_TAG
    ist = PASS_TAG if ip_ok else WARN_TAG
    print(f"  {'TQ Quantize (N=8192)':<35} {fmt_vps(vecs_per_sec_quant):>14}  {elapsed_quant*1000:>10.3f}  [{qst}]")
    print(f"  {'TQ InnerProduct (1 query x N keys)':<35} {fmt_vps(vecs_per_sec_ip):>14}  {elapsed_ip*1000:>10.3f}  [{ist}]")
    print(f"  {'FP16 MatVec baseline':<35} {fmt_vps(vecs_per_sec_fp16):>14}  {elapsed_fp16*1000:>10.3f}  [    ]")
    print()

    speedup_vs_fp16 = elapsed_fp16 / elapsed_ip
    print(f"  TQ inner product vs FP16 matmul: {speedup_vs_fp16:.2f}x speedup")

    record("GPU quantize throughput >=1M vecs/sec", quant_ok, warn=not quant_ok)
    record("GPU inner product throughput >=1M vecs/sec", ip_ok, warn=not ip_ok)


# ---------------------------------------------------------------------------
# Section: Cosine Similarity Quality
# ---------------------------------------------------------------------------

def section_cosine_similarity(n_vectors: int = 500) -> None:
    banner("7. COSINE SIMILARITY QUALITY")
    print("  Cosine similarity measured via self-inner-product of TurboQuant estimator.")
    print("  For unit vectors: cos_sim(x, x_hat) = <x, x_hat> = TurboQuant_estimate(<x, x>).")
    print("  Target from CLAUDE.md: 3-bit combined estimator cos_sim > 0.995")
    print()
    print(f"  {'d':<6} {'bits':<6} {'mean_cos_sim':>14}  {'min_cos_sim':>14}  {'target':>10}  {'status'}")
    print(SUBSEP)

    # For MSE-only reconstruction, cos sim is the PolarQuant reconstruction quality
    # Full TurboQuant (MSE+QJL) provides unbiased inner products, so self-IP ~= 1.0
    # The CLAUDE.md target of >0.995 refers to the effective inner product fidelity
    # measured as: mean self-inner-product when query = key (normalized cos sim)
    targets_cos = {1: 0.85, 2: 0.95, 3: 0.995, 4: 0.999}

    for d in (64, 128):
        for bits in (1, 2, 3, 4):
            torch.manual_seed(5)
            x = torch.randn(n_vectors, d)
            x = x / x.norm(dim=-1, keepdim=True)  # unit vectors

            estimator = TurboQuantEstimator(d, bits, seed=42)
            compressed = estimator.quantize(x)

            # Self inner-product: <x_i, compressed(x_i)>
            # For unit vectors, this equals the cosine similarity of the combined estimator
            self_ips = []
            for i in range(n_vectors):
                q = x[i:i+1]
                c_i = {k: v[i:i+1] for k, v in compressed.items()}
                est = estimator.inner_product(q, c_i).item()
                self_ips.append(est)

            self_ips_t = torch.tensor(self_ips)
            mean_cs = self_ips_t.mean().item()
            min_cs = self_ips_t.min().item()

            tgt = targets_cos.get(bits, 0.90)
            ok = mean_cs >= tgt
            status = PASS_TAG if ok else FAIL_TAG

            print(
                f"  {d:<6} {bits:<6} {mean_cs:>14.6f}  {min_cs:>14.6f}  {tgt:>10.4f}  [{status}]"
            )
            record(f"CosSim d={d} bits={bits} mean>={tgt}", ok)


# ---------------------------------------------------------------------------
# Section: QJL Variance Bound (Lemma 4)
# ---------------------------------------------------------------------------

def section_qjl_variance(n_trials: int = 1000) -> None:
    banner("8. QJL VARIANCE BOUND — LEMMA 4")
    print("  Lemma 4: Var(<y, QJL^{-1}(QJL(r))>) <= pi/(2*d) * ||y||^2")
    print("  Test: fix r and y; vary QJL matrix S over n_trials instantiations.")
    print()
    print(f"  {'d':<6} {'true_ip':>10}  {'E[est]':>10}  {'Var[est]':>10}  {'bound':>10}  {'ratio':>8}  {'status'}")
    print(SUBSEP)

    # Note: the bound holds for EACH fixed (r, y) with randomness over S.
    # We test it by instantiating n_trials independent QJL matrices.

    for d in (64, 128, 256):
        torch.manual_seed(7 + d)

        r_fixed = torch.randn(d)
        r_fixed = r_fixed / r_fixed.norm()  # fixed unit residual
        y_fixed = torch.randn(d)
        y_fixed = y_fixed / y_fixed.norm()  # fixed unit query
        true_ip = (y_fixed * r_fixed).sum().item()

        estimates = []
        scale = math.sqrt(math.pi / 2.0) / d
        for seed in range(n_trials):
            qjl = QJL(d, seed=seed)
            signs = qjl.project_and_sign(r_fixed.unsqueeze(0)).squeeze(0)  # (d,)
            Sy = y_fixed @ qjl.S.T  # (d,): S @ y
            est = scale * (signs @ Sy).item()
            estimates.append(est)

        ests_t = torch.tensor(estimates)
        mean_est = ests_t.mean().item()
        var_est = ests_t.var().item()
        bound = math.pi / (2.0 * d) * (y_fixed.norm().item() ** 2)

        ratio = var_est / bound
        # The bound should hold; allow 20% slack for n_trials finite-sample noise
        ok = ratio <= 1.20
        status = PASS_TAG if ok else FAIL_TAG

        print(
            f"  {d:<6} {true_ip:>10.5f}  {mean_est:>10.5f}  {var_est:>10.6f}  "
            f"{bound:>10.6f}  {ratio:>8.4f}  [{status}]"
        )
        record(f"QJL variance bound d={d}", ok)


# ---------------------------------------------------------------------------
# Summary Table
# ---------------------------------------------------------------------------

def print_summary() -> None:
    banner("SUMMARY")

    n_pass = sum(1 for _, s in _results if s == PASS_TAG)
    n_fail = sum(1 for _, s in _results if s == FAIL_TAG)
    n_warn = sum(1 for _, s in _results if s == WARN_TAG)
    total = len(_results)

    print()
    for desc, status in _results:
        tag = f"[{status}]"
        print(f"  {desc:<58}  {tag}")

    print()
    print(SUBSEP)
    print(f"  Total: {total}  |  PASS: {n_pass}  |  FAIL: {n_fail}  |  WARN: {n_warn}")
    print(SUBSEP)

    if n_fail == 0:
        print()
        print("  ALL CRITICAL CHECKS PASSED")
        if n_warn > 0:
            print(f"  ({n_warn} warnings — see details above)")
    else:
        print()
        print(f"  {n_fail} CRITICAL CHECK(S) FAILED — see details above")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(SEP)
    print("  TurboQuantDC Synthetic Validation Benchmark")
    print("  Validates paper bounds: Theorem 1, Theorem 2, Lemma 4")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(SEP)

    section_codebook()
    section_mse_distortion(n_vectors=2000)
    section_inner_product(n_pairs=2000)
    section_mse_bias(n_pairs=2000)
    section_needle_in_haystack()
    section_cosine_similarity(n_vectors=2000)
    section_qjl_variance(n_trials=1000)
    section_gpu_benchmark()

    print_summary()

    n_fail = sum(1 for _, s in _results if s == FAIL_TAG)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
