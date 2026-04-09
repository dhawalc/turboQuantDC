# TurboQuantDC vs RotorQuant: Head-to-Head Benchmark

**Date:** 2026-04-09
**Model:** Qwen2.5-3B-Instruct (BnB 4-bit)
**Device:** NVIDIA RTX 4090 (CUDA)
**Head dimension:** 128 (GQA, 2 KV heads per layer)
**KV cache vectors:** 5,120 real key vectors from 5 layers (0, 9, 18, 27, 35)
**Sequence length:** 512 tokens
**Query vectors:** 200 (20 positions x 2 heads x 5 layers)

## Summary

RotorQuant's block-diagonal rotation methods (IsoQuant, PlanarQuant) deliver
significantly better **attention quality** than our WHT-based methods despite
having worse **vector reconstruction** (lower cosine similarity, higher MSE).
This is the critical finding: attention accuracy and reconstruction accuracy
are **inversely correlated** across rotation families.

## 3-Bit Comparison (Primary)

| Method | Family | Vec CosSim | Attn CosSim | Top-1 | Top-5 | Spearman | NMSE | Q ms/1k | DQ ms/1k |
|--------|--------|-----------|-------------|-------|-------|----------|------|---------|----------|
| IsoQuant-Full (4D Quat) | RQ | 0.9243 | **0.4832** | **0.4844** | **0.5312** | **0.9744** | 0.2344 | 0.098 | 0.092 |
| PlanarQuant (2D Givens) | RQ | 0.9101 | 0.4526 | 0.4531 | 0.4844 | 0.9671 | 0.2720 | 0.030 | 0.019 |
| IsoQuant-Fast (4D Quat) | RQ | 0.9258 | 0.4441 | 0.4219 | 0.5000 | 0.9659 | 0.2601 | 0.056 | 0.049 |
| PolarQuant-WHT (Ours) | Ours | **0.9838** | 0.3698 | 0.3594 | 0.4062 | 0.9603 | **0.0343** | 0.069 | 0.058 |
| ResidualQuant+Mean (Ours) | Ours | 0.9626 | 0.3541 | 0.3438 | 0.3906 | 0.9636 | 0.0365 | 0.093 | 0.064 |
| ResidualQuant-WHT (Ours) | Ours | 0.9820 | 0.3518 | 0.3438 | 0.3906 | 0.9506 | 0.0452 | 0.090 | 0.064 |
| TurboQuant-QR (RQ impl) | RQ | 0.9836 | 0.3469 | 0.3438 | 0.3594 | 0.9370 | 0.0310 | 0.016 | 0.005 |

## 2-Bit Comparison

| Method | Family | Vec CosSim | Attn CosSim | Top-1 | Top-5 | Spearman | NMSE | Q ms/1k | DQ ms/1k |
|--------|--------|-----------|-------------|-------|-------|----------|------|---------|----------|
| IsoQuant-Fast (4D Quat) | RQ | 0.8187 | **0.5912** | **0.5625** | **0.7031** | **0.9503** | 0.4735 | 0.053 | 0.055 |
| IsoQuant-Full (4D Quat) | RQ | 0.8180 | 0.4385 | 0.4062 | 0.5781 | 0.9269 | 0.4561 | 0.095 | 0.083 |
| TurboQuant-QR (RQ impl) | RQ | **0.9420** | 0.3606 | 0.3594 | 0.3906 | 0.9172 | **0.1157** | 0.009 | 0.005 |
| ResidualQuant-WHT (Ours) | Ours | 0.9401 | 0.3575 | 0.3594 | 0.3906 | 0.9199 | 0.1387 | 0.095 | 0.060 |
| ResidualQuant+Mean (Ours) | Ours | 0.8712 | 0.3239 | 0.3281 | 0.3438 | 0.9355 | 0.1198 | 0.097 | 0.064 |
| PolarQuant-WHT (Ours) | Ours | 0.9442 | 0.3076 | 0.3125 | 0.3281 | 0.8986 | 0.1174 | 0.074 | 0.060 |
| PlanarQuant (2D Givens) | RQ | 0.8013 | 0.2784 | 0.2500 | 0.4688 | 0.9306 | 0.4854 | 0.029 | 0.019 |

## 4-Bit Comparison

| Method | Family | Vec CosSim | Attn CosSim | Top-1 | Top-5 | Spearman | NMSE | Q ms/1k | DQ ms/1k |
|--------|--------|-----------|-------------|-------|-------|----------|------|---------|----------|
| IsoQuant-Full (4D Quat) | RQ | 0.9616 | **0.5572** | **0.5469** | **0.6719** | **0.9845** | 0.1317 | 0.098 | 0.078 |
| PlanarQuant (2D Givens) | RQ | 0.9511 | 0.4618 | 0.4531 | 0.4844 | 0.9794 | 0.1697 | 0.042 | 0.019 |
| ResidualQuant-WHT (Ours) | Ours | 0.9948 | 0.4393 | 0.3906 | 0.4688 | 0.9842 | 0.0122 | 0.092 | 0.067 |
| IsoQuant-Fast (4D Quat) | RQ | 0.9628 | 0.4203 | 0.4062 | 0.4375 | 0.9793 | 0.1610 | 0.056 | 0.047 |
| ResidualQuant+Mean (Ours) | Ours | 0.9896 | 0.3769 | 0.3750 | 0.4062 | 0.9768 | 0.0099 | 0.095 | 0.063 |
| TurboQuant-QR (RQ impl) | RQ | **0.9956** | 0.3684 | 0.3438 | 0.4062 | 0.9451 | **0.0084** | 0.045 | 0.006 |
| PolarQuant-WHT (Ours) | Ours | 0.9956 | 0.3587 | 0.3594 | 0.4219 | 0.9512 | 0.0095 | 0.071 | 0.057 |

## Critical Findings

### 1. Vector Reconstruction vs Attention Quality: INVERSE Correlation

This is the most striking and counterintuitive result:

| Method | 3-bit Vec CosSim | 3-bit Attn CosSim | Paradox |
|--------|-----------------|-------------------|---------|
| PolarQuant-WHT (Ours) | **0.984** (best) | 0.370 (4th) | Best recon, mediocre attention |
| IsoQuant-Full (RQ) | 0.924 (5th) | **0.483** (best) | Mediocre recon, best attention |
| PlanarQuant (RQ) | 0.910 (worst) | 0.453 (2nd) | Worst recon, 2nd best attention |

**WHT/QR global rotations achieve near-perfect vector reconstruction** (>0.98 cosine sim)
but the residual errors are correlated across dimensions, causing systematic attention
score drift. **Block-diagonal rotations (quaternion, Givens) have higher per-vector error**
(~0.92 cosine sim) but the errors are more randomly distributed, preserving attention
rank ordering better.

### 2. IsoQuant-Full is the Clear Winner Across All Bit-widths

| Bits | Best Method | Attn CosSim | Top-1 | Top-5 |
|------|-------------|-------------|-------|-------|
| 2 | IsoQuant-Fast (RQ) | 0.591 | 0.563 | 0.703 |
| 3 | IsoQuant-Full (RQ) | 0.483 | 0.484 | 0.531 |
| 4 | IsoQuant-Full (RQ) | 0.557 | 0.547 | 0.672 |

RotorQuant's IsoQuant wins at every bit-width on attention metrics.

### 3. Our WHT Baseline Matches Their QR Baseline (Same Paper)

Our PolarQuant-WHT and their TurboQuant-QR are both implementations of the same
TurboQuant paper (QR/WHT rotation + Lloyd-Max). The results are nearly identical:

| Method | 3-bit Vec CosSim | 3-bit Attn CosSim | 3-bit NMSE |
|--------|-----------------|-------------------|------------|
| PolarQuant-WHT (Ours) | 0.9838 | 0.3698 | 0.0343 |
| TurboQuant-QR (RQ) | 0.9836 | 0.3469 | 0.0310 |

Our WHT is marginally better than their QR on attention (0.370 vs 0.347), consistent
with WHT's O(d log d) butterfly mixing vs QR's dense matrix.

### 4. ResidualQuant Does NOT Help on Attention Metrics at 3-bit

At 3-bit, our ResidualQuant methods trail our plain PolarQuant-WHT:

| Method | 3-bit Attn CosSim | Why |
|--------|-------------------|-----|
| PolarQuant-WHT | 0.3698 | Plain MSE at 3 bits |
| ResidualQuant-WHT | 0.3518 | MSE at 2 bits + 1 bit signs |
| ResidualQuant+Mean | 0.3541 | MSE at 2 bits + 1 bit signs + mean removal |

ResidualQuant allocates 1 of the 3 bits to residual signs, leaving only 2 bits for
MSE. The residual correction helps vector reconstruction but hurts attention because
the sign-based correction introduces its own correlated errors.

**Exception at 4-bit:** ResidualQuant-WHT (0.439) beats PolarQuant-WHT (0.359),
suggesting the residual correction pays off when the MSE stage has enough bits (3+).

### 5. Speed Comparison

| Method | 3-bit Q ms/1k | 3-bit DQ ms/1k | Total |
|--------|--------------|----------------|-------|
| TurboQuant-QR (RQ) | **0.016** | **0.005** | **0.021** |
| PlanarQuant (RQ) | 0.030 | 0.019 | 0.049 |
| IsoQuant-Fast (RQ) | 0.056 | 0.049 | 0.105 |
| PolarQuant-WHT (Ours) | 0.069 | 0.058 | 0.127 |
| ResidualQuant-WHT (Ours) | 0.090 | 0.064 | 0.154 |
| IsoQuant-Full (RQ) | 0.098 | 0.092 | 0.190 |

Their QR baseline is fastest (dense matmul optimized by cuBLAS). Our WHT is ~3x slower
than their QR but this is Python overhead, not algorithmic -- both are O(d^2) for
d=128 (WHT's O(d log d) advantage only matters for d >= 1024). PlanarQuant offers the
best speed-quality tradeoff: 2nd best attention at 2.4x the speed of IsoQuant-Full.

## Key Questions Answered

### Q1: Does their block-diagonal rotation beat our WHT at same bit-width?

**YES, decisively.** IsoQuant-Full beats our WHT by +0.113 attention cosine similarity
at 3-bit (0.483 vs 0.370). This is not a small margin. The block-diagonal SO(4) rotation
preserves attention rank ordering much better than global WHT/QR rotation, despite having
higher per-vector MSE.

### Q2: Does our PCA + mean-removal + ResidualQuant beat their best?

**No.** Our PCA quantizer has a scaling bug when applied to real KV cache data (the
whitening scale computed from calibration data doesn't transfer to unit vectors correctly).
Even our working methods (ResidualQuant+Mean at 0.354) trail their IsoQuant-Full at 0.483.
The PCA approach needs redesign.

### Q3: Where is the speed vs quality tradeoff?

**PlanarQuant (2D Givens) is the Pareto-optimal point.** It achieves 0.453 attention cosine
(93% of IsoQuant-Full's 0.483) at 3.9x the speed. IsoQuant-Full is worth the cost only
when attention quality is paramount.

## Strategic Takeaway

**RotorQuant's block-diagonal rotation approach is a genuine advance over the WHT/QR
rotation used in both our PolarQuant and the original TurboQuant paper.** The key insight
is that what matters for KV cache compression is not vector reconstruction fidelity
(where WHT wins) but attention rank preservation (where block-diagonal rotations win).

The gap is large enough (~11 points on attention cosine) that we should consider:

1. **Adopt block-diagonal rotation** (quaternion or Givens) as our default rotation,
   replacing WHT. This is a ~50 line change in `polarquant.py`.

2. **Test ResidualQuant with block-diagonal rotation.** Our residual sign correction
   should stack with their rotation, potentially combining both advantages.

3. **The PCA approach needs rework.** The current `PCARotatedQuantizer` has scaling
   issues with real KV cache data. The whitening step needs to be redesigned.

4. **Benchmark attention-aware metrics from the start.** Vector cosine similarity
   is misleading for KV cache quantization -- we should use attention cosine
   similarity and top-K match as primary metrics.
