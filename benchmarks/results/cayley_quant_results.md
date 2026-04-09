# Cayley Full Rotation Benchmark Results

**Model:** Qwen/Qwen2.5-3B-Instruct (BnB 4-bit NF4)
**Bits:** 3
**Head dim:** 128
**Layers tested:** [0, 9, 18, 27, 35]
**Date:** 2026-04-02

## Key Finding

**Cayley learned full rotation (8128 DOF) decisively beats WHT (896 DOF) at 3-bit KV compression.**

- Cosine similarity: 0.9438 -> 0.9738 (53% of gap to perfect closed)
- KL divergence: 0.193 -> 0.081 (58% reduction)
- Top-5 match: 96.2% -> 99.5%
- Calibration: 100 steps, ~230ms per layer

**Cold start (identity init) beats warm start (WHT init).**
This is surprising but explained: WHT has eigenvalues at -1, which the Cayley transform cannot represent exactly. The WHT-init puts parameters near the boundary of the representable space, making optimization harder.

## DOF Comparison

| Rotation | DOF | % of max |
|----------|-----|----------|
| Givens block-diagonal | 64 | 0.8% |
| WHT (butterfly) | 896 | 11.0% |
| Cayley full SO(d) | 8128 | 100% |

## Aggregated Results (3-bit, averaged across 5 test layers)

| Method | Cosine | Top-1 | Top-5 | KL | vs WHT KL |
|--------|--------|-------|-------|-----|-----------|
| WHT + mean-removal | 0.9438 | 88.5% | 96.2% | 0.1931 | baseline |
| Givens learned (64 DOF) | 0.9095 | 84.6% | 96.0% | 0.3597 | +86% worse |
| **Cayley cold (8128 DOF)** | **0.9738** | **93.3%** | **99.5%** | **0.0809** | **-58% better** |
| Cayley WHT-warm (8128 DOF) | 0.9597 | 91.6% | 99.0% | 0.1123 | -42% better |

## Per-Layer Breakdown

| Layer | Method | Cosine | KL |
|-------|--------|--------|-----|
| 0 (embedding) | WHT | 0.7591 | 0.8479 |
| 0 | **Cayley cold** | **0.8900** | **0.3415** |
| 9 (early-mid) | WHT | 0.9976 | 0.0139 |
| 9 | **Cayley cold** | **0.9990** | **0.0059** |
| 18 (middle) | WHT | 0.9899 | 0.0366 |
| 18 | **Cayley cold** | **0.9977** | **0.0139** |
| 27 (late-mid) | WHT | 0.9810 | 0.0421 |
| 27 | **Cayley cold** | **0.9881** | **0.0266** |
| 35 (final) | WHT | 0.9914 | 0.0251 |
| 35 | **Cayley cold** | **0.9943** | **0.0168** |

Cayley wins on EVERY layer. Biggest gains on layer 0 (hardest to quantize).

## Step Sweep (single layer)

### Cayley cold start (identity init)

| Steps | Cosine | KL | Cal.Time |
|-------|--------|----|----------|
| 25 | 0.9970 | 0.0140 | 62ms |
| 50 | 0.9958 | 0.0130 | 119ms |
| **100** | **0.9983** | **0.0095** | **229ms** |
| 200 | 0.9981 | 0.0092 | 420ms |

### Cayley WHT-warm start

| Steps | Cosine | KL | Cal.Time |
|-------|--------|----|----------|
| 25 | 0.9971 | 0.0120 | 57ms |
| 50 | 0.9971 | 0.0120 | 594ms |
| 100 | 0.9967 | 0.0126 | 238ms |
| 200 | 0.9980 | 0.0101 | 453ms |

**50-100 steps is the sweet spot.** Cold start converges to a better optimum.

## Transfer Test (calibrate on prompt A, evaluate on prompt B)

| Layer | WHT (no cal.) | Cayley transfer | Cayley direct | Transfer gap |
|-------|---------------|-----------------|---------------|--------------|
| 0 | 1.7949 | 0.9100 | 0.5301 | 49% |
| 8 | 0.0340 | 0.0235 | 0.0159 | 31% |
| 16 | 0.0313 | 0.0417 | 0.0157 | -33% |
| 35 | 0.0348 | 0.0222 | 0.0169 | 36% |

Transfer works on 3/4 layers (beats WHT baseline). Layer 16 shows prompt-specific overfitting.
Direct calibration (upper bound) is 2x better than WHT on all layers.

## Why Cayley Wins

1. **8128 DOF vs 896**: Full d*(d-1)/2 free parameters can express ALL rotations (not just butterfly patterns)
2. **Gradient-based**: Directly optimizes for attention KL, not a proxy like MSE
3. **Mean-removal synergy**: Combined with the proven shift-invariance trick
4. **Frozen centroids**: Lloyd-Max codebook is mathematically optimal; no need to learn it

## Why Cold Start > Warm Start

WHT rotation has eigenvalues at exactly -1, which lie OUTSIDE the Cayley parameterization's image (Cayley maps only cover SO(d) matrices without eigenvalue -1). The gradient-based WHT-init approximation puts parameters near this boundary, creating a harder optimization landscape. Identity init starts at the center of the parameterization space where gradients are well-conditioned.

## Practical Implications

1. **Per-layer calibration in ~230ms** (100 steps) -- negligible vs model load time
2. **36 layers x 230ms = 8.3s** total calibration for Qwen2.5-3B
3. **Transfer partially works** -- calibrate on one prompt, generalize to others
4. **Storage**: One 128x128 float32 rotation matrix per layer per KV head = 64KB per head
5. **Inference cost**: Same as WHT (one matmul per vector) -- Cayley is only used during calibration
