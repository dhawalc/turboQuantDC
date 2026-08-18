# RotorQuant vs TurboQuantDC: Comprehensive Head-to-Head

**Date:** 2026-04-15
**GPU:** NVIDIA RTX 4090 (24GB)
**Dataset:** wikitext-2 test (8,192 tokens, window=512, stride=256)
**Models:** Qwen2.5-3B (2 KV heads), 7B (4 KV heads), 14B (8 KV heads) — all BnB 4-bit NF4
**RotorQuant:** scrya-com/rotorquant (IsoQuant, PlanarQuant)
**TurboQuantDC:** dhawalc/turboQuantDC (PolarQuant-WHT, Givens)

## TL;DR

**WHT rotation is universally the best for PPL.** At every model size and bit-width tested,
PolarQuant-WHT (with or without mean-removal) beats all block-diagonal rotations on perplexity.

**Mean-removal is critical for low-KV-head models** (2-4 heads) where it transforms catastrophic
PPL into near-lossless. On higher-KV-head models (8+ heads), mean-removal is nearly neutral for
WHT and actually HURTS block-diagonal rotations.

**Attention cosine similarity does not predict PPL.** The method with the best attention cosine
often has the worst PPL, and vice versa.

---

## Results by Model

### Qwen2.5-14B-Instruct (8 KV heads, FP16 PPL: 4.94)

#### 3-bit

| Rank | Method | Family | PPL | vs FP16 |
|------|--------|--------|-----|---------|
| 1 | **PolarQuant-WHT** | ours | **5.54** | **+12.1%** |
| 2 | PolarQuant-WHT+Mean | hybrid | 5.58 | +12.9% |
| 3 | IsoQuant-Full | rotorquant | 18.39 | +272% |
| 4 | IsoQuant-Full+Mean | hybrid | 30.87 | +525% |
| 5 | PlanarQuant | rotorquant | 40.66 | +723% |
| 6 | Givens+Mean | ours | 49.95 | +911% |
| 7 | PlanarQuant+Mean | hybrid | 50.40 | +920% |

#### 4-bit

| Rank | Method | Family | PPL | vs FP16 |
|------|--------|--------|-----|---------|
| 1 | **PolarQuant-WHT+Mean** | hybrid | **5.07** | **+2.6%** |
| 2 | PolarQuant-WHT | ours | 5.15 | +4.2% |
| 3 | IsoQuant-Full | rotorquant | 10.12 | +105% |
| 4 | IsoQuant-Full+Mean | hybrid | 13.21 | +167% |
| 5 | PlanarQuant | rotorquant | 18.59 | +276% |
| 6 | Givens+Mean | ours | 37.84 | +666% |
| 7 | PlanarQuant+Mean | hybrid | 38.24 | +674% |

### Qwen2.5-7B-Instruct (4 KV heads, FP16 PPL: 8.43)

#### 3-bit

| Rank | Method | Family | PPL | vs FP16 |
|------|--------|--------|-----|---------|
| 1 | **PolarQuant-WHT+Mean** | hybrid | **9.06** | **+7.5%** |
| 2 | IsoQuant-Full+Mean | hybrid | 14.50 | +72% |
| 3 | Givens+Mean | ours | 15.57 | +85% |
| 4 | PlanarQuant+Mean | hybrid | 15.65 | +86% |
| 5 | IsoQuant-Full | rotorquant | 83.92 | +895% |
| 6 | PlanarQuant | rotorquant | 132.87 | +1476% |
| 7 | PolarQuant-WHT | ours (no mean) | 13,225 | catastrophic |

#### 4-bit

| Rank | Method | Family | PPL | vs FP16 |
|------|--------|--------|-----|---------|
| 1 | **PolarQuant-WHT+Mean** | hybrid | **8.63** | **+2.4%** |
| 2 | IsoQuant-Full+Mean | hybrid | 13.41 | +59% |
| 3 | Givens+Mean | ours | 15.29 | +81% |
| 4 | PlanarQuant+Mean | hybrid | 15.32 | +82% |
| 5 | IsoQuant-Full | rotorquant | 80.26 | +852% |
| 6 | PlanarQuant | rotorquant | 541.72 | +6327% |
| 7 | PolarQuant-WHT | ours (no mean) | 9,731 | catastrophic |

### Qwen2.5-3B-Instruct (2 KV heads, FP16 PPL: 11.44)

#### 3-bit

| Rank | Method | Family | PPL | vs FP16 |
|------|--------|--------|-----|---------|
| 1 | **PolarQuant-WHT+Mean** | hybrid | **11.87** | **+3.8%** |
| 2 | IsoQuant-Full+Mean | hybrid | 32.60 | +185% |
| 3 | Givens+Mean | ours | 40.51 | +254% |
| 4 | PlanarQuant+Mean | hybrid | 41.13 | +260% |
| 5 | IsoQuant-Full | rotorquant | 49.85 | +336% |
| 6 | PlanarQuant | rotorquant | 103.04 | +801% |
| 7 | PolarQuant-WHT | ours (no mean) | 2,340 | catastrophic |

#### 4-bit

| Rank | Method | Family | PPL | vs FP16 |
|------|--------|--------|-----|---------|
| 1 | **PolarQuant-WHT+Mean** | hybrid | **11.55** | **+1.0%** |
| 2 | IsoQuant-Full+Mean | hybrid | 17.77 | +55% |
| 3 | Givens+Mean | ours | 24.61 | +115% |
| 4 | IsoQuant-Full | rotorquant | 24.64 | +115% |
| 5 | PlanarQuant+Mean | hybrid | 24.76 | +117% |
| 6 | PolarQuant-WHT | ours (no mean) | 51.13 | +347% |
| 7 | PlanarQuant | rotorquant | 51.21 | +348% |

---

## Key Findings

### 1. WHT Rotation Wins Everywhere on PPL

PolarQuant-WHT (our implementation of the original TurboQuant rotation) is the best
rotation for perplexity at every model size tested:

| Model | KV Heads | Best 3-bit Method | PPL | vs FP16 |
|-------|----------|-------------------|-----|---------|
| 14B | 8 | PolarQuant-WHT | 5.54 | +12.1% |
| 7B | 4 | PolarQuant-WHT+Mean | 9.06 | +7.5% |
| 3B | 2 | PolarQuant-WHT+Mean | 11.87 | +3.8% |

WHT's global decorrelation (O(d log d) butterfly) produces coordinates that match
the Gaussian distribution assumed by Lloyd-Max codebooks. Block-diagonal rotations
only decorrelate within blocks, leaving inter-block correlations that the scalar
quantizer cannot exploit.

### 2. Mean-Removal: Critical for Low-KV-Head Models, Neutral/Harmful for High-KV-Head

The effect of mean-removal depends strongly on the number of KV heads:

| Model | KV Heads | WHT PPL (no mean) | WHT+Mean PPL | Mean effect |
|-------|----------|--------------------|--------------|-------------|
| 3B | 2 | 2,340 (catastrophic) | **11.87** | **197x fix** |
| 7B | 4 | 13,225 (catastrophic) | **9.06** | **1,460x fix** |
| 14B | 8 | **5.54** (already good) | 5.58 | -0.7% (neutral) |

**Mean-removal on block-diagonal rotations at 14B is harmful:**

| Method | 14B 3-bit (no mean) | 14B 3-bit (+mean) | Effect |
|--------|--------------------|--------------------|--------|
| IsoQuant-Full | 18.39 | 30.87 | mean HURTS (+68%) |
| PlanarQuant | 40.66 | 50.40 | mean HURTS (+24%) |

**Explanation:** With 8 KV heads, each head represents a narrower slice of information.
The per-head mean is already close to zero, so mean-removal subtracts near-zero and
the re-centering slightly distorts the block rotation's expected distribution.
With 2-4 KV heads, the per-head mean is significantly non-zero, causing the codebook
to waste levels on the mean offset — mean-removal fixes this.

### 3. Attention Cosine Sim Does NOT Predict PPL

Across all models, attention cosine similarity is uncorrelated or inversely correlated
with PPL:

| Model | Best attn cos method | Its PPL | Best PPL method | Its attn cos |
|-------|---------------------|---------|-----------------|-------------|
| 14B 3-bit | Givens+Mean (0.896) | 49.95 | WHT (0.890) | 5.54 |
| 7B 3-bit | WHT (0.129) | 13,225 | WHT+Mean (0.089) | 9.06 |
| 3B 3-bit | IsoQuant+Mean (0.642) | 32.60 | WHT+Mean (0.548) | 11.87 |

**Benchmarks reporting only attention cosine are misleading. PPL is the metric
that predicts generation quality.**

### 4. RotorQuant Consistently Worse on PPL

RotorQuant's block-diagonal rotations (IsoQuant, PlanarQuant) never win on PPL:

| Model | Best RotorQuant PPL | Best TurboQuantDC PPL | Gap |
|-------|--------------------|-----------------------|-----|
| 14B 3-bit | IsoQuant 18.39 | WHT 5.54 | **3.3x worse** |
| 7B 3-bit | IsoQuant 83.92 | WHT+Mean 9.06 | **9.3x worse** |
| 3B 3-bit | IsoQuant 49.85 | WHT+Mean 11.87 | **4.2x worse** |

RotorQuant's published advantage (Llama 3.1 8B: 6.91 vs 7.07 PPL) is likely because
Llama has 8 KV heads where the rotation choice matters less, and their benchmark
doesn't test on low-KV-head models where the gap is catastrophic.

---

## Recommended Configuration

| Scenario | Method | Bits | Expected PPL delta |
|----------|--------|------|--------------------|
| Low KV heads (2-4) | WHT + mean-removal | 3 | +4-8% |
| Low KV heads (2-4) | WHT + mean-removal | 4 | +1-2.5% |
| High KV heads (8+) | WHT (no mean-removal) | 3 | +12% |
| High KV heads (8+) | WHT + mean-removal | 4 | +2.6% |
| Speed-critical | PlanarQuant + mean-removal | 3 | +86% (low heads) |

**Adaptive rule:** If KV heads <= 4, apply mean-removal. If KV heads >= 8, skip it
(or apply only at 4-bit where it provides small benefit).

---

## Methodology

**PPL:** Sliding-window on wikitext-2 test (8,192 tokens). DynamicCache.update patched
to quantize keys. K-only compression (V uncompressed). BnB 4-bit model weights.

**Mean-removal:** Subtracts per-head key mean across sequence dim, adds back after
dequantization. Exploits softmax shift-invariance.

**Speed:** Python-level only (not fused kernels). RotorQuant's published llama.cpp
speed advantages (5.3x prefill) are from fused CUDA kernels not measured here.

---

## E8 Lattice VQ Breakthrough (April 15, 2026)

**E8+WHT+Mean replaces scalar Lloyd-Max with 8D lattice VQ after WHT rotation.**
E8 achieves 14% lower NSM than scalar quantization (Zador's theorem). On real models,
the PPL improvement is even larger:

### 3-bit Results

| Model | E8+WHT+Mean | WHT+Mean (prev best) | Scalar only | FP16 |
|-------|-------------|----------------------|-------------|------|
| 3B | **11.44 (+0.1%)** | 11.87 (+3.8%) | 2,340 (catastrophic) | 11.44 |
| 7B | **8.49 (+0.8%)** | 9.06 (+7.5%) | 13,225 (catastrophic) | 8.43 |
| 14B | **5.02 (+1.5%)** | 5.58 (+12.9%) | 5.54 (+12.1%) | 4.94 |

### 4-bit Results

| Model | E8+WHT+Mean | WHT+Mean (prev best) | FP16 |
|-------|-------------|----------------------|------|
| 14B | **4.97 (+0.5%)** | 5.07 (+2.6%) | 4.94 |

**E8 reduces PPL degradation by 38x on 3B and 9x on 7B** compared to scalar Lloyd-Max
with mean-removal. This is near-lossless 3-bit KV cache compression at 5x memory reduction.

Algorithm: Conway-Sloane two-coset nearest E8 point finding. O(1) per 8D block.
No calibration data, no learned parameters, no lookup table. Implementation: 180 lines Python.

22 unit tests pass covering lattice properties, MSE comparison, and WHT integration.

---

## Honest Limitations

1. **Qwen-only.** Llama/Gemma results may differ (Llama has 8+ KV heads).
2. **K-only.** Production compresses both K and V.
3. **BnB 4-bit weights** compound quantization errors.
4. **512-token window.** Longer contexts may shift dynamics.
5. **Python speed only.** Fused kernel speeds not compared.
