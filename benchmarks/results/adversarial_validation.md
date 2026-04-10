# Adversarial Validation Report -- TurboQuantDC

Generated: 2026-04-09
Device: RTX 4090, CUDA
Models tested: Qwen2.5-3B-Instruct, Qwen2.5-14B-Instruct (both BnB 4-bit)
Seeds: [42, 137, 2024]

---

## Executive Summary

**The core algorithm (WHT + mean-removal at 3-bit) is REAL and HOLDS UP.**
Several of the "breakthrough" numbers are OVERFITTED or have CRITICAL CAVEATS.

| Claimed Result | Verdict | Details |
|---|---|---|
| Cayley rotation: 0.974 cosine | REAL but OVERSTATED | +0.003-0.006 lift on non-layer-0 layers. Layer 0 inflates the average. |
| Expected Attention: 10x | BROKEN on 2 of 4 adversarial cases | Fails on uniform attention (0.08 Spearman) and distribution shift (-0.03). |
| KVSculpt: 19.7x at 0.829 | REAL but fragile | 0.83-0.98 output cosine depending on ratio. |
| Triple stack: 59.8x at 0.90 | OVERSTATED | At 39x we get 0.891, at 65x we get 0.855. The 59.8x/0.90 claim is borderline. |
| Mean-removal: +25% quality | REAL AND ROBUST | Never hurts across 5 diverse prompts. +0.5% to +2.8% cosine lift. |

---

## Test 1: Cayley Rotation -- Cross-Model Validation

Tested on both 3B and 14B with 3 seeds each across 5 layers.

### 3B Results
| Layer | WHT cos (mean) | Cayley cos (mean) | Lift |
|---|---|---|---|
| 0 | 0.643 | 0.784 | +0.141 |
| 7 | 0.997 | 0.999 | +0.002 |
| 15 | 0.991 | 0.994 | +0.004 |
| 23 | 0.988 | 0.993 | +0.005 |
| 35 | 0.988 | 0.994 | +0.006 |

### 14B Results
| Layer | WHT cos (mean) | Cayley cos (mean) | Lift |
|---|---|---|---|
| 0 | 0.944 | 0.961 | +0.017 |
| 10 | 0.998 | 0.999 | +0.001 |
| 20 | 0.996 | 0.998 | +0.002 |
| 30 | 0.986 | 0.993 | +0.007 |
| 47 | 0.990 | 0.995 | +0.005 |

**VERDICT**: Cayley works on both models, but the lift is TINY on all layers
except layer 0. Layer 0 is anomalous on 3B (WHT gets ~0.64 cosine there) and
inflates the overall average. On non-layer-0 layers the lift is +0.001 to
+0.007 -- real but not transformative. The previously claimed 0.974 average
was likely dominated by the layer 0 recovery (which goes from 0.64 to 0.78
-- a huge gain on a terrible baseline).

**HIGH VARIANCE WARNING**: Overall lift std (0.056 on 3B, 0.006 on 14B) is
comparable to the mean lift. Layer 0 dominates the variance.

---

## Test 2: Cayley Transfer (3B calibration -> 14B evaluation)

Does a Cayley rotation trained on 3B data transfer to 14B?

| Layer | 3B (in-dist) | 14B (transfer) | 14B (WHT baseline) | Beats WHT? |
|---|---|---|---|---|
| 0 | 0.771 | 0.940 | 0.943 | NO |
| 1 | 0.932 | 0.980 | 0.979 | YES (barely) |
| 2 | 0.939 | 0.918 | 0.916 | YES (barely) |
| 3 | 1.000 | 0.941 | 0.948 | NO |
| 4 | 1.000 | 0.979 | 0.977 | YES (barely) |

**VERDICT**: Transfer is marginal. 3 out of 5 layers "beat" WHT but by <0.003.
2 layers lose. **Cayley must be calibrated per-model**. Cross-model transfer
does not justify the calibration cost.

---

## Test 3: Sequence Length Sensitivity

| Seq Len | Actual Tokens | WHT cos | Cayley cos | EA Spearman | Distill cos |
|---|---|---|---|---|---|
| 128 | 128 | 0.9953 | 0.9982 | 0.164 | 0.938 |
| 256 | 256 | 0.9919 | 0.9968 | 0.373 | 0.950 |
| 512 | 512 | 0.9893 | 0.9940 | 0.223 | 0.955 |
| 1024 | 724 | 0.9853 | 0.9907 | 0.146 | 0.960 |

**KEY FINDINGS**:
- WHT and Cayley both **degrade gracefully** with longer sequences (0.995 -> 0.985 for WHT).
- Expected Attention Spearman is **POOR across all lengths** (0.15-0.37). It never
  reaches the 0.5+ needed to be useful for ranking.
- Distillation **improves with length** (0.938 -> 0.960), which makes sense --
  more tokens = more redundancy to exploit.
- Note: 1024 and 2048 produced the same 724 tokens (model hit EOS), so they are duplicates.

**CONCERN**: EA Spearman never exceeded 0.37 on real model data. The "10x compression"
claim from Expected Attention was measured on synthetic power-law data (0.87 Spearman).
On real LLM data, the ranking is much weaker.

---

## Test 4: Prompt Diversity

All at 512 tokens, 3B model.

| Prompt | Layer 0 WHT | Layer 0 Cayley | Mid-Layer WHT | Mid-Layer Cayley | Last-Layer WHT | Last-Layer Cayley |
|---|---|---|---|---|---|---|
| code | 0.639 | 0.798 | 0.986 | 0.994 | 0.970 | 0.985 |
| math | 0.569 | 0.825 | 0.980 | 0.987 | 0.957 | 0.972 |
| creative | 0.580 | 0.787 | 0.990 | 0.994 | 0.991 | 0.994 |
| factual | 0.665 | 0.771 | 0.989 | 0.994 | 0.988 | 0.995 |
| adversarial | 0.524 | 0.659 | 0.985 | 0.992 | 0.983 | 0.991 |

**VERDICT**: Methods are robust across prompt types. No prompt causes catastrophic
failure. The adversarial prompt (repeated tokens) is slightly worse but not by much.
Math prompts are hardest on later layers (0.957 WHT vs 0.991 for creative).

**Layer 0 is ALWAYS bad for WHT** (0.52-0.67 cosine). This is a real issue -- the
embedding layer has very different statistical properties. Cayley helps there (+0.10-0.26)
but never fully fixes it (best is 0.825).

---

## Test 5: Expected Attention Failure Modes (CRITICAL)

Tested on synthetic data with controlled distributions.

| Scenario | EA Spearman | Assessment |
|---|---|---|
| Power-law (normal) | **0.866** | WORKS well |
| Uniform attention | **0.083** | BROKEN -- basically random |
| Distribution shift | **-0.035** | BROKEN -- worse than random |
| Cold start (4 queries) | **0.427** | DEGRADED but partially functional |

**VERDICT**: Expected Attention has **TWO CRITICAL FAILURE MODES**:
1. **Uniform attention**: When all tokens are roughly equally important (e.g.,
   factual recall, uniform information density), EA cannot distinguish them.
   Spearman 0.08 = noise.
2. **Distribution shift**: When the query distribution changes (e.g., model
   switches from reciting to reasoning), EA uses stale statistics and produces
   NEGATIVE correlation -- it would evict the IMPORTANT tokens.

These failure modes are not academic -- real LLM conversations exhibit both patterns.
The 10x compression claim from EA is only valid when attention is highly concentrated
(power-law) AND stationary. **EA needs a shift-detection mechanism to be safe.**

---

## Test 6: Mean-Removal Impact

| Prompt | With Center | Without | Diff | Verdict |
|---|---|---|---|---|
| code | 0.9863 | 0.9756 | +0.0107 | HELPS |
| math | 0.9800 | 0.9519 | +0.0281 | HELPS |
| creative | 0.9895 | 0.9802 | +0.0093 | HELPS |
| factual | 0.9893 | 0.9845 | +0.0048 | HELPS |
| adversarial | 0.9845 | 0.9720 | +0.0125 | HELPS |

**VERDICT**: Mean-removal is the most ROBUST technique we have. It NEVER hurts
across 5 diverse prompts. The lift ranges from +0.5% to +2.8% cosine sim.
Math prompts benefit most (+2.8%), likely because math text has more structured
patterns that create consistent key offsets.

The "+25% quality" claim needs clarification: it was likely measured as relative
improvement in error, not absolute cosine. The absolute improvement is +0.5-2.8%
cosine points, which is real and consistent.

---

## Test 7: Triple Stack Compression Limits

| Config | Compression | Output Cosine | Status |
|---|---|---|---|
| 2x distill only | 10.0x | 0.977 | GOOD |
| 30% evict + 2x distill | 14.1x | 0.938 | DEGRADED |
| 50% evict + 4x distill | 39.3x | 0.891 | DEGRADED |
| 70% evict + 4x distill | 64.6x | 0.855 | DEGRADED |
| 80% evict + 4x distill | 95.1x | 0.827 | DEGRADED |
| 90% evict + 4x distill | 180.8x | 0.762 | GARBAGE |
| 50% evict + 8x distill | 78.6x | 0.870 | DEGRADED |
| 70% evict + 8x distill | 129.1x | 0.834 | DEGRADED |

**VERDICT**: The "59.8x at 0.90 cosine" claim is **at the edge of reality**.
At 39.3x we get 0.891 (close to 0.90 but not quite), and at 64.6x we get 0.855.
The claim likely used a more favorable prompt/layer combination. The true
relationship is approximately:

- 10x: 0.98 cosine (SOLID)
- 40x: 0.89 cosine (ACCEPTABLE for some use cases)
- 65x: 0.86 cosine (MARGINAL)
- 95x: 0.83 cosine (POOR)
- 180x: 0.76 cosine (GARBAGE)

**Garbage threshold**: ~100x compression is where output cosine drops below 0.80.

---

## Test 8: End-to-End Generation Quality

FP16 baseline perplexity: 1.09 (200 tokens, math prompt)

Per-layer WHT quantization quality (3-bit with mean-removal):

| Layer | MSE | Vector Cosine |
|---|---|---|
| 0 | 0.054 | 0.9998 |
| 9 | 0.038 | 0.9801 |
| 18 | 0.040 | 0.9948 |
| 27 | 0.042 | 0.9894 |
| 35 | 0.025 | 0.9944 |

Mean vector cosine across layers: 0.9917

**VERDICT**: Individual vector reconstruction is excellent (0.98-1.00 cosine).
The MSE is low and consistent across layers. Layer 0 has highest MSE but also
highest vector cosine because its vectors are more aligned.

---

## Test 9: Reproducibility

| Method | Cosine (mean +/- std) | Top-5 (mean +/- std) | Reliable? |
|---|---|---|---|
| WHT + mean-removal | 0.9895 +/- 0.0002 | 1.000 +/- 0.000 | YES |
| Cayley (100 steps) | 0.9947 +/- 0.0005 | 1.000 +/- 0.000 | YES |
| Cayley (200 steps) | 0.9953 +/- 0.0003 | 1.000 +/- 0.000 | YES |

**VERDICT**: All three methods are highly reproducible. Std is <0.1% of mean
in all cases. The seed does not meaningfully affect results -- the algorithms
are deterministic enough.

---

## Test 10: Full 14B Validation

WHT+mean-removal vs raw PolarQuant on Qwen2.5-14B-Instruct:

| Prompt | Layer | WHT+mean cos | PolarQuant cos | WHT Advantage |
|---|---|---|---|---|
| factual | 0 | 0.943 | 0.398 | +0.545 |
| factual | 10 | 0.998 | 0.997 | +0.001 |
| factual | 30 | 0.985 | 0.978 | +0.007 |
| factual | 47 | 0.991 | 0.955 | +0.036 |
| code | 0 | 0.948 | 0.522 | +0.426 |
| code | 47 | 0.987 | 0.920 | +0.067 |
| adversarial | 0 | 0.936 | 0.352 | +0.584 |
| adversarial | 47 | 0.984 | 0.922 | +0.062 |

**VERDICT**: WHT + mean-removal MASSIVELY outperforms raw PolarQuant, especially
on layer 0 (+0.4-0.6 cosine) and late layers (+0.04-0.07). The improvement is
consistent across models and prompts. This is the single most impactful technique.

---

## FINAL VERDICTS

### WHAT HOLDS UP (REAL):
1. **WHT + mean-removal at 3-bit**: The foundation. 0.985-0.998 cosine sim across
   models, prompts, and sequence lengths. Highly reproducible. The single best
   bang-for-buck technique.
2. **Mean-removal**: Never hurts, always helps (+0.5-2.8%). Robust across all
   tested conditions. Essentially free.
3. **KVSculpt distillation**: Works at moderate ratios (2-4x distillation).
   10x total compression (2x distill + 5x quant) gives 0.977 cosine = solid.
4. **Cayley rotation**: Real but small lift (+0.001-0.007 on non-layer-0 layers).
   Worth using if calibration budget is available.

### WHAT IS OVERSTATED:
1. **Cayley "0.974 cosine"**: The average was inflated by layer 0 recovery.
   On typical middle/late layers the lift is +0.002-0.006. Still positive, but
   not the breakthrough it was presented as.
2. **Triple stack "59.8x at 0.90"**: More accurately: 40x gets you ~0.89,
   65x gets you ~0.86. The 0.90 target requires careful tuning.
3. **"10x from Expected Attention"**: Only valid with power-law attention AND
   stationary query distribution. Both conditions fail in realistic scenarios.

### WHAT IS BROKEN:
1. **Expected Attention on uniform distributions**: Spearman 0.08 = useless.
   If the model is doing broad information retrieval (not focused attention),
   EA will evict randomly.
2. **Expected Attention on distribution shifts**: Spearman -0.03 = ANTI-correlated.
   EA actively recommends evicting the tokens that will become important.
   This is DANGEROUS in production.
3. **Triple stack beyond 100x**: Output cosine drops below 0.80. Not usable
   for any quality-sensitive application.
4. **Layer 0 quantization on 3B**: WHT gets 0.52-0.67 cosine on layer 0.
   Even Cayley only reaches 0.66-0.83. Layer 0 needs special handling
   (keep at FP16, or use higher bit-width).

### RECOMMENDATIONS:
1. Ship WHT + mean-removal as the default. It is proven, robust, reproducible.
2. Keep Cayley as an OPTIONAL per-model calibration step. Don't claim 0.974.
3. Add a distribution shift detector to Expected Attention before using it
   for eviction. Without it, EA is a liability.
4. Claim triple stack at 20-40x, not 60x. The quality at 60x is marginal.
5. Always use FP16 for layer 0 (or at minimum, 4-bit). Layer 0 is special.
