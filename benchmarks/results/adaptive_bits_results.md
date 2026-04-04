# Adaptive Bits Results

**Date:** 2026-04-04 04:10
**Model:** Qwen/Qwen2.5-3B-Instruct
**Context:** 1138 tokens
**Layers:** 36
**Heads:** 2
**Head dim:** 128
**Runtime:** 64.5s

## 1. Attention Distribution (Power Law Analysis)

**Power-law strength (Gini coefficient):** 0.8310
  - 0.0 = perfectly uniform attention
  - 1.0 = all attention on one token
  - **0.83** indicates strong concentration

**Normalized entropy:** 0.6510

### Attention Concentration

| Top % of tokens | % of total attention captured |
|---|---|
| 1% | 40.1% |
| 5% | 66.3% |
| 10% | 79.4% |
| 20% | 87.1% |
| 30% | 91.1% |
| 50% | 95.8% |

### Per-Layer Gini Coefficient

| Layer | Gini | Top 5% captures | Top 10% captures |
|---|---|---|---|
| 0 | 0.8109 | 63.0% | 79.0% |
| 1 | 0.7124 | 53.0% | 69.4% |
| 2 | 0.7330 | 55.8% | 72.2% |
| 3 | 0.8226 | 71.2% | 79.7% |
| 4 | 0.8906 | 80.5% | 88.4% |
| 5 | 0.9289 | 84.5% | 92.2% |
| 6 | 0.8635 | 74.7% | 84.6% |
| 7 | 0.7980 | 65.7% | 75.1% |
| 8 | 0.8583 | 73.4% | 82.5% |
| 9 | 0.9203 | 84.2% | 91.4% |
| 10 | 0.8607 | 70.5% | 83.2% |
| 11 | 0.8707 | 72.1% | 85.1% |
| 12 | 0.8853 | 72.4% | 85.5% |
| 13 | 0.8777 | 73.9% | 86.0% |
| 14 | 0.9143 | 79.0% | 92.0% |
| 15 | 0.9151 | 77.7% | 92.5% |
| 16 | 0.9356 | 80.9% | 95.7% |
| 17 | 0.8439 | 65.9% | 79.5% |
| 18 | 0.9057 | 74.9% | 91.5% |
| 19 | 0.8096 | 58.0% | 73.9% |
| 20 | 0.8076 | 58.0% | 73.3% |
| 21 | 0.7327 | 42.1% | 58.6% |
| 22 | 0.7128 | 44.5% | 59.3% |
| 23 | 0.8662 | 66.6% | 83.7% |
| 24 | 0.9014 | 75.3% | 89.8% |
| 25 | 0.8709 | 72.6% | 84.6% |
| 26 | 0.8502 | 68.5% | 82.5% |
| 27 | 0.7274 | 45.7% | 59.8% |
| 28 | 0.8391 | 68.6% | 79.8% |
| 29 | 0.8389 | 67.0% | 79.6% |
| 30 | 0.8376 | 70.2% | 80.8% |
| 31 | 0.6814 | 44.4% | 58.2% |
| 32 | 0.7483 | 55.0% | 65.4% |
| 33 | 0.7251 | 53.6% | 62.8% |
| 34 | 0.8186 | 61.1% | 82.0% |
| 35 | 0.8008 | 62.2% | 78.6% |

**Key Finding:** Top 10% of tokens capture **79.4%** of attention, top 20% capture **87.1%**. This validates the power-law hypothesis.

## 2. Tiered Compression Results

### Quality Comparison: Adaptive vs Uniform 3-bit

| Config | Eff. Bits | Compression | Adaptive CosSim | Uniform 3b CosSim | Delta CosSim | Adaptive Top-5 | Uniform 3b Top-5 | Delta Top-5 |
|---|---|---|---|---|---|---|---|---|
| same_budget_3bit | 2.97 | 5.4x | 0.9489 | 0.9207 | +0.0282 | 97.4% | 97.1% | +0.3% |
| aggressive | 3.30 | 4.8x | 0.9575 | 0.9207 | +0.0368 | 97.3% | 97.1% | +0.1% |
| moderate | 4.10 | 3.9x | 0.9655 | 0.9207 | +0.0448 | 97.3% | 97.1% | +0.1% |
| conservative | 3.75 | 4.3x | 0.9613 | 0.9207 | +0.0406 | 97.2% | 97.1% | +0.1% |
| eviction_sim | 3.50 | 4.6x | 0.9548 | 0.9207 | +0.0341 | 97.2% | 97.1% | +0.1% |
| ultra_aggressive | 2.05 | 7.8x | 0.9116 | 0.9207 | -0.0091 | 97.6% | 97.1% | +0.5% |

### Tier Distribution by Config

**same_budget_3bit** (~3.0 eff bits: top 3% FP16, next 7% 4-bit, next 40% 3-bit, bottom 50% 2-bit):
  - Tier 0 (FP16): 3.1%
  - Tier 1 (4-bit): 6.9%
  - Tier 2 (3-bit): 40.0%
  - Tier 3 (2-bit): 50.0%
  - Effective bits: 2.97
  - Compression ratio: 5.4x

**aggressive** (Top 5% FP16, next 15% 4-bit, next 30% 3-bit, bottom 50% 2-bit):
  - Tier 0 (FP16): 5.0%
  - Tier 1 (4-bit): 15.0%
  - Tier 2 (3-bit): 30.0%
  - Tier 3 (2-bit): 50.0%
  - Effective bits: 3.30
  - Compression ratio: 4.8x

**moderate** (Top 10% FP16, next 20% 4-bit, next 30% 3-bit, bottom 40% 2-bit):
  - Tier 0 (FP16): 10.0%
  - Tier 1 (4-bit): 20.0%
  - Tier 2 (3-bit): 30.0%
  - Tier 3 (2-bit): 40.0%
  - Effective bits: 4.10
  - Compression ratio: 3.9x

**conservative** (Top 5% FP16, next 10% 4-bit, next 25% 3-bit, bottom 60% 3-bit):
  - Tier 0 (FP16): 5.0%
  - Tier 1 (4-bit): 10.0%
  - Tier 2 (3-bit): 25.0%
  - Tier 3 (3-bit): 59.9%
  - Effective bits: 3.75
  - Compression ratio: 4.3x

**eviction_sim** (Top 10% FP16, next 20% 4-bit, next 20% 3-bit, bottom 50% 1-bit (near-eviction)):
  - Tier 0 (FP16): 10.0%
  - Tier 1 (4-bit): 20.0%
  - Tier 2 (3-bit): 19.9%
  - Tier 3 (1-bit): 50.0%
  - Effective bits: 3.50
  - Compression ratio: 4.6x

**ultra_aggressive** (~2.0 eff bits: top 5% FP16, next 15% 3-bit, bottom 80% 1-bit):
  - Tier 0 (FP16): 5.0%
  - Tier 1 (3-bit): 15.0%
  - Tier 2 (1-bit): 80.0%
  - Effective bits: 2.05
  - Compression ratio: 7.8x

## 3. Eviction vs Adaptive Comparison

| Strategy | Eff. Bits | CosSim | Top-5 |
|---|---|---|---|
| Evict 50% (keep_50pct) | 8.0 | 0.9820 | 99.9% |
| Evict 25% (keep_75pct) | 12.0 | 0.9901 | 100.0% |
| Adaptive (same_budget_3bit) | 2.97 | 0.9489 | 97.4% |
| Adaptive (aggressive) | 3.30 | 0.9575 | 97.3% |
| Adaptive (moderate) | 4.10 | 0.9655 | 97.3% |
| Adaptive (conservative) | 3.75 | 0.9613 | 97.2% |
| Adaptive (eviction_sim) | 3.50 | 0.9548 | 97.2% |
| Adaptive (ultra_aggressive) | 2.05 | 0.9116 | 97.6% |
| Uniform 3-bit | 3.00 | 0.9207 | 97.1% |

## 4. Summary

**Best quality adaptive:** moderate (4.10 bits, CosSim 0.9655)

**Most compressed adaptive:** ultra_aggressive (2.05 bits, 7.8x compression, CosSim 0.9116)

**Adaptive vs Uniform 3-bit:**
  Adaptive at 4.10 bits exceeds uniform 3-bit quality (0.9655 vs 0.9207)

### Conclusion

The attention distribution shows **strong power-law behavior** (Gini=0.8310). Adaptive bit allocation is highly effective: important tokens get more bits, unimportant tokens can be aggressively compressed with minimal quality loss.
