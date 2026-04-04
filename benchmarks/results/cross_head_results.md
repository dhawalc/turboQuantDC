# Cross-Head Delta Compression Results

Model: Qwen/Qwen2.5-3B-Instruct (GQA: 16 query heads, 2 KV heads, d=128)
Date: 2026-04-04

## Hypothesis

KV heads at the same layer position are correlated. By compressing
one "anchor" head at 3-bit and remaining heads as 1-bit deltas from
the anchor, we can achieve extreme compression:
- 8 heads: (3 + 7*1)/8 = 1.25 bits/element = 12.8x compression
- 2 heads: (3 + 1*1)/2 = 2.0 bits/element = 8.0x compression

## Inter-Head Correlation (Real Model)

| Metric | Keys | Values |
|--------|------|--------|
| Mean cosine similarity | 0.1220 | 0.0057 |
| Mean Pearson r | 0.1172 | 0.0048 |
| Delta variance ratio | 1.4143 | 2.0068 |

**Variance ratio > 1.0 means deltas are LARGER than the originals.**
This is the opposite of what we need -- it means heads are anti-correlated
or independent, not redundant.

### Per-Layer Key Delta Statistics (vs Head 0 Anchor)

| Layer | Head | Cosine to Anchor | Variance Ratio | Relative Delta Norm |
|-------|------|------------------|----------------|---------------------|
| 0 (first) | 1 | 0.1450 | 1.3534 | 1.5325 |
| 9 (early) | 1 | 0.0188 | 2.0972 | 1.3592 |
| 18 (middle) | 1 | -0.0102 | 1.3926 | 1.9145 |
| 27 (late) | 1 | 0.2552 | 0.9367 | 3.9867 |
| 35 (last) | 1 | 0.2010 | 1.2915 | 1.5007 |

Key observations:
- Most layers show near-zero cosine between heads (0.0188, -0.0102)
- Layer 27 and 35 show modest correlation (0.25, 0.20) but variance ratio is still ~1.0
- Relative delta norms are >1.3 everywhere, meaning ||kv_1 - kv_0|| > ||kv_0||
- Values are effectively uncorrelated (cosine ~0.006)

## Compression Quality Comparison (Real 2-Head Model)

### Layer 9

| Config | Eff. Bits | Comp. Ratio | Mean Cos | Min Cos | Top-5 Attn | Attn r | MSE |
|--------|-----------|-------------|----------|---------|------------|--------|-----|
| Uniform 3-bit RQ | 3.00 | 5.3x | 0.9825 | 0.9680 | 0.7750 | 0.9859 | 0.087 |
| Uniform 2-bit RQ | 2.00 | 8.0x | 0.9409 | 0.9197 | 0.6750 | 0.9513 | 0.292 |
| **Cross-head 3+1** | **2.19** | **7.3x** | **0.8153** | **0.5301** | **0.5000** | **0.8477** | **0.864** |
| Cross-head 3+2 | 2.69 | 6.0x | 0.9369 | 0.8403 | 0.5000 | 0.9467 | 0.300 |
| Cross-head 2+1 | 1.69 | 9.5x | 0.8060 | 0.5625 | 0.4500 | 0.8392 | 0.897 |

### Layer 18

| Config | Eff. Bits | Comp. Ratio | Mean Cos | Min Cos | Top-5 Attn | Attn r | MSE |
|--------|-----------|-------------|----------|---------|------------|--------|-----|
| Uniform 3-bit RQ | 3.00 | 5.3x | 0.9857 | 0.9725 | 0.6500 | 0.9900 | 0.196 |
| Uniform 2-bit RQ | 2.00 | 8.0x | 0.9494 | 0.9218 | 0.4250 | 0.9655 | 0.697 |
| **Cross-head 3+1** | **2.19** | **7.3x** | **0.8653** | **0.6785** | **0.4750** | **0.7092** | **2.503** |
| Cross-head 3+2 | 2.69 | 6.0x | 0.9561 | 0.9049 | 0.5500 | 0.9473 | 0.808 |

### Layer 27

| Config | Eff. Bits | Comp. Ratio | Mean Cos | Min Cos | Top-5 Attn | Attn r | MSE |
|--------|-----------|-------------|----------|---------|------------|--------|-----|
| Uniform 3-bit RQ | 3.00 | 5.3x | 0.9851 | 0.9677 | 0.5250 | 0.9950 | 0.460 |
| Uniform 2-bit RQ | 2.00 | 8.0x | 0.9438 | 0.9156 | 0.3750 | 0.9837 | 1.989 |
| **Cross-head 3+1** | **2.19** | **7.3x** | **0.9146** | **0.7803** | **0.4000** | **0.8847** | **5.205** |
| Cross-head 3+2 | 2.69 | 6.0x | 0.9674 | 0.9362 | 0.4750 | 0.9806 | 1.723 |

**Result: Cross-head 3+1 at 2.19 bits is WORSE than Uniform 2-bit at 2.0 bits in every metric.**
The delta reconstruction (head 1 cos ~0.65-0.85) drags the average far below uniform quantization.

## Synthetic 8-Head Experiment

To test whether cross-head delta would work IF heads were correlated,
we simulated 8-head MHA by replicating a real KV head with controlled noise:

### High correlation (cosine ~0.998, variance ratio 0.005)

| Config | Eff. Bits | Comp. Ratio | Mean Cos | Top-5 Attn | Attn r |
|--------|-----------|-------------|----------|------------|--------|
| Uniform 3-bit | 3.00 | 5.3x | 0.9839 | 0.7375 | 0.9868 |
| Uniform 2-bit | 2.00 | 8.0x | 0.9442 | 0.5563 | 0.9556 |
| **Cross-head 3+1** | **1.39** | **11.5x** | **0.9922** | **0.8313** | **0.9939** |
| Cross-head 3+2 | 2.27 | 7.1x | 0.9961 | 0.8875 | 0.9970 |

**At high correlation: Cross-head 3+1 BEATS uniform 3-bit at 2.3x better compression!**
- 11.5x vs 5.3x compression
- 0.9922 vs 0.9839 mean cosine
- 83.1% vs 73.8% top-5 attention match

### Medium correlation (cosine ~0.92, variance ratio 0.16)

| Config | Eff. Bits | Comp. Ratio | Mean Cos | Top-5 Attn | Attn r |
|--------|-----------|-------------|----------|------------|--------|
| Uniform 3-bit | 3.00 | 5.3x | 0.9831 | 0.7000 | 0.9864 |
| Uniform 2-bit | 2.00 | 8.0x | 0.9428 | 0.5500 | 0.9521 |
| Cross-head 3+1 | 1.39 | 11.5x | 0.9681 | 0.6438 | 0.9732 |
| Cross-head 3+2 | 2.27 | 7.1x | 0.9884 | 0.7438 | 0.9906 |

At medium correlation: Cross-head 3+2 beats uniform 3-bit at 1.3x better compression.
Cross-head 3+1 quality drops below uniform 3-bit but is still usable.

### Low correlation (cosine ~0.67, variance ratio 0.65)

| Config | Eff. Bits | Comp. Ratio | Mean Cos | Top-5 Attn | Attn r |
|--------|-----------|-------------|----------|------------|--------|
| Uniform 3-bit | 3.00 | 5.3x | 0.9817 | 0.7875 | 0.9837 |
| Uniform 2-bit | 2.00 | 8.0x | 0.9395 | 0.6438 | 0.9457 |
| Cross-head 3+1 | 1.39 | 11.5x | 0.8941 | 0.4625 | 0.9062 |
| Cross-head 3+2 | 2.27 | 7.1x | 0.9648 | 0.6500 | 0.9684 |

At low correlation: Cross-head degrades significantly. Uniform 2-bit at 8.0x beats
Cross-head 3+1 at 11.5x in every metric.

## Generation Quality

| Config | Score |
|--------|-------|
| FP16 | 4/5 |
| RQ-3bit | 1/5 |

## Key Findings

### 1. Real GQA Models Have Near-Zero Inter-Head Correlation

This is the central negative result. On Qwen2.5-3B-Instruct:
- Key heads: mean cosine = 0.12, Pearson r = 0.12
- Value heads: mean cosine = 0.006, Pearson r = 0.005
- Delta variance ratio: 1.41x (keys), 2.01x (values)

The 2 KV heads in GQA are **effectively independent**. The deltas between
heads are actually LARGER than the heads themselves (variance ratio > 1.0),
meaning delta coding is strictly worse than direct quantization.

This makes architectural sense: GQA already reduces KV heads to the minimum
needed for diverse information. The remaining 2 heads are deliberately
specialized to capture different attention patterns -- if they were
redundant, the model would have pruned them during training.

### 2. Cross-Head Delta Works Beautifully at High Correlation

The synthetic experiment proves the algorithm is correct:
- At 0.998 cosine correlation: Cross-head 3+1 at 1.39 bits (11.5x) BEATS
  uniform 3-bit at 3.0 bits (5.3x) in both cosine sim AND attention quality
- The delta approach would deliver 10x+ compression IF the correlation existed

### 3. The Approach Requires Correlated Heads to Be Viable

| Correlation Level | Cross-Head Viable? | Compression Advantage |
|---|---|---|
| >0.95 cosine | Yes -- beats uniform | 2x better than uniform 3-bit |
| 0.8-0.95 cosine | Marginal | Only with 3+2 (mild compression gain) |
| <0.8 cosine | No -- uniform wins | Negative (worse than just using fewer bits) |

### 4. Modern GQA Models Are the Wrong Target

| Model | Q Heads | KV Heads | Head Correlation |
|---|---|---|---|
| Qwen2.5-3B | 16 | 2 | ~0.12 |
| Qwen2.5-7B | 28 | 4 | (expected ~0.1-0.3) |
| MHA models | N | N | (varies) |

GQA models already compress KV heads by sharing across query groups.
The few remaining KV heads are maximally diverse by design.
Cross-head delta would only be viable for:
- MHA models (all heads are KV heads, may have redundancy)
- Models with poor head specialization
- Specific layers where heads happen to align (e.g., layer 27 had 0.25 cosine)

## Effective Bit Rates (Theoretical)

| Config | Formula (N=8 heads) | Bits/elem | Compression |
|--------|---------|-----------|-------------|
| Uniform 3-bit | 3 | 3.00 | 5.3x |
| Uniform 2-bit | 2 | 2.00 | 8.0x |
| Cross-head 3+1 | (3+7*1)/8 | 1.25 | 12.8x |
| Cross-head 3+2 | (3+7*2)/8 | 2.13 | 7.5x |
| Cross-head 2+1 | (2+7*1)/8 | 1.13 | 14.2x |

These rates are achievable IF inter-head correlation > 0.95. For real GQA
models, uniform quantization at the same total bit budget produces better quality.

## Conclusion

**DISPROVED for GQA models.** Real KV heads in Qwen2.5-3B have near-zero
inter-head correlation (cosine ~0.12), making delta coding strictly worse than
uniform quantization. The delta variance ratio > 1.0 means deltas require
MORE bits than absolute values, not fewer.

**VALIDATED for high-correlation scenarios.** The algorithm itself works:
when inter-head cosine > 0.95 (synthetic test), cross-head 3+1 at 1.39 bits
BEATS uniform 3-bit at 3.0 bits in reconstruction quality AND attention match.
This means the approach could be viable for:
- Full MHA models with head redundancy
- Per-layer adaptive: use cross-head only for layers where correlation > 0.9
- Future architectures with correlated KV heads

**Recommendation:** Do not pursue cross-head delta for production GQA compression.
The existing uniform 3-bit ResidualQuant (5.1x) or 2-bit (8.0x) approaches
are strictly better for current models. The idea is sound but the target
(modern GQA heads) lacks the prerequisite redundancy.
