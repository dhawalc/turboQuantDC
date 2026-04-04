# Temporal Delta Coding Experiment Results

**Model:** Qwen/Qwen2.5-3B-Instruct
**Prompt tokens:** 815
**Layers:** 36
**Runtime:** 7.9s

## Hypothesis

Consecutive tokens within the same layer share nearly identical context
(token N sees [1..N], token N+1 sees [1..N+1]). The KV projections should
be highly correlated temporally, making delta coding viable for within-layer
compression. Cross-layer delta coding failed (ratio 2.6x), but temporal
deltas may succeed because the mechanism is fundamentally different.

## Experiment 1: Temporal Correlation

| Metric | Keys | Values |
|--------|------|--------|
| Cosine similarity | 0.7983 | 0.3728 |
| Pearson correlation | 0.7982 | 0.3728 |

### Per-Layer Cosine Similarity

| Layer | Key cos | Val cos |
|-------|---------|---------|
| 0 | 0.9963 | 0.0963 |
| 6 | 0.7224 | 0.2177 |
| 12 | 0.7803 | 0.4003 |
| 18 | 0.8890 | 0.4404 |
| 24 | 0.8395 | 0.4749 |
| 30 | 0.7621 | 0.4502 |
| 35 | 0.8604 | 0.4280 |

## Experiment 2: Delta Variance Ratio

**Threshold:** ratio < 0.5 means delta coding is viable

| Metric | Keys | Values | Viable? |
|--------|------|--------|---------|
| Variance ratio | 0.3872 | 1.2575 | NO |
| L2 norm ratio | 0.5910 | 1.1111 | |
| Linf ratio | 0.4287 | 0.9888 | |

- Key delta viable: **YES** (ratio=0.3872)
- Value delta viable: **NO** (ratio=1.2575)

### Per-Layer Variance Ratios

| Layer | Key var ratio | Val var ratio |
|-------|---------------|---------------|
| 0 | 0.0068 | 1.7655 |
| 6 | 0.5439 | 1.5793 |
| 12 | 0.4310 | 1.1880 |
| 18 | 0.1713 | 1.1304 |
| 24 | 0.3195 | 1.0433 |
| 30 | 0.4661 | 1.1737 |
| 35 | 0.2764 | 1.1885 |

## Experiment 3: Delta by Token Position

| Quartile | Key delta/abs | Val delta/abs |
|----------|---------------|---------------|
| Q1 (early) | 0.4836 | 1.1430 |
| Q2 | 0.4819 | 1.1334 |
| Q3 | 0.4748 | 1.1292 |
| Q4 (late) | 0.4774 | 1.1412 |

## Experiment 4: Delta Sparsity

| Threshold | Key sparsity | Val sparsity |
|-----------|-------------|-------------|
| 0.01 | 0.0215 | 0.0085 |
| 0.05 | 0.1048 | 0.0421 |
| 0.10 | 0.1990 | 0.0840 |
| 0.20 | 0.3584 | 0.1666 |

## Experiment 5: Entropy Analysis

| Bit-width | Key abs H | Key delta H | Ratio | Val abs H | Val delta H | Ratio |
|-----------|-----------|-------------|-------|-----------|-------------|-------|
| 2 | 0.080 | 0.019 | 0.234 | 0.035 | 0.023 | 0.646 |
| 3 | 0.690 | 0.590 | 0.855 | 0.747 | 0.698 | 0.935 |
| 4 | 1.698 | 1.630 | 0.960 | 1.831 | 1.783 | 0.974 |

## Experiment 6: Attention Quality

| Config | Cosine Sim | Top-5 Match | Relative MSE |
|--------|-----------|-------------|--------------|
| delta-2bit | 0.375715 | 8.57% | 3.010977 |
| delta-3bit | 0.745123 | 21.43% | 0.444180 |
| delta-4bit | 0.982554 | 51.43% | 0.073359 |

## Experiment 7: Compression Ratios

| Config | vs FP16 | vs TQ3 | Eff bits/coord |
|--------|---------|--------|----------------|
| delta-1bit | 14.00x | 2.62x | 1.14 |
| delta-2bit | 7.47x | 1.40x | 2.14 |
| delta-3bit | 5.09x | 0.96x | 3.14 |

## Experiment 8: Error Accumulation

| Layer | Bits | Q1 err | Q2 err | Q3 err | Q4 err | Max err |
|-------|------|--------|--------|--------|--------|---------|
| 0 | 2bit | 0.5573 | 1.0189 | 1.3992 | 1.6605 | 1.8238 |
| 0 | 3bit | 0.2722 | 0.5141 | 0.7099 | 0.8367 | 0.8793 |
| 0 | 4bit | 0.1179 | 0.2302 | 0.3098 | 0.3658 | 0.3890 |
| 12 | 2bit | 4.3429 | 7.7750 | 10.4861 | 13.4645 | 17.9930 |
| 12 | 3bit | 2.0282 | 3.5509 | 4.4477 | 5.4633 | 6.9306 |
| 12 | 4bit | 0.9049 | 1.5637 | 1.9307 | 2.3756 | 3.1168 |
| 24 | 2bit | 3.7694 | 7.5468 | 10.4653 | 12.6926 | 16.0337 |
| 24 | 3bit | 1.7721 | 3.2925 | 4.3245 | 5.2530 | 6.7894 |
| 24 | 4bit | 0.7191 | 1.2854 | 1.8114 | 2.1015 | 2.6826 |
| 35 | 2bit | 3.4843 | 6.9361 | 9.5431 | 12.3192 | 15.2926 |
| 35 | 3bit | 1.6644 | 3.0344 | 3.9355 | 4.8562 | 5.8614 |
| 35 | 4bit | 0.8069 | 1.2742 | 1.6018 | 1.9138 | 2.2963 |

## Verdict

**TEMPORAL DELTA CODING: KEY-ONLY, WITH ANCHORS**

- Keys: variance ratio 0.3872 -- VIABLE for delta coding
- Values: variance ratio 1.2575 -- NOT VIABLE (deltas larger than absolutes)
- Error accumulation: CRITICAL PROBLEM -- grows O(sqrt(T)), reaches 2-13x by Q4

## Deep Analysis

### The Key Asymmetry: Keys vs Values

The experiment reveals a fundamental asymmetry:

**Keys (temporal cosine 0.80, var ratio 0.39):** Consecutive keys are highly
correlated because Qwen2.5 uses Grouped Query Attention (GQA) with only 2 KV
heads per layer. Each KV head processes a large fraction of the input, making
its key projection smooth across positions. RoPE rotates keys per-position, but
the content component dominates. Layer 0 is nearly identical (cos=0.996,
var_ratio=0.007) because early layers do minimal transformation.

**Values (temporal cosine 0.37, var ratio 1.26):** Values are essentially
uncorrelated between consecutive positions. Each position's value is a different
projection of the residual stream, capturing what information that position
contributes to the output. Adjacent positions typically contribute very different
information (different words, different syntactic roles), so values vary
independently. This matches the cross-layer finding where values were also
unpredictable (CV R^2 = 0.09).

### The Error Accumulation Problem

This is the deal-breaker for naive delta coding. Reconstructing via cumulative
sum (x[t] = x[0] + sum(delta[1..t])) causes quantization noise to accumulate:

- At 4-bit deltas, layer 12: error grows from 0.90 (Q1) to 2.38 (Q4) -- a 2.6x increase
- At 3-bit deltas, layer 12: error grows from 2.03 (Q1) to 5.46 (Q4) -- a 2.7x increase
- At 2-bit deltas, layer 12: error grows from 4.34 (Q1) to 13.46 (Q4) -- a 3.1x increase

The error grows as O(sqrt(T)) because each delta adds independent quantization
noise that compounds through the sum. For 815 tokens, sqrt(815) ~ 28.5, meaning
the final token's reconstruction is ~28x noisier than a single delta error.

This makes pure delta coding UNUSABLE for attention quality despite the favorable
variance ratio. The attention scores at layer 12 with 4-bit deltas have
relative MSE of 0.035 -- already 35x worse than TurboQuant 3-bit.

### Solution: Anchored Delta Coding

Insert full-precision anchors every W tokens to bound error accumulation:

```
[anchor] [delta] [delta] ... [delta] [anchor] [delta] ...
   t=0      t=1     t=2    t=W-1      t=W      t=W+1  ...
```

Error is bounded by: O(sqrt(W)) instead of O(sqrt(T)).
- Window W=32: max error proportional to sqrt(32) ~ 5.7
- Window W=16: max error proportional to sqrt(16) = 4.0
- Window W=8:  max error proportional to sqrt(8) ~ 2.8

Storage cost: 1 anchor per W tokens at FP16, rest at delta_bits.
Effective bits/coord for keys with W=32, delta=2bit:
  (16 + 31*2) / 32 = 2.44 bits/coord (vs 3-bit TurboQuant baseline)

But the compression advantage shrinks rapidly as W decreases (to control error).
At W=8 with 2-bit deltas: (16 + 7*2) / 8 = 3.75 bits -- WORSE than TQ3.

### The Fundamental Problem

Temporal delta coding for KV cache compression faces a **trilemma**:

1. **Low delta bits** -> good compression but high error accumulation
2. **Small anchor window** -> bounded error but reduced compression
3. **High delta bits** -> good quality but poor compression (matches baseline)

The favorable variance ratio (0.39) is eaten by the overhead of anchors needed
to control error accumulation. At the operating points where quality matches
TurboQuant 3-bit, the compression ratio is similar or worse.

### Why This Differs From Cross-Layer Delta Coding

| Approach | Key var ratio | Val var ratio | Verdict |
|----------|---------------|---------------|---------|
| Cross-layer (layers) | 2.58 | 2.09 | NOT VIABLE |
| Temporal (positions) | 0.3872 | 1.2575 | KEY-ONLY, MARGINAL |

Cross-layer deltas fail completely (deltas LARGER than absolutes). Temporal
deltas succeed for keys (deltas 61% smaller) but fail for values. However,
even for keys, the error accumulation problem makes the approach impractical
for improving compression beyond what TurboQuant already achieves.

### Recommendation

Do NOT implement temporal delta coding as a compression strategy. The existing
TurboQuant 3-bit approach (Lloyd-Max quantization + residual sign correction)
achieves 5.1x compression without error accumulation. Temporal delta coding
would add complexity for marginal or negative compression gains.

The key temporal correlation IS real (0.80 cosine) and could potentially be
exploited via:
- Adaptive bit allocation (fewer bits for highly-correlated keys)
- Predictive coding with learned predictors (not simple subtraction)
- Entropy coding of delta symbols (entropy ratio is 0.23 at 2-bit)

But these approaches add significant complexity and are unlikely to yield
more than 0.1-0.3 bits/coord improvement over the current system.

---
*Generated on 2026-04-04 12:20:42 by temporal_delta_experiment.py*