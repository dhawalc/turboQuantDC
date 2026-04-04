# Cross-Layer KV Cache Prediction Experiment

**Model:** Qwen/Qwen2.5-3B-Instruct
**Prompt tokens:** 235
**Layers:** 36
**Runtime:** 7.6s

## Mission

Can we PREDICT KV cache values across layers instead of storing them?
If layer N's KV can predict layer N+1's KV, we only need to store the residual.

## Results Summary

### Approach 1: Delta Coding

- Key variance ratio (delta/abs): **2.5843**
- Value variance ratio (delta/abs): **2.0858**
- Need < 0.5 for viable delta coding
- Verdict: **NOT VIABLE**

### Approach 2: Linear Predictor (R^2)

| Metric | Keys | Values |
|--------|------|--------|
| Raw R^2 | 0.8193 | 0.6501 |
| Adjusted R^2 | 0.7507 | 0.5174 |
| Cross-validated R^2 | 0.5920 | 0.0912 |
| Random baseline | 0.2717 | 0.2717 |

- Real signal (CV R^2 > random+0.05): Keys=YES, Values=NO
- Verdict: **REAL SIGNAL**

### Approach 3: Per-Head Correlation

- Heads with key cosine > 0.5: **0/2**
- Heads with value cosine > 0.5: **0/2**
- Best key head cosine: **0.0081**
- Best value head cosine: **-0.0012**

### Approach 4: Token-Position Correlation

- Key position cos mean: **0.0135**
- Value position cos mean: **0.0001**
- Key position cos max: **0.0465**
- Value position cos max: **0.0491**

### Approach 5: Subspace Alignment (PCA)

- Key top-16 subspace overlap: **0.1946**
- Value top-16 subspace overlap: **0.1257**
- Verdict: **LIMITED**

### Approach 6: Skip-Layer Correlation

| Skip | Key cos | Value cos |
|------|---------|-----------|
| 1 | 0.0031 | -0.0023 |
| 2 | 0.0068 | 0.0024 |
| 4 | -0.0062 | -0.0000 |
| 8 | 0.0020 | 0.0002 |
| 16 | -0.0075 | -0.0045 |

### Approach 7: Norm vs Direction Decomposition

- Key norm Pearson: **0.2808**
- Value norm Pearson: **0.3510**
- Key direction cosine: **0.0031**
- Value direction cosine: **-0.0023**

## Conclusions

### Mixed Verdict: Keys Have Signal, Values Do Not

The linear predictor reveals an asymmetry between keys and values:

- **Keys:** CV R^2=0.5920 vs random baseline 0.2717 -- 
  genuine signal exists. A learned 128x128 rotation matrix can predict
  ~59% of key variance from the previous layer.
- **Values:** CV R^2=0.0912 vs random baseline 0.2717 -- 
  NO real signal. Value prediction is pure overfitting.

### The Paradox: Zero Cosine But High R^2

How can cosine similarity be ~0 yet a linear predictor work?
The answer is that the predictor learns a **rotation between subspaces**.
Cosine similarity measures whether vectors POINT in the same direction.
But a linear predictor can learn that KV_n in direction X maps to
KV_{n+1} in direction Y -- a completely different direction but still
a deterministic linear relationship.

However, note the raw R^2=0.8193 drops to CV R^2=0.5920
after cross-validation. This means ~28% of the apparent 
signal is overfitting, and only ~72% is real.

### Is Key Prediction Useful for Compression?

Probably not, for several reasons:

1. **Predictor cost:** Each layer pair needs a 128x128 = 64 KB matrix.
   For 35 pairs, that is 2.2 MB of predictor storage -- comparable to
   the KV cache savings themselves.
2. **Residual still large:** Even with 59% variance explained,
   the 41% residual must still be quantized. The residual
   variance reduction translates to maybe 0.5-1 fewer bit at best.
3. **Compute overhead:** Matrix multiply per layer per token during
   decoding adds latency on the critical path.
4. **Values are unpredictable:** Values have no cross-layer signal,
   and values represent most of the KV cache memory.

### Other Six Approaches: Uniformly Negative

1. **Delta coding (var ratio ~2.0):** Deltas are LARGER than absolutes.
2. **Per-head correlation (cos ~ 0.00):** No head is predictable.
3. **Token-position (cos ~ 0.01):** No position is more predictable.
4. **Subspace alignment (overlap ~ 0.15):** PCs rotate between layers.
5. **Skip-layer (cos ~ 0.00):** All distances are independent.
6. **Norm vs direction:** Norms weakly correlated (~0.3), directions at zero.

### Root Cause

Each transformer layer applies a different learned projection (W_K, W_V)
to the shared residual stream. For keys, successive W_K projections
happen to have a partially learnable rotation between them (hence the
linear predictor signal). For values, the V projections are more
independent -- perhaps because different layers' value heads extract
genuinely different features.

### Bottom Line

Cross-layer prediction is **not a viable compression strategy**.
The modest key signal does not justify the overhead. Each layer's KV
cache should be compressed independently, and the only viable cross-layer
optimization is statistical sharing (codebook + rotation), already
implemented in `cross_layer_kv.py`.

---
*Generated on 2026-04-04 09:30:29 by cross_layer_predict.py*