# Comment for vLLM Issue #38171 and PR #38280

---

I have been working on an independent TurboQuant implementation (github.com/dhawalc/turboQuantDC, MIT license) and may have found the root cause of the generation quality failures reported here.

## The protocol bug

In my implementation, I traced every generation failure to `get_mask_sizes` -- the Cache protocol method that tells the attention mask builder how long the KV sequence is. If this returns the wrong length, the causal mask is incorrect: the model either attends to padding positions or misses real cached tokens.

The specific bug: `get_mask_sizes` returning `(query_length, 0)` instead of `(cached_length + query_length, 0)`. This is invisible in single-step attention benchmarks (cosine similarity looks fine) but produces garbled output during autoregressive generation.

The fix is one line:

```python
# BEFORE (wrong):
def get_mask_sizes(self, cache_position, layer_idx):
    return cache_position.shape[0], 0

# AFTER (correct):
def get_mask_sizes(self, cache_position, layer_idx):
    kv_length = self._layers[layer_idx].get_seq_length() + cache_position.shape[0]
    return kv_length, 0
```

I noticed that the 0% gsm8k accuracy reported in the issue discussion has the same symptom pattern: per-step attention quality is fine, end-to-end generation produces garbage. This is exactly what a mask bug looks like. If the vLLM TurboQuant integration has its own KV length reporting mechanism, it would be worth verifying that the attention mask sees the correct sequence length at each decode step.

## The 5.1x result with the fix

After fixing the mask bug, I still needed to solve a second problem: QJL (the paper's 1-bit bias correction) adds variance that compounds across layers. This has been independently confirmed by multiple teams. I replaced QJL with a novel approach called ResidualQuant:

**QJL:** project residual through random Gaussian, store signs. Unbiased but high variance.
**ResidualQuant:** store signs of the actual residual in rotated space. Biased but low variance.

Same bit budget (1 bit per coordinate). Same storage format. But ResidualQuant preserves the residual direction instead of destroying it through a random projection.

The full stack that matches FP16 generation quality:

| Component | Bits/dim | Purpose |
|---|---|---|
| 3-bit keys (MSE indices) | 2 | Lloyd-Max codebook |
| 1-bit residual signs | 1 | Direct residual correction |
| 2-bit values (MSE) | 2 | Value reconstruction |
| FP16 window (last 128 tokens) | 16 (0.4% of tokens at 32K) | Error accumulation break |

Effective compression: 5.1x. Tested on Qwen2.5-3B-Instruct, 8 test prompts (factual, math, code, reasoning), all matching FP16 output.

## ResidualQuant approach

The implementation is straightforward. In the quantize path:

```python
# Standard TurboQuant: MSE quantization in rotated space
x_rotated = x_normalized @ rotation_matrix
mse_indices = codebook.quantize(x_rotated)
x_mse_rotated = codebook.centroids[mse_indices]

# ResidualQuant: store actual residual signs (NO random projection)
residual_rotated = x_rotated - x_mse_rotated
residual_signs = sign(residual_rotated)       # 1 bit per coordinate
residual_scale = mean(abs(residual_rotated))   # 16 bits per vector
```

In the dequantize path:

```python
# Reconstruct with residual correction
x_corrected_rotated = x_mse_rotated + residual_scale * residual_signs
x_corrected = x_corrected_rotated @ rotation_matrix.T
```

Drop-in replacement for QJL. No random projection matrix needed. No analytical inner product formula needed (just reconstruct and do standard matmul).

## Potential relevance to vLLM

If the vLLM TurboQuant integration (PR #38280) or the feature request (#38171) are experiencing generation quality issues:

1. Check the attention mask construction. The mask must reflect the actual KV cache length at each decode step.
2. Consider MSE-only or MSE+ResidualQuant instead of full QJL for the key path. QJL's unbiasedness guarantee is about expectations over many samples, but each decode step is a single sample.
3. The FP16 window (keeping recent tokens uncompressed) costs almost nothing at long context and prevents error accumulation in the most-attended positions.

The full implementation is at github.com/dhawalc/turboQuantDC with 568+ tests and MIT license. The ResidualQuant module is at `turboquantdc/residual_quant.py`. The autoresearch sweep that found the optimal configuration is at `autoresearch.py`.
