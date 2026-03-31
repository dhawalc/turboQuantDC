# Comment for llama.cpp Discussion #20969

---

Following up on this discussion with a finding that might be relevant to everyone implementing TurboQuant, including the llama.cpp implementations.

## The get_mask_sizes root cause

I built a from-scratch TurboQuant implementation in PyTorch (github.com/dhawalc/turboQuantDC) and hit the same generation quality wall everyone else has: attention metrics look great (0.9969 cosine similarity at 3-bit), but autoregressive generation produces garbage after about 100 tokens.

After extensive debugging, I traced every generation failure to a single root cause: the attention mask protocol.

In HuggingFace's Cache protocol, `get_mask_sizes(cache_position, layer_idx)` returns `(kv_length, kv_offset)`. The attention mask builder uses `kv_length` to determine which positions the model can attend to. If `kv_length` is wrong -- for example, returning only the query length instead of cached_length + query_length -- the causal mask is incorrect. The model either attends to padding positions or fails to attend to real cached tokens.

This bug is invisible in single-step attention benchmarks (cosine similarity, top-k match) because those benchmarks construct the mask from ground truth. It only manifests during `generate()`.

I cannot speak to how llama.cpp handles this internally since the mask management is different from HF's Cache protocol, but the general pattern is worth checking: is the KV cache length reported to the attention mask builder correct after each decode step? Does it account for both the existing cached tokens and the newly appended tokens?

## ResidualQuant vs QJL data

Confirming what TheTom and others have found: QJL hurts generation. I built a direct replacement called ResidualQuant that uses the same 1-bit budget differently:

**QJL (standard TurboQuant):**
- Compute residual r = x - x_mse in original space
- Project through random Gaussian S, store sign(S @ r)
- Correction uses ||r|| * sqrt(pi/2)/m * <S@q, signs>
- Unbiased inner products, high variance from random projection

**ResidualQuant:**
- Compute residual in rotated space: r_rot = x_rot - centroids[indices]
- Store sign(r_rot) directly, no random projection
- Correction: k_corrected_rot = centroids[indices] + mean(|r_rot|) * sign(r_rot)
- Biased reconstruction, low variance because residual direction is preserved

Generation results on Qwen2.5-3B-Instruct:

| Method | Compression | Generation Quality |
|---|---|---|
| MSE-only 3-bit (no QJL, no ResidualQuant) | 5.0x | Garbled at 100+ tokens |
| MSE + QJL 3-bit (paper's approach) | 5.0x | Worse than MSE-only |
| MSE + ResidualQuant 3-bit | 5.0x | Matches FP16 |

Same bit budget for all three. The difference is entirely in how the 1-bit correction is used.

## The FP16 window

The other key ingredient: keeping the last 128 tokens at FP16 eliminates the error accumulation chain for the most-attended positions. At long context this costs almost nothing (128/32K = 0.4% of tokens). Combined with ResidualQuant, the full stack is:

- 3-bit keys (2-bit MSE indices + 1-bit residual signs)
- 2-bit values (MSE-only, values only need reconstruction)
- FP16 for last 128 tokens
- Effective compression: 5.1x

This produced factually correct, coherent output matching FP16 on 8 test prompts covering factual recall, math, code, and reasoning.

## Validated numbers

All measured on RTX 4090 with real LLM KV caches:

| Model | Bits | Cosine Sim | Top-5 Attn Match | Compression |
|---|---|---|---|---|
| Qwen2.5-3B (d=128) | 3 | 0.9969 | 94.4% | 5.0x |
| Qwen2.5-14B (d=128) | 3 | 0.9964 | 95.3% | 5.0x |
| Qwen3.5-27B (d=256) | 3 | 0.9932 | 100% | 5.2x |

568+ tests, MIT license: github.com/dhawalc/turboQuantDC

The ResidualQuant approach and the full stack configuration were found via an autoresearch sweep (600 configurations, auto-scored overnight). Happy to share details on the methodology if useful for anyone here.
