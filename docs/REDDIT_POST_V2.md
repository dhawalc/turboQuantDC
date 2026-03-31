# Reddit Post V2 -- r/LocalLLaMA

## Title

**I found the root cause of why TurboQuant generation fails everywhere -- and fixed it. 5.1x KV cache compression matching FP16 quality.**

---

## Body

Six days ago I posted about TurboQuantDC, a from-scratch implementation of Google's TurboQuant algorithm for KV cache compression. The attention metrics were good (0.9969 cosine similarity at 3-bit, 94.4% top-5 match). But autoregressive generation was broken at 3-bit, producing garbled output after about 100 tokens. Every other TurboQuant implementation had the same problem.

I spent the last three days trying to fix it. What I found surprised me: every failure had a single root cause, and it was not the algorithm.

Repo: [github.com/dhawalc/turboQuantDC](https://github.com/dhawalc/turboQuantDC) | PyPI: `pip install turboquantdc` | [Colab](https://colab.research.google.com/github/dhawalc/turboQuantDC)

### The root cause: get_mask_sizes

HuggingFace's Cache protocol requires a `get_mask_sizes(cache_position, layer_idx)` method that returns `(kv_length, kv_offset)`. The attention mask is built from these values. If `kv_length` is wrong, the mask is wrong, and the model attends to garbage positions or fails to attend to real ones.

In every custom cache implementation I checked -- ours, and what I could find in other TurboQuant repos -- this method was either missing, returning `(query_length, 0)` instead of `(cached_length + query_length, 0)`, or not accounting for the fact that `update()` has already appended the new tokens to the cache before `get_mask_sizes` is called.

The fix is one line. Before:

```python
def get_mask_sizes(self, cache_position, layer_idx):
    return cache_position.shape[0], 0  # WRONG: returns query length only
```

After:

```python
def get_mask_sizes(self, cache_position, layer_idx):
    kv_length = self._layers[layer_idx].get_seq_length() + cache_position.shape[0]
    return kv_length, 0  # RIGHT: returns total KV length including cached tokens
```

This bug does not affect single-step attention metrics (cosine similarity, top-k match), because those benchmarks build the mask correctly from ground truth. It only manifests during autoregressive generation, where HF's `generate()` calls `get_mask_sizes` to build the causal mask.

I suspect the vLLM maintainer who reported 0% gsm8k accuracy with TurboQuant-3 hit the same issue. The symptom is identical: attention-level metrics look perfect, generation produces garbage.

### The autoresearch story

After the mask fix, I still was not getting FP16-matching quality because there was a second problem: QJL (the bias-correction stage of TurboQuant) adds variance that compounds across layers during generation. This was known in the community (TheTom, DEJAN, 0xSero all reported it), but nobody had a replacement that used the same bit budget.

So I built an autoresearch loop. The idea is simple: load the model once, sweep 600 configurations of (key_bits, val_bits, anchor_interval, fp16_window, use_residual_quant) combinations, auto-score each on 8 test prompts (factual recall, math, code, reasoning), and report the Pareto frontier of compression vs quality.

The sweep includes a novel approach I called ResidualQuant: instead of QJL's random projection + signs, store the actual sign of each residual coordinate in rotated space. Same bit budget (1 bit per coordinate), but no random projection noise. The trade is: QJL gives you unbiased inner products with high variance. ResidualQuant gives you biased reconstruction with low variance. For generation, low variance wins.

The autoresearch ran overnight. Results across rounds:

- Round 0: K4/V4 with anchors, 3.1x compression, coherent but heavily repetitive
- Round 3: K4/V2 no anchors + ResidualQuant, 5.0x compression, mostly coherent
- Round 6: K3/V2 + ResidualQuant + FP16 window, 5.1x compression, matches FP16 on all 8 prompts

The winning configuration: 3-bit keys with 1-bit residual signs + 2-bit values + last 128 tokens at FP16 = 5.1x compression. Every test prompt produced factually correct, coherent output indistinguishable from the FP16 baseline.

### ResidualQuant: the novel contribution

Standard TurboQuant (from the ICLR 2026 paper) uses QJL for its second stage:

1. Compute residual: r = x - x_mse
2. Project through random Gaussian matrix S
3. Store sign(S @ r) -- 1 bit per dimension
4. Correction formula: ||r|| * sqrt(pi/2)/m * <S@q, signs>

This is mathematically unbiased (E[error] = 0) but the random projection adds variance proportional to pi/(2m) * ||y||^2 per attention score. That variance compounds across 36 layers of softmax.

ResidualQuant does this instead:

1. Compute residual in rotated space: r_rot = x_rot - centroids[indices]
2. Store sign(r_rot) -- 1 bit per coordinate, no random projection
3. Correction: k_corrected_rot = centroids[indices] + scale * sign(r_rot)

Same storage (1 bit per coordinate + 16-bit scale). But the sign of the actual residual preserves its direction perfectly -- just not its per-coordinate magnitude. QJL's random projection destroys direction information while gaining unbiasedness. For generation, this is a bad trade.

The numbers:

| Method | Compression | Attention CosSim | Generation Quality |
|---|---|---|---|
| MSE-only 3-bit | 5.0x | 0.9959 | Garbled at 100+ tokens |
| MSE + QJL 3-bit | 5.0x | 0.9969 | Worse than MSE-only |
| MSE + ResidualQuant 3-bit | 5.0x | 0.9955 | Matches FP16 |
| Full stack (K3-RQ + V2 + FP16 window) | 5.1x | -- | Matches FP16 on all 8 test prompts |

### The full compression stack

| Component | Bits | Role |
|---|---|---|
| 3-bit keys (MSE) | 2 bits/dim | Lloyd-Max codebook indices |
| 1-bit residual signs | 1 bit/dim | Direct residual correction |
| 2-bit values (MSE) | 2 bits/dim | Value reconstruction |
| FP16 window | 16 bits/dim, last 128 tokens | Recent tokens at full precision |
| Per-vector norms | 16 bits/vector | Scale factors |

Effective compression: 5.1x overall (at 2K context; higher at longer context as the FP16 window fraction shrinks).

### What this means for the community

1. **The mask bug might affect other implementations.** If you are implementing a custom HF Cache and your generation quality is bad despite good attention metrics, check your `get_mask_sizes`. This is the single most common failure mode I have seen.

2. **QJL is dead for generation.** The community consensus is confirmed: MSE-only or MSE+ResidualQuant beats MSE+QJL for autoregressive generation. The paper's unbiasedness guarantee is about expectations, not individual samples, and each decode step is a single sample.

3. **ResidualQuant is a strictly better use of the 1-bit budget.** Same storage, better generation quality. I have not seen this exact approach in any other implementation.

4. **The FP16 window is cheap insurance.** Keeping the last 128 tokens at FP16 costs almost nothing at long context (128/32K = 0.4% of tokens) but eliminates the error accumulation chain for the most recent and most-attended positions.

### Limitations

I am being honest about what this is and what it is not.

- All generation quality tests are on Qwen2.5-3B-Instruct. I have not run lm-eval benchmarks or tested on Llama/Mistral architectures.
- The "matches FP16" claim is based on 8 test prompts (factual, math, code, reasoning). A proper lm-eval sweep would be more rigorous.
- The 5.1x compression ratio depends on context length. At very short context, the FP16 window is a large fraction and effective compression is lower.
- No fused GPU kernels. Pure PyTorch. Throughput is not production-grade.
- Only tested on Qwen-family models so far. The algorithm is architecture-agnostic in theory, but empirical validation on other families is missing.

### Credit

TheTom (turboquant_plus, llama.cpp Metal), 0xSero (Triton/vLLM), DEJAN (Triton kernel blog), hackimov (fused attention), tonbistudio (reference implementation), scos-lab (8-model benchmark). The QJL-hurts-generation finding was independently confirmed by multiple teams before I got there. The get_mask_sizes fix and ResidualQuant are, to my knowledge, novel.

### Build process

Built by 40+ AI agents coordinated in a single session over 6 days. If that sounds unusual, it is. The full war room transcript (92+ messages) is in the repo at `docs/WARROOM_TRANSCRIPT.md`. The autoresearch dashboard that ran overnight is at `autoresearch_dashboard.py`.

568+ tests, 21 source modules, 8,859 lines of library code, MIT license.

### Links

- Repo: [github.com/dhawalc/turboQuantDC](https://github.com/dhawalc/turboQuantDC)
- PyPI: `pip install turboquantdc`
- Paper: [arxiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [Colab notebook](https://colab.research.google.com/github/dhawalc/turboQuantDC) (interactive demo)
