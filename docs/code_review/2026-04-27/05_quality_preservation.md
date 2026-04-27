# Quality Preservation Review (Opus 4.7)

Reviewer: Opus 4.7 (1M context)
Date: 2026-04-27
Scope: 15 quality-preservation modules, mean-removal audit, vLLM 27B integration recommendations.

## Headline Recommendation for the 6-Hour Qwen3.6-27B vLLM Demo

The ONE thing that must ship: **mean-removal at the per-head running mean, plumbed end-to-end through `_CompressedLayer.update`**. The current `vllm_integration.TurboQuantAttentionBackend` does NOT use it (uses raw `TurboQuantEstimator`). This is a critical gap. Without mean-removal on Qwen, the 7B benchmark shows PPL 13,225 (catastrophic). Everything else listed below is secondary.

Order of operations for the demo:
1. (CRITICAL, ~30 min) Wire `vllm_integration` to `_CompressedLayer` / `ResidualQuantCache` instead of raw `TurboQuantEstimator`. Or replace it entirely with a `GenerationCache.from_preset("hybrid_max_quality")` adapter.
2. (~15 min) Use `anchor_strategy="boundary"` from `GenerationCache` (built into the production preset). Cheap, validated.
3. (~30 min) Verify `center_before_quantize=True` propagates through every cache layer.
4. SKIP everything else for the demo. Test in-progress techniques afterward.

## Technique Catalog

| Technique | File | Status | Recommendation for Qwen3.6-27B vLLM demo |
|---|---|---|---|
| Mean-removal (running per-head) | `residual_quant.py`, `generation_layers.py`, `attention_optimal.py` (`MeanRemovedQuantizer`) | PRODUCTION | INCLUDE — non-negotiable. The single biggest delta in the entire repo. |
| Layer-adaptive (anchor_strategy="boundary"/"gradient") | `generation_strategy.py` + `generation_core.py` | PRODUCTION | INCLUDE — already used by `hybrid_max_quality` preset. Cheap. |
| Layer-adaptive (`LayerAdaptiveKVCache`) | `layer_adaptive.py` | ARCHIVE | SKIP — uses old `TurboQuantKVCache`, lacks mean-removal, duplicates anchor strategies. Dead path. |
| Channel-adaptive (`ChannelAdaptiveCache`) | `channel_adaptive.py` | KEEP-RESEARCHING | SKIP — does NOT use mean-removal or ResidualQuant; bypasses the production pipeline. No PPL benchmark. |
| Outlier (`OutlierTurboQuant`) | `outlier.py` | KEEP-RESEARCHING | SKIP — fractional bit rates, isolated from production. No mean-removal. No real-model benchmark. |
| Adaptive bits / `AdaptiveBitsCache` | `adaptive_bits.py` | KEEP-RESEARCHING | SKIP — needs ground-truth attention to score importance, runs offline reclassify, no mean-removal. Tested only on cosine sim, not PPL. |
| Expected Attention | `expected_attention.py` | ARCHIVE (broken) | SKIP — confirmed broken on distribution shift (Spearman -0.035 in adversarial validation). HANDOFF says "NEEDS shift guard". Anti-correlated on real workloads. |
| Entropy coding (ANS, zlib, LZMA) | `entropy_coding.py`, `entropy_analysis.py` | KEEP-RESEARCHING | SKIP for inference — 4-6% lossless gain at 3-bit, but ANS/zlib decode is on the critical path of every cache read. Cost > gain at decode time. Use for cold storage / serialization only. |
| Learned quantization | `learned_quant.py` | ARCHIVE | SKIP — `wht_mean_removal` (no learning) beats `learned_full` and `learned_rotation_mean` on cosine, KL, top-5, Spearman. Calibration is wasted. |
| Sparse-V attention | `sparse_v.py` | KEEP-RESEARCHING | SKIP — wraps the OLD `TurboQuantKVCache`, not `GenerationCache`. Quality story is "0 PPL delta at 32K" but never re-validated against the mean-removal pipeline. Real win is at 32K+ context, not the demo's 27B short-context use. |
| Cayley quant | `cayley_quant.py` | ARCHIVE | SKIP — HANDOFF says "+0.002-0.006 on typical layers (layer 0 inflated average)". Real but tiny. Calibration cost not justified. |
| Cross-head compress | `cross_head_compress.py` | ARCHIVE (broken hypothesis) | SKIP — measured inter-head correlation is 0.12 (keys) and 0.005 (values). Variance ratio of deltas > 1, meaning deltas are LARGER than the originals. The premise is false on Qwen GQA. |
| Code retrieval | `code_retrieval.py` | KEEP-RESEARCHING | SKIP — clever idea (quantization codes as LSH) but works as a retrieval index, not a quality preservation technique. Orthogonal to the demo. |
| PCA code retrieval | `pca_code_retrieval.py` | KEEP-RESEARCHING | SKIP — research finding (binary PCA hash beats whitened PCA). Same scope as `code_retrieval.py`. |
| Retrieval attention | `retrieval_attention.py` | KEEP-RESEARCHING | SKIP — requires sparsity that does not exist below ~16K context (HANDOFF: "TurboRetrievalCache > 2K tokens: FAISS undertrained"). Not relevant to 27B short-context demo. |
| Evolving compressor | `evolving_compressor.py` | ARCHIVE | SKIP — autoresearch playground, frozen at "PolarQuant + norm correction" with NO mean-removal. The `.backup` file in the dir is the giveaway. |

## Mean-Removal Implementation Audit

### Where it lives

The HANDOFF claim ("PPL 9,410 → 7.90 on Qwen2.5-7B") refers to mean-removal in 4 places:

1. **`turboquantdc/residual_quant.py:152`** (`ResidualQuantEstimator.quantize`)
   - Centers per-batch: `vec_mean = x.mean(dim=0, keepdim=True).expand_as(x)`.
   - This is the "primitive". Stored as 16-bit overhead per vector.

2. **`turboquantdc/residual_quant.py:386-397`** (`ResidualQuantLayer.update`)
   - Online running mean per (batch, head): `mean_new = (mean_old * n + sum_new) / (n + new_seq)`.
   - Updates as new tokens stream in. This is the correct shape.

3. **`turboquantdc/generation_layers.py:454-471`** (`_CompressedLayer.update` and `compress_only`)
   - Same online formula. This is the path used by `GenerationCache` (the production cache).
   - Snapshot of the running mean is stored per chunk (`self._key_means.append(...)`) for dequantization. **Storage overhead is per-token, not per-head**: this is a memory cost the docstrings do not advertise.

4. **`turboquantdc/attention_optimal.py:153`** (`MeanRemovedQuantizer`)
   - Standalone benchmark wrapper. Not used in production.

### Default state

`GenerationCache(center_before_quantize=True)` is the default (`generation_core.py:170`). Both `hybrid_max_quality` and `hybrid_max_compression` presets inherit this default. Good.

### Where it is NOT wired

- **`turboquantdc/vllm_integration.py`**: Uses raw `TurboQuantEstimator` (line 108). NO mean-removal. THIS IS A SHIP-BLOCKER for the demo if the demo path goes through `vllm_integration`. (The HANDOFF acknowledges "Don't try to integrate with vLLM/SGLang until the core algorithm is validated" — that integration was written before mean-removal landed.)
- **`turboquantdc/channel_adaptive.py`**: No mean-removal in `_AdaptiveCompressedLayer`.
- **`turboquantdc/adaptive_bits.py`**: No mean-removal in `AdaptiveBitsCache`.
- **`turboquantdc/layer_adaptive.py`**: Uses old `TurboQuantKVCache`, no mean-removal.
- **`turboquantdc/outlier.py`**: No mean-removal.
- **`turboquantdc/evolving_compressor.py`**: No mean-removal.
- **`turboquantdc/cross_head_compress.py`**: No mean-removal.

### Does the implementation match the claim?

**Yes, for the production path.** The flow `GenerationCache → _CompressedLayer → mean-centered keys → ResidualQuant → cached compressed indices + per-chunk mean → dequant + add mean back` is correct, and adversarial validation (`benchmarks/results/adversarial_validation.md` lines 147-164) shows mean-removal "NEVER hurts across 5 diverse prompts" with +0.5% to +2.8% absolute cosine lift on real Qwen2.5-3B. The 7B PPL 13,225 → 9.06 result in `rotorquant_comprehensive.md` (line 64-65) is also consistent.

### Storage cost (HANDOFF says 36 KB; reality says larger)

The HANDOFF docstring claim is `2*d` bytes per head per layer (~36 KB total), but `_CompressedLayer` stores a SNAPSHOT of the running mean **per token chunk** (`self._key_means.append(chunk_mean.to(...))`, line 483). This is `(B, H, new_seq, D)` per chunk. Over N tokens, this is `N * H * D * 2` bytes — the same order as the unquantized FP16 KV. This is a quiet bug-or-feature. For 27B at 32K context, that is ~200 MB just for the mean snapshots. **The 36 KB figure is wrong unless they squeeze the snapshot dimension to (B, H, 1, D).** See HIGH-1 below.

## CRITICAL findings

### CRIT-1: vLLM integration bypasses mean-removal entirely
- File: `turboquantdc/vllm_integration.py:108-179`
- Issue: `TurboQuantAttentionBackend` instantiates `TurboQuantEstimator` (raw QJL). No mean-centering. This is the public API the README and integration docs point at, but it is exactly the configuration that produced PPL 13,225 on Qwen2.5-7B.
- Fix: Replace `TurboQuantEstimator` with the `_CompressedLayer` path or `ResidualQuantCache`. If a vLLM-native attention backend is needed, port the running-mean update inside `compress_kv`.
- Impact for demo: If `demo_27b` invokes vLLM through this module, the demo is going to look broken. Must be fixed before showing the model to anyone.

### CRIT-2: `vec_mean` stored per-batch instead of per-head in `ResidualQuantEstimator.quantize`
- File: `turboquantdc/residual_quant.py:153`
- Issue: `vec_mean = x.mean(dim=0, keepdim=True).expand_as(x)`. When `x` is `(batch, d)` with batch == one head's tokens, this is correct. But the docstring says "per-head mean" and downstream `_CompressedLayer` flattens `(B, H, T, D)` to `(B*H*T, D)` before calling `quantize`. So `mean(dim=0)` is taken over **all heads and all tokens at once**, giving a single vector — not a per-head mean.
- However, `_CompressedLayer.update` (`generation_layers.py:456`) does the per-head running mean correctly BEFORE flattening, and tells `ResidualQuantEstimator` to use `center_before_quantize=False` (`residual_quant.py:362`). So the production path is OK. But anyone calling `ResidualQuantEstimator.quantize` directly with `center_before_quantize=True` and stacked-head input will get a wrong (cross-head) mean.
- Fix: Either delete `center_before_quantize` from `ResidualQuantEstimator` (make centering only the layer-level concern), or document loudly that `quantize` expects a single head's worth of vectors.

## HIGH

### HIGH-1: Mean snapshot stored per-token, not per-head — quietly inflates memory
- File: `turboquantdc/generation_layers.py:469-471`, `483`
- Issue: `chunk_mean = self._key_running_mean.expand(B, H, new_seq, D).clone()`. The `.expand(...).clone()` materializes the broadcasted snapshot and appends to `self._key_means`. Memory is `O(N * H * D)` even though the actual unique data is `O(H * D)`.
- Fix: Store one `(B, H, 1, D)` snapshot per chunk and broadcast at dequant time. Drop the `.clone()` and store the `(B, H, 1, D)` original. This is a one-liner that recovers ~all of the supposed "negligible 36 KB" claim.

### HIGH-2: Channel-adaptive cache silently does WORSE than mean-removal alone
- File: `turboquantdc/channel_adaptive.py:415-422`, full pipeline
- Issue: `ChannelAdaptivePolarQuant` is built around the assumption that "after random orthogonal rotation, different coordinate positions have different sensitivity". This is provably false for a strictly orthogonal rotation: the channels are exchangeable, by symmetry. The sensitivity it measures (`analyze_channel_sensitivity`) on synthetic N(0,1/d) data will produce essentially uniform values up to noise. Their "boost top 25% to 4-bit" is then noise-driven channel selection.
- Empirically: there is no PPL benchmark for `ChannelAdaptiveCache` in `benchmarks/results/`. The only empirical signal (`adaptive_bits_results.md`) is for the OTHER adaptive-bits scheme (token tier, not channel tier).
- Fix: Either drop or pivot to a non-rotated channel allocator (which would require coordinated changes to the rotation step). Do not include in the demo.

### HIGH-3: Expected Attention is broken on real workloads — keep, but flag
- File: `turboquantdc/expected_attention.py`
- Evidence: `adversarial_validation.md:130` shows Spearman = -0.035 on distribution shift (worse than random); 0.083 on uniform attention (random); only 0.866 on synthetic power-law data.
- The internal benchmark (`expected_attention_results.md`) shows Spearman 0.83 on Qwen2.5-3B but admits Layer 27 drops to 0.05 EMA / 0.25 EA. The HANDOFF and adversarial validation are both correct: this technique fails on real conversation patterns.
- Fix: Either delete the module or guard it behind a "EXPERIMENTAL — fails on distribution shift" flag and remove from `__init__.py` exports. Do not include in the demo path.

### HIGH-4: Cross-head compression hypothesis is empirically false
- File: `turboquantdc/cross_head_compress.py`
- Evidence: `cross_head_results.md:18-23` shows mean inter-head cosine is 0.12 (keys) and 0.005 (values), with variance ratio > 1 across all tested layers. The deltas are LARGER than the originals on Qwen2.5-3B GQA.
- Fix: Archive. Do not include.

### HIGH-5: Layer adaptive scheme `LayerAdaptiveKVCache` is dead code
- File: `turboquantdc/layer_adaptive.py`
- Issue: Uses old `TurboQuantKVCache` (no mean-removal, no residual signs). The production `GenerationCache.anchor_strategy` superseded this. Two competing implementations of the same idea.
- Fix: Mark deprecated or delete. Do not include.

### HIGH-6: Entropy coding has wrong cost model for inference
- File: `turboquantdc/entropy_coding.py`
- Issue: 4-6% lossless savings at 3-bit. ANS/zlib decode adds latency to every cache READ (each token in attention computation). At decode-time throughput, this is many milliseconds per layer per token for a sub-10% memory win.
- The benchmark in `entropy_coding_results.md` measures only encoded SIZE, not decode latency. The "RECOMMEND: 5-10% free compression is meaningful at scale" is true for cold storage, false for the 27B vLLM hot path.
- Fix: Reserve for serialization use case only. Skip in the demo.

## MEDIUM

### MED-1: Mean-removal docstrings claim "+1 free bit", which is overstated
- Files: `residual_quant.py:55`, `generation_layers.py:451`
- Reality: adversarial validation shows +0.5% to +2.8% absolute cosine sim, not +1 bit (which would require a step change of ~+0.05 cosine to match a +1 bit move). The "FREE +1 bit" framing is marketing, not measurement. The PPL story (9,410 → 7.90) is real, but it's not a "free bit"; it's "Qwen models are catastrophically broken without it; the catastrophe is fixed by it". Two different claims.
- Fix: Update docstrings to match reality. Recommended language: "Mean-removal is mandatory for Qwen2.5/Qwen3 models. Without it, Qwen2.5-7B PPL diverges to 13,225 at 3-bit. With it, PPL is 9.06 (ref `benchmarks/results/rotorquant_comprehensive.md`)."

### MED-2: Outlier module duplicates rotation/QJL machinery without sharing the test path
- File: `turboquantdc/outlier.py`
- Issue: Re-implements rotation and QJL stages instead of composing them. Means it does not benefit from any of the production pipeline's improvements (mean-removal, ResidualQuant, Triton kernels). Stale code.
- Fix: Either rewrite atop `_CompressedLayer` or archive. Don't ship.

### MED-3: Importance scorer in `adaptive_bits.py` requires attention weights, but production path uses Flash Attention
- File: `turboquantdc/adaptive_bits.py:72-121`
- Issue: `update()` requires the materialized softmax `(batch, heads, q, kv)` matrix. Flash Attention does not materialize this. So the importance scorer cannot be fed real attention weights from a production model without disabling Flash. This is a fundamental integration problem the module does not address.
- Fix: Document the requirement. Provide either an EA-style query-statistics surrogate (broken per HIGH-3) or a sampled-attention path. Do not include in the demo.

### MED-4: Sparse-V attention wraps the old cache interface
- File: `turboquantdc/sparse_v.py:54`
- Issue: Takes `TurboQuantKVCache` (old `kv_cache.py`), not `GenerationCache`. Cannot stack with mean-removal or anchor strategies. Quality result was measured against a cache that does not exist in production.
- Fix: Port to `_CompressedLayer` and re-validate, OR archive. Skip in the demo.

### MED-5: Code/PCA/retrieval attention modules ignore mean-removal entirely
- Files: `code_retrieval.py:340`, `pca_code_retrieval.py`, `retrieval_attention.py`
- Issue: All set `center_before_quantize=False` or never use a centered estimator. Means any retrieval comparison against the production cache is apples-to-oranges.
- Fix: Out of scope for the demo. Tag for re-validation.

## LOW

- **LOW-1**: `cayley_quant.py` constructs a `(d, d)` matrix every forward pass via `torch.linalg.solve`. Fine for offline calibration, terrible for hot-path attention. Module docstring promises "Cache R during inference; only recompute during calibration" — verify the cache path is implemented before any demo claim of "Cayley speedup".
- **LOW-2**: `evolving_compressor.py.backup` exists in the source tree. Either delete or move to `.git`-ignored.
- **LOW-3**: `learned_quant.py` straight-through estimator is correct but the calibration loss (KL) is not the same surface as wikitext PPL. The benchmark numbers cannot be directly compared to PPL claims.
- **LOW-4**: `expected_attention.py:506` does `from .adaptive_bits import ImportanceScorer` inside `compare_scorers` — circular import risk. Hoist to module level if the scorer survives the archive purge.
- **LOW-5**: `entropy_coding.py:73` builds the `_symbol_probabilities` table via `scipy.integrate.quad` per symbol. One-time cost, but should be cached on the codebook object alongside the centroids.

## Closing recommendation

The 6-hour budget is well-spent on mean-removal end-to-end and a careful smoke test. Adding any single one of {channel-adaptive, expected-attention, entropy-coding, sparse-V, retrieval, cross-head} during the demo window introduces risk without proven Qwen3.6 quality numbers. The "research" pile is genuine research — keep it for the publication, not the demo.

If there is residual time after the vLLM integration is fixed, the highest-confidence add-on is:
- Confirm `anchor_strategy="boundary"` works at 27B (HANDOFF + 32B benchmark say boundary alone is not enough at 32B/64-layer; 27B is similar). Consider `"gradient"` instead, which keeps the first/last 10% of layers at FP16 and graduates the rest. Has not been benchmarked at 27B but is extremely cheap to test.

Files referenced (all absolute):
- /home/dhawal/turboQuantDC/HANDOFF.md
- /home/dhawal/turboQuantDC/turboquantdc/residual_quant.py
- /home/dhawal/turboQuantDC/turboquantdc/generation_core.py
- /home/dhawal/turboQuantDC/turboquantdc/generation_layers.py
- /home/dhawal/turboQuantDC/turboquantdc/generation_strategy.py
- /home/dhawal/turboQuantDC/turboquantdc/vllm_integration.py
- /home/dhawal/turboQuantDC/turboquantdc/outlier.py
- /home/dhawal/turboQuantDC/turboquantdc/layer_adaptive.py
- /home/dhawal/turboQuantDC/turboquantdc/channel_adaptive.py
- /home/dhawal/turboQuantDC/turboquantdc/adaptive_bits.py
- /home/dhawal/turboQuantDC/turboquantdc/expected_attention.py
- /home/dhawal/turboQuantDC/turboquantdc/entropy_analysis.py
- /home/dhawal/turboQuantDC/turboquantdc/entropy_coding.py
- /home/dhawal/turboQuantDC/turboquantdc/learned_quant.py
- /home/dhawal/turboQuantDC/turboquantdc/sparse_v.py
- /home/dhawal/turboQuantDC/turboquantdc/cayley_quant.py
- /home/dhawal/turboQuantDC/turboquantdc/cross_head_compress.py
- /home/dhawal/turboQuantDC/turboquantdc/code_retrieval.py
- /home/dhawal/turboQuantDC/turboquantdc/pca_code_retrieval.py
- /home/dhawal/turboQuantDC/turboquantdc/retrieval_attention.py
- /home/dhawal/turboQuantDC/turboquantdc/evolving_compressor.py
- /home/dhawal/turboQuantDC/turboquantdc/attention_optimal.py
- /home/dhawal/turboQuantDC/benchmarks/mean_removal_benchmark.py
- /home/dhawal/turboQuantDC/benchmarks/results/adversarial_validation.md
- /home/dhawal/turboQuantDC/benchmarks/results/mean_removal_integration_results.md
- /home/dhawal/turboQuantDC/benchmarks/results/rotorquant_comprehensive.md
- /home/dhawal/turboQuantDC/benchmarks/results/cross_head_results.md
- /home/dhawal/turboQuantDC/benchmarks/results/expected_attention_results.md
- /home/dhawal/turboQuantDC/benchmarks/results/adaptive_bits_results.md
- /home/dhawal/turboQuantDC/benchmarks/results/entropy_coding_results.md
- /home/dhawal/turboQuantDC/benchmarks/results/learned_quant_results.md
- /home/dhawal/turboQuantDC/benchmarks/results/cayley_quant_results.md
- /home/dhawal/turboQuantDC/benchmarks/results/32b_long_generation.md
