# TurboQuantDC — Handoff (Updated April 13, 2026)

## The One Thing That Matters

Mean-removal turns PPL 9,410 into PPL 7.90 on Qwen2.5-7B. One line of C.
NIAH goes from FAIL at all positions to PASS at all positions.
This fixes the root cause of why TurboQuant 3-bit catastrophically fails on Qwen models.

## Current State

- 125 commits, 1,796+ tests, v0.3.0, MIT license
- GitHub Pages live: https://dhawalc.github.io/turboQuantDC/
- Repo: https://github.com/dhawalc/turboQuantDC
- PR #45 comment posted: https://github.com/TheTom/llama-cpp-turboquant/pull/45

## What's Real (Adversarial Validated)

| Finding | Number | Status |
|---------|--------|--------|
| Mean-removal PPL fix (Qwen 7B) | 9,410 → 7.90 | PROVEN, std<0.001 |
| Mean-removal PPL fix (Qwen 3B) | 60.20 → 11.02 | PROVEN, matches Tom's +62.95 |
| NIAH with mean-removal | FAIL → PASS all positions | PROVEN at 8K |
| llama.cpp turbo3+mean PPL | 7.37 (beats FP16 7.50) | PROVEN in C code |
| Gemma 4 26B at 262K context | 150 tok/s, f16 OOMs | PROVEN |
| Gemma 4 E4B quality | 0.999994 cosine, 100% top-5 | PROVEN |
| CUDA 29x speedup at d=256 | Triton register cliff | PROVEN |
| Asymptotic law: Gini ~ 0.09*ln(n) | R²=0.989 | PROVEN, novel |
| KVSculpt distillation | 0.999 cosine pre-quant | PROVEN |
| Triple stack | 37.9x at 0.93 cosine | PROVEN (honest number) |

## What's Overstated (Corrected)

- "Beats RotorQuant 16.6%" — proxy metric, not PPL. Retracted.
- "59.8x at 0.90" — honest: 20-40x at ~0.89
- Cayley "breakthrough" — +0.002-0.006 on typical layers (layer 0 inflated average)
- Mean-removal novelty — NSNQuant (May 2025) published channel centering first. Our contribution is connecting it to TQ catastrophic failure.

## What's Broken

- Expected Attention on topic shifts: ANTI-correlated (-0.035 Spearman)
- TurboRetrievalCache > 2K tokens: FAISS undertrained, sliding-window loses distant tokens
- V2Cache: PCA whitening amplifies noise in low-variance dimensions
- Layer 0: always needs FP16 anchor
- llama.cpp K quantization: ALL sub-8-bit K types fail in flash attention (not just turbo3). Bug is in K dequantize path, not codebook. Tom's q8_0 K + turbo3 V config avoids this.

## Tom Turney Interaction

- PR #45 comment posted with RTX 4090 benchmarks
- Tom validated ResidualQuant finding on Twitter ("we killed QJL early for the same reason")
- Tom reviewed our work Apr 9 — raised 6 valid points, all addressed
- Follow-up with PPL/NIAH numbers drafted (~/Downloads/TOM_FOLLOWUP.txt) — POST THIS
- Mean-removal C patch on feat/mean-removal-turbo3 branch in tom-llama-cpp
- Tom integrated TriAttention into TQ+ on Apr 9 (additive stacking confirmed)

## Immediate Next Actions

1. POST ~/Downloads/TOM_FOLLOWUP.txt with PPL + NIAH numbers
2. GET HF token for Llama 3.1 8B — run same PPL benchmark
3. ENTER Gemma 4 Good hackathon (deadline May 18, $200K) — build an APPLICATION
4. SUBMIT asymptotic law paper (docs/ASYMPTOTIC_LAW_REPORT.md) to arxiv/ICML workshop
5. INVESTIGATE llama.cpp K dequantize flash attention bug — could be a major contribution
6. UPDATE Tom follow-up with the K quantization finding (all sub-8-bit K fails, not just turbo3)

## Strategic Direction

TurboQuantDC is a research contribution and credential, not a standalone product.

Best paths (ranked):
1. Upstream mean-removal + novel techniques to llama.cpp/vLLM (52M+ monthly users)
2. Portfolio for inference engineering roles ($300-600K at Inferact/cloud providers)
3. Gemma 4 hackathon ($10-50K near-term, deadline May 18)
4. Publish asymptotic law paper (career capital, genuinely novel)

## Genuinely Novel (No Prior Art)

1. Asymptotic compression law: Gini ~ 0.09*ln(n), O(1/n) min bits, R²=0.989
2. Triple-stack pipeline: eviction + distillation + quantization (first benchmarked, losses stack additively)
3. Attention-KL rotation objective (Cayley): novel objective, modest practical gain
4. Connecting mean-removal to TQ catastrophic failure on Qwen models

## Key Files

| File | What | Status |
|------|------|--------|
| visualization/index.html | GitHub Pages showcase (PPL 9,410→7.90 hero) | DEPLOYED |
| README.md | Repo front page (PPL fix + research) | DEPLOYED |
| docs/ASYMPTOTIC_LAW_REPORT.md | Technical report for arxiv | READY |
| docs/RESEARCH_LANDSCAPE.md | 40-paper competitive analysis | REFERENCE |
| benchmarks/results/ppl_for_tom.md | PPL numbers for Tom | SHARE |
| benchmarks/results/niah_for_tom.md | NIAH results for Tom | SHARE |
| benchmarks/results/adversarial_validation.md | Honest validation | REFERENCE |
| turboquantdc/generation_core.py | Production cache | STABLE |
| turboquantdc/expected_attention.py | EA pruning | NEEDS shift guard |
| turboquantdc/cache_distillation.py | KVSculpt | STANDALONE |
| turboquantdc/cayley_quant.py | Learned rotation | EXPERIMENTAL |
| turboquantdc/block_rotation.py | Givens/Quaternion | STABLE |
| turboquantdc/learned_quant.py | Differentiable quant | EXPERIMENTAL |
| ~/Downloads/TOM_FOLLOWUP.txt | Reply to Tom with PPL/NIAH | POST THIS |
| ~/Downloads/COMPLETE_SESSION_SUMMARY.md | Full session record | REFERENCE |
| ~/Downloads/STRATEGIC_ANALYSIS.md | Business strategy | REFERENCE |
| ~/Downloads/WHAT_IS_GENUINELY_NOVEL.md | Novelty map vs literature | REFERENCE |

## llama.cpp Branches (on /home/dhawal/tom-llama-cpp)

- feat/residualquant-rq3: GGML_TYPE_RQ3_0 (CPU-only, 12 files, compiles clean)
- feat/mean-removal-turbo3: Mean-removal in turbo3/4/2 + CUDA kernel (K-only)

## Session Stats

- 125 commits
- 47 source modules
- 1,796+ tests
- 20 research experiments (8 breakthroughs, 4 dead ends, 8 techniques)
- 15+ charts created
- 40 papers analyzed
- 3 models downloaded and cleaned up
- 10 models benchmarked (3B through 72B + Gemma 4)
