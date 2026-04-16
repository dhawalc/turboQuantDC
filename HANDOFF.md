# TurboQuantDC — Handoff (Updated April 15, 2026)

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
| WHT beats all rotations on PPL (3 models) | 5.54/9.06/11.87 vs best RQ 18.39/83.92/49.85 | PROVEN, 3B/7B/14B |
| Mean-removal critical for low KV heads | 7B: 13,225→9.06 (2-4 heads), 14B: neutral (8 heads) | PROVEN, KV-head dependent |
| Mean-removal HURTS block rotations on 14B | IsoQuant 18.39→30.87 with mean (+68%) | PROVEN, unexpected |
| Attn cosine sim misleads on PPL | worst attn cos → best PPL across all models | PROVEN, counterintuitive |
| **E8 lattice VQ** (new technique) | **PPL +0.1% on 3B, +0.8% on 7B** at 3-bit | **PROVEN, near-lossless** |
| E8 vs scalar Lloyd-Max MSE | 86-89% lower MSE at same bit rate | PROVEN on synthetic+real |
| **E8 2-bit viable** | PPL +1.3% (3B), +3.5% (7B) at 8x compression | PROVEN (scalar 2-bit is +22-29%) |
| **E8 3-bit near-lossless** | FP16 weights: +0.001% (3B), +0.20% (7B) | PROVEN on FP16 weights |
| E8 3-bit beats FP16 on BnB | 7B: -0.08%, Mistral: -0.02% (BnB regularization) | PROVEN, BnB-specific |
| E8 on 14B improved | +1.53% → +0.53% via scale optimization | PROVEN, 3x better |

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
2. **SHARE** RotorQuant comparison results (benchmarks/results/rotorquant_comprehensive.md) — mean-removal universality is publishable
3. GET HF token for Llama 3.1 8B — run same PPL benchmark (verify on Llama where RotorQuant claims advantage)
4. ~~ENTER Gemma 4 Good hackathon (deadline May 18, $200K)~~ — SKIPPED (unavailable)
5. SUBMIT asymptotic law paper (docs/ASYMPTOTIC_LAW_REPORT.md) to arxiv/ICML workshop
6. INVESTIGATE llama.cpp K dequantize flash attention bug — could be a major contribution
7. UPDATE Tom follow-up with the K quantization finding (all sub-8-bit K fails, not just turbo3)
8. **PROPOSE** mean-removal as standard preprocessing to RotorQuant repo (scrya-com/rotorquant Issue or PR)
9. **PUBLISH** E8 lattice VQ arXiv report — need: (a) E8P encoding for actual compression, (b) Llama-3.1-8B results, (c) speed benchmarks. Target: arXiv in 2 weeks, NeurIPS 2026 workshop
10. **COMPETE** against NestQuant (ICML 2025, Gosset lattice on KV) — our differentiator: calibration-free + WHT + mean-removal

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

## April 15 Session — New Files

| File | What | Status |
|------|------|--------|
| turboquantdc/e8_lattice.py | **E8 lattice VQ** (near-lossless 3-bit) | NEW, 22 tests |
| tests/test_e8_lattice.py | E8 unit tests | NEW, 22/22 pass |
| benchmarks/rotorquant_comprehensive.py | 9-method head-to-head benchmark | NEW |
| benchmarks/results/rotorquant_comprehensive.md | Full comparison report (3 models) | NEW |
| docs/KV_COMPRESSION_SURVEY_2026.md | 60+ technique survey | NEW |
| docs/RESEARCH_FINDINGS_APR15.md | All research findings | NEW |

## April 15 Session — Research Tracks Completed

1. **E8 lattice VQ** — BREAKTHROUGH (+0.1% PPL on 3B, near-lossless 3-bit)
2. RotorQuant head-to-head (WHT wins everywhere, mean-removal KV-head-dependent)
3. llama.cpp FA bug (3 issues, not 1; MMA kernel is the contribution opportunity)
4. Mean-removal prior art (35 papers; integration with rotation-VQ is novel)
5. KV compression survey (60+ techniques; top gap: KVTC 20-40x)
6. Clifford algebra (no advantage; 9 sign errors in RotorQuant)
7. KVTC Procrustes (motivating analysis not compression; DP bit allocation is portable)
8. D4 vs E8 (E8 wins, D4 not worth it)
9. AQUA-KV (78% MSE gain per-layer, needs full pipeline for PPL)
10. xKV cross-layer SVD (negative scaling result on Qwen)
11. NSNQuant double normalization (doesn't help our pipeline)

## Session Stats

- 127 commits (125 + 2 this session)
- 48 source modules (47 + e8_lattice.py)
- 1,818+ tests (1,796 + 22 E8 tests)
- 31 research experiments (20 + 11 this session: 2 breakthroughs, 3 dead ends, 6 analyses)
- 15+ charts created
- 100+ papers analyzed (40 + 60+ KV survey + 35 mean-removal survey)
- 3 models downloaded and cleaned up
- 10 models benchmarked (3B through 72B + Gemma 4)
