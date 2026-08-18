# HANDOFF — current snapshot 2026-08-18 (end of measurement campaign)

**Branch:** `phase-d-vllm-attention-impl` (16+ commits ahead of `master`, unmerged by choice)
**Master:** carries ONLY the reproduction registry (`cc2638e`) — see "Repo layout" below.

## What this session produced

A full end-to-end measurement campaign on the KV-cache quantization failure,
plus public infrastructure for outside reproduction.

### 1. End-to-end perplexity harness (closed the biggest gap)

`paper/experiments/ppl_harness.py` patches the repo's production quantizer into
`DynamicCache` and computes wikitext-2 sliding-window PPL **and** the cheap proxy
metrics in ONE forward pass, yielding paired (proxy, ground-truth) data.

Validated: reproduces the April-9 `ppl_for_tom.py` baseline to 4 decimals
(Qwen2.5-7B = 7.5225) and the catastrophe (3-bit uncentered ×1416).

### 2. MAIN RESULT — reconstruction metrics do not predict damage

**108 configurations, 19 models, 7 families.**

| proxy | Spearman | Pearson | separates broken/working? |
|---|---:|---:|---|
| per-vector cosine similarity | −0.498 | **−0.113** | **NO (ranges overlap)** |
| worst-layer attention-logit correlation | −0.704 | **−0.939** | **YES — 0/108 errors** |

The 0.995 cosine criterion is wrong in BOTH directions:
- qwen2.5-1.5b 4-bit uncentered: cos **0.9986** PASSES → PPL **×376**
- qwen2.5-1.5b 2-bit centered: cos **0.9904** FAILS → PPL **×1.07**

Held-out (threshold fitted on Qwen2.5 only, tested on 15 unseen models):
worst-layer logit r = **1/86** misclassified; vec_cos = **21/86**.

Separating gap: worst broken 0.8056 < best fine 0.8475 (any threshold between works).

### 3. Causal proof the mechanism is NOT Qwen-specific (§6.18)

`ppl_harness.py --inject ALPHA` adds a shared per-head direction before
quantization and subtracts it after — a no-op in exact arithmetic, so damage is
purely the quantizer's. On **Llama-3.2-1B** (naturally immune):

| inject | uncentered | centered |
|---:|---:|---:|
| 0 | ×1.07 | ×1.06 |
| 3× | ×5.22 | ×1.06 |
| 10× | ×646 | ×1.07 |
| 30× | **×1831** | **×1.09** |

A shared key component is SUFFICIENT to reproduce the full Qwen2.5 catastrophe in
an immune model. Dose-dependent. Centering neutralizes it at every dose.

**Honest caveat (in the paper):** this does NOT extend the metric claim. In the
injected regime cosine DOES track damage. Cosine's blindness is specific to the
NATURAL regime. §6.15 still rests on 8 natural failures, all Qwen2.5.

### 4. Two of our own hypotheses REFUTED

- **KV-head-count hypothesis (§6.3) is dead.** It was a within-Qwen2.5 artifact.
  Qwen3.5-0.8B has 2 KV heads (same as Qwen2.5 models failing ×1348) and is immune.
  Gemma2/3, Falcon3 share Qwen2.5-7B's 4 KV heads and are immune. SmolLM2 has 32
  and is no safer.
- **P1 "not bit-starvation" downgraded to PARTIALLY confirmed.** Bit sweep
  2/3/4/5/6/8 shows enough bits DO eventually fix it (at 6), just not in the
  practical 2–4 bit range.

### 5. Most actionable number: centering is worth ~4 BITS

2-bit **centered** (×1.07) beats 6-bit **uncentered** (×1.17) on Qwen2.5-1.5B.
Same-or-better quality at one third the storage. Side info is `d` floats/head,
amortized to nothing.

### 6. "Better than Qwen2.5-7B" — every larger/newer model is immune

| Model | worst uncentered |
|---|---:|
| Qwen2.5-14B | ×1.19 |
| Qwen3-14B | ×1.07 |
| Qwen3.5-9B | ×1.00 |

Every catastrophic cell in all 108 is Qwen2.5 (1.5B/3B/7B). Qwen2.5-14B and up
are fine. Qwen3 intermediate, Qwen3.5 immune.

### 7. Negative result: nothing beats plain mean-removal for free

`preconditioning.py`: per-channel whitening TIES exactly with centering;
standardizing rotated coords actively HURTS (0.82 vs 0.98); PCA-k wins only by
paying per-token storage that §6.16 shows is better spent on precision.

## Repo layout — IMPORTANT

**GitHub serves issue templates ONLY from the default branch.** So:
- `master` carries ONLY `REPRODUCTIONS.md` + `.github/ISSUE_TEMPLATE/independent-reproduction.yml`
- ALL paper/experiment work stays on `phase-d-vllm-attention-impl`, unmerged
- `REPRODUCTIONS.md` therefore uses ABSOLUTE IMMUTABLE PERMALINKS, not relative
  paths, so it resolves from master even though `paper/` is not there
- The template links to `blob/HEAD/REPRODUCTIONS.md` so it follows the default branch
- Label `independent-reproduction` created

Registry documents TWO reference configs deliberately (different numbers; a
reproducer would otherwise file a false mismatch):
- Reference A (`ppl_for_tom.py`, keys+values 3-bit): 7.5225 / 9410.4876 / 7.9029
- Reference B (`ppl_harness.py`, keys only): 7.5225 / 10655.2268 / 7.7235

**Rule: never enter a third party in the registry table unless they ran it and
reported it themselves.**

## Open items / next steps

1. **HIGHEST VALUE — the metric claim's real weakness (Limitation 18).** 108 cells
   but only 8 broken, all Qwen2.5. A detector validated on 8 positives from one
   family is not validated. Need MORE NATURAL failures from other families, or a
   way to make injected failures cosine-invisible.
2. **Check prior art before claiming novelty:** read HeadQ (arXiv 2605.03562,
   May 2026, "score-space correction") — may already be this idea.
3. Qwen2.5-32B run FAILED (harness error, likely OOM) — rerun with more headroom.
4. Three atlas models failed on GPU contention: OLMo-2-1B, Granite-3.3-2B,
   Ministral-8B.
5. Single seed, single corpus (wikitext-2), 512-token window, keys-only. No
   variance estimates.
6. Tomorrow's outreach: send `https://github.com/dhawalc/turboQuantDC/blob/HEAD/REPRODUCTIONS.md`
   to ONE qualified external engineer, requesting reproduction not endorsement.

## Operational gotchas learned

- Background jobs launched from a Bash tool call DIE when that call times out at
  2 min. Use `setsid nohup ... < /dev/null & disown` AND verify with `pgrep`.
- An ollama runner can grab 22.5 GB of GPU with no warning (KEEP_ALIVE short).
  Check `nvidia-smi --query-compute-apps` before launching GPU work.
- Running two GPU jobs concurrently caused the OOM failures above.

---

# TurboQuantDC — Handoff

## Current snapshot — 2026-04-27 ~10:55 PDT

**Session today:** see `HANDOFF_2026-04-27.md` for the full overnight + post-wake run. Key outputs:

- **PR #1 open:** https://github.com/dhawalc/turboQuantDC/pull/1 — `Phase D: vLLM custom AttentionImpl for TurboQuant 3-bit KV` (branch `phase-d-vllm-attention-impl`).
- **8-way Opus 4.7 code review** at `docs/code_review/2026-04-27/CODE_REVIEW_2026-04-27.md` (synthesis) + per-area files `01_*` through `08_*`.
- **Working vLLM serve recipe:** `scripts/serve_qwen36_flawless.sh` (Qwen3.6-27B-AWQ-INT4 on RTX 4090, max-model-len 1024, max-num-seqs 1, FlashInfer workspace 128 MiB). Real constraint: 4090 + your `uvicorn :8110` + prod + colleague leaves ~2 GiB residual after weights. Goal of 4500 tok/s blocked by VRAM, not software, in this configuration.
- **Algorithmic correctness:** 5 of 7 stub-listed requirements DONE in PR #1 (subclass, GQA, stateful per-layer/sequence, fp32 norms, mean-removal). 1 partial (paged KV layout — Phase E). 1 inherited (int16 indices).
- **Bug fixes shipped to master:** `e8_lattice.nearest_d8` argmin → argmax (commit `2ba205b`); 935-LOC docstring-sketch `vllm_integration.py` → 168-LOC honest stub (commit `a97abd6`); `qwen3.5-27` config corrected to 64/4/256, `qwen3.6-27` added.

**Honest claim retractions surfaced by the review (not yet applied — queued):**
- "E8 lattice VQ" results were obtained with half-integer scalar quantisation, not actual E8. `nearest_e8_relaxed` should be relabelled.
- "PPL 9410 → 7.90 with mean-removal" likely held only at full-prefill; the autoregressive `residual_quant.py:382-397` has a softmax shift-invariance bug.
- README quickstart code does not run (every kwarg wrong).
- CLAUDE.md says "All source files in turboquantdc/ and tests/ are empty stubs" — repo is at v0.3.0 with 67 modules, 43 test files.

**Next-session queue:** real benchmarks on freed VRAM (kill `uvicorn :8110` for +1.86 GiB) → mean-removal autoregressive fix → README/CLAUDE.md/E8 retraction wording → Phase E paged-layout integration → Phase F vLLM backend registration.

**Disabled scheduled routine:** `trig_01ViWwvugsYBVvkeF9dptwks` (Sonnet 4.6 fallback for Phase D — disabled because we did Phase D locally on Opus 4.7). Re-enable from https://claude.ai/code/routines if useful.

---

## Original handoff (Updated April 15, 2026)

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
