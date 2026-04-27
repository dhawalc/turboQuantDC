# Tests & Benchmarks Review (Opus 4.7)

Reviewer: Opus 4.7 (1M)
Date: 2026-04-27
Scope: `/home/dhawal/turboQuantDC/tests/` (43 test files, 24,308 LOC) and
`/home/dhawal/turboQuantDC/benchmarks/` (~62 benchmark files, ~39K LOC).
HANDOFF claim: "1,796+ tests" — by name count this is plausible (`def test_` count
is high; sampled `test_residual_quant.py` alone has ~30 named tests, `test_generation_cache.py` ~70).

## Test Coverage on Hot-Path Files

| File | Test file | Coverage | Notes |
|---|---|---|---|
| `turboquantdc/vllm_integration.py` (935 LOC) | **NONE** | **0%** | `grep "vllm_integration\|TurboQuantAttentionBackend\|TurboQuantCacheManager"` against `tests/` returns **zero hits**. No unit tests for `_flatten_to_2d`, `quantize_kv`, the GQA path, the cache budget calculator, or the monkey-patch hooks. Given this is the integration point that the demo will exercise, this is a serious gap — and reviewer 01 already flagged a GQA correctness bug at `vllm_integration.py:300-329`. |
| `turboquantdc/kv_cache.py` (256 LOC) | `tests/test_integration.py` (and indirectly `test_estimator.py`, `test_sparse_v.py`) | Partial | `TurboQuantKVCache` is exercised in 3 integration-test classes (`test_single_vector_kv_cache`, `test_large_batch_kv_cache`, `test_estimator_matches_kv_cache`). No dedicated `test_kv_cache.py`. The vLLM-shaped GQA path is **not** covered. |
| `turboquantdc/e8_lattice.py` (233 LOC) | `tests/test_e8_lattice.py` (272 LOC, 22 tests) | High | Excellent coverage: lattice property tests (D8 even-sum, E8 coset structure), MSE monotonicity, scale calibration, cosine similarity, WHT integration, beats-Lloyd-Max comparison. **All 22 named tests look meaningful** — no dummy assertions. This is the cleanest test file in the repo. |
| `turboquantdc/residual_quant.py` (635 LOC) | `tests/test_residual_quant.py` (458 LOC, ~30 tests) | High | Tests the estimator, layer accumulation, cache protocol (HF DynamicCache compatible), bit-budget invariants, deterministic seeding, and a quantitative comparison against MSE-only and TurboQuantEstimator. Solid. |
| `turboquantdc/cuda_kernels.py` (536 LOC) | `tests/test_cuda_kernels.py` (339 LOC) | Adequate | Numerical correctness only: dequantize-MSE, dequantize-residual, WHT forward/inverse, fallback chain. **Skips entirely** if CUDA not available (`pytestmark = pytest.mark.skipif(not torch.cuda.is_available())`). Does **not** test the `CUDATurboQuant` wrapper under realistic batch sizes — only smoke-tests it. No tests at d=256 (Qwen3.5/3.6 head_dim). |
| `turboquantdc/generation_core.py` (553 LOC — production cache for the demo) | **NONE** | **0%** | No test file imports `from turboquantdc.generation_core`. `GenerationCache` is tested via `tests/test_generation_cache.py` (1,100 LOC, ~70 tests) but that imports from `generation_cache.py`, not `generation_core.py`. These are two different modules. The demo path (per HANDOFF and `niah_for_tom.py`) uses `generation_core.GenerationCache`. **Untested.** |

### Skipped / conditional tests of note
- `tests/test_cuda_kernels.py` — entire file skipped on CPU.
- `tests/test_estimator.py:655,670`, `test_polarquant.py:312,327`, `test_qjl.py:335,347`, `test_wht.py:273` — GPU-gated.
- `tests/test_streaming_70b.py:334-362` — three CUDA-prefetch tests skipped without GPU.
- `tests/test_generation_cache.py:1001` — Triton dispatch class fully skipped without Triton+CUDA.
- No `@pytest.mark.skip` for "broken/known issues"; no `pytest.xfail` markers.

### Test-coverage red flags
1. The pipeline tested by `test_integration.py` is `TurboQuantKVCache` (paper-faithful original). The pipeline used by **production benchmarks and the Qwen demo** is `generation_core.GenerationCache`. These have different code paths (different rotation type, mean-removal, ResidualQuant integration). The integration tests do not cover the demo's actual hot path.
2. No GQA-shaped tests anywhere. Every test fixture uses `num_heads == num_kv_heads` or treats heads as a uniform first dim. The vLLM backend will hit GQA shapes.
3. No tests for `cache_distillation.py`, `expected_attention.py`, or the triple-stack (despite being headlined in HANDOFF).

## Recommended Benchmarks to Run for Qwen3.6-27B Demo

Ranked by signal-per-minute. The model exists at `/home/dhawal/turboQuantDC/models/Qwen3.6-27B-AWQ-INT4/` (AWQ-INT4, 27B, head_dim probably 128, 8 KV heads).

| Benchmark | File | What it measures | Rough wall-clock |
|---|---|---|---|
| **PPL on wikitext-2** | `benchmarks/ppl_for_tom.py` | Real perplexity using `GenerationCache`, sliding window 512/256, on real model with BnB-4bit. Configs: FP16 / 3-bit / 3-bit+mean / 4-bit / 4-bit+mean. Hard-coded for `Qwen2.5-7B/3B`; **swap `MODELS` to include `Qwen3.6-27B-AWQ-INT4`**. This is the most important number — the +0.30 PPL claim must reproduce on Qwen3.6-27B. | 8–15 min per model × 5 configs ≈ 60–90 min on a 27B AWQ model. |
| **NIAH (needle-in-haystack)** | `benchmarks/niah_for_tom.py` | Greedy generation with the compressed cache; checks string `PINEAPPLE-77` is recovered at 10/50/90% positions. Auto-binary-searches max context that fits. Tests FP16, WHT-3, WHT-3+mean. **Swap `MODEL_NAME` to Qwen3.6-27B-AWQ.** This is the single most demo-friendly number: pass/fail on real long context. | 15–25 min for full sweep. |
| **Generation-quality token match** | `benchmarks/generation_quality.py` | 200-token greedy generation against FP16 baseline; reports exact token-match rate and PPL on wikitext excerpt. Most "honest" generation benchmark; will catch the same divergence found in `32b_long_generation.md` (token 52 collapse on 32B). | 10–15 min. |
| **Synthetic paper bounds** | `benchmarks/synthetic.py` | Validates Theorems 1, 2, Lemma 4 from the TurboQuant paper (MSE distortion, IP unbiasedness, codebook properties). Pure GPU/CPU, no model load. Sanity-check the algorithm before model integration. | 1–3 min. |
| **CUDA kernel speed** | `benchmarks/bench_cuda_kernels.py` | Vectors/sec for dequantize-MSE / dequantize-residual / WHT at d=128, d=256. Distinguishes "is the kernel actually fast" from "does the integration get bottlenecked elsewhere". | 2–5 min. |

If time permits as a sixth: `benchmarks/large_model_validation.py` (cosine + top-5
on extracted Qwen2.5-32B/72B KV) is reasonable, but its `large_model_results.md`
already shows the test (32B 3-bit RQ3 generation 13.7s vs FP16 4.1s — i.e. **3.3×
slower**, not faster), so it's mostly archival.

## Misleading Benchmarks (claim vs reality)

- **`benchmarks/gemma4_showcase.py:424,458`** — claims `"TurboQuantDC 3-bit: 150 tok/s at 262K on RTX 4090"`. The number is **hardcoded as a print statement**, not measured by this script. Same number is repeated as a label in `gemma4_charts.py:185`. There is no end-to-end benchmark in the repo that actually measured 150 tok/s at 262K context. Treat as marketing copy, not a measurement.
- **`benchmarks/triple_stack_benchmark.py`** — claims "37.9× compression at 0.93 cosine" (HANDOFF). What it actually measures: cosine similarity of `softmax(QK^T)V` between full and compressed cache, on synthetic queries from a single forward pass on Qwen2.5-3B. **It does NOT run generation, does NOT compute PPL, and does NOT use the production GenerationCache.** The 0.93 cosine is "attention-output cosine similarity at one layer", which the team's own `rotorquant_comprehensive.md` explicitly warns "does not predict PPL".
- **`benchmarks/large_model_validation.py` → `large_model_results.md`** — claims "IDENTICAL generation" at 50 tokens on 32B/72B. The team's own `32b_long_generation.md` corrected this: divergence happens at **token 52** on 32B; the 50-token match was 2 tokens shy of the divergence point. The "IDENTICAL" cells in the table are technically true but operationally meaningless. Use `32b_long_generation.py`, not `large_model_validation.py`, when claiming identical generation.
- **`benchmarks/impossible_inference.py`** — name suggests a benchmark; it is actually a **VRAM/throughput projection calculator**. It prints "tok/s" for hypothetical model+compression stacks based on bandwidth assumptions, not measurements. It does not run a model. Do not present its outputs as benchmark results.
- **`benchmarks/hf_benchmark.py`** — labels measurements "tok/s" (correct), but uses `BENCHMARK_RUNS=3` greedy generation runs of `MAX_NEW_TOKENS=100` on a 3B model. It is a microbenchmark of one prompt at one batch size, not a sustained-throughput measurement. Single-stream, no concurrency, no prefill overhead amortization.
- **`benchmarks/long_context.py`** — header docstring claims "demonstrates that TurboQuant enables long-context inference on Qwen3.5-27B"; in practice the script defaults to `Qwen/Qwen2.5-14B-Instruct` (line 41) and only measures attention-score top-k match on extracted KV, not generation. The "VERDICT" at the end is a memory-arithmetic projection, not a runtime test.

## CRITICAL findings (broken tests, missing coverage on hot path)

- **CRITICAL — vLLM integration has zero tests.** `turboquantdc/vllm_integration.py` is the largest hot-path file (935 LOC) and has no test coverage at all. Reviewer 01's analysis at `01_hotpath_correctness.md:46` already flagged a GQA correctness bug there (cross-product instead of grouped attention). For Qwen3.6-27B (`num_kv_heads=8`, `num_heads` typically 28–40), this means attention is mathematically wrong unless the GQA path is fixed AND tested before the demo runs.
- **CRITICAL — `generation_core.py` has zero direct tests.** This is the cache the demo benchmarks (`niah_for_tom.py`, `ppl_for_tom.py`) use. `tests/test_generation_cache.py` covers a **different module** (`generation_cache.py`). Changes to `generation_core.py` are not regression-protected.
- **CRITICAL — No test exercises GQA shapes.** Every fixture in the test suite assumes `num_heads == num_kv_heads`. Qwen3.6-27B is GQA. The first GQA-shaped tensor that hits production code will land on untested paths.

## HIGH

- **Bench/test mismatch on production cache.** Tests cover `TurboQuantKVCache` (paper original) and `GenerationCache` from `generation_cache.py`. Demos use `GenerationCache` from `generation_core.py`. These are not the same class hierarchy.
- **No dedicated `test_kv_cache.py`.** Only indirect coverage via integration tests. Direct unit tests for the public KV cache wrapper would catch shape regressions cheaply.
- **`large_model_results.md` numbers are stale and misleading.** The "IDENTICAL generation at 50 tokens" claim was corrected by `32b_long_generation.md` (April 2) but the original table still ships in the repo unchanged. Anyone running `large_model_validation.py` for the demo will print these numbers.
- **`rotorquant_comprehensive.md` (April 15) explicitly says attention cosine ≠ PPL.** Yet `triple_stack_benchmark.py` and `long_context.py` still report only attention cosine + top-k as quality. Replace those with PPL or token-match for the demo.
- **`pytest.skipif` on CUDA fixtures will SILENTLY skip ~25% of tests on a CPU runner.** The "1,796+ tests" headline number is gross, not net-of-skipped. CI on a CPU box will under-test by a wide margin.

## MEDIUM

- **`benchmarks/results/` contains stale JSON/MD from prior runs.** Numbers reported here may not reproduce on Qwen3.6-27B. Treat anything dated before April 25 as "directional, not authoritative."
- **62 benchmark files is too many.** Many are exploratory experiments (`adversarial_validation.py`, `delta_quant_experiment.py`, `learned_rotation_experiment.py`, `temporal_delta_experiment.py`, `cayley_quant_benchmark.py`, etc.). They produce results files but most won't reproduce in a tight demo window. Move experimental ones under `benchmarks/experiments/` to clarify the gold-standard set.
- **Test files with names like `test_run_70b.py`, `test_streaming_70b.py`, `test_ultra_streaming.py` are fragile.** They depend on model availability, GPU, and exact HF cache locations (`/media/dhawal/Beast/cache/hub`). Likely break on a clean checkout.
- **`benchmarks/synthetic.py` exits 0/1 based on bounds — ideal for CI**, but it's not wired into the test runner. Add it as a make target or a pytest entry.
- **No benchmark targets the AWQ-INT4 weight quantization × KV-cache compression interaction.** All "large model" benchmarks use BnB-NF4. Demo runs on AWQ-INT4 (`models/Qwen3.6-27B-AWQ-INT4/`) — this combination has never been measured in this repo.

## LOW

- `tests/test_benchmark.py` is named confusingly — it is a unit test for the prompt-list, not a "benchmark of tests".
- `evolving_compressor.py.backup` and `autoresearch_results_*.jsonl` files in repo root suggest an autoresearch loop dumping results into version control. Cosmetic, but inflates repo size.
- Several benchmarks read `os.environ["HF_HOME"] = "/media/dhawal/Beast/cache"` at import time. Will silently set the env for any test importer.

## Summary: which 3–5 benchmarks should the orchestrator run tonight?

Run **in this order**, gating on each step:

1. `/home/dhawal/turboQuantDC/benchmarks/synthetic.py` — 1–3 min, CPU-only, no model. **Sanity-checks the algorithm**. If this fails, every subsequent number is suspect. Exit code 0/1 is wired in.
2. `/home/dhawal/turboQuantDC/benchmarks/bench_cuda_kernels.py` — 2–5 min, CUDA. **Confirms kernels actually load and outperform the PyTorch fallback at d=128 and d=256** (Qwen3.6-27B head_dim).
3. `/home/dhawal/turboQuantDC/benchmarks/ppl_for_tom.py` — **patched to point at `models/Qwen3.6-27B-AWQ-INT4/`** (currently hardcoded to Qwen2.5-7B/3B). Configs: FP16, WHT-3, WHT-3+mean, WHT-4+mean. ~60–90 min total. **This is the headline PPL number.** If WHT-3+mean gets within +1 PPL of FP16 on Qwen3.6-27B, the integration is validated.
4. `/home/dhawal/turboQuantDC/benchmarks/niah_for_tom.py` — patched to point at the same Qwen3.6-27B model. ~15–25 min. **Demo-friendly pass/fail on long context.** If 3 of 3 needles recover with WHT-3+mean, ship.
5. `/home/dhawal/turboQuantDC/benchmarks/generation_quality.py` — patched to point at Qwen3.6-27B. ~10–15 min. **Catches the token-52 divergence problem** that `large_model_validation.py` missed; reports actual token-match rate, not just cosine.

**Do NOT run for the demo** (low signal, will eat the 6-hour budget):

- `large_model_validation.py` / `large_model_72b.py` — measures cosine only, has the misleading "IDENTICAL at 50 tokens" claim baked in.
- `triple_stack_benchmark.py` — attention-output cosine only, no PPL/generation.
- `long_context.py` — defaults to wrong model, measures only attention cosine.
- `gemma4_showcase.py` — prints hardcoded "150 tok/s" claim that the script does not measure.
- `impossible_inference.py` — calculator, not benchmark.
- The 50+ exploratory `*_experiment.py` benches.

**Pre-flight fixes required before running**:

- Patch `niah_for_tom.py:66` and `ppl_for_tom.py:53-56` to load Qwen3.6-27B from the local AWQ-INT4 directory.
- Verify `generation_core.GenerationCache` (used by these scripts) handles GQA — there is currently zero test coverage proving this.
- If reviewer 01's `vllm_integration.py` GQA bug is in the hot path for these scripts (it is **not** — these use `GenerationCache`, not the vLLM backend), the demo can proceed; otherwise it must be fixed first.
