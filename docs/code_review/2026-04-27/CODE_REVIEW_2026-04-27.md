# TurboQuantDC Code Review Synthesis — 2026-04-27

**Reviewers:** 8 parallel Opus 4.7 subagents (read-only)
**Scope:** Full repo audit. Existing code authored by Opus 4.6 / Sonnet across 125+ commits, v0.3.0.
**Goal of review:** Determine whether/how to ship Qwen3.6-27B inference on a single RTX 4090 via vLLM in a 6-hour window, and surface every correctness, performance, and hygiene issue worth knowing.

> Per-area reports are at `docs/code_review/2026-04-27/0[1-8]_*.md`. This file is the consolidated executive summary.

---

## Verdict

**Do NOT use `turboquantdc/vllm_integration.py` for tonight's demo.** Three independent reviewers (#1 hot-path, #4 cache architecture, #5 quality, #8 public API) found independent showstoppers in this single file:

- **#1**: 7 CRITICAL bugs — stateless backend, broken GQA, fp16 underflow, paged-attention layout mismatch.
- **#4**: not PagedAttention compatible; flat `(max_seq_len, H, D)` layout instead of `(num_blocks, block_size, ...)`.
- **#5**: missing mean-removal — would silently produce PPL ~13,225 on Qwen2.5-7B (vs. 7.90 with mean-removal).
- **#8**: file does not `import vllm`; targets a vLLM internal layout (`llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers`) that was removed in V1 engine refactor (default since vLLM 0.8). It is, in #8's words, "a docstring sketch."

**Use vLLM with native FP8 KV instead.** Production-tested, well-supported, hits the user's UX goal directly.

**Use `GenerationCache` + HF transformers as the TurboQuant single-sequence research demo** — verified working by reviewer #8. But note reviewer #3's finding that some quality claims are mislabeled (E8 lattice is not actually E8; mean-removal fix is mostly a prefill artifact in autoregressive code path).

---

## Per-area highlights

### 01 — Hot-Path Correctness

**7 CRITICAL** including:
- `vllm_integration.py:127-199` — `TurboQuantAttentionBackend` stateless; `compress_kv` never appends to `_compressed_*_store`.
- `vllm_integration.py:300-329` — GQA broken; queries flattened across all heads attend to keys from all KV heads.
- `vllm_integration.py:622-683` — `TurboQuantCacheManager.fetch` incompatible with vLLM PagedAttention block layout.
- `estimator.py:99-105` — fp16 underflow; `1e-8` epsilon is below fp16 min normal (6e-5).
- `polarquant.py:144` — `quantize` does not normalize input; saturates the codebook.
- `qjl.py:78` — sign rule maps zeros to +1; small upward bias.
- `vllm_integration.py:586` — `mse_indices.long()` produces 8x int64 memory blowup.

### 02 — CUDA & Kernels

**3 CRITICAL** including:
- `cuda/dequantize.cu:181-184` — no bounds check on `idx`; corrupted indices kill the CUDA stream.
- `cuda_kernels.py:208,247` — WHT result silently upcast to fp32 and never restored.
- `cuda_kernels.py:111-116` — `R.contiguous().float()` allocates fresh fp32 copy of R on every call.

**Verification of 29x speedup claim:** PLAUSIBLE BUT MIS-FRAMED. Triton kernel materializes a `(BLOCK_D, BLOCK_D)` tile that spills at d=256 on Ada (64 KB regfile). CUDA streams R from global memory. Speedup is "Triton baseline broken", not "CUDA fast". **HANDOFF.md should retract "register cliff at d=256" wording.**

`turboquantdc/kernels/` is empty. `fused_attention.py` is misnamed (eager PyTorch with 7 separate launches).

### 03 — Algorithm & Math

**5 CRITICAL** including:
- `e8_lattice.py:52` — `nearest_d8` flips wrong coordinate (uses `argmin` instead of `argmax`).
- `e8_lattice.py:95-115` — `nearest_e8_relaxed` is NOT E8; it's (1/2)·Z^8. **HANDOFF "E8 lattice VQ" results were obtained with half-integer scalar quantization, not E8.** Conway-Sloane / Viazovska / QuIP# E8P12 references should be retracted.
- `e8p_codec.py:24-75` — Source set is not QuIP# E8P12; 222/256 patterns wrong; 5 duplicates.
- `residual_quant.py:152-156` — `mean(dim=0)` collapses across batch×heads×positions, contradicting per-head docstring.
- `residual_quant.py:382-397, 459` — running-mean centering breaks softmax shift-invariance in autoregressive use. **The "PPL 9410 → 7.90" headline likely held only for full-prefill, not autoregressive.**

**What IS correct (verified numerically):** WHT orthogonality, QR Haar uniformity, QJL inner-product unbiasedness, Lloyd-Max distortion bounds, Givens/Quaternion/Cayley rotations, DCT, ResidualVQ stage-2, PCA whitening.

### 04 — Cache Architecture

**Production-ready (KEEP):**
- `generation_cache.py` — load-bearing; FP16 anchor schedule; proven Qwen PPL numbers; what `run_70b.py` actually uses. **Use `GenerationCache.from_preset("balanced")` or `"hybrid_max_quality"`.**

**Confirmed broken:**
- `v2_cache.py` — PCA whitening blows up on tiny eigenvalues.
- `retrieval_cache.py` — IVF undertrained, degrades to single cluster.

**Archive (16 files):** `adaptive_hf_cache.py`, `adaptive_generation_cache.py`, `ultimate_cache.py`, `self_correcting_cache.py`, `streaming.py`, `streaming_70b.py`, `ultra_streaming*.py` (3), `cross_layer_kv.py`, `cross_layer_predict.py` (hardcoded `/media/dhawal/Beast/`!), `chunked_prefill.py`, `temporal_decay.py`, `temporal_delta.py`, `cache_distillation.py`, `sparse_loading.py`.

### 05 — Quality Preservation

**Mean-removal status:** REAL claim, REAL implementation in `_CompressedLayer` / `ResidualQuantLayer` / `GenerationCache`. **NOT wired** in `vllm_integration.py`, `channel_adaptive`, `adaptive_bits`, `layer_adaptive`, `outlier`, `evolving_compressor`, `cross_head_compress`. **If the demo path goes through `vllm_integration.py`, it produces PPL 13,225 on Qwen2.5-7B (vs 7.90).**

**Production-ready (INCLUDE):** mean-removal (running per-head, in `_CompressedLayer.update`); `anchor_strategy="boundary"`.

**Archive (broken or superseded):** `expected_attention` (Spearman -0.035 on dist shift), `cross_head_compress`, `learned_quant`, `cayley_quant`, `layer_adaptive.LayerAdaptiveKVCache`, `evolving_compressor`.

**HIGH-1 latent bug:** `generation_layers.py:469-471` — per-chunk mean snapshot materialized to `(B, H, new_seq, D)` via `.expand().clone()`. Memory is `O(N·H·D)`. Fix: store `(B, H, 1, D)` and broadcast at dequant.

### 06 — Tests & Benchmarks

**Test coverage on hot-path files:**
- `vllm_integration.py` (935 LOC) — **0 tests**.
- `generation_core.py` (553 LOC) — **0 tests**.
- `e8_lattice.py`, `residual_quant.py` — well covered (22+ tests each).
- `cuda_kernels.py` — adequate but skips on CPU and **never tests d=256** (Qwen3.x head dim).
- **No GQA-shaped fixtures anywhere.**

**Theatrical benchmarks:**
- `gemma4_showcase.py` prints "150 tok/s at 262K" as a hardcoded string.
- `triple_stack_benchmark.py` reports cosine only — no PPL.
- `large_model_validation.py` claims "IDENTICAL at 50 tokens" — `32b_long_generation.md` shows divergence at token 52.
- `impossible_inference.py` is a VRAM calculator masquerading as a benchmark.
- `long_context.py` docstring says "Qwen3.5-27B" but defaults to Qwen2.5-14B.

**Recommended demo run-order:**
1. `benchmarks/synthetic.py` (1–3 min, paper bounds)
2. `benchmarks/bench_cuda_kernels.py` (2–5 min)
3. `benchmarks/ppl_for_tom.py` (patch model path → ~60–90 min)
4. `benchmarks/niah_for_tom.py` (patch → ~15–25 min)
5. `benchmarks/generation_quality.py` (patch → ~10–15 min)

**Pre-flight:** hardcoded model names in `niah_for_tom.py:66` and `ppl_for_tom.py:53-56` (Qwen2.5). **AWQ-INT4 × KV-compression has never been measured in this repo** (only BnB-NF4).

### 07 — Repo Hygiene

**CLAUDE.md is dramatically stale** — line 11 says "All source files in turboquantdc/ and tests/ are empty stubs"; reality is 67 modules, 43 test files, v0.3.0.

**6 tracked `_buggy`/`_broken_scoring`/`_pre_*`.jsonl files at repo root** (~7 MB), gitignored AFTER tracking (no-op).

**17 Python scripts at repo root**; only `setup.py` belongs.

**`GROWTH_PLAYBOOK.md` ("Codename: MACHIAVELLI", 30 KB)** — internal hype playbook visible publicly.

**Proposed archival** (no deletions, all moves): repo root → 1 .py + ~4 .md. ~7,653 LOC moved out of root. ~166 MB on-disk freed by deleting gitignore-leak caches.

### 08 — Public API & Integration

**Working integration paths:**
- ✅ `from transformers import AutoModelForCausalLM` + `from turboquantdc import GenerationCache` + `model.generate(past_key_values=cache)` — verified end-to-end.
- ✅ `hf_integration.TurboQuantCache(bits=3)` — verified end-to-end.

**Broken paths:**
- ❌ `vllm_integration.py` — sketchwork; no `import vllm`; V0 layout; no GQA broadcasting; `compute_attention` materializes full softmax incompatible with paged attention.
- ❌ `README.md:140-150` quickstart — every kwarg wrong; `TypeError` on first call.
- ❌ `turboquantdc.run_model` — `ModuleNotFoundError` after `pip install` (`__init__.py:234` does `from run_70b import run_model`; `run_70b.py` is at repo root, never shipped).
- ❌ CUDA `.cu` files missing from wheel (`MANIFEST.in:6` includes only `*.py`).
- ❌ `vllm_integration.py:795` — static config for `qwen3.5-27` says num_layers=62, num_kv_heads=8 — actual is 64, 4.

**Concrete shipping recipe:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquantdc import GenerationCache
import torch

MODEL = "./models/Qwen3.6-27B-AWQ-INT4"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype=torch.float16)

cache = GenerationCache(
    key_bits=3, val_bits=3, fp16_window=128,
    anchor_strategy="boundary",
    num_layers=model.config.num_hidden_layers,
    use_residual_quant=True, center_before_quantize=True,
    quantizer_type="e8",
)
out = model.generate(**tok("…", return_tensors="pt").to(model.device),
                     past_key_values=cache, max_new_tokens=256)
```

---

## Top 5 priorities for daylight

1. **Strip out `vllm_integration.py` or rewrite from scratch** as a real vLLM custom backend (subclass `AttentionImpl`, integrate with PagedAttention block layout, support GQA via repeat-interleave).
2. **Fix README.md quickstart** — every kwarg wrong.
3. **Update CLAUDE.md** — line 11 etc. factually incorrect.
4. **Archival sweep per #7's plan** — unblocks open-source readiness.
5. **Add GQA test fixtures** — Qwen3.x family is GQA-only.

## Top 5 priorities for tonight (this run)

1. ✅ Stand up vLLM + Qwen3.6-27B-AWQ-INT4 + native FP8 KV — script: `scripts/serve_qwen36_flawless.sh`.
2. **Build benchmark harness** — concurrency × context × output-length sweep over the running server.
3. **Single-sequence TurboQuant comparison** via the verified-working `GenerationCache` path (HF transformers).
4. **Quality validation** — NIAH @ 16K/32K, tool-call validity, code spot-checks.
5. **HANDOFF write-up** — recipe, headline numbers, link to this code review for follow-ups.

## What's correct in the codebase (verified by review)

- TurboQuant paper math (PolarQuant + QJL): correct. Inner products are unbiased.
- WHT, QR rotations, Lloyd-Max, Cayley, Givens, Quaternion, DCT, PCA whitening, ResidualVQ stage-2: all verified correct.
- `generation_cache.py` is the load-bearing production cache; pattern works.
- `hf_integration.TurboQuantCache(bits=3)` works end-to-end against transformers 5.5.

## What needs retraction

- "E8 lattice VQ" — the implementation is half-integer scalar quantization, not E8.
- "QuIP# E8P12" reference in `e8p_codec.py` — source set is not E8P12.
- "29x CUDA speedup at d=256 (Triton register cliff)" — Triton baseline was broken; reframe.
- "PPL 9410 → 7.90" mean-removal claim — autoregressive layer code path has shift-invariance bug; claim likely held only at full prefill.
- "All source files in turboquantdc/ and tests/ are empty stubs" (CLAUDE.md:11) — repo is at v0.3.0 with 67 modules.
