# Plan — Qwen3.6-27B "Flawless Local" Execution

**Date:** 2026-04-27
**Spec:** `docs/superpowers/specs/2026-04-27-qwen-flawless-design.md`
**Owner:** Claude (Opus 4.7), autonomous overnight run
**Window:** 01:51 PDT 2026-04-27 — 08:00 PDT 2026-04-27 (~6h 9min)

## Phase 0 — Setup (DONE)

- ✅ Created `docs/superpowers/specs/`, `docs/superpowers/plans/`, `docs/code_review/2026-04-27/`, `benchmarks/results/qwen_flawless/`, `logs/2026-04-27/`
- ✅ Stopped ollama (freed 6.8 GB VRAM)
- ✅ Created `.venv-vllm/` Python venv
- ✅ Installed `huggingface_hub` + `hf_transfer` in venv
- ✅ Verified `cyankiwi/Qwen3.6-27B-AWQ-INT4` on HF (vLLM-compatible, 19 GB)
- ✅ Confirmed user has `qwen3.6:27b` GGUF locally as fallback

## Phase 1 — Parallel kickoff (RUNNING)

1. **8 Opus 4.7 code reviewers** — files in `docs/code_review/2026-04-27/`:
   - ✅ #1 hot-path correctness (7 CRITICAL bugs in `vllm_integration.py`)
   - ✅ #2 CUDA & kernels (29x claim mis-framed; PagedAttention incompat)
   - ⏳ #3 algorithm & math
   - ✅ #4 cache architecture (`generation_cache.py` is load-bearing; 16 archives)
   - ✅ #5 quality preservation (`vllm_integration.py` MISSING mean-removal — would PPL-explode demo)
   - ⏳ #6 tests & benchmarks
   - ⏳ #7 repo hygiene
   - ⏳ #8 public API & integration

2. **vLLM install** — ✅ DONE (vllm 0.19.1 + flashinfer + torch + flashinfer-cubin in `.venv-vllm/`)

3. **Model download** — ✅ DONE (20 GB on disk at `models/Qwen3.6-27B-AWQ-INT4/`)

## Phase 2 — Synthesize, decide path (NEXT)

**Strategic call confirmed by reviewers #1, #4, #5:** the vLLM monkey-patch path through `vllm_integration.py` is NOT viable for tonight. Three independent reviewers found independent showstoppers:

- #1: stateless backend, broken GQA, fp16 underflow (correctness)
- #4: not PagedAttention compatible; flat layout (architecture)
- #5: missing mean-removal — the very fix that takes Qwen PPL from 9,410 to 7.90 (quality)

**Revised path:**

| Deliverable | Path | Status |
|---|---|---|
| **Primary** — Qwen3.6-27B serving on 4090 with frontier-API UX | vLLM + AWQ-Int4 + native FP8 KV cache | New, planned for Phase 3 |
| **Stretch** — TurboQuant 3-bit KV demo (single-sequence) | HF transformers + `GenerationCache.from_preset("balanced")` | New, planned for Phase 5 |
| **Documentation** — patch list for vLLM TurboQuant integration | Synthesized code review + minimum-viable patch sketch | New, planned for Phase 7 |

## Phase 3 — vLLM baseline server (~03:00 PDT)

```bash
.venv-vllm/bin/vllm serve ./models/Qwen3.6-27B-AWQ-INT4 \
    --quantization awq_marlin \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.92 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 32 \
    --port 8000 \
    --reasoning-parser qwen3 \
    > logs/2026-04-27/vllm_server_baseline.log 2>&1 &
```

Validation gates before benchmarking:
- Server boots within 5 minutes
- `curl http://localhost:8000/v1/models` returns the model
- Single completion request returns coherent text (no garbage)
- GPU memory utilization is steady ≤ 22 GB

If the AWQ-Int4 path fails with a Qwen3.6-specific error: try without `--quantization awq_marlin` (let vLLM auto-detect), or add `--language-model-only`. If still fails: switch to `Qwen3-32B-AWQ-Int4` from QwenLM/Qwen3 repo (real public ~30B Qwen3 dense model, vanilla attention).

## Phase 4 — Benchmark harness + baseline numbers (~03:30 PDT)

Build `scripts/bench_qwen_flawless.py`:

- aiohttp-based concurrency, hits `/v1/chat/completions` with `stream=true`
- Measures per-request: TTFT, ITL, decode tok/s, total tok/s
- Aggregates per-run: TTFT p50/p95, decode p50/p95, aggregate tok/s, success rate
- Writes JSON to `benchmarks/results/qwen_flawless/baseline_<concurrency>x<input>x<output>.json`

Sweep:
- Concurrency: {1, 4, 8, 16, 32}
- Input length: {1K, 4K}
- Output length: {256, 1024}

20 configurations × ~30 seconds each = 10 minutes wall clock.

Headline number = max aggregate tok/s across configurations.

## Phase 5 — TurboQuant single-sequence quality demo (~05:00 PDT)

Skipping vLLM-integrated TurboQuant for tonight (per Phase 2 strategic call).

Build `scripts/bench_turboquant_qwen36.py` (new):

- Loads Qwen3.6-27B-AWQ-INT4 via HF transformers (using vLLM's same weights)
- Wraps the attention layer with `GenerationCache.from_preset("balanced")` (mean-removal enabled per reviewer #5)
- Runs a 50-prompt code corpus
- Measures: PPL vs FP16 baseline, decode tok/s/stream, memory used at 32K context
- Optional: NIAH at 8K, 16K, 32K

If this is technically infeasible in the time budget (HF transformers + the cache wrapper may not work cleanly with AWQ-Int4 weights since AWQ uses custom kernels): defer the TurboQuant demo to daylight, document why, and ship just the FP8 baseline.

## Phase 6 — Quality validation on FP8-KV server (~06:00 PDT)

Against the running vLLM server:
- NIAH at 8K, 16K, 32K context (5 needles per length, 3 different needle positions each)
- 30-prompt tool-call validity benchmark (JSON schema compliance rate)
- 10 spot-check code generation prompts (eyeball)

Write `benchmarks/results/qwen_flawless/quality_validation.json` with per-test pass/fail.

## Phase 7 — HANDOFF (~07:00 PDT)

Create `HANDOFF_2026-04-27.md` at repo root. Sections:

1. **TL;DR** (2-3 sentences + headline number)
2. **What's running** — exact recipe, `serve_qwen36_flawless.sh`
3. **Hardware & software stack**
4. **Headline numbers** — TTFT, decode, aggregate, concurrency
5. **Quality validation** — NIAH, tool-call, code spot-checks
6. **Code review summary** — link to `CODE_REVIEW_2026-04-27.md`, top 5 priorities
7. **What works** — FP8-KV vLLM path, the recipe
8. **What didn't ship** — TurboQuant vLLM integration; documented patch list
9. **Recommended next steps** — prioritized
10. **Files added** — exhaustive list

## Phase 8 — Buffer / cleanup (~07:30 PDT)

- Kill vLLM server (clean GPU for morning user)
- Verify `git status` is clean
- Final commit: "session: 2026-04-27 overnight Qwen3.6-27B flawless run"
- Update `MEMORY.md` pointer to new session entry

## Failure-mode playbook

| If… | Then |
|---|---|
| vLLM serve fails on Qwen3.6 hybrid attention | Try `--no-enable-prefix-caching --no-enable-chunked-prefill`, then fall back to Qwen3-32B-AWQ-Int4 |
| Server OOMs at 32K context | Drop `--max-model-len` to 16K, then 8K. Document realistic ceiling |
| Aggregate < 1500 tok/s | Investigate FlashInfer fallback; try `--attention-backend FLASH_ATTN` or default backend |
| NIAH fails at 32K | Drop max ctx to 16K. Document |
| Single-sequence TurboQuant demo blocked by AWQ kernel incompat | Defer to daylight; ship FP8 baseline only |
| vLLM venv breaks unexpectedly | HF transformers fallback (lower throughput, but ships) |

## Decisions log

| Time | Decision | Rationale |
|---|---|---|
| 01:51 PDT | Stop ollama | Frees 6.8 GB VRAM; idle |
| 01:55 PDT | `cyankiwi/Qwen3.6-27B-AWQ-INT4` over GGUF | Marlin kernels in vLLM; GGUF spotty |
| 02:00 PDT | 8-way parallel review (Opus 4.7) | Faster per-domain depth |
| 02:10 PDT | TurboQuant vLLM = stretch, not primary | Reviewer #1 found 7 CRITICAL bugs |
| 02:18 PDT | Inline plan vs invoking writing-plans skill | User: "do not stop and wait" |
| 02:22 PDT | Confirm: skip vllm_integration.py entirely tonight | Reviewer #5 found mean-removal missing — would silently PPL-explode demo |
