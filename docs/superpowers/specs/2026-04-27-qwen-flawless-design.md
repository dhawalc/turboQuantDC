# Spec — Qwen3.6-27B "Flawless Local" on RTX 4090

**Date:** 2026-04-27
**Author:** Claude (Opus 4.7) — autonomous overnight run
**Status:** Active. User asleep until 08:00 PDT 2026-04-27.

## Problem

OpenClaw and similar agent applications should be able to point at a **local Qwen3.6-27B** inference endpoint and have it feel **indistinguishable from a frontier cloud API** (Opus 4.7, GPT-5.4). Concretely: low TTFT, smooth per-stream decode, multiple concurrent agents without queueing, long context, no quality cliff vs FP16.

The user's stated proxy metric is **~4500 tok/s aggregate decode throughput on a single RTX 4090**. The number itself is a *symptom* of "frontier-API parity at the workload shape OpenClaw uses" — not the goal in isolation.

## Goal (success criterion)

Single-process vLLM-compatible HTTP server on `localhost:8000` running Qwen3.6-27B (or the largest Qwen3.6 variant that fits a 4090), exposing an OpenAI-compatible API, achieving:

| Dimension | Target |
|---|---|
| TTFT (p50, ≤4K prompt, 16 concurrent) | < 500 ms |
| Per-stream decode (16 concurrent) | ≥ 60 tok/s/stream |
| Aggregate decode throughput (peak) | ≥ 3000 tok/s; aspirational 4500 tok/s |
| Concurrent streams without queueing | 16 minimum, 32 aspirational |
| Context per request | 32K supported, 64K aspirational, 8K is the realistic agent workload |
| Quality regression vs FP16 | Tool-call validity within 5%; NIAH passes at 32K; PPL within +1% on a code-shaped corpus |
| Reliability under churn | No mid-generation OOM; continuous batching tolerant of mixed prompt lengths |

## Non-Goals

- Multi-GPU / tensor-parallel deployment.
- Multi-model serving (this is a 1-model, 1-GPU plan).
- Production-grade vLLM upstream PR. (Tracked as a follow-up.)
- Vision capability of Qwen3.6-27B (we use `--language-model-only` to skip the vision encoder).
- Ollama / llama.cpp pipelines (vLLM is the chosen path).

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4090, 24 GB VRAM, SM 89 (Ada Lovelace) |
| CUDA | 13.0 |
| PyTorch | 2.11.0+cu130 |
| Python | 3.12 |
| Disk free | 217 GB |
| OS | Linux 6.8.0 |

Pre-existing GPU residents (untouchable): gnome-remote-desktop (~474 MB), prod python (~490 MB), colleague python (~490 MB). User's own python at `/srv/work/dev/dhawal/venv` (~1.9 GB) — left running unless user opts in. **Ollama was stopped to free 6.8 GB**; not restarted.

Effective free VRAM after ollama stop: ~21 GB.

## Model

**Primary:** `cyankiwi/Qwen3.6-27B-AWQ-INT4` — 19 GB on disk, AWQ-Int4 weight quantization, dense Qwen3.6 architecture with gated-delta-network (GDN) hybrid attention, native 262K context.

**Why this variant:**
- Official `Qwen/Qwen3.6-27B-FP8` is 28 GB — won't fit alongside KV + activations.
- Official `Qwen/Qwen3.6-27B` (BF16) is 56 GB — needs 2× H100.
- AWQ-Int4 + Marlin kernels is the throughput-optimal path on Ada Lovelace.
- vLLM ≥ 0.19 supports this format natively.

**User also has** `qwen3.6:27b` in ollama (17 GB GGUF). Keeping as a fallback for HF-based runs if vLLM path fails. The HF download is the primary ingestion path because vLLM's GGUF support is limited.

## Stack

| Layer | Choice | Notes |
|---|---|---|
| Inference engine | **vLLM ≥ 0.19.1** | Required for Qwen3.6 hybrid attention support |
| Weight quantization | AWQ-Int4 (Marlin kernels) | Best throughput on SM 89 |
| KV cache (baseline) | `--kv-cache-dtype fp8_e4m3` | vLLM-native, well-tested |
| KV cache (TurboQuant path, if integration patched) | TurboQuant 3-bit E8 lattice via `turboquantdc/vllm_integration.py` | Stretch goal — requires fixes per code review |
| Attention backend | `FLASHINFER` | vLLM auto-detects on Ada |
| Speculative decoding | MTP (Multi-Token Prediction) | Built into Qwen3.6, ~1.5–2x decode boost |
| Prefix caching | enabled | Repeated agent system prompts |
| Chunked prefill | enabled | Smooth TTFT under concurrency |
| API | OpenAI-compatible (`/v1/chat/completions`) | Drop-in for OpenClaw |

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  OpenClaw / agent app                                          │
│  → POST localhost:8000/v1/chat/completions                     │
└──────────────┬─────────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────────┐
│  vLLM 0.19.1 (1 process, 1 GPU)                                │
│   - Qwen3.6-27B-AWQ-INT4 weights (~19 GB)                      │
│   - FP8 KV cache (baseline) OR TurboQuant 3-bit E8 (stretch)   │
│   - Continuous batching, prefix caching, chunked prefill       │
│   - FlashInfer attention backend                               │
│   - MTP speculative decoding                                   │
└──────────────┬─────────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────────┐
│  Benchmark harness (turboQuantDC repo, separate process)       │
│   - Replays agent-shaped traces                                │
│   - Measures TTFT p50/p95, decode tok/s/stream, aggregate      │
│   - Sweeps concurrency × context × KV config                   │
│   - Validates quality: NIAH @32K, JSON tool-call accuracy      │
│   - Writes results to benchmarks/results/qwen_flawless/        │
└────────────────────────────────────────────────────────────────┘
```

Three deliverables, in dependency order:

1. **vLLM server config** — model + AWQ + FlashInfer + MTP + FP8 KV. The baseline. Provable to hit user's UX goal even without TurboQuant.
2. **Benchmark + quality harness** — agent-shaped traces, NIAH, tool-call validity. Same harness for baseline and TurboQuant runs (apples-to-apples).
3. **TurboQuant integration patches** — apply CRITICAL fixes from code review (8 parallel Opus 4.7 reviewers), wire into vLLM. Stretch goal; the demo holds even if this slips.

## Code Review (parallel Opus 4.7 subagents)

8 Opus 4.7 subagents launched at 02:00 PDT, reviewing the existing repo (written by Opus 4.6 / Sonnet across 125+ commits). Areas:

1. Hot-path correctness (`vllm_integration.py`, `kv_cache.py`, `estimator.py`, `polarquant.py`, `qjl.py`)
2. CUDA & kernels (`cuda_kernels.py`, `cuda/*.cu`, `kernels/*`)
3. Algorithm & math (E8 lattice, residual quant, codebooks, rotations)
4. Cache architecture (~20 cache implementations — identify load-bearing vs research)
5. Quality preservation (mean removal, layer/channel/outlier adaptive)
6. Tests & benchmarks (which to run for the demo)
7. Repo hygiene (archival plan)
8. Public API & integration (minimum-viable vLLM wiring)

Findings synthesize into `docs/code_review/2026-04-27/CODE_REVIEW_2026-04-27.md` with CRITICAL/HIGH/MEDIUM/LOW buckets. CRITICAL+HIGH fixes get applied before the TurboQuant integration step. MEDIUM/LOW queue for daylight.

**Early returns confirm the spec's risk assessment:**

- Reviewer #1 (hot-path correctness): 7 CRITICAL bugs in `vllm_integration.py`. Stateless backend, broken GQA, fp16 underflow, paged-attention layout incompatibility. The existing monkey-patch is *not* viable for batched inference without major fixes.
- Reviewer #2 (CUDA & kernels): "29x speedup" claim mis-framed — Triton baseline is broken, not CUDA fast. PagedAttention incompatibilities throughout. `kernels/` directory is empty. `fused_attention.py` is misnamed (eager PyTorch).

This re-affirms making TurboQuant a *stretch* goal, with FP8 KV baseline as the primary path.

## Decision Tree

```
Did vLLM install in venv succeed?
├─ Yes → continue
└─ No → fall back to HF transformers + custom batcher (lower throughput ceiling, but ships)

Did Qwen3.6-27B-AWQ-INT4 download succeed?
├─ Yes → use HF safetensors path
└─ No → fall back to local ollama GGUF blob via vLLM GGUF support (slower)

Did vLLM serve boot cleanly with FP8 KV?
├─ Yes → measure baseline; this IS the deliverable if everything else fails
└─ No → triage error; may indicate Qwen3.6 hybrid-attention not supported in installed vLLM version

Did benchmark hit aggregate ≥ 3000 tok/s on baseline?
├─ Yes → user's UX goal met; TurboQuant becomes pure upside
└─ No → tune: --max-num-batched-tokens, --max-num-seqs, --max-model-len; if still no, document realistic ceiling

Did TurboQuant CRITICAL fixes apply cleanly?
├─ Yes → wire patched integration; A/B vs FP8
└─ No → document fixes-needed list; ship FP8 baseline only
```

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| vLLM install fails on CUDA 13.0 | Medium | High | Fall back to torch 2.10 + cu128 in venv; or HF transformers path |
| Qwen3.6 hybrid-attention not supported in vLLM 0.19.1 | Low | High | Verify in recipe, fall back to Qwen3-32B (real public Qwen3 dense ~30B) |
| Model download interrupted | Low | Medium | hf_transfer enabled; if fails, use local ollama GGUF |
| `vllm_integration.py` patches take >2 hours | High | Medium | Ship FP8 baseline only; document TurboQuant as follow-up |
| 4500 tok/s unreachable in 6 hours | Medium | Low | Document realistic ceiling; user's UX goal is the deliverable, not the number |
| Colleague's process or prod gets killed | Low | High | Strict policy: only ollama and own venv at /srv/work/dev/dhawal touched |
| Disk fills during model download | Low | Medium | 217 GB free, model is 19 GB — comfortable |

## Time Budget (PDT, 27 Apr 2026)

| Slot | Phase | Status |
|---|---|---|
| 01:51 — 02:15 | Setup, env check, kick off background work | DONE |
| 02:00 — 02:45 | 8 code reviewers run; vLLM install + model download in background | RUNNING |
| 02:45 — 03:30 | Synthesize review; apply CRITICAL fixes to repo | PENDING |
| 03:30 — 04:30 | Stand up vLLM server; baseline benchmark (FP8 KV) | PENDING |
| 04:30 — 06:00 | Wire TurboQuant (if patches green); A/B benchmark + quality eval | PENDING |
| 06:00 — 07:30 | Final headline numbers; write HANDOFF + recipe | PENDING |
| 07:30 — 08:00 | Buffer, commit, final cleanup | PENDING |

## Deliverables (by 08:00 PDT)

- `docs/superpowers/specs/2026-04-27-qwen-flawless-design.md` (this file)
- `docs/superpowers/plans/2026-04-27-qwen-flawless-plan.md` (implementation plan)
- `docs/code_review/2026-04-27/*.md` (8 review files + 1 synthesis)
- `HANDOFF_2026-04-27.md` (top-level: what shipped, headline numbers, recipe, follow-ups)
- `benchmarks/results/qwen_flawless/*.json` (raw benchmark data)
- `scripts/serve_qwen36_flawless.sh` (the recipe — one-liner to reproduce)
- `.venv-vllm/` (working venv — not committed, but documented)
- Git commits at every phase boundary so progress is recoverable

## Open Questions (deferred to user, not blocking)

- Does the user have a HF_TOKEN we should use? (Not blocking; downloads work without one but are slower.)
- Is `/srv/work/dev/dhawal/venv/bin/python3.12` (1.9 GB GPU) safe to kill for cleaner VRAM? Default: leave alone.
- Long-term: should the TurboQuant vLLM integration be upstreamed as a PR, or kept as a fork?
