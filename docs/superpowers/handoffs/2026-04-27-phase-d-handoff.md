# Phase D Sub-Handoff — vLLM TurboQuant AttentionImpl

**Date:** 2026-04-27 (~10:47 PDT, same-day continuation of 01:51 overnight session)
**Operator:** Claude Opus 4.7 (interactive, on user's local 4090 — NOT the deferred remote routine)
**Branch:** `phase-d-vllm-attention-impl`
**Parent handoff:** `HANDOFF_2026-04-27.md` ("Post-wake continuation" section)
**Code review reference:** `docs/code_review/2026-04-27/CODE_REVIEW_2026-04-27.md`

## TL;DR

Replaced the previous `vllm_integration.py` stub's blockers with a real
`TurboQuantAttentionImpl` that subclasses vLLM 0.19+'s V1 `AttentionImpl`.
Single-request reference path: works correctly, all 19 unit tests pass,
verified at Qwen3.6-27B production shapes (24/4/256, 64 layers) on the
RTX 4090 at ~3 ms/decode step per layer in synthetic-tensor mode. NOT
yet wired into vLLM's paged scheduler — that's the next phase.

## What was built

| Path | Purpose |
|---|---|
| `turboquantdc/vllm_attention_impl.py` | Real `TurboQuantAttentionImpl(AttentionImpl)` — 333 LOC |
| `tests/test_vllm_attention_impl.py` | 19 tests covering construction, GQA, state, fp16, mean-removal, baseline cosine — 17 CPU + 4 GPU |
| `scripts/verify_phase_d.py` | On-4090 verification at Qwen3.6-27B shapes — writes `benchmarks/results/qwen_flawless/phase_d_verification.json` |

## Status against the 7 requirements (from the prior stub docstring)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Subclass vLLM 0.19+ V1 `AttentionImpl` | DONE | Module imports cleanly; constructor mirrors the abstract signature; `forward()` matches the interface |
| 2 | PagedAttention block layout | PARTIAL | `forward()` *bypasses* the paged buffer and uses our own per-(layer, sequence) state via `GenerationCache`. This is correct but doesn't yet co-exist with vLLM's continuous batching scheduler. See "What's deferred" |
| 3 | GQA repeat-interleave | DONE | `_sdpa_with_gqa()` uses SDPA `enable_gqa=True` with manual fallback; verified by `test_sdpa_with_gqa_matches_manual_repeat` (numerical equivalence to manual `repeat_interleave`) |
| 4 | Stateful per (layer, sequence) | DONE | `_caches: dict[sequence_id, GenerationCache]`; `forward()` calls `cache.update()` per layer; `test_forward_multi_step_state_grows` and `test_forward_multiple_layers_isolated` confirm growth and isolation |
| 5 | fp32 cast for vec_norm / divisor | DONE | Inherited from `TurboQuantEstimator.quantize` (already fp32 internally); `test_forward_fp16_no_nan_or_inf` confirms tiny-magnitude fp16 input doesn't produce inf/nan on CUDA |
| 6 | Mean-removal wired in | DONE | Default `center_before_quantize=True` in `_tq_config`; `test_construction_defaults_have_mean_removal_on` and `test_mean_removal_active_via_compressed_layer` confirm |
| 7 | int16 (or packed) index storage | INHERITED | `_CompressedLayer` already stores indices in non-int64 dtypes; we don't introduce new int64 storage. Future packing optimisation deferred |

**Net:** 5 requirements fully done, 1 partial (paged-layout integration), 1 inherited+verified.

## Verified on the 4090

`scripts/verify_phase_d.py` ran cleanly:

- Qwen3.6-27B-shape tensors: `num_heads=24, num_kv_heads=4, head_size=256, num_layers=64, gqa_factor=6`.
- Scenario: 64-token prefill + 16 decode steps × 4 layers exercised.
- All 4 layers PASS shape, finiteness, sequence-length-growth checks.
- Layer 0 is an `_FP16Layer` (boundary-anchor strategy); layers 1–3 are `_CompressedLayer`.
- Per-layer decode latency: ~3 ms/step (synthetic random tensors only — does NOT include model weight matmuls).
- GPU memory: 16 MiB allocated / 23.5 MiB peak (synthetic only; this scales linearly with sequence length).

These numbers do not predict end-to-end Qwen3.6-27B serving throughput. They prove the math + plumbing are correct at the production tensor shapes.

## What's not yet done (deferred Phase E and beyond)

1. **vLLM paged KV layout integration.** The current path uses `GenerationCache` instances keyed by sequence id, bypassing vLLM's `(2, num_blocks, block_size, num_kv_heads, head_size)` shared block pool. To use TurboQuant inside vLLM's continuous-batching scheduler we need to either:
   - Pack the compressed representation into the existing `(2, B, ...)` byte buffer with reshape-and-cache kernels (hard; needs custom dtype registration); or
   - Extend vLLM's KV cache spec to support a custom backend-allocated buffer (also non-trivial).
2. **Multi-request continuous batching.** Today `_caches: dict[__default__, GenerationCache]` — single sentinel id. The scheduler needs to pass per-request ids and we need an eviction policy.
3. **Backend registration.** Even with the impl working, vLLM doesn't know about it. Need to add it to vLLM's backend registry (a fork or upstream PR) before `vllm serve` can pick it.
4. **Fused CUDA kernels.** Current path uses `GenerationCache.update()` which uses Triton when available. A fused compress+attn kernel would meaningfully cut the ~3 ms/step overhead.
5. **End-to-end serving validation.** No actual Qwen3.6-27B forward pass through this backend yet. Would require steps 1–3 first.
6. **Mean-removal autoregressive shift-invariance fix** (Reviewer #3 CRIT-5) — affects PPL claim accuracy, queued for daylight.
7. **E8 lattice retraction** (Reviewer #3 CRIT-2) — `nearest_e8_relaxed` is half-integer scalar, not E8. Phase A's bug fix to `nearest_d8` doesn't change the labeling.

## Verification recipe (for the user)

After merging this branch:

```bash
# Confirm no regressions in existing tests
.venv-vllm/bin/python -m pytest tests/test_e8_lattice.py tests/test_estimator.py \
    tests/test_code_review_2026_04_27_fixes.py

# Run the new vLLM AttentionImpl unit tests (19 tests; needs CUDA for 4 of them)
.venv-vllm/bin/python -m pytest tests/test_vllm_attention_impl.py -v

# Production-shape verification on the 4090
.venv-vllm/bin/python scripts/verify_phase_d.py
# Writes: benchmarks/results/qwen_flawless/phase_d_verification.json
# Exit 0 if all_passed; non-zero on any layer failure.
```

The "I want 4500 tok/s on Qwen3.6-27B" goal still requires Phase E (paged-layout integration) and Phase F (vLLM upstream PR or fork) before it's reachable. This phase removes the algorithmic correctness blockers and ships a tested foundation.

## Open questions / known limitations

- The `boundary` anchor strategy makes layer 0 (and likely the last layer) `_FP16Layer` instead of compressed. That's the documented Qwen-quality-preservation behaviour, but it means the compression ratio at small `num_layers` is lower than ideal. Verify on Qwen3.6-27B (64 layers) whether the boundary schedule keeps just 2 layers FP16 or more.
- `_layer_idx_from_layer` parses layer_name like `"model.layers.7.self_attn"`. Qwen3.6 multimodal models prefix with `language_model.model.layers.7...` — covered by the parametrized test, but worth verifying against an actual Qwen3.6 vLLM run.
- The single-request `__default__` sentinel will collide if ever called from a multi-request scheduler. Wire to vLLM's request id before that path is enabled.

## Why this was done locally on Opus 4.7 instead of via the scheduled remote agent

The user opted to use the local Opus 4.7 session over the scheduled `claude-sonnet-4-6` cloud routine (`trig_01ViWwvugsYBVvkeF9dptwks`, now disabled). The local session has GPU access; the remote agent would have shipped only CPU-runnable code. Both paths produce the same kind of artifact (PR + sub-handoff); the local run is strictly more rigorous because it can run the GPU verification script.
