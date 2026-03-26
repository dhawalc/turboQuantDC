# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity
**TurboQuantDC** — a from-scratch PyTorch + CUDA implementation of Google's TurboQuant algorithm (ICLR 2026) for compressing LLM key-value caches.

**Objective:** Compress KV cache to 3-bit with <0.5% attention quality loss on an RTX 4090. Enable running 27B+ parameter models locally with long context that would normally OOM.

## Current Status
All source files in `turboquantdc/` and `tests/` are empty stubs. `setup.py` is empty. Phase 1 (Core Algorithm) has not started. Check `PLAN.md` for the latest task checklist.

The reference implementation in `reference/tonbistudio-ref/` has a complete working implementation — use it for understanding the algorithm, NOT for copying code.

## Development Commands
```bash
# Dependencies (not yet in a requirements.txt — create one when starting)
pip install torch scipy

# Later phases will also need:
pip install transformers accelerate bitsandbytes

# Run tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_codebook.py -v

# Run reference implementation tests (for understanding)
cd reference/tonbistudio-ref && python test_turboquant.py

# Run reference validation suite
cd reference/tonbistudio-ref && python validate.py
```

## Hardware
- GPU: NVIDIA RTX 4090 (24GB VRAM, SM 89)
- CUDA: 12.8.93
- PyTorch: 2.10.0+cu128
- Python: 3.12

## Core Algorithm (from the paper)
TurboQuant is a two-stage vector quantization algorithm:

**Stage 1 — PolarQuant (MSE-optimal):**
1. Random orthogonal rotation of input vector (QR decomposition of Gaussian matrix)
2. After rotation, coordinates follow concentrated Beta distribution ≈ N(0, 1/d)
3. Apply precomputed Lloyd-Max scalar quantizer per coordinate
4. Store quantized indices (2-4 bits per coordinate)

**Stage 2 — QJL (1-bit bias correction):**
1. Compute residual: r = x - x_mse (Stage 1 reconstruction error)
2. Project through random Gaussian matrix S
3. Store only signs: 1 bit per dimension
4. This makes the inner product estimate mathematically unbiased

**Combined estimator:**
```
<q, k> ≈ <q, k_mse> + ||r|| * sqrt(π/2) / m * <S @ q, sign(S @ r)>
```

**Key Insight:** TurboQuant does NOT need accurate vector reconstruction. Individual vectors can have 23-44% reconstruction error. What matters is accurate **inner products** (attention scores). QJL ensures these are unbiased with variance O(1/d).

## Module Pipeline

The data flows through a chain of dependent modules. Each must be implemented and tested in order:

```
codebook.py ─────────────────────────────────────────────────────────┐
  Lloyd-Max quantizer for Beta/Gaussian distribution.               │
  Uses scipy.integrate.quad for continuous 1-D k-means.             │
  Precomputed once per (dimension, bit-width) pair, then cached.    │
  Output: centroids tensor + boundaries tensor                      │
                                                                    │
rotation.py                                                         │
  QR decomposition of random Gaussian matrix → orthogonal R.        │
  One matrix per head dimension, reused for all vectors.             │
  MUST be orthogonal (not just random) — breaks distribution         │
  assumptions otherwise.                                            │
                                                                    │
polarquant.py ◄──── codebook.py, rotation.py                        │
  Stage 1: rotate → per-coordinate Lloyd-Max quantize → indices.    │
  Dequantize: lookup centroids → inverse rotate.                    │
  Input: (batch, d) float vectors                                   │
  Output: integer indices + reconstructed vectors                   │
                                                                    │
qjl.py                                                              │
  Random Gaussian projection matrix S of shape (m, d), m defaults   │
  to d. Project residual through S, store only signs (1 bit each).  │
  Also stores ||residual|| as fp16 scalar per vector.               │
                                                                    │
estimator.py ◄──── polarquant.py, qjl.py                           │
  Combines both stages for unbiased inner product estimation.       │
  The total storage per key vector at b bits:                       │
    (b-1)*d bits MSE indices + d bits QJL signs + 16 bits norm      │
                                                                    │
kv_cache.py ◄──── estimator.py                                     │
  Drop-in KV cache wrapper.                                         │
  Keys: TurboQuantProd (needs unbiased inner products for attention)│
  Values: TurboQuantMSE only (needs MSE reconstruction, not IP)     │
```

## Key Constants (from paper)
- Head dimension d: typically 128 (Qwen, Llama) or 64 (some models)
- Optimal bit-widths: 3-bit (sweet spot), 3.5-bit (quality-neutral), 4-bit (near-lossless)
- QJL projection dimension m: paper uses m = d (same as head dimension)
- Rotation matrix: d × d orthogonal, generated once per head dimension

## Success Metrics
| Metric | Target | Paper Claims |
|---|---|---|
| 3-bit cosine similarity | >0.995 | 0.9945-0.9961 |
| 3-bit compression ratio | >4.5x | 5.0x |
| 3-bit top-5 attention match | >90% | 88-94% |
| Quantize throughput | >1M vectors/sec | Near-zero overhead |

## Implementation Rules

### DO:
- Implement from the paper directly. Read the math, implement the math.
- Use the reference implementation in `reference/tonbistudio-ref/` for understanding, NOT for copying code.
- Write tests first (TDD). Every module gets tests before implementation.
- Validate against the paper's theoretical bounds at each step.
- Use PyTorch for the initial implementation, Triton/CUDA for optimization.

### DON'T:
- Don't copy code from the reference implementation. Write it fresh.
- Don't skip the QJL stage. Stage 1 alone has biased inner products.
- Don't optimize prematurely. Get correctness first, speed second.
- Don't try to integrate with vLLM/SGLang until the core algorithm is validated.

## Reference Implementation Notes
The reference in `reference/tonbistudio-ref/` contains:
- `lloyd_max.py` — Lloyd-Max solver using `scipy.integrate.quad` for the exact Beta PDF and Gaussian approximation (N(0, 1/d), accurate for d >= 64)
- `turboquant.py` — `TurboQuantMSE` (Stage 1), `TurboQuantProd` (Stage 1+2), `TurboQuantKVCache` wrapper
- `validate.py` — Verification suite: codebook properties, MSE distortion bounds, inner product unbiasedness, needle-in-haystack, GPU benchmarks
- `test_turboquant.py` — Tests using real HuggingFace model (Qwen2.5-3B) with needle-in-haystack

Key patterns to note (implement differently, but understand the math):
- Codebook uses `scipy.integrate.quad` for continuous Lloyd-Max iterations — this is a one-time precomputation
- Keys use `TurboQuantProd` (both stages), values use `TurboQuantMSE` (Stage 1 only) — this asymmetry is intentional
- The QJL correction scale is `sqrt(π/2) / m` applied to `||residual|| * <S@q, sign(S@r)>`

## Workflow
1. Read PLAN.md for current status
2. Pick the next uncompleted task
3. Write tests first
4. Implement
5. Validate against paper bounds
6. Update PLAN.md and MEMORY.md
