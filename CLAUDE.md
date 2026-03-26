# CLAUDE.md — TurboQuantDC

## Project Identity
You are building **TurboQuantDC** — a from-scratch PyTorch + CUDA implementation of Google's TurboQuant algorithm (ICLR 2026) for compressing LLM key-value caches.

## Objective
Compress KV cache to 3-bit with <0.5% attention quality loss on an RTX 4090. Enable running 27B+ parameter models locally with long context that would normally OOM.

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

## Key Insight
TurboQuant does NOT need accurate vector reconstruction. Individual vectors can have 23-44% reconstruction error. What matters is accurate **inner products** (attention scores). QJL ensures these are unbiased with variance O(1/d).

## Architecture

```
turboQuantDC/
├── CLAUDE.md              # This file
├── PLAN.md                # Current progress + next steps
├── MEMORY.md              # Accumulated decisions and gotchas
├── docs/
│   ├── IMPLEMENTATION_PLAN.md   # Full implementation plan
│   ├── turboquant_paper.pdf     # Original paper (arxiv 2504.19874)
│   └── google_blog.html        # Google Research blog post
├── reference/
│   └── tonbistudio-ref/        # Reference implementation (for analysis only, DO NOT COPY)
├── turboquantdc/
│   ├── __init__.py
│   ├── codebook.py             # Lloyd-Max codebook generation
│   ├── rotation.py             # Random orthogonal rotation matrix
│   ├── polarquant.py           # Stage 1: PolarQuant quantizer
│   ├── qjl.py                  # Stage 2: QJL bias correction
│   ├── estimator.py            # Combined inner product estimator
│   ├── kv_cache.py             # KV cache wrapper (drop-in replacement)
│   └── kernels/
│       ├── quantize.cu         # CUDA kernel for batch quantization
│       └── attention.cu        # Fused attention with TurboQuant
├── benchmarks/
│   ├── synthetic.py            # Validate against paper's theoretical bounds
│   ├── real_model.py           # Test on actual LLM KV cache
│   └── compare.py              # Compare bit-widths, measure fidelity
├── tests/
│   ├── test_codebook.py
│   ├── test_polarquant.py
│   ├── test_qjl.py
│   └── test_estimator.py
└── setup.py
```

## Implementation Rules

### DO:
- Implement from the paper directly. Read the math, implement the math.
- Use the reference implementation in `reference/tonbistudio-ref/` for understanding, NOT for copying code.
- Write tests first (TDD). Every module gets tests before implementation.
- Validate against the paper's theoretical bounds at each step.
- Use PyTorch for the initial implementation, Triton/CUDA for optimization.
- Target RTX 4090 (24GB VRAM, CUDA 12.8, SM 89).

### DON'T:
- Don't copy code from the reference implementation. Write it fresh.
- Don't skip the QJL stage. Stage 1 alone has biased inner products.
- Don't optimize prematurely. Get correctness first, speed second.
- Don't try to integrate with vLLM/SGLang until the core algorithm is validated.
- Don't use any paid API models for automation. Use local Ollama if needed.

## Hardware
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- CUDA: 12.8.93
- PyTorch: 2.10.0+cu128
- Python: 3.12

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

## Workflow
1. Read PLAN.md for current status
2. Pick the next uncompleted task
3. Write tests first
4. Implement
5. Validate against paper bounds
6. Update PLAN.md and MEMORY.md
