# TurboQuant Implementation Plan

*Our own from-scratch implementation for RTX 4090. March 25, 2026.*

---

## Objective

Build a clean PyTorch implementation of TurboQuant that:
1. Compresses KV cache to 3-bit with <0.5% attention quality loss
2. Runs on our RTX 4090 with real models (Qwen 27B, GLM-5, MiniMax M2.5)
3. Integrates with vLLM/SGLang for actual inference
4. Publishable as open-source (cortex-quant or similar)

## Architecture (from the paper)

### Stage 1: PolarQuant (MSE-optimal compression)
```
Input vector x (FP16, d dimensions)
    → Random orthogonal rotation: x_rot = R @ x
    → Coordinates now follow Beta distribution ≈ Gaussian N(0, 1/d)  
    → Apply Lloyd-Max scalar quantizer per coordinate
    → Store quantized indices (2-4 bits per coordinate)
    → Reconstruct: x_mse = R^T @ dequantize(indices)
```

Key insight: random rotation makes coordinates nearly independent with known distribution, so optimal scalar quantization works perfectly.

### Stage 2: QJL (1-bit bias correction)
```
Residual = x - x_mse
    → Project through random Gaussian matrix S: projected = S @ residual  
    → Store only signs: sign_bits = sign(projected)
    → 1 bit per dimension
```

### Combined inner product estimator:
```
<q, k> ≈ <q, k_mse> + ||residual|| * sqrt(π/2) / m * <S @ q, sign_bits>
```

This is mathematically unbiased — the Stage 1 quantization has bias, QJL corrects it.

## Implementation Modules

### Module 1: Lloyd-Max Codebook Generator
- Precompute optimal quantization codebooks for target distribution
- Inputs: bit-width (2,3,4), dimension, distribution params
- Output: centroids + boundaries lookup table
- One-time computation, cached

### Module 2: Random Rotation Matrix
- Generate orthogonal matrix via QR decomposition of Gaussian
- Fixed per model head dimension (typically 128)
- Stored and reused across all vectors

### Module 3: PolarQuant Quantizer
- Rotate → quantize per-coordinate → store indices
- Dequantize → lookup centroids → inverse rotate
- CUDA kernel for batch processing on 4090

### Module 4: QJL Bias Correction
- Random Gaussian projection matrix (fixed per head)
- Compute residual, project, store signs (1 bit each)
- Estimator computation at attention time

### Module 5: KV Cache Wrapper
- Drop-in replacement for standard KV cache
- Intercept cache writes → compress
- Intercept attention → use TurboQuant estimator
- Track compression stats

### Module 6: Integration Layer
- vLLM PagedAttention hook
- SGLang RadixAttention hook
- Standalone benchmark mode (no inference engine needed)

## Research Team (Subagent Delegation)

### Agent 1: Math Researcher (Opus)
**Task:** Read the full paper (arxiv 2504.19874), extract every equation, constant, and algorithm detail. Produce a complete mathematical spec with:
- Exact Lloyd-Max quantizer formulas for Beta distribution
- QJL projection dimensions and scaling constants
- The combined estimator with all normalization terms
- Theoretical bounds we should validate against

### Agent 2: Reference Analyzer (Sonnet)
**Task:** Analyze tonbistudio/turboquant-pytorch code. Extract:
- Their Lloyd-Max implementation approach
- How they handle the rotation matrix
- QJL implementation details
- Any deviations from the paper
- Bugs or limitations they mention

### Agent 3: CUDA/Performance Engineer (Opus)
**Task:** Design the CUDA kernel strategy for RTX 4090:
- Batch quantization kernel (rotate + quantize in one pass)
- Fused attention kernel (TurboQuant estimator instead of standard dot product)
- Memory layout for 3-bit packed storage
- Benchmark plan: throughput, latency, VRAM usage

### Agent 4: Integration Engineer (Sonnet)
**Task:** Map how to hook into vLLM and SGLang:
- Where KV cache is created/accessed in each engine
- Hook points for custom quantization
- How to make it a config flag (--kv-quant turboquant-3bit)
- Testing plan with Qwen 27B

## Execution Plan

### Phase 1: Core Algorithm (Week 1)
- [ ] Lloyd-Max codebook generation
- [ ] Random rotation matrix (QR decomposition)
- [ ] PolarQuant: quantize/dequantize in PyTorch
- [ ] QJL: projection + sign storage
- [ ] Combined estimator
- [ ] Validate against paper's synthetic benchmarks

### Phase 2: Real Model Testing (Week 2)
- [ ] KV cache wrapper for HuggingFace models
- [ ] Test on Qwen2.5:7b (already on machine)
- [ ] Test on phi4:14b (already on machine)
- [ ] Attention fidelity benchmarks (cosine sim, top-k match)
- [ ] Compression ratio verification

### Phase 3: Big Model Testing (Week 3)
- [ ] Pull Qwen 27B or MiniMax M2.5
- [ ] Test if 27B fits on 4090 with TurboQuant 3-bit KV
- [ ] Long context benchmarks (32k, 64k, 128k tokens)
- [ ] Compare: baseline FP16 vs TQ-3bit vs TQ-4bit

### Phase 4: Engine Integration (Week 4)
- [ ] vLLM integration (PagedAttention hook)
- [ ] SGLang integration (RadixAttention hook)
- [ ] End-to-end inference benchmarks
- [ ] Publish to GitHub

## Success Criteria

| Metric | Target | Paper Claims |
|---|---|---|
| 3-bit cosine similarity | >0.995 | 0.9945-0.9961 |
| 3-bit compression ratio | >4.5x | 5.0x |
| 3-bit top-5 attention match | >90% | 88-94% |
| 27B model on 4090 at 32k ctx | Fits | New (our contribution) |
| Quantization overhead | <5% of total inference | Near-zero per paper |

## Risks

1. **CUDA kernels are hard** — pure PyTorch might be too slow for real inference. May need Triton kernels.
2. **vLLM/SGLang internals change fast** — integration may break across versions.
3. **Paper's claims may not hold at 27B scale** — they tested on Gemma/Mistral (smaller models).
4. **Random rotation matrix storage** — one per head × layer. For a 64-layer model with 64 heads = 4,096 matrices. Need to verify memory impact.

## Budget

- All research agents: Opus with --max-budget-usd $3.00 each
- Implementation: Claude Code sessions (local, no Codex)
- Total estimated: <$15 in API costs
