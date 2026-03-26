# PLAN.md — TurboQuantDC

## Current Phase: Phase 1 — Core Algorithm

### Phase 1: Core Algorithm ⬜
- [ ] Lloyd-Max codebook generation for Beta/Gaussian distribution
- [ ] Random orthogonal rotation matrix (QR decomposition)
- [ ] PolarQuant: quantize + dequantize in PyTorch
- [ ] QJL: random projection + sign storage + estimator
- [ ] Combined inner product estimator
- [ ] Synthetic validation against paper bounds (Table 1 & 2)
- [ ] Unit tests for all modules

### Phase 2: Real Model Testing ⬜
- [ ] KV cache wrapper for HuggingFace models
- [ ] Test on Qwen2.5:7b (already on machine via Ollama)
- [ ] Test on phi4:14b (already on machine)
- [ ] Attention fidelity benchmarks (cosine sim, top-k match)
- [ ] Compression ratio verification
- [ ] Needle-in-haystack test

### Phase 3: Big Model Testing ⬜
- [ ] Pull Qwen 27B (or MiniMax M2.5)
- [ ] Test if 27B fits on 4090 with 3-bit KV cache
- [ ] Long context benchmarks (32k, 64k, 128k)
- [ ] Quality comparison: FP16 vs TQ-3bit vs TQ-4bit

### Phase 4: Engine Integration ⬜
- [ ] vLLM PagedAttention hook
- [ ] SGLang RadixAttention hook
- [ ] End-to-end inference benchmarks
- [ ] Publish to GitHub

## Decisions Made
_(none yet)_

## Blockers
_(none yet)_
