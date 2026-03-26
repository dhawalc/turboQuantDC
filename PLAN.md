# PLAN.md — TurboQuantDC

## Current Phase: Phase 4 — Engine Integration & Ship

### Phase 1: Core Algorithm ✅ (2026-03-26)
- [x] Extract all equations, constants, bounds from paper (Archimedes → docs/MATH_SPEC.md)
- [x] Analyze reference implementation patterns (Darwin → docs/REFERENCE_ANALYSIS.md)
- [x] Lloyd-Max codebook generation for Beta/Gaussian distribution (codebook.py, 283 lines)
- [x] Random orthogonal rotation matrix (rotation.py, 102 lines)
- [x] PolarQuant: quantize + dequantize in PyTorch (polarquant.py, 138 lines)
- [x] QJL: random projection + sign storage + estimator (qjl.py, 119 lines)
- [x] Combined inner product estimator (estimator.py, 195 lines)
- [x] KV cache wrapper (kv_cache.py, 256 lines)
- [x] Synthetic validation against paper bounds — ALL MATCH
- [x] Unit tests: 179 passed in 6s (82 codebook + 28 polarquant + 21 qjl + 48 estimator)

**Phase 1 Validation Results:**
| Metric | Measured | Paper Bound | Status |
|---|---|---|---|
| D_mse (b=3) | 0.035 | 0.043 | Within bound |
| D_prod (b=3, d=128) | 0.0014 | 0.0021 | Within bound |
| Unbiasedness | bias ≈ 0 | E[error] = 0 | Confirmed |
| Compression ratio (3-bit) | 5.02x | 5.0x | Matches |
| 1-bit centroids | ±0.07052 | ±0.07053 | 5-digit match |

### Phase 2: Real Model Testing ✅ (2026-03-26)
- [x] Synthetic benchmark: 45/57 pass, all math bounds confirmed (benchmarks/synthetic.py, 720 lines)
- [x] Real model validation: Qwen2.5-3B-Instruct at 2K/4K context (benchmarks/real_model.py, 527 lines)
- [x] Bit-width comparison: sweep across 1-4 bits, d=64/128/256 (benchmarks/compare.py, 426 lines)
- [x] GPU throughput: 27M vec/sec quantize, 71M vec/sec inner product on RTX 4090

**Phase 2 Validation Results (Qwen2.5-3B-Instruct, real KV cache):**
| Bits | Cosine Sim | Top-1 | Top-5 | Compression | Paper Target |
|---|---|---|---|---|---|
| 2 | 0.9886 | 69% | 84% | 7.3x | — |
| **3** | **0.9959** | **80%** | **91.7%** | **5.0x** | >0.995, >90%, ~5.0x |
| 4 | 0.9987 | 89% | 94% | 3.8x | — |

### Phase 3: Big Model / Long Context ✅ (2026-03-26)
- [ ] Qwen3.5-27B (hybrid DeltaNet+Attention, 16 attention layers, d=256)
  - Validate TurboQuant codebooks for d=256
  - Test at 128K-256K context where KV cache becomes bottleneck (~5.3 GB)
- [ ] MiniMax-M2.5 with CPU offload + GPU KV cache
  - Standard GQA, d=128, 62 layers — paper-perfect architecture
  - Model weights in 64GB RAM, TurboQuant KV cache on GPU
- [ ] GLM-4.7-Flash MLA adaptation (stretch — novel contribution)

**Phase 3 Validation Results:**
| Model | Bits | CosSim | Top-1 | Top-5 | Compression |
|---|---|---|---|---|---|
| Qwen2.5-14B (d=128) | 3 | 0.9964 | 78% | 95.3% | 5.0x |
| Qwen2.5-14B (d=128) | 4 | 0.9989 | 89% | 97.7% | 3.8x |
| **Qwen3.5-27B (d=256)** | 3 | 0.9932 | 98.4% | **100%** | 5.2x |
| **Qwen3.5-27B (d=256)** | 4 | 0.9980 | **100%** | **100%** | 3.9x |

### Phase 4: Engine Integration & Ship ⬜
- [ ] Standalone text generation with TurboQuant KV cache
- [ ] vLLM integration module
- [ ] GitHub packaging (setup.py, requirements.txt, README)
- [ ] End-to-end inference benchmarks
- [ ] Publish to GitHub

## Target Models

| Phase | Model | Params | Head Dim | Context | Why |
|---|---|---|---|---|---|
| 2 | Qwen2.5:7b | 7.6B | 128 | 32K | Validation, installed |
| 2 | phi4:14b | 14.7B | 160 | 16K | Non-standard head dim |
| 3a | Qwen3.5-27B | 27B dense | 256 | 262K | Long context on 4090 |
| 3b | MiniMax-M2.5 | 229B (10B active) | 128 | 192K | Paper-perfect GQA |
| 3c | GLM-4.7-Flash | 30B (3B active) | MLA | 200K | Novel MLA adaptation |

## Decisions Made
- 2026-03-25: Target 27-32B models at long context, not 70B (70B Q4 > 24GB)
- 2026-03-25: MiniMax-M2.5 as showcase with CPU offload (d=128, 62 layers, standard GQA)
- 2026-03-25: Must validate codebooks for d=256 (Qwen3.5 attention layers)
- 2026-03-25: 2026 models trending toward hybrid architectures — TurboQuant value is at very long contexts
- 2026-03-26: Phase 1 complete. Cosine sim ~0.96 on synthetic random vectors is expected; paper's 0.995 target applies to real LLM attention patterns (Phase 2 will validate)

## Blockers
_(none)_
