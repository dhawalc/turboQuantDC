# Research Findings — April 15, 2026

## Summary of 4 Deep Research Tracks

### 1. llama.cpp Flash Attention K Dequantize Bug

NOT a single bug — three separate issues:
- **Compile gating**: q4_1/q5_0/q5_1 K types need `GGML_CUDA_FA_ALL_QUANTS=ON` (not compiled by default)
- **MMA kernel limitation**: Prefill path (tensor core MMA) only handles f16/bf16. Quantized K only works on vec kernel (decode, batch=1). No one is working on upstream MMA quantized K support.
- **Historical MMQ bugs**: Fixed October 2024 (PR #10021, #10032)

**Contribution opportunity**: Expand default FA compile set (~10 lines) or add quantized K to MMA kernel (~500-1000 lines CUDA). The MMA kernel change would be a significant contribution — no one else is doing it.

### 2. Mean-Removal Prior Art (35 papers surveyed)

Mean-removal on KV cache is **not novel standalone**:
- TaDA (ACL Industry Jun 2025): per-head mean centering K+V
- NSNQuant (NeurIPS 2025): channel-wise centering + Hadamard + VQ
- SVDq (Feb 2025): per-channel mean on keys before SVD
- SageAttention2 (ICML 2025): only paper to note softmax shift-invariance for K centering

**Our novel contributions**:
1. Integration with rotation-based VQ (TurboQuant pipeline) — no prior art
2. Explicit softmax shift-invariance motivation for KV VQ — unique framing
3. Catastrophic→near-lossless magnitude (13,225→9.06 PPL) — unique to our pipeline
4. The KV-head-dependent effect (critical at 2-4 heads, harmful at 8+ heads on block rotations) — not reported anywhere

### 3. KV Compression Landscape (60+ techniques surveyed)

Top gaps to fill:
1. **KVTC** (ICLR 2026): Procrustes cross-head alignment + entropy coding = 20-40x
2. **xKV**: Cross-layer SVD (singular vector alignment even when raw r=0.001)
3. **AQUA-KV** (ICML 2025): Linear cross-layer predictors (additive compression)
4. **SnapKV**: Per-head observation window eviction (widely adopted)
5. **KVzip** (NeurIPS 2025 Oral): Query-agnostic eviction (3-4x)
6. **DMS** (NVIDIA): Delayed eviction (8x compression maintaining reasoning)
7. **PALU** (ICLR 2025): Low-rank projection (>91% reduction, complementary)
8. **CommVQ** (ICML 2025, Apple): RoPE-commutative codebooks

Novel untried combinations:
- TurboQuant + KVTC Procrustes → 30-50x
- AQUA-KV linear predictors + TurboQuant residual
- KVzip eviction + TurboQuant + KVTC entropy → 50-100x

### 4. Clifford Algebra for Quantization

**No mathematical advantage over simpler approaches.**
- RotorQuant has 9 sign errors in Cl(3,0) geometric product — trivector components are artifacts
- Grade-aware quantization wastes 23% extra bits on spurious components
- Block size 3 misaligns with power-of-2 dimensions
- Quaternion SO(4) (our IsoQuant) is strictly better: same DOF, perfect d=128 alignment
- WHT+Mean wins on PPL everywhere regardless of rotation type

**Recommendation**: Keep WHT as default. Investigate E8/Leech lattice VQ as a separate direction (30%+ distortion reduction over scalar Lloyd-Max in 8D/24D).

## xKV Cross-Layer SVD Investigation (Empirical)

Tested on Qwen2.5-3B (36 layers, d=128, 512 tokens):
- Raw correlation: r ≈ 0.025 (confirming our prior r=0.001 finding)
- **Singular vector alignment**: avg cos(principal angle) = 0.38 for adjacent layers
- **Shared subspace**: top-16 shared SVD preserves only 41% energy (group=2)
- xKV reports 0.9+ alignment on larger models — the effect may scale with model size
- Retested on 7B and 14B:

| Model | Layers | KV Heads | Gap-1 cos | Group-2 energy | Group-4 energy |
|-------|--------|----------|-----------|----------------|----------------|
| 3B | 36 | 2 | 0.379 | 41.2% | 36.6% |
| **7B** | 28 | 4 | **0.435** | **47.3%** | **43.5%** |
| 14B | 48 | 8 | 0.426 | 36.7% | 30.1% |

**Negative scaling result:** 14B has WORSE shared subspace quality than 7B. More layers + more heads = more diversity. xKV's 0.9+ alignment is likely Llama-specific, not universal. **Cross-layer SVD is NOT a promising direction for Qwen models.**

## Build/Test Results

- NSNQuant double normalization: PPL 122.66 vs WHT+Mean 11.87 — NSN designed for VQ codebooks, not scalar Lloyd-Max. **Our mean-removal is better for our pipeline.**
- RotorQuant comprehensive benchmark (3B/7B/14B): WHT wins everywhere on PPL. Mean-removal critical at low KV heads, neutral/harmful at high KV heads.

## E8 LATTICE VQ BREAKTHROUGH (April 15, 2026)

**E8+WHT+Mean achieves near-lossless 3-bit KV cache compression:**
- Qwen2.5-3B: PPL 11.44 (+0.1% vs FP16 11.44) — was 11.87 (+3.8%) with scalar Lloyd-Max
- Qwen2.5-7B: PPL 8.49 (+0.8% vs FP16 8.43) — was 9.06 (+7.5%) with scalar Lloyd-Max
- 86-89% lower MSE than scalar Lloyd-Max at same bit rate on synthetic data
- Algorithm: Conway-Sloane two-coset nearest E8 point, O(1) per 8D block, no calibration
- Implementation: turboquantdc/e8_lattice.py (~180 lines)
- Pipeline: mean-remove → normalize → WHT rotate → E8 quantize per 8D block → inverse
- Generation quality: 67.6% exact token match vs FP16 (vs 13.2% for scalar WHT+Mean)
- 2-bit E8 also viable: +1.3% PPL on 3B, +3.5% on 7B (scalar 2-bit is +22-29%)
- 14B results: +1.5% at 3-bit, +0.5% at 4-bit
- Entropy: E8 coords have 5.05 bits/dim entropy. Need QuIP# E8P encoding (2 bits/dim) for memory win
- 22 unit tests passing

### Full E8 Results Table

| Model | KV Heads | E8 2-bit | E8 3-bit | E8 4-bit | Scalar 3-bit+Mean | FP16 |
|-------|----------|----------|----------|----------|-------------------|------|
| 3B | 2 | +1.3% | **+0.1%** | — | +3.8% | baseline |
| 7B | 4 | +3.5% | **+0.8%** | — | +7.5% | baseline |
| 14B | 8 | — | **+1.5%** | **+0.5%** | +12.9% | baseline |

### AQUA-KV Cross-Layer Prediction (Prototype Result)

- Per-layer MSE: 78% improvement when predicting K[l] from K[l-1] + E8 on residual
- Cross-layer R²: 75.8% average on Qwen2.5-3B (range 69-95% across layers)
- **PPL: +11.2% (WORSE than E8 direct at +6.8%)** — error propagation kills gains
- Root cause: insufficient calibration (1 sentence vs needed 256 x 8K sequences)
- Needs full AQUA-KV pipeline: sequential training, RoPE-aware, reconstructed inputs
- D4 lattice NOT worth implementing (only 54% of E8's improvement over scalar)
