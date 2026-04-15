# KV Cache Compression Survey — April 2026

## Top Gaps to Fill (ranked by impact x feasibility)

| Rank | Technique | Paper | Compression | Difficulty | Status |
|------|-----------|-------|-------------|------------|--------|
| 1 | **KVTC** Procrustes cross-head | ICLR 2026 | 20-40x | MED-HIGH | GAP |
| 2 | **xKV** cross-layer SVD | Mar 2025 | 8.5x | MEDIUM | GAP (contradicts our r=0.001) |
| 3 | **AQUA-KV** linear cross-layer predictors | ICML 2025 | Additive | MEDIUM | GAP |
| 4 | **NSNQuant** double normalization | NeurIPS 2025 | Quality boost | LOW | PARTIAL (we have step 2/3) |
| 5 | **SnapKV** per-head observation window | NeurIPS 2024 | 8.2x memory | MEDIUM | GAP |
| 6 | **KVzip** query-agnostic eviction | NeurIPS 2025 Oral | 3-4x | HIGH | GAP |
| 7 | **DMS** delayed eviction | NVIDIA 2025 | 8x | MED-HIGH | GAP |
| 8 | **PALU** low-rank projection | ICLR 2025 | >91% reduction | MED-HIGH | GAP |
| 9 | **CommVQ** RoPE-commutative codebooks | ICML 2025 Apple | 87.5% | HIGH | GAP |
| 10 | **RotateKV** channel reorder | IJCAI 2025 | Quality boost | MEDIUM | PARTIAL |

## Novel Combinations Not Yet Tried (Anywhere)

1. **TurboQuant + KVTC Procrustes** — cross-head alignment before rotation → 30-50x
2. **Mean-removal + NSNQuant double norm** — add token-wise normalization steps 1+3
3. **AQUA-KV + TurboQuant residual** — inter-layer prediction reduces storage, TQ on residual
4. **KVzip eviction + TurboQuant + KVTC entropy** — three-stage → 50-100x
5. **xKV SVD + TurboQuant** — cross-layer subspace sharing of quantized indices

## Critical Insight: xKV Contradicts Our Prior Finding

Our `cross_layer_kv.py` measured raw KV correlation (r=0.001) and concluded cross-layer
sharing is not viable. xKV (Mar 2025) discovered that **singular vectors** are remarkably
aligned across layers even when raw vectors aren't. We measured the wrong thing.

## What We Already Do Well (no gap)

Rotation-based quant, ResidualQuant/QJL, asymmetric K/V, layer-adaptive bits, temporal
decay, token eviction, retrieval attention, cache distillation, entropy coding, channel-
adaptive, mixed-precision, attention sinks, fused attention, streaming, weight compression,
fractional bits, block rotations, cross-head delta, DeltaQuant, mean-removal, expected
attention pruning, learned rotation.
