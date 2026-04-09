# Triple Stack Benchmark: EA + KVSculpt + TurboQuant

**Date:** 2026-04-09 16:49
**Model:** Qwen/Qwen2.5-3B-Instruct
**Sequence length:** 1086 tokens
**Head dimension:** 128
**Layers tested:** [0, 7, 15, 23, 35]
**Past/Future split:** 60% / 40%
**Distillation steps:** 50, lr=0.01

## The Triple Stack

```
Input: N tokens in FP16 KV cache
  |
  v
Stage 1: Expected Attention eviction (remove unimportant tokens)
  N -> N * (1 - eviction_rate)
  |
  v
Stage 2: KVSculpt distillation (synthesize fewer tokens)
  N' -> N' / distill_ratio
  |
  v
Stage 3: TurboQuant 3-bit (compress each token)
  FP16 -> 3-bit per coordinate
  |
  v
Output: Compressed cache for attention computation
```

## Results Summary (Averaged Across Layers)

| Configuration | Compression | Cosine | Rel Error | Top-5 | Time (ms) |
|---------------|-------------|--------|-----------|-------|-----------|
| FP16 (baseline) | 1.0x | 1.0000 | 0.0000 | 1.000 | 0.1 |
| EA-only 50% | 1.9x | 0.9937 | 0.0717 | 0.929 | 3.1 |
| Distill-only 4x | 4.0x | 0.9613 | 0.2196 | 0.867 | 529.2 |
| Quant-only 3-bit | 4.9x | 0.8261 | 0.5451 | 0.705 | 19.6 |
| EA 50% + Distill 4x | 7.7x | 0.9758 | 0.1873 | 0.864 | 240.0 |
| EA 50% + Quant 3-bit | 9.4x | 0.8185 | 0.5812 | 0.678 | 21.3 |
| Distill 4x + Quant 3-bit | 19.7x | 0.7692 | 5.7887 | 0.613 | 668.7 |
| Triple: EA50% -> D4x -> Q3 (40x) | 37.7x | 0.7715 | 0.9942 | 0.640 | 262.0 |
| Triple: D4x -> EA50% -> Q3 (40x) | 34.1x | 0.7027 | 1.8252 | 0.526 | 495.8 |
| Triple: EA30% -> D4x -> Q3 (28x) | 27.5x | 0.7002 | 2.4214 | 0.604 | 330.9 |
| Triple: EA70% -> D4x -> Q3 (66x) | 59.2x | 0.6993 | 1.7044 | 0.593 | 220.4 |

## Per-Layer Results

### FP16 (baseline)

| Layer | N_orig | N_final | Compression | Cosine | Rel Error | Top-5 | Time (ms) |
|-------|--------|---------|-------------|--------|-----------|-------|-----------|
| 0 | 1086 | 1086 | 1.0x | 1.0000 | 0.0000 | 1.000 | 0.3 |
| 7 | 1086 | 1086 | 1.0x | 1.0000 | 0.0000 | 1.000 | 0.1 |
| 15 | 1086 | 1086 | 1.0x | 1.0000 | 0.0000 | 1.000 | 0.0 |
| 23 | 1086 | 1086 | 1.0x | 1.0000 | 0.0000 | 1.000 | 0.1 |
| 35 | 1086 | 1086 | 1.0x | 1.0000 | 0.0000 | 1.000 | 0.1 |

### EA-only 50%

| Layer | N_orig | N_final | Compression | Cosine | Rel Error | Top-5 | Time (ms) |
|-------|--------|---------|-------------|--------|-----------|-------|-----------|
| 0 | 1086 | 586 | 1.9x | 0.9999 | 0.0103 | 0.989 | 9.1 |
| 7 | 1086 | 567 | 1.9x | 0.9979 | 0.0621 | 0.922 | 1.8 |
| 15 | 1086 | 558 | 1.9x | 1.0000 | 0.0052 | 0.998 | 2.4 |
| 23 | 1086 | 584 | 1.9x | 0.9757 | 0.1829 | 0.866 | 1.1 |
| 35 | 1086 | 553 | 2.0x | 0.9949 | 0.0980 | 0.868 | 1.3 |

### Distill-only 4x

| Layer | N_orig | N_final | Compression | Cosine | Rel Error | Top-5 | Time (ms) |
|-------|--------|---------|-------------|--------|-----------|-------|-----------|
| 0 | 1086 | 271 | 4.0x | 0.9559 | 0.1744 | 0.884 | 783.1 |
| 7 | 1086 | 271 | 4.0x | 0.9905 | 0.1418 | 0.884 | 474.8 |
| 15 | 1086 | 271 | 4.0x | 0.9823 | 0.1771 | 0.889 | 458.2 |
| 23 | 1086 | 271 | 4.0x | 0.8872 | 0.4906 | 0.803 | 458.7 |
| 35 | 1086 | 271 | 4.0x | 0.9908 | 0.1140 | 0.877 | 471.1 |

### Quant-only 3-bit

| Layer | N_orig | N_final | Compression | Cosine | Rel Error | Top-5 | Time (ms) |
|-------|--------|---------|-------------|--------|-----------|-------|-----------|
| 0 | 1086 | 1086 | 4.9x | 0.2035 | 1.8840 | 0.090 | 19.3 |
| 7 | 1086 | 1086 | 4.9x | 0.9751 | 0.2931 | 0.825 | 19.4 |
| 15 | 1086 | 1086 | 4.9x | 0.9755 | 0.2425 | 0.852 | 19.9 |
| 23 | 1086 | 1086 | 4.9x | 0.9875 | 0.1618 | 0.894 | 19.7 |
| 35 | 1086 | 1086 | 4.9x | 0.9887 | 0.1440 | 0.862 | 19.9 |

### EA 50% + Distill 4x

| Layer | N_orig | N_final | Compression | Cosine | Rel Error | Top-5 | Time (ms) |
|-------|--------|---------|-------------|--------|-----------|-------|-----------|
| 0 | 1086 | 146 | 7.4x | 0.9842 | 0.1197 | 0.910 | 246.6 |
| 7 | 1086 | 141 | 7.7x | 0.9896 | 0.1488 | 0.866 | 239.4 |
| 15 | 1086 | 139 | 7.8x | 0.9881 | 0.1325 | 0.913 | 230.8 |
| 23 | 1086 | 146 | 7.4x | 0.9289 | 0.3896 | 0.792 | 243.4 |
| 35 | 1086 | 138 | 7.9x | 0.9880 | 0.1457 | 0.838 | 239.8 |

### EA 50% + Quant 3-bit

| Layer | N_orig | N_final | Compression | Cosine | Rel Error | Top-5 | Time (ms) |
|-------|--------|---------|-------------|--------|-----------|-------|-----------|
| 0 | 1086 | 586 | 9.1x | 0.2034 | 1.8985 | 0.090 | 21.0 |
| 7 | 1086 | 567 | 9.4x | 0.9720 | 0.3053 | 0.818 | 22.2 |
| 15 | 1086 | 558 | 9.6x | 0.9753 | 0.2438 | 0.852 | 20.9 |
| 23 | 1086 | 584 | 9.2x | 0.9610 | 0.2634 | 0.837 | 21.5 |
| 35 | 1086 | 553 | 9.7x | 0.9809 | 0.1951 | 0.790 | 20.9 |

### Distill 4x + Quant 3-bit

| Layer | N_orig | N_final | Compression | Cosine | Rel Error | Top-5 | Time (ms) |
|-------|--------|---------|-------------|--------|-----------|-------|-----------|
| 0 | 1086 | 271 | 19.7x | 0.3143 | 26.8274 | 0.280 | 527.0 |
| 7 | 1086 | 271 | 19.7x | 0.9022 | 0.4911 | 0.712 | 488.0 |
| 15 | 1086 | 271 | 19.7x | 0.8045 | 0.6275 | 0.587 | 761.0 |
| 23 | 1086 | 271 | 19.7x | 0.8601 | 0.7376 | 0.753 | 1069.0 |
| 35 | 1086 | 271 | 19.7x | 0.9648 | 0.2601 | 0.733 | 498.5 |

### Triple: EA50% -> D4x -> Q3 (40x)

| Layer | N_orig | N_final | Compression | Cosine | Rel Error | Top-5 | Time (ms) |
|-------|--------|---------|-------------|--------|-----------|-------|-----------|
| 0 | 1086 | 146 | 36.6x | 0.1332 | 2.9955 | 0.217 | 267.4 |
| 7 | 1086 | 141 | 37.9x | 0.9467 | 0.6382 | 0.735 | 264.7 |
| 15 | 1086 | 139 | 38.5x | 0.9370 | 0.5258 | 0.800 | 254.3 |
| 23 | 1086 | 146 | 36.6x | 0.8781 | 0.5500 | 0.714 | 264.3 |
| 35 | 1086 | 138 | 38.7x | 0.9623 | 0.2616 | 0.736 | 259.0 |

### Triple: D4x -> EA50% -> Q3 (40x)

| Layer | N_orig | N_final | Compression | Cosine | Rel Error | Top-5 | Time (ms) |
|-------|--------|---------|-------------|--------|-----------|-------|-----------|
| 0 | 1086 | 161 | 33.2x | 0.0608 | 7.0788 | 0.092 | 556.1 |
| 7 | 1086 | 155 | 34.5x | 0.7917 | 0.5818 | 0.486 | 488.8 |
| 15 | 1086 | 153 | 34.9x | 0.8734 | 0.5274 | 0.686 | 474.5 |
| 23 | 1086 | 156 | 34.3x | 0.8445 | 0.6014 | 0.706 | 471.2 |
| 35 | 1086 | 160 | 33.4x | 0.9434 | 0.3368 | 0.660 | 488.6 |

### Triple: EA30% -> D4x -> Q3 (28x)

| Layer | N_orig | N_final | Compression | Cosine | Rel Error | Top-5 | Time (ms) |
|-------|--------|---------|-------------|--------|-----------|-------|-----------|
| 0 | 1086 | 196 | 27.3x | -0.1622 | 10.5266 | 0.035 | 354.7 |
| 7 | 1086 | 195 | 27.4x | 0.9131 | 0.4170 | 0.707 | 341.7 |
| 15 | 1086 | 192 | 27.8x | 0.9295 | 0.3611 | 0.782 | 303.0 |
| 23 | 1086 | 197 | 27.1x | 0.8516 | 0.5604 | 0.736 | 340.3 |
| 35 | 1086 | 191 | 28.0x | 0.9691 | 0.2419 | 0.758 | 314.9 |

### Triple: EA70% -> D4x -> Q3 (66x)

| Layer | N_orig | N_final | Compression | Cosine | Rel Error | Top-5 | Time (ms) |
|-------|--------|---------|-------------|--------|-----------|-------|-----------|
| 0 | 1086 | 94 | 56.9x | -0.1052 | 6.5950 | 0.179 | 224.7 |
| 7 | 1086 | 89 | 60.1x | 0.8934 | 0.6068 | 0.633 | 214.7 |
| 15 | 1086 | 87 | 61.5x | 0.9265 | 0.4038 | 0.766 | 216.2 |
| 23 | 1086 | 95 | 56.3x | 0.8336 | 0.6034 | 0.704 | 230.1 |
| 35 | 1086 | 87 | 61.5x | 0.9484 | 0.3132 | 0.680 | 216.5 |


## Results Summary (Layers 7-35 Only, Excluding Layer 0)

Layer 0 is the embedding-like first layer that always needs FP16 anchor treatment.
Excluding it gives the true picture for non-anchor layers.

| Configuration | Compression | Cosine | Rel Error | Top-5 | Time (ms) |
|---------------|-------------|--------|-----------|-------|-----------|
| FP16 (baseline) | 1.0x | 1.0000 | 0.0000 | 1.000 | 0.1 |
| EA-only 50% | 1.9x | 0.9921 | 0.0871 | 0.914 | 1.6 |
| Distill-only 4x | 4.0x | 0.9627 | 0.2309 | 0.863 | 465.7 |
| Quant-only 3-bit | 4.9x | 0.9817 | 0.2104 | 0.858 | 19.7 |
| EA 50% + Distill 4x | 7.7x | 0.9737 | 0.2041 | 0.852 | 238.4 |
| EA 50% + Quant 3-bit | 9.5x | 0.9723 | 0.2519 | 0.824 | 21.4 |
| Distill 4x + Quant 3-bit | 19.7x | 0.8829 | 0.5291 | 0.696 | 704.1 |
| Triple: EA50% -> D4x -> Q3 (40x) | 37.9x | 0.9310 | 0.4939 | 0.746 | 260.6 |
| Triple: D4x -> EA50% -> Q3 (40x) | 34.3x | 0.8632 | 0.5118 | 0.635 | 480.7 |
| Triple: EA30% -> D4x -> Q3 (28x) | 27.6x | 0.9158 | 0.3951 | 0.746 | 325.0 |
| Triple: EA70% -> D4x -> Q3 (66x) | 59.8x | 0.9005 | 0.4818 | 0.696 | 219.4 |

## Compression-Quality Curve (All Layers)

Sorted by compression ratio, showing the quality frontier:

| Rank | Configuration | Compression | Cosine | Pass >0.9? | Pass >0.95? |
|------|---------------|-------------|--------|------------|-------------|
| 1 | FP16 (baseline) | 1.0x | 1.0000 | YES | YES |
| 2 | EA-only 50% | 1.9x | 0.9937 | YES | YES |
| 3 | Distill-only 4x | 4.0x | 0.9613 | YES | YES |
| 4 | Quant-only 3-bit | 4.9x | 0.8261 | no | no |
| 5 | EA 50% + Distill 4x | 7.7x | 0.9758 | YES | YES |
| 6 | EA 50% + Quant 3-bit | 9.4x | 0.8185 | no | no |
| 7 | Distill 4x + Quant 3-bit | 19.7x | 0.7692 | no | no |
| 8 | Triple: EA30% -> D4x -> Q3 (28x) | 27.5x | 0.7002 | no | no |
| 9 | Triple: D4x -> EA50% -> Q3 (40x) | 34.1x | 0.7027 | no | no |
| 10 | Triple: EA50% -> D4x -> Q3 (40x) | 37.7x | 0.7715 | no | no |
| 11 | Triple: EA70% -> D4x -> Q3 (66x) | 59.2x | 0.6993 | no | no |

## Compression-Quality Curve (Layers 7-35 Only -- Excluding Layer 0)

| Rank | Configuration | Compression | Cosine | Pass >0.9? | Pass >0.95? |
|------|---------------|-------------|--------|------------|-------------|
| 1 | FP16 (baseline) | 1.0x | 1.0000 | YES | YES |
| 2 | EA-only 50% | 1.9x | 0.9921 | YES | YES |
| 3 | Distill-only 4x | 4.0x | 0.9627 | YES | YES |
| 4 | Quant-only 3-bit | 4.9x | 0.9817 | YES | YES |
| 5 | EA 50% + Distill 4x | 7.7x | 0.9737 | YES | YES |
| 6 | EA 50% + Quant 3-bit | 9.5x | 0.9723 | YES | YES |
| 7 | Distill 4x + Quant 3-bit | 19.7x | 0.8829 | no | no |
| 8 | Triple: EA30% -> D4x -> Q3 (28x) | 27.6x | 0.9158 | YES | no |
| 9 | Triple: D4x -> EA50% -> Q3 (40x) | 34.3x | 0.8632 | no | no |
| 10 | Triple: EA50% -> D4x -> Q3 (40x) | 37.9x | 0.9310 | YES | no |
| 11 | Triple: EA70% -> D4x -> Q3 (66x) | 59.8x | 0.9005 | YES | no |

## Stacking Order Comparison

Does the order of eviction vs distillation matter?

- **Order A (Evict then Distill):** cos=0.7715, compression=37.7x
- **Order B (Distill then Evict):** cos=0.7027, compression=34.1x

**Evict-first is better** by 0.0687 cosine.

## Aggressiveness Sweep (EA eviction rate)

| Eviction Rate | Compression | Cosine | Relative Error |
|---------------|-------------|--------|----------------|
| 30% | 27.5x | 0.7002 | 2.4214 |
| 50% | 37.7x | 0.7715 | 0.9942 |
| 70% | 59.2x | 0.6993 | 1.7044 |

## Key Findings

### All Layers (Including Layer 0)

- **Max compression with >0.90 cosine:** 7.7x (EA 50% + Distill 4x, cos=0.9758)
- **Max compression with >0.95 cosine:** 7.7x (EA 50% + Distill 4x, cos=0.9758)

### Non-Anchor Layers Only (Layers 7-35, Excluding Layer 0)

Layer 0 is the embedding-like first layer where 3-bit quantization catastrophically
fails (cos~0.20). In practice this layer uses an FP16 anchor. Excluding it shows
the true compression-quality frontier for the remaining 35 layers.

- **Max compression with >0.90 cosine:** 59.8x (Triple: EA70% -> D4x -> Q3 (66x), cos=0.9005)
- **Max compression with >0.95 cosine:** 9.5x (EA 50% + Quant 3-bit, cos=0.9723)

### Quality Stacking Analysis (layers 7-35)

- EA-only 50% loss: 0.0079 (cos=0.9921)
- Distill-only 4x loss: 0.0373 (cos=0.9627)
- Quant-only 3-bit loss: 0.0183 (cos=0.9817)
- **Triple stack actual loss: 0.0690 (cos=0.9310)**

- Additive prediction (independent losses): 0.0635
- Multiplicative prediction (compound cosines): 0.0624

**Approximately additive stacking:** Triple loss is roughly the sum of individual losses, suggesting the techniques degrade quality independently.

---
*Benchmark completed in 24.3s*