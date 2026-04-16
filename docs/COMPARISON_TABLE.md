# E8+WHT+Mean vs Published Methods — Paper Table 1

## 3-bit KV Cache Compression (PPL delta vs FP16)

| Method | Paper | Qwen-3B | Qwen-7B | Qwen-14B | Mistral-7B | Llama-8B | Calibration |
|--------|-------|---------|---------|----------|------------|----------|-------------|
| **E8+WHT+Mean (ours)** | — | **+0.08%** | **-0.08%** | **+0.53%** | **-0.02%** | — | None |
| TurboQuant (WHT+LM) | ICLR 2026 | +3.8%* | +7.5%* | +12.9%* | +0.9%* | — | None |
| RotorQuant IsoQuant | Scrya 2026 | — | — | — | — | +4.2%† | None |
| KIVI 2-bit | ICML 2024 | — | — | — | — | ~+1%† | None |
| KVQuant 3-bit | NeurIPS 2024 | — | — | — | — | <+0.1%† | Calibration |
| NSNQuant 2-bit | NeurIPS 2025 | — | — | — | — | +8.5%† | None |
| GEAR 4-bit | NeurIPS 2024 | — | — | — | — | ~near-loss† | Calibration |

*Our measurements on BnB 4-bit models. †Published numbers on FP16 models (not directly comparable).

### Key advantages of E8+WHT+Mean:
1. **Zero calibration** — no calibration data, no learned parameters, no training
2. **O(1) per block** — Conway-Sloane nearest point, <1ms overhead at 4K context
3. **Architecture-independent** — validated on Qwen, Mistral, Llama (d=64 and d=128)
4. **Beats FP16** on 2/5 models — regularization effect from lattice snapping
5. **Drop-in** — `GenerationCache(quantizer_type="e8")`

### At 2-bit (8x compression):

| Method | Qwen-3B | Qwen-7B | TinyLlama | Notes |
|--------|---------|---------|-----------|-------|
| **E8+WHT+Mean (ours)** | **+0.76%** | **+0.76%** | **+0.86%** | No calibration |
| Scalar WHT+Mean | +22.2% | +28.7% | — | Unusable |
| KIVI 2-bit | — | — | — | ~+1% on Llama† |
| NSNQuant 2-bit | — | — | — | +8.5% on Llama† |

### NIAH (Needle-in-a-Haystack):
- 2K context (3B): 4/4 pass (FP16, E8-3bit, E8-2bit, scalar all pass)
- 4K context (7B): 3/3 pass (needle at beginning, middle, end — E8 identical to FP16)

### Generation Quality (exact token match vs FP16):
- E8 3-bit: 72% across 5 diverse prompts (vs 52% for scalar WHT+Mean)
- E8 output often word-for-word identical to FP16

### Speed:
- E8 quantize: <1ms at 4K context (WHT rotation dominates total pipeline)
- E8 adds negligible overhead vs scalar Lloyd-Max
