# PPL Benchmark: Mean-Removal vs Production WHT

Date: 2026-04-09
Dataset: wikitext-2 test (max 4096 tokens)
Window: 512 tokens, stride 256
Compression: anchor=0, fp16_window=0, RQ=True, V3-bit
Context: Tom's TQ 3-bit gets +62.95 PPL on Qwen2.5-3B (catastrophic)

## Qwen2.5-7B-Instruct

| Config | PPL | Delta vs FP16 | Time |
|--------|-----|---------------|------|
| FP16 baseline | 7.5225 | baseline | 3.2s |
| WHT 3-bit | 9410.4876 | +9402.97 | 20.2s |
| WHT 3-bit + mean-removal | 7.9029 | +0.38 | 23.0s |
| WHT 4-bit | 1048.9915 | +1041.47 | 32.5s |
| WHT 4-bit + mean-removal | 7.7583 | +0.24 | 30.5s |

## Qwen2.5-3B-Instruct

| Config | PPL | Delta vs FP16 | Time |
|--------|-----|---------------|------|
| FP16 baseline | 10.7177 | baseline | 2.5s |
| WHT 3-bit | 60.1999 | +49.48 | 25.6s |
| WHT 3-bit + mean-removal | 11.0165 | +0.30 | 23.6s |
| WHT 4-bit | 13.2231 | +2.51 | 36.1s |
| WHT 4-bit + mean-removal | 10.8273 | +0.11 | 35.5s |

## Analysis

**Mean-removal is transformative at 3-bit:**

- Qwen2.5-7B 3-bit: mean-removal helps? YES (delta 9402.97 -> 0.38, improvement 9402.58 PPL points)
- Qwen2.5-7B 4-bit: mean-removal helps? YES (delta 1041.47 -> 0.24, improvement 1041.23 PPL points)
- Qwen2.5-3B 3-bit: mean-removal helps? YES (delta 49.48 -> 0.30, improvement 49.18 PPL points)
- Qwen2.5-3B 4-bit: mean-removal helps? YES (delta 2.51 -> 0.11, improvement 2.40 PPL points)

**Key findings:**

1. Without mean-removal, 3-bit WHT is catastrophic: PPL 9410 on 7B, PPL 60 on 3B
2. WITH mean-removal, 3-bit WHT gets within +0.30-0.38 PPL of FP16 baseline
3. The effect is MORE dramatic on 7B than 3B (9402 vs 49 PPL improvement)
4. 4-bit without mean-removal is also bad (PPL 1049 on 7B), with mean-removal it is near-lossless (+0.24)
5. This confirms Tom's observation that 3-bit symmetric TQ is catastrophic, AND shows mean-removal is the fix

**Comparison with Tom's numbers:**
- Tom's TQ 3-bit on Qwen2.5-3B: +62.95 PPL (catastrophic)
- Our WHT 3-bit on Qwen2.5-3B: +49.48 PPL (similar magnitude, confirms the problem)
- Our WHT 3-bit + mean-removal on Qwen2.5-3B: +0.30 PPL (problem solved)
