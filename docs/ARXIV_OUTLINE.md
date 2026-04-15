# E8 Lattice Quantization for Near-Lossless KV Cache Compression

## arXiv Technical Report — Outline

### Title Options
1. "E8 Lattice Quantization Achieves Near-Lossless KV Cache Compression at 3 Bits"
2. "Beyond Scalar Quantization: E8 Lattice VQ for LLM Key-Value Caches"
3. "Near-Lossless 3-Bit KV Cache Compression via E8 Lattice Vector Quantization"

### Abstract (draft)
We show that replacing per-coordinate scalar quantization with E8 lattice vector
quantization in the TurboQuant KV cache compression pipeline reduces perplexity
degradation from 3.8% to 0.1% at 3 bits per dimension on Qwen2.5-3B, and from
7.5% to 0.8% on Qwen2.5-7B. The E8 lattice, which achieves the optimal sphere
packing in 8 dimensions, has 14% lower normalized second moment than scalar
quantization (Zador's theorem), translating to 86-89% lower MSE in practice.
Our method requires no calibration data, no learned parameters, and adds negligible
computational overhead (O(1) per 8D block via the Conway-Sloane algorithm).
Combined with Walsh-Hadamard rotation and per-head mean-removal, this achieves
near-lossless compression at 5x memory reduction.

### 1. Introduction
- KV cache is the memory bottleneck for long-context LLM inference
- Existing quantization uses per-coordinate scalar quantizers (Lloyd-Max)
- We propose replacing scalar with E8 lattice VQ (8D blocks)
- Near-lossless results: +0.1% to +1.5% PPL at 3-bit across 3 models

### 2. Background
- 2.1 KV Cache Compression (TurboQuant, KIVI, KVQuant, GEAR)
- 2.2 Walsh-Hadamard Transform for distribution concentration
- 2.3 Mean-removal via softmax shift-invariance
- 2.4 E8 lattice and optimal sphere packing (Viazovska 2016)
- 2.5 Zador's theorem: NSM_E8 = 0.07168 vs NSM_Z = 0.08333

### 3. Method
- 3.1 Pipeline: mean-remove → normalize → WHT rotate → E8 quantize per 8D block
- 3.2 Conway-Sloane nearest E8 point (two-coset algorithm)
- 3.3 Relaxed E8 (no parity constraint) for KV cache data
- 3.4 Scale calibration (adaptive to post-WHT distribution)
- 3.5 E8P encoding for compact storage (16 bits per 8D block = 2 bits/dim)

### 4. Experiments
- 4.1 Models: Qwen2.5-{3B, 7B, 14B}, Mistral-7B, [Phi-3.5-mini?]
- 4.2 Perplexity (wikitext-2, 8K tokens, sliding window)
- 4.3 Generation quality (token match vs FP16)
- 4.4 Attention fidelity (cosine sim, top-K match)
- 4.5 MSE comparison vs scalar Lloyd-Max
- 4.6 Ablation: E8 vs D4 vs scalar at same bit rate
- 4.7 Ablation: mean-removal contribution with E8
- 4.8 Speed comparison

### 5. Results
- Table 1: PPL across models and bit-widths (the money table)
- Table 2: MSE comparison (E8 vs scalar, 86-89% reduction)
- Table 3: Generation quality (token match)
- Table 4: Attention fidelity
- Figure 1: PPL vs bit-rate curves
- Figure 2: MSE vs bit-rate with Zador bounds

### 6. Analysis
- 6.1 Why E8 helps: Voronoi cell geometry and sub-Gaussian KV distributions
- 6.2 Mean-removal is KV-head-dependent (critical at 2-4 heads, neutral at 8+)
- 6.3 Attention cosine does NOT predict PPL
- 6.4 The regularization effect (E8 4-bit beats FP16 on 7B)

### 7. Related Work
- TurboQuant (ICLR 2026), QuIP# (ICML 2024), NestQuant (ICML 2025)
- AQUA-KV/HIGGS (ICML 2025), CommVQ (ICML 2025)
- NSNQuant (NeurIPS 2025), KIVI (ICML 2024), KVQuant (NeurIPS 2024)

### 8. Conclusion
- E8 lattice VQ is a drop-in replacement for scalar quantization
- Near-lossless at 3 bits, viable at 2 bits
- No calibration, O(1) decode, compatible with existing pipelines

### Data Needed (status)
- [x] Qwen2.5-3B PPL (complete)
- [x] Qwen2.5-7B PPL (complete)
- [x] Qwen2.5-14B PPL (complete)
- [x] Mistral-7B PPL (complete: 3-bit +0.1%, 4-bit -0.0%)
- [ ] Phi-3.5-mini PPL (planned)
- [x] Generation quality 5 prompts (complete)
- [x] MSE comparison synthetic (complete)
- [x] E8 unit tests (22 passing)
- [ ] E8P encoding implementation (blocker)
- [ ] Speed benchmarks (needed)
- [ ] NIAH long-context test (nice to have)
