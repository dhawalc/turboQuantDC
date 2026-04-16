# TurboQuantDC

**The one-line fix that turns TurboQuant from PPL 9,410 to 7.90 on Qwen models.**

From-scratch KV cache compression library implementing TurboQuant (ICLR 2026). Built in one week. Found why TurboQuant catastrophically fails on Qwen models, proved the fix in production llama.cpp C code, and produced 3 publishable findings along the way.

155+ commits. 1,818+ tests. 31 experiments (10 breakthroughs, 7 dead ends -- all published). MIT license.

**NEW (April 15, 2026): E8 lattice VQ achieves near-lossless 3-bit KV cache compression — PPL +0.001% on Qwen2.5-3B (FP16 weights).** See [E8 results](#e8-lattice-vq) below.

[Live showcase](https://dhawalc.github.io/turboQuantDC/) | [PyPI](https://pypi.org/project/turboquantdc/)

---

## The Patch

Per-head key mean carries 57% of total variance but is invisible to attention (softmax is shift-invariant). Standard quantization wastes its entire codebook encoding something the model never uses.

```c
// Before quantizing keys, remove per-head channel mean.
// Softmax is shift-invariant: softmax(x + c) == softmax(x).
for (int h = 0; h < n_head; h++) {
    float mean = ggml_vec_mean(head_dim, keys + h * head_dim);
    ggml_vec_sub1(head_dim, keys + h * head_dim, mean);
}
```

### PPL Impact (wikitext-2)

| Model | Config | PPL | vs FP16 |
|-------|--------|-----|---------|
| Qwen2.5-7B | FP16 baseline | 7.52 | -- |
| Qwen2.5-7B | WHT 3-bit | **9,410** | +9,403 |
| Qwen2.5-7B | WHT 3-bit + mean-removal | **7.90** | +0.38 |
| Qwen2.5-3B | FP16 baseline | 10.72 | -- |
| Qwen2.5-3B | WHT 3-bit | **60.20** | +49.48 |
| Qwen2.5-3B | WHT 3-bit + mean-removal | **11.02** | +0.30 |
| Llama 3.1 8B (llama.cpp) | Q4_0 + FP16 KV | 7.50 | -- |
| Llama 3.1 8B (llama.cpp) | Q4_0 + turbo3 KV + mean-removal | **7.37** | **-0.13 (beats FP16)** |

### Needle-in-a-Haystack (8K context, Qwen2.5-7B)

| Config | 10% depth | 50% depth | 90% depth |
|--------|-----------|-----------|-----------|
| FP16 baseline | PASS | PASS | PASS |
| WHT 3-bit | FAIL | FAIL | FAIL |
| WHT 3-bit + mean-removal | PASS | PASS | PASS |

Without mean-removal, the model generates `0000., . numberWith); .0..0..` instead of `PINEAPPLE-77`.

---

## What's in the Box

**20 research experiments.** All run on RTX 4090 with real model KV caches. Full results, including failures.

### 8 Breakthroughs

| Finding | Result |
|---------|--------|
| Mean-removal quantization | PPL 9,410 -> 7.90 (1,190x improvement from one line) |
| ResidualQuant > QJL | Matches FP16 generation; QJL hurts it. Confirmed by TQ author. |
| Asymptotic compression law | Gini ~ 0.08\*ln(n), R^2=0.989. Longer context = better compression. |
| Block rotation + mean-removal | Beats RotorQuant at 3-bit without calibration data |
| Triple-stack compression | 37.9x at 0.93 cosine (eviction + distillation + quantization) |
| KVSculpt cache distillation | 19.7x near-lossless token synthesis |
| Expected Attention pruning | 10x at 0.978 cosine (analytical importance prediction) |
| Cayley learned rotation | Novel attention-KL objective; modest practical gain (+0.002-0.006) |

### E8 Lattice VQ (April 15, 2026) {#e8-lattice-vq}

Replacing scalar Lloyd-Max with E8 lattice vector quantization (8D blocks) after WHT rotation reduces MSE by 86-89% at the same bit rate. The E8 lattice achieves optimal sphere packing in 8 dimensions (Viazovska 2016, Fields Medal).

```python
from turboquantdc import GenerationCache

# Drop-in: just set quantizer_type="e8"
cache = GenerationCache(key_bits=3, val_bits=3, quantizer_type="e8")
output = model.generate(inputs, past_key_values=cache, max_new_tokens=100)
```

| Model | Weights | E8 3-bit PPL | vs FP16 | Scalar 3-bit |
|-------|---------|-------------|---------|-------------|
| Qwen2.5-3B | FP16 | 9.7262 | **+0.001%** | +3.8% |
| Qwen2.5-7B | FP16 | 6.9881 | +0.20% | +7.5% |
| Qwen2.5-3B | BnB 4-bit | 11.4438 | +0.08% | +3.8% |
| Qwen2.5-7B | BnB 4-bit | 8.4228 | -0.08% | +7.5% |
| Mistral-7B | BnB 4-bit | 8.2206 | -0.02% | +0.9% |
| TinyLlama-1.1B | BnB 4-bit | 10.9608 | +0.20% | +8.3% |

**No calibration. No learned parameters. O(1) per 8D block. <1ms overhead at 4K context.**

Run the demo: `python demo_e8.py --model Qwen/Qwen2.5-3B-Instruct --bits 3`

### 4 Dead Ends (published)

| Experiment | Why It Failed |
|------------|---------------|
| XQuant rematerialization | X is 4x larger than K+V in GQA; 0.815 cosine |
| Expected Attention on topic shifts | Anti-correlated (-0.035 Spearman) when distribution shifts |
| Cross-model Cayley transfer | Must recalibrate per model; no transfer across architectures |
| TurboRetrievalCache at scale | FAISS index corruption above 2K tokens |

### Model Validation (3B to 72B)

| Model | Bits | Cosine Sim | Top-5 Match | Compression | Generation |
|-------|------|-----------|-------------|-------------|------------|
| Qwen2.5-3B | 3 | 0.9969 | 94.4% | 5.0x | 5/5 match |
| Gemma 4 E4B | 3 | 0.999994 | 100% | 5.12x | 150 tok/s @ 262K |
| Qwen2.5-14B | 3 | 0.9964 | 95.3% | 5.0x | 5/5 match |
| Qwen3.5-27B | 3 | 0.9932 | 100% | 5.2x | -- |
| Qwen2.5-32B | 3 | -- | -- | 5.0x | 5/5 match |
| Llama 3.1 70B | 3 | -- | -- | 5.0x | 4x context (4K->16K) |

Gemma 4 26B at full 262K native context on a single RTX 4090 at 150 tok/s. FP16 KV OOMs.

---

## Install

```bash
pip install turboquantdc

# With all dependencies (scipy for codebook, triton for CUDA kernels)
pip install turboquantdc[all]

# From source
git clone https://github.com/dhawalc/turboQuantDC.git
cd turboQuantDC
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from turboquantdc import GenerationCache

# Drop-in KV cache replacement for HuggingFace models
cache = GenerationCache(
    num_layers=36,
    num_heads=2,
    head_dim=128,
    bits=3,
    mean_removal=True,       # the fix
    residual_quant=True,     # better than QJL
    fp16_window=128,         # recent tokens at full precision
    anchor_layers=[0, 1, -2, -1],  # boundary protection
    device="cuda"
)

# Use with any HuggingFace model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

inputs = tokenizer("Explain quantum computing:", return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    past_key_values=cache,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Run Tests

```bash
python -m pytest tests/ -v          # 1,796 tests
python -m pytest tests/ -v -x -q    # stop on first failure
```

---

## Research Findings

### 1. Mean-Removal is Not an Optimization. It Fixes Catastrophic Failure.

The per-head channel mean in Qwen KV caches is large relative to the signal. At 3-bit, the Lloyd-Max codebook cannot represent both the mean and the variation. It clips the signal. The model sees near-uniform attention and generates random tokens.

Mean-removal is trivial: subtract the mean before quantizing. Softmax is shift-invariant, so attention scores are identical. The codebook now has full resolution for the actual signal.

Prior art: NSNQuant (May 2025) published channel centering for KV quantization. The technique itself is known. Connecting it to TurboQuant's catastrophic failure on Qwen models, and proving it is the single root cause (not just an optimization), is the new contribution.

### 2. ResidualQuant Replaces QJL

The paper's QJL stage (random Gaussian projection for 1-bit residual correction) introduces variance that compounds during autoregressive decoding. ResidualQuant stores `sign(residual)` directly in the rotated space -- same 1-bit budget, no random projection matrix, lower variance. Generation goes from garbled to FP16-matching.

Tom Turney confirmed his team had independently been working on removing QJL.

### 3. Asymptotic Compression Law

Attention concentration increases logarithmically with context length:

```
G(n) = 0.08 * ln(n) + beta    (R^2 = 0.989)
```

At 128 tokens, 12.8% of tokens get >1% attention. At 2,094 tokens, only 0.3% do. The theoretical minimum bits per token scales as O(1/n). Longer context is inherently more compressible.

### 4. Weight + KV Stacking

When combined with TurboQuant weight compression in llama.cpp:

- Llama 3.1 8B: max context extends from ~48K to ~100K (2.1x) on RTX 4090
- Llama 3.1 70B: max context extends from ~4K to ~16K (4x) on RTX 4090
- At 8K context, turbo3 KV is actually faster than FP16 (86.5 vs 78.4 tok/s) due to less VRAM pressure

---

## Architecture

```
codebook.py         Lloyd-Max quantizer (scipy.integrate.quad, precomputed once)
rotation.py         WHT butterfly rotation, O(d log d), CUDA kernel
polarquant.py       Stage 1: rotate -> per-coordinate quantize -> indices
qjl.py              Stage 2: random projection + sign (paper's approach)
residual_quant.py   Stage 2 replacement: sign(residual) in rotated space
estimator.py        Combined inner product estimation
generation_core.py  Production GenerationCache with mean-removal + anchoring
kv_cache.py         HuggingFace Cache protocol integration
```

Key design decisions:
- Keys use both stages (need unbiased inner products for attention)
- Values use Stage 1 only (need MSE reconstruction, not inner products)
- Layer 0 always needs FP16 (anomalous activation patterns)

## Hardware

All benchmarks on: NVIDIA RTX 4090 (24 GB VRAM, SM 89), CUDA 12.8, PyTorch 2.10+

## Links

- Paper: Ashkboos et al., "TurboQuant: Online Vector Quantization for KV Cache Compression" (ICLR 2026)
- Tom Turney's implementation: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
- Live results: [dhawalc.github.io/turboQuantDC](https://dhawalc.github.io/turboQuantDC/)

## License

MIT
