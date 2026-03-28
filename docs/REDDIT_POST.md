# Reddit Post — r/LocalLLaMA

## Title

**TurboQuantDC: 5x KV cache compression at 3-bit with <0.5% quality loss -- 331 tests, validated on real models up to 27B, runs on RTX 4090**

---

## Body

I built a from-scratch PyTorch implementation of Google's TurboQuant algorithm (ICLR 2026, [arxiv 2504.19874](https://arxiv.org/abs/2504.19874)) for compressing LLM key-value caches. It compresses 16-bit KV vectors down to 3-bit with 5x compression, and the attention scores stay accurate. 331 tests, MIT license. Repo: [github.com/dhawal/turboQuantDC](https://github.com/dhawal/turboQuantDC)

### The problem

Every token your LLM generates stores key and value vectors in FP16. The model weights might fit on your GPU, but at long context the KV cache eats the rest of your VRAM. A Qwen3.5-27B at 262K context needs 16 GB just for KV cache. With Q4 weights already taking ~15 GB, that's 31 GB on a 24 GB card. You're out of memory before the interesting context lengths even start.

### What TurboQuant does

It compresses those FP16 vectors to 3 bits per dimension. 5x smaller. The key insight: it does not try to reconstruct individual vectors accurately (reconstruction error is 23-44%, which sounds bad). Instead, it guarantees that attention *scores* -- the inner products between queries and keys -- are mathematically unbiased. Attention only cares about relative scores, not per-vector fidelity. Two stages: a rotated Lloyd-Max quantizer for MSE-optimal compression, then a 1-bit QJL correction that kills the bias in inner product estimates. The math is clean and the paper is worth reading.

### My results (measured on RTX 4090)

All numbers below are from real LLM KV caches, not synthetic random vectors.

**Attention quality (Qwen2.5-3B-Instruct, showcase benchmark):**

| Bits | Cosine Similarity | Top-5 Attention Match | Compression |
|---|---|---|---|
| 2-bit | 0.9913 | 93.1% | 7.3x |
| **3-bit** | **0.9969** | **94.4%** | **5.0x** |
| 4-bit | 0.9990 | 94.4% | 3.8x |

Paper targets: cosine sim > 0.995, top-5 > 90%. Both met at 3-bit. All 72 attention heads score >= 0.99 individually. Worst head: 0.9902.

**Cross-model validation:**

| Model | Params | Head dim | Cosine Sim | Top-5 | Compression |
|---|---|---|---|---|---|
| Qwen2.5-3B | 3B | 128 | 0.9969 | 94.4% | 5.0x |
| Qwen2.5-14B | 14B | 128 | 0.9964 | 95.3% | 5.0x |
| Qwen3.5-27B | 27B | 256 | 0.9932 | 100% | 5.2x |

The 27B model uses d=256 (hybrid DeltaNet+Attention architecture) -- a head dimension the paper never tested. It works. 100% top-5 attention match across all attention heads.

**GPU throughput:**

| Operation | Vectors/sec |
|---|---|
| Quantize (3-bit, d=128) | 34M |
| Inner product estimate | 68M |

Note: this is pure PyTorch, no fused CUDA/Triton kernels yet. These numbers will improve.

**Memory projections** (calculated, not measured -- marking these clearly):

| Setup | FP16 KV Cache | TurboQuant 3-bit | Fits 24GB? |
|---|---|---|---|
| Qwen3.5-27B Q4 (~15GB weights) + 262K context | 16.0 GB KV = 31 GB total | 3.1 GB KV = 18 GB total | FP16: No. TQ-3: Yes, 6 GB spare |

### What makes this different

There are other TurboQuant implementations out there -- TheTom's turboquant_plus, 0xSero's work, the llama.cpp discussions. Credit to all of them for pushing on this. Here is where this one differs:

- **331 tests** in 13 seconds. Most implementations have fewer than 20. Every mathematical bound from the paper is validated in code.
- **Validated on 3 real models** including Qwen3.5-27B at d=256, a dimension the paper never tested.
- **5 features beyond the paper** (details below).
- **Packaged as a Python library**: `pip install turboquantdc`, proper setup.py, 10,000+ lines total.
- **Paper bounds verified**: MSE distortion, inner product distortion, unbiasedness, compression ratio, Lloyd-Max centroids matching to 5 decimal places.

### Beyond-paper features

These are extensions not in the original Google paper, inspired partly by community work (TheTom/turboquant_plus) and partly by what seemed useful:

1. **Sparse V dequantization** -- At long context, 90%+ of softmax attention weights are negligible. Skip decoding those value vectors entirely. 22% faster decode, zero quality loss (PPL delta = 0.0000).

2. **Fractional bit rates** -- Support for non-integer bit-widths like 2.5-bit (5.56x compression) and 3.5-bit (4.13x). Splits channels after rotation into two groups quantized at floor/ceil of target bits. The rotation preserves optimality for both groups.

3. **Layer-adaptive compression** -- Not all layers are equally sensitive. Keep the last N layers at higher precision, compress the rest more aggressively. Three strategies: tail_preserve, gradient, custom per-layer assignment.

4. **Walsh-Hadamard Transform** -- O(d log d) butterfly-based rotation as an alternative to QR decomposition. 256x less memory than storing a full d x d rotation matrix. Same mathematical guarantees.

5. **Temporal decay** -- Three-tier hot/warm/cold cache. Recent tokens at 4-bit, older tokens at 3-bit, oldest at 2-bit. At 32K context with default windows, 85% of tokens sit in the cold tier. ~30% additional memory savings on top of base compression.

### Try it

```bash
pip install turboquantdc
```

```python
import torch
from turboquantdc import TurboQuantEstimator

estimator = TurboQuantEstimator(d=128, bits=3, device="cuda")
keys = torch.randn(4096, 128, device="cuda")
compressed = estimator.quantize(keys)

query = torch.randn(1, 128, device="cuda")
scores = estimator.inner_product(query, compressed)  # (1, 4096), unbiased
```

[Colab notebook](link) (interactive demo with Qwen2.5-3B)

You can also run the standalone demo: `python demo.py --prompt "Explain quantum computing" --max-tokens 100 --bits 3`

### What is next

- HuggingFace transformers integration (PR in progress)
- Triton kernels for actual fused quantize/dequantize ops -- the current throughput numbers are pure PyTorch and leave performance on the table
- SGLang / vLLM backend integration (vLLM module exists but needs real-world testing)
- Combining temporal decay + sparse V + eviction for a push toward 10x effective compression

### Limitations (read before commenting)

- No fused GPU kernels. Pure PyTorch. The throughput numbers are good but not production-grade yet.
- No end-to-end perplexity benchmarks on full generation tasks. The attention score metrics are solid but someone should run a proper lm-eval sweep.
- The memory projections for 27B models are calculated, not measured end-to-end with a running model.
- Only tested on Qwen-family models so far. Llama, Mistral, etc. should work (the algorithm is architecture-agnostic) but I have not validated them.

---

331 tests, 10,000+ lines, MIT license. The full test suite runs in 13 seconds.

One more thing: this was built by a team of 15 specialized AI agents coordinated through a war room dashboard. The full transcript (92 messages) is in the repo at `docs/WARROOM_TRANSCRIPT.md`. Yes, really. If you want to see what coordinated multi-agent development looks like in practice, that file is the raw log.

Paper: [arxiv 2504.19874](https://arxiv.org/abs/2504.19874)

Repo: [github.com/dhawal/turboQuantDC](https://github.com/dhawal/turboQuantDC)
