# Add TurboQuant KV Cache Compression

## What does this PR do?

Adds `TurboQuantCache`, a new KV cache implementation based on Google's
[TurboQuant algorithm](https://arxiv.org/abs/2504.19874) (ICLR 2026).
TurboQuant compresses key-value pairs to 2-4 bit using a two-stage vector
quantization approach: PolarQuant (rotation + Lloyd-Max MSE quantization)
plus QJL (1-bit sign correction for unbiased inner products).

The cache plugs into `model.generate()` via the `past_key_values` parameter
with no model changes required.

**Important caveat (see Limitations):** While TurboQuant achieves excellent
*attention-level* metrics (cosine similarity >0.999 on individual score
vectors), the MSE-only key reconstruction used in the HF integration
introduces errors that compound during autoregressive generation. At 4-bit,
generation quality is acceptable for many tasks. At 3-bit and below,
generation quality degrades noticeably. The full unbiased inner product
estimator from the paper requires a custom attention kernel, which this
drop-in cache cannot provide.

## Motivation

The KV cache is the dominant memory bottleneck for long-context LLM
inference. At 32K context with a 7B model (28 layers, 28 KV heads,
d=128), the FP16 KV cache consumes ~3.5 GB of VRAM -- often more than
the model weights themselves under quantized inference. This limits
context length, batch size, and the models that can fit on consumer GPUs.

Existing approaches in transformers (the `QuantizedCache` using quanto/HQQ
backends) apply standard integer quantization per-channel. TurboQuant takes
a fundamentally different approach: it applies a random orthogonal rotation
before quantization, which makes coordinates nearly independent and enables
MSE-optimal Lloyd-Max scalar quantization. A second QJL (Quantized
Johnson-Lindenstrauss) stage stores 1-bit sign corrections that make inner
product estimates mathematically unbiased.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquantdc import TurboQuantCache

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)

# Create TurboQuant cache -- this is the only change needed
cache = TurboQuantCache(bits=4)  # 4-bit recommended for generation

output = model.generate(
    **inputs,
    max_new_tokens=100,
    past_key_values=cache,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# Check memory savings
savings = cache.memory_savings()
print(f"Compression ratio: {savings['overall_compression_ratio']:.1f}x")
```

Supported bit-widths: `bits=2` (7.3x, experimental), `bits=3` (5.0x, attention-quality),
`bits=4` (3.8x, recommended for generation).

## Benchmark Results

Benchmarked on Qwen2.5-3B-Instruct, RTX 4090, 100 new tokens generated.

### Attention Score Quality (Single-Step, Real KV Cache)

Measured on real Qwen2.5-3B KV cache vectors, comparing compressed attention
score vectors against FP16 ground truth at a single step:

| Bits | Attn Cosine Sim | Compression | Notes |
|------|-----------------|-------------|-------|
| 4 | 0.9997 | 3.8x | Near-lossless attention scores |
| **3** | **0.9993** | **5.0x** | Matches paper's >0.995 target |
| 2 | 0.9990 | 7.3x | Still high per-step quality |

These confirm the paper's theoretical bounds at the individual attention
computation level.

### Generation Quality (Autoregressive, End-to-End)

| Config | Compression | Tok/s | Output Quality |
|--------|-------------|-------|---------------|
| FP16 (baseline) | 1.0x | 46.5 | Reference |
| TQ-4 (4-bit) | 3.8x | 11.5 | Mostly coherent, some repetition artifacts |
| TQ-3 (3-bit) | 5.0x | 12.9 | Degraded -- garbled output on most prompts |
| TQ-2 (2-bit) | 7.3x | 18.3 | Severely degraded -- unusable |

**Why the gap?** Attention-level cosine similarity measures quality at a
single step. During autoregressive generation, small per-step errors compound
across 100+ decoding steps. The MSE-only key reconstruction introduces a
systematic bias that shifts attention distributions slightly at each step.
Over many steps, this causes the model to drift off the FP16 trajectory.

This is the fundamental limitation of the drop-in cache approach: TurboQuant's
full power comes from its *unbiased inner product estimator* (MSE + QJL
combined), but standard HF attention computes `Q @ K^T` on raw FP16 tensors
and cannot use the QJL correction term. A custom attention kernel is needed
to realize the paper's full quality promises for generation.

### Prior Validation (Attention-Level, Matches Paper)

These results from Phase 2/3 validation (not the HF integration) confirm
the algorithm works as described in the paper when using the full estimator:

| Model | Bits | Cosine Sim | Top-1 | Top-5 | Compression |
|-------|------|-----------|-------|-------|-------------|
| Qwen2.5-3B | 3 | 0.9959 | 80% | 91.7% | 5.0x |
| Qwen2.5-14B | 3 | 0.9964 | 78% | 95.3% | 5.0x |
| Qwen3.5-27B | 3 | 0.9932 | 98.4% | 100% | 5.2x |

## How it works

TurboQuant is a two-stage vector quantization algorithm:

**Stage 1 -- PolarQuant (MSE-optimal):**
1. Apply a random orthogonal rotation matrix (via QR decomposition) to each
   KV vector. After rotation, coordinates become nearly independent.
2. Quantize each coordinate independently using a precomputed Lloyd-Max
   codebook optimized for the theoretical post-rotation distribution.
3. Store `(b-1)` bits per coordinate as codebook indices.

**Stage 2 -- QJL (1-bit bias correction, keys only):**
1. Compute the residual between the original and MSE-reconstructed vector.
2. Project through a random Gaussian matrix and store only the signs (1 bit each).
3. Combined with the residual norm, this provides a mathematically unbiased
   correction to inner product estimates.

**Key insight:** TurboQuant does NOT need accurate individual vector
reconstruction. What matters for attention is accurate **inner products**
between query and key vectors. The QJL stage ensures these estimates are
unbiased with variance O(1/d). However, exploiting this requires a custom
attention kernel (see Limitations).

The `TurboQuantCache` implementation:
- Keys are compressed with the full two-stage estimator (MSE + QJL stored)
- Values are compressed with MSE-only PolarQuant (they need reconstruction, not inner products)
- On retrieval, compressed data is dequantized to FP16 using **MSE reconstruction only**
  (the QJL signs are stored but not used in dequantization, because standard HF
  attention expects dense FP16 tensors)

## Architecture

```
TurboQuantCache                    # Main cache class (Cache protocol)
  |-- TurboQuantLayer[]            # Per-layer compressed storage
        |-- TurboQuantEstimator    # Key compression (MSE + QJL)
        |      |-- PolarQuant      #   Stage 1: rotate + Lloyd-Max quantize
        |      |-- QJL             #   Stage 2: 1-bit sign correction (stored)
        |-- PolarQuant             # Value compression (MSE-only)
```

The cache implements the full HF Cache protocol:
- `update()` -- compress and store new KV pairs
- `get_seq_length()` -- return cached token count
- `reorder_cache()` -- beam search support
- `crop()` -- sequence truncation
- `batch_repeat_interleave()` / `batch_select_indices()` -- beam expansion
- `__iter__` / `__len__` -- standard iteration

## Limitations (Honest Assessment)

**CRITICAL: Generation quality at 3-bit and below is currently degraded.**
The benchmark shows garbled output at 3-bit and unusable output at 2-bit.
Only 4-bit produces acceptable (though imperfect) generation quality.

Root causes:

1. **Key reconstruction is MSE-only, not unbiased inner product.** The HF
   attention mechanism expects FP16 tensors and computes `Q @ K^T` directly.
   The QJL signs are stored but cannot be used in dequantization because
   the QJL correction requires access to the query at attention time, not at
   cache retrieval time. The MSE reconstruction at 3-bit uses only 2-bit
   codebook indices (since 1 bit is allocated to QJL), resulting in
   significant per-vector error that compounds during generation.

2. **Error compounding in autoregressive generation.** Even small attention
   distribution shifts at each step accumulate over 50-100+ generation
   steps. This is why per-step cosine similarity >0.999 does not translate
   to good generation quality. The paper's quality claims are for
   single-step attention score accuracy, not end-to-end generation.

3. **Speed overhead from compression/decompression.** Each `update()` call
   runs: rotation (matrix multiply), codebook lookup, QJL projection, and
   sign computation. Each retrieval runs: inverse rotation and centroid
   lookup. Generation throughput is 3-4x slower than FP16 baseline.
   The crossover point where memory savings offset speed cost is at long
   contexts (>2K-4K tokens).

4. **Not compatible with FlashAttention.** FlashAttention requires FP16/BF16
   inputs and does not support custom inner product estimators. The cache
   works with standard eager attention and SDPA.

5. **Codebook precomputation uses scipy.** The Lloyd-Max codebook solver
   depends on `scipy.integrate.quad` for numerical integration. This runs
   once per (dimension, bit-width) pair on first use and is cached, but adds
   scipy as a dependency.

6. **No fused attention kernel.** To realize TurboQuant's full potential,
   a custom attention kernel that computes:
   `<q, k> = <q, k_mse> + ||r|| * sqrt(pi/2)/m * <S@q, signs>`
   directly from compressed data is needed. This is implemented in
   `turboquantdc.estimator.TurboQuantEstimator.inner_product()` but cannot
   be plugged into HF's attention without modifying the model.

### Path Forward

The most promising approach for high-quality generation with TurboQuant
compression would be to:
1. Register a custom attention implementation (similar to how FlashAttention
   is registered) that operates directly on compressed KV cache
2. Use the full two-stage estimator for Q@K^T computation
3. Reconstruct values MSE-only (as currently done)

This would require changes to the HF attention dispatch mechanism, not just
the cache class.

## Tests

The test suite covers:
- **Basic cache operations:** update, seq_length, getitem, accumulation (7 tests)
- **Multi-layer support:** independent layers, iteration, seed diversity (3 tests)
- **Bit-width parameters:** all valid widths (2/3/4), invalid rejection (4 tests)
- **Beam search:** reorder_cache, duplicate beams (2 tests)
- **Crop:** truncation, no-op, negative, preservation, multi-update (5 tests)
- **Memory savings:** positive ratio, bit-width ordering, paper bounds (4 tests)
- **Quality:** key/value cosine similarity per bit-width, monotonic quality (7 tests)
- **HF protocol:** is_initialized, is_sliding, mask_sizes, reset, batch ops (8 tests)
- **Generate simulation:** mock generate loop, prefill+decode quality (2 tests)
- **Layer internals:** lazy_init, clear, memory scaling (3 tests)

Total: 47 tests in `tests/test_hf_integration.py`, all passing.

```bash
python -m pytest tests/test_hf_integration.py -v
# 47 passed in 3.02s
```

## Related Work

- Existing `QuantizedCache` in transformers uses quanto/HQQ per-channel integer quantization
- [KIVI](https://arxiv.org/abs/2402.02750) (asymmetric 2-bit) -- per-channel, no rotation
- [KVQuant](https://arxiv.org/abs/2401.18079) -- per-channel with outlier handling
- TurboQuant's rotation + Lloyd-Max approach achieves theoretically optimal MSE distortion
  because it exploits the mathematical properties of rotated coordinates

## Checklist

- [x] Implementation: `TurboQuantCache` with full Cache protocol
- [x] Unit tests: 47 tests covering all protocol methods and quality
- [x] Examples: Usage examples in `examples/hf_turboquant_example.py`
- [x] Benchmarks: `benchmarks/hf_benchmark.py` with generation and attention quality
- [x] Documentation: This PR description with honest limitations section
- [x] Honest quality assessment: Generation quality gap documented and explained
- [ ] CI: Tests pass in transformers CI (pending)
- [ ] Custom attention kernel: Would unlock full paper quality (future work)
- [ ] Review: HF team review (pending)
