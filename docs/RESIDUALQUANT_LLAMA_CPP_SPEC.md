# ResidualQuant llama.cpp Integration Specification

## Overview

ResidualQuant is a drop-in replacement for the QJL (Stage 2) correction in TurboQuant's
KV cache compression. Instead of projecting residuals through a random Gaussian matrix
and storing only signs (QJL), ResidualQuant stores the sign of each residual coordinate
*directly in the rotated space* along with a mean absolute deviation scale factor.

**Trade-off:** Biased estimator with lower variance. For autoregressive generation
where the same compressed keys are reused across many decode steps, the variance
reduction outweighs the small bias. Validated by Tom against turbo3 in llama.cpp.

**Storage:** Identical to TurboQuant at the same bit-width.

## Binary Block Format

### Proposed GGML type: `GGML_TYPE_RQ3_0`

3-bit ResidualQuant: 2-bit PolarQuant MSE indices + 1-bit residual signs + metadata.

Block size: 128 elements (one rotation group = one head dimension).

```
struct block_rq3_0 {
    ggml_half  norm;              //  2 bytes: original vector L2 norm (FP16)
    ggml_half  rscale;            //  2 bytes: residual scale = mean(|r_rot|) (FP16)
    uint8_t    qs[128 / 4];      //  32 bytes: 2-bit MSE indices (4 per byte)
    uint8_t    signs[128 / 8];   //  16 bytes: residual sign bits (8 per byte)
};                                // = 52 bytes per 128 elements
                                  // = 3.25 bits/value
                                  // = 4.9x compression vs FP16
```

**Comparison with existing turbo types:**

| Type | Block Size | Bytes/Block | Bits/Value | Compression | Stage 2 |
|------|-----------|-------------|------------|-------------|---------|
| turbo2_0 | 128 | 34 | 2.125 | 7.5x | None (MSE only) |
| turbo3_0 | 128 | 50 | 3.125 | 5.1x | QJL (bit 3 encodes high bit) |
| **rq3_0** | **128** | **52** | **3.25** | **4.9x** | **ResidualQuant** |
| turbo4_0 | 128 | 68 | 4.25 | 3.8x | 4-bit PolarQuant |

Note on turbo3_0 vs rq3_0: turbo3_0 packs a 3-bit codebook index split as 2 bits
in qs[] + 1 bit in signs[] (the high bit of the index). The "signs" field there
is NOT a residual sign; it is the third bit of the codebook index (selecting among
8 centroids). In RQ3, the qs[] 2-bit index and signs[] residual bit are *independent*
fields with different semantics. RQ3 adds 2 bytes per block for the explicit residual
scale (rscale), making it 52 vs 50 bytes.

### Proposed GGML type: `GGML_TYPE_RQ4_0`

4-bit ResidualQuant: 3-bit PolarQuant MSE indices + 1-bit residual signs + metadata.

```
struct block_rq4_0 {
    ggml_half  norm;              //  2 bytes: original vector L2 norm (FP16)
    ggml_half  rscale;            //  2 bytes: residual scale (FP16)
    uint8_t    qs[128 * 3 / 8];  //  48 bytes: 3-bit MSE indices (packed bitstream)
    uint8_t    signs[128 / 8];   //  16 bytes: residual sign bits
};                                // = 68 bytes per 128 elements
                                  // = 4.25 bits/value
                                  // = 3.8x compression vs FP16
```

Same total bytes as turbo4_0 but with different Stage 2 semantics.

### Block Layout (Byte-level Detail)

For block_rq3_0 (d=128, mse_bits=2):

```
Offset  Size   Field          Encoding
------  ----   -----          --------
0       2      norm           IEEE 754 FP16, little-endian
2       2      rscale         IEEE 754 FP16, little-endian
4       32     qs[32]         2-bit indices packed 4 per byte, LSB first
                              qs[i] bits [(j%4)*2 +: 2] = index for element i*4+j
36      16     signs[16]      1 bit per element, LSB first
                              signs[i] bit (j%8) = sign for element i*8+j
                              1 = positive (+1), 0 = negative (-1)
------
52 bytes total
```

## Dequantize Algorithm

For one block (d=128 elements):

```c
void dequantize_row_rq3_0(const block_rq3_0 *x, float *y, int64_t k) {
    const int d = 128;
    const int nb = k / d;

    for (int block = 0; block < nb; block++) {
        float norm   = GGML_FP16_TO_FP32(x[block].norm);
        float rscale = GGML_FP16_TO_FP32(x[block].rscale);

        // Step 1: Unpack indices + signs -> corrected rotated-space vector
        float corrected_rot[128];
        for (int j = 0; j < d; j++) {
            uint8_t idx  = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            float sign   = (x[block].signs[j/8] & (1 << (j%8))) ? 1.0f : -1.0f;
            corrected_rot[j] = CENTROIDS_2BIT[idx] + rscale * sign;
        }

        // Step 2: Inverse rotation (WHT or dense)
        float unrotated[128];
        inverse_rotate(corrected_rot, unrotated, d);

        // Step 3: Rescale
        for (int j = 0; j < d; j++) {
            y[block * d + j] = unrotated[j] * norm;
        }
    }
}
```

**Key difference from turbo3_0 dequant:** turbo3_0 just does `centroids[idx] * norm`.
RQ3 adds `rscale * sign` correction *before* inverse rotation, then applies norm.

## Rotation Strategy

Two rotation backends, selectable at runtime:

### WHT (Default, O(d log d))

The Walsh-Hadamard Transform with random sign flips. Matches the existing
turbo3_0/turbo2_0 CPU and CUDA paths.

- Forward: `y = WHT(D1 * x) * D2 / sqrt(d)` where D1, D2 are diagonal sign matrices
- Inverse: `x = D1 * WHT(y * D2) / sqrt(d)` (WHT is self-inverse up to scaling)
- Storage: 2 * d sign values (256 bytes for d=128), shared across all vectors

The sign arrays are generated deterministically from a seed and compiled into
header files (see `turbo-rotation-data.h` in Tom's repo).

### Dense QR (Fallback)

Full d x d orthogonal matrix from QR decomposition of random Gaussian.
Required when d is not a power of 2.

- Forward: `y = x @ Pi^T`
- Inverse: `x = y @ Pi`
- Storage: d*d floats (64KB for d=128), shared across all vectors

## Codebook

Lloyd-Max optimal centroids for N(0, 1/d) at the given bit-width.

**2-bit codebook (for RQ3, d=128):**
```c
static const float CENTROIDS_2BIT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};
```

**3-bit codebook (for RQ4, d=128):**
```c
static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
```

These are the same centroids used by turbo3_0/turbo4_0. The codebook is a
function of (d, mse_bits) only and can be hard-coded for d=128.

## Integration Points in llama.cpp

### 1. ggml.h — Type enum

```c
// In the GGML_TYPE enum, after TURBO2_0:
GGML_TYPE_RQ3_0 = 47,   // ResidualQuant 3-bit: 2-bit PolarQuant + 1-bit residual signs
GGML_TYPE_RQ4_0 = 48,   // ResidualQuant 4-bit: 3-bit PolarQuant + 1-bit residual signs
GGML_TYPE_COUNT = 49,
```

### 2. ggml-common.h — Block definition

```c
#define QK_RQ3 128
typedef struct {
    ggml_half  norm;
    ggml_half  rscale;
    uint8_t    qs[QK_RQ3 / 4];       // 32 bytes: 2-bit MSE indices
    uint8_t    signs[QK_RQ3 / 8];    // 16 bytes: residual sign bits
} block_rq3_0;                        // 52 bytes total
static_assert(sizeof(block_rq3_0) == 52, "wrong rq3_0 block size");
```

### 3. ggml.c — Type traits

```c
[GGML_TYPE_RQ3_0] = {
    .type_name     = "rq3",
    .blck_size     = QK_RQ3,
    .type_size     = sizeof(block_rq3_0),
    .is_quantized  = true,
    .to_float      = (ggml_to_float_t) dequantize_row_rq3_0,
    .from_float_ref = (ggml_from_float_t) quantize_row_rq3_0_ref,
},
```

### 4. ggml-turbo-quant.c (or new ggml-rq-quant.c) — CPU implementation

The reference C implementation from `tools/residualquant_reference.c` provides
the algorithm. For llama.cpp integration:

- `quantize_row_rq3_0_ref()` — Quantize path (CPU SET_ROWS)
- `dequantize_row_rq3_0()` — Dequantize path (CPU to_float)
- `quantize_rq3_0()` — Batch quantize wrapper

### 5. llama-kv-cache.cpp — Cache type selection

```c
// In the cache-type-k argument handling:
case "rq3":
    cache_type_k = GGML_TYPE_RQ3_0;
    break;
```

### 6. CUDA kernel (ggml-cuda/)

The CUDA dequantize kernel follows the same pattern as turbo3_0:
- Unpack indices from shared memory
- Lookup centroids (compile-time constant)
- Add `rscale * sign` correction
- Apply WHT inverse rotation using butterfly operations
- Scale by norm

The flash-attention integration would follow the same `fattn-vec-instance-*`
template pattern, with `dequantize_1_rq3_0()` providing the per-element decode.

### 7. Metal kernel (ggml-metal/)

Similar pattern to the CUDA kernel, using Metal shader language.
The WHT butterfly and centroid lookup translate directly.

## Performance Expectations

### Memory

For d=128, 3-bit ResidualQuant:
- Per vector: 52 bytes (vs 256 bytes for FP16) = 4.9x compression
- For 8K context, 32 layers, 32 heads: ~430 MB (vs ~2.1 GB) for keys only

### Compute

Dequantize cost per vector (d=128):
- Index unpack + centroid lookup: 128 operations
- Sign unpack + correction: 128 FMA operations
- WHT inverse rotation: 128 * 7 = 896 butterfly operations
- Norm scaling: 128 multiplications

Total: ~1280 FLOPs per vector. This is identical to turbo3_0 dequantize
since the correction step replaces the QJL inner-product formula rather
than adding new work.

### Quality

Based on TurboQuantDC benchmarks (d=128, 10K random vectors):

| Metric | turbo3 (QJL) | RQ3 (ResidualQuant) | RQ4 |
|--------|-------------|---------------------|-----|
| Cosine similarity | 0.9945 | 0.9960 | 0.9985 |
| Top-5 attention match | 88% | 92% | 97% |
| Inner product bias | Unbiased | ~2% bias | ~1% bias |
| Inner product std | 0.15 | 0.08 | 0.04 |

The quality improvement comes from ResidualQuant preserving the direction of the
residual (just quantizing its magnitude to a single scale), while QJL's random
projection destroys direction information.

## File Format for Offline Export

For development and validation, the `export_residualquant_gguf.py` tool writes
a standalone binary file with format version "RQ01". See that file for the
complete header specification. This is NOT a GGUF file — it is a development
artifact for validating the C implementation before integrating into ggml.

For production llama.cpp integration, the block data would be stored inline in
GGUF tensor data, exactly as turbo3_0/turbo4_0 blocks are today.

## Migration Path

1. **Phase 1 (this PR):** Reference C implementation + validation against Python
2. **Phase 2:** Register GGML_TYPE_RQ3_0 in ggml.h/ggml.c with CPU-only dequantize
3. **Phase 3:** CPU quantize (SET_ROWS handler) + basic attention
4. **Phase 4:** CUDA kernel (dequantize + flash-attention template instance)
5. **Phase 5:** Metal kernel (Apple Silicon support)
6. **Phase 6:** Performance benchmarks and quality validation with real models
