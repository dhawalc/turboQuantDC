/*
 * residualquant_reference.c — Reference C99 implementation of ResidualQuant dequantization
 *
 * This is the pure-C reference that would eventually go into llama.cpp's
 * ggml-quants.c as the CPU dequantize path for a new GGML_TYPE_RQ3 type.
 *
 * Algorithm (per vector of d elements):
 *   1. Unpack MSE indices from packed bits -> lookup centroids
 *   2. Unpack residual sign bits -> {-1, +1}
 *   3. In rotated space: corrected[j] = centroid[idx[j]] + scale * sign[j]
 *   4. Inverse rotation: result = corrected @ Pi  (Pi is orthogonal, so Pi^T = Pi^{-1})
 *   5. Rescale: result *= vec_norm
 *
 * Binary block layout (per vector, d elements):
 *   [0 .. mse_byte_count-1]        : packed MSE indices, mse_bits per element
 *   [mse_byte_count .. +sign_count] : packed residual sign bits, 1 bit per element
 *   [.. +2]                         : residual_scale as FP16 (little-endian)
 *   [.. +2]                         : vec_norm as FP16 (little-endian)
 *
 * Compile: gcc -std=c99 -O2 -shared -fPIC -o librqref.so residualquant_reference.c -lm
 * Or:      gcc -std=c99 -O2 -c residualquant_reference.c -lm
 */

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

/* ================================================================
 * FP16 <-> FP32 conversion (IEEE 754 half-precision)
 * Matches ggml's GGML_FP16_TO_FP32 / GGML_FP32_TO_FP16
 * ================================================================ */

static float fp16_to_fp32(uint16_t h) {
    /* IEEE 754 half-precision to single-precision */
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x03FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            /* +/- zero */
            union { uint32_t u; float f; } result;
            result.u = sign;
            return result.f;
        }
        /* Subnormal: normalize */
        while (!(mantissa & 0x0400)) {
            mantissa <<= 1;
            exponent--;
        }
        exponent++;
        mantissa &= ~0x0400;
    } else if (exponent == 31) {
        /* Inf or NaN */
        union { uint32_t u; float f; } result;
        result.u = sign | 0x7F800000 | ((uint32_t)mantissa << 13);
        return result.f;
    }

    exponent = exponent + (127 - 15);
    uint32_t bits = sign | (exponent << 23) | ((uint32_t)mantissa << 13);

    union { uint32_t u; float f; } result;
    result.u = bits;
    return result.f;
}


/* ================================================================
 * Bit unpacking helpers
 * ================================================================ */

/*
 * Unpack a single index of `bits_per_index` bits from a packed byte array.
 * Indices are packed LSB-first: index i starts at bit (i * bits_per_index).
 */
static int unpack_index(const uint8_t *data, int i, int bits_per_index) {
    int bit_offset = i * bits_per_index;
    int val = 0;

    for (int b = 0; b < bits_per_index; b++) {
        int byte_pos = (bit_offset + b) / 8;
        int bit_pos  = (bit_offset + b) % 8;
        if (data[byte_pos] & (1 << bit_pos)) {
            val |= (1 << b);
        }
    }
    return val;
}

/*
 * Unpack a single sign bit: 1 = positive (+1.0), 0 = negative (-1.0).
 */
static float unpack_sign(const uint8_t *data, int i) {
    int byte_pos = i / 8;
    int bit_pos  = i % 8;
    return (data[byte_pos] & (1 << bit_pos)) ? 1.0f : -1.0f;
}

/*
 * Read a little-endian FP16 value from a byte pointer.
 */
static float read_fp16_le(const uint8_t *ptr) {
    uint16_t h = (uint16_t)ptr[0] | ((uint16_t)ptr[1] << 8);
    return fp16_to_fp32(h);
}


/* ================================================================
 * Rotation matrix generation (must match Python exactly)
 * ================================================================ */

/*
 * LCG PRNG — same constants as Tom's ggml-turbo-quant.c
 * Produces deterministic sequence from a seed.
 */
typedef struct {
    uint64_t state;
} rq_prng_t;

static void rq_prng_seed(rq_prng_t *rng, uint64_t seed) {
    rng->state = seed;
}

static double rq_prng_normal(rq_prng_t *rng) {
    /* Box-Muller from LCG uniform */
    rng->state = rng->state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(rng->state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    rng->state = rng->state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(rng->state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

/*
 * Generate rotation matrix via QR decomposition (Modified Gram-Schmidt).
 * Result stored row-major in `out`, which must have d*d floats allocated.
 *
 * This matches the LCG-seeded rotation generation in ggml-turbo-quant.c.
 * NOTE: TurboQuantDC's Python code uses torch.randn with a different PRNG,
 * so the matrices will NOT match across Python and C for the same seed.
 * The validation script handles this by exporting the Python-generated
 * rotation matrix explicitly.
 */
static void rq_generate_rotation_qr(float *out, int d, uint32_t seed) {
    rq_prng_t rng;
    rq_prng_seed(&rng, (uint64_t)seed);

    /* Fill with Gaussian samples */
    for (int i = 0; i < d * d; i++) {
        out[i] = (float)rq_prng_normal(&rng);
    }

    /* Modified Gram-Schmidt (column-major storage in row-major array):
     * We treat out[i*d + j] as the (i,j) entry.
     * Orthogonalize columns. */
    for (int j = 0; j < d; j++) {
        /* Normalize column j */
        float norm = 0.0f;
        for (int i = 0; i < d; i++) {
            norm += out[i * d + j] * out[i * d + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int i = 0; i < d; i++) {
                out[i * d + j] /= norm;
            }
        }

        /* Orthogonalize subsequent columns against j */
        for (int k = j + 1; k < d; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += out[i * d + j] * out[i * d + k];
            }
            for (int i = 0; i < d; i++) {
                out[i * d + k] -= dot * out[i * d + j];
            }
        }
    }
}

/*
 * Generate WHT sign vector (for WHT-based rotation).
 * signs must have d floats allocated. Each is +1 or -1.
 */
static void rq_generate_wht_signs(float *signs, int d, uint32_t seed) {
    rq_prng_t rng;
    rq_prng_seed(&rng, (uint64_t)seed);

    for (int i = 0; i < d; i++) {
        /* Generate a uniform [0,1) value using LCG */
        rng.state = rng.state * 6364136223846793005ULL + 1442695040888963407ULL;
        double u = (double)(rng.state >> 11) / (double)(1ULL << 53);
        signs[i] = (u < 0.5) ? 1.0f : -1.0f;
    }
}

/*
 * In-place Walsh-Hadamard Transform (butterfly algorithm).
 * x must have d elements, d must be a power of 2.
 */
static void rq_fwht(float *x, int d) {
    for (int h = 1; h < d; h *= 2) {
        for (int i = 0; i < d; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

/*
 * Apply inverse WHT rotation: x = D @ WHT(y) / sqrt(d)
 * (WHT is self-inverse up to scaling, D is self-inverse)
 */
static void rq_inverse_wht_rotation(float *x, const float *signs, int d) {
    /* Inverse: WHT first, then multiply by signs */
    rq_fwht(x, d);
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) {
        x[i] *= inv_sqrt_d * signs[i];
    }
}


/* ================================================================
 * Dense matrix-vector multiply
 * ================================================================ */

/*
 * y = M^T @ x, where M is d x d row-major.
 * For orthogonal M, M^T @ x is the inverse rotation.
 * Note: we compute M^T @ x = sum_i M[i][j] * x[i] for each j.
 */
static void matvec_transpose(const float *M, const float *x, float *y, int d) {
    for (int j = 0; j < d; j++) {
        float sum = 0.0f;
        for (int i = 0; i < d; i++) {
            sum += M[i * d + j] * x[i];
        }
        y[j] = sum;
    }
}


/* ================================================================
 * Core dequantize function
 * ================================================================ */

/*
 * Dequantize one ResidualQuant vector from packed binary format.
 *
 * Parameters:
 *   dst         - Output buffer for d float32 elements
 *   src         - Packed binary data for one vector (see layout above)
 *   centroids   - Codebook centroid values, length n_centroids
 *   rotation    - Rotation matrix, d*d floats, row-major (NULL if using WHT)
 *   wht_signs   - WHT sign vector, d floats (NULL if using QR rotation)
 *   d           - Head dimension
 *   mse_bits    - Bits per MSE index
 *   use_wht     - 1 for WHT rotation, 0 for dense QR rotation
 */
void residualquant_dequantize(
    float       *dst,
    const void  *src,
    const float *centroids,
    const float *rotation,     /* d*d row-major, or NULL for WHT */
    const float *wht_signs,    /* d floats, or NULL for QR */
    int          d,
    int          mse_bits,
    int          use_wht
) {
    const uint8_t *ptr = (const uint8_t *)src;

    /* Compute byte offsets within the packed vector */
    int mse_byte_count  = (mse_bits * d + 7) / 8;
    int sign_byte_count = (d + 7) / 8;

    const uint8_t *mse_data  = ptr;
    const uint8_t *sign_data = ptr + mse_byte_count;
    const uint8_t *scale_ptr = sign_data + sign_byte_count;
    const uint8_t *norm_ptr  = scale_ptr + 2;

    /* Read FP16 scalars */
    float residual_scale = read_fp16_le(scale_ptr);
    float vec_norm       = read_fp16_le(norm_ptr);

    /* Step 1: Unpack MSE indices -> centroid lookup
     * Step 2: Unpack signs -> apply correction
     * Combined: corrected_rot[j] = centroids[idx_j] + scale * sign_j */
    float *corrected_rot = (float *)malloc(d * sizeof(float));
    assert(corrected_rot != NULL);

    for (int j = 0; j < d; j++) {
        int idx = unpack_index(mse_data, j, mse_bits);
        float centroid_val = centroids[idx];
        float sign_val = unpack_sign(sign_data, j);
        corrected_rot[j] = centroid_val + residual_scale * sign_val;
    }

    /* Step 3: Inverse rotation */
    if (use_wht) {
        assert(wht_signs != NULL);
        /* In-place inverse WHT rotation on corrected_rot */
        rq_inverse_wht_rotation(corrected_rot, wht_signs, d);
        /* Copy to dst */
        for (int j = 0; j < d; j++) {
            dst[j] = corrected_rot[j];
        }
    } else {
        assert(rotation != NULL);
        /* Dense: dst = rotation^T @ corrected_rot */
        matvec_transpose(rotation, corrected_rot, dst, d);
    }

    /* Step 4: Rescale by original vector norm */
    for (int j = 0; j < d; j++) {
        dst[j] *= vec_norm;
    }

    free(corrected_rot);
}

/*
 * Dequantize a row of n_elements ResidualQuant values.
 * Each vector (block) covers d elements.
 *
 * This matches the signature pattern of ggml's dequantize_row_* functions:
 *   dequantize_row_rq3(const block_rq3 *x, float *y, int64_t k)
 *
 * Parameters:
 *   dst         - Output buffer for n_elements float32 values
 *   src         - Packed binary data (consecutive blocks)
 *   centroids   - Codebook centroid values
 *   rotation    - Rotation matrix (NULL for WHT)
 *   wht_signs   - WHT sign vector (NULL for QR)
 *   n_elements  - Total number of float elements (must be divisible by d)
 *   d           - Head dimension (block size)
 *   mse_bits    - Bits per MSE index
 *   use_wht     - 1 for WHT rotation, 0 for QR
 */
void residualquant_dequantize_row(
    float       *dst,
    const void  *src,
    const float *centroids,
    const float *rotation,
    const float *wht_signs,
    int          n_elements,
    int          d,
    int          mse_bits,
    int          use_wht
) {
    assert(n_elements % d == 0);
    int n_blocks = n_elements / d;
    int block_bytes = (mse_bits * d + 7) / 8  /* MSE indices */
                    + (d + 7) / 8              /* sign bits */
                    + 2                        /* residual_scale fp16 */
                    + 2;                       /* vec_norm fp16 */

    const uint8_t *ptr = (const uint8_t *)src;

    for (int b = 0; b < n_blocks; b++) {
        residualquant_dequantize(
            dst + b * d,
            ptr + b * block_bytes,
            centroids,
            rotation,
            wht_signs,
            d,
            mse_bits,
            use_wht
        );
    }
}


/* ================================================================
 * Quantize (for testing: matches Python ResidualQuantEstimator)
 * ================================================================ */

/*
 * Find nearest centroid index for a scalar value.
 * Brute-force linear scan — fine for n_centroids <= 16.
 */
static int nearest_centroid(float val, const float *centroids, int n_centroids) {
    int best = 0;
    float best_dist = fabsf(val - centroids[0]);
    for (int i = 1; i < n_centroids; i++) {
        float dist = fabsf(val - centroids[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }
    return best;
}

/*
 * Pack a single index into a packed byte array at position i.
 */
static void pack_index(uint8_t *data, int i, int val, int bits_per_index) {
    int bit_offset = i * bits_per_index;
    for (int b = 0; b < bits_per_index; b++) {
        if (val & (1 << b)) {
            int byte_pos = (bit_offset + b) / 8;
            int bit_pos  = (bit_offset + b) % 8;
            data[byte_pos] |= (uint8_t)(1 << bit_pos);
        }
    }
}

/*
 * Pack a single sign bit at position i: positive -> 1, negative -> 0.
 */
static void pack_sign(uint8_t *data, int i, float val) {
    if (val >= 0.0f) {
        data[i / 8] |= (uint8_t)(1 << (i % 8));
    }
}

/*
 * Write a FP16 value (little-endian) at ptr.
 */
static void write_fp16_le(uint8_t *ptr, float val) {
    /* Simple FP32 -> FP16 conversion */
    union { float f; uint32_t u; } fu;
    fu.f = val;
    uint32_t f32 = fu.u;

    uint16_t sign = (uint16_t)((f32 >> 16) & 0x8000);
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = f32 & 0x007FFFFF;

    uint16_t h;
    if (exponent <= 0) {
        h = sign; /* flush to zero for simplicity */
    } else if (exponent >= 31) {
        h = sign | 0x7C00; /* infinity */
    } else {
        h = sign | (uint16_t)(exponent << 10) | (uint16_t)(mantissa >> 13);
    }

    ptr[0] = (uint8_t)(h & 0xFF);
    ptr[1] = (uint8_t)(h >> 8);
}

/*
 * Quantize one vector using ResidualQuant algorithm and pack into binary format.
 *
 * Parameters:
 *   dst         - Output packed binary data (must have block_bytes allocated)
 *   src         - Input vector, d float32 elements
 *   centroids   - Codebook centroid values
 *   n_centroids - Number of centroids (2^mse_bits)
 *   rotation    - Rotation matrix, d*d row-major (NULL for WHT)
 *   wht_signs   - WHT sign vector, d floats (NULL for QR)
 *   d           - Head dimension
 *   mse_bits    - Bits per MSE index
 *   use_wht     - 1 for WHT, 0 for QR
 */
void residualquant_quantize(
    void        *dst,
    const float *src,
    const float *centroids,
    int          n_centroids,
    const float *rotation,
    const float *wht_signs,
    int          d,
    int          mse_bits,
    int          use_wht
) {
    int mse_byte_count  = (mse_bits * d + 7) / 8;
    int sign_byte_count = (d + 7) / 8;
    int block_bytes     = mse_byte_count + sign_byte_count + 4;

    uint8_t *ptr = (uint8_t *)dst;
    memset(ptr, 0, block_bytes);

    uint8_t *mse_data  = ptr;
    uint8_t *sign_data = ptr + mse_byte_count;
    uint8_t *scale_ptr = sign_data + sign_byte_count;
    uint8_t *norm_ptr  = scale_ptr + 2;

    /* Step 1: Compute norm and normalize */
    float norm_sq = 0.0f;
    for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
    float vec_norm = sqrtf(norm_sq);

    float *normalized = (float *)malloc(d * sizeof(float));
    assert(normalized != NULL);

    if (vec_norm > 1e-8f) {
        float inv = 1.0f / vec_norm;
        for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
    } else {
        memset(normalized, 0, d * sizeof(float));
    }

    /* Step 2: Rotate */
    float *rotated = (float *)malloc(d * sizeof(float));
    assert(rotated != NULL);

    if (use_wht) {
        assert(wht_signs != NULL);
        /* Forward WHT: multiply by signs, then WHT, then normalize */
        for (int i = 0; i < d; i++) rotated[i] = normalized[i] * wht_signs[i];
        rq_fwht(rotated, d);
        float inv_sqrt_d = 1.0f / sqrtf((float)d);
        for (int i = 0; i < d; i++) rotated[i] *= inv_sqrt_d;
    } else {
        assert(rotation != NULL);
        /* Dense: rotated = x @ Pi^T, i.e., rotated[j] = sum_i x[i] * Pi[j][i]
         * With row-major Pi: rotated[j] = sum_i normalized[i] * rotation[j*d + i] */
        for (int j = 0; j < d; j++) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += normalized[i] * rotation[j * d + i];
            }
            rotated[j] = sum;
        }
    }

    /* Step 3: MSE quantize */
    int *indices = (int *)malloc(d * sizeof(int));
    assert(indices != NULL);

    for (int j = 0; j < d; j++) {
        indices[j] = nearest_centroid(rotated[j], centroids, n_centroids);
    }

    /* Step 4: Compute residual in rotated space, extract signs and scale */
    float scale_sum = 0.0f;
    for (int j = 0; j < d; j++) {
        float residual = rotated[j] - centroids[indices[j]];
        float sign = (residual >= 0.0f) ? 1.0f : -1.0f;

        pack_index(mse_data, j, indices[j], mse_bits);
        pack_sign(sign_data, j, sign);
        scale_sum += fabsf(residual);
    }
    float residual_scale = scale_sum / (float)d;

    /* Step 5: Write FP16 scalars */
    write_fp16_le(scale_ptr, residual_scale);
    write_fp16_le(norm_ptr, vec_norm);

    free(indices);
    free(rotated);
    free(normalized);
}
