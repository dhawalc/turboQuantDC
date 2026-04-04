/*
 * TurboQuantDC — CUDA Walsh-Hadamard Transform kernel with random sign flipping.
 *
 * Extends the Triton WHT kernel from d<=512 to d<=2048:
 *   - d=64..2048: shared memory butterfly, threads cooperate
 *   - One block per vector, NTHREADS threads per block
 *   - Shared memory: D floats = D*4 bytes (2048*4 = 8KB, well within 48KB)
 *
 * Forward:  out = WHT(signs * x) / sqrt(d)
 * Inverse:  out = signs * WHT(x) / sqrt(d)
 *
 * The butterfly decomposition: for each stage h = 1, 2, 4, ..., d/2:
 *   For each group g in [0, d/(2*h)):
 *     For each element e in [0, h):
 *       a = x[g*2*h + e]       (lo half)
 *       b = x[g*2*h + h + e]   (hi half)
 *       x[g*2*h + e]     = a + b
 *       x[g*2*h + h + e] = a - b
 *
 * Build: JIT compiled via torch.utils.cpp_extension.load()
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>


// ---------------------------------------------------------------------------
// Shared memory butterfly kernel: threads cooperate on one vector per block.
// Template D = dimension, NTHREADS = threads per block.
// Uses D*sizeof(float) bytes of shared memory.
// ---------------------------------------------------------------------------

template <int D, int NTHREADS>
__global__ void wht_shared_kernel(
    const float* __restrict__ x_in,    // (batch, D)
    float*       __restrict__ x_out,   // (batch, D)
    const float* __restrict__ signs,   // (D,)
    int batch_size,
    int inverse
) {
    const int vid = blockIdx.x;
    if (vid >= batch_size) return;

    const int tid = threadIdx.x;

    __shared__ float s_v[D];

    // Load vector into shared memory (coalesced)
    const float* row_in = x_in + vid * D;
    #pragma unroll 4
    for (int i = tid; i < D; i += NTHREADS) {
        s_v[i] = row_in[i];
    }
    __syncthreads();

    // Forward: apply signs before WHT
    if (inverse == 0) {
        #pragma unroll 4
        for (int i = tid; i < D; i += NTHREADS) {
            s_v[i] *= signs[i];
        }
        __syncthreads();
    }

    // Butterfly stages: h = 1, 2, 4, ..., D/2
    // Each stage has D/2 independent butterfly pairs.
    // Threads divide these pairs among themselves.
    //
    // For pair index p in [0, D/2):
    //   group   = p / h
    //   element = p % h
    //   lo_idx  = group * 2 * h + element
    //   hi_idx  = lo_idx + h
    //
    // The integer divide/mod is optimized by the compiler when h is a
    // loop variable (power-of-2 strength reduction).

    constexpr int N_PAIRS = D / 2;

    #pragma unroll 1
    for (int h = 1; h < D; h <<= 1) {
        for (int p = tid; p < N_PAIRS; p += NTHREADS) {
            int group = p / h;
            int element = p - group * h;  // p % h via subtraction (faster)
            int lo = group * 2 * h + element;
            int hi = lo + h;
            float a = s_v[lo];
            float b = s_v[hi];
            s_v[lo] = a + b;
            s_v[hi] = a - b;
        }
        __syncthreads();
    }

    // Normalize and apply inverse signs if needed
    const float inv_sqrt_d = 1.0f / sqrtf((float)D);
    float* row_out = x_out + vid * D;

    if (inverse == 1) {
        #pragma unroll 4
        for (int i = tid; i < D; i += NTHREADS) {
            row_out[i] = s_v[i] * inv_sqrt_d * signs[i];
        }
    } else {
        #pragma unroll 4
        for (int i = tid; i < D; i += NTHREADS) {
            row_out[i] = s_v[i] * inv_sqrt_d;
        }
    }
}


// ---------------------------------------------------------------------------
// C++ dispatch function
// ---------------------------------------------------------------------------

torch::Tensor cuda_wht(
    torch::Tensor x,       // (batch, d) float32
    torch::Tensor signs,   // (d,) float32
    bool inverse
) {
    TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
    TORCH_CHECK(signs.is_cuda(), "signs must be on CUDA");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (batch, d)");

    const int batch = x.size(0);
    const int d = x.size(1);
    const int inv = inverse ? 1 : 0;

    TORCH_CHECK(signs.size(0) == d, "signs must have size d");
    TORCH_CHECK((d & (d - 1)) == 0 && d > 0, "d must be a power of 2");
    TORCH_CHECK(d >= 64 && d <= 2048, "d must be 64..2048");

    auto output = torch::empty_like(x);

    // One block per vector. Thread count chosen for occupancy:
    // - d=64..128:  128 threads (64 pairs / 128 threads = good utilization)
    // - d=256..512: 128 threads (128-256 pairs / 128 threads = full utilization)
    // - d=1024:     256 threads (512 pairs / 256 threads = 2 pairs per thread)
    // - d=2048:     256 threads (1024 pairs / 256 threads = 4 pairs per thread)

    switch (d) {
        case 64:
            wht_shared_kernel<64, 32><<<batch, 32>>>(
                x.data_ptr<float>(), output.data_ptr<float>(),
                signs.data_ptr<float>(), batch, inv);
            break;
        case 128:
            wht_shared_kernel<128, 64><<<batch, 64>>>(
                x.data_ptr<float>(), output.data_ptr<float>(),
                signs.data_ptr<float>(), batch, inv);
            break;
        case 256:
            wht_shared_kernel<256, 128><<<batch, 128>>>(
                x.data_ptr<float>(), output.data_ptr<float>(),
                signs.data_ptr<float>(), batch, inv);
            break;
        case 512:
            wht_shared_kernel<512, 256><<<batch, 256>>>(
                x.data_ptr<float>(), output.data_ptr<float>(),
                signs.data_ptr<float>(), batch, inv);
            break;
        case 1024:
            wht_shared_kernel<1024, 256><<<batch, 256>>>(
                x.data_ptr<float>(), output.data_ptr<float>(),
                signs.data_ptr<float>(), batch, inv);
            break;
        case 2048:
            wht_shared_kernel<2048, 256><<<batch, 256>>>(
                x.data_ptr<float>(), output.data_ptr<float>(),
                signs.data_ptr<float>(), batch, inv);
            break;
        default:
            TORCH_CHECK(false, "Unsupported d: ", d, ". Use 64, 128, 256, 512, 1024, or 2048.");
    }

    return output;
}


// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wht", &cuda_wht,
          "CUDA Walsh-Hadamard Transform with random sign flipping",
          py::arg("x"), py::arg("signs"), py::arg("inverse") = false);
}
