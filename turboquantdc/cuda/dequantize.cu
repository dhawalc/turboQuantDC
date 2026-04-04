/*
 * TurboQuantDC — Raw CUDA dequantize kernels for SM 89 (RTX 4090).
 *
 * Two kernels:
 *   1. dequantize_mse:      centroid lookup -> inverse rotation -> rescale
 *   2. dequantize_residual: centroid lookup -> residual correction -> inverse rotation -> rescale
 *
 * Optimization strategy:
 *   - Codebook (<=16 centroids for 4-bit) loaded into shared memory (fits in 64 bytes)
 *   - Rotation matrix loaded into shared memory (d*d*4 bytes; 128*128*4 = 64KB -> use
 *     column tiles for d=256 to fit in 48KB shared memory)
 *   - One threadblock per vector, 128 threads per block
 *   - Coalesced global loads for indices (int32) and stores for output (float32)
 *   - Template on D for compile-time unrolling (d=128 and d=256)
 *
 * Build: JIT compiled via torch.utils.cpp_extension.load()
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Threads per block — one warp per 4 elements of d=128, or 2 elements of d=256
constexpr int THREADS_PER_BLOCK = 128;

// Maximum supported codebook size (4-bit = 16 centroids)
constexpr int MAX_CENTROIDS = 16;

// ---------------------------------------------------------------------------
// Kernel 1: MSE-only dequantize
//   x_hat[j] = vec_norm * sum_k( centroids[indices[k]] * R[k][j] )
//
// Template D: head dimension (128 or 256), allows compile-time loop unrolling.
// Each block handles one vector. Threads cooperate on the matrix-vector multiply.
// ---------------------------------------------------------------------------

template <int D>
__global__ void dequantize_mse_kernel(
    const int32_t* __restrict__ indices,   // (batch, D)
    const float*   __restrict__ centroids, // (n_centroids,)
    const float*   __restrict__ R,         // (D, D) row-major
    const float*   __restrict__ vec_norms, // (batch,)
    float*         __restrict__ output,    // (batch, D)
    int n_centroids,
    int batch_size
) {
    const int vec_id = blockIdx.x;
    if (vec_id >= batch_size) return;

    const int tid = threadIdx.x;

    // --- Load codebook into shared memory (all threads cooperate) ---
    __shared__ float s_centroids[MAX_CENTROIDS];
    if (tid < n_centroids) {
        s_centroids[tid] = centroids[tid];
    }
    __syncthreads();  // ensure codebook is ready before centroid lookup

    // --- Load y_hat = centroids[indices] into shared memory ---
    // For D=128: 128 threads each load 1 element
    // For D=256: 128 threads each load 2 elements
    __shared__ float s_y_hat[D];
    const int32_t* row_indices = indices + vec_id * D;

    #pragma unroll
    for (int i = tid; i < D; i += THREADS_PER_BLOCK) {
        int idx = row_indices[i];
        s_y_hat[i] = s_centroids[idx];
    }
    __syncthreads();  // ensure y_hat is ready before matmul

    // --- Inverse rotation: x[j] = sum_k y_hat[k] * R[k][j] ---
    // Each thread computes one or more output elements.
    // R is row-major: R[k][j] at offset k*D + j.
    const float vn = vec_norms[vec_id];
    float* out_row = output + vec_id * D;

    #pragma unroll
    for (int j = tid; j < D; j += THREADS_PER_BLOCK) {
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < D; k++) {
            sum += s_y_hat[k] * R[k * D + j];
        }
        out_row[j] = sum * vn;
    }
}


// ---------------------------------------------------------------------------
// Kernel 2: Dequantize with residual correction
//   y_corrected[k] = centroids[indices[k]] + res_signs[k] * res_scale
//   x_hat[j] = vec_norm * sum_k( y_corrected[k] * R[k][j] )
// ---------------------------------------------------------------------------

template <int D>
__global__ void dequantize_residual_kernel(
    const int32_t* __restrict__ indices,    // (batch, D)
    const float*   __restrict__ centroids,  // (n_centroids,)
    const float*   __restrict__ R,          // (D, D) row-major
    const float*   __restrict__ vec_norms,  // (batch,)
    const float*   __restrict__ res_signs,  // (batch, D)
    const float*   __restrict__ res_scale,  // (batch,)
    float*         __restrict__ output,     // (batch, D)
    int n_centroids,
    int batch_size
) {
    const int vec_id = blockIdx.x;
    if (vec_id >= batch_size) return;

    const int tid = threadIdx.x;

    // --- Load codebook into shared memory ---
    __shared__ float s_centroids[MAX_CENTROIDS];
    if (tid < n_centroids) {
        s_centroids[tid] = centroids[tid];
    }

    // --- Load y_corrected = centroids[indices] + res_signs * res_scale ---
    __shared__ float s_y_hat[D];
    const int32_t* row_indices = indices + vec_id * D;
    const float* row_signs = res_signs + vec_id * D;
    const float scale = res_scale[vec_id];

    __syncthreads();  // ensure s_centroids is ready

    #pragma unroll
    for (int i = tid; i < D; i += THREADS_PER_BLOCK) {
        int idx = row_indices[i];
        s_y_hat[i] = s_centroids[idx] + row_signs[i] * scale;
    }
    __syncthreads();

    // --- Inverse rotation + rescale ---
    const float vn = vec_norms[vec_id];
    float* out_row = output + vec_id * D;

    #pragma unroll
    for (int j = tid; j < D; j += THREADS_PER_BLOCK) {
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < D; k++) {
            sum += s_y_hat[k] * R[k * D + j];
        }
        out_row[j] = sum * vn;
    }
}


// ---------------------------------------------------------------------------
// Fallback kernel for arbitrary D (not template-specialized)
// Uses dynamic shared memory for y_hat.
// ---------------------------------------------------------------------------

__global__ void dequantize_mse_generic_kernel(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ centroids,
    const float*   __restrict__ R,
    const float*   __restrict__ vec_norms,
    float*         __restrict__ output,
    int n_centroids,
    int D,
    int batch_size
) {
    const int vec_id = blockIdx.x;
    if (vec_id >= batch_size) return;

    const int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* s_centroids = shared;
    float* s_y_hat = shared + MAX_CENTROIDS;

    if (tid < n_centroids) {
        s_centroids[tid] = centroids[tid];
    }
    __syncthreads();

    const int32_t* row_indices = indices + vec_id * D;

    for (int i = tid; i < D; i += THREADS_PER_BLOCK) {
        int idx = row_indices[i];
        s_y_hat[i] = s_centroids[idx];
    }
    __syncthreads();

    const float vn = vec_norms[vec_id];
    float* out_row = output + vec_id * D;

    for (int j = tid; j < D; j += THREADS_PER_BLOCK) {
        float sum = 0.0f;
        for (int k = 0; k < D; k++) {
            sum += s_y_hat[k] * R[k * D + j];
        }
        out_row[j] = sum * vn;
    }
}


__global__ void dequantize_residual_generic_kernel(
    const int32_t* __restrict__ indices,
    const float*   __restrict__ centroids,
    const float*   __restrict__ R,
    const float*   __restrict__ vec_norms,
    const float*   __restrict__ res_signs,
    const float*   __restrict__ res_scale,
    float*         __restrict__ output,
    int n_centroids,
    int D,
    int batch_size
) {
    const int vec_id = blockIdx.x;
    if (vec_id >= batch_size) return;

    const int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* s_centroids = shared;
    float* s_y_hat = shared + MAX_CENTROIDS;

    if (tid < n_centroids) {
        s_centroids[tid] = centroids[tid];
    }

    const int32_t* row_indices = indices + vec_id * D;
    const float* row_signs = res_signs + vec_id * D;
    const float scale = res_scale[vec_id];

    __syncthreads();

    for (int i = tid; i < D; i += THREADS_PER_BLOCK) {
        int idx = row_indices[i];
        s_y_hat[i] = s_centroids[idx] + row_signs[i] * scale;
    }
    __syncthreads();

    const float vn = vec_norms[vec_id];
    float* out_row = output + vec_id * D;

    for (int j = tid; j < D; j += THREADS_PER_BLOCK) {
        float sum = 0.0f;
        for (int k = 0; k < D; k++) {
            sum += s_y_hat[k] * R[k * D + j];
        }
        out_row[j] = sum * vn;
    }
}


// ---------------------------------------------------------------------------
// C++ dispatch functions (called from Python via pybind11)
// ---------------------------------------------------------------------------

torch::Tensor cuda_dequantize_mse(
    torch::Tensor indices,    // (batch, d) int32
    torch::Tensor centroids,  // (n_centroids,) float32
    torch::Tensor R,          // (d, d) float32
    torch::Tensor vec_norms   // (batch,) float32
) {
    TORCH_CHECK(indices.is_cuda(), "indices must be on CUDA");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be on CUDA");
    TORCH_CHECK(R.is_cuda(), "R must be on CUDA");
    TORCH_CHECK(vec_norms.is_cuda(), "vec_norms must be on CUDA");

    const int batch = indices.size(0);
    const int d = indices.size(1);
    const int n_cent = centroids.size(0);

    TORCH_CHECK(R.size(0) == d && R.size(1) == d, "R must be (d, d)");
    TORCH_CHECK(n_cent <= MAX_CENTROIDS, "n_centroids must be <= ", MAX_CENTROIDS);

    auto output = torch::empty({batch, d}, torch::dtype(torch::kFloat32).device(indices.device()));

    const int grid = batch;
    const int block = THREADS_PER_BLOCK;

    if (d == 128) {
        dequantize_mse_kernel<128><<<grid, block>>>(
            indices.data_ptr<int32_t>(),
            centroids.data_ptr<float>(),
            R.data_ptr<float>(),
            vec_norms.data_ptr<float>(),
            output.data_ptr<float>(),
            n_cent, batch
        );
    } else if (d == 256) {
        dequantize_mse_kernel<256><<<grid, block>>>(
            indices.data_ptr<int32_t>(),
            centroids.data_ptr<float>(),
            R.data_ptr<float>(),
            vec_norms.data_ptr<float>(),
            output.data_ptr<float>(),
            n_cent, batch
        );
    } else {
        // Generic path with dynamic shared memory
        int smem = (MAX_CENTROIDS + d) * sizeof(float);
        dequantize_mse_generic_kernel<<<grid, block, smem>>>(
            indices.data_ptr<int32_t>(),
            centroids.data_ptr<float>(),
            R.data_ptr<float>(),
            vec_norms.data_ptr<float>(),
            output.data_ptr<float>(),
            n_cent, d, batch
        );
    }

    return output;
}


torch::Tensor cuda_dequantize_residual(
    torch::Tensor indices,    // (batch, d) int32
    torch::Tensor centroids,  // (n_centroids,) float32
    torch::Tensor R,          // (d, d) float32
    torch::Tensor vec_norms,  // (batch,) float32
    torch::Tensor res_signs,  // (batch, d) float32
    torch::Tensor res_scale   // (batch,) float32
) {
    TORCH_CHECK(indices.is_cuda(), "indices must be on CUDA");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be on CUDA");
    TORCH_CHECK(R.is_cuda(), "R must be on CUDA");
    TORCH_CHECK(vec_norms.is_cuda(), "vec_norms must be on CUDA");
    TORCH_CHECK(res_signs.is_cuda(), "res_signs must be on CUDA");
    TORCH_CHECK(res_scale.is_cuda(), "res_scale must be on CUDA");

    const int batch = indices.size(0);
    const int d = indices.size(1);
    const int n_cent = centroids.size(0);

    TORCH_CHECK(R.size(0) == d && R.size(1) == d, "R must be (d, d)");
    TORCH_CHECK(n_cent <= MAX_CENTROIDS, "n_centroids must be <= ", MAX_CENTROIDS);
    TORCH_CHECK(res_signs.size(0) == batch && res_signs.size(1) == d);
    TORCH_CHECK(res_scale.size(0) == batch);

    auto output = torch::empty({batch, d}, torch::dtype(torch::kFloat32).device(indices.device()));

    const int grid = batch;
    const int block = THREADS_PER_BLOCK;

    if (d == 128) {
        dequantize_residual_kernel<128><<<grid, block>>>(
            indices.data_ptr<int32_t>(),
            centroids.data_ptr<float>(),
            R.data_ptr<float>(),
            vec_norms.data_ptr<float>(),
            res_signs.data_ptr<float>(),
            res_scale.data_ptr<float>(),
            output.data_ptr<float>(),
            n_cent, batch
        );
    } else if (d == 256) {
        dequantize_residual_kernel<256><<<grid, block>>>(
            indices.data_ptr<int32_t>(),
            centroids.data_ptr<float>(),
            R.data_ptr<float>(),
            vec_norms.data_ptr<float>(),
            res_signs.data_ptr<float>(),
            res_scale.data_ptr<float>(),
            output.data_ptr<float>(),
            n_cent, batch
        );
    } else {
        int smem = (MAX_CENTROIDS + d) * sizeof(float);
        dequantize_residual_generic_kernel<<<grid, block, smem>>>(
            indices.data_ptr<int32_t>(),
            centroids.data_ptr<float>(),
            R.data_ptr<float>(),
            vec_norms.data_ptr<float>(),
            res_signs.data_ptr<float>(),
            res_scale.data_ptr<float>(),
            output.data_ptr<float>(),
            n_cent, d, batch
        );
    }

    return output;
}


// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequantize_mse", &cuda_dequantize_mse,
          "CUDA dequantize MSE-only (centroid lookup + inverse rotation + rescale)");
    m.def("dequantize_residual", &cuda_dequantize_residual,
          "CUDA dequantize with residual correction");
}
