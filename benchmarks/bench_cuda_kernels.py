"""Speed benchmark: CUDA vs Triton vs PyTorch dequantize and WHT.

Measures throughput in vectors/second for:
  - Dequantize MSE: centroid lookup + inverse rotation + rescale
  - Dequantize Residual: centroid lookup + correction + inverse rotation + rescale
  - WHT: forward Walsh-Hadamard Transform with sign flipping

Configurations:
  - n = 100, 1000, 10000 vectors
  - d = 128, 256 (and 1024 for WHT)
"""

import os
import sys
import time

import torch

# Ensure CUDA_HOME is set
if "CUDA_HOME" not in os.environ:
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"

# ---------------------------------------------------------------------------
# Load backends
# ---------------------------------------------------------------------------


def load_backends():
    """Load all available backends."""
    backends = {}

    # CUDA
    try:
        from turboquantdc.cuda.build import load_dequantize, load_wht
        deq_mod = load_dequantize()
        wht_mod = load_wht()
        if deq_mod:
            backends["cuda_deq"] = deq_mod
        if wht_mod:
            backends["cuda_wht"] = wht_mod
    except Exception as e:
        import traceback
        print(f"CUDA backend: {e}")
        traceback.print_exc()

    # Triton
    try:
        from turboquantdc.triton_kernels import (
            triton_dequantize,
            triton_dequantize_residual,
            triton_wht_rotate,
        )
        backends["triton_deq"] = triton_dequantize
        backends["triton_deq_res"] = triton_dequantize_residual
        backends["triton_wht"] = triton_wht_rotate
    except Exception as e:
        print(f"Triton backend: {e}")

    return backends


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def bench_fn(fn, warmup=20, iters=200):
    """Benchmark a CUDA function, return mean time in milliseconds."""
    torch.cuda.synchronize()

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms per call


def make_test_data(batch, d, n_centroids=8, seed=42):
    """Create test tensors for dequantize benchmarks."""
    torch.manual_seed(seed)
    centroids = torch.randn(n_centroids, device="cuda")
    R = torch.linalg.qr(torch.randn(d, d))[0].cuda().contiguous()
    indices = torch.randint(0, n_centroids, (batch, d), dtype=torch.int32, device="cuda")
    vec_norms = torch.randn(batch, device="cuda").abs() + 0.1
    res_signs = torch.where(torch.rand(batch, d, device="cuda") > 0.5, 1.0, -1.0)
    res_scale = torch.rand(batch, device="cuda") * 0.1
    return centroids, R, indices, vec_norms, res_signs, res_scale


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def main():
    print("=" * 80)
    print("TurboQuantDC Speed Benchmark: CUDA vs Triton vs PyTorch")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    backends = load_backends()
    print(f"Available backends: {list(backends.keys())}")
    print()

    batch_sizes = [100, 1000, 10000]
    dims = [128, 256]

    # -----------------------------------------------------------------------
    # Dequantize MSE Benchmark
    # -----------------------------------------------------------------------
    print("-" * 80)
    print("DEQUANTIZE MSE (centroid lookup + inverse rotation + rescale)")
    print("-" * 80)
    print(f"{'n':>8} {'d':>5} {'CUDA (ms)':>12} {'Triton (ms)':>12} {'PyTorch (ms)':>14} {'CUDA vps':>14} {'Triton vps':>14} {'Speedup':>10}")
    print("-" * 80)

    for d in dims:
        for n in batch_sizes:
            centroids, R, indices, vec_norms, _, _ = make_test_data(n, d)

            results = {}

            # CUDA
            if "cuda_deq" in backends:
                mod = backends["cuda_deq"]
                t = bench_fn(lambda: mod.dequantize_mse(indices, centroids, R, vec_norms))
                results["cuda"] = t

            # Triton
            if "triton_deq" in backends:
                fn = backends["triton_deq"]
                t = bench_fn(lambda: fn(indices, centroids, R, vec_norms))
                results["triton"] = t

            # PyTorch
            def pytorch_deq():
                y_hat = centroids[indices.long()]
                return (y_hat @ R) * vec_norms.unsqueeze(-1)
            t = bench_fn(pytorch_deq)
            results["pytorch"] = t

            cuda_ms = results.get("cuda", float("nan"))
            triton_ms = results.get("triton", float("nan"))
            pytorch_ms = results.get("pytorch", float("nan"))

            cuda_vps = n / (cuda_ms / 1000) if cuda_ms == cuda_ms else 0
            triton_vps = n / (triton_ms / 1000) if triton_ms == triton_ms else 0

            if cuda_ms == cuda_ms and triton_ms == triton_ms and triton_ms > 0:
                speedup = f"{triton_ms / cuda_ms:.2f}x"
            else:
                speedup = "N/A"

            print(f"{n:>8} {d:>5} {cuda_ms:>12.4f} {triton_ms:>12.4f} {pytorch_ms:>14.4f} {cuda_vps:>14,.0f} {triton_vps:>14,.0f} {speedup:>10}")

    # -----------------------------------------------------------------------
    # Dequantize Residual Benchmark
    # -----------------------------------------------------------------------
    print()
    print("-" * 80)
    print("DEQUANTIZE RESIDUAL (centroid + correction + inverse rotation + rescale)")
    print("-" * 80)
    print(f"{'n':>8} {'d':>5} {'CUDA (ms)':>12} {'Triton (ms)':>12} {'PyTorch (ms)':>14} {'Speedup':>10}")
    print("-" * 80)

    for d in dims:
        for n in batch_sizes:
            centroids, R, indices, vec_norms, res_signs, res_scale = make_test_data(n, d)

            results = {}

            # CUDA
            if "cuda_deq" in backends:
                mod = backends["cuda_deq"]
                t = bench_fn(lambda: mod.dequantize_residual(
                    indices, centroids, R, vec_norms, res_signs, res_scale))
                results["cuda"] = t

            # Triton
            if "triton_deq_res" in backends:
                fn = backends["triton_deq_res"]
                t = bench_fn(lambda: fn(indices, centroids, R, vec_norms, res_signs, res_scale))
                results["triton"] = t

            # PyTorch
            def pytorch_deq_res():
                y_hat = centroids[indices.long()]
                y_corr = y_hat + res_signs * res_scale.unsqueeze(-1)
                return (y_corr @ R) * vec_norms.unsqueeze(-1)
            t = bench_fn(pytorch_deq_res)
            results["pytorch"] = t

            cuda_ms = results.get("cuda", float("nan"))
            triton_ms = results.get("triton", float("nan"))
            pytorch_ms = results.get("pytorch", float("nan"))

            if cuda_ms == cuda_ms and triton_ms == triton_ms and triton_ms > 0:
                speedup = f"{triton_ms / cuda_ms:.2f}x"
            else:
                speedup = "N/A"

            print(f"{n:>8} {d:>5} {cuda_ms:>12.4f} {triton_ms:>12.4f} {pytorch_ms:>14.4f} {speedup:>10}")

    # -----------------------------------------------------------------------
    # WHT Benchmark
    # -----------------------------------------------------------------------
    print()
    print("-" * 80)
    print("WHT (Walsh-Hadamard Transform with random sign flipping)")
    print("-" * 80)
    wht_dims = [128, 256, 512, 1024, 2048]
    print(f"{'n':>8} {'d':>5} {'CUDA (ms)':>12} {'Triton (ms)':>12} {'PyTorch (ms)':>14} {'Speedup':>10}")
    print("-" * 80)

    from turboquantdc.rotation import generate_wht_rotation, apply_wht_rotation, fast_wht
    import math

    for d in wht_dims:
        params = generate_wht_rotation(d, seed=42, device="cuda")
        signs = params["signs"]

        for n in batch_sizes:
            x = torch.randn(n, d, device="cuda")

            results = {}

            # CUDA
            if "cuda_wht" in backends:
                mod = backends["cuda_wht"]
                x_c = x.contiguous()
                s_c = signs.contiguous()
                t = bench_fn(lambda: mod.wht(x_c, s_c, False))
                results["cuda"] = t

            # Triton (only d<=512)
            if "triton_wht" in backends and d <= 512:
                fn = backends["triton_wht"]
                t = bench_fn(lambda: fn(x, signs))
                results["triton"] = t

            # PyTorch
            def pytorch_wht():
                return apply_wht_rotation(x, params, inverse=False)
            t = bench_fn(pytorch_wht)
            results["pytorch"] = t

            cuda_ms = results.get("cuda", float("nan"))
            triton_ms = results.get("triton", float("nan"))
            pytorch_ms = results.get("pytorch", float("nan"))

            if cuda_ms == cuda_ms and triton_ms == triton_ms and triton_ms > 0:
                speedup = f"{triton_ms / cuda_ms:.2f}x"
            elif cuda_ms == cuda_ms and pytorch_ms == pytorch_ms and pytorch_ms > 0:
                speedup = f"{pytorch_ms / cuda_ms:.2f}x vs PT"
            else:
                speedup = "N/A"

            triton_str = f"{triton_ms:.4f}" if triton_ms == triton_ms else "N/A"
            print(f"{n:>8} {d:>5} {cuda_ms:>12.4f} {triton_str:>12} {pytorch_ms:>14.4f} {speedup:>10}")

    print()
    print("=" * 80)
    print("vps = vectors per second")
    print("Speedup = Triton / CUDA (or PyTorch / CUDA when Triton not available)")
    print("=" * 80)


if __name__ == "__main__":
    main()
