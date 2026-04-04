#!/usr/bin/env python3
"""Validate C reference dequantize against Python ResidualQuantEstimator.

Strategy:
  1. Generate test vectors using Python ResidualQuantEstimator
  2. Export compressed data to the binary format
  3. Load the shared library (librqref.so) via ctypes
  4. Call C dequantize on the same packed binary data
  5. Compare Python vs C reconstruction — must match to FP32 precision

The rotation matrix is the tricky part: Python uses torch.randn (Mersenne Twister)
while the C code uses LCG PRNG. Since these produce different sequences, we
explicitly pass the Python-generated rotation matrix to the C function rather
than having C regenerate it from a seed.

Usage:
  cd tools/
  gcc -std=c99 -O2 -shared -fPIC -o librqref.so residualquant_reference.c -lm
  python validate_c_vs_python.py
"""

from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.export_residualquant_gguf import (
    get_binary_blob,
    pack_indices,
    pack_signs,
)
from turboquantdc.residual_quant import ResidualQuantEstimator


TOOLS_DIR = Path(__file__).resolve().parent
LIB_PATH = TOOLS_DIR / "librqref.so"
SRC_PATH = TOOLS_DIR / "residualquant_reference.c"


def compile_lib() -> Path:
    """Compile the C reference implementation into a shared library."""
    if LIB_PATH.exists():
        # Recompile if source is newer
        if SRC_PATH.stat().st_mtime > LIB_PATH.stat().st_mtime:
            print("Source newer than library, recompiling...")
        else:
            print(f"Using existing {LIB_PATH}")
            return LIB_PATH

    print(f"Compiling {SRC_PATH} -> {LIB_PATH}")
    result = subprocess.run(
        ["gcc", "-std=c99", "-O2", "-shared", "-fPIC",
         "-o", str(LIB_PATH), str(SRC_PATH), "-lm"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}")
        sys.exit(1)
    print("Compilation successful.")
    return LIB_PATH


def load_lib(lib_path: Path):
    """Load the shared library and set up function signatures."""
    lib = ctypes.CDLL(str(lib_path))

    # residualquant_dequantize(dst, src, centroids, rotation, wht_signs, d, mse_bits, use_wht)
    lib.residualquant_dequantize.restype = None
    lib.residualquant_dequantize.argtypes = [
        ctypes.POINTER(ctypes.c_float),    # dst
        ctypes.c_void_p,                   # src (packed binary)
        ctypes.POINTER(ctypes.c_float),    # centroids
        ctypes.POINTER(ctypes.c_float),    # rotation (or NULL)
        ctypes.POINTER(ctypes.c_float),    # wht_signs (or NULL)
        ctypes.c_int,                      # d
        ctypes.c_int,                      # mse_bits
        ctypes.c_int,                      # use_wht
    ]

    # residualquant_dequantize_row(dst, src, centroids, rotation, wht_signs, n_elements, d, mse_bits, use_wht)
    lib.residualquant_dequantize_row.restype = None
    lib.residualquant_dequantize_row.argtypes = [
        ctypes.POINTER(ctypes.c_float),    # dst
        ctypes.c_void_p,                   # src
        ctypes.POINTER(ctypes.c_float),    # centroids
        ctypes.POINTER(ctypes.c_float),    # rotation (or NULL)
        ctypes.POINTER(ctypes.c_float),    # wht_signs (or NULL)
        ctypes.c_int,                      # n_elements
        ctypes.c_int,                      # d
        ctypes.c_int,                      # mse_bits
        ctypes.c_int,                      # use_wht
    ]

    return lib


def numpy_to_c_float_ptr(arr: np.ndarray):
    """Convert a numpy float32 array to a ctypes float pointer."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _fp32_to_fp16_bytes(val: float) -> bytes:
    """Convert float32 to FP16 bytes (little-endian)."""
    return struct.pack("<e", val)


def build_binary_blob(
    mse_indices: np.ndarray,
    residual_signs: np.ndarray,
    residual_scale: float,
    vec_norm: float,
    mse_bits: int,
) -> bytes:
    """Build the packed binary blob for one vector, matching export format."""
    parts = []
    parts.append(pack_indices(mse_indices.astype(np.int32), mse_bits))
    parts.append(pack_signs(residual_signs.astype(np.float32)))
    parts.append(_fp32_to_fp16_bytes(residual_scale))
    parts.append(_fp32_to_fp16_bytes(vec_norm))
    return b"".join(parts)


def run_python_dequantize(
    estimator: ResidualQuantEstimator,
    compressed: dict,
) -> np.ndarray:
    """Run Python dequantize and return as numpy."""
    result = estimator.dequantize(compressed)
    return result.cpu().numpy()


def run_c_dequantize(
    lib,
    blob: bytes,
    centroids: np.ndarray,
    rotation_matrix: np.ndarray | None,
    wht_signs: np.ndarray | None,
    d: int,
    mse_bits: int,
    use_wht: bool,
) -> np.ndarray:
    """Call the C dequantize function via ctypes."""
    dst = (ctypes.c_float * d)()
    src = ctypes.create_string_buffer(blob)
    c_centroids = numpy_to_c_float_ptr(centroids)

    if use_wht:
        c_rotation = None
        c_wht_signs = numpy_to_c_float_ptr(wht_signs)
    else:
        c_rotation = numpy_to_c_float_ptr(rotation_matrix)
        c_wht_signs = None

    lib.residualquant_dequantize(
        dst,
        ctypes.cast(src, ctypes.c_void_p),
        c_centroids,
        c_rotation,
        c_wht_signs,
        d,
        mse_bits,
        1 if use_wht else 0,
    )

    return np.array(dst[:d], dtype=np.float32)


def run_c_dequantize_row(
    lib,
    blob: bytes,
    centroids: np.ndarray,
    rotation_matrix: np.ndarray | None,
    wht_signs: np.ndarray | None,
    n_elements: int,
    d: int,
    mse_bits: int,
    use_wht: bool,
) -> np.ndarray:
    """Call the C row dequantize function via ctypes."""
    dst = (ctypes.c_float * n_elements)()
    src = ctypes.create_string_buffer(blob)
    c_centroids = numpy_to_c_float_ptr(centroids)

    if use_wht:
        c_rotation = None
        c_wht_signs = numpy_to_c_float_ptr(wht_signs)
    else:
        c_rotation = numpy_to_c_float_ptr(rotation_matrix)
        c_wht_signs = None

    lib.residualquant_dequantize_row(
        dst,
        ctypes.cast(src, ctypes.c_void_p),
        c_centroids,
        c_rotation,
        c_wht_signs,
        n_elements,
        d,
        mse_bits,
        1 if use_wht else 0,
    )

    return np.array(dst[:n_elements], dtype=np.float32)


def test_single_vector(lib, d: int, bits: int, seed: int, use_wht: bool) -> dict:
    """Test dequantize of a single vector, comparing Python vs C.

    Returns dict with results and max error.
    """
    rotation_type = "wht" if (use_wht and (d & (d - 1)) == 0) else "qr"
    estimator = ResidualQuantEstimator(d=d, bits=bits, seed=seed, device="cpu")

    # Force rotation type if needed
    if estimator.polar.rotation_type != rotation_type:
        from turboquantdc.polarquant import PolarQuant
        estimator.polar = PolarQuant(d, estimator.mse_bits, seed=seed, device="cpu",
                                      rotation_type=rotation_type)

    # Generate a random test vector
    torch.manual_seed(12345)
    x = torch.randn(d)
    x = x * 5.0  # scale to non-trivial magnitude

    # Python quantize + dequantize
    compressed = estimator.quantize(x)
    py_result = run_python_dequantize(estimator, compressed)

    # Extract data for C
    mse_indices = compressed["mse_indices"].numpy().astype(np.int32)
    residual_signs = compressed["residual_signs"].numpy().astype(np.float32)
    residual_scale = float(compressed["residual_scale"])
    vec_norm = float(compressed["vec_norm"])

    centroids = estimator.polar.codebook.centroids.numpy().astype(np.float32)
    mse_bits = estimator.mse_bits

    # Build binary blob
    blob = build_binary_blob(mse_indices, residual_signs, residual_scale, vec_norm, mse_bits)

    # Extract rotation data from Python
    if rotation_type == "wht":
        wht_signs = estimator.polar.wht_signs.numpy().astype(np.float32)
        rotation_matrix = None
    else:
        rotation_matrix = estimator.polar.Pi.numpy().astype(np.float32)
        wht_signs = None

    # C dequantize
    c_result = run_c_dequantize(
        lib, blob, centroids, rotation_matrix, wht_signs, d, mse_bits,
        use_wht=(rotation_type == "wht"),
    )

    # Compare
    max_abs_error = np.max(np.abs(py_result - c_result))
    mean_abs_error = np.mean(np.abs(py_result - c_result))

    # Also check relative error (avoiding division by near-zero)
    norms = np.abs(py_result)
    mask = norms > 1e-6
    if mask.any():
        max_rel_error = np.max(np.abs(py_result[mask] - c_result[mask]) / norms[mask])
    else:
        max_rel_error = 0.0

    return {
        "d": d,
        "bits": bits,
        "rotation_type": rotation_type,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_rel_error": max_rel_error,
        "py_norm": np.linalg.norm(py_result),
        "c_norm": np.linalg.norm(c_result),
        "vec_norm": vec_norm,
    }


def test_batch(lib, d: int, bits: int, n_vectors: int, seed: int, use_wht: bool) -> dict:
    """Test dequantize_row with multiple vectors."""
    rotation_type = "wht" if (use_wht and (d & (d - 1)) == 0) else "qr"
    estimator = ResidualQuantEstimator(d=d, bits=bits, seed=seed, device="cpu")

    if estimator.polar.rotation_type != rotation_type:
        from turboquantdc.polarquant import PolarQuant
        estimator.polar = PolarQuant(d, estimator.mse_bits, seed=seed, device="cpu",
                                      rotation_type=rotation_type)

    torch.manual_seed(99999)
    x_batch = torch.randn(n_vectors, d) * 3.0

    # Python: quantize + dequantize batch
    compressed = estimator.quantize(x_batch)
    py_result = run_python_dequantize(estimator, compressed).flatten()

    # Build concatenated binary blob for all vectors
    centroids = estimator.polar.codebook.centroids.numpy().astype(np.float32)
    mse_bits_val = estimator.mse_bits

    blob_parts = []
    for i in range(n_vectors):
        idx = compressed["mse_indices"][i].numpy().astype(np.int32)
        sgn = compressed["residual_signs"][i].numpy().astype(np.float32)
        scl = float(compressed["residual_scale"][i])
        nrm = float(compressed["vec_norm"][i])
        blob_parts.append(build_binary_blob(idx, sgn, scl, nrm, mse_bits_val))
    blob = b"".join(blob_parts)

    # Extract rotation data
    if rotation_type == "wht":
        wht_signs = estimator.polar.wht_signs.numpy().astype(np.float32)
        rotation_matrix = None
    else:
        rotation_matrix = estimator.polar.Pi.numpy().astype(np.float32)
        wht_signs = None

    # C row dequantize
    c_result = run_c_dequantize_row(
        lib, blob, centroids, rotation_matrix, wht_signs,
        n_vectors * d, d, mse_bits_val,
        use_wht=(rotation_type == "wht"),
    )

    max_abs_error = np.max(np.abs(py_result - c_result))
    mean_abs_error = np.mean(np.abs(py_result - c_result))

    return {
        "d": d,
        "bits": bits,
        "n_vectors": n_vectors,
        "rotation_type": rotation_type,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
    }


def main():
    print("=" * 70)
    print("ResidualQuant: C vs Python Validation")
    print("=" * 70)
    print()

    # Compile
    lib_path = compile_lib()
    lib = load_lib(lib_path)
    print()

    # FP16 introduces quantization error of ~1e-3 for typical values.
    # We allow up to 2e-3 absolute error to account for FP16 round-trip
    # on both scale and norm, plus any float32 accumulation differences.
    FP16_TOLERANCE = 2e-3
    PASS_THRESHOLD = 5e-3  # generous threshold for combined FP16 + float ops

    all_passed = True
    test_configs = []

    # ---- Single vector tests ----
    print("--- Single Vector Tests ---")
    print(f"{'d':>5}  {'bits':>4}  {'rotation':>8}  {'max_abs':>10}  {'mean_abs':>10}  {'max_rel':>10}  {'status':>6}")
    print("-" * 65)

    for d in [128, 256]:
        for bits in [3, 4]:
            for use_wht in [True, False]:
                # Skip WHT for non-power-of-2 dimensions
                if use_wht and (d & (d - 1)) != 0:
                    continue

                result = test_single_vector(lib, d, bits, seed=42, use_wht=use_wht)
                passed = result["max_abs_error"] < PASS_THRESHOLD
                status = "PASS" if passed else "FAIL"
                if not passed:
                    all_passed = False

                print(f"{result['d']:>5}  {result['bits']:>4}  {result['rotation_type']:>8}  "
                      f"{result['max_abs_error']:>10.2e}  {result['mean_abs_error']:>10.2e}  "
                      f"{result['max_rel_error']:>10.2e}  {status:>6}")

                test_configs.append(result)

    print()

    # ---- Batch tests ----
    print("--- Batch Tests (dequantize_row) ---")
    print(f"{'d':>5}  {'bits':>4}  {'n_vec':>5}  {'rotation':>8}  {'max_abs':>10}  {'mean_abs':>10}  {'status':>6}")
    print("-" * 65)

    for d in [128, 256]:
        for bits in [3]:
            for n_vectors in [8, 32]:
                for use_wht in [True]:
                    result = test_batch(lib, d, bits, n_vectors, seed=42, use_wht=use_wht)
                    passed = result["max_abs_error"] < PASS_THRESHOLD
                    status = "PASS" if passed else "FAIL"
                    if not passed:
                        all_passed = False

                    print(f"{result['d']:>5}  {result['bits']:>4}  {result['n_vectors']:>5}  "
                          f"{result['rotation_type']:>8}  {result['max_abs_error']:>10.2e}  "
                          f"{result['mean_abs_error']:>10.2e}  {status:>6}")

    print()

    # ---- Binary format round-trip test ----
    print("--- Binary Format Round-Trip Test ---")
    from tools.export_residualquant_gguf import (
        export_residualquant,
        load_residualquant,
        unpack_indices,
        unpack_signs,
    )

    d = 128
    bits = 3
    estimator = ResidualQuantEstimator(d=d, bits=bits, seed=42, device="cpu")

    torch.manual_seed(77777)
    x = torch.randn(4, d) * 2.0
    compressed = estimator.quantize(x)

    # Export
    tmpfile = TOOLS_DIR / "_test_roundtrip.rq"
    export_residualquant(estimator, [compressed], str(tmpfile))

    # Load
    loaded = load_residualquant(str(tmpfile))

    # Verify header
    hdr = loaded["header"]
    assert hdr["d"] == d, f"d mismatch: {hdr['d']} vs {d}"
    assert hdr["bits"] == bits, f"bits mismatch: {hdr['bits']} vs {bits}"
    assert hdr["n_vectors"] == 4, f"n_vectors mismatch: {hdr['n_vectors']} vs 4"
    assert hdr["n_centroids"] == (1 << (bits - 1)), \
        f"n_centroids mismatch: {hdr['n_centroids']} vs {1 << (bits - 1)}"

    # Verify codebook
    orig_centroids = estimator.polar.codebook.centroids.numpy()
    assert np.allclose(loaded["centroids"], orig_centroids, atol=1e-7), "Centroid mismatch"

    # Verify vectors
    for i, vec in enumerate(loaded["vectors"]):
        orig_idx = compressed["mse_indices"][i].numpy()
        orig_sgn = compressed["residual_signs"][i].numpy()
        orig_scl = float(compressed["residual_scale"][i])
        orig_nrm = float(compressed["vec_norm"][i])

        assert np.array_equal(vec["mse_indices"], orig_idx), f"Vector {i}: index mismatch"
        assert np.array_equal(vec["residual_signs"], orig_sgn), f"Vector {i}: sign mismatch"
        # FP16 round-trip tolerance: absolute for small values, relative for large.
        # FP16 precision degrades with magnitude (step size = 2^(e-10) for exponent e).
        scl_tol = max(FP16_TOLERANCE, abs(orig_scl) * 1e-3)
        nrm_tol = max(FP16_TOLERANCE, abs(orig_nrm) * 1e-3)
        assert abs(vec["residual_scale"] - orig_scl) < scl_tol, \
            f"Vector {i}: scale mismatch {vec['residual_scale']} vs {orig_scl}"
        assert abs(vec["vec_norm"] - orig_nrm) < nrm_tol, \
            f"Vector {i}: norm mismatch {vec['vec_norm']} vs {orig_nrm}"

    # Cleanup
    tmpfile.unlink()
    print("Binary format round-trip: PASS (4 vectors, d=128, 3-bit)")
    print()

    # ---- Summary ----
    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
        print(f"Max absolute error across all tests: "
              f"{max(r['max_abs_error'] for r in test_configs):.2e}")
        print(f"FP16 quantization accounts for ~{FP16_TOLERANCE:.0e} of this error.")
    else:
        print("SOME TESTS FAILED")
        for r in test_configs:
            if r.get("max_abs_error", 0) >= PASS_THRESHOLD:
                print(f"  FAIL: d={r['d']}, bits={r['bits']}, "
                      f"rotation={r.get('rotation_type', '?')}, "
                      f"max_abs_error={r['max_abs_error']:.2e}")

    print("=" * 70)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
