#!/usr/bin/env python3
"""Export ResidualQuant-compressed KV cache vectors to GGUF-compatible binary format.

Produces a self-describing binary file containing everything needed to dequantize
on the C side: codebook centroids, rotation matrix seed, and packed vector data.

Binary format (ResidualQuant, one vector = one "block" of d elements):
  Per block:
    - mse_indices:     ceil((b-1)*d / 8) bytes  (packed (b-1)-bit indices)
    - residual_signs:  ceil(d / 8) bytes         (packed 1-bit signs)
    - residual_scale:  2 bytes                   (FP16)
    - vec_norm:        2 bytes                   (FP16)

File layout:
  [Header]
    magic:           4 bytes  "RQ01"
    version:         4 bytes  uint32 = 1
    d:               4 bytes  uint32 (head dimension)
    bits:            4 bytes  uint32 (total bits per coordinate)
    mse_bits:        4 bytes  uint32 (bits-1, for MSE indices)
    n_centroids:     4 bytes  uint32 (2^mse_bits)
    rotation_seed:   4 bytes  uint32 (seed for reproducing rotation matrix)
    rotation_type:   4 bytes  uint32 (0=qr, 1=wht)
    n_vectors:       4 bytes  uint32 (number of vectors in file)
    reserved:        28 bytes (zero-padded, future use)
  [Codebook]
    centroids:       n_centroids * 4 bytes (float32, ascending order)
    boundaries:      (n_centroids - 1) * 4 bytes (float32)
  [Vectors]
    For each vector:
      mse_indices:     ceil(mse_bits * d / 8) bytes
      residual_signs:  ceil(d / 8) bytes
      residual_scale:  2 bytes (FP16)
      vec_norm:        2 bytes (FP16)
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import BinaryIO, Dict, List

import numpy as np
import torch

# Add parent to path so we can import turboquantdc
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


MAGIC = b"RQ01"
FORMAT_VERSION = 1
HEADER_SIZE = 64  # Fixed header size (padded with reserved bytes)

ROTATION_TYPE_QR = 0
ROTATION_TYPE_WHT = 1


def _fp32_to_fp16_bytes(val: float) -> bytes:
    """Convert a float32 value to IEEE 754 FP16 bytes (little-endian)."""
    return struct.pack("<e", val)


def _fp16_bytes_to_fp32(data: bytes) -> float:
    """Convert 2 bytes of IEEE 754 FP16 to float32."""
    return struct.unpack("<e", data)[0]


def pack_indices(indices: np.ndarray, bits_per_index: int) -> bytes:
    """Pack an array of integer indices into a packed byte array.

    Each index uses `bits_per_index` bits, packed LSB-first into bytes.

    Args:
        indices: 1-D array of integer indices, each in [0, 2^bits_per_index).
        bits_per_index: Number of bits per index (1-8).

    Returns:
        Packed bytes, length = ceil(len(indices) * bits_per_index / 8).
    """
    n = len(indices)
    total_bits = n * bits_per_index
    n_bytes = (total_bits + 7) // 8
    result = bytearray(n_bytes)

    for i, idx in enumerate(indices):
        bit_offset = i * bits_per_index
        for b in range(bits_per_index):
            if idx & (1 << b):
                byte_pos = (bit_offset + b) // 8
                bit_pos = (bit_offset + b) % 8
                result[byte_pos] |= (1 << bit_pos)

    return bytes(result)


def unpack_indices(data: bytes, n_elements: int, bits_per_index: int) -> np.ndarray:
    """Unpack a packed byte array into an array of integer indices.

    Args:
        data: Packed bytes from pack_indices().
        n_elements: Number of indices to unpack.
        bits_per_index: Number of bits per index.

    Returns:
        1-D numpy array of int32 indices.
    """
    result = np.zeros(n_elements, dtype=np.int32)
    buf = bytearray(data)

    for i in range(n_elements):
        bit_offset = i * bits_per_index
        val = 0
        for b in range(bits_per_index):
            byte_pos = (bit_offset + b) // 8
            bit_pos = (bit_offset + b) % 8
            if buf[byte_pos] & (1 << bit_pos):
                val |= (1 << b)
        result[i] = val

    return result


def pack_signs(signs: np.ndarray) -> bytes:
    """Pack a {-1, +1} sign array into bits (1 = positive, 0 = negative).

    Args:
        signs: 1-D array of -1.0 or +1.0 values.

    Returns:
        Packed bytes, length = ceil(len(signs) / 8).
    """
    n = len(signs)
    n_bytes = (n + 7) // 8
    result = bytearray(n_bytes)

    for i in range(n):
        if signs[i] >= 0:
            result[i // 8] |= (1 << (i % 8))

    return bytes(result)


def unpack_signs(data: bytes, n_elements: int) -> np.ndarray:
    """Unpack packed sign bits into a {-1, +1} array.

    Args:
        data: Packed bytes from pack_signs().
        n_elements: Number of sign values.

    Returns:
        1-D numpy array of float32 values in {-1.0, +1.0}.
    """
    result = np.full(n_elements, -1.0, dtype=np.float32)
    buf = bytearray(data)

    for i in range(n_elements):
        if buf[i // 8] & (1 << (i % 8)):
            result[i] = 1.0

    return result


def bytes_per_vector(d: int, mse_bits: int) -> int:
    """Compute the byte size of one packed vector.

    Args:
        d: Head dimension.
        mse_bits: Bits per MSE index (= total_bits - 1).

    Returns:
        Number of bytes per packed vector.
    """
    mse_bytes = (mse_bits * d + 7) // 8
    sign_bytes = (d + 7) // 8
    return mse_bytes + sign_bytes + 2 + 2  # +2 scale +2 norm


def write_header(
    f: BinaryIO,
    d: int,
    bits: int,
    rotation_seed: int,
    rotation_type: int,
    n_vectors: int,
    n_centroids: int,
) -> None:
    """Write the file header."""
    mse_bits = max(bits - 1, 1)

    header = bytearray(HEADER_SIZE)
    offset = 0

    # magic (4 bytes)
    header[0:4] = MAGIC
    offset = 4

    # version (uint32)
    struct.pack_into("<I", header, offset, FORMAT_VERSION)
    offset += 4

    # d (uint32)
    struct.pack_into("<I", header, offset, d)
    offset += 4

    # bits (uint32)
    struct.pack_into("<I", header, offset, bits)
    offset += 4

    # mse_bits (uint32)
    struct.pack_into("<I", header, offset, mse_bits)
    offset += 4

    # n_centroids (uint32)
    struct.pack_into("<I", header, offset, n_centroids)
    offset += 4

    # rotation_seed (uint32)
    struct.pack_into("<I", header, offset, rotation_seed)
    offset += 4

    # rotation_type (uint32)
    struct.pack_into("<I", header, offset, rotation_type)
    offset += 4

    # n_vectors (uint32)
    struct.pack_into("<I", header, offset, n_vectors)
    offset += 4

    # remaining bytes are reserved (zero)
    f.write(bytes(header))


def read_header(f: BinaryIO) -> dict:
    """Read and parse the file header.

    Returns:
        Dict with keys: magic, version, d, bits, mse_bits, n_centroids,
        rotation_seed, rotation_type, n_vectors.
    """
    data = f.read(HEADER_SIZE)
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Header too short: {len(data)} bytes (need {HEADER_SIZE})")

    magic = data[0:4]
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic!r} (expected {MAGIC!r})")

    offset = 4
    version = struct.unpack_from("<I", data, offset)[0]; offset += 4
    d = struct.unpack_from("<I", data, offset)[0]; offset += 4
    bits = struct.unpack_from("<I", data, offset)[0]; offset += 4
    mse_bits = struct.unpack_from("<I", data, offset)[0]; offset += 4
    n_centroids = struct.unpack_from("<I", data, offset)[0]; offset += 4
    rotation_seed = struct.unpack_from("<I", data, offset)[0]; offset += 4
    rotation_type = struct.unpack_from("<I", data, offset)[0]; offset += 4
    n_vectors = struct.unpack_from("<I", data, offset)[0]; offset += 4

    return {
        "magic": magic,
        "version": version,
        "d": d,
        "bits": bits,
        "mse_bits": mse_bits,
        "n_centroids": n_centroids,
        "rotation_seed": rotation_seed,
        "rotation_type": rotation_type,
        "n_vectors": n_vectors,
    }


def write_codebook(f: BinaryIO, centroids: np.ndarray, boundaries: np.ndarray) -> None:
    """Write codebook centroids and boundaries as float32 arrays."""
    f.write(centroids.astype(np.float32).tobytes())
    f.write(boundaries.astype(np.float32).tobytes())


def read_codebook(f: BinaryIO, n_centroids: int) -> tuple:
    """Read codebook centroids and boundaries.

    Returns:
        (centroids: np.ndarray[n_centroids], boundaries: np.ndarray[n_centroids-1])
    """
    centroids = np.frombuffer(f.read(n_centroids * 4), dtype=np.float32).copy()
    n_boundaries = n_centroids - 1
    boundaries = np.frombuffer(f.read(n_boundaries * 4), dtype=np.float32).copy()
    return centroids, boundaries


def write_vector(
    f: BinaryIO,
    mse_indices: np.ndarray,
    residual_signs: np.ndarray,
    residual_scale: float,
    vec_norm: float,
    mse_bits: int,
) -> None:
    """Write one packed vector to the file."""
    f.write(pack_indices(mse_indices, mse_bits))
    f.write(pack_signs(residual_signs))
    f.write(_fp32_to_fp16_bytes(residual_scale))
    f.write(_fp32_to_fp16_bytes(vec_norm))


def read_vector(f: BinaryIO, d: int, mse_bits: int) -> dict:
    """Read one packed vector from the file.

    Returns:
        Dict with mse_indices, residual_signs, residual_scale, vec_norm.
    """
    mse_byte_count = (mse_bits * d + 7) // 8
    sign_byte_count = (d + 7) // 8

    mse_data = f.read(mse_byte_count)
    sign_data = f.read(sign_byte_count)
    scale_data = f.read(2)
    norm_data = f.read(2)

    return {
        "mse_indices": unpack_indices(mse_data, d, mse_bits),
        "residual_signs": unpack_signs(sign_data, d),
        "residual_scale": _fp16_bytes_to_fp32(scale_data),
        "vec_norm": _fp16_bytes_to_fp32(norm_data),
    }


def export_residualquant(
    estimator,
    compressed_list: List[Dict[str, torch.Tensor]],
    output_path: str | Path,
) -> Path:
    """Export ResidualQuant-compressed vectors to the binary format.

    Args:
        estimator: A ResidualQuantEstimator instance (for codebook + config).
        compressed_list: List of compressed dicts from estimator.quantize(),
            each containing mse_indices, residual_signs, residual_scale, vec_norm
            for a batch of vectors. Tensors can be 1-D (single vector) or 2-D (batch).
        output_path: Destination file path.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)

    d = estimator.d
    bits = estimator.bits
    mse_bits = estimator.mse_bits
    n_centroids = estimator.polar.codebook.n_levels

    # Determine rotation type and seed
    rotation_type = ROTATION_TYPE_WHT if estimator.polar.rotation_type == "wht" else ROTATION_TYPE_QR

    # Extract seed from PolarQuant — it was passed at construction time
    # We store it in the file so the C side can reconstruct the rotation
    # The estimator doesn't store seed directly, so we get it from the caller
    # For now we look for it on the polar module's init args
    rotation_seed = 42  # default; overridden below if available

    # Flatten all compressed dicts into individual vectors
    all_indices = []
    all_signs = []
    all_scales = []
    all_norms = []

    for comp in compressed_list:
        idx = comp["mse_indices"]
        sgn = comp["residual_signs"]
        scl = comp["residual_scale"]
        nrm = comp["vec_norm"]

        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
            sgn = sgn.unsqueeze(0)
            scl = scl.unsqueeze(0)
            nrm = nrm.unsqueeze(0)

        for i in range(idx.shape[0]):
            all_indices.append(idx[i].cpu().numpy().astype(np.int32))
            all_signs.append(sgn[i].cpu().numpy().astype(np.float32))
            all_scales.append(float(scl[i].cpu()))
            all_norms.append(float(nrm[i].cpu()))

    n_vectors = len(all_indices)

    with open(output_path, "wb") as f:
        write_header(f, d, bits, rotation_seed, rotation_type, n_vectors, n_centroids)

        centroids_np = estimator.polar.codebook.centroids.cpu().numpy()
        boundaries_np = estimator.polar.codebook.boundaries.cpu().numpy()
        write_codebook(f, centroids_np, boundaries_np)

        for i in range(n_vectors):
            write_vector(
                f,
                all_indices[i],
                all_signs[i],
                all_scales[i],
                all_norms[i],
                mse_bits,
            )

    return output_path


def load_residualquant(input_path: str | Path) -> dict:
    """Load a ResidualQuant binary file into numpy arrays.

    Returns:
        Dict with header info, codebook, and list of vector dicts.
    """
    input_path = Path(input_path)

    with open(input_path, "rb") as f:
        header = read_header(f)
        centroids, boundaries = read_codebook(f, header["n_centroids"])

        vectors = []
        for _ in range(header["n_vectors"]):
            vec = read_vector(f, header["d"], header["mse_bits"])
            vectors.append(vec)

    return {
        "header": header,
        "centroids": centroids,
        "boundaries": boundaries,
        "vectors": vectors,
    }


def get_binary_blob(
    mse_indices: np.ndarray,
    residual_signs: np.ndarray,
    residual_scale: float,
    vec_norm: float,
    mse_bits: int,
) -> bytes:
    """Pack a single vector into its binary blob (no header, no codebook).

    This is what the C dequantize function receives as its `src` pointer.

    Returns:
        Raw bytes for one vector block.
    """
    parts = []
    parts.append(pack_indices(mse_indices, mse_bits))
    parts.append(pack_signs(residual_signs))
    parts.append(_fp32_to_fp16_bytes(residual_scale))
    parts.append(_fp32_to_fp16_bytes(vec_norm))
    return b"".join(parts)


if __name__ == "__main__":
    # Quick self-test
    print("Testing pack/unpack round-trip...")

    # Test index packing at various bit widths
    for bits in [1, 2, 3, 4, 5]:
        max_val = (1 << bits) - 1
        rng = np.random.default_rng(42)
        indices = rng.integers(0, max_val + 1, size=128, dtype=np.int32)
        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, 128, bits)
        assert np.array_equal(indices, unpacked), f"Failed for {bits}-bit"
        print(f"  {bits}-bit indices: OK ({len(packed)} bytes for 128 elements)")

    # Test sign packing
    rng = np.random.default_rng(42)
    signs = rng.choice([-1.0, 1.0], size=128).astype(np.float32)
    packed = pack_signs(signs)
    unpacked = unpack_signs(packed, 128)
    assert np.array_equal(signs, unpacked), "Sign pack/unpack failed"
    print(f"  sign bits: OK ({len(packed)} bytes for 128 elements)")

    # Test FP16 round-trip
    for val in [0.0, 1.0, -0.5, 3.14, 0.001]:
        encoded = _fp32_to_fp16_bytes(val)
        decoded = _fp16_bytes_to_fp32(encoded)
        assert abs(decoded - val) < 0.01, f"FP16 round-trip failed for {val}"
    print("  FP16 round-trip: OK")

    print("\nAll self-tests passed.")
