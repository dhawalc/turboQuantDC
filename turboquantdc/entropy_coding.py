"""Entropy coding for quantized indices — free 5-10% compression.

Lloyd-Max quantization on a Gaussian distribution produces NON-UNIFORM index
usage: middle centroids (near zero) are used far more often than tail centroids.
This means the actual Shannon entropy is LESS than the allocated bit-width.

Example at 3-bit (8 centroids) on N(0, 1/128):
    Middle centroids (3,4): ~19% usage each
    Edge centroids (0,7):   ~4% usage each
    Shannon entropy:         ~2.82 bits (vs allocated 3.0 bits)
    Free compression:        ~6%

Savings scale with bit-width:
    2-bit: ~4.4% savings (entropy 1.91 vs 2.0 bits)
    3-bit: ~5.8% savings (entropy 2.82 vs 3.0 bits)
    4-bit: ~5.9% savings (entropy 3.77 vs 4.0 bits)
    7-bit: ~10% savings (entropy 6.30 vs 7.0 bits)

This module provides:
    1. Analysis functions to measure the compression opportunity
    2. A rANS (range ANS) entropy encoder/decoder for quantized indices
    3. A zlib-based fast fallback encoder
    4. CompressedPolarQuant: PolarQuant + entropy coding integration

The rANS encoder achieves near-Shannon-limit compression. The zlib
backend provides high throughput via its C implementation.
"""

from __future__ import annotations

import math
import struct
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from scipy import integrate

from .codebook import LloydMaxCodebook, gaussian_pdf
from .polarquant import PolarQuant


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def _symbol_probabilities(codebook: LloydMaxCodebook) -> np.ndarray:
    """Compute the probability of each centroid being selected.

    For a coordinate drawn from N(0, 1/d), the probability that it falls
    into partition i is: P_i = integral_{b_{i-1}}^{b_i} f(x) dx,
    where f is the Gaussian PDF and [b_{i-1}, b_i] are the boundaries.

    Args:
        codebook: A solved Lloyd-Max codebook.

    Returns:
        Array of shape (n_levels,) with probabilities summing to 1.
    """
    d = codebook.d
    n_levels = codebook.n_levels
    sigma = 1.0 / math.sqrt(d)
    lo = -10.5 * sigma  # match codebook integration range
    hi = 10.5 * sigma

    boundaries_list = codebook.boundaries.tolist()
    edges = [lo] + boundaries_list + [hi]

    probs = np.zeros(n_levels, dtype=np.float64)
    for i in range(n_levels):
        left = edges[i]
        right = edges[i + 1]
        val, _ = integrate.quad(lambda x: gaussian_pdf(x, d), left, right)
        probs[i] = val

    # Normalize to handle integration rounding
    total = probs.sum()
    if total > 0:
        probs /= total

    return probs


def measure_index_entropy(indices: torch.Tensor, n_levels: int) -> float:
    """Measure the empirical Shannon entropy of quantized indices.

    H = -sum(p_i * log2(p_i)) for observed symbol frequencies.

    Args:
        indices: Integer tensor of quantized indices.
        n_levels: Number of quantization levels (2^bits).

    Returns:
        Empirical entropy in bits per symbol.
    """
    flat = indices.reshape(-1).cpu().numpy()
    counts = np.bincount(flat, minlength=n_levels).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0

    probs = counts / total
    # Filter zero probabilities to avoid log(0)
    nonzero = probs[probs > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero))
    return float(entropy)


def theoretical_index_entropy(codebook: LloydMaxCodebook) -> float:
    """Compute the theoretical Shannon entropy from Gaussian PDF + codebook.

    This is the expected entropy for indices produced by quantizing
    coordinates drawn from N(0, 1/d) using the given codebook.

    Args:
        codebook: A solved Lloyd-Max codebook.

    Returns:
        Theoretical entropy in bits per symbol.
    """
    probs = _symbol_probabilities(codebook)
    nonzero = probs[probs > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero))
    return float(entropy)


def compression_opportunity(codebook: LloydMaxCodebook) -> Dict[str, float]:
    """Compute the compression opportunity for a given codebook.

    Args:
        codebook: A solved Lloyd-Max codebook.

    Returns:
        Dict with:
            allocated_bits: bits per symbol (the bit-width)
            theoretical_entropy: Shannon entropy from PDF
            entropy_ratio: entropy / allocated_bits (< 1.0 means room to compress)
            savings_pct: (1 - ratio) * 100, the percentage savings
    """
    allocated = float(codebook.bits)
    entropy = theoretical_index_entropy(codebook)
    ratio = entropy / allocated if allocated > 0 else 1.0
    savings = (1.0 - ratio) * 100.0

    return {
        "allocated_bits": allocated,
        "theoretical_entropy": entropy,
        "entropy_ratio": ratio,
        "savings_pct": savings,
    }


# ---------------------------------------------------------------------------
# rANS Entropy Coder (Range Asymmetric Numeral Systems)
# ---------------------------------------------------------------------------


class ANSEncoder:
    """Range ANS (rANS) entropy coder for quantized indices.

    rANS achieves near-Shannon-limit compression. It encodes symbols into
    a state integer, emitting bytes when the state grows too large. Decoding
    processes the byte stream in reverse (LIFO) to recover the original
    symbols.

    The key invariant: state is always in [M, 2M) after renormalization,
    where M = sum of all quantized frequencies.

    Args:
        codebook: Lloyd-Max codebook providing symbol probabilities.
        table_bits: Log2 of the frequency table resolution (default: 12).
    """

    def __init__(self, codebook: LloydMaxCodebook, table_bits: int = 12):
        self.n_symbols = codebook.n_levels
        self.table_bits = table_bits
        self.table_size = 1 << table_bits  # M

        # Compute symbol probabilities from the codebook
        probs = _symbol_probabilities(codebook)
        self._probs = probs

        # Quantize probabilities to integer frequencies summing to table_size
        self._freqs = self._quantize_freqs(probs)

        # Cumulative frequencies
        self._cum_freqs = np.zeros(self.n_symbols + 1, dtype=np.int64)
        for i in range(self.n_symbols):
            self._cum_freqs[i + 1] = self._cum_freqs[i] + self._freqs[i]

        # Build reverse lookup table for decoding: given a cumulative offset,
        # find which symbol it belongs to
        self._cum_to_sym = np.zeros(self.table_size, dtype=np.int32)
        for sym in range(self.n_symbols):
            start = int(self._cum_freqs[sym])
            end = int(self._cum_freqs[sym + 1])
            self._cum_to_sym[start:end] = sym

        # Cache compressed bits per symbol
        self._compressed_bps: Optional[float] = None

    def _quantize_freqs(self, probs: np.ndarray) -> np.ndarray:
        """Quantize probabilities to integer frequencies summing to table_size.

        Uses the largest-remainder method to ensure exact sum while
        preserving relative proportions. Every symbol gets at least
        frequency 1 (required for rANS correctness).
        """
        n = self.n_symbols
        M = self.table_size

        # Start with floor values, minimum 1
        scaled = probs * M
        freqs = np.maximum(np.floor(scaled).astype(np.int64), 1)

        # Distribute remaining counts by largest remainder
        remainder = M - freqs.sum()
        if remainder > 0:
            fractional = scaled - freqs
            order = np.argsort(-fractional)
            for i in range(int(remainder)):
                freqs[order[i % n]] += 1
        elif remainder < 0:
            order = np.argsort(-freqs)
            for i in range(int(-remainder)):
                idx = order[i % n]
                if freqs[idx] > 1:
                    freqs[idx] -= 1

        assert freqs.sum() == M, f"Freq sum {freqs.sum()} != {M}"
        assert np.all(freqs >= 1), "All symbols must have freq >= 1"
        return freqs

    @property
    def compressed_bits_per_symbol(self) -> float:
        """Expected bits per symbol after entropy coding.

        Computed from the quantized frequency distribution. The actual
        rate may be slightly higher due to per-block state overhead.
        """
        if self._compressed_bps is not None:
            return self._compressed_bps

        M = self.table_size
        probs_q = self._freqs.astype(np.float64) / M
        nonzero = probs_q[probs_q > 0]
        self._compressed_bps = float(-np.sum(nonzero * np.log2(nonzero)))
        return self._compressed_bps

    def encode(self, indices: torch.Tensor) -> bytes:
        """Encode quantized indices to a compressed byte stream using rANS.

        Standard rANS byte-level coding. Encoder processes symbols in
        REVERSE order; decoder reads them back in forward order.

        State invariant (maintained by renormalization):
            After encoding symbol s, state is in [M, M * 256^k) for some k.
            Before encoding symbol s, state must be in [f_s, f_s * L)
            where L = (M // f_s) * 256^k to ensure the post-encode state
            stays bounded.

        We use the standard byte-aligned rANS approach:
            L = 1 << 23  (lower bound for decoder renormalization)
            Encoder emits bytes when state >= (f_s / M) * (L << 8)

        Format:
            [4B: n_symbols][8B: final_state][4B: n_emitted_bytes][emitted...]

        Args:
            indices: Integer tensor with values in [0, n_symbols).

        Returns:
            Compressed bytes.
        """
        flat = indices.reshape(-1).cpu().numpy().astype(np.int32)
        n = len(flat)
        M = self.table_size
        L = 1 << 23  # decoder lower bound

        # Encode in REVERSE order
        state = L  # Initial state at lower bound
        emitted: List[int] = []

        freqs = self._freqs
        cum_freqs = self._cum_freqs

        for i in range(n - 1, -1, -1):
            sym = int(flat[i])
            f_s = int(freqs[sym])
            c_s = int(cum_freqs[sym])

            # Renormalize: we need state in [f_s, f_s * (L * 256 // M))
            # after encode, which means before encode: state < f_s * (L * 256 // M)
            # Simpler: emit bytes while (state // f_s) >= (L * 256 // M)
            # i.e. while state >= f_s * ((L << 8) // M)
            threshold = ((L << 8) // M) * f_s
            while state >= threshold:
                emitted.append(state & 0xFF)
                state >>= 8

            # Encode: state' = (state // f_s) * M + (state % f_s) + c_s
            state = (state // f_s) * M + (state % f_s) + c_s

        output = bytearray()
        output.extend(struct.pack("<I", n))
        output.extend(struct.pack("<Q", state))
        output.extend(struct.pack("<I", len(emitted)))
        output.extend(bytes(emitted))

        return bytes(output)

    def decode(self, data: bytes, length: int = 0) -> torch.Tensor:
        """Decode compressed bytes back to quantized indices.

        Processes symbols in forward order (encoder encoded in reverse).

        Args:
            data: Compressed bytes from encode().
            length: Expected number of symbols (if 0, read from header).

        Returns:
            Integer tensor of decoded indices.
        """
        off = 0
        n = struct.unpack_from("<I", data, off)[0]
        off += 4
        state = struct.unpack_from("<Q", data, off)[0]
        off += 8
        n_emitted = struct.unpack_from("<I", data, off)[0]
        off += 4

        if length > 0:
            n = length

        emitted = list(data[off:off + n_emitted])

        M = self.table_size
        L = 1 << 23
        freqs = self._freqs
        cum_freqs = self._cum_freqs
        cum_to_sym = self._cum_to_sym

        symbols = np.zeros(n, dtype=np.int32)
        byte_idx = len(emitted) - 1

        for i in range(n):
            # Extract symbol
            slot = int(state % M)
            sym = int(cum_to_sym[slot])
            symbols[i] = sym

            f_s = int(freqs[sym])
            c_s = int(cum_freqs[sym])

            # Decode: reverse the encoding step
            state = f_s * (state // M) + (slot - c_s)

            # Reverse renormalization: read bytes until state >= L
            while state < L and byte_idx >= 0:
                state = (state << 8) | emitted[byte_idx]
                byte_idx -= 1

        return torch.from_numpy(symbols.astype(np.int64))


# ---------------------------------------------------------------------------
# Fast batch encoder using zlib (fallback for maximum throughput)
# ---------------------------------------------------------------------------


class ZlibEncoder:
    """Fast entropy encoder using zlib for batch compression.

    While not optimal (zlib uses LZ77+Huffman, not ANS), it is extremely
    fast due to C implementation and provides reasonable compression for
    the non-uniform index distributions from Lloyd-Max quantization.

    For the quantized indices, zlib achieves ~70-80% of the theoretical
    entropy limit, which is good enough for a production fallback.

    Args:
        codebook: Lloyd-Max codebook (used for metadata only).
        level: zlib compression level (1-9, default 6).
    """

    def __init__(self, codebook: LloydMaxCodebook, level: int = 6):
        import zlib
        self._zlib = zlib
        self.n_symbols = codebook.n_levels
        self.bits = codebook.bits
        self._level = level
        self._codebook = codebook
        self._compressed_bps: Optional[float] = None

    @property
    def compressed_bits_per_symbol(self) -> float:
        """Estimated bits per symbol (theoretical entropy of the distribution).

        This is the theoretical Shannon entropy, which is the best
        achievable rate. Zlib typically gets within 5-15% of this bound.
        """
        if self._compressed_bps is not None:
            return self._compressed_bps
        self._compressed_bps = theoretical_index_entropy(self._codebook)
        return self._compressed_bps

    def encode(self, indices: torch.Tensor) -> bytes:
        """Compress quantized indices using zlib.

        Packs indices into the minimum number of bytes per symbol,
        then applies zlib compression.

        Args:
            indices: Integer tensor of indices.

        Returns:
            Compressed bytes with shape header.
        """
        flat = indices.reshape(-1).cpu().numpy().astype(np.uint8)
        shape_bytes = struct.pack("<I", len(flat))

        # If indices fit in a byte (bits <= 8), pack directly
        raw = flat.tobytes()
        compressed = self._zlib.compress(raw, self._level)

        return shape_bytes + compressed

    def decode(self, data: bytes, length: int = 0) -> torch.Tensor:
        """Decompress bytes back to indices.

        Args:
            data: Compressed bytes from encode().
            length: Ignored (read from header).

        Returns:
            Integer tensor of indices.
        """
        n = struct.unpack_from("<I", data, 0)[0]
        compressed = data[4:]
        raw = self._zlib.decompress(compressed)
        flat = np.frombuffer(raw, dtype=np.uint8)[:n]
        return torch.from_numpy(flat.astype(np.int64))


# ---------------------------------------------------------------------------
# Unified EntropyEncoder interface
# ---------------------------------------------------------------------------


class EntropyEncoder:
    """Unified entropy encoder for quantized indices.

    Selects the best available backend:
    - 'ans': tANS encoder (near-optimal compression, pure Python)
    - 'zlib': zlib-based encoder (fast C implementation, ~80% optimal)
    - 'auto': ANS for small batches, zlib for large batches

    Args:
        codebook: Lloyd-Max codebook providing symbol statistics.
        backend: Encoder backend ('ans', 'zlib', or 'auto').
    """

    def __init__(
        self,
        codebook: LloydMaxCodebook,
        backend: str = "auto",
    ):
        self._codebook = codebook
        self._backend = backend

        # Build both encoders lazily
        self._ans: Optional[ANSEncoder] = None
        self._zlib: Optional[ZlibEncoder] = None

        if backend in ("ans", "auto"):
            self._ans = ANSEncoder(codebook)
        if backend in ("zlib", "auto"):
            self._zlib = ZlibEncoder(codebook)

    @property
    def compressed_bits_per_symbol(self) -> float:
        """Expected compressed bits per symbol."""
        if self._ans is not None:
            return self._ans.compressed_bits_per_symbol
        return self._zlib.compressed_bits_per_symbol

    def encode(self, indices: torch.Tensor) -> bytes:
        """Compress quantized indices.

        For 'auto' backend, uses zlib for batches > 10000 symbols
        (faster throughput) and ANS for smaller batches (better ratio).

        Args:
            indices: Integer tensor of quantized indices.

        Returns:
            Compressed bytes.
        """
        n = indices.numel()

        if self._backend == "auto":
            # Use zlib for large batches (faster), ANS for small (better ratio)
            if n > 10000:
                tag = b"\x01"  # zlib tag
                return tag + self._zlib.encode(indices)
            else:
                tag = b"\x00"  # ANS tag
                return tag + self._ans.encode(indices)
        elif self._backend == "zlib":
            return b"\x01" + self._zlib.encode(indices)
        else:
            return b"\x00" + self._ans.encode(indices)

    def decode(self, data: bytes, length: int = 0) -> torch.Tensor:
        """Decompress bytes back to quantized indices.

        Args:
            data: Compressed bytes from encode().
            length: Expected number of symbols (optional hint).

        Returns:
            Integer tensor of indices.
        """
        tag = data[0]
        payload = data[1:]

        if tag == 0x01:
            return self._zlib.decode(payload, length)
        else:
            return self._ans.decode(payload, length)


# ---------------------------------------------------------------------------
# CompressedPolarQuant
# ---------------------------------------------------------------------------


class CompressedPolarQuant(PolarQuant):
    """PolarQuant with entropy-coded index storage.

    Same quantize/dequantize as PolarQuant, but stores indices in
    entropy-coded form for improved compression. The actual compression
    ratio includes the entropy coding gain.

    Args:
        d: Head dimension.
        bits: Bits per coordinate.
        use_entropy_coding: If True, apply entropy coding to indices.
        entropy_backend: Encoder backend ('ans', 'zlib', or 'auto').
        seed: Random seed.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        bits: int,
        use_entropy_coding: bool = True,
        entropy_backend: str = "auto",
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__(d=d, bits=bits, seed=seed, device=device)
        self.use_entropy_coding = use_entropy_coding
        self._entropy_encoder: Optional[EntropyEncoder] = None

        if use_entropy_coding:
            self._entropy_encoder = EntropyEncoder(
                self.codebook, backend=entropy_backend
            )

    def compress_indices(self, indices: torch.Tensor) -> bytes:
        """Compress quantized indices using entropy coding.

        Args:
            indices: Integer tensor from quantize().

        Returns:
            Compressed bytes.
        """
        if not self.use_entropy_coding or self._entropy_encoder is None:
            # Fallback: raw bytes
            return indices.reshape(-1).cpu().numpy().astype(np.uint8).tobytes()
        return self._entropy_encoder.encode(indices)

    def decompress_indices(
        self, data: bytes, shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Decompress indices from entropy-coded bytes.

        Args:
            data: Compressed bytes from compress_indices().
            shape: Original shape of the indices tensor.

        Returns:
            Integer tensor of shape `shape`.
        """
        if not self.use_entropy_coding or self._entropy_encoder is None:
            flat = np.frombuffer(data, dtype=np.uint8)
            return torch.from_numpy(flat.astype(np.int64)).reshape(shape)

        flat = self._entropy_encoder.decode(data, length=math.prod(shape))
        return flat.reshape(shape)

    def compression_stats(
        self, indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Report compression statistics.

        Args:
            indices: Optional indices tensor to measure empirical entropy.

        Returns:
            Dict with allocated_bits, effective_bits, compression_ratio,
            and optional empirical_entropy.
        """
        allocated = float(self.bits)
        effective = allocated

        if self.use_entropy_coding and self._entropy_encoder is not None:
            effective = self._entropy_encoder.compressed_bits_per_symbol

        stats = {
            "allocated_bits": allocated,
            "effective_bits_per_symbol": effective,
            "compression_ratio": allocated / effective if effective > 0 else 1.0,
            "savings_pct": (1.0 - effective / allocated) * 100 if allocated > 0 else 0.0,
        }

        if indices is not None:
            stats["empirical_entropy"] = measure_index_entropy(
                indices, self.codebook.n_levels
            )

        return stats


# ---------------------------------------------------------------------------
# Convenience: sweep all bit-widths
# ---------------------------------------------------------------------------


def entropy_analysis_sweep(
    d: int = 128,
    bit_range: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
) -> List[Dict[str, float]]:
    """Analyze compression opportunity across bit-widths.

    For each bit-width, computes the Lloyd-Max codebook for dimension d,
    then measures the theoretical entropy and compression savings.

    Args:
        d: Head dimension (default 128).
        bit_range: Tuple of bit-widths to analyze.

    Returns:
        List of dicts, one per bit-width, with:
            bits, n_levels, theoretical_entropy, entropy_ratio, savings_pct,
            and per-symbol probabilities.
    """
    results = []
    for bits in bit_range:
        codebook = LloydMaxCodebook(d=d, bits=bits)
        probs = _symbol_probabilities(codebook)
        opp = compression_opportunity(codebook)

        results.append({
            "bits": bits,
            "n_levels": codebook.n_levels,
            "theoretical_entropy": opp["theoretical_entropy"],
            "allocated_bits": opp["allocated_bits"],
            "entropy_ratio": opp["entropy_ratio"],
            "savings_pct": opp["savings_pct"],
            "symbol_probabilities": probs.tolist(),
        })

    return results
