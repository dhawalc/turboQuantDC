"""Tests for entropy coding of quantized indices.

Tests verify:
    1. Round-trip encode/decode produces identical indices
    2. Compressed size < uncompressed size
    3. Empirical entropy matches theoretical prediction (within 5%)
    4. CompressedPolarQuant integration
    5. Entropy analysis sweep produces correct results
"""

import math

import numpy as np
import pytest
import torch

from turboquantdc.codebook import LloydMaxCodebook
from turboquantdc.entropy_coding import (
    ANSEncoder,
    CompressedPolarQuant,
    EntropyEncoder,
    ZlibEncoder,
    _symbol_probabilities,
    compression_opportunity,
    entropy_analysis_sweep,
    measure_index_entropy,
    theoretical_index_entropy,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DIM = 128
SIGMA = 1.0 / math.sqrt(DEFAULT_DIM)
SEED = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def codebook_3bit():
    """3-bit Lloyd-Max codebook for d=128."""
    return LloydMaxCodebook(d=DEFAULT_DIM, bits=3)


@pytest.fixture
def codebook_2bit():
    """2-bit Lloyd-Max codebook for d=128."""
    return LloydMaxCodebook(d=DEFAULT_DIM, bits=2)


@pytest.fixture
def codebook_4bit():
    """4-bit Lloyd-Max codebook for d=128."""
    return LloydMaxCodebook(d=DEFAULT_DIM, bits=4)


@pytest.fixture
def sample_indices_3bit(codebook_3bit):
    """Quantized indices from N(0, 1/d) samples using 3-bit codebook."""
    torch.manual_seed(SEED)
    samples = torch.randn(10000) * SIGMA
    return codebook_3bit.quantize(samples)


@pytest.fixture
def sample_indices_2d(codebook_3bit):
    """2D batch of quantized indices (batch=50, d=128)."""
    torch.manual_seed(SEED)
    samples = torch.randn(50, DEFAULT_DIM) * SIGMA
    return codebook_3bit.quantize(samples)


# ---------------------------------------------------------------------------
# Tests: symbol probabilities
# ---------------------------------------------------------------------------
class TestSymbolProbabilities:
    """Verify that computed symbol probabilities are valid and match theory."""

    def test_probabilities_sum_to_one(self, codebook_3bit):
        """Symbol probabilities must sum to 1."""
        probs = _symbol_probabilities(codebook_3bit)
        assert abs(probs.sum() - 1.0) < 1e-6, (
            f"Probabilities sum to {probs.sum()}, expected 1.0"
        )

    def test_probabilities_all_positive(self, codebook_3bit):
        """All symbol probabilities must be positive."""
        probs = _symbol_probabilities(codebook_3bit)
        assert np.all(probs > 0), "All probabilities must be positive"

    def test_middle_centroids_most_probable(self, codebook_3bit):
        """Middle centroids should have highest probability (Gaussian peak)."""
        probs = _symbol_probabilities(codebook_3bit)
        n = len(probs)
        mid = n // 2
        # Middle two should be the most probable
        mid_prob = probs[mid - 1] + probs[mid]
        edge_prob = probs[0] + probs[-1]
        assert mid_prob > edge_prob, (
            f"Middle probability {mid_prob} should exceed edge probability {edge_prob}"
        )

    def test_symmetric_probabilities(self, codebook_3bit):
        """Probabilities should be symmetric: P(i) = P(n-1-i)."""
        probs = _symbol_probabilities(codebook_3bit)
        n = len(probs)
        for i in range(n // 2):
            assert abs(probs[i] - probs[n - 1 - i]) < 1e-4, (
                f"P({i})={probs[i]:.6f} != P({n-1-i})={probs[n-1-i]:.6f}"
            )

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_probabilities_match_empirical(self, bits):
        """Theoretical probabilities should match empirical frequencies."""
        cb = LloydMaxCodebook(d=DEFAULT_DIM, bits=bits)
        probs = _symbol_probabilities(cb)

        # Generate large sample and measure empirical frequencies
        torch.manual_seed(SEED)
        samples = torch.randn(100000) * SIGMA
        indices = cb.quantize(samples)
        counts = torch.bincount(indices, minlength=cb.n_levels).float()
        empirical = (counts / counts.sum()).numpy()

        # Should match within 2% for 100k samples
        max_diff = np.abs(probs - empirical).max()
        assert max_diff < 0.02, (
            f"Max probability difference: {max_diff:.4f} (expected < 0.02)"
        )


# ---------------------------------------------------------------------------
# Tests: entropy measurement
# ---------------------------------------------------------------------------
class TestEntropyMeasurement:
    """Verify Shannon entropy measurement and theoretical predictions."""

    def test_entropy_less_than_bits(self, codebook_3bit, sample_indices_3bit):
        """Empirical entropy should be less than allocated bits."""
        entropy = measure_index_entropy(sample_indices_3bit, codebook_3bit.n_levels)
        assert entropy < codebook_3bit.bits, (
            f"Entropy {entropy:.3f} should be < {codebook_3bit.bits} bits"
        )

    def test_entropy_positive(self, codebook_3bit, sample_indices_3bit):
        """Entropy should be positive."""
        entropy = measure_index_entropy(sample_indices_3bit, codebook_3bit.n_levels)
        assert entropy > 0, "Entropy should be positive"

    def test_theoretical_entropy_less_than_bits(self, codebook_3bit):
        """Theoretical entropy should be less than allocated bits."""
        entropy = theoretical_index_entropy(codebook_3bit)
        assert entropy < codebook_3bit.bits, (
            f"Theoretical entropy {entropy:.3f} should be < {codebook_3bit.bits}"
        )

    def test_empirical_matches_theoretical(self, codebook_3bit, sample_indices_3bit):
        """Empirical entropy should match theoretical within 5%."""
        empirical = measure_index_entropy(
            sample_indices_3bit, codebook_3bit.n_levels
        )
        theoretical = theoretical_index_entropy(codebook_3bit)

        rel_diff = abs(empirical - theoretical) / theoretical
        assert rel_diff < 0.05, (
            f"Empirical entropy {empirical:.4f} differs from "
            f"theoretical {theoretical:.4f} by {rel_diff*100:.1f}%"
        )

    @pytest.mark.parametrize("bits", [2, 3, 4, 5])
    def test_entropy_decreases_ratio_with_bits(self, bits):
        """Entropy ratio (entropy/bits) should generally be < 1.0."""
        cb = LloydMaxCodebook(d=DEFAULT_DIM, bits=bits)
        entropy = theoretical_index_entropy(cb)
        ratio = entropy / bits
        assert ratio < 1.0, (
            f"Entropy ratio {ratio:.4f} should be < 1.0 at {bits}-bit"
        )

    def test_uniform_distribution_has_max_entropy(self):
        """Uniform indices should have entropy = log2(n_levels)."""
        n = 8
        indices = torch.randint(0, n, (10000,))
        entropy = measure_index_entropy(indices, n)
        expected = math.log2(n)
        assert abs(entropy - expected) < 0.05, (
            f"Uniform entropy {entropy:.4f} should be ~{expected:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: compression opportunity
# ---------------------------------------------------------------------------
class TestCompressionOpportunity:
    """Verify compression opportunity analysis."""

    def test_savings_positive(self, codebook_3bit):
        """Savings percentage should be positive for Gaussian distribution."""
        opp = compression_opportunity(codebook_3bit)
        assert opp["savings_pct"] > 0, (
            f"Savings should be positive, got {opp['savings_pct']:.1f}%"
        )

    def test_3bit_savings_in_expected_range(self, codebook_3bit):
        """3-bit savings should be roughly 3-15% for d=128.

        The Lloyd-Max codebook is well-adapted to the Gaussian distribution,
        so the probabilities are not extremely skewed. The savings are modest
        but still meaningful (entropy ~2.82 bits vs 3.0 allocated).
        """
        opp = compression_opportunity(codebook_3bit)
        assert 3.0 < opp["savings_pct"] < 15.0, (
            f"3-bit savings {opp['savings_pct']:.1f}% outside expected 3-15% range"
        )

    def test_entropy_ratio_below_one(self, codebook_3bit):
        """Entropy ratio should be < 1.0 (room to compress)."""
        opp = compression_opportunity(codebook_3bit)
        assert opp["entropy_ratio"] < 1.0

    def test_allocated_bits_correct(self, codebook_3bit):
        """Allocated bits should match codebook bit-width."""
        opp = compression_opportunity(codebook_3bit)
        assert opp["allocated_bits"] == 3.0


# ---------------------------------------------------------------------------
# Tests: ANS encoder round-trip
# ---------------------------------------------------------------------------
class TestANSEncoder:
    """Test ANS encoder/decoder correctness and compression."""

    def test_roundtrip_small(self, codebook_3bit):
        """Encode then decode should produce identical indices (small)."""
        enc = ANSEncoder(codebook_3bit)
        indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 3, 4])
        compressed = enc.encode(indices)
        decoded = enc.decode(compressed)
        assert torch.equal(indices, decoded), (
            f"Round-trip failed: {indices.tolist()} != {decoded.tolist()}"
        )

    def test_roundtrip_gaussian_sample(self, codebook_3bit, sample_indices_3bit):
        """Round-trip with realistic Gaussian-distributed indices."""
        enc = ANSEncoder(codebook_3bit)
        # Use a reasonable subset for speed
        subset = sample_indices_3bit[:1000]
        compressed = enc.encode(subset)
        decoded = enc.decode(compressed)
        assert torch.equal(subset, decoded), "Round-trip failed on Gaussian sample"

    def test_compressed_smaller(self, codebook_3bit, sample_indices_3bit):
        """Compressed size should be less than raw size."""
        enc = ANSEncoder(codebook_3bit)
        subset = sample_indices_3bit[:5000]
        compressed = enc.encode(subset)

        raw_bits = len(subset) * codebook_3bit.bits
        raw_bytes = (raw_bits + 7) // 8
        compressed_bytes = len(compressed)

        assert compressed_bytes < raw_bytes, (
            f"Compressed {compressed_bytes} bytes >= raw {raw_bytes} bytes"
        )

    def test_compressed_bps_less_than_bits(self, codebook_3bit):
        """Compressed bits per symbol should be less than allocated bits."""
        enc = ANSEncoder(codebook_3bit)
        assert enc.compressed_bits_per_symbol < codebook_3bit.bits

    def test_all_same_symbol(self, codebook_3bit):
        """Encoding all-same symbols should compress very well."""
        enc = ANSEncoder(codebook_3bit)
        indices = torch.full((100,), 3, dtype=torch.long)
        compressed = enc.encode(indices)
        decoded = enc.decode(compressed)
        assert torch.equal(indices, decoded)

    def test_roundtrip_2bit(self, codebook_2bit):
        """Round-trip at 2-bit quantization."""
        enc = ANSEncoder(codebook_2bit)
        torch.manual_seed(SEED)
        samples = torch.randn(500) * SIGMA
        indices = codebook_2bit.quantize(samples)
        compressed = enc.encode(indices)
        decoded = enc.decode(compressed)
        assert torch.equal(indices, decoded)


# ---------------------------------------------------------------------------
# Tests: Zlib encoder round-trip
# ---------------------------------------------------------------------------
class TestZlibEncoder:
    """Test zlib-based encoder correctness."""

    def test_roundtrip(self, codebook_3bit, sample_indices_3bit):
        """Encode then decode should produce identical indices."""
        enc = ZlibEncoder(codebook_3bit)
        subset = sample_indices_3bit[:5000]
        compressed = enc.encode(subset)
        decoded = enc.decode(compressed)
        assert torch.equal(subset, decoded), "Zlib round-trip failed"

    def test_compressed_smaller(self, codebook_3bit, sample_indices_3bit):
        """Compressed size should be less than raw size."""
        enc = ZlibEncoder(codebook_3bit)
        compressed = enc.encode(sample_indices_3bit)
        raw_bytes = len(sample_indices_3bit) * 1  # 1 byte per index (uint8)
        compressed_bytes = len(compressed)
        assert compressed_bytes < raw_bytes, (
            f"Compressed {compressed_bytes} >= raw {raw_bytes}"
        )

    def test_roundtrip_2d(self, codebook_3bit, sample_indices_2d):
        """Round-trip with 2D batch of indices."""
        enc = ZlibEncoder(codebook_3bit)
        flat = sample_indices_2d.reshape(-1)
        compressed = enc.encode(flat)
        decoded = enc.decode(compressed)
        assert torch.equal(flat, decoded)


# ---------------------------------------------------------------------------
# Tests: unified EntropyEncoder
# ---------------------------------------------------------------------------
class TestEntropyEncoder:
    """Test the unified EntropyEncoder interface."""

    @pytest.mark.parametrize("backend", ["ans", "zlib", "auto"])
    def test_roundtrip(self, codebook_3bit, backend):
        """Round-trip works with all backends."""
        enc = EntropyEncoder(codebook_3bit, backend=backend)
        torch.manual_seed(SEED)
        samples = torch.randn(500) * SIGMA
        indices = codebook_3bit.quantize(samples)
        compressed = enc.encode(indices)
        decoded = enc.decode(compressed)
        assert torch.equal(indices, decoded), (
            f"Round-trip failed with backend={backend}"
        )

    @pytest.mark.parametrize("backend", ["ans", "zlib", "auto"])
    def test_compressed_bps(self, codebook_3bit, backend):
        """Compressed bits per symbol should be less than allocated."""
        enc = EntropyEncoder(codebook_3bit, backend=backend)
        assert enc.compressed_bits_per_symbol < codebook_3bit.bits

    def test_auto_uses_zlib_for_large(self, codebook_3bit):
        """Auto backend should use zlib for large batches."""
        enc = EntropyEncoder(codebook_3bit, backend="auto")
        large = torch.randint(0, 8, (20000,))
        compressed = enc.encode(large)
        # Check the tag byte
        assert compressed[0] == 0x01, "Auto should use zlib (tag 0x01) for large batch"

    def test_auto_uses_ans_for_small(self, codebook_3bit):
        """Auto backend should use ANS for small batches."""
        enc = EntropyEncoder(codebook_3bit, backend="auto")
        small = torch.randint(0, 8, (100,))
        compressed = enc.encode(small)
        assert compressed[0] == 0x00, "Auto should use ANS (tag 0x00) for small batch"


# ---------------------------------------------------------------------------
# Tests: CompressedPolarQuant
# ---------------------------------------------------------------------------
class TestCompressedPolarQuant:
    """Test PolarQuant with entropy coding integration."""

    def test_quantize_same_as_polarquant(self):
        """Quantize should produce same indices as plain PolarQuant."""
        from turboquantdc.polarquant import PolarQuant

        pq = PolarQuant(d=DEFAULT_DIM, bits=3, seed=SEED)
        cpq = CompressedPolarQuant(d=DEFAULT_DIM, bits=3, seed=SEED)

        torch.manual_seed(77)
        x = torch.randn(100, DEFAULT_DIM)
        x = x / x.norm(dim=1, keepdim=True)

        indices_pq = pq.quantize(x)
        indices_cpq = cpq.quantize(x)
        assert torch.equal(indices_pq, indices_cpq)

    def test_compress_decompress_roundtrip(self):
        """Compress then decompress should recover identical indices."""
        cpq = CompressedPolarQuant(d=DEFAULT_DIM, bits=3, seed=SEED)

        torch.manual_seed(77)
        x = torch.randn(100, DEFAULT_DIM)
        x = x / x.norm(dim=1, keepdim=True)

        indices = cpq.quantize(x)
        compressed = cpq.compress_indices(indices)
        recovered = cpq.decompress_indices(compressed, indices.shape)
        assert torch.equal(indices, recovered)

    def test_dequantize_from_decompressed(self):
        """Full pipeline: quantize -> compress -> decompress -> dequantize."""
        cpq = CompressedPolarQuant(d=DEFAULT_DIM, bits=3, seed=SEED)

        torch.manual_seed(77)
        x = torch.randn(100, DEFAULT_DIM)
        x = x / x.norm(dim=1, keepdim=True)

        indices = cpq.quantize(x)
        compressed = cpq.compress_indices(indices)
        recovered = cpq.decompress_indices(compressed, indices.shape)
        x_hat = cpq.dequantize(recovered)

        # Should be identical to dequantizing the original indices
        x_hat_direct = cpq.dequantize(indices)
        assert torch.allclose(x_hat, x_hat_direct, atol=1e-6)

    def test_compression_stats(self):
        """Compression stats should report savings."""
        cpq = CompressedPolarQuant(d=DEFAULT_DIM, bits=3, seed=SEED)

        torch.manual_seed(77)
        x = torch.randn(100, DEFAULT_DIM)
        x = x / x.norm(dim=1, keepdim=True)
        indices = cpq.quantize(x)

        stats = cpq.compression_stats(indices)
        assert stats["allocated_bits"] == 3.0
        assert stats["effective_bits_per_symbol"] < 3.0
        assert stats["compression_ratio"] > 1.0
        assert stats["savings_pct"] > 0

    def test_no_entropy_coding(self):
        """With use_entropy_coding=False, should use raw bytes."""
        cpq = CompressedPolarQuant(
            d=DEFAULT_DIM, bits=3, seed=SEED, use_entropy_coding=False
        )

        torch.manual_seed(77)
        x = torch.randn(50, DEFAULT_DIM)
        x = x / x.norm(dim=1, keepdim=True)

        indices = cpq.quantize(x)
        compressed = cpq.compress_indices(indices)
        recovered = cpq.decompress_indices(compressed, indices.shape)
        assert torch.equal(indices, recovered)

        # Raw bytes: 1 byte per index
        expected_size = indices.numel()
        assert len(compressed) == expected_size


# ---------------------------------------------------------------------------
# Tests: entropy analysis sweep
# ---------------------------------------------------------------------------
class TestEntropySweep:
    """Test the analysis sweep across bit-widths."""

    def test_sweep_returns_all_widths(self):
        """Sweep should return results for all requested bit-widths."""
        results = entropy_analysis_sweep(d=DEFAULT_DIM, bit_range=(2, 3, 4))
        assert len(results) == 3
        assert [r["bits"] for r in results] == [2, 3, 4]

    def test_sweep_savings_positive(self):
        """All bit-widths should have positive savings."""
        results = entropy_analysis_sweep(d=DEFAULT_DIM, bit_range=(2, 3, 4, 5))
        for r in results:
            assert r["savings_pct"] > 0, (
                f"Savings should be positive at {r['bits']}-bit"
            )

    def test_sweep_entropy_less_than_bits(self):
        """Theoretical entropy should be less than allocated bits."""
        results = entropy_analysis_sweep(d=DEFAULT_DIM, bit_range=(2, 3, 4, 5, 6, 7, 8))
        for r in results:
            assert r["theoretical_entropy"] < r["allocated_bits"], (
                f"Entropy {r['theoretical_entropy']:.3f} should be < "
                f"{r['allocated_bits']} at {r['bits']}-bit"
            )

    def test_sweep_probabilities_valid(self):
        """Symbol probabilities should sum to ~1 and be positive."""
        results = entropy_analysis_sweep(d=DEFAULT_DIM, bit_range=(3,))
        probs = results[0]["symbol_probabilities"]
        assert abs(sum(probs) - 1.0) < 1e-6
        assert all(p > 0 for p in probs)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_symbol(self, codebook_3bit):
        """Encoding a single symbol should work."""
        enc = EntropyEncoder(codebook_3bit, backend="zlib")
        indices = torch.tensor([4])
        compressed = enc.encode(indices)
        decoded = enc.decode(compressed)
        assert torch.equal(indices, decoded)

    def test_empty_tensor_entropy(self):
        """Empty tensor should have 0 entropy."""
        entropy = measure_index_entropy(torch.tensor([], dtype=torch.long), 8)
        assert entropy == 0.0

    def test_large_batch_roundtrip(self, codebook_3bit):
        """Large batch should round-trip correctly."""
        enc = EntropyEncoder(codebook_3bit, backend="zlib")
        torch.manual_seed(SEED)
        samples = torch.randn(100000) * SIGMA
        indices = codebook_3bit.quantize(samples)
        compressed = enc.encode(indices)
        decoded = enc.decode(compressed)
        assert torch.equal(indices, decoded)
