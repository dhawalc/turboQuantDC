"""Tests for SparseVAttention — attention-gated value dequantization.

Validates that sparse V dequantization produces equivalent results to
dense attention while skipping negligible positions for efficiency.

Categories:
    1. Correctness: sparse matches dense at short context
    2. Sparsity: ratio increases with context length
    3. Quality: cosine similarity preserved
    4. Threshold: lower threshold = more positions, higher quality
    5. Stats: correct tracking of sparsity metrics
    6. Shape: single query, batched queries, empty cache
    7. GPU: works on CUDA if available
"""

import math

import pytest
import torch

from turboquantdc.kv_cache import TurboQuantKVCache
from turboquantdc.sparse_v import SparseVAttention, sparse_attention


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
D = 64  # Smaller dimension for fast tests (codebook precomputation)
BITS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_cache(
    n_tokens: int,
    d: int = D,
    bits: int = BITS,
    seed: int = SEED,
    device: str = "cpu",
    query_seed: int = 99,
    n_similar: int = 3,
    similar_norm: float = 1.0,
) -> tuple:
    """Build a cache with realistic attention patterns.

    Creates n_tokens key/value pairs where n_similar keys are close to a
    reference query (high attention weight) and the rest are random (low
    weight after softmax). This mimics real LLM attention where only a
    few positions dominate.

    Args:
        similar_norm: Norm multiplier for the "important" keys. Higher
            values amplify the score gap between important and random
            positions, producing sharper attention and more sparsity.
            Default 1.0 (unit norm). Use 8-10 for sparsity tests since
            TurboQuant compressed inner products have quantization noise
            that flattens the softmax at low signal-to-noise ratios.

    Returns:
        (cache, queries) tuple. queries has shape (1, d).
    """
    torch.manual_seed(seed)
    cache = TurboQuantKVCache(d_key=d, d_value=d, bits=bits, seed=seed, device=device)

    # Generate a query direction
    torch.manual_seed(query_seed)
    query = torch.randn(1, d, device=device)
    query = query / query.norm(dim=-1, keepdim=True)

    # Generate keys: most are random, a few are similar to query
    torch.manual_seed(seed + 1)
    keys = torch.randn(n_tokens, d, device=device)
    values = torch.randn(n_tokens, d, device=device)

    # Make the first n_similar keys very similar to query (high dot product).
    # Apply similar_norm to amplify the score gap through the quantizer.
    if n_similar > 0 and n_tokens >= n_similar:
        noise = torch.randn(n_similar, d, device=device) * 0.05
        keys[:n_similar] = query.expand(n_similar, -1) + noise
        keys[:n_similar] = (
            keys[:n_similar]
            / keys[:n_similar].norm(dim=-1, keepdim=True)
            * similar_norm
        )

    # Normalize remaining keys to unit norm
    keys[n_similar:] = keys[n_similar:] / keys[n_similar:].norm(dim=-1, keepdim=True)

    # Append one at a time (as the cache API is designed)
    for i in range(n_tokens):
        cache.append(keys[i], values[i])

    return cache, query


def dense_attention(cache: TurboQuantKVCache, queries: torch.Tensor, scale: float = None):
    """Reference dense attention: dequantize ALL values, compute weighted sum."""
    if queries.dim() == 1:
        queries = queries.unsqueeze(0)

    if cache.seq_len == 0:
        return torch.zeros(queries.shape[0], cache.d_value, device=queries.device)

    scores = cache.attention_scores(queries)
    if scale is None:
        scale = 1.0 / math.sqrt(cache.d_key)
    weights = torch.softmax(scores * scale, dim=-1)
    values = cache.get_values()
    return weights @ values


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors (flattened)."""
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-10)).item()


# ===========================================================================
# Tests
# ===========================================================================


class TestSparseVMatchesDense:
    """At short context, sparse V should match dense exactly (all weights significant)."""

    def test_sparse_v_matches_dense(self):
        """With 32 tokens, all positions should be significant at threshold=1e-6.
        Sparse output should closely match dense output."""
        n_tokens = 32
        cache, query = make_cache(n_tokens, n_similar=3)

        # Dense reference
        dense_out = dense_attention(cache, query)

        # Sparse with very low threshold (should decode nearly everything)
        sv = SparseVAttention(cache, threshold=1e-6)
        sparse_out = sv.attend(query)

        # At short context, softmax weights are more distributed, so most
        # positions have weight > 1e-6. The outputs should be very close.
        sim = cosine_sim(sparse_out, dense_out)
        assert sim > 0.99, (
            f"Sparse V should closely match dense at short context. "
            f"Got cosine similarity {sim:.6f}"
        )


class TestSparseVSparsityIncreasesWithContext:
    """Sparsity should increase with longer context.

    TurboQuant compressed inner products have quantization noise that
    flattens the softmax distribution at the default 1/sqrt(d) scale.
    To produce realistic sparsity in tests we amplify the "important"
    key norms (similar_norm=10) and use a scale factor of 1.0 instead
    of 1/sqrt(d). This mimics real LLM attention where a few keys
    dominate and the softmax concentrates on them.
    """

    SCALE = 1.0  # Sharper softmax than 1/sqrt(d) for test visibility

    def test_sparsity_short_context(self):
        """At 64 tokens with n_similar=3, some sparsity expected."""
        cache, query = make_cache(64, n_similar=3, similar_norm=10.0)
        sv = SparseVAttention(cache, threshold=1e-3)
        sv.attend(query, scale=self.SCALE)
        assert sv.last_stats["total_positions"] == 64

    def test_sparsity_medium_context(self):
        """At 512 tokens with only 3 high-norm similar keys, most positions
        should fall below the threshold."""
        cache, query = make_cache(512, n_similar=3, similar_norm=10.0)
        sv = SparseVAttention(cache, threshold=1e-3)
        sv.attend(query, scale=self.SCALE)
        ratio = sv.last_stats["sparsity_ratio"]
        assert ratio > 0.3, (
            f"Expected sparsity > 0.3 at 512 tokens with threshold=1e-3 "
            f"and similar_norm=10 and scale=1.0, got {ratio:.4f}"
        )

    def test_sparsity_increases(self):
        """Sparsity at 512 tokens should be higher than at 64 tokens.
        More random positions = more mass below threshold."""
        # Short context
        cache_short, query_short = make_cache(
            64, n_similar=3, query_seed=99, similar_norm=10.0
        )
        sv_short = SparseVAttention(cache_short, threshold=1e-3)
        sv_short.attend(query_short, scale=self.SCALE)
        sparsity_short = sv_short.last_stats["sparsity_ratio"]

        # Longer context (same query seed for fair comparison)
        cache_long, query_long = make_cache(
            512, n_similar=3, query_seed=99, similar_norm=10.0
        )
        sv_long = SparseVAttention(cache_long, threshold=1e-3)
        sv_long.attend(query_long, scale=self.SCALE)
        sparsity_long = sv_long.last_stats["sparsity_ratio"]

        assert sparsity_long > sparsity_short, (
            f"Sparsity should increase with context. "
            f"Short: {sparsity_short:.4f}, Long: {sparsity_long:.4f}"
        )


class TestSparseVQualityPreserved:
    """Cosine similarity between sparse and dense output should be very high."""

    def test_quality_preserved(self):
        """At 256 tokens, sparse V should produce output with cosine sim > 0.999
        vs dense attention."""
        cache, query = make_cache(256, n_similar=5)
        dense_out = dense_attention(cache, query)
        sv = SparseVAttention(cache, threshold=1e-6)
        sparse_out = sv.attend(query)

        sim = cosine_sim(sparse_out, dense_out)
        assert sim > 0.999, (
            f"Quality should be preserved. Got cosine sim {sim:.6f}, expected > 0.999"
        )


class TestSparseVThresholdEffect:
    """Lower threshold should decode more positions."""

    def test_threshold_effect(self):
        """Decreasing threshold should monotonically increase positions decoded."""
        cache, query = make_cache(256, n_similar=3)

        thresholds = [1e-2, 1e-4, 1e-6, 1e-8]
        positions_decoded = []

        for thr in thresholds:
            sv = SparseVAttention(cache, threshold=thr)
            sv.attend(query)
            positions_decoded.append(sv.last_stats["positions_decoded"])

        # Each lower threshold should decode >= as many positions
        for i in range(len(positions_decoded) - 1):
            assert positions_decoded[i] <= positions_decoded[i + 1], (
                f"Lower threshold should decode more positions. "
                f"threshold={thresholds[i]}: {positions_decoded[i]}, "
                f"threshold={thresholds[i+1]}: {positions_decoded[i+1]}"
            )


class TestSparseVStatsTracking:
    """Verify last_stats reports correct values."""

    def test_stats_fields_present(self):
        """last_stats should have sparsity_ratio, positions_decoded, total_positions."""
        cache, query = make_cache(64, n_similar=3)
        sv = SparseVAttention(cache, threshold=1e-6)
        sv.attend(query)

        assert "sparsity_ratio" in sv.last_stats
        assert "positions_decoded" in sv.last_stats
        assert "total_positions" in sv.last_stats

    def test_stats_consistency(self):
        """sparsity_ratio should equal 1 - positions_decoded / total_positions."""
        cache, query = make_cache(128, n_similar=3)
        sv = SparseVAttention(cache, threshold=1e-4)
        sv.attend(query)

        n_dec = sv.last_stats["positions_decoded"]
        n_tot = sv.last_stats["total_positions"]
        expected_ratio = 1.0 - (n_dec / max(n_tot, 1))

        assert abs(sv.last_stats["sparsity_ratio"] - expected_ratio) < 1e-10, (
            f"Sparsity ratio inconsistent: reported {sv.last_stats['sparsity_ratio']}, "
            f"expected {expected_ratio}"
        )

    def test_stats_total_positions(self):
        """total_positions should match cache.seq_len."""
        n = 100
        cache, query = make_cache(n, n_similar=3)
        sv = SparseVAttention(cache, threshold=1e-6)
        sv.attend(query)
        assert sv.last_stats["total_positions"] == n


class TestSparseVSingleQuery:
    """Works with 1-D query input."""

    def test_single_query(self):
        """attend() with a 1-D query should return a 1-D output."""
        cache, query_2d = make_cache(64, n_similar=3)
        query_1d = query_2d.squeeze(0)  # (d,)

        sv = SparseVAttention(cache, threshold=1e-6)
        output = sv.attend(query_1d)

        assert output.dim() == 1, f"Expected 1-D output, got shape {output.shape}"
        assert output.shape[0] == D, f"Expected dim {D}, got {output.shape[0]}"


class TestSparseVBatchQueries:
    """Works with batched queries."""

    def test_batch_queries(self):
        """attend() with (n_queries, d) input should return (n_queries, d_value)."""
        cache, _ = make_cache(128, n_similar=3)

        torch.manual_seed(200)
        queries = torch.randn(5, D)

        sv = SparseVAttention(cache, threshold=1e-6)
        output = sv.attend(queries)

        assert output.shape == (5, D), f"Expected (5, {D}), got {output.shape}"

    def test_batch_matches_individual(self):
        """Batched queries should give the same results as individual queries."""
        cache, _ = make_cache(64, n_similar=3)

        torch.manual_seed(300)
        queries = torch.randn(3, D)

        # Batched
        sv = SparseVAttention(cache, threshold=1e-6)
        batched_out = sv.attend(queries)

        # Individual
        individual_outs = []
        for i in range(3):
            sv_i = SparseVAttention(cache, threshold=1e-6)
            out_i = sv_i.attend(queries[i])
            individual_outs.append(out_i)
        individual_out = torch.stack(individual_outs, dim=0)

        # They should be close (not necessarily identical due to the union
        # of significant positions differing between batch and individual)
        sim = cosine_sim(batched_out, individual_out)
        assert sim > 0.99, (
            f"Batched should match individual queries. Got cosine sim {sim:.6f}"
        )


class TestSparseVEmptyCache:
    """Handles empty cache gracefully."""

    def test_empty_cache_2d_query(self):
        """attend() on empty cache should return zeros with correct shape."""
        cache = TurboQuantKVCache(d_key=D, d_value=D, bits=BITS, seed=SEED)
        query = torch.randn(2, D)

        sv = SparseVAttention(cache, threshold=1e-6)
        output = sv.attend(query)

        assert output.shape == (2, D)
        assert (output == 0).all(), "Empty cache should produce zero output"

    def test_empty_cache_1d_query(self):
        """attend() on empty cache with 1-D query returns 1-D zeros."""
        cache = TurboQuantKVCache(d_key=D, d_value=D, bits=BITS, seed=SEED)
        query = torch.randn(D)

        sv = SparseVAttention(cache, threshold=1e-6)
        output = sv.attend(query)

        assert output.shape == (D,)
        assert (output == 0).all()

    def test_empty_cache_stats(self):
        """Stats for empty cache should show zero positions."""
        cache = TurboQuantKVCache(d_key=D, d_value=D, bits=BITS, seed=SEED)
        query = torch.randn(1, D)

        sv = SparseVAttention(cache, threshold=1e-6)
        sv.attend(query)

        assert sv.last_stats["total_positions"] == 0
        assert sv.last_stats["positions_decoded"] == 0
        assert sv.last_stats["sparsity_ratio"] == 0.0


class TestSparseVThresholdZero:
    """threshold=0 should decode everything (equivalent to dense)."""

    def test_threshold_zero_decodes_all(self):
        """With threshold=0, all positions should be decoded."""
        cache, query = make_cache(64, n_similar=3)

        sv = SparseVAttention(cache, threshold=0.0)
        sv.attend(query)

        assert sv.last_stats["positions_decoded"] == 64, (
            f"threshold=0 should decode all 64 positions, "
            f"got {sv.last_stats['positions_decoded']}"
        )

    def test_threshold_zero_matches_dense(self):
        """threshold=0 output should be very close to dense attention."""
        cache, query = make_cache(128, n_similar=5)

        dense_out = dense_attention(cache, query)

        sv = SparseVAttention(cache, threshold=0.0)
        sparse_out = sv.attend(query)

        sim = cosine_sim(sparse_out, dense_out)
        assert sim > 0.999, (
            f"threshold=0 should match dense. Got cosine sim {sim:.6f}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSparseVGPU:
    """Works on CUDA if available."""

    def test_gpu_basic(self):
        """Sparse V attention runs on GPU and produces reasonable output."""
        device = "cuda"
        cache, query = make_cache(64, n_similar=3, device=device)

        sv = SparseVAttention(cache, threshold=1e-6)
        output = sv.attend(query)

        assert output.device.type == "cuda"
        assert output.shape == (1, D)
        assert not torch.isnan(output).any(), "GPU output contains NaN"

    def test_gpu_matches_cpu(self):
        """GPU and CPU should produce the same results on the same data.

        We build the cache on CPU first (so random sequences are identical),
        then rebuild on GPU from the same tensors to ensure an apples-to-apples
        comparison.
        """
        n_tokens = 64
        n_similar = 3

        # Generate data on CPU for reproducibility
        torch.manual_seed(SEED + 1)
        keys = torch.randn(n_tokens, D)
        values = torch.randn(n_tokens, D)
        torch.manual_seed(99)
        query = torch.randn(1, D)
        query = query / query.norm(dim=-1, keepdim=True)

        noise = torch.randn(n_similar, D) * 0.05
        keys[:n_similar] = query.expand(n_similar, -1) + noise
        keys[:n_similar] = keys[:n_similar] / keys[:n_similar].norm(dim=-1, keepdim=True)
        keys[n_similar:] = keys[n_similar:] / keys[n_similar:].norm(dim=-1, keepdim=True)

        # Build CPU cache
        cache_cpu = TurboQuantKVCache(d_key=D, d_value=D, bits=BITS, seed=SEED, device="cpu")
        for i in range(n_tokens):
            cache_cpu.append(keys[i], values[i])
        sv_cpu = SparseVAttention(cache_cpu, threshold=1e-6)
        out_cpu = sv_cpu.attend(query)

        # Build GPU cache from same data
        cache_gpu = TurboQuantKVCache(d_key=D, d_value=D, bits=BITS, seed=SEED, device="cuda")
        keys_gpu = keys.cuda()
        values_gpu = values.cuda()
        query_gpu = query.cuda()
        for i in range(n_tokens):
            cache_gpu.append(keys_gpu[i], values_gpu[i])
        sv_gpu = SparseVAttention(cache_gpu, threshold=1e-6)
        out_gpu = sv_gpu.attend(query_gpu)

        sim = cosine_sim(out_cpu, out_gpu.cpu())
        assert sim > 0.99, f"GPU and CPU should match. Got cosine sim {sim:.6f}"


class TestSparseAttentionFunctionalAPI:
    """Test the sparse_attention() convenience function."""

    def test_functional_api(self):
        """sparse_attention() should produce the same output as class API."""
        cache, query = make_cache(64, n_similar=3)

        sv = SparseVAttention(cache, threshold=1e-6)
        class_out = sv.attend(query)

        func_out = sparse_attention(cache, query, threshold=1e-6)

        sim = cosine_sim(class_out, func_out)
        assert sim > 0.9999, f"Functional API should match class API. Got {sim:.6f}"
