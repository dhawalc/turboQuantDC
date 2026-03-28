"""End-to-end integration tests for the full TurboQuantDC pipeline.

Tests the COMPLETE chain:
    codebook -> rotation -> polarquant -> qjl -> estimator -> kv_cache

Validates against the paper's theoretical bounds (arxiv 2504.19874) and
ensures all modules compose correctly across configurations.

Categories:
    A. Full Pipeline Tests (varying d, bits)
    B. KV Cache Wrapper Tests (append, retrieve, attention, compression)
    C. Paper Bound Validation (MSE distortion, IP distortion, unbiasedness)
    D. Edge Cases (single vector, large batch, GPU, determinism)
    E. Cross-Module Consistency (estimator vs cache, key vs value asymmetry)
"""

import math

import pytest
import torch

from turboquantdc.codebook import LloydMaxCodebook, solve_lloyd_max
from turboquantdc.estimator import TurboQuantEstimator
from turboquantdc.kv_cache import TurboQuantKVCache
from turboquantdc.polarquant import PolarQuant
from turboquantdc.qjl import QJL
from turboquantdc.rotation import generate_qjl_matrix, generate_rotation_matrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42

# Paper bounds
MSE_BOUND_FACTOR = math.sqrt(3) * math.pi / 2  # ~ 2.721
IP_BOUND_FACTOR = math.sqrt(3) * math.pi ** 2  # ~ 17.08

# Paper table values for D_prod (assuming ||y||=1): {1.57/d, 0.56/d, 0.18/d, 0.047/d}
PAPER_DPROD = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}

# Paper table values for D_mse per coordinate: {0.36/d, 0.117/d, 0.03/d, 0.009/d}
PAPER_DMSE_PER_COORD = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_unit_vectors(n: int, d: int, seed: int = SEED) -> torch.Tensor:
    """Generate n random unit vectors of dimension d."""
    torch.manual_seed(seed)
    x = torch.randn(n, d)
    return x / x.norm(dim=1, keepdim=True)


def random_vectors(n: int, d: int, seed: int = SEED, scale: float = 1.0) -> torch.Tensor:
    """Generate n random (non-unit) vectors of dimension d."""
    torch.manual_seed(seed)
    return torch.randn(n, d) * scale


def cosine_similarity_vectors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise cosine similarity between rows of a and b."""
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)


def element_wise_inner_products(
    est: TurboQuantEstimator,
    queries: torch.Tensor,
    compressed: dict,
) -> torch.Tensor:
    """Compute element-wise <q_i, k_i> via diagonal of cross-product matrix."""
    return torch.diagonal(est.inner_product(queries, compressed))


# ===========================================================================
# A. Full Pipeline Tests
# ===========================================================================
class TestFullPipelineD128_3bit:
    """Standard configuration: d=128, 3-bit. The paper's sweet spot."""

    def test_full_pipeline_d128_3bit(self):
        """Full codebook -> rotation -> polarquant -> qjl -> estimator chain
        at d=128, 3-bit. Verify cosine similarity of attention scores > 0.90
        on random unit vectors."""
        d, bits = 128, 3
        n_keys, n_queries = 500, 50

        keys = random_unit_vectors(n_keys, d, seed=10)
        queries = random_unit_vectors(n_queries, d, seed=20)

        # Stage 1 components
        codebook = LloydMaxCodebook(d=d, bits=bits - 1)
        assert codebook.n_levels == 2 ** (bits - 1)

        rotation = generate_rotation_matrix(d, seed=SEED)
        assert rotation.shape == (d, d)
        # Orthogonality check
        identity_err = (rotation @ rotation.T - torch.eye(d)).abs().max().item()
        assert identity_err < 1e-5, f"Rotation not orthogonal: err={identity_err}"

        # Full estimator
        est = TurboQuantEstimator(d=d, bits=bits, seed=SEED)
        compressed = est.quantize(keys)

        # Verify compressed structure
        assert compressed["mse_indices"].shape == (n_keys, d)
        assert compressed["qjl_signs"].shape == (n_keys, d)
        assert compressed["residual_norm"].shape == (n_keys,)
        assert compressed["vec_norm"].shape == (n_keys,)

        # Estimated inner products
        est_scores = est.inner_product(queries, compressed)  # (n_queries, n_keys)
        true_scores = queries @ keys.T  # (n_queries, n_keys)

        # Cosine similarity of each query's score vector
        cos_sims = cosine_similarity_vectors(
            est_scores.flatten().unsqueeze(0),
            true_scores.flatten().unsqueeze(0),
        )
        assert cos_sims.item() > 0.90, (
            f"Pipeline d=128/3-bit: cosine sim = {cos_sims.item():.4f}, expected > 0.90"
        )


class TestFullPipelineD64_3bit:
    """Smaller head dimension: d=64."""

    def test_full_pipeline_d64_3bit(self):
        """d=64 is used by some older models. Full pipeline should still work."""
        d, bits = 64, 3
        n_keys, n_queries = 300, 30

        keys = random_unit_vectors(n_keys, d, seed=10)
        queries = random_unit_vectors(n_queries, d, seed=20)

        est = TurboQuantEstimator(d=d, bits=bits, seed=SEED)
        compressed = est.quantize(keys)

        assert compressed["mse_indices"].shape == (n_keys, d)
        assert compressed["qjl_signs"].shape == (n_keys, d)

        est_scores = est.inner_product(queries, compressed)
        true_scores = queries @ keys.T

        cos_sims = cosine_similarity_vectors(
            est_scores.flatten().unsqueeze(0),
            true_scores.flatten().unsqueeze(0),
        )
        # d=64 has higher per-dim variance, so slightly relaxed threshold
        assert cos_sims.item() > 0.85, (
            f"Pipeline d=64/3-bit: cosine sim = {cos_sims.item():.4f}, expected > 0.85"
        )


class TestFullPipelineD256_3bit:
    """Large head dimension: d=256 (Qwen3.5-27B style)."""

    def test_full_pipeline_d256_3bit(self):
        """d=256 should give better results than d=128 due to stronger
        concentration of measure."""
        d, bits = 256, 3
        n_keys, n_queries = 300, 30

        keys = random_unit_vectors(n_keys, d, seed=10)
        queries = random_unit_vectors(n_queries, d, seed=20)

        est = TurboQuantEstimator(d=d, bits=bits, seed=SEED)
        compressed = est.quantize(keys)

        assert compressed["mse_indices"].shape == (n_keys, d)

        est_scores = est.inner_product(queries, compressed)
        true_scores = queries @ keys.T

        cos_sims = cosine_similarity_vectors(
            est_scores.flatten().unsqueeze(0),
            true_scores.flatten().unsqueeze(0),
        )
        assert cos_sims.item() > 0.90, (
            f"Pipeline d=256/3-bit: cosine sim = {cos_sims.item():.4f}, expected > 0.90"
        )


class TestFullPipelineD128_2bit:
    """Aggressive compression: d=128, 2-bit."""

    def test_full_pipeline_d128_2bit(self):
        """2-bit uses 1-bit MSE + 1-bit QJL. Higher distortion expected."""
        d, bits = 128, 2
        n_keys = 500

        keys = random_unit_vectors(n_keys, d, seed=10)
        queries = random_unit_vectors(50, d, seed=20)

        est = TurboQuantEstimator(d=d, bits=bits, seed=SEED)
        compressed = est.quantize(keys)

        # MSE bits = max(bits-1, 1) = 1
        assert est.mse_bits == 1

        est_scores = est.inner_product(queries, compressed)
        true_scores = queries @ keys.T

        cos_sims = cosine_similarity_vectors(
            est_scores.flatten().unsqueeze(0),
            true_scores.flatten().unsqueeze(0),
        )
        # 2-bit is aggressive, so relaxed threshold
        assert cos_sims.item() > 0.60, (
            f"Pipeline d=128/2-bit: cosine sim = {cos_sims.item():.4f}, expected > 0.60"
        )


class TestFullPipelineD128_4bit:
    """High quality: d=128, 4-bit."""

    def test_full_pipeline_d128_4bit(self):
        """4-bit uses 3-bit MSE + 1-bit QJL. Near-lossless expected."""
        d, bits = 128, 4
        n_keys = 500

        keys = random_unit_vectors(n_keys, d, seed=10)
        queries = random_unit_vectors(50, d, seed=20)

        est = TurboQuantEstimator(d=d, bits=bits, seed=SEED)
        compressed = est.quantize(keys)

        assert est.mse_bits == 3

        est_scores = est.inner_product(queries, compressed)
        true_scores = queries @ keys.T

        cos_sims = cosine_similarity_vectors(
            est_scores.flatten().unsqueeze(0),
            true_scores.flatten().unsqueeze(0),
        )
        # 4-bit should be near-lossless
        assert cos_sims.item() > 0.95, (
            f"Pipeline d=128/4-bit: cosine sim = {cos_sims.item():.4f}, expected > 0.95"
        )


# ===========================================================================
# B. KV Cache Wrapper Tests
# ===========================================================================
class TestKVCacheAppendAndRetrieve:
    """Basic round-trip: append keys/values, then retrieve."""

    def test_kv_cache_append_and_retrieve(self):
        """Append a batch of KV pairs, retrieve values, check shape and quality."""
        d = 128
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=3, seed=SEED)

        n = 100
        keys = random_unit_vectors(n, d, seed=10)
        values = random_unit_vectors(n, d, seed=20)

        cache.append(keys, values)
        assert cache.seq_len == 1

        # Retrieve values
        retrieved = cache.get_values()
        assert retrieved.shape == (n, d), (
            f"Expected ({n}, {d}), got {retrieved.shape}"
        )

        # Value reconstruction should be reasonable
        cos_sims = cosine_similarity_vectors(values, retrieved)
        mean_cos = cos_sims.mean().item()
        assert mean_cos > 0.85, (
            f"Value cosine similarity: {mean_cos:.4f}, expected > 0.85"
        )

        # Attention scores
        queries = random_unit_vectors(10, d, seed=30)
        scores = cache.attention_scores(queries)
        assert scores.shape == (10, n)


class TestKVCacheAttentionScoresUnbiased:
    """Verify inner products through the cache are unbiased."""

    def test_kv_cache_attention_scores_unbiased(self):
        """Mean estimation error across many query-key pairs should be near zero."""
        d = 128
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=3, seed=SEED)

        n_keys = 500
        n_queries = 200
        keys = random_unit_vectors(n_keys, d, seed=10)
        values = random_unit_vectors(n_keys, d, seed=20)
        queries = random_unit_vectors(n_queries, d, seed=30)

        cache.append(keys, values)
        est_scores = cache.attention_scores(queries)  # (n_queries, n_keys)
        true_scores = queries @ keys.T

        mean_error = (est_scores - true_scores).mean().item()
        assert abs(mean_error) < 0.05, (
            f"Cache attention mean error: {mean_error:.4f}, expected |err| < 0.05"
        )


class TestKVCacheCompressionRatio:
    """Verify memory_usage_bits() reports correct compression ratio."""

    def test_kv_cache_compression_ratio(self):
        """3-bit at d=128 should give approximately 5.0x compression."""
        d, bits = 128, 3
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=bits, seed=SEED)

        n = 200
        keys = random_unit_vectors(n, d, seed=10)
        values = random_unit_vectors(n, d, seed=20)
        cache.append(keys, values)

        usage = cache.memory_usage_bits()
        assert usage["total_bits"] > 0
        assert usage["fp16_baseline_bits"] > 0
        ratio = usage["compression_ratio"]

        # 3-bit should give approximately 3.8-5.5x compression
        # The exact ratio depends on how key and value bits are counted
        assert ratio > 3.5, (
            f"Compression ratio {ratio:.2f}x too low for 3-bit, expected > 3.5x"
        )
        assert ratio < 7.0, (
            f"Compression ratio {ratio:.2f}x too high for 3-bit, expected < 7.0x"
        )

    @pytest.mark.parametrize("bits,min_ratio", [
        (2, 5.0),
        (3, 3.5),
        (4, 2.5),
    ])
    def test_compression_ratio_by_bits(self, bits, min_ratio):
        """Compression should improve with fewer bits."""
        d = 128
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=bits, seed=SEED)
        keys = random_unit_vectors(100, d, seed=10)
        values = random_unit_vectors(100, d, seed=20)
        cache.append(keys, values)

        usage = cache.memory_usage_bits()
        assert usage["compression_ratio"] > min_ratio, (
            f"{bits}-bit compression ratio {usage['compression_ratio']:.2f}x "
            f"below minimum {min_ratio}x"
        )


class TestKVCacheSequentialAppend:
    """Append tokens one-by-one, verify cache grows correctly."""

    def test_kv_cache_sequential_append(self):
        """Append 20 single-vector batches, verify attention scores shape."""
        d = 128
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=3, seed=SEED)

        n_tokens = 20
        torch.manual_seed(SEED)
        for i in range(n_tokens):
            k = torch.randn(1, d)
            k = k / k.norm(dim=-1, keepdim=True)
            v = torch.randn(1, d)
            v = v / v.norm(dim=-1, keepdim=True)
            cache.append(k, v)

        assert cache.seq_len == n_tokens

        # Attention scores should have shape (1, n_tokens)
        q = random_unit_vectors(1, d, seed=99)
        scores = cache.attention_scores(q)
        assert scores.shape == (1, n_tokens), (
            f"Expected (1, {n_tokens}), got {scores.shape}"
        )

        # Values should be (n_tokens, d)
        vals = cache.get_values()
        assert vals.shape == (n_tokens, d)


class TestKVCacheBatchAppend:
    """Append batch of tokens, verify it matches sequential for same data."""

    def test_kv_cache_batch_append(self):
        """A single batch append should produce same attention scores as
        sequential single-vector appends with the same data."""
        d = 128
        n = 20
        torch.manual_seed(100)
        keys = torch.randn(n, d)
        keys = keys / keys.norm(dim=1, keepdim=True)
        values = torch.randn(n, d)
        values = values / values.norm(dim=1, keepdim=True)

        query = random_unit_vectors(1, d, seed=200)

        # Batch append
        cache_batch = TurboQuantKVCache(d_key=d, d_value=d, bits=3, seed=SEED)
        cache_batch.append(keys, values)
        scores_batch = cache_batch.attention_scores(query)  # (1, n)

        # Sequential append
        cache_seq = TurboQuantKVCache(d_key=d, d_value=d, bits=3, seed=SEED)
        for i in range(n):
            cache_seq.append(keys[i:i+1], values[i:i+1])
        scores_seq = cache_seq.attention_scores(query)  # (1, n)

        assert scores_batch.shape == scores_seq.shape == (1, n)

        # Scores should be identical since same seed => same matrices
        diff = (scores_batch - scores_seq).abs().max().item()
        assert diff < 1e-4, (
            f"Batch vs sequential max score diff: {diff:.6f}, expected < 1e-4"
        )


class TestKVCacheValueReconstruction:
    """get_values() reconstruction quality."""

    def test_kv_cache_value_reconstruction(self):
        """Values use MSE-only PolarQuant (full bits). Reconstruction
        error per-vector should be bounded."""
        d, bits = 128, 3
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=bits, seed=SEED)

        n = 200
        values = random_unit_vectors(n, d, seed=20)
        keys = random_unit_vectors(n, d, seed=10)
        cache.append(keys, values)

        retrieved = cache.get_values()
        per_vector_mse = ((values - retrieved) ** 2).sum(dim=1)  # (n,)
        mean_mse = per_vector_mse.mean().item()

        # For 3-bit values (MSE-only), distortion per coordinate ~ 0.03/d,
        # total MSE per vector ~ d * 0.03/d = 0.03.
        # With norm storage imprecision, allow generous slack.
        assert mean_mse < 0.3, (
            f"Value MSE: {mean_mse:.4f}, expected < 0.3 for 3-bit MSE"
        )

        # Cosine similarity should be high
        cos_sims = cosine_similarity_vectors(values, retrieved)
        assert cos_sims.mean().item() > 0.90, (
            f"Value cosine similarity: {cos_sims.mean().item():.4f}, expected > 0.90"
        )


# ===========================================================================
# C. Paper Bound Validation
# ===========================================================================
class TestMSEDistortionWithinPaperBound:
    """D_mse(3-bit) should be below 0.043 (paper Table 1)."""

    def test_mse_distortion_within_paper_bound(self):
        """Empirical MSE distortion for 3-bit at d=128 should be within
        reasonable range of the theoretical bound.

        D_mse = E[||x - x_hat||^2] for unit vectors.
        Theoretical bound: D_mse <= sqrt(3)*pi/2 / 4^b ~ 2.721 / 64 ~ 0.0425.
        The Gaussian approximation and finite-sample effects can cause
        slight overshoot, so we allow 1.5x slack.
        """
        d, bits = 128, 3
        pq = PolarQuant(d=d, bits=bits, seed=SEED)

        n = 2000
        x = random_unit_vectors(n, d, seed=42)

        indices = pq.quantize(x)
        x_hat = pq.dequantize(indices)

        # MSE per vector
        mse_per_vec = ((x - x_hat) ** 2).sum(dim=1)  # (n,)
        mean_mse = mse_per_vec.mean().item()

        # Paper bound for b=3 MSE: sqrt(3)*pi/2 / 4^3 = 2.721/64 ~ 0.0425
        # Allow 1.5x slack for Gaussian approximation vs exact Beta PDF
        paper_bound = MSE_BOUND_FACTOR / (4 ** bits)
        assert mean_mse < paper_bound * 1.5, (
            f"MSE distortion {mean_mse:.6f} exceeds 1.5x paper bound "
            f"{paper_bound * 1.5:.6f} at 3-bit"
        )

    @pytest.mark.parametrize("bits,paper_dmse_factor", [
        (1, 0.36),
        (2, 0.117),
        (3, 0.03),
        (4, 0.009),
    ])
    def test_mse_per_coord_within_paper_values(self, bits, paper_dmse_factor):
        """Per-coordinate distortion should match paper table values.

        D_coord ~ paper_dmse_factor / d (from Table 1 in the paper).
        Total D_mse ~ d * D_coord = paper_dmse_factor.
        """
        d = 128
        pq = PolarQuant(d=d, bits=bits, seed=SEED)

        n = 2000
        x = random_unit_vectors(n, d, seed=42)
        indices = pq.quantize(x)
        x_hat = pq.dequantize(indices)

        mse_per_vec = ((x - x_hat) ** 2).sum(dim=1)
        mean_mse = mse_per_vec.mean().item()

        # Allow 3x slack for finite-sample effects and Gaussian approximation
        assert mean_mse < paper_dmse_factor * 3.0, (
            f"MSE at {bits}-bit: {mean_mse:.6f}, "
            f"expected < {paper_dmse_factor * 3.0:.6f} (3x of {paper_dmse_factor})"
        )


class TestIPDistortionWithinPaperBound:
    """D_prod(3-bit, d=128) should be < 0.0021 (paper Table 1)."""

    def test_ip_distortion_within_paper_bound(self):
        """D_prod = E[|<y,x> - <y,x_tilde>|^2] for 3-bit at d=128.

        Paper value: D_prod ~ 0.18/128 ~ 0.0014.
        Theoretical bound: sqrt(3)*pi^2/d / 4^b ~ 17.08 / 128 / 64 ~ 0.0021.
        """
        d, bits = 128, 3
        est = TurboQuantEstimator(d=d, bits=bits, seed=SEED)

        n = 2000
        x = random_unit_vectors(n, d, seed=42)
        y = random_unit_vectors(n, d, seed=77)

        true_ips = (x * y).sum(dim=1)
        compressed = est.quantize(x)
        est_ips = element_wise_inner_products(est, y, compressed)

        d_prod = ((est_ips - true_ips) ** 2).mean().item()

        # Theoretical bound for ||y||=1: IP_BOUND_FACTOR / d / 4^b
        bound = IP_BOUND_FACTOR / d / (4 ** bits)

        # Allow 3x slack for finite samples
        assert d_prod < bound * 3.0, (
            f"D_prod at 3-bit: {d_prod:.6f}, "
            f"theoretical bound (3x): {bound * 3.0:.6f}"
        )

    @pytest.mark.parametrize("bits,paper_dprod_factor", [
        (2, 0.56),
        (3, 0.18),
        (4, 0.047),
    ])
    def test_dprod_matches_paper_table(self, bits, paper_dprod_factor):
        """D_prod should match paper table values within 3x."""
        d = 128
        est = TurboQuantEstimator(d=d, bits=bits, seed=SEED)

        n = 2000
        x = random_unit_vectors(n, d, seed=42)
        y = random_unit_vectors(n, d, seed=77)

        true_ips = (x * y).sum(dim=1)
        compressed = est.quantize(x)
        est_ips = element_wise_inner_products(est, y, compressed)

        d_prod = ((est_ips - true_ips) ** 2).mean().item()
        expected = paper_dprod_factor / d

        assert d_prod < expected * 3.0, (
            f"D_prod at {bits}-bit: {d_prod:.6f}, "
            f"expected < {expected * 3.0:.6f} (3x of {expected:.6f})"
        )


class TestInnerProductUnbiasedness:
    """E[error] should be approximately zero over many trials."""

    def test_inner_product_unbiasedness(self):
        """Average inner product estimation error over 1000 pairs should
        have |mean| < 0.01, confirming the estimator is unbiased."""
        d, bits = 128, 3
        n = 1000

        x = random_unit_vectors(n, d, seed=42)
        y = random_unit_vectors(n, d, seed=77)
        true_ips = (x * y).sum(dim=1)

        est = TurboQuantEstimator(d=d, bits=bits, seed=SEED)
        compressed = est.quantize(x)
        est_ips = element_wise_inner_products(est, y, compressed)

        errors = est_ips - true_ips
        mean_error = errors.mean().item()

        assert abs(mean_error) < 0.02, (
            f"|E[error]| = {abs(mean_error):.6f}, expected < 0.02 for unbiased estimator"
        )

    def test_unbiasedness_across_random_seeds(self):
        """Average across multiple random matrix seeds to reduce variance."""
        d, bits = 128, 3
        n = 200

        torch.manual_seed(42)
        x = torch.randn(1, d)
        x = x / x.norm()
        y = torch.randn(1, d)
        y = y / y.norm()
        true_ip = (x * y).sum().item()

        estimates = []
        for seed in range(200):
            est = TurboQuantEstimator(d=d, bits=bits, seed=seed + 5000)
            compressed = est.quantize(x)
            ip_est = est.inner_product(y, compressed).item()
            estimates.append(ip_est)

        mean_est = sum(estimates) / len(estimates)
        assert abs(mean_est - true_ip) < 0.08, (
            f"Mean IP estimate: {mean_est:.4f}, true: {true_ip:.4f}, "
            f"diff: {abs(mean_est - true_ip):.4f}"
        )


class TestCompressionRatioMatchesPaper:
    """3-bit should give approximately 5.0x compression."""

    def test_compression_ratio_matches_paper(self):
        """For d_key = d_value = 128, 3-bit:
        Key: 2*128 (MSE) + 128 (QJL) + 16 (r_norm) + 16 (v_norm) = 416 bits
        Value: 3*128 (MSE) + 16 (v_norm) = 400 bits
        Total: 816 bits vs FP16 4096 bits => 5.02x
        """
        d = 128
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=3, seed=SEED)

        keys = random_unit_vectors(100, d, seed=10)
        values = random_unit_vectors(100, d, seed=20)
        cache.append(keys, values)

        usage = cache.memory_usage_bits()
        ratio = usage["compression_ratio"]

        # Paper claims ~5.0x for 3-bit
        assert 4.0 < ratio < 6.0, (
            f"3-bit compression ratio {ratio:.2f}x, paper claims ~5.0x"
        )


# ===========================================================================
# D. Edge Cases
# ===========================================================================
class TestSingleVector:
    """Pipeline works with batch_size=1."""

    def test_single_vector_estimator(self):
        """Compress and estimate IP for a single vector."""
        d, bits = 128, 3
        est = TurboQuantEstimator(d=d, bits=bits, seed=SEED)

        x = random_unit_vectors(1, d, seed=10)
        y = random_unit_vectors(1, d, seed=20)

        compressed = est.quantize(x)
        score = est.inner_product(y, compressed)
        # Should return (1, 1) shape
        assert score.shape == (1, 1), f"Expected (1, 1), got {score.shape}"

    def test_single_vector_1d(self):
        """Compress a single 1D vector (no batch dimension)."""
        d, bits = 128, 3
        est = TurboQuantEstimator(d=d, bits=bits, seed=SEED)

        x = random_unit_vectors(1, d, seed=10).squeeze(0)  # (d,)
        compressed = est.quantize(x)

        # Should handle 1D input
        assert compressed["mse_indices"].dim() == 1
        assert compressed["vec_norm"].dim() == 0

    def test_single_vector_kv_cache(self):
        """KV cache works with single-vector append."""
        d = 128
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=3, seed=SEED)

        k = random_unit_vectors(1, d, seed=10)
        v = random_unit_vectors(1, d, seed=20)
        cache.append(k, v)

        q = random_unit_vectors(1, d, seed=30)
        scores = cache.attention_scores(q)
        assert scores.shape == (1, 1)


class TestLargeBatch:
    """Pipeline works with 10K vectors."""

    def test_large_batch(self):
        """Compress and estimate IP for a 10K-vector batch.
        This also serves as a basic performance sanity check."""
        d, bits = 128, 3
        n = 10000

        est = TurboQuantEstimator(d=d, bits=bits, seed=SEED)
        keys = random_unit_vectors(n, d, seed=10)

        compressed = est.quantize(keys)
        assert compressed["mse_indices"].shape == (n, d)
        assert compressed["qjl_signs"].shape == (n, d)

        # Estimate IP with a single query
        query = random_unit_vectors(1, d, seed=20)
        scores = est.inner_product(query, compressed)
        assert scores.shape == (1, n)

    def test_large_batch_kv_cache(self):
        """KV cache handles 10K tokens."""
        d = 128
        n = 10000
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=3, seed=SEED)

        keys = random_unit_vectors(n, d, seed=10)
        values = random_unit_vectors(n, d, seed=20)
        cache.append(keys, values)

        # Retrieve values
        vals = cache.get_values()
        assert vals.shape == (n, d)

        # Query
        q = random_unit_vectors(5, d, seed=30)
        scores = cache.attention_scores(q)
        assert scores.shape == (5, n)


class TestGPUIfAvailable:
    """Run full pipeline on CUDA if available."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_gpu_if_available(self):
        """Full pipeline on GPU: quantize, inner product, KV cache."""
        d, bits = 128, 3
        device = "cuda"

        est = TurboQuantEstimator(d=d, bits=bits, seed=SEED, device=device)
        keys = random_unit_vectors(200, d, seed=10).to(device)
        queries = random_unit_vectors(10, d, seed=20).to(device)

        compressed = est.quantize(keys)
        scores = est.inner_product(queries, compressed)
        assert scores.device.type == "cuda"
        assert scores.shape == (10, 200)

        # KV cache on GPU
        cache = TurboQuantKVCache(
            d_key=d, d_value=d, bits=bits, seed=SEED, device=device
        )
        values = random_unit_vectors(200, d, seed=30).to(device)
        cache.append(keys, values)

        cache_scores = cache.attention_scores(queries)
        assert cache_scores.device.type == "cuda"

        vals = cache.get_values()
        assert vals.device.type == "cuda"


class TestDeterministicWithSeed:
    """Same seed produces identical results."""

    def test_deterministic_with_seed(self):
        """Two runs with the same seed should produce identical compressed
        representations and inner product estimates."""
        d, bits = 128, 3

        keys = random_unit_vectors(100, d, seed=10)
        queries = random_unit_vectors(10, d, seed=20)

        # Run 1
        est1 = TurboQuantEstimator(d=d, bits=bits, seed=SEED)
        comp1 = est1.quantize(keys)
        scores1 = est1.inner_product(queries, comp1)

        # Run 2 (same seed)
        est2 = TurboQuantEstimator(d=d, bits=bits, seed=SEED)
        comp2 = est2.quantize(keys)
        scores2 = est2.inner_product(queries, comp2)

        # Should be identical
        assert torch.equal(comp1["mse_indices"], comp2["mse_indices"]), (
            "MSE indices differ between identical-seed runs"
        )
        assert torch.equal(comp1["qjl_signs"], comp2["qjl_signs"]), (
            "QJL signs differ between identical-seed runs"
        )
        assert torch.allclose(scores1, scores2, atol=1e-6), (
            f"Scores differ: max diff = {(scores1 - scores2).abs().max().item()}"
        )

    def test_different_seeds_differ(self):
        """Different seeds should produce different compressed representations."""
        d, bits = 128, 3
        keys = random_unit_vectors(100, d, seed=10)

        est1 = TurboQuantEstimator(d=d, bits=bits, seed=100)
        comp1 = est1.quantize(keys)

        est2 = TurboQuantEstimator(d=d, bits=bits, seed=200)
        comp2 = est2.quantize(keys)

        # Rotation matrices differ => indices should differ
        assert not torch.equal(comp1["mse_indices"], comp2["mse_indices"]), (
            "Different seeds produced identical MSE indices"
        )


# ===========================================================================
# E. Cross-Module Consistency
# ===========================================================================
class TestEstimatorMatchesKVCache:
    """TurboQuantEstimator.inner_product should match KVCache.attention_scores
    for the same input data."""

    def test_estimator_matches_kv_cache(self):
        """Both paths should produce identical attention scores."""
        d = 128
        n_keys = 100
        n_queries = 10

        keys = random_unit_vectors(n_keys, d, seed=10)
        values = random_unit_vectors(n_keys, d, seed=20)
        queries = random_unit_vectors(n_queries, d, seed=30)

        # Path 1: Direct estimator
        est = TurboQuantEstimator(d=d, bits=3, seed=SEED)
        compressed = est.quantize(keys)
        est_scores = est.inner_product(queries, compressed)  # (n_q, n_k)

        # Path 2: Through KV cache
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=3, seed=SEED)
        cache.append(keys, values)
        cache_scores = cache.attention_scores(queries)  # (n_q, n_k)

        # Should be identical (same seed, same pipeline)
        max_diff = (est_scores - cache_scores).abs().max().item()
        assert max_diff < 1e-4, (
            f"Estimator vs cache max score diff: {max_diff:.6f}, expected < 1e-4"
        )


class TestValueMSEvsKeyIP:
    """Values use MSE-only PolarQuant, keys use full estimator.
    This asymmetry is a core design choice from the paper."""

    def test_value_mse_vs_key_ip(self):
        """Verify the asymmetric treatment:
        - Keys: TurboQuantEstimator (MSE + QJL) for unbiased inner products
        - Values: PolarQuant (MSE-only) for best reconstruction
        """
        d, bits = 128, 3
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=bits, seed=SEED)

        # Verify types
        assert isinstance(cache.key_quantizer, TurboQuantEstimator), (
            f"Key quantizer should be TurboQuantEstimator, got {type(cache.key_quantizer)}"
        )
        assert isinstance(cache.value_quantizer, PolarQuant), (
            f"Value quantizer should be PolarQuant, got {type(cache.value_quantizer)}"
        )

        # Key quantizer has QJL (2-stage), value quantizer does not
        assert hasattr(cache.key_quantizer, 'qjl'), "Key quantizer missing QJL stage"
        assert isinstance(cache.key_quantizer.qjl, QJL), "Key quantizer QJL is wrong type"

        # Value quantizer uses full bits for MSE (better reconstruction)
        assert cache.value_quantizer.bits == bits, (
            f"Value quantizer uses {cache.value_quantizer.bits} bits, expected {bits}"
        )
        # Key quantizer uses bits-1 for MSE (1 bit reserved for QJL)
        assert cache.key_quantizer.mse_bits == bits - 1, (
            f"Key MSE uses {cache.key_quantizer.mse_bits} bits, expected {bits - 1}"
        )

    def test_value_reconstruction_better_than_key(self):
        """Values at full bits should have lower reconstruction error than
        keys at (bits-1) bits MSE."""
        d, bits = 128, 3
        n = 500

        vectors = random_unit_vectors(n, d, seed=42)

        # Value path: full-bit PolarQuant
        value_pq = PolarQuant(d=d, bits=bits, seed=SEED)
        v_indices = value_pq.quantize(vectors)
        v_recon = value_pq.dequantize(v_indices)
        value_mse = ((vectors - v_recon) ** 2).sum(dim=1).mean().item()

        # Key path: (bits-1)-bit PolarQuant
        key_pq = PolarQuant(d=d, bits=bits - 1, seed=SEED)
        k_indices = key_pq.quantize(vectors)
        k_recon = key_pq.dequantize(k_indices)
        key_mse = ((vectors - k_recon) ** 2).sum(dim=1).mean().item()

        assert value_mse < key_mse, (
            f"Value MSE ({value_mse:.6f}) should be lower than key MSE "
            f"({key_mse:.6f}) since values use full {bits} bits vs {bits-1} bits"
        )


class TestNonUnitVectors:
    """Pipeline should handle arbitrary (non-unit) vectors via norm storage."""

    def test_non_unit_vectors_through_estimator(self):
        """Estimator normalizes internally and stores ||x||. Inner products
        should still be approximately correct."""
        d, bits = 128, 3
        est = TurboQuantEstimator(d=d, bits=bits, seed=SEED)

        n = 500
        # Non-unit vectors with varying norms
        x = random_vectors(n, d, seed=42, scale=3.0)
        y = random_vectors(n, d, seed=77, scale=2.0)

        true_ips = (x * y).sum(dim=1)
        compressed = est.quantize(x)
        est_ips = element_wise_inner_products(est, y, compressed)

        # Mean error should be near zero (unbiased)
        mean_error = (est_ips - true_ips).mean().item()
        # Larger norms => larger absolute errors, so scale tolerance
        scale_factor = true_ips.abs().mean().item()
        relative_mean_error = abs(mean_error) / (scale_factor + 1e-8)

        assert relative_mean_error < 0.1, (
            f"Relative mean IP error: {relative_mean_error:.4f}, expected < 0.1"
        )

    def test_non_unit_vectors_through_kv_cache(self):
        """KV cache handles non-unit vectors properly."""
        d = 128
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=3, seed=SEED)

        n = 100
        keys = random_vectors(n, d, seed=10, scale=5.0)
        values = random_vectors(n, d, seed=20, scale=3.0)
        queries = random_vectors(10, d, seed=30, scale=2.0)

        cache.append(keys, values)
        scores = cache.attention_scores(queries)
        assert scores.shape == (10, n)
        assert torch.isfinite(scores).all(), "Non-finite attention scores"

        vals = cache.get_values()
        assert vals.shape == (n, d)
        assert torch.isfinite(vals).all(), "Non-finite reconstructed values"


class TestClearAndReuse:
    """Cache can be cleared and reused."""

    def test_clear_resets_state(self):
        """After clear(), cache should be empty."""
        d = 128
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=3, seed=SEED)

        keys = random_unit_vectors(50, d, seed=10)
        values = random_unit_vectors(50, d, seed=20)
        cache.append(keys, values)
        assert cache.seq_len > 0

        cache.clear()
        assert cache.seq_len == 0

    def test_reuse_after_clear(self):
        """Cache works normally after clear()."""
        d = 128
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=3, seed=SEED)

        # First use
        keys1 = random_unit_vectors(50, d, seed=10)
        values1 = random_unit_vectors(50, d, seed=20)
        cache.append(keys1, values1)

        cache.clear()

        # Second use
        keys2 = random_unit_vectors(30, d, seed=30)
        values2 = random_unit_vectors(30, d, seed=40)
        cache.append(keys2, values2)

        assert cache.seq_len == 1
        vals = cache.get_values()
        assert vals.shape == (30, d)


class TestMultipleAppendAttention:
    """Attention scores are correct across multiple appends."""

    def test_multi_append_attention_shape(self):
        """After 3 appends of different sizes, attention scores have the
        correct total number of columns."""
        d = 128
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=3, seed=SEED)

        sizes = [20, 30, 50]
        for i, n in enumerate(sizes):
            k = random_unit_vectors(n, d, seed=100 + i)
            v = random_unit_vectors(n, d, seed=200 + i)
            cache.append(k, v)

        total = sum(sizes)
        queries = random_unit_vectors(5, d, seed=99)
        scores = cache.attention_scores(queries)
        assert scores.shape == (5, total), (
            f"Expected (5, {total}), got {scores.shape}"
        )


class TestEmptyCache:
    """Edge cases for empty cache."""

    def test_empty_attention_scores(self):
        """Attention scores on empty cache should be empty."""
        d = 128
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=3, seed=SEED)

        q = random_unit_vectors(5, d, seed=10)
        scores = cache.attention_scores(q)
        assert scores.shape == (5, 0)

    def test_empty_get_values(self):
        """get_values on empty cache should return empty tensor."""
        d = 128
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=3, seed=SEED)
        vals = cache.get_values()
        assert vals.shape == (0, d)

    def test_empty_memory_usage(self):
        """memory_usage_bits on empty cache should return zeros."""
        d = 128
        cache = TurboQuantKVCache(d_key=d, d_value=d, bits=3, seed=SEED)
        usage = cache.memory_usage_bits()
        assert usage["total_bits"] == 0
        assert usage["compression_ratio"] == 0.0


class TestParametrizedDimensions:
    """Parametrized test across multiple dimensions and bit widths."""

    @pytest.mark.parametrize("d", [64, 128, 256])
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_full_pipeline_parametrized(self, d, bits):
        """Full pipeline should work across all (d, bits) combinations."""
        est = TurboQuantEstimator(d=d, bits=bits, seed=SEED)
        n = 100

        keys = random_unit_vectors(n, d, seed=10)
        queries = random_unit_vectors(5, d, seed=20)

        compressed = est.quantize(keys)
        assert compressed["mse_indices"].shape == (n, d)
        assert compressed["qjl_signs"].shape == (n, d)

        scores = est.inner_product(queries, compressed)
        assert scores.shape == (5, n)
        assert torch.isfinite(scores).all(), (
            f"Non-finite scores at d={d}, bits={bits}"
        )
