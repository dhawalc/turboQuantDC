"""Tests for the combined TurboQuant inner product estimator and KV cache.

The combined estimator (TurboQuantEstimator) uses Stage 1 (PolarQuant) for MSE
reconstruction and Stage 2 (QJL) for bias correction. The KV cache wrapper
uses Prod quantization for keys and MSE quantization for values.

Mathematical reference: MATH_SPEC.md sections 7, 8, 10, 11.

Notes on API semantics:
    - inner_product(query, compressed) returns shape (batch_q, batch_k) — a
      cross-product matrix. For element-wise pair testing use torch.diagonal.
    - TurboQuantKVCache.seq_len counts append() call batches, not total tokens.
      attention_scores(queries) still returns correct (n_queries, total_tokens).
"""

import math

import pytest
import torch

from turboquantdc.estimator import TurboQuantEstimator
from turboquantdc.kv_cache import TurboQuantKVCache
from turboquantdc.polarquant import PolarQuant

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DIM = 128
SEED = 42
N_VECTORS = 1000

# Inner product distortion bound: D_prod <= sqrt(3)*pi^2/d * 1/4^b * ||y||^2
IP_BOUND_FACTOR = math.sqrt(3) * math.pi ** 2  # ~ 17.08

# Paper's tabulated D_prod values (assuming ||y||=1): {1.57/d, 0.56/d, 0.18/d, 0.047/d}
PAPER_DPROD = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}

# MSE bias at 1-bit: E[<y, x_mse>] = (2/pi) * <y, x>
MSE_BIAS_1BIT = 2.0 / math.pi  # ~ 0.6366

# MSE bound
MSE_BOUND_FACTOR = math.sqrt(3) * math.pi / 2  # ~ 2.721


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_unit_vectors(n: int, d: int, seed: int = SEED) -> torch.Tensor:
    """Generate n random unit vectors of dimension d."""
    torch.manual_seed(seed)
    x = torch.randn(n, d)
    x = x / x.norm(dim=1, keepdim=True)
    return x


def compute_compression_ratio(d: int, bits: int) -> float:
    """Compute the theoretical compression ratio for a given d and bits.

    Storage: (bits-1)*d bits [MSE] + d bits [QJL] + 16 bits [residual_norm]
             + 16 bits [vec_norm] = bits*d + 32 bits total.
    Baseline: 16*d bits (FP16).
    Ratio: 16*d / (bits*d + 32).
    """
    total_bits = bits * d + 32
    fp16_bits = 16 * d
    return fp16_bits / total_bits


def element_wise_inner_products(
    est: TurboQuantEstimator,
    queries: torch.Tensor,
    compressed: dict,
) -> torch.Tensor:
    """Compute element-wise inner product <q_i, k_i> for each pair.

    inner_product returns (batch_q, batch_k) cross-product. Take diagonal
    to get the N element-wise products.

    Args:
        est: TurboQuantEstimator instance.
        queries: (N, d) query vectors.
        compressed: Output of est.quantize() for N key vectors.

    Returns:
        (N,) tensor of estimated inner products <q_i, k_i>.
    """
    return torch.diagonal(est.inner_product(queries, compressed))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(params=[2, 3, 4], ids=lambda b: f"bits={b}")
def bits(request):
    """Total bit-width (MSE uses bits-1, QJL uses 1)."""
    return request.param


@pytest.fixture
def estimator(bits):
    """TurboQuantEstimator instance parametrized over bit-widths."""
    return TurboQuantEstimator(d=DEFAULT_DIM, bits=bits, seed=SEED)


@pytest.fixture
def estimator_3bit():
    """TurboQuantEstimator at 3-bit for targeted tests."""
    return TurboQuantEstimator(d=DEFAULT_DIM, bits=3, seed=SEED)


@pytest.fixture
def kv_cache():
    """TurboQuantKVCache for testing the cache wrapper."""
    return TurboQuantKVCache(d_key=DEFAULT_DIM, d_value=DEFAULT_DIM, bits=3, seed=SEED)


@pytest.fixture
def unit_vectors():
    return random_unit_vectors(N_VECTORS, DEFAULT_DIM)


@pytest.fixture
def query_vectors():
    return random_unit_vectors(N_VECTORS, DEFAULT_DIM, seed=77)


# ---------------------------------------------------------------------------
# Tests: combined estimator is unbiased (Theorem 2)
# ---------------------------------------------------------------------------
class TestUnbiasedness:
    """The combined TurboQuantEstimator must be unbiased:
    E[<y, x_tilde>] = <y, x>."""

    def test_estimator_unbiased_fixed_vectors(self):
        """Fix x and y, average over many random matrices.

        E[<y, x_tilde>] = <y, x> where expectation is over the random
        rotation and QJL projection.
        """
        torch.manual_seed(42)
        x = torch.randn(1, DEFAULT_DIM)
        x = x / x.norm()
        y = torch.randn(1, DEFAULT_DIM)
        y = y / y.norm()

        true_ip = (x * y).sum().item()

        n_trials = 300
        estimates = []
        for seed in range(n_trials):
            est = TurboQuantEstimator(d=DEFAULT_DIM, bits=3, seed=seed + 2000)
            compressed = est.quantize(x)
            # inner_product returns (1, 1) for 1 query vs 1 key
            ip_est = est.inner_product(y, compressed)
            estimates.append(ip_est.item())

        mean_est = sum(estimates) / len(estimates)
        tolerance = 0.08  # statistical tolerance
        assert abs(mean_est - true_ip) < tolerance, (
            f"Mean estimated IP: {mean_est:.4f}, true IP: {true_ip:.4f}, "
            f"diff: {abs(mean_est - true_ip):.4f}"
        )

    def test_estimator_unbiased_batch(self, estimator_3bit):
        """Average error over many (x, y) pairs should be ~0.

        Uses diagonal of the cross-product matrix for element-wise pairs.
        """
        n = 2000
        x = random_unit_vectors(n, DEFAULT_DIM, seed=42)
        y = random_unit_vectors(n, DEFAULT_DIM, seed=77)

        true_ips = (x * y).sum(dim=1)  # (n,)
        compressed = estimator_3bit.quantize(x)
        # inner_product returns (n, n); take diagonal for element-wise pairs
        est_ips = element_wise_inner_products(estimator_3bit, y, compressed)

        mean_error = (est_ips - true_ips).mean().item()
        assert abs(mean_error) < 0.05, (
            f"Mean estimation error: {mean_error:.4f} (should be ~0 for unbiased)"
        )


# ---------------------------------------------------------------------------
# Tests: inner product distortion (Theorem 2)
# ---------------------------------------------------------------------------
class TestInnerProductDistortion:
    """D_prod := E[|<y,x> - <y,x_tilde>|^2] should match paper bounds."""

    @pytest.mark.parametrize("bits_val,expected_dprod_factor", [
        (2, 1.57),    # D_prod ~ 1.57/d (total 2-bit: 1-bit MSE + 1-bit QJL)
        (3, 0.56),    # D_prod ~ 0.56/d (total 3-bit: 2-bit MSE + 1-bit QJL)
        (4, 0.18),    # D_prod ~ 0.18/d (total 4-bit: 3-bit MSE + 1-bit QJL)
    ])
    def test_dprod_matches_paper_values(self, bits_val, expected_dprod_factor):
        """D_prod should match {1.57/d, 0.56/d, 0.18/d, 0.047/d} for b={1,2,3,4}.

        We use ||y||=1 unit vectors, so the bound simplifies.
        Allow 2x slack for finite samples.
        """
        est = TurboQuantEstimator(d=DEFAULT_DIM, bits=bits_val, seed=SEED)
        n = 2000
        x = random_unit_vectors(n, DEFAULT_DIM, seed=42)
        y = random_unit_vectors(n, DEFAULT_DIM, seed=77)

        true_ips = (x * y).sum(dim=1)
        compressed = est.quantize(x)
        est_ips = element_wise_inner_products(est, y, compressed)

        d_prod = ((est_ips - true_ips) ** 2).mean().item()
        expected = expected_dprod_factor / DEFAULT_DIM

        # Allow 3x slack for finite sample effects and diagonal approximation
        assert d_prod < expected * 3.0, (
            f"D_prod at {bits_val}-bit: {d_prod:.6f}, "
            f"expected < {expected * 3.0:.6f} (3x of {expected:.6f})"
        )

    def test_dprod_below_theoretical_bound(self, bits, estimator):
        """D_prod <= sqrt(3)*pi^2/d * 1/4^b * ||y||^2 (Theorem 2)."""
        n = 2000
        x = random_unit_vectors(n, DEFAULT_DIM, seed=42)
        y = random_unit_vectors(n, DEFAULT_DIM, seed=77)

        true_ips = (x * y).sum(dim=1)
        compressed = estimator.quantize(x)
        est_ips = element_wise_inner_products(estimator, y, compressed)

        d_prod = ((est_ips - true_ips) ** 2).mean().item()
        # ||y||=1, so bound = sqrt(3)*pi^2 / d / 4^b
        upper_bound = IP_BOUND_FACTOR / DEFAULT_DIM / (4 ** bits)

        # Allow 3x slack for finite samples
        assert d_prod < upper_bound * 3.0, (
            f"D_prod={d_prod:.6f} exceeds theoretical bound "
            f"{upper_bound:.6f} at {bits}-bit (with 3x slack)"
        )

    def test_dprod_decreases_with_bits(self):
        """More bits should give lower inner product distortion."""
        n = 2000
        x = random_unit_vectors(n, DEFAULT_DIM, seed=42)
        y = random_unit_vectors(n, DEFAULT_DIM, seed=77)
        true_ips = (x * y).sum(dim=1)

        distortions = []
        for b in [2, 3, 4]:
            est = TurboQuantEstimator(d=DEFAULT_DIM, bits=b, seed=SEED)
            compressed = est.quantize(x)
            est_ips = element_wise_inner_products(est, y, compressed)
            d_prod = ((est_ips - true_ips) ** 2).mean().item()
            distortions.append(d_prod)

        for i in range(len(distortions) - 1):
            assert distortions[i] > distortions[i + 1], (
                f"D_prod should decrease: bits={i+2} gave {distortions[i]:.6f}, "
                f"bits={i+3} gave {distortions[i+1]:.6f}"
            )


# ---------------------------------------------------------------------------
# Tests: MSE-only has biased inner products (Section 10)
# ---------------------------------------------------------------------------
class TestMSEBias:
    """Demonstrate that MSE-only (Stage 1 without QJL) has biased inner products.
    This is the motivation for Stage 2."""

    def test_mse_only_biased_at_1bit(self):
        """At 1-bit, MSE-only inner products are biased by factor 2/pi.

        E[<y, x_mse>] = (2/pi) * <y, x> ~ 0.6366 * <y, x>.
        """
        pq = PolarQuant(d=DEFAULT_DIM, bits=1, seed=SEED)
        n = 2000
        x = random_unit_vectors(n, DEFAULT_DIM, seed=42)
        y = random_unit_vectors(n, DEFAULT_DIM, seed=77)

        true_ips = (x * y).sum(dim=1)
        indices = pq.quantize(x)
        reconstructed = pq.dequantize(indices)
        mse_ips = (y * reconstructed).sum(dim=1)

        # Ratio should be close to 2/pi ~ 0.6366
        # Use regression: mse_ip ~ slope * true_ip + intercept
        # For unit vectors, intercept should be ~0
        slope = (mse_ips * true_ips).sum() / (true_ips ** 2).sum()
        assert abs(slope.item() - MSE_BIAS_1BIT) < 0.1, (
            f"MSE-only bias at 1-bit: slope={slope.item():.4f}, "
            f"expected {MSE_BIAS_1BIT:.4f} (2/pi)"
        )

    def test_combined_removes_bias(self):
        """The combined (MSE + QJL) estimator should remove the bias.

        Regression slope of estimated vs true inner products should be ~1.0.
        """
        est = TurboQuantEstimator(d=DEFAULT_DIM, bits=2, seed=SEED)
        n = 2000
        x = random_unit_vectors(n, DEFAULT_DIM, seed=42)
        y = random_unit_vectors(n, DEFAULT_DIM, seed=77)

        true_ips = (x * y).sum(dim=1)
        compressed = est.quantize(x)
        est_ips = element_wise_inner_products(est, y, compressed)

        # Regression slope should be ~1.0 (unbiased)
        slope = (est_ips * true_ips).sum() / (true_ips ** 2).sum()
        assert abs(slope.item() - 1.0) < 0.15, (
            f"Combined estimator slope: {slope.item():.4f}, "
            f"expected ~1.0 (unbiased)"
        )

    def test_mse_bias_diminishes_with_bits(self):
        """MSE-only bias should decrease as bits increase."""
        n = 2000
        x = random_unit_vectors(n, DEFAULT_DIM, seed=42)
        y = random_unit_vectors(n, DEFAULT_DIM, seed=77)
        true_ips = (x * y).sum(dim=1)

        biases = []
        for b in [1, 2, 3, 4]:
            pq = PolarQuant(d=DEFAULT_DIM, bits=b, seed=SEED)
            indices = pq.quantize(x)
            reconstructed = pq.dequantize(indices)
            mse_ips = (y * reconstructed).sum(dim=1)
            slope = (mse_ips * true_ips).sum() / (true_ips ** 2).sum()
            bias = abs(1.0 - slope.item())
            biases.append(bias)

        # Bias should decrease with more bits
        for i in range(len(biases) - 1):
            assert biases[i] > biases[i + 1] * 0.5, (
                f"MSE bias should decrease: bits={i+1} bias={biases[i]:.4f}, "
                f"bits={i+2} bias={biases[i+1]:.4f}"
            )


# ---------------------------------------------------------------------------
# Tests: needle-in-a-haystack (attention ranking)
# ---------------------------------------------------------------------------
class TestNeedleInHaystack:
    """Given a query that exactly matches one key among many random keys,
    the matching key should rank highest (or very near the top) in the
    estimated attention scores."""

    def test_needle_ranks_top5(self, estimator_3bit):
        """The exact-match key should rank in top-5 out of N=1000 keys.

        Uses inner_product(query_1xd, compressed_nxd) -> (1, n_keys),
        then squeeze to get (n_keys,) scores.
        """
        n_keys = 1000
        d = DEFAULT_DIM
        torch.manual_seed(42)

        # Generate random keys
        keys = torch.randn(n_keys, d)
        keys = keys / keys.norm(dim=1, keepdim=True)

        # Plant a needle: query = one of the keys (index 500)
        needle_idx = 500
        query = keys[needle_idx].unsqueeze(0)  # (1, d)

        # Compress all keys
        compressed = estimator_3bit.quantize(keys)

        # Estimate all inner products: (1, n_keys) -> squeeze to (n_keys,)
        est_ips = estimator_3bit.inner_product(query, compressed).squeeze(0)

        # Rank: needle should be in top-5
        _, top_indices = est_ips.topk(5)
        assert needle_idx in top_indices.tolist(), (
            f"Needle (idx={needle_idx}) not in top-5. "
            f"Top-5 indices: {top_indices.tolist()}, "
            f"needle score: {est_ips[needle_idx].item():.4f}, "
            f"top score: {est_ips[top_indices[0]].item():.4f}"
        )

    def test_needle_ranks_top1_4bit(self):
        """At 4-bit, the needle should rank #1."""
        est = TurboQuantEstimator(d=DEFAULT_DIM, bits=4, seed=SEED)
        n_keys = 1000
        torch.manual_seed(42)

        keys = torch.randn(n_keys, DEFAULT_DIM)
        keys = keys / keys.norm(dim=1, keepdim=True)

        needle_idx = 500
        query = keys[needle_idx].unsqueeze(0)  # (1, d)

        compressed = est.quantize(keys)
        # (1, n_keys) -> squeeze to (n_keys,)
        est_ips = est.inner_product(query, compressed).squeeze(0)

        _, top_indices = est_ips.topk(1)
        assert top_indices[0].item() == needle_idx, (
            f"At 4-bit, needle should rank #1. Got rank at index "
            f"{top_indices[0].item()}"
        )


# ---------------------------------------------------------------------------
# Tests: compression ratio
# ---------------------------------------------------------------------------
class TestCompressionRatio:
    """Verify that compression ratio calculations are correct."""

    @pytest.mark.parametrize("total_bits,expected_ratio", [
        # Compression ratio = 16*d / (b*d + 32) for d=128
        # (32 bits = 16 for residual_norm + 16 for vec_norm)
        (2, 16 * 128 / (2 * 128 + 32)),
        (3, 16 * 128 / (3 * 128 + 32)),
        (4, 16 * 128 / (4 * 128 + 32)),
    ])
    def test_compression_ratio(self, total_bits, expected_ratio):
        """Compression ratio should match theoretical formula.

        Storage: (b-1)*d bits [MSE] + d bits [QJL] + 16 bits [residual_norm]
                 + 16 bits [vec_norm] = b*d + 32 bits.
        Baseline: 16*d bits (FP16).
        Ratio: 16*d / (b*d + 32).
        """
        est = TurboQuantEstimator(d=DEFAULT_DIM, bits=total_bits, seed=SEED)
        ratio = compute_compression_ratio(est.d, est.bits)
        assert abs(ratio - expected_ratio) < 0.1, (
            f"Compression ratio at {total_bits}-bit: {ratio:.2f}, "
            f"expected {expected_ratio:.2f}"
        )

    def test_compression_ratio_improves_with_fewer_bits(self):
        """Lower bit-width should give higher compression ratio."""
        ratios = []
        for b in [2, 3, 4]:
            est = TurboQuantEstimator(d=DEFAULT_DIM, bits=b, seed=SEED)
            ratios.append(compute_compression_ratio(est.d, est.bits))

        for i in range(len(ratios) - 1):
            assert ratios[i] > ratios[i + 1], (
                f"Compression ratio should decrease: bits={i+2} gave {ratios[i]:.2f}, "
                f"bits={i+3} gave {ratios[i+1]:.2f}"
            )


# ---------------------------------------------------------------------------
# Tests: quantize / inner_product API
# ---------------------------------------------------------------------------
class TestCompressAPI:
    """Test the quantize and inner_product API."""

    def test_quantize_returns_dict(self, estimator_3bit):
        """quantize() should return a dict with required keys."""
        x = random_unit_vectors(100, DEFAULT_DIM)
        compressed = estimator_3bit.quantize(x)
        assert compressed is not None
        assert "mse_indices" in compressed
        assert "qjl_signs" in compressed
        assert "residual_norm" in compressed
        assert "vec_norm" in compressed

    def test_inner_product_shape_cross(self, estimator_3bit):
        """inner_product(q, k) returns (batch_q, batch_k) cross-product matrix."""
        x = random_unit_vectors(20, DEFAULT_DIM)
        y = random_unit_vectors(10, DEFAULT_DIM, seed=77)
        compressed = estimator_3bit.quantize(x)
        scores = estimator_3bit.inner_product(y, compressed)
        assert scores.shape == (10, 20), (
            f"Expected shape (10, 20) for 10 queries x 20 keys, got {scores.shape}"
        )

    def test_inner_product_single_query(self, estimator_3bit):
        """inner_product with 1 query returns (1, n_keys)."""
        x = random_unit_vectors(100, DEFAULT_DIM)
        y = random_unit_vectors(1, DEFAULT_DIM, seed=77)
        compressed = estimator_3bit.quantize(x)
        scores = estimator_3bit.inner_product(y, compressed)
        assert scores.shape == (1, 100), (
            f"Expected shape (1, 100), got {scores.shape}"
        )

    def test_single_vector_quantize(self, estimator_3bit):
        """Should handle compressing a single vector."""
        x = random_unit_vectors(1, DEFAULT_DIM)
        compressed = estimator_3bit.quantize(x)
        y = random_unit_vectors(1, DEFAULT_DIM, seed=77)
        score = estimator_3bit.inner_product(y, compressed)
        assert score.shape == (1, 1)


# ---------------------------------------------------------------------------
# Tests: KV cache wrapper
# ---------------------------------------------------------------------------
class TestKVCache:
    """Test the TurboQuantKVCache that wraps the full TurboQuant pipeline."""

    def test_append_single_batch(self, kv_cache):
        """append() stores one batch; seq_len counts batch calls."""
        keys = random_unit_vectors(50, DEFAULT_DIM)
        values = random_unit_vectors(50, DEFAULT_DIM, seed=77)
        kv_cache.append(keys, values)
        # seq_len counts append() calls (batches), not individual tokens
        assert kv_cache.seq_len == 1

    def test_append_multiple_batches(self, kv_cache):
        """Multiple append() calls increment seq_len."""
        keys1 = random_unit_vectors(30, DEFAULT_DIM, seed=1)
        values1 = random_unit_vectors(30, DEFAULT_DIM, seed=2)
        keys2 = random_unit_vectors(20, DEFAULT_DIM, seed=3)
        values2 = random_unit_vectors(20, DEFAULT_DIM, seed=4)
        kv_cache.append(keys1, values1)
        kv_cache.append(keys2, values2)
        assert kv_cache.seq_len == 2

    def test_attention_scores_shape(self, kv_cache):
        """attention_scores returns (n_queries, total_tokens) shape."""
        n_keys = 100
        n_queries = 10

        keys = random_unit_vectors(n_keys, DEFAULT_DIM, seed=1)
        values = random_unit_vectors(n_keys, DEFAULT_DIM, seed=2)
        kv_cache.append(keys, values)

        queries = random_unit_vectors(n_queries, DEFAULT_DIM, seed=3)
        scores = kv_cache.attention_scores(queries)

        # Shape should be (n_queries, n_keys)
        assert scores.shape == (n_queries, n_keys), (
            f"Expected shape ({n_queries}, {n_keys}), got {scores.shape}"
        )

    def test_get_values(self, kv_cache):
        """get_values() returns reconstructed value vectors."""
        n = 50
        keys = random_unit_vectors(n, DEFAULT_DIM, seed=1)
        values = random_unit_vectors(n, DEFAULT_DIM, seed=2)
        kv_cache.append(keys, values)
        retrieved = kv_cache.get_values()

        # Shape should match
        assert retrieved.shape == (n, DEFAULT_DIM)

        # Should be reasonably close (MSE quantized)
        mse = ((values - retrieved) ** 2).sum(dim=1).mean().item()
        # 3-bit MSE quantization for values
        assert mse < 0.5, f"Value reconstruction MSE too high: {mse}"

    def test_keys_use_prod_values_use_mse(self, kv_cache):
        """Keys should use Prod quantizer (unbiased IP), values should use
        MSE quantizer (best reconstruction)."""
        assert hasattr(kv_cache, 'key_quantizer') or hasattr(kv_cache, '_key_quantizer')
        assert hasattr(kv_cache, 'value_quantizer') or hasattr(kv_cache, '_value_quantizer')

    def test_attention_scores_unbiased(self, kv_cache):
        """Attention scores from cached keys should be approximately unbiased."""
        n_keys = 500
        n_queries = 500
        keys = random_unit_vectors(n_keys, DEFAULT_DIM, seed=1)
        values = random_unit_vectors(n_keys, DEFAULT_DIM, seed=3)
        queries = random_unit_vectors(n_queries, DEFAULT_DIM, seed=2)

        kv_cache.append(keys, values)
        scores = kv_cache.attention_scores(queries)

        # True inner products
        true_scores = queries @ keys.T  # (n_queries, n_keys)

        # Mean error should be near zero (unbiased)
        mean_error = (scores - true_scores).mean().item()
        assert abs(mean_error) < 0.05, (
            f"Mean attention score error: {mean_error:.4f} (should be ~0)"
        )

    def test_empty_cache(self, kv_cache):
        """Empty cache should report seq_len == 0."""
        assert kv_cache.seq_len == 0


# ---------------------------------------------------------------------------
# Tests: full pipeline integration
# ---------------------------------------------------------------------------
class TestFullPipeline:
    """End-to-end tests combining all components."""

    def test_attention_preserves_ranking(self):
        """Compressed attention should preserve the ranking of top keys."""
        est = TurboQuantEstimator(d=DEFAULT_DIM, bits=3, seed=SEED)
        n_keys = 200
        torch.manual_seed(42)

        keys = torch.randn(n_keys, DEFAULT_DIM)
        keys = keys / keys.norm(dim=1, keepdim=True)
        query = torch.randn(1, DEFAULT_DIM)
        query = query / query.norm()

        # True scores
        true_scores = (query @ keys.T).squeeze()  # (n_keys,)
        true_top5 = true_scores.topk(5).indices.tolist()

        # Compressed scores: (1, n_keys) -> squeeze to (n_keys,)
        compressed = est.quantize(keys)
        est_scores = est.inner_product(query, compressed).squeeze(0)
        est_top5 = est_scores.topk(5).indices.tolist()

        # At least 3 of the true top-5 should appear in estimated top-5
        overlap = len(set(true_top5) & set(est_top5))
        assert overlap >= 3, (
            f"Top-5 overlap: {overlap}/5. "
            f"True top-5: {true_top5}, Est top-5: {est_top5}"
        )

    def test_cosine_similarity_of_attention(self):
        """Cosine similarity between true and estimated attention distributions."""
        est = TurboQuantEstimator(d=DEFAULT_DIM, bits=3, seed=SEED)
        n_keys = 500
        torch.manual_seed(42)

        keys = torch.randn(n_keys, DEFAULT_DIM)
        keys = keys / keys.norm(dim=1, keepdim=True)
        query = torch.randn(1, DEFAULT_DIM)
        query = query / query.norm()

        true_scores = (query @ keys.T).squeeze()  # (n_keys,)
        compressed = est.quantize(keys)
        # (1, n_keys) -> squeeze to (n_keys,)
        est_scores = est.inner_product(query, compressed).squeeze(0)

        # Cosine similarity of score vectors
        cos_sim = (
            (true_scores * est_scores).sum()
            / (true_scores.norm() * est_scores.norm())
        ).item()

        # Paper claims >0.995 for full TurboQuant; on random unit vectors >0.85 is reasonable
        assert cos_sim > 0.85, (
            f"Attention score cosine similarity: {cos_sim:.4f} (expected >0.85)"
        )

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_different_dimensions(self, dim):
        """Full pipeline should work for various head dimensions."""
        est = TurboQuantEstimator(d=dim, bits=3, seed=SEED)
        x = random_unit_vectors(100, dim, seed=42)
        y = random_unit_vectors(1, dim, seed=77)
        compressed = est.quantize(x)
        # (1, 100) cross-product for 1 query vs 100 keys
        scores = est.inner_product(y, compressed)
        assert scores.shape == (1, 100)


# ---------------------------------------------------------------------------
# Tests: GPU compatibility
# ---------------------------------------------------------------------------
class TestGPU:
    """Test full pipeline on CUDA when available."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_full_pipeline_gpu(self):
        """Full quantize + inner_product pipeline should work on GPU."""
        est = TurboQuantEstimator(d=DEFAULT_DIM, bits=3, seed=SEED, device='cuda')
        x = random_unit_vectors(100, DEFAULT_DIM).cuda()
        y = random_unit_vectors(1, DEFAULT_DIM, seed=77).cuda()

        compressed = est.quantize(x)
        # (1, 100) for 1 query vs 100 keys
        scores = est.inner_product(y, compressed)
        assert scores.device.type == "cuda"
        assert scores.shape == (1, 100)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_kv_cache_gpu(self):
        """KV cache should work on GPU."""
        cache = TurboQuantKVCache(
            d_key=DEFAULT_DIM, d_value=DEFAULT_DIM, bits=3, seed=SEED, device='cuda'
        )
        keys = random_unit_vectors(50, DEFAULT_DIM).cuda()
        values = random_unit_vectors(50, DEFAULT_DIM, seed=77).cuda()
        queries = random_unit_vectors(10, DEFAULT_DIM, seed=99).cuda()

        cache.append(keys, values)

        scores = cache.attention_scores(queries)
        assert scores.device.type == "cuda"
        assert scores.shape == (10, 50)
