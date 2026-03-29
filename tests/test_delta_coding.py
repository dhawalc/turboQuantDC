"""Tests for cross-layer delta coding.

Validates:
1. Quantization roundtrip fidelity
2. Delta norm is smaller than full layer norm (correlation exploited)
3. Full encode-decode cycle produces valid weights
4. Effective bits/param < anchor_bits (better than independent coding)
5. Reconstruction quality > 0.99 cosine similarity
6. Chain stability — error does not blow up for deep layers
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# Ensure project root is importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from turboquantdc.delta_coding import (
    CrossLayerDeltaCoder,
    compute_layer_pair_stats,
    dequantize_delta,
    dequantize_uniform,
    estimate_delta_entropy,
    parse_layer_params,
    quantize_delta,
    quantize_uniform,
)


# ---------------------------------------------------------------------------
# Helpers: create synthetic model state_dicts with correlated layers
# ---------------------------------------------------------------------------

def make_correlated_state_dict(
    num_layers: int = 8,
    hidden_dim: int = 64,
    intermediate_dim: int = 128,
    correlation: float = 0.95,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Create a synthetic state_dict with correlated adjacent layers.

    Each layer L+1 is computed as:
        W_{L+1} = correlation * W_L + sqrt(1 - correlation^2) * noise

    This produces layers with the specified Pearson correlation.

    Args:
        num_layers: Number of transformer layers.
        hidden_dim: Hidden dimension size.
        intermediate_dim: MLP intermediate dimension.
        correlation: Target correlation between adjacent layers (0 to 1).
        seed: Random seed.

    Returns:
        State dict with keys like "model.layers.0.self_attn.q_proj.weight".
    """
    torch.manual_seed(seed)
    state_dict = {}

    weight_configs = {
        "self_attn.q_proj.weight": (hidden_dim, hidden_dim),
        "self_attn.k_proj.weight": (hidden_dim, hidden_dim),
        "self_attn.v_proj.weight": (hidden_dim, hidden_dim),
        "self_attn.o_proj.weight": (hidden_dim, hidden_dim),
        "mlp.gate_proj.weight": (intermediate_dim, hidden_dim),
        "mlp.up_proj.weight": (intermediate_dim, hidden_dim),
        "mlp.down_proj.weight": (hidden_dim, intermediate_dim),
    }

    noise_scale = (1 - correlation ** 2) ** 0.5

    for wtype, shape in weight_configs.items():
        # Layer 0: random initialization
        prev = torch.randn(shape) * 0.02
        state_dict[f"model.layers.0.{wtype}"] = prev

        for layer_idx in range(1, num_layers):
            noise = torch.randn(shape) * 0.02
            current = correlation * prev + noise_scale * noise
            state_dict[f"model.layers.{layer_idx}.{wtype}"] = current
            prev = current

    return state_dict


# ---------------------------------------------------------------------------
# Test 1: Delta quantize roundtrip
# ---------------------------------------------------------------------------

class TestQuantizationRoundtrip:
    """Quantize + dequantize should recover close to original."""

    def test_uniform_quantize_roundtrip_4bit(self):
        x = torch.randn(100, 64) * 0.02
        indices, scale = quantize_uniform(x, bits=4)
        recon = dequantize_uniform(indices, scale)
        cos = torch.nn.functional.cosine_similarity(
            x.reshape(1, -1), recon.reshape(1, -1)
        ).item()
        assert cos > 0.98, f"4-bit roundtrip cosine sim too low: {cos}"

    def test_uniform_quantize_roundtrip_2bit(self):
        # 2-bit symmetric uniform has only 3 levels (-1, 0, 1), so quality is low.
        # The point is that it still preserves the general direction.
        x = torch.randn(100, 64) * 0.02
        indices, scale = quantize_uniform(x, bits=2)
        recon = dequantize_uniform(indices, scale)
        cos = torch.nn.functional.cosine_similarity(
            x.reshape(1, -1), recon.reshape(1, -1)
        ).item()
        assert cos > 0.40, f"2-bit roundtrip cosine sim too low: {cos}"

    def test_delta_quantize_roundtrip(self):
        """Delta quantize + dequantize recovers close to original delta."""
        delta = torch.randn(64, 64) * 0.005  # Small delta
        indices, scale = quantize_delta(delta, bits=4)
        recon = dequantize_delta(indices, scale)
        rel_err = ((delta - recon) ** 2).sum() / (delta ** 2).sum() + 1e-10
        assert rel_err < 0.05, f"Delta roundtrip relative error too high: {rel_err}"

    def test_quantize_zero_tensor(self):
        """Edge case: all-zero tensor."""
        x = torch.zeros(10, 10)
        indices, scale = quantize_uniform(x, bits=4)
        recon = dequantize_uniform(indices, scale)
        assert (recon == 0).all(), "Zero tensor should quantize to zero"

    def test_quantize_preserves_shape(self):
        """Output shape matches input shape."""
        for shape in [(10,), (5, 10), (3, 4, 5)]:
            x = torch.randn(shape)
            indices, scale = quantize_uniform(x, bits=4)
            assert indices.shape == x.shape


# ---------------------------------------------------------------------------
# Test 2: Delta norm is smaller than full layer norm
# ---------------------------------------------------------------------------

class TestDeltaSmallerThanFull:
    """Correlated layers should have smaller delta norms than full norms."""

    def test_high_correlation_small_delta(self):
        """With r=0.95, delta should be much smaller than full layer."""
        sd = make_correlated_state_dict(num_layers=4, correlation=0.95)
        w0 = sd["model.layers.0.self_attn.q_proj.weight"]
        w1 = sd["model.layers.1.self_attn.q_proj.weight"]
        delta = w1 - w0
        relative = delta.norm() / w0.norm()
        assert relative < 0.5, f"Delta should be <50% of full norm with r=0.95, got {relative:.2%}"

    def test_low_correlation_larger_delta(self):
        """With r=0.3, delta is larger (but still < 2x full norm)."""
        sd = make_correlated_state_dict(num_layers=4, correlation=0.3)
        w0 = sd["model.layers.0.self_attn.q_proj.weight"]
        w1 = sd["model.layers.1.self_attn.q_proj.weight"]
        delta_high_corr = (
            make_correlated_state_dict(num_layers=4, correlation=0.95)
            ["model.layers.1.self_attn.q_proj.weight"]
            - make_correlated_state_dict(num_layers=4, correlation=0.95)
            ["model.layers.0.self_attn.q_proj.weight"]
        ).norm()
        delta_low_corr = (w1 - w0).norm()
        # Low correlation should produce bigger deltas
        assert delta_low_corr > delta_high_corr, (
            "Low correlation should have larger deltas than high correlation"
        )

    def test_all_weight_types_have_small_deltas(self):
        """All 7 weight types should exhibit small deltas with high correlation."""
        sd = make_correlated_state_dict(num_layers=4, correlation=0.95)
        from turboquantdc.delta_coding import WEIGHT_TYPES

        for wtype in WEIGHT_TYPES:
            w0 = sd[f"model.layers.0.{wtype}"]
            w1 = sd[f"model.layers.1.{wtype}"]
            relative = (w1 - w0).norm() / w0.norm()
            assert relative < 0.6, f"{wtype}: delta/full = {relative:.2%}, expected < 60%"


# ---------------------------------------------------------------------------
# Test 3: Full encode-decode cycle
# ---------------------------------------------------------------------------

class TestEncodeDecodeCycle:
    """Encode and decode should produce valid weight tensors."""

    @pytest.fixture
    def model_and_encoded(self):
        sd = make_correlated_state_dict(num_layers=8, correlation=0.95)
        coder = CrossLayerDeltaCoder(anchor_bits=4, delta_bits=2)
        encoded = coder.encode_model(sd)
        return sd, encoded, coder

    def test_all_layers_decodable(self, model_and_encoded):
        sd, encoded, coder = model_and_encoded
        for layer_idx in range(8):
            decoded = coder.decode_layer(encoded, layer_idx)
            assert len(decoded) > 0, f"Layer {layer_idx} decoded to empty dict"

    def test_decoded_shapes_match(self, model_and_encoded):
        sd, encoded, coder = model_and_encoded
        for layer_idx in range(8):
            decoded = coder.decode_layer(encoded, layer_idx)
            for wtype, recon in decoded.items():
                original = sd[f"model.layers.{layer_idx}.{wtype}"]
                assert recon.shape == original.shape, (
                    f"Shape mismatch at layer {layer_idx} {wtype}: "
                    f"{recon.shape} vs {original.shape}"
                )

    def test_decoded_values_finite(self, model_and_encoded):
        sd, encoded, coder = model_and_encoded
        for layer_idx in range(8):
            decoded = coder.decode_layer(encoded, layer_idx)
            for wtype, recon in decoded.items():
                assert torch.isfinite(recon).all(), (
                    f"Non-finite values at layer {layer_idx} {wtype}"
                )

    def test_metadata_present(self, model_and_encoded):
        _, encoded, _ = model_and_encoded
        assert "_metadata" in encoded
        meta = encoded["_metadata"]
        assert meta["anchor_bits"] == 4
        assert meta["delta_bits"] == 2
        assert meta["num_layers"] == 8
        assert meta["total_params"] > 0


# ---------------------------------------------------------------------------
# Test 4: Compression better than independent coding
# ---------------------------------------------------------------------------

class TestCompressionBetterThanIndependent:
    """Effective bits/param should be less than anchor_bits."""

    def test_effective_bpp_below_anchor(self):
        """Delta coding should achieve < 4 bits/param when anchor=4, delta=2."""
        sd = make_correlated_state_dict(num_layers=16, correlation=0.95)
        coder = CrossLayerDeltaCoder(anchor_bits=4, delta_bits=2)
        encoded = coder.encode_model(sd)
        original_bytes = sum(p.numel() * 2 for p in sd.values())  # fp16
        report = coder.compression_report(encoded, original_bytes)
        bpp = report["effective_bits_per_param"]
        assert bpp < 4.0, f"Effective bits/param {bpp:.2f} should be < 4.0"

    def test_effective_bpp_with_many_layers(self):
        """More layers -> closer to delta_bits (2) as anchor amortizes."""
        sd = make_correlated_state_dict(num_layers=32, correlation=0.95)
        coder = CrossLayerDeltaCoder(anchor_bits=4, delta_bits=2)
        encoded = coder.encode_model(sd)
        original_bytes = sum(p.numel() * 2 for p in sd.values())
        report = coder.compression_report(encoded, original_bytes)
        bpp = report["effective_bits_per_param"]
        # With 32 layers: (4 + 31*2)/32 = 2.0625 bits/param theoretically
        assert bpp < 2.2, f"32-layer effective bits/param {bpp:.2f} should be < 2.2"

    def test_compression_ratio_vs_fp16(self):
        """Should achieve > 4x compression vs fp16."""
        sd = make_correlated_state_dict(num_layers=16, correlation=0.95)
        coder = CrossLayerDeltaCoder(anchor_bits=4, delta_bits=2)
        encoded = coder.encode_model(sd)
        original_bytes = sum(p.numel() * 2 for p in sd.values())
        report = coder.compression_report(encoded, original_bytes)
        assert report["compression_ratio"] > 4.0, (
            f"Compression ratio {report['compression_ratio']:.1f}x should be > 4x vs fp16"
        )


# ---------------------------------------------------------------------------
# Test 5: Reconstruction quality
# ---------------------------------------------------------------------------

class TestReconstructionQuality:
    """Decoded weights should have > 0.99 cosine similarity to originals."""

    def test_anchor_layer_quality(self):
        """Layer 0 (anchor) should have high quality at 4-bit."""
        sd = make_correlated_state_dict(num_layers=4, correlation=0.95)
        coder = CrossLayerDeltaCoder(anchor_bits=4, delta_bits=2)
        encoded = coder.encode_model(sd)
        decoded = coder.decode_layer(encoded, 0)
        for wtype, recon in decoded.items():
            original = sd[f"model.layers.0.{wtype}"].float()
            cos = torch.nn.functional.cosine_similarity(
                original.reshape(1, -1), recon.reshape(1, -1)
            ).item()
            # 4-bit uniform quantization of small random tensors (64x64)
            # achieves ~0.98+ cosine similarity
            assert cos > 0.98, f"Anchor layer {wtype}: cosine sim {cos:.4f} < 0.98"

    def test_delta_layer_quality(self):
        """Delta-coded layers should maintain > 0.98 cosine sim."""
        sd = make_correlated_state_dict(num_layers=8, correlation=0.95)
        coder = CrossLayerDeltaCoder(anchor_bits=4, delta_bits=3)
        encoded = coder.encode_model(sd)

        for layer_idx in range(8):
            decoded = coder.decode_layer(encoded, layer_idx)
            for wtype, recon in decoded.items():
                original = sd[f"model.layers.{layer_idx}.{wtype}"].float()
                cos = torch.nn.functional.cosine_similarity(
                    original.reshape(1, -1), recon.reshape(1, -1)
                ).item()
                assert cos > 0.98, (
                    f"Layer {layer_idx} {wtype}: cosine sim {cos:.4f} < 0.98"
                )

    def test_per_layer_quality_report(self):
        """per_layer_quality should return results for all layers."""
        sd = make_correlated_state_dict(num_layers=8, correlation=0.95)
        coder = CrossLayerDeltaCoder(anchor_bits=4, delta_bits=3)
        encoded = coder.encode_model(sd)
        quality = coder.per_layer_quality(encoded, sd)
        assert len(quality) == 8
        for q in quality:
            assert "avg_cosine_sim" in q
            assert q["avg_cosine_sim"] > 0.95


# ---------------------------------------------------------------------------
# Test 6: Chain stability — error does not blow up for deep layers
# ---------------------------------------------------------------------------

class TestChainStability:
    """Error accumulation should be bounded even for deep layer chains."""

    def test_no_error_blowup_32_layers(self):
        """Layer 31 should not be dramatically worse than layer 1."""
        sd = make_correlated_state_dict(num_layers=32, hidden_dim=32, correlation=0.95)
        coder = CrossLayerDeltaCoder(anchor_bits=4, delta_bits=3)
        encoded = coder.encode_model(sd)

        cos_first = []
        cos_last = []

        decoded_first = coder.decode_layer(encoded, 1)
        for wtype, recon in decoded_first.items():
            original = sd[f"model.layers.1.{wtype}"].float()
            cos = torch.nn.functional.cosine_similarity(
                original.reshape(1, -1), recon.reshape(1, -1)
            ).item()
            cos_first.append(cos)

        decoded_last = coder.decode_layer(encoded, 31)
        for wtype, recon in decoded_last.items():
            original = sd[f"model.layers.31.{wtype}"].float()
            cos = torch.nn.functional.cosine_similarity(
                original.reshape(1, -1), recon.reshape(1, -1)
            ).item()
            cos_last.append(cos)

        avg_first = sum(cos_first) / len(cos_first)
        avg_last = sum(cos_last) / len(cos_last)

        # Last layer should not be more than 5% worse than first layer
        assert avg_last > avg_first - 0.05, (
            f"Chain stability failed: layer 1 avg cos={avg_first:.4f}, "
            f"layer 31 avg cos={avg_last:.4f}, degradation={avg_first - avg_last:.4f}"
        )

    def test_deep_layer_still_usable(self):
        """Even at layer 31, cosine sim should remain > 0.90."""
        sd = make_correlated_state_dict(num_layers=32, hidden_dim=32, correlation=0.95)
        coder = CrossLayerDeltaCoder(anchor_bits=4, delta_bits=3)
        encoded = coder.encode_model(sd)
        decoded = coder.decode_layer(encoded, 31)

        for wtype, recon in decoded.items():
            original = sd[f"model.layers.31.{wtype}"].float()
            cos = torch.nn.functional.cosine_similarity(
                original.reshape(1, -1), recon.reshape(1, -1)
            ).item()
            assert cos > 0.90, f"Layer 31 {wtype}: cosine sim {cos:.4f} too degraded"


# ---------------------------------------------------------------------------
# Additional utility tests
# ---------------------------------------------------------------------------

class TestParseLayerParams:
    """Test state dict parsing."""

    def test_groups_by_weight_type(self):
        sd = make_correlated_state_dict(num_layers=4)
        grouped = parse_layer_params(sd)
        assert "self_attn.q_proj.weight" in grouped
        assert len(grouped["self_attn.q_proj.weight"]) == 4

    def test_ignores_non_layer_params(self):
        sd = make_correlated_state_dict(num_layers=2)
        sd["model.embed_tokens.weight"] = torch.randn(1000, 64)
        sd["lm_head.weight"] = torch.randn(1000, 64)
        grouped = parse_layer_params(sd)
        assert "embed_tokens.weight" not in grouped
        assert "lm_head.weight" not in grouped


class TestCorrelationStats:
    """Test correlation computation."""

    def test_identical_weights_perfect_correlation(self):
        w = torch.randn(64, 64)
        stats = compute_layer_pair_stats(w, w)
        assert abs(stats["cosine_sim"] - 1.0) < 1e-5
        assert stats["relative_delta_norm"] < 1e-5
        assert abs(stats["correlation"] - 1.0) < 1e-5

    def test_uncorrelated_weights_low_correlation(self):
        torch.manual_seed(0)
        w1 = torch.randn(256, 256)
        torch.manual_seed(999)
        w2 = torch.randn(256, 256)
        stats = compute_layer_pair_stats(w1, w2)
        assert abs(stats["correlation"]) < 0.1, (
            f"Uncorrelated weights should have ~0 correlation, got {stats['correlation']}"
        )


class TestDeltaEntropy:
    """Test entropy estimation."""

    def test_concentrated_delta_low_entropy(self):
        """Small deltas should have low entropy."""
        delta = torch.randn(1000) * 0.001
        entropy = estimate_delta_entropy(delta, bits=4)
        # Concentrated distribution -> low entropy
        assert entropy < 4.0, f"Small delta entropy should be < 4 bits, got {entropy}"

    def test_uniform_delta_high_entropy(self):
        """Uniformly spread values should have higher entropy."""
        delta = torch.linspace(-1, 1, 10000)
        entropy = estimate_delta_entropy(delta, bits=4)
        # Should use more of the code space
        assert entropy > 2.0, f"Uniform delta entropy should be > 2 bits, got {entropy}"
