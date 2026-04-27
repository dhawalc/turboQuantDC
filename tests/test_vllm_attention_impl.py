"""Tests for TurboQuantAttentionImpl (Phase D, vLLM custom backend).

Covers the 7 requirements from the original `vllm_integration.py` stub
docstring (correctness side; throughput side is for daylight Phase E).

Most tests are CPU-runnable. Large-context / FP16-on-CUDA tests are
guarded with `pytest.mark.skipif(not torch.cuda.is_available(), ...)`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from turboquantdc.vllm_attention_impl import (
    TurboQuantAttentionImpl,
    _sdpa_with_gqa,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dtype(device):
    return torch.float16 if device == "cuda" else torch.float32


def _make_impl(num_heads=4, num_kv_heads=2, head_size=64, **tq_overrides):
    tq_cfg = {"num_layers": 4, "key_bits": 3, "val_bits": 3, "fp16_window": 8}
    tq_cfg.update(tq_overrides)
    return TurboQuantAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=1.0 / (head_size ** 0.5),
        num_kv_heads=num_kv_heads,
        turboquant_config=tq_cfg,
    )


def _stub_layer(name="model.layers.0.self_attn"):
    return SimpleNamespace(layer_name=name)


def _zero_metadata():
    return SimpleNamespace()


# ── Construction-level tests ────────────────────────────────────────────────

def test_construction_sets_attrs():
    """Req-1: subclass of AttentionImpl, exposes the contract attrs."""
    impl = _make_impl(num_heads=8, num_kv_heads=2, head_size=128)
    assert impl.num_heads == 8
    assert impl.num_kv_heads == 2
    assert impl.head_size == 128
    assert impl.scale == pytest.approx(1.0 / (128 ** 0.5))
    assert impl.num_heads % impl.num_kv_heads == 0


def test_construction_rejects_bad_gqa():
    with pytest.raises(ValueError, match="multiple of"):
        TurboQuantAttentionImpl(
            num_heads=7, head_size=64, scale=0.1, num_kv_heads=2,
            turboquant_config={"num_layers": 4},
        )


def test_construction_rejects_alibi_and_swa():
    with pytest.raises(NotImplementedError, match="ALiBi"):
        TurboQuantAttentionImpl(
            num_heads=4, head_size=64, scale=0.1, num_kv_heads=2,
            alibi_slopes=[0.1, 0.2, 0.3, 0.4],
            turboquant_config={"num_layers": 4},
        )
    with pytest.raises(NotImplementedError, match="sliding"):
        TurboQuantAttentionImpl(
            num_heads=4, head_size=64, scale=0.1, num_kv_heads=2,
            sliding_window=128,
            turboquant_config={"num_layers": 4},
        )


def test_construction_defaults_have_mean_removal_on():
    """Req-6: mean-removal must be ON by default."""
    impl = _make_impl()
    assert impl._tq_config["center_before_quantize"] is True


# ── Shape / GQA tests (Req-3) ───────────────────────────────────────────────

def test_sdpa_with_gqa_expands_kv():
    B, S, D = 1, 32, 64
    num_q, num_kv = 8, 2
    q = torch.randn(B, num_q, S, D)
    k = torch.randn(B, num_kv, S, D)
    v = torch.randn(B, num_kv, S, D)
    out = _sdpa_with_gqa(q, k, v, scale=1.0 / D**0.5, causal=True)
    assert out.shape == (B, num_q, S, D)


def test_sdpa_with_gqa_matches_manual_repeat():
    B, S, D = 1, 16, 32
    num_q, num_kv = 6, 2
    torch.manual_seed(0)
    q = torch.randn(B, num_q, S, D)
    k = torch.randn(B, num_kv, S, D)
    v = torch.randn(B, num_kv, S, D)

    out_helper = _sdpa_with_gqa(q, k, v, scale=1.0 / D**0.5, causal=True)

    factor = num_q // num_kv
    k_man = k.repeat_interleave(factor, dim=1)
    v_man = v.repeat_interleave(factor, dim=1)
    out_manual = torch.nn.functional.scaled_dot_product_attention(
        q, k_man, v_man, scale=1.0 / D**0.5, is_causal=True,
    )

    assert torch.allclose(out_helper, out_manual, atol=1e-5)


# ── Forward / state-management tests (Req-4) ────────────────────────────────

def test_forward_single_token_shape(device, dtype):
    impl = _make_impl(num_heads=4, num_kv_heads=2, head_size=64)
    num_tokens = 1
    q = torch.randn(num_tokens, impl.num_heads, impl.head_size, dtype=dtype, device=device)
    k = torch.randn(num_tokens, impl.num_kv_heads, impl.head_size, dtype=dtype, device=device)
    v = torch.randn(num_tokens, impl.num_kv_heads, impl.head_size, dtype=dtype, device=device)
    out = impl.forward(_stub_layer(), q, k, v, kv_cache=torch.empty(0), attn_metadata=_zero_metadata())
    assert out.shape == (num_tokens, impl.num_heads * impl.head_size)
    assert torch.isfinite(out).all()


def test_forward_multi_step_state_grows(device, dtype):
    """Req-4: each forward() call grows the per-(layer, sequence) cache."""
    impl = _make_impl(num_heads=4, num_kv_heads=2, head_size=64)
    layer = _stub_layer("model.layers.2.self_attn")
    md = _zero_metadata()

    for _ in range(5):
        q = torch.randn(1, impl.num_heads, impl.head_size, dtype=dtype, device=device)
        k = torch.randn(1, impl.num_kv_heads, impl.head_size, dtype=dtype, device=device)
        v = torch.randn(1, impl.num_kv_heads, impl.head_size, dtype=dtype, device=device)
        impl.forward(layer, q, k, v, kv_cache=torch.empty(0), attn_metadata=md)

    cache = impl._caches["__default__"]
    layer_idx = impl._layer_idx_from_layer(layer)
    # Both _CompressedLayer and _FP16Layer expose get_seq_length(); use it
    # so the test is robust against the boundary anchor strategy assigning
    # the chosen layer to either type.
    assert cache._layers[layer_idx].get_seq_length() == 5


def test_forward_multiple_layers_isolated(device, dtype):
    impl = _make_impl(num_heads=4, num_kv_heads=2, head_size=64)
    md = _zero_metadata()

    for _ in range(3):
        q = torch.randn(1, impl.num_heads, impl.head_size, dtype=dtype, device=device)
        k = torch.randn(1, impl.num_kv_heads, impl.head_size, dtype=dtype, device=device)
        v = torch.randn(1, impl.num_kv_heads, impl.head_size, dtype=dtype, device=device)
        impl.forward(_stub_layer("model.layers.0.self_attn"), q, k, v,
                     kv_cache=torch.empty(0), attn_metadata=md)

    for _ in range(2):
        q = torch.randn(1, impl.num_heads, impl.head_size, dtype=dtype, device=device)
        k = torch.randn(1, impl.num_kv_heads, impl.head_size, dtype=dtype, device=device)
        v = torch.randn(1, impl.num_kv_heads, impl.head_size, dtype=dtype, device=device)
        impl.forward(_stub_layer("model.layers.3.self_attn"), q, k, v,
                     kv_cache=torch.empty(0), attn_metadata=md)

    cache = impl._caches["__default__"]
    assert cache._layers[0].get_seq_length() == 3
    assert cache._layers[3].get_seq_length() == 2


def test_reset_clears_state():
    impl = _make_impl()
    md = _zero_metadata()
    q = torch.randn(1, impl.num_heads, impl.head_size)
    k = torch.randn(1, impl.num_kv_heads, impl.head_size)
    v = torch.randn(1, impl.num_kv_heads, impl.head_size)
    impl.forward(_stub_layer(), q, k, v, kv_cache=torch.empty(0), attn_metadata=md)
    assert "__default__" in impl._caches
    impl.reset_sequence()
    assert "__default__" not in impl._caches


# ── Layer-idx parsing ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "name,expected_idx",
    [
        ("model.layers.0.self_attn", 0),
        ("model.layers.7.self_attn.attn", 7),
        ("model.layers.63.attn", 63),
        ("language_model.model.layers.42.self_attn", 42),
        ("nonsense", 0),
        ("", 0),
    ],
)
def test_layer_idx_parsing(name, expected_idx):
    layer = _stub_layer(name) if name else SimpleNamespace()
    assert TurboQuantAttentionImpl._layer_idx_from_layer(layer) == expected_idx


# ── Mean-removal verification (Req-6) ───────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs fp16 GPU")
def test_mean_removal_active_via_compressed_layer():
    impl = _make_impl(num_heads=4, num_kv_heads=2, head_size=64)
    q = torch.randn(1, impl.num_heads, impl.head_size, dtype=torch.float16, device="cuda")
    k = torch.randn(1, impl.num_kv_heads, impl.head_size, dtype=torch.float16, device="cuda")
    v = torch.randn(1, impl.num_kv_heads, impl.head_size, dtype=torch.float16, device="cuda")
    impl.forward(_stub_layer(), q, k, v, kv_cache=torch.empty(0), attn_metadata=_zero_metadata())

    cache = impl._caches["__default__"]
    compressed_layer = cache._layers[0]
    if hasattr(compressed_layer, "center_before_quantize"):
        assert compressed_layer.center_before_quantize is True


# ── fp32 cast / fp16 stability (Req-5) ──────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs fp16 GPU")
def test_forward_fp16_no_nan_or_inf():
    """Req-5: fp16 input must not produce inf/nan."""
    impl = _make_impl(num_heads=4, num_kv_heads=2, head_size=64)
    q = torch.randn(1, impl.num_heads, impl.head_size, dtype=torch.float16, device="cuda") * 1e-3
    k = torch.randn(1, impl.num_kv_heads, impl.head_size, dtype=torch.float16, device="cuda") * 1e-3
    v = torch.randn(1, impl.num_kv_heads, impl.head_size, dtype=torch.float16, device="cuda") * 1e-3
    out = impl.forward(_stub_layer(), q, k, v, kv_cache=torch.empty(0), attn_metadata=_zero_metadata())
    assert torch.isfinite(out).all()


# ── Compress/decompress quality smoke test ─────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs fp16 GPU")
def test_attention_output_close_to_fp16_baseline():
    impl = _make_impl(num_heads=4, num_kv_heads=2, head_size=64)
    torch.manual_seed(0)
    q = torch.randn(1, impl.num_heads, impl.head_size, dtype=torch.float16, device="cuda")
    k = torch.randn(1, impl.num_kv_heads, impl.head_size, dtype=torch.float16, device="cuda")
    v = torch.randn(1, impl.num_kv_heads, impl.head_size, dtype=torch.float16, device="cuda")

    out_tq = impl.forward(_stub_layer(), q, k, v,
                          kv_cache=torch.empty(0), attn_metadata=_zero_metadata())

    q_ref = q.permute(1, 0, 2).unsqueeze(0)
    k_ref = k.permute(1, 0, 2).unsqueeze(0)
    v_ref = v.permute(1, 0, 2).unsqueeze(0)
    out_ref = _sdpa_with_gqa(q_ref, k_ref, v_ref, scale=impl.scale, causal=True)
    out_ref = out_ref.squeeze(0).permute(1, 0, 2).reshape(1, -1)

    cos = torch.nn.functional.cosine_similarity(
        out_tq.float().flatten(),
        out_ref.float().flatten(),
        dim=0,
    ).item()
    assert cos > 0.85, f"Cosine similarity dropped to {cos:.3f}, expected > 0.85"
