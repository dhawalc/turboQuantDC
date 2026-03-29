"""Tests for the Streaming Layer Inference Engine.

Tests the StreamingInferenceEngine which loads one transformer layer at a time
from CPU to GPU, enabling models of ANY size to run inference on fixed VRAM.

Uses Qwen2.5-3B-Instruct as the test model. Tests cover:
    - Model loading and architecture detection
    - Single token generation
    - Multi-token text generation
    - VRAM bounds (proving only one layer is on GPU at a time)
    - Memory report accuracy
    - KV cache compression
"""

import pytest
import torch

from turboquantdc.streaming import StreamingInferenceEngine


# ---------------------------------------------------------------------------
# Fixture: shared engine instance (expensive to load, reuse across tests)
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


@pytest.fixture(scope="module")
def engine():
    """Load the streaming engine once for all tests in this module."""
    eng = StreamingInferenceEngine(MODEL_NAME, bits=3)
    eng.load_model_streaming()
    return eng


# ---------------------------------------------------------------------------
# Test: initialization and architecture
# ---------------------------------------------------------------------------
class TestStreamingLoads:
    """Engine initializes correctly and reports correct architecture."""

    def test_architecture_info(self, engine):
        """Engine should report correct Qwen2.5-3B architecture."""
        info = engine.architecture_info()
        assert info["num_layers"] == 36
        assert info["hidden_size"] == 2048
        assert info["num_attention_heads"] == 16
        assert info["num_kv_heads"] == 2
        assert info["head_dim"] == 128
        assert info["vocab_size"] == 151936

    def test_layers_on_cpu(self, engine):
        """All transformer layers should be on CPU after loading."""
        for i, layer in enumerate(engine.layers):
            for name, param in layer.named_parameters():
                assert param.device.type == "cpu", (
                    f"Layer {i} param {name} is on {param.device}, expected CPU"
                )

    def test_embeddings_on_gpu(self, engine):
        """Embeddings and lm_head should be on GPU after loading."""
        for name, param in engine.embed_tokens.named_parameters():
            assert param.device.type == "cuda", (
                f"embed_tokens param {name} is on {param.device}"
            )
        for name, param in engine.lm_head.named_parameters():
            assert param.device.type == "cuda", (
                f"lm_head param {name} is on {param.device}"
            )

    def test_rotary_emb_on_gpu(self, engine):
        """Rotary position embeddings should be on GPU."""
        assert engine.rotary_emb is not None
        # Rotary emb uses buffers, not parameters
        for name, buf in engine.rotary_emb.named_buffers():
            assert buf.device.type == "cuda", (
                f"rotary_emb buffer {name} is on {buf.device}"
            )

    def test_layer_count(self, engine):
        """Should have the correct number of layers."""
        assert len(engine.layers) == 36
        assert engine.num_layers == 36

    def test_tq_cache_initialized(self, engine):
        """TurboQuantCache should be initialized."""
        assert engine.tq_cache is not None
        assert engine.tq_cache.bits == 3

    def test_model_size_reasonable(self, engine):
        """Model size should be approximately 5-6 GB for Qwen2.5-3B FP16."""
        total_mb = engine._model_total_bytes / (1024 ** 2)
        assert 4000 < total_mb < 8000, f"Model size {total_mb:.0f} MB unexpected"

    def test_layer_size_reasonable(self, engine):
        """Each layer should be ~140-160 MB for Qwen2.5-3B FP16."""
        layer_mb = engine._layer_size_bytes / (1024 ** 2)
        assert 100 < layer_mb < 250, f"Layer size {layer_mb:.0f} MB unexpected"


# ---------------------------------------------------------------------------
# Test: single token generation
# ---------------------------------------------------------------------------
class TestStreamingSingleToken:
    """Engine can generate a single token."""

    def test_generate_one_token(self, engine):
        """generate_token() should return a valid token ID."""
        # Reset cache for this test
        from turboquantdc.hf_integration import TurboQuantCache
        engine.tq_cache = TurboQuantCache(bits=3, seed=42)

        input_ids = engine.tokenizer("Hello", return_tensors="pt")["input_ids"]
        next_token = engine.generate_token(input_ids, past_seq_len=0)

        assert next_token.shape == (1, 1)
        assert next_token.device.type == "cuda"
        token_id = next_token.item()
        assert 0 <= token_id < engine.vocab_size

    def test_token_is_decodeable(self, engine):
        """Generated token should decode to a string."""
        from turboquantdc.hf_integration import TurboQuantCache
        engine.tq_cache = TurboQuantCache(bits=3, seed=42)

        input_ids = engine.tokenizer("The capital of France is", return_tensors="pt")["input_ids"]
        next_token = engine.generate_token(input_ids, past_seq_len=0)
        decoded = engine.tokenizer.decode(next_token[0])
        assert isinstance(decoded, str)
        assert len(decoded) > 0


# ---------------------------------------------------------------------------
# Test: multi-token generation
# ---------------------------------------------------------------------------
class TestStreamingGeneratesText:
    """Engine can generate coherent multi-token text."""

    def test_generates_multiple_tokens(self, engine):
        """generate() should produce text with multiple new tokens."""
        output = engine.generate("What is 1+1?", max_new_tokens=5)
        assert isinstance(output, str)
        # Output should be longer than just the prompt
        assert len(output) > len("What is 1+1?")

    def test_generates_20_tokens(self, engine):
        """generate() should be able to produce 20 tokens."""
        output = engine.generate("Count from 1 to 10:", max_new_tokens=20)
        assert isinstance(output, str)
        assert engine._tokens_generated > 0
        assert engine._tokens_generated <= 20

    def test_eos_stops_generation(self, engine):
        """Generation should stop at EOS token or max tokens."""
        # Keep max_new_tokens small -- each token requires 36 layer
        # transfers so large values make tests very slow.
        output = engine.generate("Hi!", max_new_tokens=10)
        assert isinstance(output, str)
        # May generate fewer than 10 tokens if EOS hit
        assert engine._tokens_generated <= 10


# ---------------------------------------------------------------------------
# Test: VRAM bounded
# ---------------------------------------------------------------------------
class TestStreamingVRAMBounded:
    """Peak VRAM should prove only one layer is on GPU at a time."""

    def test_peak_vram_bounded(self, engine):
        """Peak VRAM should be much less than full model size.

        VRAM budget should be approximately:
            embed_tokens (~594 MB) + lm_head (~594 MB, possibly tied)
            + one_layer (~147 MB) + activations + overhead

        For Qwen2.5-3B: expect ~700-1500 MB peak, not 5886 MB.
        We use 2x one_layer as the threshold above embeddings.
        """
        # Generate some tokens to exercise the layer streaming
        output = engine.generate("Test VRAM:", max_new_tokens=3)

        report = engine.memory_report()
        peak_mb = report["peak_vram_mb"]
        model_mb = report["model_total_mb"]
        layer_mb = report["one_layer_mb"]

        # Peak should be well below full model
        assert peak_mb < model_mb * 0.5, (
            f"Peak VRAM {peak_mb:.0f} MB >= 50% of model {model_mb:.0f} MB. "
            f"Layers may not be evicted properly."
        )

        # Peak should not exceed embeds + lm_head + 2*layer + margin
        embed_lmhead = report["embeddings_mb"] + report["lm_head_mb"]
        reasonable_peak = embed_lmhead + 2 * layer_mb + 200  # 200 MB for overhead
        assert peak_mb < reasonable_peak, (
            f"Peak VRAM {peak_mb:.0f} MB exceeds expected bound "
            f"{reasonable_peak:.0f} MB (embed+head+2*layer+200)"
        )

    def test_layers_back_on_cpu_after_generation(self, engine):
        """After generation, all layers should be back on CPU."""
        engine.generate("Check CPU:", max_new_tokens=2)
        for i, layer in enumerate(engine.layers):
            for name, param in layer.named_parameters():
                assert param.device.type == "cpu", (
                    f"Layer {i} param {name} still on {param.device} after generation"
                )


# ---------------------------------------------------------------------------
# Test: memory report
# ---------------------------------------------------------------------------
class TestStreamingMemoryReport:
    """Memory report returns accurate and complete data."""

    def test_report_has_all_keys(self, engine):
        """Report should contain all expected keys."""
        engine.generate("Report test:", max_new_tokens=3)
        report = engine.memory_report()
        expected_keys = {
            "embeddings_mb", "lm_head_mb", "one_layer_mb", "kv_cache_mb",
            "peak_vram_mb", "model_total_mb", "compression_factor",
            "num_layers", "tokens_generated", "tokens_per_sec",
            "load_time_sec", "bits",
        }
        assert expected_keys.issubset(report.keys())

    def test_report_values_positive(self, engine):
        """All memory values should be positive after generation."""
        engine.generate("Positive test:", max_new_tokens=3)
        report = engine.memory_report()
        assert report["embeddings_mb"] > 0
        assert report["one_layer_mb"] > 0
        assert report["peak_vram_mb"] > 0
        assert report["model_total_mb"] > 0
        assert report["compression_factor"] > 1.0

    def test_report_num_layers(self, engine):
        """Report should correctly identify 36 layers."""
        report = engine.memory_report()
        assert report["num_layers"] == 36

    def test_report_bits(self, engine):
        """Report should reflect the configured bit-width."""
        report = engine.memory_report()
        assert report["bits"] == 3

    def test_report_tokens_generated(self, engine):
        """Report should track tokens generated."""
        engine.generate("Token count:", max_new_tokens=5)
        report = engine.memory_report()
        assert report["tokens_generated"] > 0
        assert report["tokens_generated"] <= 5


# ---------------------------------------------------------------------------
# Test: KV cache compression
# ---------------------------------------------------------------------------
class TestStreamingKVCompression:
    """KV cache is actually compressed (smaller than FP16)."""

    def test_kv_cache_populated(self, engine):
        """After generation, TQ cache should contain data."""
        engine.generate("Cache test:", max_new_tokens=5)
        assert engine.tq_cache.is_initialized
        assert len(engine.tq_cache) > 0

    def test_kv_cache_seq_length(self, engine):
        """Cache should have correct sequence length."""
        engine.generate("Length test:", max_new_tokens=5)
        # Cache should have prompt_tokens + generated_tokens entries
        seq_len = engine.tq_cache.get_seq_length(0)
        assert seq_len > 0

    def test_kv_compression_ratio(self, engine):
        """Cache should achieve compression vs FP16 baseline."""
        engine.generate("Compress:", max_new_tokens=5)
        savings = engine.tq_cache.memory_savings()
        ratio = savings["overall_compression_ratio"]
        # 3-bit TurboQuant should compress at least 2x vs FP16
        assert ratio > 2.0, f"Compression ratio {ratio:.2f}x is too low"

    def test_kv_cache_has_all_layers(self, engine):
        """Cache should have entries for all transformer layers."""
        engine.generate("All layers:", max_new_tokens=3)
        # The cache should have been populated for all 36 layers
        assert len(engine.tq_cache) == engine.num_layers


# ---------------------------------------------------------------------------
# Test: parameter validation
# ---------------------------------------------------------------------------
class TestStreamingValidation:
    """Parameter validation and error handling."""

    def test_invalid_bits(self):
        """bits outside {2,3,4} should raise ValueError."""
        with pytest.raises(ValueError, match="bits must be"):
            StreamingInferenceEngine(MODEL_NAME, bits=5)
        with pytest.raises(ValueError, match="bits must be"):
            StreamingInferenceEngine(MODEL_NAME, bits=1)

    def test_generate_before_load_raises(self):
        """generate() before load_model_streaming() should raise."""
        eng = StreamingInferenceEngine(MODEL_NAME, bits=3)
        with pytest.raises(RuntimeError, match="not loaded"):
            eng.generate("test")
