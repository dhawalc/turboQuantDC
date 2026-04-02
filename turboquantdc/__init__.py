"""TurboQuantDC — TurboQuant KV cache compression for LLMs.

A from-scratch implementation of Google's TurboQuant algorithm (ICLR 2026)
for compressing key-value caches to 3-bit with <0.5% attention quality loss.

Modules:
    codebook         — Lloyd-Max optimal scalar quantizer
    rotation         — Random orthogonal rotation and QJL projection matrices
    polarquant       — Stage 1: MSE-optimal vector quantization
    qjl              — Stage 2: 1-bit QJL bias correction
    estimator        — Combined unbiased inner product estimator
    kv_cache         — Drop-in compressed KV cache wrapper
    vllm_integration — vLLM attention backend and cache manager
"""

from .codebook import LloydMaxCodebook, beta_pdf, gaussian_pdf, solve_lloyd_max
from .estimator import TurboQuantEstimator
from .kv_cache import TurboQuantKVCache
from .polarquant import PolarQuant
from .qjl import QJL
from .rotation import generate_qjl_matrix, generate_rotation_matrix
from .vllm_integration import (
    TurboQuantAttentionBackend,
    TurboQuantCacheManager,
    get_turboquant_config,
)

# Phase 5: Beyond the Paper
from .rotation import apply_wht_rotation, fast_wht, generate_wht_rotation
from .sparse_v import SparseVAttention, sparse_attention
from .outlier import OutlierTurboQuant
from .layer_adaptive import FP16Cache, LayerAdaptiveKVCache, estimate_memory, recommended_schedule
from .temporal_decay import TemporalDecayCache
from .hf_integration import TurboQuantCache
from .custom_attention import turboquant_attention, patch_model_attention
from .fused_attention import (
    fused_turboquant_attention,
    fused_mse_attention,
    compute_norm_correction,
    patch_model_fused_attention,
)
from .streaming import StreamingInferenceEngine
from .chunked_prefill import ChunkedPrefillEngine
from .asymmetric import (
    AsymmetricKVCache,
    AsymmetricTurboQuantCache,
    AsymmetricTurboQuantLayer,
    PRESETS as ASYMMETRIC_PRESETS,
    create_asymmetric_cache,
    analyze_kv_norms,
)
from .generation_cache import (
    GenerationCache,
    HybridCache,
    compute_anchor_schedule,
    compute_layer_key_bits,
    ANCHOR_STRATEGIES,
    _compute_attention_entropy,
)
GENERATION_PRESETS = GenerationCache.PRESETS
from .channel_adaptive import (
    ChannelAdaptiveCache,
    ChannelAdaptivePolarQuant,
    analyze_channel_sensitivity,
    get_channel_priority,
)
from .entropy_coding import (
    ANSEncoder,
    CompressedPolarQuant,
    EntropyEncoder,
    ZlibEncoder,
    compression_opportunity,
    entropy_analysis_sweep,
    measure_index_entropy,
    theoretical_index_entropy,
)
from .token_eviction import EvictionCache
from .self_correcting_cache import SelfCorrectingCache
from .ultra_value_quant import (
    UltraValueQuantizer,
    UltraValueCache,
    compute_value_layer_schedule,
    sweep_value_bits,
)
from .residual_vq import ResidualVQ, ResidualVQCache, ResidualVQLayer
from .weight_compression import (
    CompressedLinear,
    TurboQuantWeightCompressor,
    compress_model,
    compute_weight_bit_schedule,
    effective_bpw,
    estimate_compressed_size,
)
from .streaming_70b import (
    AsyncPrefetcher,
    LayerGPUCache,
    MemoryPlanner,
    StreamingModel,
)
from .cross_layer_kv import (
    CrossLayerKVCache,
    measure_cross_layer_kv_correlation,
    measure_distribution_similarity,
    correlation_report,
)
from .ultra_streaming import (
    KNOWN_ARCHITECTURES,
    KVManager,
    ModelAnalyzer,
    UltraStreamingEngine,
    WeightManager,
    format_plan_report,
    plan_memory,
)


def run_model(
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    context_target: int = 144_000,
    speed_mode: str = "balanced",
    prompt=None,
) -> None:
    """One-line API to run any 70B model on an RTX 4090.

    Convenience wrapper that delegates to run_70b.run_model().
    See ``python run_70b.py --help`` for full CLI options.

    Args:
        model_name: HuggingFace model name or path.
        context_target: Target context length in tokens.
        speed_mode: One of "safe", "balanced", "fast".
        prompt: Optional single prompt (None = interactive chat).
    """
    from run_70b import run_model as _run_model
    _run_model(
        model_name=model_name,
        context_target=context_target,
        speed_mode=speed_mode,
        prompt=prompt,
    )


__all__ = [
    # Codebook
    "beta_pdf",
    "gaussian_pdf",
    "solve_lloyd_max",
    "LloydMaxCodebook",
    # Rotation
    "generate_rotation_matrix",
    "generate_qjl_matrix",
    # Fast Rotation (Walsh-Hadamard)
    "fast_wht",
    "generate_wht_rotation",
    "apply_wht_rotation",
    # Stage 1
    "PolarQuant",
    # Stage 2
    "QJL",
    # Combined
    "TurboQuantEstimator",
    # KV Cache
    "TurboQuantKVCache",
    # Sparse V Attention
    "SparseVAttention",
    "sparse_attention",
    # Fractional Bit Rates
    "OutlierTurboQuant",
    # Layer-Adaptive
    "LayerAdaptiveKVCache",
    "FP16Cache",
    "recommended_schedule",
    "estimate_memory",
    # Temporal Decay
    "TemporalDecayCache",
    # HuggingFace Integration
    "TurboQuantCache",
    # Custom Attention (unbiased inner products)
    "turboquant_attention",
    "patch_model_attention",
    # Fused Attention (compute IP directly from compressed indices)
    "fused_turboquant_attention",
    "fused_mse_attention",
    "compute_norm_correction",
    "patch_model_fused_attention",
    # vLLM Integration
    "TurboQuantAttentionBackend",
    "TurboQuantCacheManager",
    "get_turboquant_config",
    # Streaming Inference
    "StreamingInferenceEngine",
    # Chunked Prefill
    "ChunkedPrefillEngine",
    # Asymmetric K/V Compression
    "AsymmetricKVCache",
    "AsymmetricTurboQuantCache",
    "AsymmetricTurboQuantLayer",
    "ASYMMETRIC_PRESETS",
    "create_asymmetric_cache",
    "analyze_kv_norms",
    # Production Generation Cache
    "GenerationCache",
    "HybridCache",
    "GENERATION_PRESETS",
    "compute_anchor_schedule",
    "compute_layer_key_bits",
    "ANCHOR_STRATEGIES",
    "_compute_attention_entropy",
    # Channel-Adaptive Mixed Precision
    "ChannelAdaptiveCache",
    "ChannelAdaptivePolarQuant",
    "analyze_channel_sensitivity",
    "get_channel_priority",
    # Entropy Coding
    "ANSEncoder",
    "ZlibEncoder",
    "EntropyEncoder",
    "CompressedPolarQuant",
    "measure_index_entropy",
    "theoretical_index_entropy",
    "compression_opportunity",
    "entropy_analysis_sweep",
    # Token Eviction
    "EvictionCache",
    # Self-Correcting Cache (periodic refresh)
    "SelfCorrectingCache",
    # Ultra Value Quantization (1-bit V)
    "UltraValueQuantizer",
    "UltraValueCache",
    "compute_value_layer_schedule",
    "sweep_value_bits",
    # Residual Vector Quantization (2-stage RVQ)
    "ResidualVQ",
    "ResidualVQCache",
    "ResidualVQLayer",
    # Weight Compression (TQ-W)
    "CompressedLinear",
    "TurboQuantWeightCompressor",
    "compress_model",
    "compute_weight_bit_schedule",
    "effective_bpw",
    "estimate_compressed_size",
    # Streaming 70B Engine
    "StreamingModel",
    "LayerGPUCache",
    "MemoryPlanner",
    "AsyncPrefetcher",
    # Unified 70B Launcher
    "run_model",
    # Ultra-Streaming (200B+)
    "KNOWN_ARCHITECTURES",
    "KVManager",
    "ModelAnalyzer",
    "UltraStreamingEngine",
    "WeightManager",
    "format_plan_report",
    "plan_memory",
    # Cross-Layer KV Sharing
    "CrossLayerKVCache",
    "measure_cross_layer_kv_correlation",
    "measure_distribution_similarity",
    "correlation_report",
]

__version__ = "0.2.0"
