"""TurboQuantDC -- TurboQuant KV cache compression for LLMs.

A from-scratch implementation of Google's TurboQuant algorithm (ICLR 2026)
for compressing key-value caches to 3-bit with <0.5% attention quality loss.

Modules:
    codebook         -- Lloyd-Max optimal scalar quantizer
    rotation         -- Random orthogonal rotation and QJL projection matrices
    polarquant       -- Stage 1: MSE-optimal vector quantization
    qjl              -- Stage 2: 1-bit QJL bias correction
    estimator        -- Combined unbiased inner product estimator
    kv_cache         -- Drop-in compressed KV cache wrapper
    vllm_integration -- vLLM attention backend and cache manager
"""

__version__ = "0.3.0"

import sys

_MIN_PYTHON = (3, 10)
if sys.version_info < _MIN_PYTHON:
    raise RuntimeError(
        f"TurboQuantDC requires Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+, "
        f"but you are running {sys.version_info.major}.{sys.version_info.minor}"
    )


# ---------------------------------------------------------------------------
# Helper for optional dependency errors
# ---------------------------------------------------------------------------
def _optional_import_error(module_name: str, extra: str, pkg: str) -> None:
    """Raise an ImportError with install instructions for an optional dep."""
    raise ImportError(
        f"'{module_name}' requires '{pkg}' which is not installed. "
        f"Install it with:  pip install turboquantdc[{extra}]"
    )


# ---------------------------------------------------------------------------
# Core modules (always available -- depend only on torch + scipy)
# ---------------------------------------------------------------------------
from .asymmetric import (
    PRESETS as ASYMMETRIC_PRESETS,
)
from .asymmetric import (
    AsymmetricKVCache,
    AsymmetricTurboQuantCache,
    AsymmetricTurboQuantLayer,
    analyze_kv_norms,
    create_asymmetric_cache,
)
from .chunked_prefill import ChunkedPrefillEngine
from .codebook import LloydMaxCodebook, beta_pdf, gaussian_pdf, solve_lloyd_max
from .e8_lattice import E8Quantizer, nearest_e8, nearest_e8_relaxed, calibrate_scale
from .e8p_codec import E8PCodec
from .custom_attention import patch_model_attention, turboquant_attention
from .estimator import TurboQuantEstimator
from .fused_attention import (
    compute_norm_correction,
    fused_mse_attention,
    fused_turboquant_attention,
    patch_model_fused_attention,
)
from .generation_cache import (
    ANCHOR_STRATEGIES,
    GenerationCache,
    HybridCache,
    _compute_attention_entropy,
    compute_anchor_schedule,
    compute_layer_key_bits,
)
from .kv_cache import TurboQuantKVCache
from .layer_adaptive import FP16Cache, LayerAdaptiveKVCache, estimate_memory, recommended_schedule
from .outlier import OutlierTurboQuant
from .polarquant import PolarQuant
from .qjl import QJL

# Block-diagonal rotations (Givens/Quaternion)
from .block_rotation import GivensRotation, QuaternionRotation

# Phase 5: Beyond the Paper -- core extensions (torch-only)
from .rotation import (
    apply_wht_rotation,
    fast_wht,
    generate_qjl_matrix,
    generate_rotation_matrix,
    generate_wht_rotation,
)
from .sparse_v import SparseVAttention, sparse_attention
from .streaming import StreamingInferenceEngine
from .temporal_decay import TemporalDecayCache

GENERATION_PRESETS = GenerationCache.PRESETS
from .adaptive_bits import AdaptiveBitsCache, ImportanceScorer
from .adaptive_generation_cache import AdaptiveGenerationCache
from .expected_attention import ExpectedAttentionScorer, ExpectedAttentionCache
from .attention_optimal import MeanRemovedQuantizer
from .channel_adaptive import (
    ChannelAdaptiveCache,
    ChannelAdaptivePolarQuant,
    analyze_channel_sensitivity,
    get_channel_priority,
)
from .cross_layer_kv import (
    CrossLayerKVCache,
    correlation_report,
    measure_cross_layer_kv_correlation,
    measure_distribution_similarity,
)
from .delta_quant import DeltaQuantEncoder
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
from .learned_quant import LearnedQuantizer, calibrate_all_layers
from .learned_rotation import PCARotatedQuantizer, compute_pca_rotation

# ---------------------------------------------------------------------------
# v0.3.0: New research modules
# ---------------------------------------------------------------------------
from .residual_quant import ResidualQuantCache, ResidualQuantEstimator, ResidualQuantLayer
from .residual_vq import ResidualVQ, ResidualVQCache, ResidualVQLayer
from .self_correcting_cache import SelfCorrectingCache
from .token_eviction import EvictionCache
from .ultra_compress import AttentionGatedCache
from .ultra_value_quant import (
    UltraValueCache,
    UltraValueQuantizer,
    compute_value_layer_schedule,
    sweep_value_bits,
)
from .v2_cache import TurboQuantV2Cache
from .weight_compression import (
    CompressedLinear,
    TurboQuantWeightCompressor,
    compress_model,
    compute_weight_bit_schedule,
    effective_bpw,
    estimate_compressed_size,
)

try:
    from .retrieval_cache import RetrievalKVCache
except ImportError:
    pass  # faiss-gpu not installed

try:
    from .pca_code_retrieval import PCACodeIndex
except ImportError:
    pass

# ---------------------------------------------------------------------------
# HuggingFace integration (transformers imported lazily inside the module,
# so importing the *class* here is safe -- it only fails when you actually
# instantiate it without transformers installed)
# ---------------------------------------------------------------------------
from .hf_integration import TurboQuantCache

# ---------------------------------------------------------------------------
# Streaming 70B / Ultra-Streaming (transformers imported lazily inside)
# ---------------------------------------------------------------------------
from .streaming_70b import (
    AsyncPrefetcher,
    LayerGPUCache,
    MemoryPlanner,
    StreamingModel,
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

# ---------------------------------------------------------------------------
# vLLM integration (vLLM imported lazily inside the module)
# ---------------------------------------------------------------------------
from .vllm_integration import (
    TurboQuantAttentionBackend,
    TurboQuantCacheManager,
    get_turboquant_config,
)

# ---------------------------------------------------------------------------
# Triton kernels (optional -- requires triton + CUDA)
# ---------------------------------------------------------------------------
try:
    from .triton_kernels import (
        TritonTurboQuant,
        triton_wht_rotate,
        triton_wht_unrotate,
    )
except (ImportError, RuntimeError):
    # Provide a stub that gives a helpful error when someone tries to use it
    def _triton_not_available(*args, **kwargs):
        _optional_import_error("triton_kernels", "triton", "triton")

    class TritonTurboQuant:  # type: ignore[no-redef]
        """Stub -- triton is not installed."""

        def __init__(self, *args, **kwargs):
            _optional_import_error("TritonTurboQuant", "triton", "triton")

    triton_wht_rotate = _triton_not_available  # type: ignore[assignment]
    triton_wht_unrotate = _triton_not_available  # type: ignore[assignment]


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
    # Version
    "__version__",
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
    # Adaptive Generation Cache (unified system)
    "AdaptiveGenerationCache",
    # v0.3.0: ResidualQuant (improved Stage 2)
    "ResidualQuantEstimator",
    "ResidualQuantCache",
    "ResidualQuantLayer",
    # v0.3.0: Adaptive Bit Allocation
    "AdaptiveBitsCache",
    "ImportanceScorer",
    # v0.3.0: Expected Attention Pruning
    "ExpectedAttentionScorer",
    "ExpectedAttentionCache",
    # v0.3.0: DeltaQuant (cross-token delta coding)
    "DeltaQuantEncoder",
    # v0.3.0: PCA-Adaptive Rotation
    "PCARotatedQuantizer",
    "compute_pca_rotation",
    # v0.3.0: TurboQuantV2Cache (unified pipeline)
    "TurboQuantV2Cache",
    # v0.3.0: Attention-Gated 1-bit Cache
    "AttentionGatedCache",
    # v0.3.0: Mean-Removed Quantization
    "MeanRemovedQuantizer",
    # v0.3.0: Learned (differentiable attention-optimal) Quantization
    "LearnedQuantizer",
    "calibrate_all_layers",
    # v0.3.0: Retrieval Cache (requires faiss-gpu)
    "RetrievalKVCache",
    "PCACodeIndex",
    # Block-diagonal rotations
    "GivensRotation",
    "QuaternionRotation",
    # Triton Kernels (optional)
    "TritonTurboQuant",
    "triton_wht_rotate",
    "triton_wht_unrotate",
]
