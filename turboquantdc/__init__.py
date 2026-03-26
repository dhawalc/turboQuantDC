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

__all__ = [
    # Codebook
    "beta_pdf",
    "gaussian_pdf",
    "solve_lloyd_max",
    "LloydMaxCodebook",
    # Rotation
    "generate_rotation_matrix",
    "generate_qjl_matrix",
    # Stage 1
    "PolarQuant",
    # Stage 2
    "QJL",
    # Combined
    "TurboQuantEstimator",
    # KV Cache
    "TurboQuantKVCache",
    # vLLM Integration
    "TurboQuantAttentionBackend",
    "TurboQuantCacheManager",
    "get_turboquant_config",
]

__version__ = "0.1.0"
