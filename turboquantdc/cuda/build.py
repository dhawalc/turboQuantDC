"""Build system for TurboQuantDC CUDA kernels.

Uses torch.utils.cpp_extension for JIT compilation.
Caches compiled modules in ~/.cache/turboquantdc_cuda/.
Falls back to Triton if CUDA compilation fails.

Usage:
    from turboquantdc.cuda.build import load_dequantize, load_wht

    dequantize_mod = load_dequantize()  # Returns compiled module or None
    wht_mod = load_wht()               # Returns compiled module or None
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

# Cache directory for compiled CUDA extensions
_CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "turboquantdc_cuda",
)
os.makedirs(_CACHE_DIR, exist_ok=True)


def _get_cuda_arch_flags() -> list[str]:
    """Detect GPU SM version and return appropriate nvcc arch flags."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        cap = torch.cuda.get_device_capability(0)
        sm = f"{cap[0]}{cap[1]}"
        return [f"-gencode=arch=compute_{sm},code=sm_{sm}"]
    except Exception:
        # Default to SM 89 (RTX 4090)
        return ["-gencode=arch=compute_89,code=sm_89"]


def _common_flags() -> list[str]:
    """Common nvcc compilation flags for all kernels."""
    arch_flags = _get_cuda_arch_flags()
    return [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        *arch_flags,
    ]


def load_dequantize():
    """JIT-compile and load the dequantize CUDA kernel.

    Returns:
        Compiled module with dequantize_mse() and dequantize_residual(),
        or None if compilation fails.
    """
    try:
        from torch.utils.cpp_extension import load

        src_dir = os.path.dirname(os.path.abspath(__file__))
        cu_file = os.path.join(src_dir, "dequantize.cu")

        if not os.path.exists(cu_file):
            logger.warning("dequantize.cu not found at %s", cu_file)
            return None

        mod = load(
            name="turboquantdc_dequantize",
            sources=[cu_file],
            extra_cuda_cflags=_common_flags(),
            build_directory=_CACHE_DIR,
            verbose=False,
        )
        logger.info("CUDA dequantize kernel compiled successfully")
        return mod

    except Exception as e:
        logger.warning("Failed to compile CUDA dequantize kernel: %s", e)
        return None


def load_wht():
    """JIT-compile and load the WHT CUDA kernel.

    Returns:
        Compiled module with wht() function,
        or None if compilation fails.
    """
    try:
        from torch.utils.cpp_extension import load

        src_dir = os.path.dirname(os.path.abspath(__file__))
        cu_file = os.path.join(src_dir, "wht.cu")

        if not os.path.exists(cu_file):
            logger.warning("wht.cu not found at %s", cu_file)
            return None

        mod = load(
            name="turboquantdc_wht",
            sources=[cu_file],
            extra_cuda_cflags=_common_flags(),
            build_directory=_CACHE_DIR,
            verbose=False,
        )
        logger.info("CUDA WHT kernel compiled successfully")
        return mod

    except Exception as e:
        logger.warning("Failed to compile CUDA WHT kernel: %s", e)
        return None


# Convenience: try to load both at import time (lazy, cached)
_dequantize_mod = None
_wht_mod = None
_loaded = False


def get_dequantize_module():
    """Get the cached dequantize module, compiling on first call."""
    global _dequantize_mod, _loaded
    if not _loaded:
        _load_all()
    return _dequantize_mod


def get_wht_module():
    """Get the cached WHT module, compiling on first call."""
    global _wht_mod, _loaded
    if not _loaded:
        _load_all()
    return _wht_mod


def _load_all():
    """Load all CUDA modules (called once on first access)."""
    global _dequantize_mod, _wht_mod, _loaded
    _loaded = True
    _dequantize_mod = load_dequantize()
    _wht_mod = load_wht()
