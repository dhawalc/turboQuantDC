"""CUDA backend for TurboQuantDC kernels.

Provides raw CUDA dequantize and WHT kernels as a drop-in alternative
to the Triton backend, targeting SM 89 (RTX 4090).
"""
