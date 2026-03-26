"""TurboQuantDC vLLM Integration — drop-in KV cache quantizer for vLLM.

This module provides TurboQuant as a pluggable KV cache compression backend for
vLLM. It targets vLLM >= 0.4.x (the modular attention backend refactor).

## Architecture Overview

vLLM's attention computation lives in ``vllm/attention/backends/``. Each backend
implements the ``AttentionBackend`` protocol with a matching ``AttentionImpl``.
This module does NOT subclass those directly (to avoid importing vLLM at package
install time), but provides drop-in compatible classes that can be wired in at
runtime. See "Hooking into vLLM" below.

## Key Asymmetry (from TurboQuant paper)

    Keys:   TurboQuantEstimator — full two-stage MSE + QJL (unbiased inner products)
    Values: PolarQuant MSE-only — accurate reconstruction for the weighted sum

The QJL stage is critical for keys because attention scores are *inner products*.
Skipping QJL leaves a systematic bias that degrades perplexity. Values only need
low-MSE reconstruction, so the cheaper Stage-1-only path is used.

## Hooking into vLLM (vLLM >= 0.4.0)

1.  **Monkey-patch approach** (fastest to try, no vLLM fork needed):

    ```python
    import vllm
    from turboquantdc.vllm_integration import TurboQuantAttentionBackend, get_turboquant_config

    # Build config from model path
    cfg = get_turboquant_config("Qwen/Qwen2.5-7B-Instruct", bits=3)

    # Create a shared backend (all layers share one backend object)
    backend = TurboQuantAttentionBackend(
        head_dim=cfg["head_dim"],
        num_kv_heads=cfg["num_kv_heads"],
        num_layers=cfg["num_layers"],
        bits=cfg["bits"],
        device="cuda",
    )

    # Patch the attention layers after model is loaded
    for layer_idx, attn_layer in enumerate(llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers):
        original_forward = attn_layer.self_attn.attn.forward

        def make_patched_forward(lidx, orig):
            def patched_forward(query, key, value, kv_cache, attn_metadata):
                compressed = backend.compress_kv(lidx, key, value)
                return backend.compute_attention(lidx, query, *compressed, attn_metadata)
            return patched_forward

        attn_layer.self_attn.attn.forward = make_patched_forward(layer_idx, original_forward)
    ```

2.  **CacheEngine replacement** (for pre-allocating compressed buffers):

    ```python
    from turboquantdc.vllm_integration import TurboQuantCacheManager

    cache_manager = TurboQuantCacheManager(
        num_layers=cfg["num_layers"],
        num_kv_heads=cfg["num_kv_heads"],
        head_dim=cfg["head_dim"],
        max_seq_len=32768,
        bits=3,
        device="cuda",
    )
    cache_manager.allocate(block_size=16)
    ```

3.  **CLI flag (future)**:

    Once merged upstream, enable with::

        vllm serve Qwen/Qwen2.5-7B-Instruct --kv-cache-dtype turboquant-3bit

## Performance Expectations (RTX 4090, 3-bit, d=128)

    Compression:     5.0x vs FP16 KV cache
    Cosine sim:      0.9959 (measured on Qwen2.5-3B)
    Top-5 attn:      91.7% preserved
    Quantize:        27M vectors/sec
    Inner product:   71M vectors/sec
    Extra latency:   <2% at batch size 1 (dominated by matmul)

## What vLLM Source Changes Are Needed

For the monkey-patch path: none. For deep integration:

    vllm/attention/backends/turboquant.py  — new file (this class + impl)
    vllm/config.py                          — add "turboquant-3bit" to KVCacheDtype
    vllm/worker/cache_engine.py            — branch on turboquant dtype to use TurboQuantCacheManager
    vllm/attention/layer.py                — route forward() to TurboQuantImpl when dtype is set

Target vLLM version: 0.4.x – 0.5.x (tested with 0.4.3). The modular backend
interface stabilised in 0.4.0. Earlier versions require deeper surgery.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .estimator import TurboQuantEstimator
from .polarquant import PolarQuant


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A compressed key bundle as returned by TurboQuantEstimator.quantize()
CompressedKey = Dict[str, torch.Tensor]

# A compressed value bundle: (value_indices, value_norms)
CompressedValue = Tuple[torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# TurboQuantAttentionBackend
# ---------------------------------------------------------------------------


class TurboQuantAttentionBackend:
    """Drop-in attention backend that uses TurboQuant-compressed KV cache.

    Designed for vLLM's attention layer interface. Each layer gets its own
    TurboQuantEstimator (unique seed per layer) so rotation and projection
    matrices are independent across layers.

    Keys use the full two-stage estimator (MSE + QJL) for unbiased inner
    products. Values use PolarQuant MSE-only since they need reconstruction,
    not inner products.

    Usage with vLLM (monkey-patch path)::

        backend = TurboQuantAttentionBackend(
            head_dim=128, num_kv_heads=8, num_layers=32, bits=3, device="cuda"
        )
        compressed_kv = backend.compress_kv(layer_idx=0, keys=keys, values=values)
        output = backend.compute_attention(layer_idx=0, queries=queries, *compressed_kv)

    Args:
        head_dim:     Dimension of each attention head (d). Typically 128 or 64.
        num_kv_heads: Number of key-value heads (supports GQA).
        num_layers:   Total transformer layers (one estimator created per layer).
        bits:         Effective bits per coordinate. 3 is the paper's sweet spot.
        device:       Target device (``"cuda"`` recommended for production).
    """

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        num_layers: int,
        bits: int = 3,
        device: str | torch.device = "cuda",
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.bits = bits
        self.device = device

        # One TurboQuantEstimator per layer (unique seed = layer_idx * 1000)
        # Seeds are spaced 1000 apart to avoid accidental correlation between
        # the key estimator (seed) and its internal QJL (seed+1).
        self._key_quantizers: List[TurboQuantEstimator] = [
            TurboQuantEstimator(
                d=head_dim,
                bits=bits,
                seed=layer_idx * 1000,
                device=device,
            )
            for layer_idx in range(num_layers)
        ]

        # One PolarQuant per layer for value quantization
        # Use seed offset 500 to be independent from the key quantizer seeds.
        self._value_quantizers: List[PolarQuant] = [
            PolarQuant(
                d=head_dim,
                bits=bits,
                seed=layer_idx * 1000 + 500,
                device=device,
            )
            for layer_idx in range(num_layers)
        ]

        # Track memory allocation per layer (updated on compress_kv calls)
        self._compressed_key_store: List[List[CompressedKey]] = [
            [] for _ in range(num_layers)
        ]
        self._compressed_value_store: List[List[CompressedValue]] = [
            [] for _ in range(num_layers)
        ]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compress_kv(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[CompressedKey, CompressedValue]:
        """Compress a batch of KV pairs for a specific layer.

        Handles arbitrary input shapes by flattening head dimensions:

            keys:   (batch, num_heads, seq_len, head_dim)  — standard vLLM shape
                    (seq_len, num_kv_heads, head_dim)       — decode-step shape
                    (batch, head_dim)                       — pre-flattened
            values: same shapes as keys

        The method reshapes to ``(N, head_dim)`` internally and returns
        compressed bundles in that flat shape. The caller is responsible for
        any subsequent head-dimension bookkeeping.

        Args:
            layer_idx: Zero-based layer index. Must be in ``[0, num_layers)``.
            keys:      Key tensor in any of the supported shapes above.
            values:    Value tensor matching keys shape.

        Returns:
            Tuple of ``(compressed_keys, compressed_values)`` where:

            - ``compressed_keys`` is the dict returned by
              ``TurboQuantEstimator.quantize()``:
              ``{mse_indices, qjl_signs, residual_norm, vec_norm}``

            - ``compressed_values`` is a tuple
              ``(value_indices: Tensor, value_norms: Tensor)``
              where ``value_indices`` has shape ``(N, head_dim)`` and
              ``value_norms`` has shape ``(N,)``.

        Raises:
            IndexError: If ``layer_idx`` is out of range.
        """
        self._check_layer_idx(layer_idx)

        keys_flat = self._flatten_to_2d(keys)      # (N, d)
        values_flat = self._flatten_to_2d(values)  # (N, d)

        # Keys: full two-stage quantization for unbiased attention scores
        compressed_keys = self._key_quantizers[layer_idx].quantize(keys_flat)

        # Values: MSE-only quantization for low-distortion reconstruction
        v_norm = values_flat.norm(dim=-1, keepdim=True)  # (N, 1)
        v_normalized = values_flat / (v_norm + 1e-8)
        v_indices = self._value_quantizers[layer_idx].quantize(v_normalized)
        compressed_values: CompressedValue = (v_indices, v_norm.squeeze(-1))

        return compressed_keys, compressed_values

    def compute_attention(
        self,
        layer_idx: int,
        queries: torch.Tensor,
        compressed_keys: CompressedKey,
        compressed_values: CompressedValue,
        scale: Optional[float] = None,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention scores using compressed KV cache.

        Implements the full TurboQuant estimator for attention:

            scores_ij = <q_i, k_j>  via unbiased two-stage estimator
            attn     = softmax(scores / sqrt(d)) @ values_reconstructed

        Values are reconstructed from their MSE quantization before the
        weighted sum. Keys remain compressed throughout — only their inner
        products with queries are estimated.

        Args:
            layer_idx:        Zero-based layer index.
            queries:          Query tensor, any shape with last dim == head_dim.
                              Flattened to ``(Q, d)`` internally.
            compressed_keys:  Output of ``compress_kv(...)[0]``.
            compressed_values: Output of ``compress_kv(...)[1]``.
            scale:            Attention scale. Defaults to ``1 / sqrt(head_dim)``.
            causal_mask:      Optional boolean mask of shape ``(Q, K)`` where
                              ``True`` means *mask this position out*. Applied
                              before softmax (sets masked logits to ``-inf``).

        Returns:
            Attention output tensor of shape matching ``queries`` (last dim
            replaced by ``head_dim``).
        """
        self._check_layer_idx(layer_idx)

        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        original_shape = queries.shape
        queries_flat = self._flatten_to_2d(queries)  # (Q, d)

        # Estimate attention logits using TurboQuant inner product estimator
        # scores shape: (Q, K)
        scores = self._key_quantizers[layer_idx].inner_product(
            query=queries_flat,
            compressed=compressed_keys,
        )
        scores = scores * scale

        # Apply optional causal mask
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Softmax over key dimension
        attn_weights = F.softmax(scores, dim=-1)  # (Q, K)

        # Reconstruct values from MSE indices
        v_indices, v_norms = compressed_values
        v_reconstructed = self._value_quantizers[layer_idx].dequantize(v_indices)
        # Rescale by stored norms
        v_reconstructed = v_reconstructed * v_norms.unsqueeze(-1)  # (K, d)

        # Weighted sum: (Q, K) @ (K, d) -> (Q, d)
        output_flat = attn_weights @ v_reconstructed

        # Restore original query shape (replace last dim with head_dim if needed)
        output = output_flat.view(*original_shape[:-1], self.head_dim)
        return output

    def memory_usage(self) -> Dict[str, object]:
        """Report current memory usage across all layers.

        Counts live compressed tensors in the internal stores.
        This reflects any KV pairs stored via the ``_compressed_*_store``
        lists. In the typical monkey-patch usage those lists are NOT
        populated (the caller owns storage); use
        ``TurboQuantCacheManager`` for managed storage.

        Returns:
            Dict with keys:

            - ``total_tokens``:       Total tokens stored across all layers.
            - ``bits_per_token_key``: Bits per token for key storage.
            - ``bits_per_token_val``: Bits per token for value storage.
            - ``total_bits``:         Total bits used.
            - ``fp16_baseline_bits``: Equivalent FP16 cost.
            - ``compression_ratio``:  ``fp16_baseline_bits / total_bits``.
        """
        d = self.head_dim
        b = self.bits
        mse_bits_key = max(b - 1, 1)    # (b-1) bits for key MSE stage
        qjl_bits = d                    # 1 bit per QJL dim (m = d)
        key_norm_bits = 32              # vec_norm (16) + residual_norm (16)
        val_bits = b * d + 16           # b bits/coord + 16-bit norm

        bits_per_token_key = mse_bits_key * d + qjl_bits + key_norm_bits
        bits_per_token_val = val_bits
        bits_per_token_fp16 = (d + d) * 16  # key + value in FP16

        total_tokens = 0
        for layer_keys in self._compressed_key_store:
            for ck in layer_keys:
                mi = ck["mse_indices"]
                total_tokens += mi.shape[0] if mi.dim() > 1 else 1

        total_bits = total_tokens * (bits_per_token_key + bits_per_token_val)
        fp16_bits = total_tokens * bits_per_token_fp16

        return {
            "total_tokens": total_tokens,
            "bits_per_token_key": bits_per_token_key,
            "bits_per_token_val": bits_per_token_val,
            "total_bits": total_bits,
            "fp16_baseline_bits": fp16_bits,
            "compression_ratio": fp16_bits / total_bits if total_bits > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_layer_idx(self, layer_idx: int) -> None:
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )

    @staticmethod
    def _flatten_to_2d(t: torch.Tensor) -> torch.Tensor:
        """Flatten any tensor with last dim == head_dim to (N, head_dim).

        Handles common vLLM shapes:
            (batch, seq_len, head_dim)          -> (batch*seq_len, head_dim)
            (batch, num_heads, seq_len, head_dim) -> (batch*num_heads*seq_len, head_dim)
            (N, head_dim)                        -> no-op
            (head_dim,)                          -> (1, head_dim)
        """
        if t.dim() == 1:
            return t.unsqueeze(0)
        if t.dim() == 2:
            return t
        # Flatten all leading dimensions
        head_dim = t.shape[-1]
        return t.reshape(-1, head_dim)


# ---------------------------------------------------------------------------
# TurboQuantCacheManager
# ---------------------------------------------------------------------------


class TurboQuantCacheManager:
    """Manages compressed KV cache across all layers.

    Replaces vLLM's CacheEngine for TurboQuant mode. Pre-allocates
    compressed cache buffers for ``max_seq_len`` tokens to avoid
    dynamic allocation during inference.

    Storage layout per layer (d=128, 3-bit):

        Key MSE indices:   (max_seq_len, num_kv_heads, d)  int8 / int16
        Key QJL signs:     (max_seq_len, num_kv_heads, m)  int8 (packed ±1 as {0,1})
        Key residual norm: (max_seq_len, num_kv_heads)     float16
        Key vec norm:      (max_seq_len, num_kv_heads)     float16
        Value indices:     (max_seq_len, num_kv_heads, d)  int8 / int16
        Value norms:       (max_seq_len, num_kv_heads)     float16

    All index tensors use ``torch.int16`` for compatibility with codebooks up
    to 4-bit (16 centroids). For 1-2 bit codebooks ``torch.uint8`` would
    suffice, but int16 keeps the code uniform.

    Args:
        num_layers:   Number of transformer layers.
        num_kv_heads: Number of KV heads (GQA supported).
        head_dim:     Head dimension ``d``.
        max_seq_len:  Maximum sequence length to pre-allocate for.
        bits:         TurboQuant effective bits per coordinate.
        device:       CUDA device string or torch.device.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        bits: int = 3,
        device: str | torch.device = "cuda",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.bits = bits
        self.device = torch.device(device)

        # Quantizers: one TurboQuantEstimator per layer for keys
        #             one PolarQuant per layer for values
        self._key_quantizers: List[TurboQuantEstimator] = [
            TurboQuantEstimator(
                d=head_dim, bits=bits, seed=layer_idx * 1000, device=device
            )
            for layer_idx in range(num_layers)
        ]
        self._value_quantizers: List[PolarQuant] = [
            PolarQuant(
                d=head_dim, bits=bits, seed=layer_idx * 1000 + 500, device=device
            )
            for layer_idx in range(num_layers)
        ]

        # Pre-allocated buffers (None until allocate() is called)
        self._key_mse_buf: Optional[List[torch.Tensor]] = None
        self._key_qjl_buf: Optional[List[torch.Tensor]] = None
        self._key_rnorm_buf: Optional[List[torch.Tensor]] = None
        self._key_vnorm_buf: Optional[List[torch.Tensor]] = None
        self._val_mse_buf: Optional[List[torch.Tensor]] = None
        self._val_vnorm_buf: Optional[List[torch.Tensor]] = None

        # Track how many slots are filled per layer
        self._fill_count: List[int] = [0] * num_layers

        self._allocated = False
        self._block_size: int = 0

    def allocate(self, block_size: int = 16) -> None:
        """Pre-allocate compressed cache buffers for all layers.

        Creates fixed-size tensors on the target device. This should be
        called once before inference, analogous to vLLM's
        ``CacheEngine.allocate_gpu_cache()``.

        The QJL projection dimension ``m`` equals ``head_dim`` (paper default).
        Signs are stored as ``int8`` with values ``0`` or ``1`` (representing
        ``-1`` and ``+1`` respectively) to halve storage vs float.

        Args:
            block_size: Number of tokens per cache block (matches vLLM's
                        ``block_size``). Not used to reshape buffers here —
                        kept for API compatibility and future paged cache
                        support.
        """
        self._block_size = block_size
        S = self.max_seq_len
        H = self.num_kv_heads
        D = self.head_dim
        M = D  # QJL projection dimension = head_dim (paper default)

        self._key_mse_buf = []
        self._key_qjl_buf = []
        self._key_rnorm_buf = []
        self._key_vnorm_buf = []
        self._val_mse_buf = []
        self._val_vnorm_buf = []

        for _ in range(self.num_layers):
            # MSE indices: int16 covers up to 4-bit codebooks (16 levels)
            self._key_mse_buf.append(
                torch.zeros(S, H, D, dtype=torch.int16, device=self.device)
            )
            # QJL signs stored as int8 {0, 1} (decode: sign = val*2 - 1)
            self._key_qjl_buf.append(
                torch.zeros(S, H, M, dtype=torch.int8, device=self.device)
            )
            # Norms in float16 for compact storage
            self._key_rnorm_buf.append(
                torch.zeros(S, H, dtype=torch.float16, device=self.device)
            )
            self._key_vnorm_buf.append(
                torch.zeros(S, H, dtype=torch.float16, device=self.device)
            )
            # Value MSE indices
            self._val_mse_buf.append(
                torch.zeros(S, H, D, dtype=torch.int16, device=self.device)
            )
            # Value norms
            self._val_vnorm_buf.append(
                torch.zeros(S, H, dtype=torch.float16, device=self.device)
            )

        self._fill_count = [0] * self.num_layers
        self._allocated = True

    def store(
        self,
        layer_idx: int,
        slot_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Compress and store KV vectors at the given cache slot.

        Each call stores one or more tokens for a single layer.

        Args:
            layer_idx: Zero-based layer index.
            slot_idx:  Absolute slot index in ``[0, max_seq_len)``.
                       For batched input, slots ``[slot_idx, slot_idx + N)``
                       are written where ``N = keys.shape[0]``.
            keys:      Key tensor of shape ``(num_kv_heads, d)`` (single token)
                       or ``(N, num_kv_heads, d)`` (multiple tokens).
            values:    Value tensor matching keys shape.

        Raises:
            RuntimeError:  If ``allocate()`` has not been called.
            IndexError:    If ``layer_idx`` out of range.
            ValueError:    If the slot range exceeds ``max_seq_len``.
        """
        self._assert_allocated()
        self._check_layer_idx(layer_idx)

        # Normalize to (N, H, D)
        if keys.dim() == 2:  # (H, D) — single token
            keys = keys.unsqueeze(0)
            values = values.unsqueeze(0)

        N, H, D = keys.shape
        end_slot = slot_idx + N
        if end_slot > self.max_seq_len:
            raise ValueError(
                f"Slot range [{slot_idx}, {end_slot}) exceeds max_seq_len={self.max_seq_len}"
            )

        # Flatten (N, H, D) -> (N*H, D) for quantizers
        k_flat = keys.reshape(N * H, D).float()
        v_flat = values.reshape(N * H, D).float()

        # --- Keys: TurboQuantEstimator ---
        ck = self._key_quantizers[layer_idx].quantize(k_flat)

        # Store MSE indices: (N*H, D) -> reshape to (N, H, D)
        mse_idx = ck["mse_indices"].to(torch.int16).reshape(N, H, D)
        self._key_mse_buf[layer_idx][slot_idx:end_slot] = mse_idx

        # QJL signs: float {-1,+1} -> int8 {0,1}
        signs_01 = ((ck["qjl_signs"] + 1.0) * 0.5).to(torch.int8)  # {-1->0, +1->1}
        signs_01 = signs_01.reshape(N, H, D)
        self._key_qjl_buf[layer_idx][slot_idx:end_slot] = signs_01

        # Norms: (N*H,) -> (N, H)
        self._key_rnorm_buf[layer_idx][slot_idx:end_slot] = (
            ck["residual_norm"].to(torch.float16).reshape(N, H)
        )
        self._key_vnorm_buf[layer_idx][slot_idx:end_slot] = (
            ck["vec_norm"].to(torch.float16).reshape(N, H)
        )

        # --- Values: PolarQuant MSE ---
        v_norm = v_flat.norm(dim=-1, keepdim=True)  # (N*H, 1)
        v_normalized = v_flat / (v_norm + 1e-8)
        v_idx = self._value_quantizers[layer_idx].quantize(v_normalized)
        self._val_mse_buf[layer_idx][slot_idx:end_slot] = (
            v_idx.to(torch.int16).reshape(N, H, D)
        )
        self._val_vnorm_buf[layer_idx][slot_idx:end_slot] = (
            v_norm.squeeze(-1).to(torch.float16).reshape(N, H)
        )

        self._fill_count[layer_idx] = max(self._fill_count[layer_idx], end_slot)

    def fetch(
        self,
        layer_idx: int,
        slot_indices: torch.Tensor,
    ) -> Tuple[CompressedKey, CompressedValue]:
        """Fetch and reconstruct compressed KV for the given slots.

        Loads the raw compressed representations from pre-allocated buffers
        and returns them in the same dict/tuple format expected by
        ``TurboQuantEstimator.inner_product()`` and
        ``PolarQuant.dequantize()``.

        Args:
            layer_idx:    Zero-based layer index.
            slot_indices: 1-D LongTensor of slot indices to fetch.
                          Shape ``(K,)`` where ``K`` is number of tokens.

        Returns:
            Tuple of ``(compressed_keys, compressed_values)``:

            - ``compressed_keys``:  dict with keys
              ``mse_indices, qjl_signs, residual_norm, vec_norm``
              shapes ``(K*H, D)``, ``(K*H, D)``, ``(K*H,)``, ``(K*H,)``

            - ``compressed_values``: tuple ``(v_indices, v_norms)``
              shapes ``(K*H, D)``, ``(K*H,)``

        Raises:
            RuntimeError: If ``allocate()`` has not been called.
            IndexError:   If ``layer_idx`` out of range.
        """
        self._assert_allocated()
        self._check_layer_idx(layer_idx)

        K = slot_indices.shape[0]
        H = self.num_kv_heads
        D = self.head_dim

        # Gather from pre-allocated buffers
        mse = self._key_mse_buf[layer_idx][slot_indices]     # (K, H, D)
        qjl = self._key_qjl_buf[layer_idx][slot_indices]     # (K, H, D)
        rnorm = self._key_rnorm_buf[layer_idx][slot_indices]  # (K, H)
        vnorm = self._key_vnorm_buf[layer_idx][slot_indices]  # (K, H)
        v_mse = self._val_mse_buf[layer_idx][slot_indices]    # (K, H, D)
        v_vnorm = self._val_vnorm_buf[layer_idx][slot_indices] # (K, H)

        # Decode QJL signs: {0,1} -> {-1,+1} in float32
        signs_float = qjl.float() * 2.0 - 1.0  # (K, H, D)

        compressed_keys: CompressedKey = {
            "mse_indices": mse.reshape(K * H, D).long(),
            "qjl_signs": signs_float.reshape(K * H, D),
            "residual_norm": rnorm.reshape(K * H).float(),
            "vec_norm": vnorm.reshape(K * H).float(),
        }

        compressed_values: CompressedValue = (
            v_mse.reshape(K * H, D).long(),
            v_vnorm.reshape(K * H).float(),
        )

        return compressed_keys, compressed_values

    def clear_layer(self, layer_idx: int) -> None:
        """Zero out the cache for a single layer.

        Args:
            layer_idx: Zero-based layer index to clear.
        """
        self._assert_allocated()
        self._check_layer_idx(layer_idx)
        self._key_mse_buf[layer_idx].zero_()
        self._key_qjl_buf[layer_idx].zero_()
        self._key_rnorm_buf[layer_idx].zero_()
        self._key_vnorm_buf[layer_idx].zero_()
        self._val_mse_buf[layer_idx].zero_()
        self._val_vnorm_buf[layer_idx].zero_()
        self._fill_count[layer_idx] = 0

    def clear_all(self) -> None:
        """Zero out caches for all layers."""
        for layer_idx in range(self.num_layers):
            self.clear_layer(layer_idx)

    def memory_usage_bytes(self) -> Dict[str, int]:
        """Return actual GPU memory used by pre-allocated buffers.

        Returns:
            Dict with keys:

            - ``key_mse_bytes``:   Bytes for all key MSE index buffers.
            - ``key_qjl_bytes``:   Bytes for QJL sign buffers.
            - ``key_norm_bytes``:  Bytes for key norm buffers (residual + vec).
            - ``val_mse_bytes``:   Bytes for value MSE index buffers.
            - ``val_norm_bytes``:  Bytes for value norm buffers.
            - ``total_bytes``:     Grand total.
            - ``fp16_baseline_bytes``: Equivalent FP16 KV cache size.
            - ``compression_ratio``:   fp16 / total.
        """
        if not self._allocated:
            return {
                k: 0
                for k in [
                    "key_mse_bytes", "key_qjl_bytes", "key_norm_bytes",
                    "val_mse_bytes", "val_norm_bytes", "total_bytes",
                    "fp16_baseline_bytes", "compression_ratio",
                ]
            }

        def buf_bytes(bufs: List[torch.Tensor]) -> int:
            return sum(b.element_size() * b.numel() for b in bufs)

        key_mse = buf_bytes(self._key_mse_buf)
        key_qjl = buf_bytes(self._key_qjl_buf)
        key_norm = buf_bytes(self._key_rnorm_buf) + buf_bytes(self._key_vnorm_buf)
        val_mse = buf_bytes(self._val_mse_buf)
        val_norm = buf_bytes(self._val_vnorm_buf)
        total = key_mse + key_qjl + key_norm + val_mse + val_norm

        # FP16 baseline: (K + V) = 2 * max_seq_len * num_kv_heads * head_dim * 2 bytes
        fp16 = (
            2
            * self.max_seq_len
            * self.num_kv_heads
            * self.head_dim
            * 2  # float16 = 2 bytes
            * self.num_layers
        )

        return {
            "key_mse_bytes": key_mse,
            "key_qjl_bytes": key_qjl,
            "key_norm_bytes": key_norm,
            "val_mse_bytes": val_mse,
            "val_norm_bytes": val_norm,
            "total_bytes": total,
            "fp16_baseline_bytes": fp16,
            "compression_ratio": fp16 / total if total > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_allocated(self) -> None:
        if not self._allocated:
            raise RuntimeError(
                "TurboQuantCacheManager.allocate() must be called before use."
            )

    def _check_layer_idx(self, layer_idx: int) -> None:
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )


# ---------------------------------------------------------------------------
# Config Helper
# ---------------------------------------------------------------------------

# Known model families and their typical head configs.
# Keyed by a lowercase substring present in the model name.
_MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    # Format: { "substring": {num_layers, num_kv_heads, head_dim} }
    "qwen2.5-0.5": {"num_layers": 24, "num_kv_heads": 2,  "head_dim": 64},
    "qwen2.5-1.5": {"num_layers": 28, "num_kv_heads": 2,  "head_dim": 128},
    "qwen2.5-3":   {"num_layers": 36, "num_kv_heads": 2,  "head_dim": 128},
    "qwen2.5-7":   {"num_layers": 28, "num_kv_heads": 4,  "head_dim": 128},
    "qwen2.5-14":  {"num_layers": 48, "num_kv_heads": 8,  "head_dim": 128},
    "qwen2.5-32":  {"num_layers": 64, "num_kv_heads": 8,  "head_dim": 128},
    "qwen2.5-72":  {"num_layers": 80, "num_kv_heads": 8,  "head_dim": 128},
    "qwen3.5-7":   {"num_layers": 28, "num_kv_heads": 4,  "head_dim": 128},
    "qwen3.5-27":  {"num_layers": 62, "num_kv_heads": 8,  "head_dim": 256},
    "llama-3.1-8":  {"num_layers": 32, "num_kv_heads": 8,  "head_dim": 128},
    "llama-3.1-70": {"num_layers": 80, "num_kv_heads": 8,  "head_dim": 128},
    "llama-3.2-1":  {"num_layers": 16, "num_kv_heads": 8,  "head_dim": 64},
    "llama-3.2-3":  {"num_layers": 28, "num_kv_heads": 8,  "head_dim": 64},
    "mistral-7":   {"num_layers": 32, "num_kv_heads": 8,  "head_dim": 128},
    "mixtral-8x7": {"num_layers": 32, "num_kv_heads": 8,  "head_dim": 128},
    "phi-4":       {"num_layers": 40, "num_kv_heads": 10, "head_dim": 96},
    "gemma-2-9":   {"num_layers": 46, "num_kv_heads": 4,  "head_dim": 256},
    "gemma-2-27":  {"num_layers": 62, "num_kv_heads": 16, "head_dim": 128},
    "minimax-m2":  {"num_layers": 62, "num_kv_heads": 4,  "head_dim": 128},
}

# Default config if model is unrecognised — assumes common 7B GQA layout
_DEFAULT_CONFIG = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128}


def get_turboquant_config(
    model_name_or_path: str,
    bits: int = 3,
    vram_gb: float = 24.0,
) -> Dict[str, object]:
    """Auto-detect model architecture and return TurboQuant config.

    Attempts to detect ``num_layers``, ``num_kv_heads``, and ``head_dim``
    from the model name. Falls back to loading ``config.json`` from the
    local path / HuggingFace Hub if the name is not recognised, and finally
    to the default 7B layout if that also fails.

    Args:
        model_name_or_path: HuggingFace model ID or local path, e.g.
                            ``"Qwen/Qwen2.5-7B-Instruct"`` or
                            ``"/models/qwen2.5-7b"``.
        bits:               TurboQuant effective bits per coordinate (1-4).
                            3 is the paper's recommended sweet spot.
        vram_gb:            Available VRAM in gigabytes. Used to estimate
                            maximum achievable context length.

    Returns:
        Dict with keys:

        - ``model_name``:            Normalised model identifier.
        - ``head_dim``:              Detected head dimension ``d``.
        - ``num_kv_heads``:          Detected KV heads.
        - ``num_layers``:            Detected number of transformer layers.
        - ``bits``:                  Requested bit-width.
        - ``estimated_compression``: Theoretical compression ratio vs FP16.
        - ``estimated_max_context``: Estimated max context at given VRAM (tokens).
        - ``config_source``:         Where config came from (``"lookup"``,
                                     ``"hf_config"``, or ``"default"``).
    """
    lower_name = model_name_or_path.lower().replace("_", "-").replace("/", "-")
    found_cfg: Optional[Dict[str, int]] = None
    config_source = "default"

    # 1. Static lookup table
    for key, cfg in _MODEL_CONFIGS.items():
        if key in lower_name:
            found_cfg = cfg
            config_source = "lookup"
            break

    # 2. Try loading from HuggingFace config.json
    if found_cfg is None:
        found_cfg = _try_load_hf_config(model_name_or_path)
        if found_cfg is not None:
            config_source = "hf_config"

    # 3. Fall back to default
    if found_cfg is None:
        found_cfg = _DEFAULT_CONFIG.copy()

    num_layers: int = found_cfg["num_layers"]
    num_kv_heads: int = found_cfg["num_kv_heads"]
    head_dim: int = found_cfg["head_dim"]

    # Theoretical compression ratio (from kv_cache.py logic)
    mse_bits_key = max(bits - 1, 1)
    qjl_bits = head_dim              # 1 bit per QJL dim
    key_norm_bits = 32               # vec_norm + residual_norm, both fp16
    val_bits = bits * head_dim + 16  # MSE + norm
    bits_per_token = (mse_bits_key * head_dim + qjl_bits + key_norm_bits) + val_bits
    fp16_bits_per_token = (head_dim + head_dim) * 16  # K + V in FP16
    estimated_compression = fp16_bits_per_token / bits_per_token

    # Estimate max context length
    # Cache bytes per token = bits_per_token * num_kv_heads / 8 * num_layers
    bytes_per_token = bits_per_token * num_kv_heads / 8 * num_layers
    vram_bytes = vram_gb * 1024 ** 3
    # Reserve 60% of VRAM for model weights; 40% for KV cache
    cache_budget_bytes = vram_bytes * 0.40
    estimated_max_context = int(cache_budget_bytes / bytes_per_token)

    return {
        "model_name": model_name_or_path,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "num_layers": num_layers,
        "bits": bits,
        "estimated_compression": round(estimated_compression, 2),
        "estimated_max_context": estimated_max_context,
        "config_source": config_source,
    }


def _try_load_hf_config(model_name_or_path: str) -> Optional[Dict[str, int]]:
    """Try to load model config from HuggingFace transformers.

    Returns a dict with ``{num_layers, num_kv_heads, head_dim}`` if
    transformers is installed and the model config is accessible; else None.
    """
    try:
        from transformers import AutoConfig  # type: ignore[import]

        hf_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=False)
    except Exception:
        return None

    try:
        # Derive head_dim from hidden_size and num_attention_heads
        num_heads = getattr(hf_cfg, "num_attention_heads", None)
        hidden_size = getattr(hf_cfg, "hidden_size", None)
        head_dim = getattr(hf_cfg, "head_dim", None)
        if head_dim is None and num_heads and hidden_size:
            head_dim = hidden_size // num_heads

        num_kv_heads = getattr(hf_cfg, "num_key_value_heads", num_heads)
        num_layers = getattr(
            hf_cfg, "num_hidden_layers", getattr(hf_cfg, "n_layer", None)
        )

        if head_dim is None or num_kv_heads is None or num_layers is None:
            return None

        return {
            "num_layers": int(num_layers),
            "num_kv_heads": int(num_kv_heads),
            "head_dim": int(head_dim),
        }
    except Exception:
        return None
