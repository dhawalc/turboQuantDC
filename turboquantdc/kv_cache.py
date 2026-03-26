"""TurboQuant KV Cache — drop-in compressed key-value cache.

Wraps the TurboQuant estimator to provide a compressed KV cache for LLM
attention layers. Keys and values are handled differently:

    Keys:   TurboQuantEstimator (need unbiased INNER PRODUCTS for attention scores)
    Values: PolarQuant MSE-only (need accurate RECONSTRUCTION for weighted sum)

This asymmetry is a key design choice from the paper — values don't need
unbiased inner products, just low MSE reconstruction.

Storage per token (d=128, 3-bit):
    Key:   2*128 (MSE) + 128 (QJL) + 16 (r_norm) + 16 (vec_norm) = 416 bits
    Value: 3*128 (MSE) + 16 (vec_norm) = 400 bits
    Total: 816 bits vs 4096 bits (FP16) = 5.0x compression

Reference: TurboQuant paper (arxiv 2504.19874), full algorithm.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from .estimator import TurboQuantEstimator
from .polarquant import PolarQuant


class TurboQuantKVCache:
    """Compressed KV cache using TurboQuant for LLM attention.

    Keys are compressed with the full two-stage estimator (MSE + QJL) to
    provide unbiased attention score estimation. Values are compressed with
    MSE-only PolarQuant since they need reconstruction, not inner products.

    Args:
        d_key: Key head dimension.
        d_value: Value head dimension (often same as d_key).
        bits: Total effective bits per coordinate (e.g. 3).
        seed: Random seed for reproducibility.
        device: Target device.
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        bits: int = 3,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        self.d_key = d_key
        self.d_value = d_value
        self.bits = bits
        self.device = device

        # Keys: full TurboQuant (MSE + QJL) for unbiased inner products
        self.key_quantizer = TurboQuantEstimator(
            d=d_key, bits=bits, seed=seed, device=device
        )

        # Values: MSE-only PolarQuant with full bits (need reconstruction, not IP)
        self.value_quantizer = PolarQuant(
            d=d_value, bits=bits, seed=seed + 100, device=device
        )

        # Storage: list of compressed representations per sequence position
        self._key_store: List[Dict[str, torch.Tensor]] = []
        self._value_indices: List[torch.Tensor] = []
        self._value_norms: List[torch.Tensor] = []

    @property
    def seq_len(self) -> int:
        """Number of tokens in the cache."""
        return len(self._key_store)

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Compress and store new key-value pairs.

        Args:
            keys: Key vectors, shape (batch, d_key) or (d_key,).
                Each row is a key vector for one token.
            values: Value vectors, shape (batch, d_value) or (d_value,).
                Each row is a value vector for one token.
        """
        # Compress keys with full estimator (MSE + QJL)
        compressed_keys = self.key_quantizer.quantize(keys)
        self._key_store.append(compressed_keys)

        # Compress values with MSE-only (need reconstruction for weighted sum)
        squeeze = values.dim() == 1
        if squeeze:
            values = values.unsqueeze(0)

        # Store value norms for rescaling after reconstruction
        value_norms = values.norm(dim=-1, keepdim=True)  # (batch, 1)
        values_normalized = values / (value_norms + 1e-8)

        value_indices = self.value_quantizer.quantize(values_normalized)

        if squeeze:
            value_indices = value_indices.squeeze(0)
            value_norms = value_norms.squeeze(0)

        self._value_indices.append(value_indices)
        self._value_norms.append(value_norms.squeeze(-1) if not squeeze else value_norms.squeeze())

    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        """Compute attention scores between queries and all cached keys.

        Uses the unbiased TurboQuant inner product estimator:
            score_ij = <q_i, k_j> ~ <q_i, k_mse_j> + QJL_correction_j

        Args:
            queries: Query vectors, shape (n_queries, d_key) or (d_key,).

        Returns:
            Attention scores, shape (n_queries, seq_len) or (seq_len,).
        """
        if self.seq_len == 0:
            if queries.dim() == 1:
                return torch.zeros(0, device=queries.device)
            return torch.zeros(queries.shape[0], 0, device=queries.device)

        # Gather all compressed keys into batched tensors
        all_keys = self._gather_keys()

        # Use estimator's inner product method
        scores = self.key_quantizer.inner_product(queries, all_keys)

        return scores

    def get_values(self) -> torch.Tensor:
        """Reconstruct all cached values via MSE dequantization.

        Returns:
            Reconstructed values, shape (seq_len, d_value).
        """
        if self.seq_len == 0:
            return torch.zeros(0, self.d_value, device=self.device)

        all_indices = []
        all_norms = []

        for idx_tensor, norm_tensor in zip(self._value_indices, self._value_norms):
            if idx_tensor.dim() == 1:
                all_indices.append(idx_tensor.unsqueeze(0))
                all_norms.append(norm_tensor.unsqueeze(0))
            else:
                all_indices.append(idx_tensor)
                all_norms.append(norm_tensor)

        indices_cat = torch.cat(all_indices, dim=0)  # (seq_len, d_value)
        norms_cat = torch.cat(all_norms, dim=0)  # (seq_len,)

        # Dequantize: centroid lookup + unrotate
        values_normalized = self.value_quantizer.dequantize(indices_cat)

        # Rescale by original norms
        if norms_cat.dim() == 1:
            norms_cat = norms_cat.unsqueeze(-1)
        return values_normalized * norms_cat

    def memory_usage_bits(self) -> Dict[str, int]:
        """Compute memory usage statistics.

        Returns:
            Dict with:
                - key_mse_bits: Total bits for key MSE indices
                - key_qjl_bits: Total bits for key QJL signs
                - key_norm_bits: Total bits for key norms (vec_norm + residual_norm)
                - value_mse_bits: Total bits for value MSE indices
                - value_norm_bits: Total bits for value norms
                - total_bits: Grand total
                - fp16_baseline_bits: What FP16 would cost
                - compression_ratio: FP16 / total
        """
        n = self.seq_len
        if n == 0:
            return {
                "key_mse_bits": 0,
                "key_qjl_bits": 0,
                "key_norm_bits": 0,
                "value_mse_bits": 0,
                "value_norm_bits": 0,
                "total_bits": 0,
                "fp16_baseline_bits": 0,
                "compression_ratio": 0.0,
            }

        # Count actual tokens across all appended batches
        total_tokens = sum(
            ks["mse_indices"].shape[0] if ks["mse_indices"].dim() > 1 else 1
            for ks in self._key_store
        )

        mse_bits_key = self.key_quantizer.mse_bits
        qjl_m = self.key_quantizer.qjl.m

        key_mse = total_tokens * self.d_key * mse_bits_key
        key_qjl = total_tokens * qjl_m * 1  # 1 bit per QJL sign
        key_norms = total_tokens * 32  # vec_norm (16) + residual_norm (16)
        value_mse = total_tokens * self.d_value * self.bits
        value_norms = total_tokens * 16  # vec_norm in FP16

        total = key_mse + key_qjl + key_norms + value_mse + value_norms
        fp16 = total_tokens * (self.d_key + self.d_value) * 16

        return {
            "key_mse_bits": key_mse,
            "key_qjl_bits": key_qjl,
            "key_norm_bits": key_norms,
            "value_mse_bits": value_mse,
            "value_norm_bits": value_norms,
            "total_bits": total,
            "fp16_baseline_bits": fp16,
            "compression_ratio": fp16 / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        """Clear all cached key-value pairs."""
        self._key_store.clear()
        self._value_indices.clear()
        self._value_norms.clear()

    def _gather_keys(self) -> Dict[str, torch.Tensor]:
        """Concatenate all stored compressed keys into batched tensors."""
        all_mse_indices = []
        all_qjl_signs = []
        all_residual_norms = []
        all_vec_norms = []

        for ks in self._key_store:
            mi = ks["mse_indices"]
            qs = ks["qjl_signs"]
            rn = ks["residual_norm"]
            vn = ks["vec_norm"]

            if mi.dim() == 1:
                mi = mi.unsqueeze(0)
                qs = qs.unsqueeze(0)
                rn = rn.unsqueeze(0)
                vn = vn.unsqueeze(0)

            all_mse_indices.append(mi)
            all_qjl_signs.append(qs)
            all_residual_norms.append(rn)
            all_vec_norms.append(vn)

        return {
            "mse_indices": torch.cat(all_mse_indices, dim=0),
            "qjl_signs": torch.cat(all_qjl_signs, dim=0),
            "residual_norm": torch.cat(all_residual_norms, dim=0),
            "vec_norm": torch.cat(all_vec_norms, dim=0),
        }
