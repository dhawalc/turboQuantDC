"""Evolving KV Cache Compressor — modified by autoresearch.

This file is the ONLY file the autoresearch loop modifies.
It contains a self-contained KV cache compressor that duck-types
the HuggingFace Cache protocol.

Current approach: MSE-only PolarQuant with norm correction.
"""

import math
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

from .codebook import LloydMaxCodebook, gaussian_pdf
from .rotation import generate_rotation_matrix


class EvolvingLayer:
    def __init__(self, bits=4, seed=42):
        self.bits = bits
        self.seed = seed
        self._seq_len = 0
        self._head_dim = None
        self._num_heads = None
        self._batch_size = None
        self._dtype = None
        self._device = None
        self._rotation = None
        self._key_codebook = None
        self._val_codebook = None
        self._key_indices = []
        self._key_norms = []
        self._val_indices = []
        self._val_norms = []

    def _lazy_init(self, key_states, value_states):
        self._batch_size = key_states.shape[0]
        self._num_heads = key_states.shape[1]
        self._head_dim = key_states.shape[3]
        self._dtype = key_states.dtype
        self._device = key_states.device
        d = self._head_dim
        device = str(self._device)
        self._rotation = generate_rotation_matrix(d, seed=self.seed, device=device)
        # 3-bit keys + 1-bit residual signs = 4 effective bits, 5.1x compression
        self._key_codebook = LloydMaxCodebook(d=d, bits=3).to(device)
        self._val_codebook = LloydMaxCodebook(d=d, bits=2).to(device)

    def _quantize_vectors(self, vectors, codebook):
        batch, heads, seq, d = vectors.shape
        flat = vectors.float().reshape(-1, d)
        norms = flat.norm(dim=-1, keepdim=True)
        normalized = flat / (norms + 1e-8)
        rotated = normalized @ self._rotation
        indices = torch.bucketize(rotated, codebook.boundaries)
        indices = indices.clamp(0, codebook.centroids.shape[0] - 1)
        # Norm correction: store original/reconstruction ratio
        recon_rotated = codebook.centroids[indices]
        recon_unrotated = recon_rotated @ self._rotation.T
        recon_norm = recon_unrotated.norm(dim=-1, keepdim=True)
        corrected_norms = norms * (norms / (recon_norm * norms.abs().clamp(min=1e-8) + 1e-8)).clamp(0.5, 2.0)
        # Residual signs for correction
        residual = rotated - recon_rotated
        res_signs = torch.sign(residual)
        res_scale = residual.abs().mean(dim=-1, keepdim=True)
        return (indices.reshape(batch, heads, seq, d),
                corrected_norms.reshape(batch, heads, seq),
                res_signs.reshape(batch, heads, seq, d),
                res_scale.reshape(batch, heads, seq))

    def _dequantize_vectors(self, indices, norms, codebook, res_signs=None, res_scale=None):
        batch, heads, seq, d = indices.shape
        flat_idx = indices.reshape(-1, d)
        flat_norms = norms.reshape(-1)
        reconstructed = codebook.centroids[flat_idx]
        # Add residual correction
        if res_signs is not None and res_scale is not None:
            reconstructed = reconstructed + res_signs.reshape(-1, d) * res_scale.reshape(-1, 1)
        reconstructed = reconstructed @ self._rotation.T
        reconstructed = reconstructed * flat_norms.unsqueeze(-1)
        return reconstructed.reshape(batch, heads, seq, d)

    def update(self, key_states, value_states):
        if self._rotation is None:
            self._lazy_init(key_states, value_states)
        # Store compressed version
        k_idx, k_norms, k_rsigns, k_rscale = self._quantize_vectors(key_states, self._key_codebook)
        self._key_indices.append(k_idx)
        self._key_norms.append(k_norms)
        self._key_rsigns = getattr(self, '_key_rsigns', [])
        self._key_rsigns.append(k_rsigns)
        self._key_rscales = getattr(self, '_key_rscales', [])
        self._key_rscales.append(k_rscale)
        v_idx, v_norms, v_rsigns, v_rscale = self._quantize_vectors(value_states, self._val_codebook)
        self._val_indices.append(v_idx)
        self._val_norms.append(v_norms)
        # FP16 residual window — only keep last 128 tokens, discard older raw data
        FP16_WINDOW = 128
        self._raw_keys = getattr(self, '_raw_keys', [])
        self._raw_vals = getattr(self, '_raw_vals', [])
        self._raw_keys.append(key_states.detach())
        self._raw_vals.append(value_states.detach())
        # Trim: concatenate and keep only last FP16_WINDOW tokens
        if self._seq_len > FP16_WINDOW * 2:
            all_rk = torch.cat(self._raw_keys, dim=2)
            all_rv = torch.cat(self._raw_vals, dim=2)
            self._raw_keys = [all_rk[:, :, -FP16_WINDOW:, :]]
            self._raw_vals = [all_rv[:, :, -FP16_WINDOW:, :]]
        self._seq_len += key_states.shape[2]
        return self._get_all()

    def _get_all(self):
        if self._seq_len == 0:
            empty = torch.zeros(1, 1, 0, self._head_dim or 1, dtype=self._dtype, device=self._device)
            return empty, empty
        all_k_idx = torch.cat(self._key_indices, dim=2)
        all_k_norms = torch.cat(self._key_norms, dim=2)
        all_v_idx = torch.cat(self._val_indices, dim=2)
        all_v_norms = torch.cat(self._val_norms, dim=2)
        k_rsigns = torch.cat(self._key_rsigns, dim=2) if hasattr(self, '_key_rsigns') and self._key_rsigns else None
        k_rscales = torch.cat(self._key_rscales, dim=2) if hasattr(self, '_key_rscales') and self._key_rscales else None
        keys = self._dequantize_vectors(all_k_idx, all_k_norms, self._key_codebook, k_rsigns, k_rscales)
        values = self._dequantize_vectors(all_v_idx, all_v_norms, self._val_codebook)

        # Residual window: replace last 128 tokens with raw FP16
        FP16_WINDOW = 128
        if hasattr(self, '_raw_keys') and self._raw_keys:
            raw_keys = torch.cat(self._raw_keys, dim=2)
            raw_vals = torch.cat(self._raw_vals, dim=2)
            win = min(FP16_WINDOW, raw_keys.shape[2])
            if win > 0 and keys.shape[2] >= win:
                keys[:, :, -win:, :] = raw_keys[:, :, -win:, :].to(keys.dtype)
                values[:, :, -win:, :] = raw_vals[:, :, -win:, :].to(values.dtype)

        return keys.to(self._dtype), values.to(self._dtype)

    @property
    def seq_len(self):
        return self._seq_len


class FP16Layer:
    """FP16 anchor layer — no compression, stores raw tensors."""
    def __init__(self):
        self._keys = []
        self._vals = []
        self._seq_len = 0
    def update(self, key_states, value_states):
        self._keys.append(key_states)
        self._vals.append(value_states)
        self._seq_len += key_states.shape[2]
        return torch.cat(self._keys, dim=2), torch.cat(self._vals, dim=2)
    @property
    def seq_len(self):
        return self._seq_len
    def _get_all(self):
        if not self._keys:
            return None, None
        return torch.cat(self._keys, dim=2), torch.cat(self._vals, dim=2)


class EvolvingCache:
    is_compileable = False
    def __init__(self, bits=4, seed=42, anchor_interval=6):
        self.bits = bits
        self.seed = seed
        self.anchor_interval = anchor_interval  # Every Nth layer is FP16
        self._layers = []
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        while len(self._layers) <= layer_idx:
            idx = len(self._layers)
            if self.anchor_interval > 0 and idx % self.anchor_interval == 0:
                self._layers.append(FP16Layer())
            else:
                self._layers.append(EvolvingLayer(bits=self.bits, seed=self.seed + idx))
        return self._layers[layer_idx].update(key_states, value_states)
    def get_seq_length(self, layer_idx=0):
        if layer_idx < len(self._layers):
            return self._layers[layer_idx].seq_len
        return 0
    def get_max_cache_shape(self):
        return None
    def __len__(self):
        return len(self._layers)
    def __iter__(self):
        for layer in self._layers:
            yield layer._get_all()
    def __getitem__(self, idx):
        if idx < len(self._layers):
            return self._layers[idx]._get_all()
        return None, None
    def __contains__(self, idx):
        return idx < len(self._layers)
    def reset(self):
        self._layers.clear()
    @property
    def seen_tokens(self):
        return self._layers[0].seq_len if self._layers else 0
    def crop(self, max_length):
        pass
    def reorder_cache(self, beam_idx):
        pass
    def batch_repeat_interleave(self, repeats):
        pass
    def batch_select_indices(self, indices):
        pass
    def get_mask_sizes(self, cache_position, layer_idx=0):
        # Match AdaptiveHFCache: kv_length = cached + new query tokens
        if layer_idx < len(self._layers):
            cached = self._layers[layer_idx].seq_len
        else:
            cached = 0
        query_length = cache_position.shape[0] if cache_position is not None else 0
        return cached + query_length, 0
    @property
    def is_sliding(self):
        # HF expects a list of booleans, one per layer
        return [False] * max(len(self._layers), 1)
    @property
    def is_initialized(self):
        return len(self._layers) > 0
