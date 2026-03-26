"""Phase 3: Long context validation -- TurboQuant on Qwen3.5-27B.

Demonstrates that TurboQuant enables long-context inference on a single RTX 4090
where FP16 KV cache would OOM. Qwen3.5-27B is a hybrid DeltaNet+Attention model:
only 16 out of 64 layers use standard attention (every 4th layer).

Tests needle-in-haystack at 8K/32K/65K context, compresses the standard attention
KV cache with TurboQuant, and compares attention score fidelity.

Usage:
    python benchmarks/long_context.py [--context 32768] [--bits 3]
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Allow running from repo root: python benchmarks/long_context.py
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from turboquantdc import TurboQuantEstimator  # noqa: E402

# ---------------------------------------------------------------------------
# Model Configuration — auto-detected from HF config at runtime
# Override with TURBOQUANT_MODEL env var for different models
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("TURBOQUANT_MODEL", "Qwen/Qwen2.5-14B-Instruct")

# These defaults are overwritten by load_model(), but kept for VRAM budget
# display which runs before model loading
NUM_HIDDEN_LAYERS = 48
FULL_ATTENTION_INTERVAL = 1  # 1 = all layers are standard attention
NUM_ATTENTION_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
MAX_POSITION_EMBEDDINGS = 32768

# Standard attention layer indices — for non-hybrid models, all layers
STANDARD_ATTN_LAYERS = list(range(NUM_HIDDEN_LAYERS))
NUM_STANDARD_LAYERS = NUM_HIDDEN_LAYERS

DEFAULT_CONTEXT_LENGTHS = [4096, 8192, 16384]
BIT_WIDTHS = [2, 3, 4]

NEEDLE = "The secret project code name is AURORA-7749."
NEEDLE_MARKER = "AURORA-7749"

# Filler paragraph (~90 tokens) repeated to build long context
FILLER = (
    "The quarterly financial review meeting covered several topics including "
    "budget allocations for the upcoming fiscal year, departmental spending reports, "
    "and projected revenue streams from various business units. The committee discussed "
    "infrastructure upgrades planned for the western regional offices and noted that "
    "maintenance schedules should be coordinated with the facilities management team. "
    "Several action items were assigned to team leads for follow-up before the next "
    "meeting cycle.\n\n"
)


# ---------------------------------------------------------------------------
# Result containers (adapted from real_model.py)
# ---------------------------------------------------------------------------
@dataclass
class HeadResult:
    """Per-head comparison result."""
    layer: int
    head: int
    cosine_sim: float
    top1_match: bool
    top5_match: bool
    needle_rank: Optional[int]


@dataclass
class BitWidthResult:
    """Aggregate results for one bit-width at one context length."""
    bits: int
    seq_len: int
    needle_token: Optional[int]
    head_results: List[HeadResult] = field(default_factory=list)
    compressed_bits: int = 0
    fp16_bits: int = 0

    @property
    def n_heads(self) -> int:
        return len(self.head_results)

    @property
    def avg_cosine_sim(self) -> float:
        if not self.head_results:
            return 0.0
        return sum(r.cosine_sim for r in self.head_results) / self.n_heads

    @property
    def top1_pct(self) -> float:
        if not self.head_results:
            return 0.0
        return 100.0 * sum(1 for r in self.head_results if r.top1_match) / self.n_heads

    @property
    def top5_pct(self) -> float:
        if not self.head_results:
            return 0.0
        return 100.0 * sum(1 for r in self.head_results if r.top5_match) / self.n_heads

    @property
    def avg_needle_rank(self) -> float:
        ranks = [r.needle_rank for r in self.head_results if r.needle_rank is not None]
        if not ranks:
            return -1.0
        return sum(ranks) / len(ranks)

    @property
    def compression_ratio(self) -> float:
        if self.compressed_bits == 0:
            return 0.0
        return self.fp16_bits / self.compressed_bits

    @property
    def compressed_mb(self) -> float:
        return self.compressed_bits / 8 / 1024 / 1024

    @property
    def fp16_mb(self) -> float:
        return self.fp16_bits / 8 / 1024 / 1024


# ---------------------------------------------------------------------------
# VRAM budget analysis
# ---------------------------------------------------------------------------
def print_vram_budget() -> None:
    """Print theoretical VRAM breakdown before running inference."""
    print()
    print("=" * 70)
    print("VRAM Budget (RTX 4090, 24 GB)")
    print("=" * 70)

    model_weight_gb = 14.0  # approximate for 27B params in 4-bit NF4

    fp16_bytes_per_token = (
        NUM_STANDARD_LAYERS * NUM_KV_HEADS * HEAD_DIM * 2  # K + V
        * 2  # 2 bytes per FP16 element
    )
    # TQ-3bit: keys = (b-1)*d + d (QJL) + norms; values = b*d + norms
    # Simplified: ~b*d bits per coord for K, b*d for V, plus overhead
    # More precise: keys = (2*256 + 256 + 32) bits = 544+32 bits; values = (3*256 + 16) = 784 bits
    # per token per layer per head: key 576 bits + value 784 bits = 1360 bits
    # But for the budget display, use the simpler approximation
    tq3_bits_per_element = 3 + 1  # 3-bit MSE + 1 QJL (keys get extra QJL bit)
    # More accurately: effective ~3 bits per coordinate for keys, 3 bits for values
    # = 0.375 bytes per coordinate
    tq3_bytes_per_token = (
        NUM_STANDARD_LAYERS * NUM_KV_HEADS * HEAD_DIM * 2  # K + V
        * 0.375  # ~3 bits = 0.375 bytes per element
    )

    available_gb = 24.0 - model_weight_gb
    max_ctx_fp16 = int(available_gb * 1024**3 / fp16_bytes_per_token)
    max_ctx_tq3 = int(available_gb * 1024**3 / tq3_bytes_per_token)

    fp16_at_max = fp16_bytes_per_token * MAX_POSITION_EMBEDDINGS / 1024**3
    tq3_at_max = tq3_bytes_per_token * MAX_POSITION_EMBEDDINGS / 1024**3

    print(f"  Model weights (4-bit):   ~{model_weight_gb:.0f} GB")
    print(f"  Available for KV cache:  ~{available_gb:.0f} GB")
    print()
    print(f"  FP16 KV cache per token: {NUM_STANDARD_LAYERS} layers x {NUM_KV_HEADS} heads "
          f"x {HEAD_DIM} dim x 2 (K+V) x 2 bytes = {fp16_bytes_per_token:,} bytes")
    print(f"  TQ-3bit per token:       {NUM_STANDARD_LAYERS} layers x {NUM_KV_HEADS} heads "
          f"x {HEAD_DIM} dim x 2 (K+V) x 0.375 bytes = {tq3_bytes_per_token:,.0f} bytes")
    print()
    print(f"  Max context (FP16):      ~{available_gb:.0f} GB / {fp16_bytes_per_token // 1024} KB "
          f"= ~{max_ctx_fp16:,} tokens")
    print(f"  Max context (TQ-3bit):   ~{available_gb:.0f} GB / {tq3_bytes_per_token / 1024:.1f} KB "
          f"= ~{max_ctx_tq3:,} tokens")
    print()
    print(f"  NOTE: FP16 cache at {MAX_POSITION_EMBEDDINGS // 1024}K (model max) = {fp16_at_max:.1f} GB "
          f"{'-> OOM!' if fp16_at_max > available_gb else '-> fits'}")
    print(f"  TQ-3bit cache at {MAX_POSITION_EMBEDDINGS // 1024}K = {tq3_at_max:.1f} GB "
          f"{'-> OOM!' if tq3_at_max > available_gb else '-> FITS!'}")
    print()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_prompt(tokenizer, target_tokens: int = 32768, needle_pos: float = 0.25) -> str:
    """Build a needle-in-haystack prompt for a base model (no chat template).

    Places the needle at ``needle_pos`` fraction through the filler text.
    Uses raw text since Qwen3.5-27B is a base model.
    """
    filler_len = len(tokenizer.encode(FILLER, add_special_tokens=False))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)

    parts: list[str] = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Memo ---\n{NEEDLE}\n--- End ---\n\n")
        parts.append(FILLER)

    haystack = "".join(parts)
    # Base model format -- no chat template
    prompt = (
        f"Document:\n{haystack}\n"
        f"Question: What is the secret project code name mentioned in the document?\n"
        f"Answer:"
    )
    return prompt


def find_needle_token(tokenizer, input_ids: torch.Tensor) -> Optional[int]:
    """Find the token position where AURORA-7749 starts."""
    needle_tokens = tokenizer.encode(NEEDLE_MARKER, add_special_tokens=False)
    ids_list = input_ids[0].tolist()

    # Exact subsequence match
    for i in range(len(ids_list) - len(needle_tokens) + 1):
        if ids_list[i : i + len(needle_tokens)] == needle_tokens:
            return i

    # Fallback: try progressively shorter prefixes
    for width in range(len(needle_tokens), 0, -1):
        sub = needle_tokens[:width]
        for i in range(len(ids_list) - width + 1):
            if ids_list[i : i + width] == sub:
                return i

    return None


# ---------------------------------------------------------------------------
# Memory accounting
# ---------------------------------------------------------------------------
def compute_memory_bits(
    n_layers: int,
    n_kv_heads: int,
    seq_len: int,
    head_dim: int,
    bits: int,
) -> tuple[int, int]:
    """Compute compressed and FP16 bit counts for the KV cache.

    Keys: (bits-1)*d MSE indices + d QJL signs + 16 residual_norm + 16 vec_norm
    Values: bits*d MSE indices + 16 vec_norm

    Returns:
        (compressed_bits, fp16_bits)
    """
    n_vectors = n_layers * n_kv_heads * seq_len
    mse_bits_key = max(bits - 1, 1)

    # Keys
    key_mse = n_vectors * head_dim * mse_bits_key
    key_qjl = n_vectors * head_dim * 1  # 1 bit per QJL sign (m = d)
    key_norms = n_vectors * 32  # vec_norm (16) + residual_norm (16)

    # Values (MSE-only, full bits)
    val_mse = n_vectors * head_dim * bits
    val_norms = n_vectors * 16  # vec_norm in FP16

    compressed = key_mse + key_qjl + key_norms + val_mse + val_norms
    fp16 = n_vectors * head_dim * 16 * 2  # keys + values, 16 bits per coord

    return compressed, fp16


# ---------------------------------------------------------------------------
# Per-head attention comparison
# ---------------------------------------------------------------------------
def compare_head(
    keys_fp: torch.Tensor,
    head_dim: int,
    bits: int,
    layer_idx: int,
    head_idx: int,
    needle_token: Optional[int],
) -> HeadResult:
    """Compare real vs TurboQuant attention scores for one head.

    Args:
        keys_fp: Key matrix for this head, shape (seq_len, head_dim), float32.
        head_dim: Dimension d.
        bits: TurboQuant bit-width.
        layer_idx: Layer index (used as seed component).
        head_idx: Head index (used as seed component).
        needle_token: Token position of the needle, or None.

    Returns:
        HeadResult with all metrics.
    """
    seq_len = keys_fp.shape[0]
    device = keys_fp.device

    # Query = last token (simulates next-token generation attending to all keys)
    query = keys_fp[-1:, :]  # (1, head_dim)

    # --- Real attention scores (full precision) ---
    real_scores = (query @ keys_fp.T).squeeze(0)  # (seq_len,)

    # --- TurboQuant scores ---
    # Each (layer, head) pair gets a unique seed for independent random matrices
    seed = layer_idx * 1000 + head_idx
    estimator = TurboQuantEstimator(
        d=head_dim, bits=bits, seed=seed, device=device
    )
    compressed = estimator.quantize(keys_fp)
    tq_scores = estimator.inner_product(query, compressed).squeeze(0)  # (seq_len,)

    # --- Metrics ---
    cos_sim = F.cosine_similarity(
        real_scores.unsqueeze(0).float(),
        tq_scores.unsqueeze(0).float(),
    ).item()

    real_top1 = real_scores.argmax().item()
    tq_top1 = tq_scores.argmax().item()
    top1_match = real_top1 == tq_top1

    tq_top5 = tq_scores.topk(min(5, seq_len)).indices.tolist()
    top5_match = real_top1 in tq_top5

    needle_rank = None
    if needle_token is not None and needle_token < seq_len:
        sorted_indices = tq_scores.argsort(descending=True)
        rank_mask = (sorted_indices == needle_token).nonzero(as_tuple=False)
        if len(rank_mask) > 0:
            needle_rank = rank_mask[0].item()

    return HeadResult(
        layer=layer_idx,
        head=head_idx,
        cosine_sim=cos_sim,
        top1_match=top1_match,
        top5_match=top5_match,
        needle_rank=needle_rank,
    )


# ---------------------------------------------------------------------------
# Cache extraction helpers for hybrid models
# ---------------------------------------------------------------------------
def identify_standard_attention_layers(cache, model_config) -> List[int]:
    """Identify which cache entries correspond to standard attention layers.

    Qwen3.5 is a hybrid model. The cache may contain entries for:
    - All 64 layers (with DeltaNet layers having None or different format)
    - Only the 16 standard attention layers

    Returns:
        List of cache indices that correspond to standard attention layers.
    """
    # Determine cache length
    if hasattr(cache, "key_cache"):
        cache_len = len(cache.key_cache)
    elif hasattr(cache, "__len__"):
        cache_len = len(cache)
    else:
        # Try iterating to get length
        try:
            cache._layer_list = list(cache)
            cache_len = len(cache._layer_list)
        except TypeError:
            cache_len = 0

    if cache_len == 0:
        return []

    # For hybrid models: probe each cache entry to find valid (non-None) keys
    if hasattr(cache, "key_cache"):
        valid = []
        for i in range(cache_len):
            entry = cache.key_cache[i]
            if entry is not None and isinstance(entry, torch.Tensor) and entry.dim() == 4:
                valid.append(i)
        if valid:
            return valid

    # Case 1: Non-hybrid model — all layers are standard attention
    if FULL_ATTENTION_INTERVAL == 1 and cache_len == NUM_HIDDEN_LAYERS:
        return list(range(cache_len))

    # Case 2: Hybrid model — cache has entries for all layers
    if cache_len == NUM_HIDDEN_LAYERS and FULL_ATTENTION_INTERVAL > 1:
        return STANDARD_ATTN_LAYERS

    # Case 3: Cache only has entries for the standard attention layers
    if cache_len == NUM_STANDARD_LAYERS:
        return list(range(NUM_STANDARD_LAYERS))

    # Case 3: Some other number -- try to detect valid entries
    # Walk through and find entries with the expected KV shape
    valid_indices = []
    for i in range(cache_len):
        try:
            keys = _get_cache_keys(cache, i)
            if keys is not None and keys.dim() == 4:
                # Expect shape (batch, num_kv_heads, seq_len, head_dim)
                if keys.shape[1] == NUM_KV_HEADS and keys.shape[3] == HEAD_DIM:
                    valid_indices.append(i)
        except (IndexError, AttributeError, TypeError):
            continue
    return valid_indices


def _get_cache_keys(cache, layer_idx: int) -> Optional[torch.Tensor]:
    """Extract key tensor from cache at the given layer index.

    Handles DynamicCache, HybridCache, tuple-of-tuples, and other formats.
    Returns None if the entry is not a standard attention cache.
    """
    try:
        # Method 1: DynamicCache with key_cache attribute
        if hasattr(cache, "key_cache"):
            entry = cache.key_cache[layer_idx]
            if entry is not None and isinstance(entry, torch.Tensor) and entry.dim() == 4:
                return entry
            return None

        # Method 2: Iterable cache yielding (keys, values) tuples
        # Convert to list on first access and cache it
        if not hasattr(cache, "_layer_list"):
            try:
                cache._layer_list = list(cache)
            except TypeError:
                cache._layer_list = None

        if cache._layer_list is not None and layer_idx < len(cache._layer_list):
            entry = cache._layer_list[layer_idx]
            if isinstance(entry, (tuple, list)):
                k = entry[0]
                if isinstance(k, torch.Tensor) and k.dim() == 4:
                    return k
            elif isinstance(entry, torch.Tensor) and entry.dim() == 4:
                return entry
            return None

        # Method 3: Direct subscript
        if hasattr(cache, "__getitem__"):
            entry = cache[layer_idx]
            if isinstance(entry, (tuple, list)):
                k = entry[0]
                if isinstance(k, torch.Tensor) and k.dim() == 4:
                    return k
            elif isinstance(entry, torch.Tensor) and entry.dim() == 4:
                return entry
            return None
    except (IndexError, KeyError, TypeError):
        return None
    return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    """Load Qwen3.5-27B in 4-bit NF4 with BitsAndBytes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME} (4-bit NF4)...", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    # Force all modules onto GPU — accelerate needs to know the model
    # will be quantized to 4-bit (much smaller than BF16 on disk)
    max_memory = {0: "22GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()

    load_time = time.time() - t0
    gpu_mb = torch.cuda.memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0
    gpu_gb = gpu_mb / 1024

    # Extract model config
    config = model.config
    n_layers = getattr(config, "num_hidden_layers", NUM_HIDDEN_LAYERS)
    n_heads = getattr(config, "num_attention_heads", NUM_ATTENTION_HEADS)
    n_kv_heads = getattr(config, "num_key_value_heads", NUM_KV_HEADS)
    head_dim_from_config = getattr(config, "head_dim", None)
    if head_dim_from_config is None:
        hidden_size = getattr(config, "hidden_size", n_heads * HEAD_DIM)
        head_dim_from_config = hidden_size // n_heads
    full_attn_interval = getattr(config, "full_attention_interval", FULL_ATTENTION_INTERVAL)

    print(f"  Loaded in {load_time:.1f}s | GPU: {gpu_gb:.1f} GB ({gpu_mb} MB)")
    print(f"  Layers: {n_layers} | Heads: {n_heads} | KV heads: {n_kv_heads} | "
          f"head_dim: {head_dim_from_config}")
    print(f"  Full attention interval: every {full_attn_interval}th layer "
          f"({n_layers // full_attn_interval} standard attention layers)")
    print()

    return model, tokenizer, n_layers, n_kv_heads, head_dim_from_config, gpu_gb


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------
def run_validation(
    model,
    tokenizer,
    n_kv_heads: int,
    head_dim: int,
    target_tokens: int,
    bit_widths: List[int],
    model_gpu_gb: float,
) -> Optional[List[BitWidthResult]]:
    """Run validation for one context length across specified bit-widths.

    Returns None if the forward pass OOMs.
    """
    # Build prompt and tokenize
    prompt = build_prompt(tokenizer, target_tokens=target_tokens)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=target_tokens + 256,
    ).to("cuda")
    seq_len = inputs["input_ids"].shape[1]

    # Find needle position
    needle_token = find_needle_token(tokenizer, inputs["input_ids"])

    print(f"\n{'=' * 70}")
    print(f"--- Context: {seq_len} tokens ---")
    print(f"{'=' * 70}")
    print(f"  Needle at token: {needle_token}")

    # Forward pass -- capture KV cache
    print("  Running forward pass...", end="", flush=True)
    t0 = time.time()
    try:
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_attentions=False)
    except torch.cuda.OutOfMemoryError:
        print(" OOM!")
        print(f"  SKIPPED: Out of GPU memory at {seq_len} tokens.")
        print(f"  This is expected for very long contexts. Try a shorter --context.")
        torch.cuda.empty_cache()
        gc.collect()
        return None
    fwd_time = time.time() - t0
    print(f" {fwd_time:.1f}s")

    cache = outputs.past_key_values

    # Identify which cache entries are standard attention (not DeltaNet)
    standard_indices = identify_standard_attention_layers(cache, model.config)
    n_standard = len(standard_indices)

    if n_standard == 0:
        print("  ERROR: Could not identify any standard attention layers in cache.")
        print("  Cache type:", type(cache).__name__)
        if hasattr(cache, "key_cache"):
            print(f"  key_cache length: {len(cache.key_cache)}")
            # Debug: print shapes of first few entries
            for i in range(min(5, len(cache.key_cache))):
                entry = cache.key_cache[i]
                if entry is None:
                    print(f"    [{i}] None")
                elif isinstance(entry, torch.Tensor):
                    print(f"    [{i}] {entry.shape}")
                else:
                    print(f"    [{i}] {type(entry).__name__}")
        return None

    # Validate dimensions from first standard layer
    sample_keys = _get_cache_keys(cache, standard_indices[0])
    if sample_keys is None:
        # Debug: try direct indexing
        print(f"  DEBUG: _get_cache_keys returned None for index {standard_indices[0]}")
        print(f"  Cache type: {type(cache).__name__}")
        if hasattr(cache, "key_cache"):
            print(f"  key_cache len: {len(cache.key_cache)}")
            entry = cache.key_cache[standard_indices[0]]
            print(f"  Entry type: {type(entry).__name__}, shape: {entry.shape if hasattr(entry, 'shape') else 'N/A'}")
        # Try legacy tuple access
        try:
            entry = cache[standard_indices[0]]
            if isinstance(entry, (tuple, list)):
                sample_keys = entry[0]
                print(f"  Legacy tuple access: shape={sample_keys.shape}")
        except Exception as e:
            print(f"  Legacy access failed: {e}")
        if sample_keys is None:
            return None
    actual_kv_heads = sample_keys.shape[1]
    actual_head_dim = sample_keys.shape[3]
    actual_seq = sample_keys.shape[2]

    print(f"  Cache: {n_standard} standard attention layers "
          f"(out of {len(cache.key_cache) if hasattr(cache, 'key_cache') else '?'} total)")
    print(f"  Shape: {actual_kv_heads} KV heads x {actual_seq} seq x {actual_head_dim} head_dim")

    # Override with actual values
    n_kv_heads = actual_kv_heads
    head_dim = actual_head_dim

    # Memory tracking
    gpu_mem_after_fwd = torch.cuda.max_memory_allocated() / (1024**3)
    fp16_kv_gb = (n_standard * n_kv_heads * actual_seq * head_dim * 2 * 2) / (1024**3)
    compressed_3bit, _ = compute_memory_bits(n_standard, n_kv_heads, actual_seq, head_dim, 3)
    tq3_kv_gb = compressed_3bit / 8 / (1024**3)

    print(f"  GPU Memory Used:   {gpu_mem_after_fwd:.1f} GB (peak)")
    print(f"  FP16 KV would be:  {fp16_kv_gb:.2f} GB")
    print(f"  TQ-3bit KV:        {tq3_kv_gb:.2f} GB ({fp16_kv_gb / tq3_kv_gb:.1f}x compression)")

    # Check if FP16 would OOM at model max context
    fp16_at_max = (n_standard * n_kv_heads * MAX_POSITION_EMBEDDINGS * head_dim * 2 * 2) / (1024**3)
    tq3_at_max_bits, _ = compute_memory_bits(
        n_standard, n_kv_heads, MAX_POSITION_EMBEDDINGS, head_dim, 3
    )
    tq3_at_max = tq3_at_max_bits / 8 / (1024**3)
    available_gb = 24.0 - model_gpu_gb
    print(f"  At model max ({MAX_POSITION_EMBEDDINGS // 1024}K): "
          f"FP16 = {fp16_at_max:.1f} GB {'(OOM!)' if fp16_at_max > available_gb else '(fits)'}, "
          f"TQ-3bit = {tq3_at_max:.1f} GB {'(OOM!)' if tq3_at_max > available_gb else '(fits)'}")

    # Evaluate each bit-width
    results: List[BitWidthResult] = []

    for bits in bit_widths:
        print(f"\n  Evaluating TQ-{bits}bit...", end="", flush=True)
        t0 = time.time()

        compressed_bits, fp16_bits = compute_memory_bits(
            n_standard, n_kv_heads, actual_seq, head_dim, bits
        )

        bw_result = BitWidthResult(
            bits=bits,
            seq_len=actual_seq,
            needle_token=needle_token,
            compressed_bits=compressed_bits,
            fp16_bits=fp16_bits,
        )

        for cache_idx, layer_global_idx in _enumerate_standard_layers(standard_indices):
            keys = _get_cache_keys(cache, cache_idx)
            if keys is None:
                continue

            for h in range(n_kv_heads):
                # Extract this head's keys as float32 for accurate comparison
                k = keys[0, h].float()  # (seq, head_dim)

                head_result = compare_head(
                    keys_fp=k,
                    head_dim=head_dim,
                    bits=bits,
                    layer_idx=layer_global_idx,
                    head_idx=h,
                    needle_token=needle_token,
                )
                bw_result.head_results.append(head_result)

        eval_time = time.time() - t0
        print(f" {eval_time:.1f}s ({bw_result.n_heads} heads)")
        results.append(bw_result)

    return results


def _enumerate_standard_layers(standard_indices: List[int]):
    """Yield (cache_index, global_layer_index) pairs.

    If the cache has 64 entries, cache_index == global_layer_index.
    If the cache has only 16 entries, cache_index is 0..15 and
    global_layer_index maps to 3, 7, 11, ..., 63.
    """
    for i, cache_idx in enumerate(standard_indices):
        if len(standard_indices) == NUM_STANDARD_LAYERS and standard_indices == list(range(NUM_STANDARD_LAYERS)):
            # Cache only contains standard layers (16 entries)
            global_idx = STANDARD_ATTN_LAYERS[i]
        else:
            # Cache contains all layers (64 entries), cache_idx IS the global index
            global_idx = cache_idx
        yield cache_idx, global_idx


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_results(results: List[BitWidthResult]) -> None:
    """Print formatted results for one context length."""
    for r in results:
        top1_count = sum(1 for h in r.head_results if h.top1_match)
        top5_count = sum(1 for h in r.head_results if h.top5_match)

        print(f"\n  TQ-{r.bits}bit:")
        print(f"    Compression:       {r.compression_ratio:.1f}x  "
              f"({r.compressed_mb:.1f} MB vs {r.fp16_mb:.1f} MB)")
        print(f"    Score cosine sim:  {r.avg_cosine_sim:.4f}  (1.0 = perfect)")
        print(f"    Top-1 match:       {r.top1_pct:.1f}%  ({top1_count}/{r.n_heads} heads)")
        print(f"    Top-5 match:       {r.top5_pct:.1f}%  ({top5_count}/{r.n_heads} heads)")
        if r.avg_needle_rank >= 0:
            print(f"    Avg needle rank:   {r.avg_needle_rank:.1f}  (lower = better, 0 = top)")


def print_final_report(
    all_results: Dict[int, List[BitWidthResult]],
    model_gpu_gb: float,
) -> None:
    """Print final summary report."""
    available_gb = 24.0 - model_gpu_gb

    # FP16 and TQ-3bit at model max context
    fp16_max = (NUM_STANDARD_LAYERS * NUM_KV_HEADS * MAX_POSITION_EMBEDDINGS
                * HEAD_DIM * 2 * 2) / (1024**3)
    tq3_max_bits, _ = compute_memory_bits(
        NUM_STANDARD_LAYERS, NUM_KV_HEADS, MAX_POSITION_EMBEDDINGS, HEAD_DIM, 3
    )
    tq3_max = tq3_max_bits / 8 / (1024**3)
    spare_gb = available_gb - tq3_max

    print()
    print("=" * 70)
    print("Phase 3: Long Context Validation -- Qwen3.5-27B on RTX 4090")
    print("=" * 70)
    print()
    print(f"Model: {MODEL_NAME} (4-bit NF4)")
    print(f"GPU Memory: {model_gpu_gb:.1f} GB (model)")
    print(f"Architecture: {NUM_HIDDEN_LAYERS} layers, {NUM_STANDARD_LAYERS} with standard attention "
          f"(d={HEAD_DIM}, {NUM_KV_HEADS} KV heads)")
    print()

    # Per-context results
    for ctx_len in sorted(all_results.keys()):
        results = all_results[ctx_len]
        if not results:
            continue

        seq = results[0].seq_len
        fp16_kv_gb = (NUM_STANDARD_LAYERS * NUM_KV_HEADS * seq * HEAD_DIM * 2 * 2) / (1024**3)
        c3_bits, _ = compute_memory_bits(NUM_STANDARD_LAYERS, NUM_KV_HEADS, seq, HEAD_DIM, 3)
        tq3_kv_gb = c3_bits / 8 / (1024**3)

        print(f"--- Context: {seq} tokens ---")
        print(f"  FP16 KV would be: {fp16_kv_gb:.2f} GB")
        print(f"  TQ-3bit KV:       {tq3_kv_gb:.2f} GB ({fp16_kv_gb / tq3_kv_gb:.1f}x compression)")

        for r in results:
            print(f"\n  TQ-{r.bits}bit results:")
            print(f"    Cosine similarity: {r.avg_cosine_sim:.4f}")
            print(f"    Top-1 match: {r.top1_pct:.1f}%")
            print(f"    Top-5 match: {r.top5_pct:.1f}%")
        print()

    # Summary table
    print("-" * 70)
    print(f"{'Context':>8} | {'Bits':>4} | {'Compress':>8} | {'CosSim':>8} | "
          f"{'Top-1':>7} | {'Top-5':>7} | {'Needle':>7}")
    print(f"{'-' * 8}-+-{'-' * 4}-+-{'-' * 8}-+-{'-' * 8}-+-"
          f"{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}")

    for ctx_len in sorted(all_results.keys()):
        for r in all_results[ctx_len]:
            needle_str = f"{r.avg_needle_rank:.1f}" if r.avg_needle_rank >= 0 else "n/a"
            print(
                f"{r.seq_len:>8} | {r.bits:>4} | {r.compression_ratio:>7.1f}x | "
                f"{r.avg_cosine_sim:>8.4f} | {r.top1_pct:>6.1f}% | "
                f"{r.top5_pct:>6.1f}% | {needle_str:>7}"
            )
    print()

    # Verdict
    print("=" * 70)
    if fp16_max > available_gb and tq3_max < available_gb:
        print(f"VERDICT: FP16 OOMs at {MAX_POSITION_EMBEDDINGS // 1024}K context "
              f"({fp16_max:.1f} GB > {available_gb:.0f} GB available).")
        print(f"         TurboQuant 3-bit fits with {spare_gb:.1f} GB to spare "
              f"({tq3_max:.1f} GB).")
    elif fp16_max <= available_gb:
        print(f"VERDICT: FP16 fits at {MAX_POSITION_EMBEDDINGS // 1024}K ({fp16_max:.1f} GB). "
              f"TurboQuant saves {fp16_max - tq3_max:.1f} GB.")
    else:
        print(f"VERDICT: Both FP16 ({fp16_max:.1f} GB) and TQ-3bit ({tq3_max:.1f} GB) "
              f"exceed available VRAM ({available_gb:.0f} GB).")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: Long context validation with Qwen3.5-27B",
    )
    parser.add_argument(
        "--context",
        type=int,
        nargs="+",
        default=None,
        help="Target context lengths (tokens). Default: 8192 32768 65536",
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=None,
        help="Bit-widths to test. Default: 2 3 4",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    context_lengths = args.context if args.context else DEFAULT_CONTEXT_LENGTHS
    bit_widths = args.bits if args.bits else BIT_WIDTHS

    print()
    print("=" * 70)
    print("  TurboQuantDC -- Phase 3: Long Context Validation")
    print("  Algorithm: TurboQuant (PolarQuant MSE + QJL bias correction)")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Bit-widths: {bit_widths}")
    print(f"  Context lengths: {context_lengths}")
    print("=" * 70)

    # Print VRAM budget analysis first
    print_vram_budget()

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    print(f"GPU: {gpu_name} ({gpu_total_mb / 1024:.1f} GB)")
    print()

    # Reset memory tracking
    torch.cuda.reset_peak_memory_stats()

    # Load model
    try:
        model, tokenizer, n_layers, n_kv_heads, head_dim, model_gpu_gb = load_model()
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        print("  Ensure you have ~16 GB free VRAM and bitsandbytes installed.")
        traceback.print_exc()
        sys.exit(1)

    # Run validation at each context length
    all_results: Dict[int, List[BitWidthResult]] = {}

    for target_tokens in context_lengths:
        torch.cuda.reset_peak_memory_stats()

        results = run_validation(
            model=model,
            tokenizer=tokenizer,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            target_tokens=target_tokens,
            bit_widths=bit_widths,
            model_gpu_gb=model_gpu_gb,
        )

        if results is not None:
            print_results(results)
            all_results[target_tokens] = results
        else:
            print(f"\n  Skipped context={target_tokens} due to OOM.")

        # Clean up between context lengths
        gc.collect()
        torch.cuda.empty_cache()

    if all_results:
        print_final_report(all_results, model_gpu_gb)
    else:
        print("\nNo results collected. All context lengths OOM'd.")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")

    return all_results


if __name__ == "__main__":
    main()
