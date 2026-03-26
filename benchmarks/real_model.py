"""Real model validation -- TurboQuant on actual LLM KV cache.

Loads Qwen2.5-3B-Instruct, runs needle-in-haystack prompts at multiple context
lengths, captures the KV cache, and compares full-precision attention scores
against TurboQuant-compressed scores across all layers and heads.

Reports: cosine similarity, top-1/top-5 attention match, needle rank,
compression ratio, and per-bit-width breakdowns.

Usage:
    python benchmarks/real_model.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Allow running from repo root: python benchmarks/real_model.py
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from turboquantdc import TurboQuantEstimator  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

NEEDLE = "The secret project code name is AURORA-7749."
NEEDLE_MARKER = "AURORA-7749"
QUESTION = "What is the secret project code name?"

TARGET_LENGTHS = [2048, 4096]
BIT_WIDTHS = [2, 3, 4]

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
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class HeadResult:
    """Per-head comparison result."""
    layer: int
    head: int
    cosine_sim: float
    top1_match: bool
    top5_match: bool
    needle_rank: Optional[int]  # None if needle not found in tokens


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
# Prompt construction
# ---------------------------------------------------------------------------
def build_prompt(tokenizer, target_tokens: int = 2048, needle_pos: float = 0.25) -> str:
    """Build a needle-in-haystack prompt with chat template.

    Places the needle at ``needle_pos`` fraction through the filler text.
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
    prompt = (
        f"<|im_start|>user\n{haystack}\n"
        f"Question: {QUESTION}<|im_end|>\n"
        f"<|im_start|>assistant\n"
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
    """Compute compressed and FP16 bit counts for the full KV cache.

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
    seed = layer_idx * 10000 + head_idx
    estimator = TurboQuantEstimator(
        d=head_dim, bits=bits, seed=seed, device=device
    )
    compressed = estimator.quantize(keys_fp)
    tq_scores = estimator.inner_product(query, compressed).squeeze(0)  # (seq_len,)

    # --- Metrics ---
    # Cosine similarity of the score vectors
    cos_sim = F.cosine_similarity(
        real_scores.unsqueeze(0).float(),
        tq_scores.unsqueeze(0).float(),
    ).item()

    # Top-1 match: does the most-attended token stay the same?
    real_top1 = real_scores.argmax().item()
    tq_top1 = tq_scores.argmax().item()
    top1_match = real_top1 == tq_top1

    # Top-5 match: is the real top-1 in compressed top-5?
    tq_top5 = tq_scores.topk(min(5, seq_len)).indices.tolist()
    top5_match = real_top1 in tq_top5

    # Needle rank: where does the needle token rank after compression?
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
# Main validation
# ---------------------------------------------------------------------------
def load_model():
    """Load Qwen2.5-3B-Instruct in 4-bit with BitsAndBytes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME} (4-bit NF4)...", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    load_time = time.time() - t0
    gpu_mb = torch.cuda.memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0

    # Extract model architecture info
    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    head_dim = config.hidden_size // n_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)

    print(f"  Loaded in {load_time:.1f}s | GPU: {gpu_mb} MB")
    print(f"  Layers: {n_layers} | Heads: {n_heads} | KV heads: {n_kv_heads} | head_dim: {head_dim}")
    print()

    return model, tokenizer, n_layers, n_heads, n_kv_heads, head_dim


def run_validation(
    model,
    tokenizer,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    target_tokens: int,
) -> List[BitWidthResult]:
    """Run validation for one context length across all bit-widths."""

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

    print(f"{'=' * 70}")
    print(f"Context: {seq_len} tokens | Needle at token {needle_token}")
    print(f"{'=' * 70}")

    # Forward pass -- capture KV cache
    print("  Running forward pass...", end="", flush=True)
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)
    fwd_time = time.time() - t0
    print(f" {fwd_time:.1f}s")

    cache = outputs.past_key_values

    # Validate cache structure
    # HuggingFace DynamicCache has .key_cache and .value_cache lists
    # or legacy tuple-of-tuples. Handle both.
    if hasattr(cache, "key_cache"):
        # DynamicCache (newer transformers)
        def get_keys(layer_idx):
            return cache.key_cache[layer_idx]  # (batch, n_kv_heads, seq, head_dim)

        def get_values(layer_idx):
            return cache.value_cache[layer_idx]

        actual_layers = len(cache.key_cache)
    elif hasattr(cache, "layers"):
        # EncoderDecoderCache with .layers attribute
        def get_keys(layer_idx):
            return cache.layers[layer_idx].keys

        def get_values(layer_idx):
            return cache.layers[layer_idx].values

        actual_layers = len(cache.layers)
    else:
        # Legacy tuple-of-tuples: ((keys, values), ...)
        def get_keys(layer_idx):
            return cache[layer_idx][0]

        def get_values(layer_idx):
            return cache[layer_idx][1]

        actual_layers = len(cache)

    # Verify dimensions
    sample_keys = get_keys(0)
    actual_kv_heads = sample_keys.shape[1]
    actual_head_dim = sample_keys.shape[3]
    actual_seq = sample_keys.shape[2]
    print(f"  Cache: {actual_layers} layers x {actual_kv_heads} KV heads x {actual_seq} seq x {actual_head_dim} head_dim")

    # Override with actual values
    n_kv_heads = actual_kv_heads
    head_dim = actual_head_dim
    n_layers_actual = actual_layers

    results: List[BitWidthResult] = []

    for bits in BIT_WIDTHS:
        print(f"\n  Evaluating TQ-{bits}bit...", end="", flush=True)
        t0 = time.time()

        # Memory accounting
        compressed_bits, fp16_bits = compute_memory_bits(
            n_layers_actual, n_kv_heads, actual_seq, head_dim, bits
        )

        bw_result = BitWidthResult(
            bits=bits,
            seq_len=actual_seq,
            needle_token=needle_token,
            compressed_bits=compressed_bits,
            fp16_bits=fp16_bits,
        )

        for layer_idx in range(n_layers_actual):
            keys = get_keys(layer_idx)  # (1, n_kv_heads, seq, head_dim)

            for h in range(n_kv_heads):
                # Extract this head's keys as float32 for accurate comparison
                k = keys[0, h].float()  # (seq, head_dim)

                head_result = compare_head(
                    keys_fp=k,
                    head_dim=head_dim,
                    bits=bits,
                    layer_idx=layer_idx,
                    head_idx=h,
                    needle_token=needle_token,
                )
                bw_result.head_results.append(head_result)

        eval_time = time.time() - t0
        print(f" {eval_time:.1f}s ({bw_result.n_heads} heads)")
        results.append(bw_result)

    return results


def print_results(results: List[BitWidthResult]) -> None:
    """Print formatted results for one context length."""
    for r in results:
        top1_count = sum(1 for h in r.head_results if h.top1_match)
        top5_count = sum(1 for h in r.head_results if h.top5_match)

        print(f"\n  TQ-{r.bits}bit:")
        print(f"    Compression:       {r.compression_ratio:.1f}x  ({r.compressed_mb:.1f} MB vs {r.fp16_mb:.1f} MB)")
        print(f"    Score cosine sim:  {r.avg_cosine_sim:.4f}  (1.0 = perfect)")
        print(f"    Top-1 match:       {r.top1_pct:.1f}%  ({top1_count}/{r.n_heads} heads)")
        print(f"    Top-5 match:       {r.top5_pct:.1f}%  ({top5_count}/{r.n_heads} heads)")
        if r.avg_needle_rank >= 0:
            print(f"    Avg needle rank:   {r.avg_needle_rank:.1f}  (lower = better, 0 = top)")


def print_summary_table(all_results: dict[int, List[BitWidthResult]]) -> None:
    """Print a summary table across all context lengths and bit-widths."""
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    # Header
    print(f"{'Context':>8} | {'Bits':>4} | {'Compress':>8} | {'CosSim':>8} | {'Top-1':>7} | {'Top-5':>7} | {'Needle':>7}")
    print(f"{'-' * 8}-+-{'-' * 4}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}")

    for ctx_len in sorted(all_results.keys()):
        for r in all_results[ctx_len]:
            needle_str = f"{r.avg_needle_rank:.1f}" if r.avg_needle_rank >= 0 else "n/a"
            print(
                f"{r.seq_len:>8} | {r.bits:>4} | {r.compression_ratio:>7.1f}x | "
                f"{r.avg_cosine_sim:>8.4f} | {r.top1_pct:>6.1f}% | "
                f"{r.top5_pct:>6.1f}% | {needle_str:>7}"
            )

    # Paper targets
    print(f"\n  Paper targets (3-bit): cosine sim > 0.995, top-5 > 90%, compression ~5.0x")


def main():
    """Run the full real model validation."""
    print()
    print("=" * 70)
    print("  TurboQuantDC -- Real Model Validation")
    print("  Algorithm: TurboQuant (PolarQuant MSE + QJL bias correction)")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Bit-widths: {BIT_WIDTHS}")
    print(f"  Context lengths: {TARGET_LENGTHS}")
    print("=" * 70)
    print()

    model, tokenizer, n_layers, n_heads, n_kv_heads, head_dim = load_model()

    all_results: dict[int, List[BitWidthResult]] = {}

    for target_tokens in TARGET_LENGTHS:
        results = run_validation(
            model, tokenizer, n_layers, n_kv_heads, head_dim, target_tokens
        )
        print_results(results)
        all_results[target_tokens] = results
        print()

        # Free some GPU memory between runs
        torch.cuda.empty_cache()

    print_summary_table(all_results)

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")

    # Return results for programmatic use
    return all_results


if __name__ == "__main__":
    main()
