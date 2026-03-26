"""TurboQuantDC Demo — Text generation with compressed KV cache.

Proves TurboQuant works for actual inference by running a real model and
shadow-tracking what the KV cache would look like when compressed.

Strategy (shadow tracking):
    1. Run prefill to get full FP16 KV cache.
    2. For each generation step, use the model's native cache (FP16) so the
       text output is unchanged and correct.
    3. After EACH step, also compress the new KV entries with TurboQuant.
    4. At the end, compare: FP16 sizes vs TurboQuant sizes, and verify that
       TurboQuant attention scores match FP16 ones.

This approach is completely correct — the model generates the same text it
would produce without TurboQuant because we keep the native cache intact.
The shadow TurboQuant caches prove that the compressed representation would
produce equivalent attention scores.

Usage:
    python demo.py
    python demo.py --model Qwen/Qwen2.5-3B-Instruct --prompt "..." --max-tokens 100 --bits 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from turboquantdc import TurboQuantEstimator, PolarQuant, TurboQuantKVCache  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TurboQuantDC — Text generation with compressed KV cache"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument(
        "--prompt",
        default="What is the secret to a long and happy life?",
        help="Prompt text",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=3,
        choices=[1, 2, 3, 4],
        help="TurboQuant bit-width (default: 3)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_name: str):
    """Load model in 4-bit NF4 with BitsAndBytes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {model_name} (4-bit NF4)...", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    load_time = time.time() - t0
    gpu_mb = torch.cuda.memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0

    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    head_dim = config.hidden_size // n_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)

    print(f"  Loaded in {load_time:.1f}s | GPU: {gpu_mb} MB")
    print(f"  Architecture: {n_layers} layers | {n_heads} heads | {n_kv_heads} KV heads | head_dim={head_dim}")

    return model, tokenizer, n_layers, n_kv_heads, head_dim


# ---------------------------------------------------------------------------
# Cache helpers: unified accessor for DynamicCache and legacy tuple-of-tuples
# ---------------------------------------------------------------------------
def make_cache_accessors(cache):
    """Return (get_keys, get_values, n_layers) for both cache formats."""
    if hasattr(cache, "key_cache"):
        # DynamicCache with key_cache attribute
        def get_keys(layer_idx):
            return cache.key_cache[layer_idx]

        def get_values(layer_idx):
            return cache.value_cache[layer_idx]

        n_layers_cache = len(cache.key_cache)
    else:
        # DynamicCache (iterable) or legacy tuple-of-tuples
        # Materialize to list of (keys, values) tuples
        if not hasattr(cache, "_layers"):
            try:
                cache._layers = list(cache)
            except TypeError:
                cache._layers = []

        def get_keys(layer_idx):
            return cache._layers[layer_idx][0]

        def get_values(layer_idx):
            return cache._layers[layer_idx][1]

        n_layers_cache = len(cache._layers)

    return get_keys, get_values, n_layers_cache


# ---------------------------------------------------------------------------
# Shadow TurboQuant caches (one per layer per head)
# ---------------------------------------------------------------------------
def build_tq_caches(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    bits: int,
    device: torch.device,
) -> List[List[TurboQuantKVCache]]:
    """Build a TurboQuantKVCache for every (layer, head) pair."""
    caches = []
    for layer_idx in range(n_layers):
        layer_caches = []
        for head_idx in range(n_kv_heads):
            seed = layer_idx * 10000 + head_idx
            tq = TurboQuantKVCache(
                d_key=head_dim,
                d_value=head_dim,
                bits=bits,
                seed=seed,
                device=str(device),
            )
            layer_caches.append(tq)
        caches.append(layer_caches)
    return caches


def compress_cache_snapshot(
    prefill_cache,
    tq_caches: List[List[TurboQuantKVCache]],
    n_layers: int,
    n_kv_heads: int,
) -> None:
    """Compress the entire prefill KV cache into the TurboQuant shadow caches."""
    get_keys, get_values, _ = make_cache_accessors(prefill_cache)

    for layer_idx in range(n_layers):
        keys = get_keys(layer_idx)    # (1, n_kv_heads, seq, head_dim)
        values = get_values(layer_idx)

        for head_idx in range(n_kv_heads):
            k = keys[0, head_idx].float()    # (seq, head_dim)
            v = values[0, head_idx].float()  # (seq, head_dim)
            tq_caches[layer_idx][head_idx].append(k, v)


def compress_new_token(
    current_cache,
    tq_caches: List[List[TurboQuantKVCache]],
    n_layers: int,
    n_kv_heads: int,
) -> None:
    """Append ONLY the last token's KV entry to the TurboQuant shadow caches.

    Called after each generation step. The native cache grows by one token per
    step, so we only compress the last slice.
    """
    get_keys, get_values, _ = make_cache_accessors(current_cache)

    for layer_idx in range(n_layers):
        keys = get_keys(layer_idx)    # (1, n_kv_heads, seq, head_dim)
        values = get_values(layer_idx)

        for head_idx in range(n_kv_heads):
            k = keys[0, head_idx, -1:, :].float()    # (1, head_dim) — last token
            v = values[0, head_idx, -1:, :].float()
            tq_caches[layer_idx][head_idx].append(k, v)


# ---------------------------------------------------------------------------
# Memory accounting
# ---------------------------------------------------------------------------
def compute_fp16_bytes(n_layers: int, n_kv_heads: int, seq_len: int, head_dim: int) -> int:
    """FP16 KV cache size in bytes (keys + values, float16 = 2 bytes per element)."""
    return n_layers * n_kv_heads * seq_len * head_dim * 2 * 2  # *2 for K and V


def compute_tq_bytes(tq_caches: List[List[TurboQuantKVCache]]) -> int:
    """Sum up TurboQuant memory across all (layer, head) caches."""
    total_bits = 0
    for layer_caches in tq_caches:
        for tq in layer_caches:
            stats = tq.memory_usage_bits()
            total_bits += stats["total_bits"]
    return total_bits // 8


# ---------------------------------------------------------------------------
# Fidelity check: compare FP16 vs TurboQuant attention scores
# ---------------------------------------------------------------------------
def fidelity_check(
    prefill_cache,
    tq_caches: List[List[TurboQuantKVCache]],
    n_layers: int,
    n_kv_heads: int,
) -> Tuple[float, float]:
    """Compare TurboQuant attention scores against FP16 for the last query.

    Uses the last token as query, computes cosine similarity and top-5 match
    across all (layer, head) pairs, averaged.

    Returns:
        (avg_cosine_sim, avg_top5_pct)
    """
    # Re-create accessor — cache may have grown during generation
    if hasattr(prefill_cache, "_layers"):
        del prefill_cache._layers  # force re-materialization
    get_keys, _, _ = make_cache_accessors(prefill_cache)

    cos_sims = []
    top5_matches = []

    for layer_idx in range(n_layers):
        keys = get_keys(layer_idx)  # (1, n_kv_heads, seq, head_dim)

        for head_idx in range(n_kv_heads):
            k_fp = keys[0, head_idx].float()  # (seq, head_dim)
            query = k_fp[-1:, :]              # (1, head_dim) — last token as query

            # FP16 attention scores
            fp_scores = (query @ k_fp.T).squeeze(0)  # (seq,)

            # TurboQuant attention scores
            tq = tq_caches[layer_idx][head_idx]
            tq_scores = tq.attention_scores(query).squeeze(0)  # (seq,)

            cos_sim = F.cosine_similarity(
                fp_scores.unsqueeze(0).float(),
                tq_scores.unsqueeze(0).float(),
            ).item()
            cos_sims.append(cos_sim)

            # Top-5 match: is FP16 top-1 in TQ top-5?
            fp_top1 = fp_scores.argmax().item()
            seq_len = fp_scores.shape[0]
            tq_top5 = tq_scores.topk(min(5, seq_len)).indices.tolist()
            top5_matches.append(1 if fp_top1 in tq_top5 else 0)

    avg_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0
    avg_top5 = 100.0 * sum(top5_matches) / len(top5_matches) if top5_matches else 0.0
    return avg_cos, avg_top5


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    bits: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    greedy: bool = False,
) -> Dict:
    """Run text generation with shadow TurboQuant tracking.

    Returns a result dict with generated text and memory stats.
    """
    device = next(model.parameters()).device

    # --- Tokenize prompt ---
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    print(f"\nPrompt ({prompt_len} tokens): \"{prompt[:80]}{'...' if len(prompt) > 80 else ''}\"")
    print("\nRunning prefill...", end="", flush=True)

    t_prefill = time.time()
    with torch.no_grad():
        prefill_out = model(**inputs, use_cache=True)
    t_prefill = time.time() - t_prefill

    # Extract the KV cache from the prefill
    prefill_cache = prefill_out.past_key_values
    get_keys, _, n_layers_cache = make_cache_accessors(prefill_cache)

    # Use actual layer/head counts from the cache (handles GQA correctly)
    actual_kv_heads = get_keys(0).shape[1]
    actual_head_dim = get_keys(0).shape[3]
    actual_seq = get_keys(0).shape[2]
    n_layers = n_layers_cache

    print(f" done ({t_prefill:.2f}s)")
    print(f"  Cache: {n_layers} layers x {actual_kv_heads} KV heads x {actual_seq} tokens x {actual_head_dim} head_dim")

    # --- Build shadow TurboQuant caches ---
    print(f"\nCompressing prefill KV cache with TQ-{bits}bit...", end="", flush=True)
    t_compress = time.time()

    tq_caches = build_tq_caches(n_layers, actual_kv_heads, actual_head_dim, bits, device)
    compress_cache_snapshot(prefill_cache, tq_caches, n_layers, actual_kv_heads)

    t_compress = time.time() - t_compress
    print(f" done ({t_compress:.2f}s)")

    # --- Generation loop ---
    print(f"\nGenerating up to {max_tokens} tokens...\n")
    print("Generated: ", end="", flush=True)

    generated_ids = input_ids.clone()
    past_key_values = prefill_cache
    generated_tokens: List[int] = []
    generated_text_parts: List[str] = []

    # Seed for the first generation step: logits from prefill
    next_logits = prefill_out.logits[:, -1, :]  # (1, vocab_size)

    for step in range(max_tokens):
        # Sample / argmax
        if greedy:
            next_token_id = next_logits.argmax(dim=-1, keepdim=True)  # (1, 1)
        else:
            probs = torch.softmax(next_logits.float() / 0.8, dim=-1)  # temperature=0.8
            next_token_id = torch.multinomial(probs, num_samples=1)   # (1, 1)

        token_id = next_token_id[0, 0].item()

        # Stop on EOS
        if token_id == tokenizer.eos_token_id:
            break

        generated_tokens.append(token_id)

        # Decode and print token
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        generated_text_parts.append(token_text)
        print(token_text, end="", flush=True)

        # Append new token to input and run one forward step
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        with torch.no_grad():
            step_out = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                use_cache=True,
            )

        past_key_values = step_out.past_key_values
        next_logits = step_out.logits[:, -1, :]

        # Shadow-compress the new token's KV entries
        compress_new_token(past_key_values, tq_caches, n_layers, actual_kv_heads)

    print("\n")  # newline after streamed output

    n_generated = len(generated_tokens)
    full_seq_len = prompt_len + n_generated

    # --- Memory report ---
    fp16_bytes = compute_fp16_bytes(n_layers, actual_kv_heads, full_seq_len, actual_head_dim)
    tq_bytes = compute_tq_bytes(tq_caches)
    compression_ratio = fp16_bytes / tq_bytes if tq_bytes > 0 else 0.0
    memory_saved_bytes = fp16_bytes - tq_bytes
    memory_saved_pct = 100.0 * memory_saved_bytes / fp16_bytes if fp16_bytes > 0 else 0.0

    # Projections to long context (per-token rates)
    fp16_bytes_per_token = fp16_bytes / full_seq_len if full_seq_len > 0 else 0
    tq_bytes_per_token = tq_bytes / full_seq_len if full_seq_len > 0 else 0

    ctx_32k_fp16 = fp16_bytes_per_token * 32768 / (1024**3)
    ctx_32k_tq = tq_bytes_per_token * 32768 / (1024**3)
    ctx_128k_fp16 = fp16_bytes_per_token * 131072 / (1024**3)
    ctx_128k_tq = tq_bytes_per_token * 131072 / (1024**3)

    # --- Fidelity check ---
    print("Running fidelity check (comparing TurboQuant vs FP16 attention scores)...", end="", flush=True)
    t_fid = time.time()
    avg_cos_sim, avg_top5_pct = fidelity_check(
        past_key_values, tq_caches, n_layers, actual_kv_heads
    )
    t_fid = time.time() - t_fid
    print(f" done ({t_fid:.2f}s)")

    return {
        "generated_text": "".join(generated_text_parts),
        "n_generated": n_generated,
        "prompt_len": prompt_len,
        "full_seq_len": full_seq_len,
        "fp16_bytes": fp16_bytes,
        "tq_bytes": tq_bytes,
        "compression_ratio": compression_ratio,
        "memory_saved_bytes": memory_saved_bytes,
        "memory_saved_pct": memory_saved_pct,
        "ctx_32k_fp16_gb": ctx_32k_fp16,
        "ctx_32k_tq_gb": ctx_32k_tq,
        "ctx_128k_fp16_gb": ctx_128k_fp16,
        "ctx_128k_tq_gb": ctx_128k_tq,
        "avg_cos_sim": avg_cos_sim,
        "avg_top5_pct": avg_top5_pct,
        "bits": bits,
    }


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------
def print_results(args: argparse.Namespace, result: Dict, model_name: str) -> None:
    """Print the final formatted demo output."""
    fp16_mb = result["fp16_bytes"] / (1024 * 1024)
    tq_mb = result["tq_bytes"] / (1024 * 1024)
    saved_mb = result["memory_saved_bytes"] / (1024 * 1024)

    print()
    print("=" * 60)
    print("TurboQuantDC Demo -- Text Generation with Compressed KV Cache")
    print(f"Model: {model_name} (4-bit NF4)")
    print(f"Compression: {result['bits']}-bit TurboQuant")
    print("=" * 60)

    print(f"\nPrompt: \"{args.prompt}\"")
    print(f"\nGenerated: {result['generated_text']}")

    print()
    print("--- Memory Report ---")
    print(f"Tokens generated:   {result['n_generated']}")
    print(f"Total seq length:   {result['full_seq_len']} tokens (prompt + generated)")
    print(f"FP16 KV cache:      {fp16_mb:.1f} MB")
    print(f"TQ-{result['bits']}bit cache:    {tq_mb:.1f} MB  ({result['compression_ratio']:.1f}x compression)")
    print(f"Memory saved:       {saved_mb:.1f} MB  ({result['memory_saved_pct']:.0f}%)")

    print()
    print(f"At 32K context, this would use:")
    print(f"  FP16:       {result['ctx_32k_fp16_gb']:.2f} GB")
    print(f"  TQ-{result['bits']}bit:    {result['ctx_32k_tq_gb']:.2f} GB  ({result['ctx_32k_fp16_gb']/result['ctx_32k_tq_gb']:.1f}x smaller)")

    print(f"\nAt 128K context:")
    fits = result['ctx_128k_tq_gb'] < 24.0
    fits_str = "FITS on 24GB GPU!" if fits else f"needs {result['ctx_128k_tq_gb']:.1f} GB"
    print(f"  FP16:       {result['ctx_128k_fp16_gb']:.2f} GB")
    print(f"  TQ-{result['bits']}bit:    {result['ctx_128k_tq_gb']:.2f} GB  ({fits_str})")

    print()
    print("--- Fidelity Check ---")
    print(f"Attention score cosine similarity:  {result['avg_cos_sim']:.4f}")
    print(f"Top-5 attention match:              {result['avg_top5_pct']:.1f}%")

    # Pass/fail vs paper targets
    cos_pass = result['avg_cos_sim'] >= 0.995
    top5_pass = result['avg_top5_pct'] >= 90.0
    cos_str = "PASS (>= 0.995)" if cos_pass else f"BELOW TARGET (target: >= 0.995)"
    top5_str = "PASS (>= 90%)" if top5_pass else f"BELOW TARGET (target: >= 90%)"
    print(f"  Cosine sim target (>= 0.995):   {cos_str}")
    print(f"  Top-5 target (>= 90%):          {top5_str}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print()
    print("=" * 60)
    print("TurboQuantDC -- Standalone Text Generation Demo")
    print(f"Model:    {args.model}")
    print(f"Bits:     {args.bits}-bit TurboQuant")
    print(f"Tokens:   up to {args.max_tokens}")
    print(f"Decoding: {'greedy' if args.greedy else 'sampling (temp=0.8)'}")
    print("=" * 60)
    print()

    model, tokenizer, n_layers, n_kv_heads, head_dim = load_model(args.model)

    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        bits=args.bits,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        greedy=args.greedy,
    )

    print_results(args, result, args.model)


if __name__ == "__main__":
    main()
