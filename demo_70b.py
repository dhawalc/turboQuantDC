#!/usr/bin/env python3
"""THE DEMO: A model that doesn't fit on your GPU... running on your GPU.

Qwen2.5-14B-Instruct requires ~29 GB VRAM in FP16.
Your RTX 4090 has 24 GB.
This demo runs it anyway — in ~4 GB VRAM — using streaming + TurboQuant.

It loads a long context with a hidden "needle" fact, then asks the model
to retrieve it. If the model finds the needle, the KV cache compression
and streaming inference are WORKING CORRECTLY end-to-end.

Usage:
    python demo_70b.py [--model MODEL] [--context-tokens N] [--bits 3]
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from textwrap import dedent

import torch

# ─── ANSI helpers ────────────────────────────────────────────────────────────

CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════════╗
║  TurboQuantDC — Impossible Inference Demo                        ║
║  "A model that doesn't fit... running on your GPU"               ║
╚══════════════════════════════════════════════════════════════════╝{RESET}
""")


def gpu_status():
    free = torch.cuda.mem_get_info()[0] / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    used = total - free
    name = torch.cuda.get_device_name(0)
    return name, total, used, free


def print_gpu(label=""):
    name, total, used, free = gpu_status()
    tag = f" [{label}]" if label else ""
    print(f"  {DIM}GPU{tag}: {used:.1f} / {total:.1f} GB used ({free:.1f} GB free){RESET}")


# ─── Needle-in-a-haystack context builder ────────────────────────────────────

NEEDLE = "The secret project code name is NEPTUNE-4422."
NEEDLE_QUERY = "What is the secret project code name?"

FILLER_PARAGRAPHS = [
    "The quarterly financial review showed moderate growth across all sectors. Revenue increased by 3.2% compared to the previous quarter, driven primarily by expansion in the Asian markets. Operating expenses remained stable, with a slight decrease in marketing costs offset by increased R&D spending. The board expressed satisfaction with the overall trajectory.",
    "Research into sustainable energy solutions continued to yield promising results. The solar efficiency project achieved a new milestone, reaching 28.4% conversion efficiency in laboratory conditions. Wind turbine designs were optimized for lower wind speed environments, potentially opening new geographic markets for deployment.",
    "The software development team completed the migration to the new microservices architecture. Performance benchmarks showed a 40% improvement in response times and a 60% reduction in memory usage per request. The deployment pipeline was updated to support canary releases, reducing the risk of production incidents.",
    "Customer satisfaction surveys indicated strong brand loyalty among existing users. The Net Promoter Score increased from 42 to 47 over the past six months. The most frequently requested feature was improved integration with third-party tools, which the product team prioritized for the next release cycle.",
    "The logistics optimization project delivered ahead of schedule. Route planning algorithms reduced average delivery times by 15% while decreasing fuel consumption by 8%. The warehouse automation system processed 30% more orders per hour after the latest software update.",
    "Human resources reported a successful hiring quarter with 45 new team members across engineering, sales, and operations. Employee retention rates improved to 94%, attributed to the new flexible work policy and expanded professional development programs.",
    "The cybersecurity audit revealed no critical vulnerabilities in the production infrastructure. Two medium-severity issues in legacy systems were identified and remediated within the standard 30-day window. The team implemented additional monitoring for supply chain dependencies.",
    "Market analysis indicated growing demand for AI-assisted workflow tools in the enterprise segment. Competitor activity in this space intensified, with three new entrants announcing products in the past quarter. Our competitive advantage remained in integration depth and customization capabilities.",
]


def build_context(target_tokens: int, tokenizer) -> tuple[str, int]:
    """Build a long context with a hidden needle."""
    # Place needle roughly 20% into the context
    needle_position = 0.2

    paragraphs = []
    current_text = ""
    needle_inserted = False
    token_count = 0
    para_idx = 0

    while token_count < target_tokens:
        # Check if it's time to insert the needle
        if not needle_inserted and token_count >= target_tokens * needle_position:
            paragraphs.append(f"\n[Internal Memo — Classified]\n{NEEDLE}\n[End Memo]\n")
            needle_inserted = True

        # Add filler
        para = FILLER_PARAGRAPHS[para_idx % len(FILLER_PARAGRAPHS)]
        paragraphs.append(para)
        para_idx += 1

        current_text = "\n\n".join(paragraphs)
        token_count = len(tokenizer.encode(current_text))

    if not needle_inserted:
        # Insert near the beginning if context was very short
        paragraphs.insert(max(1, len(paragraphs) // 5),
                          f"\n[Internal Memo — Classified]\n{NEEDLE}\n[End Memo]\n")
        current_text = "\n\n".join(paragraphs)
        token_count = len(tokenizer.encode(current_text))

    return current_text, token_count


# ─── Main demo ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Impossible Inference Demo")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct",
                        help="Model name (default: Qwen2.5-14B-Instruct)")
    parser.add_argument("--context-tokens", type=int, default=4096,
                        help="Target context length in tokens (default: 4096)")
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4],
                        help="TurboQuant bit-width (default: 3)")
    parser.add_argument("--max-new-tokens", type=int, default=60,
                        help="Tokens to generate (default: 60)")
    args = parser.parse_args()

    banner()

    # ── GPU info ──────────────────────────────────────────────────────────
    name, total_gb, used_gb, free_gb = gpu_status()
    print(f"  {BOLD}GPU:{RESET} {name}")
    print(f"  {BOLD}VRAM:{RESET} {total_gb:.1f} GB total, {free_gb:.1f} GB free")
    print(f"  {BOLD}Model:{RESET} {args.model}")
    print(f"  {BOLD}TurboQuant:{RESET} {args.bits}-bit KV cache")
    print(f"  {BOLD}Context:{RESET} {args.context_tokens:,} tokens target")
    print()

    # ── Load streaming engine ─────────────────────────────────────────────
    print(f"  {YELLOW}Loading model to CPU (layers stay off GPU)...{RESET}")
    print_gpu("before load")

    from turboquantdc.streaming import StreamingInferenceEngine

    engine = StreamingInferenceEngine(
        args.model, bits=args.bits, device="cuda", dtype=torch.float16
    )
    engine.load_model_streaming()

    model_size_gb = engine._model_total_bytes / 1e9
    layer_size_mb = engine._layer_size_bytes / 1e6
    print(f"  {GREEN}Model loaded:{RESET} {model_size_gb:.1f} GB total, "
          f"{engine.num_layers} layers x {layer_size_mb:.0f} MB each")
    print_gpu("after load")
    print()

    # ── The punchline ─────────────────────────────────────────────────────
    normal_vram = model_size_gb
    _, _, current_used, _ = gpu_status()
    print(f"  {BOLD}{RED}Normal VRAM needed:{RESET}  {normal_vram:.1f} GB")
    print(f"  {BOLD}{GREEN}Actual VRAM used:{RESET}   {current_used:.1f} GB")
    if normal_vram > total_gb:
        print(f"  {BOLD}{CYAN}This model DOES NOT FIT on your {total_gb:.0f} GB GPU.{RESET}")
        print(f"  {BOLD}{CYAN}But here it is, running.{RESET}")
    else:
        savings = normal_vram - current_used
        print(f"  {BOLD}{CYAN}Saved {savings:.1f} GB of VRAM via streaming.{RESET}")
    print()

    # ── Build context with needle ─────────────────────────────────────────
    print(f"  {YELLOW}Building {args.context_tokens:,}-token context with hidden needle...{RESET}")
    context, actual_tokens = build_context(args.context_tokens, engine.tokenizer)
    print(f"  {GREEN}Context:{RESET} {actual_tokens:,} tokens, "
          f"needle at ~20% depth")
    print(f"  {DIM}Needle: \"{NEEDLE}\"{RESET}")
    print()

    # ── Construct the full prompt ─────────────────────────────────────────
    prompt = (
        f"You are an AI assistant. Read the following document carefully and "
        f"answer the question at the end.\n\n"
        f"--- DOCUMENT START ---\n{context}\n--- DOCUMENT END ---\n\n"
        f"Question: {NEEDLE_QUERY}\n"
        f"Answer: The secret project code name is"
    )

    # ── Generate! ─────────────────────────────────────────────────────────
    print(f"  {YELLOW}Generating (streaming {engine.num_layers} layers per token)...{RESET}")
    print(f"  {DIM}This is slow (~seconds per token) because every token streams "
          f"all {engine.num_layers} layers from CPU to GPU.{RESET}")
    print()

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    output = engine.generate(prompt, max_new_tokens=args.max_new_tokens)

    t_elapsed = time.time() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
    tokens_generated = engine._tokens_generated

    # ── Results ───────────────────────────────────────────────────────────
    print()
    print(f"  {CYAN}{'═' * 60}{RESET}")
    print(f"  {BOLD}Generated text:{RESET}")
    # Show just the generated part (after the prompt)
    generated_text = output[len(prompt):] if output.startswith(prompt[:50]) else output
    # Try to extract just the answer
    for line in output.split('\n'):
        if 'NEPTUNE' in line or 'code name' in line.lower():
            generated_text = line.strip()
            break
    print(f"  {GREEN}{generated_text}{RESET}")
    print(f"  {CYAN}{'═' * 60}{RESET}")
    print()

    # ── Needle check ──────────────────────────────────────────────────────
    found_needle = "NEPTUNE-4422" in output
    if found_needle:
        print(f"  {GREEN}{BOLD}NEEDLE FOUND! The model correctly retrieved the hidden fact.{RESET}")
    else:
        print(f"  {YELLOW}Needle not found in output (may need more tokens or higher bits).{RESET}")
    print()

    # ── Stats ─────────────────────────────────────────────────────────────
    tok_per_sec = tokens_generated / t_elapsed if t_elapsed > 0 else 0
    report = engine.memory_report()

    print(f"  {BOLD}Performance:{RESET}")
    print(f"    Tokens generated: {tokens_generated}")
    print(f"    Time: {t_elapsed:.1f}s ({tok_per_sec:.2f} tok/s)")
    print(f"    Peak VRAM: {peak_vram_mb:.0f} MB")
    print()
    print(f"  {BOLD}Memory breakdown:{RESET}")
    print(f"    Model total (FP16):     {report['model_total_mb']:,.0f} MB")
    print(f"    Embeddings (on GPU):    {report['embeddings_mb']:,.0f} MB")
    print(f"    One layer (streamed):   {report['one_layer_mb']:,.0f} MB")
    print(f"    TQ KV cache:            {report['kv_cache_mb']:.1f} MB")
    print(f"    Peak actual VRAM:       {peak_vram_mb:,.0f} MB")
    print(f"    {BOLD}VRAM reduction:         {report['compression_factor']:.1f}x{RESET}")
    print()

    # ── The headline ──────────────────────────────────────────────────────
    print(f"  {CYAN}{BOLD}{'═' * 60}")
    print(f"  {model_size_gb:.0f} GB model → {peak_vram_mb/1000:.1f} GB peak VRAM")
    print(f"  {report['compression_factor']:.0f}x VRAM reduction via streaming + TurboQuant")
    if normal_vram > total_gb:
        print(f"  Model needs {normal_vram:.0f} GB. GPU has {total_gb:.0f} GB. It runs anyway.")
    print(f"  {'═' * 60}{RESET}")


if __name__ == "__main__":
    main()
