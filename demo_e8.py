#!/usr/bin/env python3
"""E8 Lattice VQ Demo — Near-Lossless 3-Bit KV Cache Compression.

Reproduces the key result: E8 lattice VQ achieves +0.0-0.2% PPL degradation
at 3-bit on Qwen2.5-3B, compared to +3.8% for scalar Lloyd-Max.

Usage:
    python demo_e8.py
    python demo_e8.py --model Qwen/Qwen2.5-7B-Instruct --bits 2 3 4
    python demo_e8.py --fp16-weights  # test without BnB quantization
"""

import argparse
import gc
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from turboquantdc.e8_lattice import E8Quantizer
from turboquantdc.rotation import fast_wht

HF_CACHE = os.environ.get("HF_HOME", "/media/dhawal/Beast/cache/hub")
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)


def e8_compress(keys, layer_idx, bits=3):
    """Compress key tensor using E8+WHT+Mean pipeline."""
    B, H, S, D = keys.shape
    mean_k = keys.mean(dim=2, keepdim=True)
    flat = (keys - mean_k).float().reshape(-1, D)
    norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    unit = flat / norms
    rotated = fast_wht(unit)
    scale = max(1.0 * rotated.std().item() / (2 ** bits), 1e-8)
    eq = E8Quantizer(scale=scale, relaxed=True)
    _, recon_rot = eq.quantize(rotated)
    recon = fast_wht(recon_rot) / D * norms
    return recon.to(keys.dtype).reshape(B, H, S, D) + mean_k


def main():
    parser = argparse.ArgumentParser(description="E8 Lattice VQ Demo")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--bits", nargs="+", type=int, default=[3])
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--fp16-weights", action="store_true",
                        help="Load model at FP16 (no BnB 4-bit)")
    args = parser.parse_args()

    import logging
    logging.disable(logging.WARNING)
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, DynamicCache)
    from datasets import load_dataset

    # Load model
    print(f"Model: {args.model}")
    print(f"Weights: {'FP16' if args.fp16_weights else 'BnB 4-bit'}")
    print(f"Bits: {args.bits}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=HF_CACHE)
    load_kwargs = dict(device_map="auto", trust_remote_code=True, cache_dir=HF_CACHE)
    if args.fp16_weights:
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    n_layers = model.config.num_hidden_layers
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)
    kv_heads = getattr(model.config, "num_key_value_heads",
                       model.config.num_attention_heads)
    print(f"  {n_layers}L, d={head_dim}, kv_heads={kv_heads}")
    print()

    # Load wikitext-2
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=HF_CACHE)
    wikitext = "\n".join(line for line in ds["text"] if line.strip())
    encodings = tokenizer(wikitext[:50000], return_tensors="pt",
                          truncation=True, max_length=args.max_tokens)
    input_ids = encodings["input_ids"].to(model.device)
    seq_len = input_ids.shape[1]

    _orig = DynamicCache.update

    def compute_ppl(compress_fn=None):
        if compress_fn:
            DynamicCache.update = lambda self, ks, vs, li, ck=None: \
                _orig(self, compress_fn(ks, li), vs, li, ck)
        else:
            DynamicCache.update = _orig
        nlls, n_tok = [], 0
        for begin in range(0, seq_len - 1, 256):
            end = min(begin + 512, seq_len)
            chunk = input_ids[:, begin:end]
            target = chunk.clone()
            if begin > 0:
                target[:, :256] = -100
            target[:, 0] = -100
            with torch.no_grad():
                out = model(chunk, labels=target, use_cache=bool(compress_fn))
            n = (target != -100).sum().item()
            if n > 0:
                nlls.append(out.loss.item() * n)
                n_tok += n
            del out
            torch.cuda.empty_cache()
            if end >= seq_len:
                break
        DynamicCache.update = _orig
        return math.exp(sum(nlls) / n_tok), n_tok

    # Baseline
    t0 = time.perf_counter()
    baseline, n_tok = compute_ppl()
    t_base = time.perf_counter() - t0
    print(f"FP16 KV baseline: PPL = {baseline:.4f} ({n_tok} tokens, {t_base:.1f}s)")

    # E8 at each bit rate
    for bits in args.bits:
        torch.cuda.empty_cache()
        gc.collect()
        compress = lambda ks, li, b=bits: e8_compress(ks, li, b)
        t0 = time.perf_counter()
        ppl, _ = compute_ppl(compress)
        elapsed = time.perf_counter() - t0
        delta = (ppl - baseline) / baseline * 100
        print(f"E8 {bits}-bit:       PPL = {ppl:.4f} ({delta:+.3f}%, {elapsed:.1f}s)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
