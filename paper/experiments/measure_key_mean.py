#!/usr/bin/env python3
"""Measure the per-head key mean magnitude on Qwen2.5 models.

This tests prediction P3 of paper/qwen_kv_quantization_failure.md Section 4.3:

    Failure severity should scale with  rho_h = ||mu_h|| / E_t||k_t - mu_h||

where mu_h is the mean over sequence positions of the cached key vectors for
KV head h, and the expectation is over positions.

The hypothesis in Section 4.2 predicts:
    Qwen2.5-3B  (2 KV heads) -> large rho
    Qwen2.5-7B  (4 KV heads) -> large rho
    Qwen2.5-14B (8 KV heads) -> small rho

If rho is similar across all three models, the mechanism proposed in Section 4.2
is WRONG and the KV-head correlation reported in Section 6.3 has another cause.
Report the result either way.

Note on what is measured: keys are read out of the model's KV cache, i.e. they
are POST-RoPE. That is deliberate -- the quantizer under study receives post-RoPE
keys, because HuggingFace applies the rotary embedding before calling
``cache.update()``. RoPE applies a position-dependent rotation, which would tend
to average a shared component *down*; a large rho measured post-RoPE is therefore
a stronger result than one measured pre-RoPE.

Run:
    python paper/experiments/measure_key_mean.py
    python paper/experiments/measure_key_mean.py --models 7B --tokens 2048

Output:
    paper/experiments/results/key_mean_stats.json
    paper/experiments/results/key_mean_stats.md
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "results"

# Match the cache location used by benchmarks/ppl_for_tom.py. Override with
# HF_HOME in the environment if the weights live elsewhere.
HF_CACHE_DIR = os.environ.get("HF_HOME", "/media/dhawal/Beast/cache/hub")
os.environ["HF_HOME"] = HF_CACHE_DIR

MODELS = {
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
    "14B": "Qwen/Qwen2.5-14B-Instruct",
}

SEED = 42
DEFAULT_TOKENS = 4096


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_wikitext2_test() -> str:
    """Load wikitext-2 test split as one string (same source as the PPL runs)."""
    from datasets import load_dataset

    ds = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="test", cache_dir=HF_CACHE_DIR,
    )
    return "\n".join(line for line in ds["text"] if line.strip())


# ---------------------------------------------------------------------------
# Cache access
# ---------------------------------------------------------------------------

def extract_layer_keys(past_key_values: Any) -> List[torch.Tensor]:
    """Return per-layer key tensors ``[batch, kv_heads, seq, head_dim]``.

    The Cache API has changed shape across transformers versions, so try the
    known accessors in order rather than assuming one.
    """
    # transformers >= 4.5x: Cache object with a .layers list of DynamicLayer
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        keys = [getattr(layer, "keys", None) for layer in layers]
        if all(k is not None for k in keys):
            return list(keys)

    # older Cache objects expose .key_cache
    key_cache = getattr(past_key_values, "key_cache", None)
    if key_cache is not None and len(key_cache) > 0:
        return list(key_cache)

    # legacy tuple-of-tuples format
    try:
        return [entry[0] for entry in past_key_values]
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Could not extract keys from cache of type {type(past_key_values)}"
        ) from exc


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def head_stats(keys: torch.Tensor) -> List[Dict[str, float]]:
    """Per-KV-head statistics for one layer.

    Args:
        keys: ``[batch, kv_heads, seq, head_dim]`` cached keys for one layer.

    Returns:
        One dict per KV head with:
          rho          ||mu|| / E_t||k_t - mu||   (the P3 quantity)
          mean_norm    ||mu||
          dev_norm     E_t||k_t - mu||
          key_norm     E_t||k_t||
          mean_energy_frac  ||mu||^2 / E_t||k_t||^2, in [0, 1]
          cos_to_mean  E_t cos(k_t, mu) -- how aligned raw keys are with the mean
    """
    k = keys.float()[0]                       # (H, T, D) -- batch 0
    mu = k.mean(dim=1, keepdim=True)          # (H, 1, D)

    mean_norm = mu.squeeze(1).norm(dim=-1)                     # (H,)
    dev_norm = (k - mu).norm(dim=-1).mean(dim=1)               # (H,)
    key_norm = k.norm(dim=-1).mean(dim=1)                      # (H,)
    key_sq = (k.norm(dim=-1) ** 2).mean(dim=1)                 # (H,)

    cos_to_mean = torch.nn.functional.cosine_similarity(
        k, mu.expand_as(k), dim=-1,
    ).mean(dim=1)                                              # (H,)

    rho = mean_norm / dev_norm.clamp_min(1e-8)
    energy_frac = (mean_norm ** 2) / key_sq.clamp_min(1e-8)

    return [
        {
            "head": int(h),
            "rho": float(rho[h]),
            "mean_norm": float(mean_norm[h]),
            "dev_norm": float(dev_norm[h]),
            "key_norm": float(key_norm[h]),
            "mean_energy_frac": float(energy_frac[h]),
            "cos_to_mean": float(cos_to_mean[h]),
        }
        for h in range(k.shape[0])
    ]


def measure_model(model_name: str, text: str, max_tokens: int) -> Dict[str, Any]:
    """Run one forward pass and summarize the per-head key mean."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\nLoading {model_name} (BnB 4-bit NF4)...")
    t0 = time.perf_counter()

    # Same weight quantization as the PPL/NIAH runs, so the activations we
    # measure are the ones the reported failure was produced from.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=HF_CACHE_DIR,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
    )
    model.eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    cfg = model.config
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads
    )

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = enc["input_ids"].to(model.device)
    print(f"  {cfg.num_hidden_layers} layers, {kv_heads} KV heads, "
          f"head_dim={head_dim}, {input_ids.shape[1]} tokens")

    with torch.no_grad():
        out = model(input_ids, use_cache=True)

    layer_keys = extract_layer_keys(out.past_key_values)

    per_layer: List[Dict[str, Any]] = []
    for idx, keys in enumerate(layer_keys):
        if keys is None or keys.numel() == 0:
            continue
        heads = head_stats(keys)
        rhos = [h["rho"] for h in heads]
        per_layer.append({
            "layer": idx,
            "rho_mean": sum(rhos) / len(rhos),
            "rho_max": max(rhos),
            "rho_min": min(rhos),
            "heads": heads,
        })

    all_rho = [h["rho"] for L in per_layer for h in L["heads"]]
    all_frac = [h["mean_energy_frac"] for L in per_layer for h in L["heads"]]
    all_rho_sorted = sorted(all_rho)

    summary = {
        "model": model_name,
        "num_layers": int(cfg.num_hidden_layers),
        "kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "eval_tokens": int(input_ids.shape[1]),
        "rho_mean": sum(all_rho) / len(all_rho),
        "rho_median": all_rho_sorted[len(all_rho_sorted) // 2],
        "rho_max": max(all_rho),
        "rho_p90": all_rho_sorted[int(0.9 * (len(all_rho_sorted) - 1))],
        "mean_energy_frac_mean": sum(all_frac) / len(all_frac),
        "per_layer": per_layer,
    }

    print(f"  rho: mean={summary['rho_mean']:.3f} median={summary['rho_median']:.3f} "
          f"p90={summary['rho_p90']:.3f} max={summary['rho_max']:.3f}")
    print(f"  mean energy fraction: {summary['mean_energy_frac_mean']:.3f}")

    del model, tokenizer, out
    gc.collect()
    torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(results: Dict[str, Any], tokens: int) -> str:
    lines = [
        "# Per-Head Key Mean Magnitude (P3 test)",
        "",
        f"Date: {time.strftime('%Y-%m-%d %H:%M')}",
        f"Dataset: wikitext-2-raw-v1 test, first {tokens} tokens",
        "Weights: BnB 4-bit NF4 (matches the PPL/NIAH runs)",
        "Keys measured post-RoPE, as received by the KV cache.",
        "",
        "rho = ||mu_h|| / E_t||k_t - mu_h||   (larger = key mean dominates)",
        "",
        "| Model | KV heads | rho mean | rho median | rho p90 | rho max | mean energy frac |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("3B", "7B", "14B"):
        r = results.get(key)
        if not r or "error" in r:
            continue
        lines.append(
            f"| {r['model'].split('/')[-1]} | {r['kv_heads']} | {r['rho_mean']:.3f} | "
            f"{r['rho_median']:.3f} | {r['rho_p90']:.3f} | {r['rho_max']:.3f} | "
            f"{r['mean_energy_frac_mean']:.3f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Section 4.2 of the manuscript predicts LARGE rho for the models that failed",
        "(3B, 7B) and SMALL rho for the model that did not (14B).",
        "",
        "- If that ordering holds, the proposed mechanism is supported.",
        "- If rho is comparable across all three, the mechanism is WRONG and the",
        "  KV-head correlation in Section 6.3 needs a different explanation.",
        "",
        "Record whichever outcome occurs. A negative result here is a publishable",
        "correction to Section 4, not a reason to withhold the measurement.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=["3B", "7B", "14B"], choices=list(MODELS),
        help="Which Qwen2.5 sizes to measure.",
    )
    parser.add_argument(
        "--tokens", type=int, default=DEFAULT_TOKENS,
        help="Number of wikitext-2 test tokens in the forward pass.",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading wikitext-2 test split...")
    text = load_wikitext2_test()

    results: Dict[str, Any] = {}
    for key in args.models:
        name = MODELS[key]
        try:
            results[key] = measure_model(name, text, args.tokens)
        except Exception as exc:
            print(f"  FAILED on {name}: {exc}")
            results[key] = {"model": name, "error": str(exc)}

    report = write_report(results, args.tokens)
    print("\n" + report)

    (OUT_DIR / "key_mean_stats.json").write_text(json.dumps(results, indent=2))
    (OUT_DIR / "key_mean_stats.md").write_text(report)
    print(f"\nWrote {OUT_DIR / 'key_mean_stats.json'}")
    print(f"Wrote {OUT_DIR / 'key_mean_stats.md'}")


if __name__ == "__main__":
    main()
