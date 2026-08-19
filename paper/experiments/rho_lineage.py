#!/usr/bin/env python3
"""Measure mean-dominance rho_h across the Qwen lineage.

rho_h = ||mu_h|| / E_t ||k_t - mu_h||  per KV head, post-RoPE (what the
quantizer sees), same definition as measure_key_mean.py but generic over
models. The lineage question: Qwen1.5 and Qwen2 are immune end-to-end while
Qwen2.5 collapses (see ppl_qwen1.5-1.8b.json etc.) - does rho, the mechanism's
central quantity, track that?

Usage:
    python rho_lineage.py --model <hf-or-path> --name qwen2-1.5b [--load-4bit]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RESULTS = HERE / "results"

from ppl_harness import load_model, SCRATCH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--tokens", type=int, default=2048)
    a = ap.parse_args()

    torch.manual_seed(42)
    text = (SCRATCH / "wikitext2_test.txt").read_text()
    model, tok = load_model(a.model, a.load_4bit)
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=a.tokens)["input_ids"].to(model.device)

    captured = {}
    from transformers.cache_utils import DynamicCache
    orig = DynamicCache.update

    def upd(self, k, v, layer_idx, cache_kwargs=None):
        if layer_idx not in captured:              # single window: first call only
            captured[layer_idx] = k[0].float().cpu()   # (H, T, D)
        return orig(self, k, v, layer_idx, cache_kwargs)

    DynamicCache.update = upd
    try:
        with torch.no_grad():
            model(ids, use_cache=True)
    finally:
        DynamicCache.update = orig

    per_layer = []
    for li in sorted(captured):
        k = captured[li]                            # (H, T, D)
        mu = k.mean(1, keepdim=True)                # (H, 1, D)
        dev = (k - mu).norm(dim=-1).mean(1)         # (H,)
        rho = (mu[:, 0].norm(dim=-1) / dev.clamp_min(1e-9))
        # fraction of the average key's squared magnitude that is the mean
        shared = float((mu[:, 0].norm(dim=-1) ** 2).sum()
                       / (k.norm(dim=-1) ** 2).mean(1).sum())
        per_layer.append(dict(layer=li, rho_mean=float(rho.mean()),
                              rho_max=float(rho.max()),
                              frac_heads_gt1=float((rho > 1).float().mean()),
                              shared_energy=shared))

    allr = [h for l in sorted(captured)
            for h in ((captured[l].mean(1, keepdim=True)[:, 0].norm(dim=-1) /
                       (captured[l] - captured[l].mean(1, keepdim=True))
                       .norm(dim=-1).mean(1).clamp_min(1e-9)).tolist())]
    out = dict(model=a.model, name=a.name, tokens=int(ids.shape[1]),
               n_layers=len(per_layer),
               rho_mean=float(np.mean(allr)), rho_max=float(np.max(allr)),
               frac_heads_gt1=float(np.mean([r > 1 for r in allr])),
               shared_energy_mean=float(np.mean([l["shared_energy"]
                                                 for l in per_layer])),
               per_layer=per_layer)
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"rho_{a.name}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"{a.name}: rho mean {out['rho_mean']:.3f} max {out['rho_max']:.1f} "
          f"heads>1 {100*out['frac_heads_gt1']:.1f}% "
          f"shared energy {100*out['shared_energy_mean']:.1f}%")
    print("wrote", p)


if __name__ == "__main__":
    main()
