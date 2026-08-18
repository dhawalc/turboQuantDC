#!/usr/bin/env python3
"""Measure a model's key-noise sensitivity curve — the second factor of damage.

SS6.19 established that reconstruction-side proxies measure the noise a
quantizer injects but not the model's sensitivity to it (damage = noise x
sensitivity), and that the sensitivity factor is what they structurally cannot
see. This probe measures it directly.

Keys are perturbed with mean-free isotropic Gaussian noise — no quantizer, no
shared-component pathology, pure noise:

    k_hat = k + sigma * (||k||_rms / sqrt(d)) * eps,   eps ~ N(0, I)

For each sigma we record sliding-window perplexity AND the same proxy metrics
(vec_cos, per-layer logit r) the atlas records for quantizer cells, so noise
cells and quantizer cells live on a common axis. The resulting per-model curve

    score-space noise (1 - logit_r)  ->  log PPL ratio

is the model's private calibration. The factorization hypothesis (SS6.22):
uniform-damage quantizer cells (e.g. 1-bit) land ON their model's curve —
their "proxy-invisible" damage is predictable — while mean-pathology cells
(Qwen2/2.5 uncentered) land far ABOVE it, and the deviation is a principled
pathology detector.

Usage:
    python sensitivity_probe.py --model <hf-or-path> --name llama3.2-1b
"""
from __future__ import annotations
import argparse, gc, json, time
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RESULTS = HERE / "results"

from ppl_harness import (SCRATCH, SEED, load_model, sliding_ppl, proxy_metrics)
import numpy as np


class NoiseInjector:
    """Same interface as KVCompressor, but the 'compression' is calibrated
    isotropic Gaussian noise on keys. Values pass through untouched."""

    def __init__(self, sigma, n_probes=128, seed=SEED):
        self.sigma = sigma
        self.n_probes, self.seed = n_probes, seed
        self.stats = {}
        self._probes = {}
        self._gen = {}

    def probes(self, d, device):
        if d not in self._probes:
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            p = torch.randn(self.n_probes, d, generator=g)
            self._probes[d] = (p / p.norm(dim=-1, keepdim=True)).to(device)
        return self._probes[d].to(device)

    def reset(self):
        pass  # noise is stateless; keep accumulated stats across windows

    def transform(self, k, v, layer_idx):
        if layer_idx not in self._gen:
            self._gen[layer_idx] = torch.Generator(device="cpu").manual_seed(
                self.seed + 7919 * layer_idx)
        eps = torch.randn(k.shape, generator=self._gen[layer_idx]).to(
            k.device, k.dtype)
        d = k.shape[-1]
        scale = k.float().norm(dim=-1, keepdim=True).to(k.dtype) / (d ** 0.5)
        kq = k + self.sigma * scale * eps
        with torch.no_grad():
            vc, lr, sr = proxy_metrics(k[0].float(), kq[0].float(),
                                       self.probes(d, k.device))
        s = self.stats.setdefault(layer_idx,
                                  {"vec_cos": [], "logit_r": [], "spread": []})
        s["vec_cos"].append(vc); s["logit_r"].append(lr); s["spread"].append(sr)
        return kq, v

    def summary(self):
        out = {}
        for key in ("vec_cos", "logit_r", "spread"):
            per_layer = [float(np.mean(s[key])) for s in self.stats.values()]
            out[key] = float(np.mean(per_layer))
            out[key + "_min"] = float(np.min(per_layer)) if per_layer else float("nan")
        out["per_layer_logit_r"] = {int(i): float(np.mean(s["logit_r"]))
                                    for i, s in sorted(self.stats.items())}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.05, 0.1, 0.2, 0.4, 0.8])
    a = ap.parse_args()

    torch.manual_seed(SEED)
    text = (SCRATCH / "wikitext2_test.txt").read_text()
    model, tok = load_model(a.model, a.load_4bit)
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=a.tokens)["input_ids"].to(model.device)
    print(f"=== sensitivity {a.name} ({ids.shape[1]} tokens) ===", flush=True)

    base, ntok = sliding_ppl(model, ids)
    print(f"[baseline] ppl={base:.4f}", flush=True)
    rows = [dict(sigma=0.0, ppl=base, ratio=1.0)]
    for sigma in a.sigmas:
        inj = NoiseInjector(sigma)
        t0 = time.time()
        ppl, _ = sliding_ppl(model, ids, inj)
        s = inj.summary()
        rows.append(dict(sigma=sigma, ppl=ppl, ratio=ppl / base, **s))
        print(f"[sigma={sigma}] ppl={ppl:.4f} (x{ppl/base:.3f}) "
              f"vec_cos={s['vec_cos']:.4f} logit_r={s['logit_r']:.4f} "
              f"(min {s['logit_r_min']:.4f}) {time.time()-t0:.0f}s", flush=True)
        del inj
        gc.collect(); torch.cuda.empty_cache()

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"sensitivity_{a.name}.json"
    json.dump(dict(model=a.model, name=a.name, tokens=int(ids.shape[1]),
                   load_4bit=a.load_4bit, rows=rows), open(out, "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
