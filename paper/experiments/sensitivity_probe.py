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

    score-space noise (1 - worst-layer logit_r)  ->  log PPL ratio

is the model's private calibration. Result (SS6.22): quantizer cells land ON
their model's curve in BOTH regimes (R^2 0.974 over 96 cells), so damage is a
model-specific function of one cheap scalar. The hypothesis that mean-pathology
cells would deviate ABOVE the curve was refuted — such models simply have
catastrophically steep curves.

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
    synthetic noise on keys. Values pass through untouched.

    ``family`` selects the noise shape, which is the variable SS6.22's residual
    analysis points at: 1-bit quantizer error is the least Gaussian error in
    the study and is exactly where the law under-predicts.

      gauss   isotropic N(0, I)                     -- the default
      unif    uniform on [-sqrt(3), sqrt(3)]        -- flat, bounded (scalar
              quantization error is uniform within a bin)
      sign    +/- 1 Rademacher                      -- maximally structured,
              mimics 1-bit index error
      round   deterministic rounding to a grid of   -- actual scalar
              step sigma*scale*sqrt(12), the        quantization, not a
              closest thing to a real quantizer     stochastic surrogate

    All families are scaled to the same per-vector RMS perturbation, so a given
    sigma means the same relative key error regardless of shape.
    """

    def __init__(self, sigma, n_probes=128, seed=SEED, family="gauss"):
        self.sigma = sigma
        self.family = family
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
        g = self._gen[layer_idx]
        d = k.shape[-1]
        scale = k.float().norm(dim=-1, keepdim=True).to(k.dtype) / (d ** 0.5)
        fam = self.family
        if fam == "round":
            # deterministic scalar quantization to a uniform grid; error is
            # uniform in [-step/2, step/2], so step = sigma*scale*sqrt(12)
            step = (self.sigma * scale * (12 ** 0.5)).clamp_min(1e-8)
            kq = torch.round(k / step) * step
            return self._finish(k, kq, v, d, layer_idx)
        if fam == "unif":
            eps = (torch.rand(k.shape, generator=g) * 2 - 1) * (3 ** 0.5)
        elif fam == "sign":
            eps = torch.randint(0, 2, k.shape, generator=g).float() * 2 - 1
        else:
            eps = torch.randn(k.shape, generator=g)
        eps = eps.to(k.device, k.dtype)
        kq = k + self.sigma * scale * eps
        return self._finish(k, kq, v, d, layer_idx)

    def _finish(self, k, kq, v, d, layer_idx):
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
    ap.add_argument("--family", default="gauss",
                    choices=["gauss", "unif", "sign", "round"],
                    help="noise shape; see NoiseInjector docstring")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--suffix", default="",
                    help="appended to the output name, e.g. -round")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    text = (SCRATCH / "wikitext2_test.txt").read_text()
    model, tok = load_model(a.model, a.load_4bit)
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=a.tokens)["input_ids"].to(model.device)
    print(f"=== sensitivity {a.name} ({ids.shape[1]} tokens) ===", flush=True)

    base, ntok = sliding_ppl(model, ids)
    print(f"[baseline] ppl={base:.4f}", flush=True)
    rows = [dict(sigma=0.0, ppl=base, ratio=1.0)]
    for sigma in a.sigmas:
        inj = NoiseInjector(sigma, seed=a.seed, family=a.family)
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
    out = RESULTS / f"sensitivity_{a.name}{a.suffix}.json"
    if out.exists():
        # merge with an existing curve so the grid can be refined incrementally
        old = json.loads(out.read_text())
        have = {r["sigma"] for r in rows}
        rows = rows + [r for r in old.get("rows", []) if r["sigma"] not in have]
        rows.sort(key=lambda r: r["sigma"])
        print(f"merged with existing curve -> {len(rows)} points", flush=True)
    json.dump(dict(model=a.model, name=a.name, tokens=int(ids.shape[1]),
                   load_4bit=a.load_4bit, family=a.family, seed=a.seed,
                   rows=rows), open(out, "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
