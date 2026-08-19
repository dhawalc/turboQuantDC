#!/usr/bin/env python3
"""Is there a *free* improvement over mean-removal?

TurboQuant's Lloyd-Max codebook is optimal for rotated coordinates distributed
as approximately N(0, 1/d). Mean-removal fixes one way real keys violate that
assumption (a non-zero shared component). This asks whether fixing the other
obvious violations -- anisotropic per-channel scale, and non-unit per-coordinate
variance after rotation -- buys anything further.

The methods compared, all per (layer, KV head) on real cached keys:

  none          normalize -> WHT -> Lloyd-Max                     (0 side info)
  center        subtract per-head key mean, then as above         (d floats/head)
  center+white  also divide by per-channel std before rotating    (2d floats/head)
  center+std    also standardize each ROTATED coordinate          (2d floats/head)
  pca-k         project out the top-k principal directions        (k floats/TOKEN)

The first four cost side information proportional to the head dimension, i.e.
amortized to nothing over a long sequence. pca-k costs per token and is included
as a reference point for what a non-free method buys.

Quality is scored with worst-case attention-logit correlation, the proxy that
metric_analysis.py validates against true perplexity.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from turboquantdc.codebook import LloydMaxCodebook
from turboquantdc.rotation import apply_wht_rotation, generate_wht_rotation

SEED = 42
EPS = 1e-8


def quantize_roundtrip(X, codebook, wht, standardize_rotated=False):
    """X: (T, D) -> reconstruction. normalize, rotate, Lloyd-Max, invert."""
    n = X.norm(dim=-1, keepdim=True)
    U = X / (n + EPS)
    R = apply_wht_rotation(U, wht)
    if standardize_rotated:
        mu, sd = R.mean(0, keepdim=True), R.std(0, keepdim=True).clamp_min(EPS)
        Rn = (R - mu) / sd
        idx = codebook.quantize(Rn)
        Rq = codebook.dequantize(idx) * sd + mu
    else:
        idx = codebook.quantize(R)
        Rq = codebook.dequantize(idx)
    Uq = apply_wht_rotation(Rq, wht, inverse=True)
    # norm correction: the repo rescales to preserve the stored norm
    Uq = Uq / (Uq.norm(dim=-1, keepdim=True) + EPS)
    return Uq * n


def logit_score(K, Kq, probes):
    """Worst-probe-averaged correlation of mean-removed logits (softmax-relevant)."""
    L, Lq = K @ probes.T, Kq @ probes.T
    Lc = L - L.mean(0, keepdim=True)
    Lqc = Lq - Lq.mean(0, keepdim=True)
    sd, sdq = Lc.std(0), Lqc.std(0)
    r = (Lc * Lqc).mean(0) / (sd * sdq).clamp_min(1e-12)
    return float(r.mean())


def run_head(K, codebook, wht, probes, pca_k=(1, 4)):
    """K: (T, D) real keys for one head. Returns {method: logit_r}."""
    out = {}
    mu = K.mean(0, keepdim=True)

    out["none"] = logit_score(K, quantize_roundtrip(K, codebook, wht), probes)

    Kc = K - mu
    out["center"] = logit_score(K, quantize_roundtrip(Kc, codebook, wht) + mu, probes)

    sd = Kc.std(0, keepdim=True).clamp_min(EPS)
    Kw = Kc / sd
    rec = quantize_roundtrip(Kw, codebook, wht) * sd + mu
    out["center+white"] = logit_score(K, rec, probes)

    rec = quantize_roundtrip(Kc, codebook, wht, standardize_rotated=True) + mu
    out["center+std"] = logit_score(K, rec, probes)

    # PCA-k: project out the top-k directions of the centered keys and store the
    # k coefficients per token exactly. Costs k floats/token -- not free.
    try:
        U, S, Vh = torch.linalg.svd(Kc, full_matrices=False)
        for k in pca_k:
            V = Vh[:k]                       # (k, D)
            coef = Kc @ V.T                  # (T, k) kept at full precision
            resid = Kc - coef @ V
            rec = quantize_roundtrip(resid, codebook, wht) + coef @ V + mu
            out[f"pca-{k}"] = logit_score(K, rec, probes)
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True, help=".npz of cached keys (L<i> -> [1,H,T,D])")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--name", default=None)
    ap.add_argument("--probes", type=int, default=256)
    a = ap.parse_args()
    name = a.name or Path(a.keys).stem

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(a.keys)
    layers = sorted(z.files, key=lambda s: int(s[1:]))
    D = z[layers[0]].shape[-1]

    codebook = LloydMaxCodebook(d=D, bits=a.bits)
    if hasattr(codebook, "to"):
        codebook = codebook.to(dev)
    wht = generate_wht_rotation(D, seed=SEED, device=dev)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    P = torch.randn(a.probes, D, generator=g)
    P = (P / P.norm(dim=-1, keepdim=True)).to(dev)

    per_layer, acc = [], {}
    for nm in layers:
        K4 = torch.from_numpy(z[nm]).to(dev).float()
        H = K4.shape[1]
        heads = [run_head(K4[0, h], codebook, wht, P) for h in range(H)]
        row = {"layer": int(nm[1:])}
        for k in heads[0]:
            row[k] = float(np.mean([h[k] for h in heads]))
            acc.setdefault(k, []).append(row[k])
        per_layer.append(row)
        print("  L{:<3d} ".format(row["layer"])
              + "  ".join(f"{k}={row[k]:.4f}" for k in heads[0]), flush=True)

    print(f"\n=== {name} @ {a.bits}-bit: mean over layers (worst layer in parens) ===")
    order = sorted(acc, key=lambda k: -np.mean(acc[k]))
    for k in order:
        print(f"  {k:14s} mean={np.mean(acc[k]):.4f}   worst={np.min(acc[k]):.4f}")

    out = HERE / "results" / f"precondition_{name}_{a.bits}bit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"name": name, "bits": a.bits, "head_dim": int(D),
               "mean": {k: float(np.mean(v)) for k, v in acc.items()},
               "worst": {k: float(np.min(v)) for k, v in acc.items()},
               "per_layer": per_layer}, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
