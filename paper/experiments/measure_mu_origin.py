#!/usr/bin/env python3
"""Where does the shared key component come from? (Limitation 12)

The paper's mechanism needs a large per-head key mean mu_h. SS6.8 refuted the
k_proj bias as its source. The remaining hypothesis, inferred by elimination:

    mu' = E_t[k'_t] = W_k . E_t[z_t] + b_k        (pre-RoPE, exact by linearity)

with E_t[z_t] - the mean normed residual-stream input - dominated by a few
massive-activation channels. This script MEASURES that decomposition:

  1. per layer: ||W_k E[z]|| vs ||b_k|| vs ||mu'|| - which term carries the mean
  2. channel concentration of E[z]: energy share of top-1/4/16 channels
  3. ablation: reconstruct W_k E[z] from only the top-16 channels of E[z];
     report its cosine to the full W_k E[z]
  4. RoPE survival: per-head ||mu_post|| / ||mu'_pre|| from the actual cache

Usage:
    python measure_mu_origin.py --model <path> --name qwen2.5-7b --load-4bit
"""
from __future__ import annotations
import argparse, json, sys
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

    # hooks: per layer capture E_t[z] (k_proj input mean) and E_t[k'] (output
    # mean, pre-RoPE); post-RoPE means come from the cache.
    z_mean, kpre_mean, hooks = {}, {}, []
    layers = model.model.layers if hasattr(model, "model") else model.layers

    def mk(idx):
        def hook(mod, inp, out):
            z = inp[0][0].float()          # (T, hidden)
            z_mean[idx] = z.mean(0).cpu()
            kpre_mean[idx] = out[0].float().mean(0).cpu()
        return hook

    for i, lay in enumerate(layers):
        hooks.append(lay.self_attn.k_proj.register_forward_hook(mk(i)))

    post_mu = {}
    from transformers.cache_utils import DynamicCache
    orig = DynamicCache.update

    def upd(self, k, v, layer_idx, cache_kwargs=None):
        if layer_idx not in post_mu:
            post_mu[layer_idx] = k[0].float().mean(1).cpu()   # (H, D)
        return orig(self, k, v, layer_idx, cache_kwargs)

    DynamicCache.update = upd
    try:
        with torch.no_grad():
            model(ids, use_cache=True)
    finally:
        DynamicCache.update = orig
        for h in hooks:
            h.remove()

    per_layer = []
    for i in sorted(z_mean):
        kp = layers[i].self_attn.k_proj
        dev = next(p.device for p in kp.parameters())
        Ez = z_mean[i].to(dev)
        with torch.no_grad():
            full = kp(Ez.unsqueeze(0).to(model.dtype
                      if hasattr(model, "dtype") else torch.float16))[0].float()
            bias = (kp.bias.detach().float()
                    if kp.bias is not None else torch.zeros_like(full))
            WEz = full - bias                    # W_k E[z], exact
            # top-16 channel ablation of E[z]
            topk = torch.topk(Ez.abs(), k=min(16, Ez.numel())).indices
            Ez_top = torch.zeros_like(Ez); Ez_top[topk] = Ez[topk]
            WEz_top = (kp(Ez_top.unsqueeze(0).to(model.dtype
                       if hasattr(model, "dtype") else torch.float16))[0]
                       .float() - bias)
        e = Ez.float()
        e2 = (e ** 2).sum()
        srt = (e ** 2).sort(descending=True).values
        conc = {k: float(srt[:k].sum() / e2.clamp_min(1e-12)) for k in (1, 4, 16)}
        cos_top = float(torch.nn.functional.cosine_similarity(
            WEz.cpu(), WEz_top.cpu(), dim=0))
        mu_pre = kpre_mean[i]
        # exactness check of the linear decomposition (quantized weights make
        # this approximate; report the discrepancy honestly)
        recon_err = float((mu_pre - full.cpu()).norm() / mu_pre.norm().clamp_min(1e-9))
        H, D = post_mu[i].shape
        mu_pre_heads = mu_pre.reshape(H, D)
        rope_survival = float((post_mu[i].norm(dim=-1) /
                               mu_pre_heads.norm(dim=-1).clamp_min(1e-9)).mean())
        per_layer.append(dict(
            layer=i,
            norm_mu_pre=float(mu_pre.norm()),
            norm_WEz=float(WEz.norm()), norm_bias=float(bias.norm()),
            cos_WEz_mu=float(torch.nn.functional.cosine_similarity(
                WEz.cpu(), mu_pre, dim=0)),
            Ez_top_channel_energy=conc, cos_WEz_top16=cos_top,
            recon_rel_err=recon_err, rope_survival=rope_survival))

    agg = dict(model=a.model, name=a.name, tokens=int(ids.shape[1]),
               per_layer=per_layer,
               summary=dict(
                   mean_cos_WEz_mu=float(np.mean([l["cos_WEz_mu"] for l in per_layer])),
                   mean_top16_energy=float(np.mean(
                       [l["Ez_top_channel_energy"][16] for l in per_layer])),
                   mean_cos_WEz_top16=float(np.mean(
                       [l["cos_WEz_top16"] for l in per_layer])),
                   mean_bias_to_WEz=float(np.mean(
                       [l["norm_bias"] / max(l["norm_WEz"], 1e-9) for l in per_layer])),
                   mean_rope_survival=float(np.mean(
                       [l["rope_survival"] for l in per_layer]))))
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"mu_origin_{a.name}.json"
    json.dump(agg, open(p, "w"), indent=1)
    s = agg["summary"]
    print(f"{a.name}: cos(W_k E[z], mu_pre) = {s['mean_cos_WEz_mu']:.4f}")
    print(f"  top-16 of {len(z_mean[0])} channels hold "
          f"{100*s['mean_top16_energy']:.1f}% of E[z] energy; "
          f"cos(WEz_top16, WEz) = {s['mean_cos_WEz_top16']:.4f}")
    print(f"  ||b_k|| / ||W_k E[z]|| = {s['mean_bias_to_WEz']:.3f}; "
          f"RoPE survival ||mu_post||/||mu_pre|| = {s['mean_rope_survival']:.3f}")
    print("wrote", p)


if __name__ == "__main__":
    main()
