"""Causal within-model ablation: is Qwen2.5's shared key component caused by the k_proj bias?

Loads Qwen2.5-7B once from the local ollama GGUF blob, then measures
rho_h = ||mu_h|| / E_t||k_t - mu_h|| twice:

  arm A: model as shipped
  arm B: identical model with every k_proj.bias zeroed

Qwen2.5 is the only family in the local model set that HAS a k_proj bias
(Qwen3, Qwen3.5, Gemma3, Gemma4 all removed it in favour of QK-norm). If the
shared key component collapses when the bias is zeroed, the bias is its cause.

This is a mechanistic attribution, not a quality benchmark: arm B is not a
model anyone would deploy.
"""
from __future__ import annotations
import os, sys, json, gc, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)
from measure_key_mean_gguf import find_models, build_corpus, extract_layer_keys, head_stats

TAG = "qwen2.5:7b"
TOKENS = 1024


def summarize(pkv, label):
    per_layer = []
    for i, keys in enumerate(extract_layer_keys(pkv)):
        if keys is None or keys.numel() == 0:
            continue
        hs = head_stats(keys)
        per_layer.append(dict(layer=i,
                              rho_mean=float(np.mean([h["rho"] for h in hs])),
                              rho_max=float(max(h["rho"] for h in hs)),
                              cos_mean=float(np.mean([h["cos_to_mean"] for h in hs])),
                              heads=hs))
    allr = [h["rho"] for L in per_layer for h in L["heads"]]
    allc = [h["cos_to_mean"] for L in per_layer for h in L["heads"]]
    allf = [h["mean_energy_frac"] for L in per_layer for h in L["heads"]]
    s = sorted(allr)
    res = dict(arm=label, n_layers=len(per_layer),
               rho_mean=float(np.mean(allr)), rho_median=float(s[len(s) // 2]),
               rho_p90=float(s[int(0.9 * (len(s) - 1))]), rho_max=float(max(allr)),
               cos_to_mean_mean=float(np.mean(allc)),
               mean_energy_frac_mean=float(np.mean(allf)),
               frac_heads_rho_gt_1=float(np.mean([r > 1 for r in allr])),
               frac_heads_rho_gt_3=float(np.mean([r > 3 for r in allr])),
               per_layer=per_layer)
    print(f"[{label}] rho mean={res['rho_mean']:.3f} median={res['rho_median']:.3f} "
          f"p90={res['rho_p90']:.3f} max={res['rho_max']:.3f} | "
          f"cos_to_mean={res['cos_to_mean_mean']:.4f} | "
          f"energy_frac={res['mean_energy_frac_mean']:.4f} | "
          f"rho>1: {100*res['frac_heads_rho_gt_1']:.1f}%  "
          f"rho>3: {100*res['frac_heads_rho_gt_3']:.1f}%", flush=True)
    return res


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    blob = find_models()[TAG]
    d = os.path.join(SCRATCH, "gguf", TAG.replace(":", "_"))
    os.makedirs(d, exist_ok=True)
    link = os.path.join(d, "model.gguf")
    if not os.path.exists(link):
        os.symlink(blob, link)

    print("loading...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(d, gguf_file="model.gguf")
    model = AutoModelForCausalLM.from_pretrained(
        d, gguf_file="model.gguf", dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    ids = tok(build_corpus(), return_tensors="pt",
              truncation=True, max_length=TOKENS)["input_ids"]
    print(f"tokens: {ids.shape[1]}", flush=True)

    layers = model.model.layers
    biases = []
    for L in layers:
        b = L.self_attn.k_proj.bias
        biases.append(None if b is None else b.detach().clone())
    present = sum(b is not None for b in biases)
    bnorms = [float(b.float().norm()) for b in biases if b is not None]
    print(f"k_proj.bias present on {present}/{len(layers)} layers; "
          f"||b|| mean={np.mean(bnorms):.2f} max={max(bnorms):.2f}", flush=True)

    out = {}

    # ---- arm A: as shipped ----
    t0 = time.time()
    with torch.no_grad():
        o = model(ids, use_cache=True)
    print(f"arm A forward {time.time()-t0:.0f}s", flush=True)
    out["A_as_shipped"] = summarize(o.past_key_values, "A: as shipped")
    del o; gc.collect()

    # ---- arm B: k_proj bias zeroed ----
    with torch.no_grad():
        for L in layers:
            if L.self_attn.k_proj.bias is not None:
                L.self_attn.k_proj.bias.zero_()
    t0 = time.time()
    with torch.no_grad():
        o = model(ids, use_cache=True)
    print(f"arm B forward {time.time()-t0:.0f}s", flush=True)
    out["B_k_bias_zeroed"] = summarize(o.past_key_values, "B: k_proj.bias = 0")
    del o; gc.collect()

    out["meta"] = dict(model=TAG, tokens=int(ids.shape[1]),
                       layers_with_k_bias=present,
                       k_bias_norm_mean=float(np.mean(bnorms)),
                       k_bias_norm_max=float(max(bnorms)))
    p = os.path.join(SCRATCH, "bias_ablation_results.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p, flush=True)

    a, b = out["A_as_shipped"], out["B_k_bias_zeroed"]
    print("\n=== CAUSAL ATTRIBUTION ===", flush=True)
    print(f"rho mean      {a['rho_mean']:8.3f} -> {b['rho_mean']:8.3f}", flush=True)
    print(f"cos_to_mean   {a['cos_to_mean_mean']:8.4f} -> {b['cos_to_mean_mean']:8.4f}", flush=True)
    print(f"heads rho>1   {100*a['frac_heads_rho_gt_1']:7.1f}% -> {100*b['frac_heads_rho_gt_1']:7.1f}%", flush=True)


if __name__ == "__main__":
    main()
