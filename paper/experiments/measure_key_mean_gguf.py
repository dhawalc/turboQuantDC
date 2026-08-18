"""Measure rho_h = ||mu_h|| / E_t||k_t - mu_h|| on real activations, from ollama GGUF blobs.

No downloads: models are loaded from the local ollama blob store via the
transformers GGUF loader (supports qwen2, qwen3, gemma3).
"""
from __future__ import annotations
import os, sys, json, gc, glob, time, argparse

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

SCRATCH = os.path.dirname(os.path.abspath(__file__))
REPO = str(__import__("pathlib").Path(__file__).resolve().parents[2])


def find_models():
    out = {}
    for base in ("/usr/share/ollama/.ollama/models",
                 os.path.expanduser("~/.ollama/models")):
        for mf in glob.glob(base + "/manifests/**/*", recursive=True):
            if not os.path.isfile(mf):
                continue
            try:
                d = json.load(open(mf))
            except Exception:
                continue
            for lay in d.get("layers", []):
                if "model" in lay.get("mediaType", ""):
                    rel = os.path.relpath(mf, base + "/manifests/registry.ollama.ai/library")
                    out[rel.replace(os.sep, ":")] = f"{base}/blobs/{lay['digest'].replace(':','-')}"
    return out


def build_corpus() -> str:
    """Varied English from files committed in this repo.

    wikitext-2 is not cached locally and we are running download-free, so this
    substitutes committed prose: a real wikitext-2 excerpt that already lives in
    benchmarks/mean_removal_benchmark.py, plus repository documentation. This is
    a documented deviation from the paper's corpus.
    """
    parts = []
    mrb = os.path.join(REPO, "benchmarks", "mean_removal_benchmark.py")
    src = open(mrb).read()
    if "WIKITEXT_EXCERPT" in src:
        seg = src.split("WIKITEXT_EXCERPT = (", 1)[1].split(")", 1)[0]
        parts.append("".join(
            ln.strip().strip('"') for ln in seg.splitlines() if ln.strip()
        ))
    for rel in ("README.md", "PLAN.md", "paper/qwen_kv_quantization_failure.md",
                "GROWTH_PLAYBOOK.md"):
        p = os.path.join(REPO, rel)
        if os.path.exists(p):
            parts.append(open(p, encoding="utf-8", errors="ignore").read())
    return "\n\n".join(parts)


def extract_layer_keys(pkv):
    layers = getattr(pkv, "layers", None)
    if layers is not None:
        ks = [getattr(l, "keys", None) for l in layers]
        if all(k is not None for k in ks):
            return ks
    kc = getattr(pkv, "key_cache", None)
    if kc:
        return list(kc)
    return [e[0] for e in pkv]


def head_stats(keys: torch.Tensor):
    k = keys.float()[0]                                  # (H, T, D)
    mu = k.mean(dim=1, keepdim=True)
    mean_norm = mu.squeeze(1).norm(dim=-1)
    dev_norm = (k - mu).norm(dim=-1).mean(dim=1)
    key_sq = (k.norm(dim=-1) ** 2).mean(dim=1)
    cos = torch.nn.functional.cosine_similarity(k, mu.expand_as(k), dim=-1).mean(dim=1)
    rho = mean_norm / dev_norm.clamp_min(1e-8)
    frac = (mean_norm ** 2) / key_sq.clamp_min(1e-8)
    return [dict(head=h, rho=float(rho[h]), mean_norm=float(mean_norm[h]),
                 dev_norm=float(dev_norm[h]), mean_energy_frac=float(frac[h]),
                 cos_to_mean=float(cos[h])) for h in range(k.shape[0])]


def run(tag: str, blob: str, tokens: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    d = os.path.join(SCRATCH, "gguf", tag.replace(":", "_"))
    os.makedirs(d, exist_ok=True)
    link = os.path.join(d, "model.gguf")
    if not os.path.exists(link):
        os.symlink(blob, link)

    print(f"[{tag}] loading from GGUF (cpu, bfloat16)...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(d, gguf_file="model.gguf")
    model = AutoModelForCausalLM.from_pretrained(
        d, gguf_file="model.gguf", dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    print(f"[{tag}] loaded in {time.time()-t0:.0f}s", flush=True)

    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)
    ids = tok(build_corpus(), return_tensors="pt", truncation=True, max_length=tokens)["input_ids"]
    print(f"[{tag}] forward on {ids.shape[1]} tokens...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model(ids, use_cache=True)
    print(f"[{tag}] forward done in {time.time()-t0:.0f}s", flush=True)

    per_layer = []
    for i, keys in enumerate(extract_layer_keys(out.past_key_values)):
        if keys is None or keys.numel() == 0:
            continue
        hs = head_stats(keys)
        per_layer.append(dict(layer=i, kv_heads=keys.shape[1],
                              rho_mean=float(np.mean([h["rho"] for h in hs])),
                              rho_max=float(max(h["rho"] for h in hs)),
                              cos_mean=float(np.mean([h["cos_to_mean"] for h in hs])),
                              heads=hs))
    allr = [h["rho"] for L in per_layer for h in L["heads"]]
    allc = [h["cos_to_mean"] for L in per_layer for h in L["heads"]]
    allf = [h["mean_energy_frac"] for L in per_layer for h in L["heads"]]
    s = sorted(allr)
    res = dict(model=tag, arch=getattr(tcfg, "model_type", "?"),
               n_layers=len(per_layer),
               kv_heads=int(getattr(tcfg, "num_key_value_heads", -1)),
               eval_tokens=int(ids.shape[1]),
               rho_mean=float(np.mean(allr)), rho_median=float(s[len(s)//2]),
               rho_p90=float(s[int(0.9*(len(s)-1))]), rho_max=float(max(allr)),
               rho_min=float(min(allr)),
               cos_to_mean_mean=float(np.mean(allc)),
               mean_energy_frac_mean=float(np.mean(allf)),
               frac_heads_rho_gt_1=float(np.mean([r > 1 for r in allr])),
               frac_heads_rho_gt_3=float(np.mean([r > 3 for r in allr])),
               per_layer=per_layer)
    print(f"[{tag}] rho mean={res['rho_mean']:.3f} median={res['rho_median']:.3f} "
          f"p90={res['rho_p90']:.3f} max={res['rho_max']:.3f} | "
          f"cos_to_mean={res['cos_to_mean_mean']:.4f} | "
          f"heads with rho>1: {100*res['frac_heads_rho_gt_1']:.1f}%", flush=True)

    del model, out
    gc.collect()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--out", default=os.path.join(SCRATCH, "rho_results.json"))
    a = ap.parse_args()

    found = find_models()
    acc = {}
    if os.path.exists(a.out):
        acc = json.load(open(a.out))
    for m in a.models:
        try:
            acc[m] = run(m, found[m], a.tokens)
        except Exception as e:
            import traceback; traceback.print_exc()
            acc[m] = dict(model=m, error=f"{type(e).__name__}: {e}")
        json.dump(acc, open(a.out, "w"), indent=1)
    print("wrote", a.out)
