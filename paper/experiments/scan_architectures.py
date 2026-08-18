"""Structural scan: does each model carry a shared per-head key component by construction?"""
import json, os, glob, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reader import GGUF


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
                    name = rel.replace(os.sep, ":")
                    out[name] = f"{base}/blobs/{lay['digest'].replace(':', '-')}"
    return out


def scan(name, path):
    g = GGUF(path)
    arch = g.arch()
    n_head = g.a("attention.head_count")
    n_kv = g.a("attention.head_count_kv")
    n_layer = g.a("block_count")
    emb = g.a("embedding_length")
    key_len = g.a("attention.key_length")
    if isinstance(n_head, list):
        n_head = n_head[0]

    # Language-model tensors only. Multimodal GGUFs prefix the vision tower
    # with "v." / "mm."; ViTs conventionally carry a qkv bias, and counting
    # it here would be a false positive for the language model.
    names = {n for n in g.tensors if n.startswith("blk.")}
    has_kbias = any(n.endswith("attn_k.bias") for n in names)
    has_knorm = any("attn_k_norm" in n for n in names)
    has_qnorm = any("attn_q_norm" in n for n in names)
    attn_layers = sorted(int(n.split(".")[1]) for n in names if n.endswith("attn_k.weight"))

    # infer head_dim from the k projection output width
    kdim = None
    for n in names:
        if n == "blk.0.attn_k.weight":
            kdim = g.tensors[n][0][-1]
    head_dim = key_len or (kdim // n_kv if (kdim and n_kv) else None)

    if isinstance(n_kv, list):  # hybrid stacks list per-layer KV heads; 0 = not attention
        nz = [v for v in n_kv if v]
        n_kv = nz[0] if nz else None
    rec = dict(model=name, arch=arch, n_layer=n_layer,
               n_attn_layers=len(attn_layers), n_head=n_head, n_kv_head=n_kv,
               emb=emb, head_dim=head_dim, k_bias=has_kbias,
               k_norm=has_knorm, q_norm=has_qnorm)

    # If a k bias exists, measure its per-head magnitude.
    if has_kbias and head_dim:
        per_layer = []
        for L in attn_layers:
            t = f"blk.{L}.attn_k.bias"
            if t not in g.tensors:
                continue
            b = g.read_tensor(t).reshape(-1)          # (n_kv*head_dim,)
            bh = b.reshape(n_kv, head_dim)
            norms = np.linalg.norm(bh, axis=1)        # ||b_h|| per KV head
            per_layer.append(dict(layer=L,
                                  bias_norm_mean=float(norms.mean()),
                                  bias_norm_max=float(norms.max()),
                                  bias_rms=float(np.sqrt((bh ** 2).mean()))))
        rec["k_bias_stats"] = per_layer
        allm = [p["bias_norm_mean"] for p in per_layer]
        rec["k_bias_norm_mean_over_layers"] = float(np.mean(allm))
        rec["k_bias_norm_max_over_layers"] = float(max(p["bias_norm_max"] for p in per_layer))
    g.close()
    return rec


if __name__ == "__main__":
    models = find_models()
    results = []
    for name in sorted(models):
        if "nomic" in name:
            continue
        try:
            r = scan(name, models[name])
        except Exception as e:
            r = dict(model=name, error=f"{type(e).__name__}: {e}")
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "k_bias_stats"}))
    json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "scan_results.json"), "w"), indent=1)
