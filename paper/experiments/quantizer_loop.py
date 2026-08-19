"""Close the loop: do the measured shared key components actually break the quantizer?

Step 1: extract real post-RoPE keys from Qwen2.5-7B (local ollama GGUF, no download).
Step 2: run the repository's OWN production quantizer (_CompressedLayer, 3-bit,
        WHT rotation + ResidualQuant) over those keys with centering on and off.
Step 3: measure how much of the attention-logit structure survives.

Logit-structure metric (query-agnostic): softmax over keys responds only to the
spread of u.k_i across positions i, for whatever query direction u the model
produces. For random unit directions u we compare the true logit vector
{u.k_i}_i against the quantized one {u.k_hat_i}_i and report

  * pearson r  between the two logit vectors (does the ordering survive?)
  * spread ratio std_i(u.k_hat_i) / std_i(u.k_i)  (is the discriminative
    signal preserved or crushed?)

Both are computed on CENTERED logits, because a constant offset is exactly what
softmax ignores. Averaged over many random u and over all layers/heads.
"""
from __future__ import annotations
import os, sys, json, gc, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

SCRATCH = os.path.dirname(os.path.abspath(__file__))
REPO = str(__import__("pathlib").Path(__file__).resolve().parents[2])
sys.path.insert(0, SCRATCH)
sys.path.insert(0, REPO)
from measure_key_mean_gguf import find_models, build_corpus, extract_layer_keys

TAG = os.environ.get("TQ_MODEL", "qwen2.5:7b")
TOKENS = int(os.environ.get("TQ_TOKENS", "1024"))
KEYS_NPZ = os.path.join(os.environ.get("TQ_SCRATCH", SCRATCH), f"keys_{os.path.basename(TAG.rstrip(chr(47))).replace(':','_')}.npz")
N_PROBES = 256
SEED = 42


def extract_keys():
    if os.path.exists(KEYS_NPZ):
        print("reusing cached keys:", KEYS_NPZ, flush=True)
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import transformers as _tf
    kw = {}
    if os.path.isdir(TAG):
        # local HuggingFace checkpoint directory (safetensors)
        d = TAG
    else:
        # ollama GGUF blob
        blob = find_models()[TAG]
        d = os.path.join(SCRATCH, "gguf", TAG.replace(":", "_"))
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, "model.gguf")
        if not os.path.exists(link):
            os.symlink(blob, link)
        kw["gguf_file"] = "model.gguf"
    dev_map = os.environ.get("TQ_DEVICE", "cpu")
    print(f"loading model from {d} onto {dev_map}...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(d, **kw)
    model = None
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        cls = getattr(_tf, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(d, dtype=torch.bfloat16,
                                        device_map=dev_map, **kw)
            print(f"  loaded via {cls_name}", flush=True)
            break
        except Exception as e:
            print(f"  {cls_name} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
    if model is None:
        raise RuntimeError("no auto class could load this checkpoint")
    model.eval()
    print(f"loaded {time.time()-t0:.0f}s", flush=True)
    ids = tok(build_corpus(), return_tensors="pt",
              truncation=True, max_length=TOKENS)["input_ids"]
    ids = ids.to(next(model.parameters()).device)
    t0 = time.time()
    with torch.no_grad():
        out = model(ids, use_cache=True)
    print(f"forward {time.time()-t0:.0f}s on {ids.shape[1]} tokens", flush=True)
    ks = {f"L{i}": k.detach().float().cpu().numpy().astype(np.float32)
          for i, k in enumerate(extract_layer_keys(out.past_key_values))
          if k is not None and k.numel()}
    np.savez_compressed(KEYS_NPZ, **ks)
    print(f"saved {len(ks)} layers -> {KEYS_NPZ}", flush=True)
    del model, out
    gc.collect()


def logit_metrics(K: torch.Tensor, Kq: torch.Tensor, probes: torch.Tensor):
    """K, Kq: (T, D) real and reconstructed keys. probes: (P, D) unit directions."""
    L = K @ probes.T          # (T, P) true logits per probe
    Lq = Kq @ probes.T        # (T, P) reconstructed
    Lc = L - L.mean(0, keepdim=True)      # softmax ignores a constant shift
    Lqc = Lq - Lq.mean(0, keepdim=True)
    sd, sdq = Lc.std(0), Lqc.std(0)
    num = (Lc * Lqc).mean(0)
    r = num / (sd * sdq).clamp_min(1e-12)
    return float(r.mean()), float((sdq / sd.clamp_min(1e-12)).mean())


def main():
    extract_keys()
    from turboquantdc.generation_layers import _CompressedLayer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(KEYS_NPZ)
    layers = sorted(z.files, key=lambda s: int(s[1:]))
    g = torch.Generator(device="cpu").manual_seed(SEED)

    rows = []
    for name in layers:
        K = torch.from_numpy(z[name]).to(dev)           # (1, H, T, D)
        _, H, T, D = K.shape
        probes = torch.randn(N_PROBES, D, generator=g).to(dev)
        probes = probes / probes.norm(dim=-1, keepdim=True)

        rec = {"layer": int(name[1:])}
        for center in (False, True):
            layer = _CompressedLayer(
                key_bits=3, val_bits=3, fp16_window=0, seed=SEED,
                use_norm_correction=True, use_residual_quant=True,
                center_before_quantize=center,
            )
            with torch.no_grad():
                Kq, _ = layer.update(K.clone(), torch.zeros_like(K))
            Kq = Kq.to(dev).float()
            rs, sr, cs = [], [], []
            for h in range(H):
                r, s = logit_metrics(K[0, h], Kq[0, h], probes)
                rs.append(r); sr.append(s)
                cs.append(float(torch.nn.functional.cosine_similarity(
                    K[0, h], Kq[0, h], dim=-1).mean()))
            tag = "center" if center else "nocenter"
            rec[f"{tag}_logit_r"] = float(np.mean(rs))
            rec[f"{tag}_spread_ratio"] = float(np.mean(sr))
            rec[f"{tag}_vec_cos"] = float(np.mean(cs))
            del layer
        rows.append(rec)
        print(f"L{rec['layer']:>2}  logit_r  nocenter={rec['nocenter_logit_r']:.4f} "
              f"center={rec['center_logit_r']:.4f}   spread "
              f"nocenter={rec['nocenter_spread_ratio']:.3f} "
              f"center={rec['center_spread_ratio']:.3f}   veccos "
              f"nocenter={rec['nocenter_vec_cos']:.4f} "
              f"center={rec['center_vec_cos']:.4f}", flush=True)

    # rho / shared-component stats from the same keys -- one model load, both results
    from measure_key_mean_gguf import head_stats as _hs
    allr, allc, allf = [], [], []
    for name in layers:
        K = torch.from_numpy(z[name])
        for h in _hs(K):
            allr.append(h["rho"]); allc.append(h["cos_to_mean"])
            allf.append(h["mean_energy_frac"])
    ss = sorted(allr)
    rho_stats = dict(rho_mean=float(np.mean(allr)), rho_median=float(ss[len(ss)//2]),
                     rho_p90=float(ss[int(0.9*(len(ss)-1))]), rho_max=float(max(allr)),
                     cos_to_mean_mean=float(np.mean(allc)),
                     mean_energy_frac_mean=float(np.mean(allf)),
                     frac_heads_rho_gt_1=float(np.mean([r > 1 for r in allr])),
                     frac_heads_rho_gt_3=float(np.mean([r > 3 for r in allr])))
    print("\n=== SHARED-COMPONENT (rho) ===")
    for k, v in rho_stats.items():
        print(f"  {k:26s} {v:.4f}")

    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in rows[0] if k != "layer"}
    out = {"model": TAG, "tokens": TOKENS, "key_bits": 3, "n_probes": N_PROBES,
           "device": dev, "aggregate": agg, "rho_stats": rho_stats, "per_layer": rows}
    p = os.path.join(SCRATCH, "results",
                     f"quantizer_loop_{os.path.basename(TAG.rstrip(chr(47))).replace(':','_')}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    print("\n=== AGGREGATE over all layers/heads ===")
    for k, v in agg.items():
        print(f"  {k:26s} {v:.4f}")
    print("wrote", p)


if __name__ == "__main__":
    main()
