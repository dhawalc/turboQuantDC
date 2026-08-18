"""End-to-end perplexity under 3-bit KV compression, with paired proxy metrics.

For each (model, configuration) this measures, in a single pass:

  * wikitext-2 sliding-window perplexity with the repository's production
    quantizer patched into the KV cache;
  * the cheap proxy metrics people actually validate compressors on --
    per-vector cosine similarity of reconstructed keys, and the correlation of
    the attention logits those keys produce.

Producing both from the same forward pass is the point: it yields paired
(proxy, ground-truth-PPL) data, which is what is needed to ask whether any given
proxy predicts real downstream damage.

Usage:
    python ppl_harness.py --model Qwen/Qwen2.5-7B-Instruct --load-4bit
    python ppl_harness.py --model /path/to/local/dir --bits 3 4
"""
from __future__ import annotations
import argparse, gc, json, math, os, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
SCRATCH = Path(os.environ.get(
    "TQ_SCRATCH",
    "/tmp/claude-1000/-home-dhawal-turboQuantDC/0bebcfa0-db7f-400c-8b64-efaab6cc84c0/scratchpad"))

import sys
sys.path.insert(0, str(REPO))

SEED = 42
CONTEXT = 512
STRIDE = 256


# ---------------------------------------------------------------------------
# proxy metrics
# ---------------------------------------------------------------------------

def proxy_metrics(K: torch.Tensor, Kq: torch.Tensor, probes: torch.Tensor):
    """K, Kq: (H, T, D). Returns (vec_cos, logit_r, spread_ratio), head-averaged.

    Logits are compared after removing the per-probe mean, since a constant
    shift across keys is exactly what softmax discards.
    """
    vc = torch.nn.functional.cosine_similarity(K, Kq, dim=-1).mean().item()
    L = torch.einsum("htd,pd->htp", K, probes)
    Lq = torch.einsum("htd,pd->htp", Kq, probes)
    Lc = L - L.mean(1, keepdim=True)
    Lqc = Lq - Lq.mean(1, keepdim=True)
    sd = Lc.std(1); sdq = Lqc.std(1)
    r = (Lc * Lqc).mean(1) / (sd * sdq).clamp_min(1e-12)
    return vc, r.mean().item(), (sdq / sd.clamp_min(1e-12)).mean().item()


# ---------------------------------------------------------------------------
# cache patching
# ---------------------------------------------------------------------------

class KVCompressor:
    """Patches DynamicCache.update so cached keys (and optionally values) pass
    through the repository's production quantizer before being stored."""

    def __init__(self, key_bits, center, val_bits=None, n_probes=128, seed=SEED):
        from turboquantdc.generation_layers import _CompressedLayer
        self._CL = _CompressedLayer
        self.key_bits, self.center, self.val_bits = key_bits, center, val_bits
        self.n_probes, self.seed = n_probes, seed
        self.layers = {}
        self.stats = {}
        self._probes = {}

    def _layer(self, idx):
        if idx not in self.layers:
            self.layers[idx] = self._CL(
                key_bits=self.key_bits, val_bits=self.val_bits or self.key_bits,
                fp16_window=0, seed=self.seed + idx, use_norm_correction=True,
                use_residual_quant=True, center_before_quantize=self.center)
        return self.layers[idx]

    def probes(self, d, device):
        if d not in self._probes:
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            p = torch.randn(self.n_probes, d, generator=g)
            self._probes[d] = (p / p.norm(dim=-1, keepdim=True)).to(device)
        return self._probes[d].to(device)

    def reset(self):
        """New window: drop quantizer state but keep accumulated statistics."""
        self.layers.clear()

    def transform(self, k, v, layer_idx):
        lay = self._layer(layer_idx)
        kq, vq = lay.update(k.clone(), v.clone())
        kq = kq.to(k.dtype)
        with torch.no_grad():
            vc, lr, sr = proxy_metrics(k[0].float(), kq[0].float(),
                                       self.probes(k.shape[-1], k.device))
        s = self.stats.setdefault(layer_idx, {"vec_cos": [], "logit_r": [], "spread": []})
        s["vec_cos"].append(vc); s["logit_r"].append(lr); s["spread"].append(sr)
        if self.val_bits is not None:
            v = vq.to(v.dtype)
        return kq, v

    def summary(self):
        out = {}
        for k in ("vec_cos", "logit_r", "spread"):
            per_layer = [float(np.mean(s[k])) for s in self.stats.values()]
            out[k] = float(np.mean(per_layer))
            out[k + "_min"] = float(np.min(per_layer)) if per_layer else float("nan")
        out["per_layer_logit_r"] = {int(i): float(np.mean(s["logit_r"]))
                                    for i, s in sorted(self.stats.items())}
        return out


def patched_cache(compressor):
    """Context manager patching DynamicCache.update for the duration."""
    from transformers.cache_utils import DynamicCache

    class _Ctx:
        def __enter__(self):
            self.orig = DynamicCache.update

            def upd(cache_self, key_states, value_states, layer_idx, cache_kwargs=None):
                if compressor is not None:
                    key_states, value_states = compressor.transform(
                        key_states, value_states, layer_idx)
                return self.orig(cache_self, key_states, value_states,
                                 layer_idx, cache_kwargs)

            DynamicCache.update = upd
            return self

        def __exit__(self, *a):
            DynamicCache.update = self.orig
    return _Ctx()


# ---------------------------------------------------------------------------
# perplexity
# ---------------------------------------------------------------------------

def sliding_ppl(model, input_ids, compressor=None, context=CONTEXT, stride=STRIDE):
    nlls, ntok = [], 0
    with patched_cache(compressor):
        for begin in range(0, input_ids.shape[1] - 1, stride):
            end = min(begin + context, input_ids.shape[1])
            chunk = input_ids[:, begin:end]
            tgt = chunk.clone()
            if begin > 0:
                tgt[:, :stride] = -100
            tgt[:, 0] = -100
            if compressor is not None:
                compressor.reset()
            with torch.no_grad():
                out = model(chunk, labels=tgt, use_cache=True)
            n = int((tgt != -100).sum())
            if n:
                nlls.append(out.loss.item() * n); ntok += n
            del out
            if end >= input_ids.shape[1]:
                break
    return (math.exp(sum(nlls) / ntok) if ntok else float("inf")), ntok


def load_model(spec, load_4bit, device="cuda"):
    import transformers as tf
    kw = dict(dtype=torch.bfloat16, device_map=device)
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4")
        kw.pop("dtype")
    tok = tf.AutoTokenizer.from_pretrained(spec)
    model = None
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        c = getattr(tf, cls, None)
        if c is None:
            continue
        try:
            model = c.from_pretrained(spec, **kw); break
        except Exception as e:
            print(f"  {cls}: {type(e).__name__}: {str(e)[:100]}", flush=True)
    if model is None:
        raise RuntimeError(f"could not load {spec}")
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--bits", type=int, nargs="+", default=[3])
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--quant-values", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    name = a.name or a.model.rstrip("/").split("/")[-1]

    torch.manual_seed(SEED)
    text = (SCRATCH / "wikitext2_test.txt").read_text()

    print(f"=== {name} ===", flush=True)
    t0 = time.time()
    model, tok = load_model(a.model, a.load_4bit)
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=a.tokens)["input_ids"].to(model.device)
    cfg = getattr(model.config, "text_config", model.config)
    print(f"tokens={ids.shape[1]} layers={cfg.num_hidden_layers} "
          f"kv_heads={getattr(cfg,'num_key_value_heads','?')}", flush=True)

    rows = []
    t0 = time.time()
    base, ntok = sliding_ppl(model, ids)
    print(f"[baseline fp16-KV] ppl={base:.4f} ({ntok} tok, {time.time()-t0:.0f}s)", flush=True)
    rows.append(dict(config="fp16-KV", bits=None, center=None, ppl=base))

    vb = None
    for bits in a.bits:
        for center in (False, True):
            if a.quant_values:
                vb = bits
            comp = KVCompressor(key_bits=bits, center=center, val_bits=vb)
            t0 = time.time()
            ppl, _ = sliding_ppl(model, ids, comp)
            s = comp.summary()
            rows.append(dict(config=f"{bits}bit{'+center' if center else ''}",
                             bits=bits, center=center, ppl=ppl,
                             delta=ppl - base, ratio=ppl / base, **s))
            print(f"[{bits}bit center={center}] ppl={ppl:.4f} "
                  f"(x{ppl/base:.2f}) vec_cos={s['vec_cos']:.4f} "
                  f"logit_r={s['logit_r']:.4f} (min {s['logit_r_min']:.4f}) "
                  f"{time.time()-t0:.0f}s", flush=True)
            del comp
            gc.collect(); torch.cuda.empty_cache()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else RESULTS / f"ppl_{name.replace('/','_')}.json"
    json.dump(dict(model=a.model, name=name, load_4bit=a.load_4bit,
                   tokens=int(ids.shape[1]), context=CONTEXT, stride=STRIDE,
                   quant_values=a.quant_values,
                   num_layers=int(cfg.num_hidden_layers),
                   kv_heads=int(getattr(cfg, "num_key_value_heads", -1)),
                   head_dim=int(getattr(cfg, "head_dim", 0) or 0),
                   rows=rows), open(out, "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
