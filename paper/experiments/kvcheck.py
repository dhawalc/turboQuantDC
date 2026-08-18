#!/usr/bin/env python3
"""kvcheck: will low-bit KV-cache quantization break this model?

One forward pass over a short calibration text, with the repository's production
quantizer patched into the cache, is enough to compute the statistic that the
116-cell atlas in `paper/qwen_kv_quantization_failure.md` validates as the best
cheap predictor of real perplexity damage: the WORST-LAYER correlation between
the attention logits produced by original and reconstructed keys.

No labels, no perplexity run, no downstream evaluation. Typical cost: seconds.

Usage:
    python kvcheck.py --model Qwen/Qwen2.5-7B-Instruct --load-4bit
    python kvcheck.py --model meta-llama/Llama-3.2-1B --bits 2 --center

Interpretation (calibrated on 116 (model, bits, centering) cells, 23 models,
7 families — see §6.15 and §6.19 of the paper):

  worst-layer logit r < 0.83   every catastrophic failure (PPL ratio > 4x) in
                               the atlas scored below this; expect severe damage
  0.83 <= r < 0.95             caution: no catastrophic cell scored here, but
                               moderate uniform damage (PPL x2-x5, e.g. 1-bit
                               keys) can hide in this band
  r >= 0.95                    every atlas cell here had PPL ratio <= 1.4

SCOPE — read this before trusting a PASS: the statistic detects concentrated
score-space collapse (the Qwen2.5-type mean-dominance pathology) with a clean
margin, but §6.19 shows that mild damage spread uniformly across layers can pass
every cheap proxy we tested, including this one and including per-vector cosine
similarity. A PASS here is evidence, not proof; a FAIL is close to proof.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from ppl_harness import KVCompressor, patched_cache, load_model, SCRATCH

# Fallback calibration text if the wikitext snapshot is absent. Any natural
# prose works: the statistic needs realistic key distributions, not labels.
FALLBACK = (
    "The history of measurement is the history of agreeing on shared references. "
    "A metre bar in Paris, a caesium transition, a fixed speed of light: each "
    "replaced a local convention with a portable one. Instruments drift, so "
    "calibration is not an event but a discipline; the question is never whether "
    "a device errs but whether its error is known, bounded, and reported. "
) * 200

CATASTROPHIC = 0.83   # every atlas cell below this had PPL ratio > 4x
SAFE = 0.95           # every atlas cell at or above this had PPL ratio <= 1.4x


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=3, help="key bits (default 3)")
    ap.add_argument("--center", action="store_true",
                    help="subtract the running per-head key mean before quantizing")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--json", default=None, help="also write a JSON report here")
    a = ap.parse_args()

    torch.manual_seed(42)
    wt = SCRATCH / "wikitext2_test.txt"
    text = wt.read_text() if wt.exists() else FALLBACK

    t0 = time.time()
    model, tok = load_model(a.model, a.load_4bit)
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=a.tokens)["input_ids"].to(model.device)
    print(f"loaded {a.model} in {time.time()-t0:.0f}s; "
          f"calibrating on {ids.shape[1]} tokens", flush=True)

    comp = KVCompressor(key_bits=a.bits, center=a.center)
    t0 = time.time()
    with patched_cache(comp), torch.no_grad():
        model(ids, use_cache=True)
    s = comp.summary()
    dt = time.time() - t0

    r_min = s["logit_r_min"]
    worst = min(s["per_layer_logit_r"], key=s["per_layer_logit_r"].get)
    if r_min < CATASTROPHIC:
        verdict, note = "FAIL", "expect severe end-to-end damage at this setting"
    elif r_min < SAFE:
        verdict, note = "CAUTION", ("no catastrophic atlas cell scored here, but "
                                    "moderate damage can hide in this band")
    else:
        verdict, note = "PASS", ("consistent with <= x1.4 perplexity in the "
                                 "atlas; see SCOPE note, a PASS is not proof")

    print(f"\nconfig: {a.bits}-bit keys, centering={'ON' if a.center else 'OFF'}")
    print(f"worst-layer logit correlation: {r_min:.4f} (layer {worst})")
    print(f"mean logit correlation:        {s['logit_r']:.4f}")
    print(f"per-vector cosine (for scale): {s['vec_cos']:.4f}  "
          f"<- do not gate on this; see paper §6.15")
    print(f"verdict: {verdict} — {note}")
    print(f"({dt:.1f}s quantizer pass)")
    if not a.center and r_min < SAFE:
        print("hint: rerun with --center; on mean-dominated models this is "
              "worth ~4 bits (paper §6.16)")

    if a.json:
        json.dump(dict(model=a.model, bits=a.bits, center=a.center,
                       tokens=int(ids.shape[1]), verdict=verdict,
                       logit_r_min=r_min, logit_r_mean=s["logit_r"],
                       vec_cos=s["vec_cos"], worst_layer=int(worst),
                       per_layer_logit_r=s["per_layer_logit_r"]),
                  open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
