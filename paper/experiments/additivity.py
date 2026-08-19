#!/usr/bin/env python3
"""Is per-layer KV-cache damage additive, or dominated by the worst layer?

This matters outside this repository. Bit allocators in shipping systems assume
per-layer damage is ADDITIVE -- they sum a per-layer error term and spend bits to
minimise that sum (NVIDIA ModelOpt's width-weighted recipe; HIGGS's dynamic-
programming allocator over layer-wise L2 error). SS6.15 of this project's
manuscript instead found that the WORST layer carries the signal, i.e. a max.

Those are the two endpoints of one family. For a per-layer score-space noise
vector n = (n_1 ... n_L), with n_l = 1 - (attention-logit correlation at layer l):

    A_p(n) = (sum_l n_l^p)^(1/p)

p = 1 is exactly the additive assumption; p -> infinity is exactly the worst-layer
statistic; intermediate p interpolates. So "additive or max?" becomes "what is p?",
which the data can answer directly.

Method. For each aggregator we rebuild the whole SS6.22 pipeline end to end: the
model's noise-response curve is re-fitted with that aggregator on the Gaussian-
noise cells, and the quantizer cells are then predicted through it. Nothing else
changes, so any difference in accuracy is attributable to the aggregator alone.
Curves are calibrated on noise and tested on quantizer error throughout, so the
comparison stays honest.

Usage:  python additivity.py
"""
from __future__ import annotations
import json, math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
VARIANTS = ("-round", "-sign", "-seed43")


def per_layer_noise(row):
    """Per-layer score-space noise vector for one cell, ordered by layer."""
    pl = row.get("per_layer_logit_r") or {}
    if not pl:
        return None
    return np.array([1.0 - pl[k] for k in sorted(pl, key=lambda x: int(x))])


def aggregate(n, kind):
    if n is None or not len(n):
        return None
    if kind == "max":
        return float(n.max())
    if kind == "sum":
        return float(n.sum())
    if kind == "mean":
        return float(n.mean())
    if kind.startswith("top"):
        k = int(kind[3:])
        return float(np.sort(n)[-k:].sum())
    if kind.startswith("L"):
        p = float(kind[1:])
        return float((n ** p).sum() ** (1.0 / p))
    raise ValueError(kind)


def load(name):
    """(noise cells, quantizer cells) for one model, or (None, None)."""
    sp = RESULTS / f"sensitivity_{name}.json"
    if not sp.exists():
        return None, None
    noise = [r for r in json.loads(sp.read_text())["rows"] if r.get("sigma", 0) > 0]
    quant = []
    for suffix in ("", "-k1"):
        p = RESULTS / f"ppl_{name}{suffix}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        base = next((r["ppl"] for r in d["rows"] if r["config"] == "fp16-KV"), None)
        if not base:
            continue
        for r in d["rows"]:
            if r.get("bits") and np.isfinite(r["ppl"]) and r.get("per_layer_logit_r"):
                quant.append(dict(row=r, actual=math.log10(r["ppl"] / base),
                                  bits=r["bits"], center=r["center"]))
    return noise, quant


def evaluate(kind, names):
    """Predict every quantizer cell through curves built with this aggregator."""
    out = []
    for name in names:
        noise, quant = load(name)
        if not noise or not quant:
            continue
        pts = sorted([(aggregate(per_layer_noise(r), kind), math.log10(r["ratio"]))
                      for r in noise if per_layer_noise(r) is not None] + [(0.0, 0.0)])
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        if len(xs) < 3:
            continue
        for c in quant:
            x = aggregate(per_layer_noise(c["row"]), kind)
            if x is None:
                continue
            if x <= xs[-1]:
                pred = float(np.interp(x, xs, ys))
            else:                                    # linear extrapolation
                pred = ys[-1] + (ys[-1] - ys[-2]) * (x - xs[-1]) / max(xs[-1] - xs[-2], 1e-9)
            out.append(dict(model=name, bits=c["bits"], center=c["center"],
                            pred=pred, actual=c["actual"]))
    return out


def score(cells):
    a = np.array([c["actual"] for c in cells]); p = np.array([c["pred"] for c in cells])
    resid = np.abs(a - p)
    r2 = 1 - float(((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum())
    big = [c for c in cells if 10 ** c["actual"] > 1.5]
    r2b = float("nan")
    if len(big) > 2:
        ab = np.array([c["actual"] for c in big]); pb = np.array([c["pred"] for c in big])
        r2b = 1 - float(((ab - pb) ** 2).sum() / ((ab - ab.mean()) ** 2).sum())
    miss = sum(1 for c in cells if 10 ** c["pred"] <= 5 < 10 ** c["actual"])
    return r2, r2b, float(np.median(resid)), float(np.percentile(resid, 90)), miss


def main():
    names = sorted(n for n in (p.stem.replace("sensitivity_", "")
                               for p in RESULTS.glob("sensitivity_*.json"))
                   if not any(n.endswith(v) for v in VARIANTS))

    kinds = ["sum", "mean", "L1.5", "L2", "L3", "L4", "L6", "L8", "L12", "L20",
             "top2", "top4", "max"]
    print("Does per-layer damage add up, or is it set by the worst layer?")
    print("p=1 (sum) is the additive assumption shipping allocators make;")
    print("max is this project's worst-layer statistic. Curves are rebuilt per")
    print("aggregator, calibrated on Gaussian noise, tested on quantizer error.\n")
    print(f"{'aggregator':>10s} {'cells':>6s} {'R2 (all)':>9s} {'R2 (>x1.5)':>11s} "
          f"{'median':>8s} {'90th':>7s} {'x5 misses':>10s}")
    print("-" * 68)
    rows = []
    for k in kinds:
        cells = evaluate(k, names)
        if len(cells) < 20:
            continue
        r2, r2b, med, p90, miss = score(cells)
        rows.append(dict(aggregator=k, n=len(cells), r2=r2, r2_damaged=r2b,
                         median=med, p90=p90, gate_misses=miss))
        print(f"{k:>10s} {len(cells):>6d} {r2:>9.4f} {r2b:>11.4f} "
              f"{med:>8.4f} {p90:>7.4f} {miss:>10d}")

    best = max(rows, key=lambda r: r["r2"])
    add = next((r for r in rows if r["aggregator"] == "sum"), None)
    mx = next((r for r in rows if r["aggregator"] == "max"), None)
    print(f"\nbest aggregator by R2: {best['aggregator']} (R2 {best['r2']:.4f})")
    if add and mx:
        print(f"additive (sum): R2 {add['r2']:.4f}, 90th pct error x{10**add['p90']:.2f}, "
              f"{add['gate_misses']} misses at a x5 gate")
        print(f"worst-layer (max): R2 {mx['r2']:.4f}, 90th pct error x{10**mx['p90']:.2f}, "
              f"{mx['gate_misses']} misses at a x5 gate")
        verdict = ("MAX-DOMINATED: the additive assumption is materially worse"
                   if mx["r2"] - add["r2"] > 0.02 else
                   "ADDITIVE IS FINE: no material penalty for summing"
                   if add["r2"] - mx["r2"] > 0.02 else
                   "INDISTINGUISHABLE on this data")
        print(f"verdict: {verdict}")

    json.dump(rows, open(RESULTS / "additivity.json", "w"), indent=1)
    print(f"\nwrote {RESULTS/'additivity.json'}")


if __name__ == "__main__":
    main()
