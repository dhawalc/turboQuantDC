#!/usr/bin/env python3
"""Does the damage law depend on the SHAPE of the perturbation, or only its size?

SS6.22 claims that only the scalar amount of score-space noise matters. Its one
systematic residual contradicts that at the extreme: 1-bit centered cells are
under-predicted on every model where they exist, and 1-bit index error is the
least Gaussian error in the study.

This script re-runs the damage-law prediction for those models using curves
calibrated with different noise families — Gaussian, uniform, Rademacher signs,
and true deterministic scalar rounding — and asks which family predicts the
real quantizer cells best, overall and at 1 bit specifically.

If quantization-shaped calibration fixes the 1-bit residual, the claim becomes
"the amount of score-space noise determines damage, provided the calibration
noise resembles the error being calibrated for". If it does not, the shape is
genuinely irrelevant and the 1-bit residual has another cause.

Usage:  python family_test.py
"""
from __future__ import annotations
import json, math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FAMILIES = ["", "-round", "-sign", "-seed43"]
LABEL = {"": "gauss", "-round": "round (real SQ)", "-sign": "sign",
         "-seed43": "gauss seed 43"}


def curve(name, suffix):
    p = RESULTS / f"sensitivity_{name}{suffix}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    pts = sorted([(1 - r["logit_r_min"], math.log10(r["ratio"]))
                  for r in d["rows"] if r["sigma"] > 0] + [(0.0, 0.0)])
    return [p[0] for p in pts], [p[1] for p in pts]


def cells(name):
    out = []
    for suffix in ("", "-k1"):
        p = RESULTS / f"ppl_{name}{suffix}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        base = next((r["ppl"] for r in d["rows"] if r["config"] == "fp16-KV"), None)
        if not base:
            continue
        for r in d["rows"]:
            if r.get("bits") and np.isfinite(r["ppl"]):
                out.append(dict(bits=r["bits"], center=r["center"],
                                x=1 - r["logit_r_min"],
                                actual=math.log10(r["ppl"] / base)))
    return out


def main():
    names = sorted({p.stem.replace("sensitivity_", "").split("-round")[0]
                    .split("-sign")[0].split("-seed43")[0]
                    for p in RESULTS.glob("sensitivity_*-round.json")})
    if not names:
        print("no alternative-family curves yet"); return

    print(f"models with alternative-family curves: {', '.join(names)}\n")
    summary = {}
    for suffix in FAMILIES:
        errs, errs_1bit = [], []
        for n in names:
            c = curve(n, suffix)
            if c is None:
                continue
            xs, ys = c
            for cell in cells(n):
                pred = (float(np.interp(cell["x"], xs, ys)) if cell["x"] <= xs[-1]
                        else ys[-1] + (ys[-1]-ys[-2])*(cell["x"]-xs[-1])/(xs[-1]-xs[-2]))
                e = abs(pred - cell["actual"])
                errs.append(e)
                if cell["bits"] == 1:
                    errs_1bit.append((n, cell["center"], pred, cell["actual"]))
        if not errs:
            continue
        summary[suffix] = (np.median(errs), np.percentile(errs, 90), errs_1bit)

    print(f"{'calibration family':20s} {'median |err|':>13s} {'90th':>8s} "
          f"{'1-bit median':>13s}")
    print("-" * 60)
    for suffix, (med, p90, e1) in summary.items():
        m1 = np.median([abs(p - a) for _, _, p, a in e1]) if e1 else float("nan")
        print(f"{LABEL[suffix]:20s} {med:>13.4f} {p90:>8.4f} {m1:>13.4f}")

    print(f"\n1-bit cells in detail (predicted vs actual PPL ratio):")
    hdr = f"{'model':14s} {'ctr':>5s} {'actual':>9s}"
    for suffix in summary:
        hdr += f" {LABEL[suffix][:11]:>12s}"
    print(hdr); print("-" * len(hdr))
    rows = {}
    for suffix, (_, _, e1) in summary.items():
        for n, ctr, pred, act in e1:
            rows.setdefault((n, ctr), {"actual": act})[suffix] = pred
    for (n, ctr), v in sorted(rows.items()):
        line = f"{n:14s} {str(ctr):>5s} {'x%.2f' % 10**v['actual']:>9s}"
        for suffix in summary:
            line += (f" {'x%.2f' % 10**v[suffix]:>12s}" if suffix in v else f" {'-':>12s}")
        print(line)

    json.dump({LABEL[k]: dict(median=float(v[0]), p90=float(v[1]))
               for k, v in summary.items()},
              open(RESULTS / "family_test.json", "w"), indent=1)
    print(f"\nwrote {RESULTS/'family_test.json'}")


if __name__ == "__main__":
    main()
