#!/usr/bin/env python3
"""Is a model's noise-response curve predictable from one forward pass?

SS6.22's protocol costs ~5 noise passes per model to build the curve. If curve
steepness were predictable from a cheap structural statistic, the protocol
would collapse to a single pass. The natural candidate is the mean-dominance
ratio rho (SS6.20), since SS6.22 argues that mean-dominated keys are exactly
what makes a curve steep.

Summarising each curve by its NOISE TOLERANCE

    x2 = the score-space noise at which damage first reaches x2

(larger = more robust), this script correlates x2 against rho_mean, rho_max,
the fraction of heads with rho > 1, and shared energy.

Usage:  python curve_vs_rho.py
"""
from __future__ import annotations
import json, math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def spearman(x, y):
    rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/d) if d else float("nan")


def pearson(x, y):
    x = np.asarray(x, float) - np.mean(x); y = np.asarray(y, float) - np.mean(y)
    d = math.sqrt((x**2).sum() * (y**2).sum())
    return float((x*y).sum()/d) if d else float("nan")


def tolerance(name, thresh=2.0):
    """Score-noise at which this model's curve first reaches `thresh` damage."""
    d = json.loads((RESULTS / f"sensitivity_{name}.json").read_text())
    pts = sorted([(1 - r["logit_r_min"], math.log10(r["ratio"]))
                  for r in d["rows"] if r["sigma"] > 0] + [(0.0, 0.0)])
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    t = math.log10(thresh)
    if ys[-1] < t:
        return None          # never reaches the threshold in the measured range
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 <= t <= y1:
            if y1 == y0:
                return x0
            return x0 + (x1 - x0) * (t - y0) / (y1 - y0)
    return None


def main():
    rows = []
    for p in sorted(RESULTS.glob("sensitivity_*.json")):
        name = p.stem.replace("sensitivity_", "")
        rp = RESULTS / f"rho_{name}.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text())
        tol = tolerance(name)
        if tol is None:
            print(f"[skip] {name}: curve never reaches x2 in range")
            continue
        rows.append(dict(name=name, tol=tol, rho_mean=r["rho_mean"],
                         rho_max=r["rho_max"], frac=r["frac_heads_gt1"],
                         energy=r["shared_energy_mean"]))
    if len(rows) < 4:
        print(f"only {len(rows)} models have both a curve and rho; need more")
        for x in rows:
            print(" ", x["name"])
        return

    rows.sort(key=lambda r: r["tol"])
    print(f"{len(rows)} models with both a noise curve and a rho measurement\n")
    print(f"{'model':14s} {'noise tol (x2)':>14s} {'rho mean':>9s} {'rho max':>8s} "
          f"{'heads>1':>8s} {'shared E':>9s}")
    print("-" * 66)
    for r in rows:
        print(f"{r['name']:14s} {r['tol']:>14.4f} {r['rho_mean']:>9.3f} "
              f"{r['rho_max']:>8.1f} {100*r['frac']:>7.1f}% {100*r['energy']:>8.1f}%")

    tol = [r["tol"] for r in rows]
    print(f"\n{'predictor':16s} {'Spearman':>9s} {'Pearson(log)':>13s}")
    print("-" * 42)
    for key, label in [("rho_mean", "rho mean"), ("rho_max", "rho max"),
                       ("frac", "frac heads>1"), ("energy", "shared energy")]:
        v = [r[key] for r in rows]
        lv = [math.log10(max(x, 1e-6)) for x in v]
        print(f"{label:16s} {spearman(v, tol):>9.3f} "
              f"{pearson(lv, [math.log10(max(t,1e-6)) for t in tol]):>13.3f}")

    json.dump(rows, open(RESULTS / "curve_vs_rho.json", "w"), indent=1)
    print(f"\nwrote {RESULTS/'curve_vs_rho.json'}")


if __name__ == "__main__":
    main()
