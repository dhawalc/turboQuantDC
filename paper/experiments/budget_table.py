#!/usr/bin/env python3
"""Turn the damage law (SS6.22) into a deployment recommendation, and verify it.

Given a damage budget (e.g. "at most 5% perplexity increase"), each model's
noise-response curve inverts to a maximum tolerable score-space noise. Any
quantizer configuration whose measured score-noise is under that ceiling should
meet the budget. This script:

  1. inverts each model's curve at several budgets;
  2. picks the cheapest measured configuration under the ceiling;
  3. VERIFIES the pick against that configuration's true measured perplexity;
  4. reports how often the law's recommendation is actually safe.

Cost model: a configuration's per-coordinate cost is (bits + 1) - the Lloyd-Max
indices plus the pipeline's always-present 1-bit residual signs - so lower is
cheaper. Centering is free at inference (d floats per head, amortized).

Usage:  python budget_table.py [--budgets 1.02 1.05 1.10]
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def curve(name):
    d = json.loads((RESULTS / f"sensitivity_{name}.json").read_text())
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
                                x=1 - r["logit_r_min"], ratio=r["ppl"] / base,
                                cost=r["bits"] + 1))
    return out


def max_noise_for(xs, ys, budget):
    """Largest score-noise whose predicted damage stays within budget."""
    target = math.log10(budget)
    lo, hi = 0.0, xs[-1]
    for _ in range(60):
        mid = (lo + hi) / 2
        if float(np.interp(mid, xs, ys)) <= target:
            lo = mid
        else:
            hi = mid
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=float, nargs="+", default=[1.02, 1.05, 1.10])
    a = ap.parse_args()

    names = sorted(p.stem.replace("sensitivity_", "")
                   for p in RESULTS.glob("sensitivity_*.json"))
    report = {}
    for budget in a.budgets:
        rows, safe, unsafe = [], 0, 0
        for n in names:
            xs, ys = curve(n)
            ceil_ = max_noise_for(xs, ys, budget)
            ok = [c for c in cells(n) if c["x"] <= ceil_]
            if not ok:
                rows.append((n, ceil_, None, None, None)); continue
            pick = min(ok, key=lambda c: (c["cost"], c["x"]))
            good = pick["ratio"] <= budget
            safe += good; unsafe += (not good)
            rows.append((n, ceil_, pick, pick["ratio"], good))
        print(f"\n=== budget: perplexity increase <= {100*(budget-1):.0f}% ===")
        print(f"{'model':14s} {'max noise':>10s} {'recommended':>22s} "
              f"{'true PPL ratio':>15s} {'safe?':>6s}")
        print("-" * 74)
        for n, ceil_, pick, ratio, good in rows:
            if pick is None:
                print(f"{n:14s} {ceil_:10.4f} {'(no measured config)':>22s}")
                continue
            cfg = f"{pick['bits']}-bit" + (" + centering" if pick["center"] else "")
            print(f"{n:14s} {ceil_:10.4f} {cfg:>22s} {ratio:>14.3f}x "
                  f"{'yes' if good else 'NO':>6s}")
        tot = safe + unsafe
        print(f"  recommendations that actually met the budget: {safe}/{tot}")
        report[str(budget)] = dict(safe=safe, total=tot,
                                   rows=[(n, ceil_, None if p is None else
                                          dict(bits=p["bits"], center=p["center"],
                                               ratio=r, safe=g))
                                         for n, ceil_, p, r, g in rows])

    json.dump(report, open(RESULTS / "budget_table.json", "w"), indent=1)
    print(f"\nwrote {RESULTS/'budget_table.json'}")


if __name__ == "__main__":
    main()
