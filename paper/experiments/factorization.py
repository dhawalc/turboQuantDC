#!/usr/bin/env python3
"""Test the damage factorization: damage = f_model(score-space noise).

Inputs:
  sensitivity_<name>.json  — per-model noise-response curves (pure isotropic
                             key noise at several sigma; no quantizer)
  ppl_<name>[-k1].json     — quantizer cells for the same models

For every quantizer cell of a probed model, predict its log10 PPL ratio by
evaluating the model's own noise curve at the cell's measured score-space
noise (x = 1 - mean logit r), then compare with the actual ratio.

Hypotheses:
  H1  uniform-damage cells (1-bit stress, centered cells) land ON the curve —
      i.e. the damage that SS6.19 showed no proxy could see becomes
      predictable from proxy + curve.
  H2  mean-pathology cells (Qwen2/2.5/Pythia uncentered) land far ABOVE the
      curve; the deviation is a principled pathology detector.

Outputs a report and paper/figures/fig3_factorization.{png,pdf}.
"""
from __future__ import annotations
import json, math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE.parents[0] / "figures"


def curve_of(name):
    p = RESULTS / f"sensitivity_{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    pts = [(1.0 - r["logit_r"], math.log10(max(r["ratio"], 1e-9)))
           for r in d["rows"] if r["sigma"] > 0 and "logit_r" in r]
    pts.append((0.0, 0.0))
    pts.sort()
    return pts


def predict(pts, x):
    """log10 ratio at score-noise x, linear interp; linear extrapolation
    beyond the last measured point (flagged by caller via x > pts[-1][0])."""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        # extrapolate from last two points
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        return y1 + (y1 - y0) * (x - x1) / max(x1 - x0, 1e-9)
    return float(np.interp(x, xs, ys))


def cells_of(name):
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
            if r.get("bits") is None or "logit_r" not in r:
                continue
            if not np.isfinite(r["ppl"]):
                continue
            out.append(dict(bits=r["bits"], center=r["center"],
                            x=1.0 - r["logit_r"],
                            lr_min=r["logit_r_min"], lr_mean=r["logit_r"],
                            actual=math.log10(max(r["ppl"] / base, 1e-9))))
    return out


def main():
    names = sorted(p.stem.replace("sensitivity_", "")
                   for p in RESULTS.glob("sensitivity_*.json"))
    if not names:
        print("no sensitivity curves yet"); return

    rows = []
    print(f"{'model':14s} {'bits':>4s} {'ctr':>5s} {'noise x':>8s} "
          f"{'pred':>7s} {'actual':>7s} {'dev':>7s}")
    print("-" * 62)
    for name in names:
        pts = curve_of(name)
        for c in cells_of(name):
            pred = predict(pts, c["x"])
            dev = c["actual"] - pred
            extrap = c["x"] > pts[-1][0]
            rows.append(dict(model=name, **c, pred=pred, dev=dev,
                             extrap=extrap))
            print(f"{name:14s} {c['bits']:>4d} {str(c['center']):>5s} "
                  f"{c['x']:8.4f} {pred:7.3f} {c['actual']:7.3f} {dev:+7.3f}"
                  + (" *extrap" if extrap else ""))

    # H1: cells whose damage is uniform (centered, or 1-bit stress) —
    # exclude uncentered cells of models with known mean pathology.
    PATHOLOGICAL = {"qwen2.5-1.5b", "qwen3-1.7b", "pythia-2.8b", "opt-2.7b",
                    "granite3.3-2b"}
    uniform = [r for r in rows
               if r["center"] or r["model"] not in PATHOLOGICAL]
    patho = [r for r in rows
             if not r["center"] and r["model"] in PATHOLOGICAL]

    def stats(rs, label):
        if not rs:
            return
        a = np.array([r["actual"] for r in rs])
        p = np.array([r["pred"] for r in rs])
        resid = a - p
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((a - a.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
        print(f"\n{label}: n={len(rs)}  R^2={r2:.3f}  "
              f"median |dev|={np.median(np.abs(resid)):.3f} (log10)  "
              f"max dev={resid.max():+.3f}")

    stats(uniform, "H1 uniform-damage cells (centered + non-pathological)")
    stats(patho, "H2 pathology cells (uncentered, mean-dominated models)")
    if patho and uniform:
        umax = max(r["dev"] for r in uniform)
        pdevs = sorted(r["dev"] for r in patho)
        print(f"\nDeviation-from-own-curve as pathology detector:")
        print(f"  max deviation among uniform cells: {umax:+.3f}")
        print(f"  pathology-cell deviations: "
              + ", ".join(f"{d:+.2f}" for d in pdevs))
        sep = sum(1 for d in pdevs if d > umax)
        print(f"  {sep}/{len(pdevs)} pathology cells exceed every uniform cell")

    json.dump(rows, open(RESULTS / "factorization_cells.json", "w"), indent=1)

    # figure 3
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(names)
        ncol = 4
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow),
                                 sharex=True, sharey=True)
        axes = np.atleast_2d(axes)
        for i, name in enumerate(names):
            ax = axes[i // ncol][i % ncol]
            pts = curve_of(name)
            xs = [p[0] for p in pts]; ys = [10 ** p[1] for p in pts]
            ax.plot(xs, ys, "k-", lw=1.4, label="noise curve")
            for r in [r for r in rows if r["model"] == name]:
                color = "tab:green" if r["center"] else "tab:red"
                marker = "s" if r["bits"] == 1 else "o"
                ax.plot(r["x"], 10 ** r["actual"], marker, color=color,
                        ms=5, alpha=0.85)
            ax.set_yscale("log")
            ax.set_title(name, fontsize=8)
            ax.grid(alpha=0.25)
        for j in range(n, nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle("Each model's private noise curve (line) vs its quantizer "
                     "cells (red=uncentered, green=centered, square=1-bit)",
                     fontsize=10)
        fig.supxlabel("score-space noise  (1 − mean logit correlation)")
        fig.supylabel("PPL ratio")
        fig.tight_layout()
        FIGS.mkdir(exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(FIGS / f"fig3_factorization.{ext}", dpi=300)
        print(f"\nwrote fig3 ({n} models)")
    except Exception as e:
        print("figure failed:", e)


if __name__ == "__main__":
    main()
