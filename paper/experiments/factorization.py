#!/usr/bin/env python3
"""Test the damage factorization: damage = f_model(score-space noise).

Inputs:
  sensitivity_<name>.json  — per-model noise-response curves (pure isotropic
                             key noise at several sigma; no quantizer)
  ppl_<name>[-k1].json     — quantizer cells for the same models

For every quantizer cell of a probed model, predict its log10 PPL ratio by
evaluating the model's own noise curve at the cell's measured score-space
noise (x = 1 - mean logit r), then compare with the actual ratio.

Result (SS6.22): quantizer cells land ON their model's own noise curve, in
BOTH regimes. The hypothesis that mean-pathology cells would deviate above
their curve was REFUTED — mean-dominated models simply have catastrophically
steep curves (Qwen2.5-1.5B loses x348 at sigma=0.05). Damage is therefore
predicted by a single model-specific function of one cheap scalar, and there
is no universal proxy threshold because that function differs by model.

Outputs a report and paper/figures/fig3_factorization.{png,pdf}.
"""
from __future__ import annotations
import json, math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE.parents[0] / "figures"


XKEY = "logit_r_min"   # worst layer; see SS6.22 for why not the mean


def curve_of(name, xkey=XKEY):
    p = RESULTS / f"sensitivity_{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    pts = [(1.0 - r[xkey], math.log10(max(r["ratio"], 1e-9)))
           for r in d["rows"] if r["sigma"] > 0 and xkey in r]
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


def cells_of(name, xkey=XKEY):
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
                            x=1.0 - r[xkey],
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

    a = np.array([r["actual"] for r in rows])
    p = np.array([r["pred"] for r in rows])
    resid = a - p
    r2 = 1 - float((resid ** 2).sum() / ((a - a.mean()) ** 2).sum())
    print(f"\n=== FACTORIZATION ACCURACY ({len(rows)} cells, {len(names)} models) ===")
    print(f"  R^2 on log10 damage       {r2:.4f}")
    print(f"  median |error|            {np.median(abs(resid)):.4f} log10 "
          f"(x{10 ** np.median(abs(resid)):.3f})")
    print(f"  90th percentile |error|   {np.percentile(abs(resid), 90):.4f} log10 "
          f"(x{10 ** np.percentile(abs(resid), 90):.2f})")
    print(f"  max |error|               {abs(resid).max():.4f} log10 "
          f"(x{10 ** abs(resid).max():.1f})")
    print(f"  damage range covered      x{10 ** a.min():.2f} to x{10 ** a.max():,.0f}")

    # Baseline: the best MODEL-AGNOSTIC mapping from the same scalar, fitted
    # on all other models (leave-one-model-out). This isolates how much of the
    # accuracy comes from per-model calibration rather than from the scalar.
    uni = []
    for c in rows:
        o = sorted((z["x"], z["actual"]) for z in rows if z["model"] != c["model"])
        uni.append(abs(float(np.interp(c["x"], [z[0] for z in o], [z[1] for z in o]))
                       - c["actual"]))
    uni = np.array(uni)
    print(f"\n  model-agnostic baseline (leave-one-model-out):")
    print(f"    median |error| {np.median(uni):.4f} log10 (x{10 ** np.median(uni):.2f}),  "
          f"90th {np.percentile(uni, 90):.4f} (x{10 ** np.percentile(uni, 90):.1f})")

    print(f"\n  largest residuals (honest failure cases):")
    for w in sorted(rows, key=lambda r: -abs(r["dev"]))[:5]:
        print(f"    {w['model']:13s} {w['bits']}bit ctr={str(w['center']):5s} "
              f"predicted x{10 ** w['pred']:>9.2f}  actual x{10 ** w['actual']:>9.2f}")

    print(f"\n{'model':14s} {'cells':>5s} {'median err':>11s} {'max err':>8s} "
          f"{'damage span':>20s}")
    for m in names:
        cm = [r for r in rows if r["model"] == m]
        if not cm:
            continue
        e = np.array([abs(r["dev"]) for r in cm])
        rr = [10 ** r["actual"] for r in cm]
        print(f"{m:14s} {len(cm):>5d} {np.median(e):>11.4f} {e.max():>8.4f} "
              f"{'x%.2f - x%.0f' % (min(rr), max(rr)):>20s}")

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
        fig.suptitle("Damage is a model-specific function of one scalar: each "
                     "model's noise curve (line, measured with Gaussian noise "
                     "and no quantizer)\nvs its quantizer cells "
                     "(red=uncentered, green=centered, square=1-bit)",
                     fontsize=10)
        fig.supxlabel("score-space noise  (1 − worst-layer logit correlation)")
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
