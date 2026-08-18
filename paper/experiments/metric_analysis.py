#!/usr/bin/env python3
"""Do the cheap proxy metrics predict real KV-compression damage?

Reads every ppl_*.json produced by ppl_harness.py and asks, across all
(model, bit-width, centering) cells:

  * how well does each proxy rank configurations by true perplexity damage?
  * how often does a proxy pass a configuration that is in fact broken?

Damage is log10(ppl / baseline_ppl), which makes a 1000x blow-up and a 1.01x
tax commensurable on one axis.
"""
from __future__ import annotations
import json, math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BROKEN = 2.0          # ppl ratio above this = the model is meaningfully damaged
COS_PASS = 0.995      # the success criterion this project set for itself


def spearman(x, y):
    def rank(v):
        o = np.argsort(np.argsort(np.asarray(v, dtype=float)))
        return o.astype(float)
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else float("nan")


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - x.mean(); y = y - y.mean()
    d = math.sqrt((x ** 2).sum() * (y ** 2).sum())
    return float((x * y).sum() / d) if d else float("nan")


def collect():
    rows = []
    for f in sorted(RESULTS.glob("ppl_*.json")):
        d = json.loads(f.read_text())
        if "error" in d or "rows" not in d:
            continue
        base = next((r["ppl"] for r in d["rows"] if r["config"] == "fp16-KV"), None)
        if not base or not np.isfinite(base):
            continue
        for r in d["rows"]:
            if r["config"] == "fp16-KV" or "vec_cos" not in r:
                continue
            if not np.isfinite(r["ppl"]):
                continue
            rows.append(dict(
                model=d["name"], kv_heads=d.get("kv_heads"),
                head_dim=d.get("head_dim"), bits=r["bits"], center=r["center"],
                ppl=r["ppl"], base=base, ratio=r["ppl"] / base,
                damage=math.log10(max(r["ppl"] / base, 1e-9)),
                vec_cos=r["vec_cos"], vec_cos_min=r.get("vec_cos_min", r["vec_cos"]),
                logit_r=r["logit_r"], logit_r_min=r["logit_r_min"],
                spread=r["spread"], spread_min=r.get("spread_min")))
    return rows


def main():
    rows = collect()
    if not rows:
        print("no results yet"); return
    models = sorted({r["model"] for r in rows})
    print(f"{len(rows)} (model, bits, centering) cells across {len(models)} models")
    print("models:", ", ".join(models))

    dmg = [r["damage"] for r in rows]
    proxies = {
        "vec_cos (mean)":       [r["vec_cos"] for r in rows],
        "logit_r (mean)":       [r["logit_r"] for r in rows],
        "logit_r (WORST layer)": [r["logit_r_min"] for r in rows],
        "spread ratio":         [r["spread"] for r in rows],
    }
    print(f"\n{'proxy':24s} {'Spearman':>10s} {'Pearson':>9s}   (vs log10 ppl ratio)")
    print("-" * 60)
    for k, v in proxies.items():
        print(f"{k:24s} {spearman(v, dmg):10.3f} {pearson(v, dmg):9.3f}")

    broken = [r for r in rows if r["ratio"] > BROKEN]
    fine = [r for r in rows if r["ratio"] <= BROKEN]
    print(f"\nbroken cells (ppl ratio > {BROKEN}x): {len(broken)} / {len(rows)}")

    if broken and fine:
        print(f"\n{'proxy':24s} {'broken (mean)':>14s} {'fine (mean)':>13s} {'separates?':>12s}")
        print("-" * 66)
        for k, v in proxies.items():
            b = np.mean([v[i] for i, r in enumerate(rows) if r["ratio"] > BROKEN])
            g = np.mean([v[i] for i, r in enumerate(rows) if r["ratio"] <= BROKEN])
            bmax = max(v[i] for i, r in enumerate(rows) if r["ratio"] > BROKEN)
            gmin = min(v[i] for i, r in enumerate(rows) if r["ratio"] <= BROKEN)
            sep = "YES" if bmax < gmin else "no (overlap)"
            print(f"{k:24s} {b:14.4f} {g:13.4f} {sep:>12s}")

        print(f"\nFalse pass rate at the {COS_PASS} cosine criterion:")
        fp = [r for r in broken if r["vec_cos"] >= COS_PASS]
        print(f"  broken configs that PASS vec_cos >= {COS_PASS}: "
              f"{len(fp)}/{len(broken)} = {100*len(fp)/len(broken):.0f}%")
        worst = sorted(broken, key=lambda r: -r["vec_cos"])[:5]
        print("  worst offenders (highest cosine among broken configs):")
        for r in worst:
            print(f"    {r['model']:16s} {r['bits']}bit center={str(r['center']):5s} "
                  f"vec_cos={r['vec_cos']:.4f}  logit_r_min={r['logit_r_min']:.4f}  "
                  f"ppl x{r['ratio']:.1f}")

        # Is there a threshold on worst-layer logit_r that cleanly separates?
        # candidate thresholds are MIDPOINTS between observed values, so a
        # clean separating gap is actually reachable (using observed values as
        # thresholds can never sit strictly inside the gap)
        obs = sorted({r["logit_r_min"] for r in rows})
        vals = [(a + b) / 2 for a, b in zip(obs, obs[1:])] + [obs[-1] + 1e-6]
        best = None
        for t in vals:
            tp = sum(1 for r in rows if r["logit_r_min"] < t and r["ratio"] > BROKEN)
            fp_ = sum(1 for r in rows if r["logit_r_min"] < t and r["ratio"] <= BROKEN)
            fn = sum(1 for r in rows if r["logit_r_min"] >= t and r["ratio"] > BROKEN)
            err = fp_ + fn
            if best is None or err < best[1]:
                best = (t, err, tp, fp_, fn)
        t, err, tp, fp_, fn = best
        print(f"\nBest single threshold on WORST-layer logit_r: {t:.4f}")
        print(f"  misclassified {err}/{len(rows)} cells "
              f"(false alarms {fp_}, missed breakages {fn})")
        bmax = max(r["logit_r_min"] for r in broken)
        gmin = min(r["logit_r_min"] for r in fine)
        if bmax < gmin:
            print(f"  separating gap: worst broken {bmax:.4f} < best-fine floor "
                  f"{gmin:.4f}  (any threshold in between is perfect)")

        # Held-out validation: fit the threshold on Qwen2.5 only (the family the
        # failure was discovered on) and test it on everything else. This is the
        # honest test of whether the proxy generalizes or was tuned to the data.
        fit = [r for r in rows if r["model"].startswith("qwen2.5")]
        held = [r for r in rows if not r["model"].startswith("qwen2.5")]
        if fit and held and any(r["ratio"] > BROKEN for r in fit):
            fb = max(r["logit_r_min"] for r in fit if r["ratio"] > BROKEN)
            fg = min(r["logit_r_min"] for r in fit if r["ratio"] <= BROKEN)
            thr = (fb + fg) / 2
            err = sum(1 for r in held
                      if (r["logit_r_min"] < thr) != (r["ratio"] > BROKEN))
            hb = sum(1 for r in held if r["ratio"] > BROKEN)
            print(f"\nHELD-OUT VALIDATION (fit on Qwen2.5, test on everything else)")
            print(f"  threshold fitted on Qwen2.5 only: {thr:.4f}")
            print(f"  held-out cells: {len(held)} across "
                  f"{len({r['model'] for r in held})} models ({hb} truly broken)")
            print(f"  misclassified: {err}/{len(held)}")
            # the same test for cosine, using its own best in-family threshold
            cb = max(r["vec_cos"] for r in fit if r["ratio"] > BROKEN)
            cg = min(r["vec_cos"] for r in fit if r["ratio"] <= BROKEN)
            cthr = (cb + cg) / 2
            cerr = sum(1 for r in held
                       if (r["vec_cos"] < cthr) != (r["ratio"] > BROKEN))
            print(f"  same procedure with vec_cos (threshold {cthr:.4f}): "
                  f"{cerr}/{len(held)} misclassified")

    print("\nPer-cell detail (sorted by true damage):")
    print(f"{'model':16s} {'bits':>4s} {'ctr':>5s} {'ppl':>12s} {'ratio':>10s} "
          f"{'vec_cos':>8s} {'logit_r':>8s} {'lr_min':>8s}")
    print("-" * 84)
    for r in sorted(rows, key=lambda r: -r["damage"]):
        print(f"{r['model']:16s} {r['bits']:>4d} {str(r['center']):>5s} "
              f"{r['ppl']:12.3f} {r['ratio']:10.2f} {r['vec_cos']:8.4f} "
              f"{r['logit_r']:8.4f} {r['logit_r_min']:8.4f}")

    json.dump(rows, open(RESULTS / "metric_analysis_cells.json", "w"), indent=1)
    print(f"\nwrote {RESULTS/'metric_analysis_cells.json'}")


if __name__ == "__main__":
    main()
