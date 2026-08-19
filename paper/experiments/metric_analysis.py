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
            pl = list(r.get("per_layer_logit_r", {}).values())
            # damage compounds through depth: if each layer preserves a
            # fraction r_l of logit structure, end-to-end survival ~ prod r_l.
            prod = float(np.exp(sum(math.log(max(x, 1e-6)) for x in pl))) if pl else float("nan")
            geo = float(prod ** (1.0 / len(pl))) if pl else float("nan")
            rows.append(dict(
                model=d["name"], kv_heads=d.get("kv_heads"),
                head_dim=d.get("head_dim"), bits=r["bits"], center=r["center"],
                ppl=r["ppl"], base=base, ratio=r["ppl"] / base,
                damage=math.log10(max(r["ppl"] / base, 1e-9)),
                vec_cos=r["vec_cos"], vec_cos_min=r.get("vec_cos_min", r["vec_cos"]),
                logit_r=r["logit_r"], logit_r_min=r["logit_r_min"],
                logit_r_prod=prod, logit_r_geo=geo, n_layers=len(pl),
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
        "logit_r (PRODUCT)":    [r["logit_r_prod"] for r in rows],
        "logit_r (geo mean)":   [r["logit_r_geo"] for r in rows],
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

        # For each proxy: best single threshold on ALL cells, plus a
        # held-out test (threshold fitted on Qwen2.5 only, tested on the rest).
        # Candidate thresholds are MIDPOINTS between observed values so a clean
        # separating gap is actually reachable.
        fit = [r for r in rows if r["model"].startswith("qwen2.5")]
        held = [r for r in rows if not r["model"].startswith("qwen2.5")]
        hb = sum(1 for r in held if r["ratio"] > BROKEN)
        print(f"\nTHRESHOLD BATTERY (broken = ratio > {BROKEN}x; "
              f"held-out = {len(held)} cells on {len({r['model'] for r in held})} "
              f"non-Qwen2.5 models, {hb} truly broken)")
        print(f"{'proxy':24s} {'best thr':>9s} {'errors':>9s} {'gap?':>16s} "
              f"{'held-out thr':>13s} {'held-out err':>13s}")
        print("-" * 92)
        for k, v in proxies.items():
            obs = sorted(set(v))
            vals = [(a + b) / 2 for a, b in zip(obs, obs[1:])] + [obs[-1] + 1e-6]
            best = None
            for t in vals:
                e = sum(1 for i, r in enumerate(rows)
                        if (v[i] < t) != (r["ratio"] > BROKEN))
                if best is None or e < best[1]:
                    best = (t, e)
            bmax = max(v[i] for i, r in enumerate(rows) if r["ratio"] > BROKEN)
            gmin = min(v[i] for i, r in enumerate(rows) if r["ratio"] <= BROKEN)
            gap = (f"{bmax:.4f}<{gmin:.4f}" if bmax < gmin else "overlap")
            ho = ""
            hoe = ""
            if fit and held and any(r["ratio"] > BROKEN for r in fit):
                iv = {id(r): v[i] for i, r in enumerate(rows)}
                fb = max(iv[id(r)] for r in fit if r["ratio"] > BROKEN)
                fg = min(iv[id(r)] for r in fit if r["ratio"] <= BROKEN)
                thr = (fb + fg) / 2
                err = sum(1 for r in held
                          if (iv[id(r)] < thr) != (r["ratio"] > BROKEN))
                ho = f"{thr:.4f}"
                hoe = f"{err}/{len(held)}"
            print(f"{k:24s} {best[0]:9.4f} {best[1]:>4d}/{len(rows):<4d} "
                  f"{gap:>16s} {ho:>13s} {hoe:>13s}")

        # Which broken cells does each proxy FALSELY PASS at its best threshold?
        print("\nFalse passes (broken cells above each proxy's best threshold):")
        for k, v in proxies.items():
            obs = sorted(set(v))
            vals = [(a + b) / 2 for a, b in zip(obs, obs[1:])] + [obs[-1] + 1e-6]
            best = min(vals, key=lambda t: sum(
                1 for i, r in enumerate(rows) if (v[i] < t) != (r["ratio"] > BROKEN)))
            fps = [(rows[i], v[i]) for i, r in enumerate(rows)
                   if r["ratio"] > BROKEN and v[i] >= best]
            if fps:
                s = ", ".join(f"{r['model']}/{r['bits']}b/ctr={r['center']} "
                              f"(x{r['ratio']:.1f}, {val:.4f})" for r, val in fps)
                print(f"  {k}: {s}")

    # cutoff-sensitivity: the x2 "broken" line is a choice; show that the
    # ranking of proxies is not an artifact of it.
    print("\nCUTOFF SENSITIVITY (best-threshold errors at each damage cutoff):")
    print(f"{'cutoff':>8s} {'broken':>7s} {'lr_min errs':>12s} {'vec_cos errs':>13s}")
    for cut in (1.5, 2.0, 5.0, 10.0):
        for k in ("logit_r_min", "vec_cos"):
            v = [r[k] for r in rows]
            obs = sorted(set(v))
            vals = [(a + b) / 2 for a, b in zip(obs, obs[1:])] + [obs[-1] + 1e-6]
            err = min(sum(1 for i, r in enumerate(rows)
                          if (v[i] < t) != (r["ratio"] > cut)) for t in vals)
            if k == "logit_r_min":
                e1 = err
            else:
                e2 = err
        nb = sum(1 for r in rows if r["ratio"] > cut)
        print(f"{cut:>8.1f} {nb:>7d} {e1:>12d} {e2:>13d}")

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
