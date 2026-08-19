#!/usr/bin/env python3
"""Generate the paper's figures from the results JSONs.

Figure 1  per-layer attention-logit correlation profiles: the two damage
          geometries. Concentrated collapse (Qwen2.5-7B, 3-bit uncentered)
          against uniform starvation (Llama-3.2-1B, 1-bit centered) and a
          healthy profile (Qwen2.5-7B, 3-bit centered).
Figure 2  the metric result: per-vector cosine vs true damage (no structure)
          beside worst-layer logit correlation vs true damage (clean gap for
          the catastrophic regime), every atlas cell.

Writes PNG (300 dpi) and PDF to paper/figures/.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE.parents[0] / "figures"
FIGS.mkdir(exist_ok=True)


def rows_of(name):
    d = json.loads((RESULTS / f"ppl_{name}.json").read_text())
    base = next(r["ppl"] for r in d["rows"] if r["config"] == "fp16-KV")
    return d, base


def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    specs = [
        ("qwen2.5-7b", 3, False, "tab:red", "-",
         "Qwen2.5-7B 3-bit uncentered — concentrated collapse"),
        ("qwen2.5-7b", 3, True, "tab:green", "-",
         "Qwen2.5-7B 3-bit centered — healthy"),
        ("llama3.2-1b-k1", 1, True, "tab:orange", "--",
         "Llama-3.2-1B 1-bit centered — uniform starvation"),
    ]
    for name, bits, center, color, ls, label in specs:
        d, base = rows_of(name)
        r = next(r for r in d["rows"]
                 if r.get("bits") == bits and r.get("center") is center)
        pl = r["per_layer_logit_r"]
        xs = sorted(int(k) for k in pl)
        ax.plot(xs, [pl[str(x)] if str(x) in pl else pl[x] for x in xs],
                color=color, ls=ls, marker="o", ms=3.5, lw=1.6,
                label=f"{label}  (PPL ×{r['ppl']/base:,.1f})".replace(",", " "))
    ax.axhline(0.83, color="gray", lw=0.8, ls=":")
    ax.text(0.02, 0.835, "0.83 detection threshold", fontsize=8, color="gray",
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("layer")
    ax.set_ylabel("attention-logit correlation (per layer)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Two damage geometries, one statistic")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"fig1_damage_geometries.{ext}", dpi=300)
    plt.close(fig)
    print("fig1 written")


def fig2():
    cells = json.loads((RESULTS / "metric_analysis_cells.json").read_text())
    broken = [c for c in cells if c["ratio"] > 2]
    fine = [c for c in cells if c["ratio"] <= 2]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    for ax, key, label in [
            (axes[0], "vec_cos", "per-vector cosine similarity"),
            (axes[1], "logit_r_min", "worst-layer logit correlation")]:
        ax.scatter([c[key] for c in fine], [c["ratio"] for c in fine],
                   s=18, c="tab:green", alpha=0.65, label="working (≤×2)")
        ax.scatter([c[key] for c in broken], [c["ratio"] for c in broken],
                   s=26, c="tab:red", marker="x", label="broken (>×2)")
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("true perplexity ratio (log scale)")
    axes[0].axvline(0.995, color="gray", lw=0.8, ls=":")
    axes[0].text(0.995, 0.94, " 0.995 criterion", fontsize=7, color="gray",
                 transform=axes[0].get_xaxis_transform(), ha="right",
                 rotation=90)
    axes[1].axvline(0.83, color="gray", lw=0.8, ls=":")
    axes[1].text(0.83, 0.94, " 0.83 ", fontsize=7, color="gray",
                 transform=axes[1].get_xaxis_transform(), ha="right",
                 rotation=90)
    n = len(cells)
    axes[0].set_title(f"reconstruction metric: no structure ({n} cells)")
    axes[1].set_title("score-space metric: catastrophic cells separate")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"fig2_metric_scatter.{ext}", dpi=300)
    plt.close(fig)
    print(f"fig2 written ({n} cells)")


if __name__ == "__main__":
    fig1()
    fig2()
