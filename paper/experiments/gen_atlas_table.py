#!/usr/bin/env python3
"""Emit the SS6.14 atlas table (markdown) directly from the results JSONs,
so the paper's table cannot drift from the data. Excludes the -k1 stress runs
(SS6.19 has its own table) and the seed/corpus robustness runs.

Usage:  python gen_atlas_table.py
"""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

FAMILY = {
    "qwen1.5": "Qwen1.5", "qwen2-": "Qwen2", "qwen2.5": "Qwen2.5",
    "qwen3-": "Qwen3", "qwen3.5": "Qwen3.5", "llama": "Llama 3.2",
    "gemma2": "Gemma 2", "gemma3": "Gemma 3", "phi4": "Phi-4",
    "smollm": "SmolLM2", "olmo": "OLMo 2", "granite": "Granite 3.3",
    "falcon": "Falcon 3", "ministral": "Ministral", "opt": "OPT",
    "pythia": "Pythia", "yi": "Yi 1.5",
}


def fam(name):
    for k, v in FAMILY.items():
        if name.startswith(k):
            return v
    return name


def cell(ratio):
    if ratio is None:
        return "—"
    s = f"×{ratio:,.2f}" if ratio < 10 else f"×{ratio:,.0f}"
    return f"**{s}**" if ratio > 2 else s


def main():
    rows = []
    for f in sorted(RESULTS.glob("ppl_*.json")):
        if "-k1" in f.name or "seed" in f.name or "gutenberg" in f.name:
            continue
        d = json.loads(f.read_text())
        if "rows" not in d:
            continue
        base = next((r["ppl"] for r in d["rows"] if r["config"] == "fp16-KV"), None)
        if not base:
            continue
        by_bits = {}
        for r in d["rows"]:
            if r.get("bits") and r.get("center") is False:
                by_bits[r["bits"]] = r["ppl"] / base
        worst = max(by_bits.values(), default=0)
        kv = d.get("kv_heads")
        if kv in (-1, None):   # config lacks num_key_value_heads => MHA
            kv = {"pythia-2.8b": "32 (MHA)", "opt-2.7b": "32 (MHA)"}.get(d["name"], "MHA")
        elif d["name"] in ("qwen1.5-1.8b", "olmo2-1b", "smollm2-1.7b") or kv >= 16:
            kv = f"{kv} (MHA)" if kv >= 16 else kv
        rows.append((worst, d["name"], fam(d["name"]), kv, base, by_bits))
    rows.sort(reverse=True)
    print("| Model | Family | KV heads | baseline PPL | 2-bit | 3-bit | 4-bit |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for worst, name, family, kv, base, bb in rows:
        print(f"| {name} | {family} | {kv} | {base:.2f} "
              f"| {cell(bb.get(2))} | {cell(bb.get(3))} | {cell(bb.get(4))} |")
    print(f"\n{len(rows)} models, "
          f"{sum(1 for w,*_ in rows if w > 2)} with at least one broken "
          f"uncentered cell (>x2)")


if __name__ == "__main__":
    main()
