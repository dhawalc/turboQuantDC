#!/usr/bin/env python3
"""Cross-architecture atlas driver.

For each model: download, run the PPL harness at several bit-widths with
centering on and off, record paired (proxy-metric, true-perplexity) data, then
delete the weights before moving on so disk stays bounded.

Resumable: a model whose results file already exists is skipped.
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SCRATCH = Path(os.environ.get(
    "TQ_SCRATCH",
    "/tmp/claude-1000/-home-dhawal-turboQuantDC/0bebcfa0-db7f-400c-8b64-efaab6cc84c0/scratchpad"))
WORK = SCRATCH / "atlas"

# (hf repo, short name, load_in_4bit)
MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct",        "qwen2.5-1.5b",  False),
    ("Qwen/Qwen2.5-3B-Instruct",          "qwen2.5-3b",    False),
    ("Qwen/Qwen3-1.7B",                   "qwen3-1.7b",    False),
    ("Qwen/Qwen3-4B",                     "qwen3-4b",      False),
    ("Qwen/Qwen3.5-0.8B",                 "qwen3.5-0.8b",  False),
    ("unsloth/Llama-3.2-1B-Instruct",     "llama3.2-1b",   False),
    ("unsloth/Llama-3.2-3B-Instruct",     "llama3.2-3b",   False),
    ("unsloth/gemma-2-2b-it",             "gemma2-2b",     False),
    ("unsloth/gemma-3-4b-it",             "gemma3-4b",     False),
    ("microsoft/Phi-4-mini-instruct",     "phi4-mini",     False),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "smollm2-1.7b", False),
    ("allenai/OLMo-2-0425-1B-Instruct",   "olmo2-1b",      False),
    ("ibm-granite/granite-3.3-2b-instruct", "granite3.3-2b", False),
    ("tiiuae/Falcon3-1B-Instruct",        "falcon3-1b",    False),
    ("mistralai/Ministral-8B-Instruct-2410", "ministral-8b", True),
]

BITS = ["2", "3", "4"]
TOKENS = "4096"


def run(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    for repo, name, four in MODELS:
        out = RESULTS / f"ppl_{name}.json"
        if out.exists():
            print(f"[skip] {name} already done", flush=True)
            continue
        d = WORK / name
        print(f"\n===== {name} ({repo}) =====", flush=True)
        t0 = time.time()
        r = run(f"hf download {repo} --local-dir {d}")
        if r.returncode != 0:
            print(f"[FAIL download] {name}: {r.stderr[-400:]}", flush=True)
            json.dump({"name": name, "model": repo, "error": "download: " + r.stderr[-400:]},
                      open(out, "w"), indent=1)
            shutil.rmtree(d, ignore_errors=True)
            continue
        print(f"  downloaded in {time.time()-t0:.0f}s", flush=True)

        cmd = (f"python3 {HERE/'ppl_harness.py'} --model {d} --name {name} "
               f"--bits {' '.join(BITS)} --tokens {TOKENS}"
               + (" --load-4bit" if four else ""))
        t0 = time.time()
        r = run(cmd, cwd=str(HERE))
        tail = "\n".join(l for l in r.stdout.splitlines()
                         if l.startswith("[") or "wrote" in l or "tokens=" in l)
        print(tail or r.stderr[-800:], flush=True)
        if not out.exists():
            json.dump({"name": name, "model": repo,
                       "error": "harness: " + (r.stderr[-600:] or "no output")},
                      open(out, "w"), indent=1)
        print(f"  ran in {time.time()-t0:.0f}s", flush=True)
        shutil.rmtree(d, ignore_errors=True)
    print("\nATLAS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
