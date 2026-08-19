#!/usr/bin/env python3
"""What does the compressed KV cache actually BUY? Measured, not derived.

Every other experiment in this campaign measures quality. This one measures the
resource win, which is the reason to compress a KV cache at all:

  1. **Bytes actually held.** Walks the live cache objects and sums real tensor
     storage (``untyped_storage().nbytes()``, deduplicated by storage pointer so
     views and expanded means are counted once). This is the number the paper
     previously refused to quote, because the mean was materialized to full
     sequence length and cancelled the compression.
  2. **Peak GPU memory** for a forward pass at a given context length.
  3. **Longest context that fits** on this GPU, by doubling until OOM.
  4. **Prefill throughput** in tokens/second, so the compression's compute cost
     is visible next to its memory saving.

FP16-KV baseline and compressed run use the same model weights and the same
inputs, so the comparison isolates the cache.

Usage:
    python memory_bench.py --model <path> --name qwen3.5-0.8b --bits 2 --center
    python memory_bench.py --model <path> --name x --max-context-search
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
RESULTS = HERE / "results"

from ppl_harness import KVCompressor, patched_cache, load_model, SCRATCH


def cache_bytes(cache) -> int:
    """Real bytes held by a cache object, deduplicated by storage pointer.

    Deduplication matters: expanded means are stride-0 views sharing one
    storage, and counting them per-token would reproduce exactly the error this
    measurement exists to correct.
    """
    seen, total = set(), 0
    def walk(o, depth=0):
        nonlocal total
        if depth > 6:
            return
        if torch.is_tensor(o):
            st = o.untyped_storage()
            key = (st.data_ptr(), st.nbytes())
            if key[0] and key not in seen:
                seen.add(key); total += key[1]
            return
        if isinstance(o, (list, tuple)):
            for x in o: walk(x, depth + 1)
        elif isinstance(o, dict):
            for x in o.values(): walk(x, depth + 1)
        elif hasattr(o, "__dict__"):
            for x in vars(o).values(): walk(x, depth + 1)
    walk(cache)
    return total


COMPRESSED_FIELDS = ("_key_indices", "_key_norms", "_key_res_signs",
                     "_key_res_scales", "_key_means", "_val_indices",
                     "_val_norms", "_raw_keys", "_raw_vals")


def compressed_only_bytes(layer) -> int:
    """Bytes of the compressed REPRESENTATION alone.

    The harness quantizes and then dequantizes, handing FP16 reconstructions
    back to the model's own cache, so a live process holds the compressed form
    AND full FP16 copies AND a dequantization cache. That is correct for
    measuring quality and useless for measuring memory. This function isolates
    what a cache that stored only the compressed form would hold.
    """
    seen, total = set(), 0
    for name in COMPRESSED_FIELDS:
        for t in getattr(layer, name, []) or []:
            if not torch.is_tensor(t):
                continue
            st = t.untyped_storage()
            key = (st.data_ptr(), st.nbytes())
            if key[0] and key not in seen:
                seen.add(key); total += key[1]
    return total


def run_once(model, ids, comp):
    """One forward pass. Returns (peak_bytes, cache_bytes, seconds)."""
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    torch.cuda.synchronize(); t0 = time.time()
    with patched_cache(comp), torch.no_grad():
        # logits_to_keep=1: only the last position's logits are materialized.
        # A full-sequence logit tensor is vocab x context x 4 bytes (4.2 GiB at
        # 8K context on a 128k-vocab model) and would dominate the very
        # measurement this script exists to make.
        try:
            out = model(ids, use_cache=True, logits_to_keep=1)
        except TypeError:
            out = model(ids, use_cache=True, num_logits_to_keep=1)
    torch.cuda.synchronize(); dt = time.time() - t0
    cb = cache_bytes(out.past_key_values)
    rep = 0
    if comp is not None:
        cb += sum(cache_bytes(l) for l in comp.layers.values())
        rep = sum(compressed_only_bytes(l) for l in comp.layers.values())
    peak = torch.cuda.max_memory_allocated() - before
    del out
    gc.collect(); torch.cuda.empty_cache()
    return peak, cb, dt, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--max-context-search", action="store_true",
                    help="double the context until OOM, for both arms")
    a = ap.parse_args()

    torch.manual_seed(42)
    wt = SCRATCH / "wikitext2_test.txt"
    text = wt.read_text() if wt.exists() else "the quick brown fox " * 200000
    model, tok = load_model(a.model, a.load_4bit)
    full = tok(text, return_tensors="pt")["input_ids"]
    print(f"=== {a.name}: {a.bits}-bit keys, centering={a.center} ===", flush=True)
    print(f"corpus has {full.shape[1]} tokens", flush=True)

    report = {"name": a.name, "bits": a.bits, "center": a.center, "runs": []}

    ctx = a.context
    if ctx > full.shape[1]:
        ctx = full.shape[1]
    ids = full[:, :ctx].to(model.device)

    for label, comp in (("fp16-KV", None),
                        (f"{a.bits}bit", KVCompressor(key_bits=a.bits,
                                                      center=a.center))):
        peak, cb, dt, rep = run_once(model, ids, comp)
        tps = ctx / dt
        print(f"[{label:8s}] ctx={ctx}  live={cb/2**20:8.1f} MiB  "
              f"compressed-repr={rep/2**20:8.1f} MiB  peak={peak/2**20:7.1f} MiB  "
              f"{tps:7.0f} tok/s", flush=True)
        report["runs"].append(dict(config=label, context=ctx, cache_bytes=cb,
                                   repr_bytes=rep, peak_bytes=peak,
                                   seconds=dt, tok_per_s=tps))

    if len(report["runs"]) == 2:
        f, q = report["runs"]
        if q["repr_bytes"]:
            print(f"\nCOMPRESSED REPRESENTATION vs FP16 cache: "
                  f"{f['cache_bytes']/q['repr_bytes']:.2f}x "
                  f"({f['cache_bytes']/2**20:.1f} -> {q['repr_bytes']/2**20:.1f} MiB)")
        if q["cache_bytes"]:
            print(f"live process holds {q['cache_bytes']/f['cache_bytes']:.2f}x the "
                  f"FP16 cache (quality harness keeps reconstructions too)")
        print(f"prefill throughput: {q['tok_per_s']/f['tok_per_s']:.2f}x baseline")

    if a.max_context_search:
        print("\n--- longest context that fits ---", flush=True)
        for label, mk in (("fp16-KV", lambda: None),
                          (f"{a.bits}bit", lambda: KVCompressor(key_bits=a.bits,
                                                                center=a.center))):
            n, best = 1024, 0
            while n <= full.shape[1]:
                try:
                    run_once(model, full[:, :n].to(model.device), mk())
                    best = n; n *= 2
                except torch.cuda.OutOfMemoryError:
                    gc.collect(); torch.cuda.empty_cache(); break
                except Exception as e:
                    print(f"  {label} stopped at {n}: {type(e).__name__}"); break
            print(f"[{label:8s}] longest context that fits: {best}", flush=True)
            report.setdefault("max_context", {})[label] = best

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"membench_{a.name}_{a.bits}bit{'_c' if a.center else ''}.json"
    json.dump(report, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
