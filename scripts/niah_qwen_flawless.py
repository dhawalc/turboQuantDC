"""Needle-in-a-haystack quality validation for the running Qwen3.6-27B vLLM server.

Hits the OpenAI-compatible HTTP endpoint (NOT HF transformers in-process — that's
what benchmarks/niah_for_tom.py does for a different purpose). This validates the
deployed server can find a needle in long context.

Inserts a unique numeric needle ("the magic number is XYZ") at various depths in
a long context built from filler text, asks for the magic number, reports
per-depth pass/fail.

Usage:
    .venv-vllm/bin/python scripts/niah_qwen_flawless.py \\
        --url http://127.0.0.1:8000 --model qwen3.6-27b \\
        --context-tokens 4000 --depths 0.1 0.5 0.9 \\
        --output benchmarks/results/qwen_flawless/niah_4k.json
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
import urllib.request
from pathlib import Path


FILLER = (
    "The cat sat on the mat. Time passes slowly in the afternoon. Birds sing. "
    "Books are stacked on shelves. The window is open. Coffee is cooling. "
    "A notebook lies on the table. Pages turn quietly. A pen rests beside it. "
    "Numbers add up to nothing in particular. The ceiling fan turns. Dust drifts. "
)


def build_haystack(context_tokens: int) -> str:
    repeats = context_tokens // 24 + 1
    return (FILLER * repeats).strip()


def insert_needle(haystack: str, needle: str, depth_frac: float) -> str:
    n = len(haystack)
    pos = int(depth_frac * n)
    while pos < n and haystack[pos] != '.':
        pos += 1
    if pos < n:
        pos += 2
    return haystack[:pos] + needle + " " + haystack[pos:]


def make_needle() -> tuple[str, str]:
    secret = "".join(random.choices(string.digits, k=6))
    return f"The magic number is {secret}. Remember it: {secret}. ", secret


def query(url: str, model: str, context: str, question: str, max_tokens: int = 80) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You answer questions about provided text accurately and tersely."},
            {"role": "user", "content": context + "\n\n" + question},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    elapsed = time.perf_counter() - start
    txt = body["choices"][0]["message"]["content"]
    return txt, elapsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="qwen3.6-27b")
    p.add_argument("--context-tokens", type=int, default=4000)
    p.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    haystack = build_haystack(args.context_tokens)
    print(f"[niah] haystack ~{args.context_tokens} tokens, {len(haystack)} chars", file=sys.stderr)

    results: list[dict] = []
    for depth in args.depths:
        needle_sentence, secret = make_needle()
        contaminated = insert_needle(haystack, needle_sentence, depth)
        question = "Question: What is the magic number? Answer with only the digits, nothing else."
        try:
            text, elapsed = query(args.url, args.model, contaminated, question)
            passed = secret in text
            results.append({
                "depth": depth,
                "secret": secret,
                "answer": text.strip()[:200],
                "passed": passed,
                "elapsed_s": round(elapsed, 2),
            })
            status = "PASS" if passed else "FAIL"
            print(f"[niah] depth={depth:.2f} {status} secret={secret} got='{text.strip()[:60]}' elapsed={elapsed:.1f}s", file=sys.stderr)
        except Exception as exc:
            results.append({
                "depth": depth,
                "secret": secret,
                "error": f"{type(exc).__name__}: {exc}",
                "passed": False,
            })
            print(f"[niah] depth={depth:.2f} ERROR {exc}", file=sys.stderr)

    n_pass = sum(1 for r in results if r.get("passed"))
    summary = {
        "config": {
            "url": args.url, "model": args.model,
            "context_tokens": args.context_tokens, "depths": args.depths,
        },
        "n_total": len(results),
        "n_passed": n_pass,
        "pass_rate": n_pass / max(len(results), 1),
        "results": results,
    }
    print(json.dumps({"pass_rate": summary["pass_rate"], "n_passed": n_pass, "n_total": len(results)}))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"[niah] wrote {args.output}", file=sys.stderr)

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
