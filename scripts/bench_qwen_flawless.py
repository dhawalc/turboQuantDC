"""Benchmark harness for the Qwen3.6-27B vLLM server on RTX 4090.

Hits an OpenAI-compatible /v1/chat/completions endpoint with concurrent streaming
requests, measures TTFT (time to first token), inter-token latency, per-stream
decode throughput, and aggregate throughput.

Usage:
    .venv-vllm/bin/python scripts/bench_qwen_flawless.py \\
        --url http://127.0.0.1:8000 \\
        --concurrency 16 --input-tokens 1024 --output-tokens 256 \\
        --num-requests 32 \\
        --output benchmarks/results/qwen_flawless/run.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp


FILLER = (
    "tokens flow through transformers like rivers carrying meaning "
    "from question into answer through layers of attention and learned weight "
    "every position attends to every other position with softmax probability "
    "and the cache holds keys and values long after the prompt is read "
)


@dataclass
class RequestResult:
    success: bool
    error: str | None
    ttft_s: float | None
    total_s: float | None
    n_input: int
    n_output: int
    decode_tps: float | None


@dataclass
class RunSummary:
    config: dict
    n_requests: int
    n_success: int
    n_failed: int
    wall_clock_s: float
    aggregate_decode_tps: float
    ttft_p50_s: float | None
    ttft_p95_s: float | None
    decode_tps_per_stream_p50: float | None
    decode_tps_per_stream_p95: float | None
    errors: list[str] = field(default_factory=list)


def build_prompt(target_tokens: int) -> str:
    needed_words = target_tokens + 16
    repeats = needed_words // 32 + 1
    text = (FILLER * repeats).strip()
    return text + "\n\nBased on the text above, write a concise summary in exactly one paragraph."


async def stream_one(
    session: aiohttp.ClientSession, url: str, model: str,
    prompt: str, output_tokens: int,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first_token_t: float | None = None
    n_output = 0
    n_input = 0
    last_chunk_t: float | None = None
    try:
        async with session.post(
            f"{url}/v1/chat/completions", json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(False, f"HTTP {resp.status}: {body[:200]}",
                                     None, time.perf_counter() - start, 0, 0, None)
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if first_token_t is None:
                    choices = obj.get("choices", [])
                    if choices and choices[0].get("delta", {}).get("content"):
                        first_token_t = time.perf_counter()
                usage = obj.get("usage")
                if usage:
                    n_input = int(usage.get("prompt_tokens", n_input))
                    n_output = int(usage.get("completion_tokens", n_output))
                else:
                    choices = obj.get("choices", [])
                    if choices and choices[0].get("delta", {}).get("content"):
                        n_output += 1
                last_chunk_t = time.perf_counter()
        total_s = time.perf_counter() - start
        if first_token_t is None:
            return RequestResult(False, "No content tokens received", None, total_s, n_input, n_output, None)
        ttft_s = first_token_t - start
        decode_window_s = max((last_chunk_t or first_token_t) - first_token_t, 1e-6)
        decode_tps = (n_output - 1) / decode_window_s if n_output > 1 else None
        return RequestResult(True, None, ttft_s, total_s, n_input, n_output, decode_tps)
    except (asyncio.TimeoutError, aiohttp.ClientError, OSError) as exc:
        return RequestResult(False, f"{type(exc).__name__}: {exc}",
                             None, time.perf_counter() - start, 0, 0, None)


async def run_concurrent(
    url: str, model: str, prompt: str,
    output_tokens: int, concurrency: int, n_requests: int,
) -> tuple[list[RequestResult], float]:
    sem = asyncio.Semaphore(concurrency)

    async def worker(session: aiohttp.ClientSession) -> RequestResult:
        async with sem:
            return await stream_one(session, url, model, prompt, output_tokens)

    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        wall_start = time.perf_counter()
        tasks = [asyncio.create_task(worker(session)) for _ in range(n_requests)]
        results = await asyncio.gather(*tasks)
        wall_s = time.perf_counter() - wall_start
    return list(results), wall_s


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs_sorted = sorted(xs)
    return xs_sorted[int(p * (len(xs_sorted) - 1))]


def summarize(results: list[RequestResult], wall_s: float, config: dict) -> RunSummary:
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    aggregate_decode_tps = sum(r.n_output for r in successes) / max(wall_s, 1e-6)
    ttft = [r.ttft_s for r in successes if r.ttft_s is not None]
    decode = [r.decode_tps for r in successes if r.decode_tps is not None]
    return RunSummary(
        config=config,
        n_requests=len(results), n_success=len(successes), n_failed=len(failures),
        wall_clock_s=wall_s, aggregate_decode_tps=aggregate_decode_tps,
        ttft_p50_s=percentile(ttft, 0.5), ttft_p95_s=percentile(ttft, 0.95),
        decode_tps_per_stream_p50=percentile(decode, 0.5),
        decode_tps_per_stream_p95=percentile(decode, 0.95),
        errors=list({r.error for r in failures if r.error})[:5],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="qwen3.6-27b")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--input-tokens", type=int, default=1024)
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--num-requests", type=int, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    n_requests = args.num_requests or max(args.concurrency * 2, 8)
    prompt = build_prompt(args.input_tokens)
    config = {
        "url": args.url, "model": args.model,
        "concurrency": args.concurrency,
        "input_tokens_target": args.input_tokens,
        "output_tokens": args.output_tokens,
        "num_requests": n_requests,
        "prompt_chars": len(prompt),
    }
    if not args.quiet:
        print(f"[bench] {config}", file=sys.stderr)

    results, wall_s = asyncio.run(run_concurrent(
        args.url, args.model, prompt,
        args.output_tokens, args.concurrency, n_requests,
    ))
    summary = summarize(results, wall_s, config)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"summary": asdict(summary), "requests": [asdict(r) for r in results]},
            indent=2,
        ))
        if not args.quiet:
            print(f"[bench] wrote {args.output}", file=sys.stderr)

    if not args.quiet:
        s = summary
        print(json.dumps({
            "concurrency": s.config["concurrency"],
            "input_tokens": s.config["input_tokens_target"],
            "output_tokens": s.config["output_tokens"],
            "n_success": s.n_success, "n_failed": s.n_failed,
            "wall_s": round(s.wall_clock_s, 2),
            "aggregate_decode_tps": round(s.aggregate_decode_tps, 1),
            "ttft_p50_ms": round(s.ttft_p50_s * 1000, 0) if s.ttft_p50_s else None,
            "ttft_p95_ms": round(s.ttft_p95_s * 1000, 0) if s.ttft_p95_s else None,
            "decode_per_stream_p50": round(s.decode_tps_per_stream_p50, 1) if s.decode_tps_per_stream_p50 else None,
            "decode_per_stream_p95": round(s.decode_tps_per_stream_p95, 1) if s.decode_tps_per_stream_p95 else None,
            "errors": s.errors,
        }, indent=2))

    return 0 if summary.n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
