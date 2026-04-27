#!/usr/bin/env bash
# Sweep the Qwen3.6-27B vLLM server across concurrency × input × output combinations.
# Writes one JSON per config to benchmarks/results/qwen_flawless/
# Tail of each summary is appended to logs/2026-04-27/sweep_summary.jsonl

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

URL="${URL:-http://127.0.0.1:8000}"
RESULTS_DIR="${RESULTS_DIR:-benchmarks/results/qwen_flawless}"
SUMMARY_LOG="${SUMMARY_LOG:-logs/2026-04-27/sweep_summary.jsonl}"

mkdir -p "$RESULTS_DIR" "$(dirname "$SUMMARY_LOG")"

# Confirm server is live
if ! curl -s --max-time 5 "$URL/v1/models" | grep -q "qwen3.6-27b"; then
    echo "ERROR: vLLM server not reachable at $URL — start it first." >&2
    exit 2
fi

# Sweep grid. Tight VRAM budget caps concurrency.
CONCURRENCIES="${CONCURRENCIES:-1 2 4}"
INPUT_TOKENS="${INPUT_TOKENS:-512 2048}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128 512}"

echo "[$(date)] starting sweep at $URL"
echo "  CONCURRENCIES=$CONCURRENCIES"
echo "  INPUT_TOKENS=$INPUT_TOKENS"
echo "  OUTPUT_TOKENS=$OUTPUT_TOKENS"

for C in $CONCURRENCIES; do
    for I in $INPUT_TOKENS; do
        for O in $OUTPUT_TOKENS; do
            NAME="c${C}_i${I}_o${O}"
            OUTFILE="$RESULTS_DIR/${NAME}.json"
            echo "[$(date)] >>> $NAME"
            .venv-vllm/bin/python scripts/bench_qwen_flawless.py \
                --url "$URL" \
                --concurrency "$C" \
                --input-tokens "$I" \
                --output-tokens "$O" \
                --num-requests "$((C * 4))" \
                --output "$OUTFILE" \
                --quiet
            python3 -c "
import json
d = json.load(open('$OUTFILE'))['summary']
print(json.dumps({
    'name': '$NAME',
    'c': $C, 'i': $I, 'o': $O,
    'wall_s': round(d['wall_clock_s'], 2),
    'agg_tps': round(d['aggregate_decode_tps'], 1),
    'ttft_p50_ms': round(d['ttft_p50_s']*1000, 0) if d['ttft_p50_s'] else None,
    'decode_p50': round(d['decode_tps_per_stream_p50'], 1) if d['decode_tps_per_stream_p50'] else None,
    'ok': d['n_success'], 'fail': d['n_failed'],
}))
" >> "$SUMMARY_LOG"
        done
    done
done

echo "[$(date)] sweep complete. summary at $SUMMARY_LOG"
echo
echo "=== SWEEP SUMMARY ==="
cat "$SUMMARY_LOG"
