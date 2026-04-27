#!/usr/bin/env bash
# Qwen3.6-27B "Flawless Local" recipe — RTX 4090, 24 GB VRAM, vLLM 0.19.1
#
# Targets: TTFT <500ms, ≥3000 tok/s aggregate, 16-32 concurrent agents.
# KV cache: native FP8 (well-tested). TurboQuantDC integration is documented
# as a follow-up — see docs/code_review/2026-04-27/CODE_REVIEW_2026-04-27.md.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODEL="${MODEL:-./models/Qwen3.6-27B-AWQ-INT4}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
# Empirical: model loads at 18.17 GiB. Other GPU residents take ~3.9 GiB
# (gnome-remote, prod, colleague, dhawal uvicorn — all UNTOUCHABLE).
# Free GPU is ~19.6 GiB → 0.83 of 23.51 = 19.51 GiB ≤ free.
# Leaves ~1.3 GiB after weights for KV + activations (very tight).
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.83}"

mkdir -p logs/2026-04-27

echo "[$(date)] Starting vLLM server: $MODEL"
echo "  port=$PORT max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS gpu_mem_util=$GPU_MEM_UTIL"

# --enforce-eager: skip CUDA graph capture (saves ~2 GiB at startup; throughput cost ~10-15%)
# --language-model-only: skip vision encoder
# --no-enable-prefix-caching: prefix caching adds memory; disable for tight VRAM budget
exec env PYTORCH_ALLOC_CONF=expandable_segments:True \
    .venv-vllm/bin/vllm serve "$MODEL" \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enforce-eager \
    --language-model-only \
    --port "$PORT" \
    --host 127.0.0.1 \
    --served-model-name qwen3.6-27b
