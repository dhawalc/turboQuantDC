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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
# Iteration history (RTX 4090, ~3.3 GiB used by untouchable other processes):
#   0.85 → fails free-memory check (19.6 free vs 19.98 needed)
#   0.83 + FLASHINFER → boots, OOMs on prefill workspace alloc (394 MiB short)
#   0.78 + CPU offload 2 GiB → fails: AssertionError vLLM hybrid + offload
#     incompatibility (https://github.com/vllm-project/vllm/pull/18298)
#   0.83 + 64 MiB FlashInfer ws + max-len 2K + n_seqs 1 → BOOTS, serves short
#     prompts at ~21 tok/s, OOMs at 1024 input tokens (memory just too tight)
# This iteration: max-model-len 1024 — should comfortably serve short
# requests. Honest cap given hardware constraint.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.83}"
FLASHINFER_WORKSPACE_BYTES="${FLASHINFER_WORKSPACE_BYTES:-67108864}"  # 64 MiB

mkdir -p logs/2026-04-27

echo "[$(date)] Starting vLLM server: $MODEL"
echo "  port=$PORT max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS gpu_mem_util=$GPU_MEM_UTIL flashinfer_workspace=${FLASHINFER_WORKSPACE_BYTES}B"

# --enforce-eager: skip CUDA graph capture (saves ~2 GiB; throughput cost ~10-15%)
# --language-model-only: skip vision encoder
# VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE: shrink FlashInfer prefill workspace
exec env PYTORCH_ALLOC_CONF=expandable_segments:True \
         VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="$FLASHINFER_WORKSPACE_BYTES" \
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
