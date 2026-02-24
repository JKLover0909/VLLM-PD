#!/bin/bash
# ================================================
# Script khởi động vLLM server cho DocMind
# Model: meta-llama/Meta-Llama-3.1-8B-Instruct (FP16)
# ================================================

MODEL=${MODEL:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8000"}
GPU_MEM=${GPU_MEM:-"0.90"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"8192"}

echo "🚀 Starting vLLM server..."
echo "   Model: $MODEL"
echo "   Host:  $HOST:$PORT"
echo "   GPU Memory Utilization: $GPU_MEM"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype float16 \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_MODEL_LEN" \
    --served-model-name "llama3.1-8b" \
    --trust-remote-code

# Usage:
#   bash scripts/start_vllm.sh
#   MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct" PORT=8000 bash scripts/start_vllm.sh
