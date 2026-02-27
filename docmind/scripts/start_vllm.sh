# ...existing code...

MODEL="${MODEL:-hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama3.1-8b-awq}"

# giảm lỗi phân mảnh bộ nhớ CUDA
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "🚀 Starting vLLM server..."
echo "   Model: $MODEL"
echo "   Host:  $HOST:$PORT"
echo "   GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "   Max Model Len: $MAX_MODEL_LEN"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --quantization awq \
  --dtype float16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enforce-eager \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME"

# ...existing code...