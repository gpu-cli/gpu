#!/bin/bash
# vLLM Models Template - Startup Script
# Starts vLLM server with configured model and launches Web UI
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Process IDs for cleanup
VLLM_PID=""
WEB_PID=""

# Graceful shutdown handler
cleanup() {
  echo ""
  echo "Shutting down..."
  [ -n "$WEB_PID" ] && kill "$WEB_PID" 2>/dev/null
  [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null
  wait
  echo "Shutdown complete."
  exit 0
}
trap cleanup SIGTERM SIGINT

echo "=== vLLM Models Template ==="
echo "Working directory: $SCRIPT_DIR"
echo ""

# Read configuration from models.json
if [ ! -f "./models.json" ]; then
  echo "Error: models.json not found"
  exit 1
fi

MODEL=$(jq -r '.model' ./models.json)
GPU_MEM=$(jq -r '.vllm_args.gpu_memory_utilization // 0.9' ./models.json)
MAX_LEN=$(jq -r '.vllm_args.max_model_len // 32768' ./models.json)
TP_SIZE=$(jq -r '.vllm_args.tensor_parallel_size // 1' ./models.json)

if [ -z "$MODEL" ] || [ "$MODEL" = "null" ]; then
  echo "Error: No model specified in models.json"
  exit 1
fi

echo "Configuration:"
echo "  Model: $MODEL"
echo "  GPU Memory Utilization: $GPU_MEM"
echo "  Max Model Length: $MAX_LEN"
echo "  Tensor Parallel Size: $TP_SIZE"
echo ""

# Build vLLM command arguments
VLLM_ARGS=(
  "--model" "$MODEL"
  "--gpu-memory-utilization" "$GPU_MEM"
  "--max-model-len" "$MAX_LEN"
  "--tensor-parallel-size" "$TP_SIZE"
  "--host" "0.0.0.0"
  "--port" "8000"
)

# Add HuggingFace token if available (for gated models)
if [ -n "$HF_TOKEN" ]; then
  echo "HuggingFace token detected, enabling gated model access"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Start vLLM server in background
echo "Starting vLLM server..."
echo "Command: python -m vllm.entrypoints.openai.api_server ${VLLM_ARGS[*]}"
echo ""
python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" &
VLLM_PID=$!

# Wait for vLLM to be ready (model loading can take 30-120+ seconds)
echo "Waiting for vLLM API (this may take a few minutes for large models)..."
TIMEOUT=180
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
  if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "vLLM is ready!"
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "Error: vLLM process died unexpectedly"
    exit 1
  fi
  sleep 3
  ELAPSED=$((ELAPSED + 3))
  echo "  Still loading... ($ELAPSED/${TIMEOUT}s)"
done

if [ $ELAPSED -ge $TIMEOUT ]; then
  echo "Error: vLLM failed to start within ${TIMEOUT} seconds"
  exit 1
fi

# Copy models.json to ui/ so the web server can serve it
echo ""
echo "Starting Web UI on port 8080..."
cp -f ./models.json ./ui/models.json 2>/dev/null || true
cd ui && python -m http.server 8080 --bind 0.0.0.0 &
WEB_PID=$!
cd "$SCRIPT_DIR"

echo ""
echo "========================================"
echo "         vLLM MODELS READY"
echo "========================================"
echo ""
echo "  Model:     $MODEL"
echo "  vLLM API:  http://localhost:8000"
echo "  Web UI:    http://localhost:8080"
echo ""
echo "  API Examples:"
echo "    # List models"
echo "    curl http://localhost:8000/v1/models"
echo ""
echo "    # Chat (OpenAI-compatible)"
echo "    curl http://localhost:8000/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'"
echo ""
echo "  To change models:"
echo "    1. Edit models.json"
echo "    2. Run: gpu restart"
echo ""
echo "========================================"

# Wait for vLLM (main process) - keeps container running
wait $VLLM_PID
