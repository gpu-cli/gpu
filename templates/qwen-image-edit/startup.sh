#!/bin/bash
# ComfyUI + Qwen Image Edit - Startup Script
# Runs setup (idempotently) and starts ComfyUI server
set -e

# Force unbuffered output from all Python processes
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMFYUI_PID=""

cleanup() {
  echo ""
  echo "Shutting down..."
  [ -n "$COMFYUI_PID" ] && kill "$COMFYUI_PID" 2>/dev/null
  wait
  echo "Shutdown complete."
  exit 0
}
trap cleanup SIGTERM SIGINT

echo "=== ComfyUI + Qwen Image Edit ==="
echo "Working directory: $SCRIPT_DIR"
echo ""

# Ensure setup dependencies are available
# (env_spec shell steps may be skipped due to only_once markers from a prior pod session)
echo "Installing dependencies..."
pip install -q tqdm requests 'huggingface_hub[cli]'

# Run setup (idempotent - skips if already done)
echo "Checking setup..."
python install_comfyui.py

echo ""

# Check if ComfyUI is already running (from previous job on same pod)
if curl -sf http://localhost:8188/system_stats > /dev/null 2>&1; then
  echo "ComfyUI is already running (reusing existing server)"
  echo ""
  echo "========================================"
  echo "   COMFYUI + QWEN IMAGE EDIT READY"
  echo "========================================"
  echo ""
  echo "  ComfyUI:  http://localhost:8188"
  echo ""
  echo "  Your workflow is pre-loaded in:"
  echo "    Workflows menu > image_qwen_image_edit"
  echo ""
  echo "========================================"
  COMFYUI_PID=""
  # Keep the script running to maintain the job
  while curl -sf http://localhost:8188/system_stats > /dev/null 2>&1; do
    sleep 30
  done
else
  echo "Starting ComfyUI server..."
  cd ComfyUI
  python main.py --listen 0.0.0.0 &
  COMFYUI_PID=$!
  cd "$SCRIPT_DIR"

  # Wait for ComfyUI to be ready
  echo "Waiting for ComfyUI to load models..."
  TIMEOUT=600
  ELAPSED=0
  while [ $ELAPSED -lt $TIMEOUT ]; do
    if curl -sf http://localhost:8188/system_stats > /dev/null 2>&1; then
      echo "ComfyUI is ready!"
      break
    fi
    if ! kill -0 $COMFYUI_PID 2>/dev/null; then
      echo "Error: ComfyUI process died unexpectedly"
      exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  Still loading... ($ELAPSED/${TIMEOUT}s)"
  done

  if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "Error: ComfyUI failed to start within ${TIMEOUT} seconds"
    exit 1
  fi
fi

echo ""
echo "========================================"
echo "   COMFYUI + QWEN IMAGE EDIT READY"
echo "========================================"
echo ""
echo "  ComfyUI:  http://localhost:8188"
echo ""
echo "  Your workflow is pre-loaded in:"
echo "    Workflows menu > image_qwen_image_edit"
echo ""
echo "========================================"

# Wait for ComfyUI (main process) - keeps container running
wait $COMFYUI_PID
