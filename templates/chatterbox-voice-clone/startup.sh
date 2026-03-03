#!/bin/bash
# Chatterbox Voice Clone - Startup Script
# Installs dependencies, pre-loads model, and launches Gradio UI
set -e

# Force unbuffered output from all Python processes
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Store HuggingFace model cache on workspace volume (persists across sessions)
export HF_HOME="$SCRIPT_DIR/.hf_cache"
mkdir -p "$HF_HOME"

# Create outputs directory for generated audio
mkdir -p "$SCRIPT_DIR/outputs"

# Process IDs for cleanup
GRADIO_PID=""

# Graceful shutdown handler
cleanup() {
  echo ""
  echo "Shutting down..."
  [ -n "$GRADIO_PID" ] && kill "$GRADIO_PID" 2>/dev/null
  wait
  echo "Shutdown complete."
  exit 0
}
trap cleanup SIGTERM SIGINT

echo "=== Chatterbox Voice Clone ==="
echo "Working directory: $SCRIPT_DIR"
echo ""

# Chatterbox's from_pretrained() uses `token=os.getenv("HF_TOKEN") or True` which
# forces token=True when HF_TOKEN is unset. The model is public (MIT), so no auth
# is needed. Setting HF_TOKEN to a falsy-but-non-None value doesn't help since
# `"" or True == True`. Instead we patch the env so huggingface_hub.snapshot_download
# gets token=False by monkey-patching at the Python level in server.py.
# Alternatively, if user has a real HF token, it will be used automatically.
if [ -z "$HF_TOKEN" ]; then
  echo "NOTE: HF_TOKEN not set. Will attempt public download (model is MIT licensed)."
fi

# Ensure dependencies are available
# (env_spec shell steps may be skipped due to only_once markers from a prior pod session)
# NOTE: Do NOT use -q flag - output keeps SSH connection alive during long installs
if ! python -c "import chatterbox, yt_dlp, gradio" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install chatterbox-tts yt-dlp gradio 2>&1
else
  echo "Dependencies already installed, skipping pip install."
fi

# Fix torchvision incompatibility: chatterbox-tts pins torch==2.6.0 but the RunPod
# base image ships torch 2.4.0 + torchvision built for 2.4.0. After chatterbox upgrades
# torch to 2.6.0, the leftover torchvision (compiled for 2.4.0) has broken C++ ops,
# causing "operator torchvision::nms does not exist" when transformers tries to import
# LlamaModel. The PyPI torchvision wheel also lacks CUDA ops.
# Fix: reinstall torchvision from PyTorch's CUDA wheel index with matching versions.
if ! python -c "import torchvision; assert torchvision.__version__ == '0.21.0'" 2>/dev/null; then
  pip install --force-reinstall torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 --no-deps 2>&1
else
  echo "torchvision 0.21.0 already installed, skipping."
fi

# Check if Gradio is already running (from previous job on same pod)
if curl -sf http://localhost:7860/ > /dev/null 2>&1; then
  echo "Gradio UI is already running (reusing existing server)"
  GRADIO_PID=""
  # Keep the script running to maintain the job
  while curl -sf http://localhost:7860/ > /dev/null 2>&1; do
    sleep 30
  done
else
  echo ""
  echo "Starting Chatterbox Voice Clone UI..."
  echo "Model will be downloaded on first run (~2GB). This may take a few minutes."
  echo ""
  stdbuf -oL python server.py 2>&1 &
  GRADIO_PID=$!

  # Readiness hook (defined in gpu.jsonc) handles waiting for Gradio to become ready.
  # This script just needs to keep running while the model loads.
  echo "Loading Chatterbox-Turbo model... (readiness hook will detect when ready)"

  # Wait for Gradio (main process) - keeps container running
  wait $GRADIO_PID
fi
