#!/bin/bash
# SAM 3 Video Segmentation - Startup Script
# Installs SAM 3 from Meta's official repo, pre-downloads model, and launches Gradio UI
set -e

# Force unbuffered output from all Python processes
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Store HuggingFace model cache on workspace volume (persists across sessions)
export HF_HOME="$SCRIPT_DIR/.hf_cache"
mkdir -p "$HF_HOME"

# Create outputs directory for annotated videos/images
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

echo "=== SAM 3 Video Segmentation ==="
echo "Working directory: $SCRIPT_DIR"
echo ""

# Configure HuggingFace token for gated model access
if [ -n "$HF_TOKEN" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  # Store token in HF cache so huggingface_hub picks it up reliably
  python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)" 2>/dev/null || true
  echo "HF token configured for SAM 3 model download"
else
  echo "WARNING: HF_TOKEN not set. SAM 3 is a gated model and requires authentication."
  echo "Set your token: gpu auth add hf"
  echo "Accept the license: https://huggingface.co/facebook/sam3"
fi

# Ensure dependencies are available
# (env_spec shell steps may be skipped due to only_once markers from a prior pod session)
# NOTE: Do NOT use -q flag - output keeps SSH connection alive during long installs
if ! python -c "import sam3, gradio, cv2, einops, pycocotools, psutil, decord" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install git+https://github.com/facebookresearch/sam3.git gradio opencv-python-headless "imageio[pyav]" einops pycocotools psutil decord 2>&1
else
  echo "Dependencies already installed, skipping pip install."
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
  echo "Starting SAM 3 Video Segmentation UI..."
  echo "Model will be downloaded on first run (~3.4GB). This may take a few minutes."
  echo ""
  stdbuf -oL python server.py 2>&1 &
  GRADIO_PID=$!

  # Readiness hook (defined in gpu.jsonc) handles waiting for Gradio to become ready.
  echo "Loading SAM 3.1 model... (readiness hook will detect when ready)"

  # Wait for Gradio (main process) - keeps container running
  wait $GRADIO_PID
fi
