#!/bin/bash
# Unsloth Studio Template - Startup Script
# Launches the Unsloth Studio web UI for no-code LLM fine-tuning
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1
export HF_HOME="$SCRIPT_DIR/.hf_cache"
export HUGGINGFACE_HUB_CACHE="$SCRIPT_DIR/.hf_cache/hub"
export TRANSFORMERS_CACHE="$SCRIPT_DIR/.hf_cache/transformers"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p exports trained_models data_recipes

STUDIO_PID=""

cleanup() {
  echo ""
  echo "Shutting down Unsloth Studio..."

  if [ -n "$STUDIO_PID" ] && kill -0 "$STUDIO_PID" 2>/dev/null; then
    kill -TERM "$STUDIO_PID" 2>/dev/null || true
    wait "$STUDIO_PID" || true
  fi

  echo "Shutdown complete."
  exit 0
}
trap cleanup SIGINT SIGTERM

echo "=== Unsloth Studio Template ==="
echo "Working directory: $SCRIPT_DIR"
echo ""

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HuggingFace token detected. Gated model access and Hub export enabled."
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Check if Studio is already running (from a previous job on this pod)
if curl -sf http://localhost:8000/ > /dev/null 2>&1; then
  echo "Unsloth Studio is already running on port 8000 (reusing existing server)"

  echo ""
  echo "========================================"
  echo "       UNSLOTH STUDIO READY"
  echo "========================================"
  echo ""
  echo "  Studio UI:     http://localhost:8000"
  echo "  Inference API: http://localhost:8001/v1"
  echo ""
  echo "========================================"

  # Keep alive while Studio is running
  while curl -sf http://localhost:8000/ > /dev/null 2>&1; do
    sleep 30
  done
  exit 0
fi

# First-run setup (compiles llama.cpp with CUDA, builds frontend)
# Skips automatically if already done from a previous run on this volume.
echo "Running Unsloth Studio setup (first run may take a few minutes)..."

# Workaround 1: bun install hangs in containers (oven-sh/bun#22846).
# Shim makes bun install fail fast so setup.sh falls back to npm.
cat > /usr/local/bin/bun << 'SHIM'
#!/bin/sh
case "$1" in install) exit 1 ;; *) exit 0 ;; esac
SHIM
chmod +x /usr/local/bin/bun

# Workaround 2: setup.sh requires a venv at ~/.unsloth/studio/ that only
# install.sh creates. Setting a COLAB_* env var triggers Colab mode which
# installs backend deps into system Python instead (no venv needed).
export COLAB_GPU_CLI=1

# Run setup with a progress indicator — some steps (npm install, Python deps)
# run silently via run_quiet and can take 2-5 minutes with no output.
(
  while true; do
    echo "  [setup still running...]"
    sleep 30
  done
) &
PROGRESS_PID=$!

unsloth studio setup || true

kill $PROGRESS_PID 2>/dev/null
wait $PROGRESS_PID 2>/dev/null || true

rm -f /usr/local/bin/bun

# The launch command checks for ~/.unsloth/studio/unsloth_studio/bin/python.
# Since we installed into system Python (Colab mode), create a symlink so
# the launch code finds it and uses system Python to run the server.
mkdir -p "$HOME/.unsloth/studio/unsloth_studio/bin"
ln -sf "$(which python3)" "$HOME/.unsloth/studio/unsloth_studio/bin/python"

echo ""

echo "Launching Unsloth Studio..."
echo ""

# Run Studio in foreground to keep SSH connection alive
unsloth studio -H 0.0.0.0 -p 8000 &
STUDIO_PID=$!

# Wait for Studio to be ready
echo "Waiting for Studio UI to start..."
for i in {1..120}; do
  if curl -sf http://localhost:8000/ > /dev/null 2>&1; then
    echo "Unsloth Studio is ready!"
    break
  fi
  if [ $i -eq 120 ]; then
    echo "Warning: Studio UI did not respond within 120 seconds (it may still be initializing)"
  fi
  sleep 2
done

echo ""
echo "========================================"
echo "       UNSLOTH STUDIO READY"
echo "========================================"
echo ""
echo "  Studio UI:     http://localhost:8000"
echo "  Inference API: http://localhost:8001/v1"
echo ""
echo "  Features:"
echo "    - Fine-tune 500+ models with LoRA/QLoRA"
echo "    - Visual Data Recipes for dataset preparation"
echo "    - Side-by-side model comparison"
echo "    - Export to GGUF, Safetensors, or HuggingFace Hub"
echo ""
echo "========================================"

# Wait for Studio process — keeps container running
wait $STUDIO_PID
