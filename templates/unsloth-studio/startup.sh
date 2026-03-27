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

# First-run setup: builds frontend (React/Vite) + compiles llama.cpp with CUDA.
# Skips automatically if already done from a previous run on this volume.
#
# Force npm over bun: bun install hangs in containers (oven-sh/bun#22846).
# The shim makes bun fail fast so unsloth's setup.sh falls back to npm.
echo "Running Unsloth Studio setup (first run may take a few minutes)..."

# Shim bun: only fail on `install` (which hangs in containers).
# Other commands (--version, pm cache rm) must succeed or the setup
# script's set -e will abort before reaching the npm fallback.
cat > /usr/local/bin/bun << 'SHIM'
#!/bin/sh
case "$1" in
  install) exit 1 ;;
  *) exit 0 ;;
esac
SHIM
chmod +x /usr/local/bin/bun

unsloth studio update < /dev/null || unsloth studio setup < /dev/null

rm -f /usr/local/bin/bun

echo ""
echo "Launching Unsloth Studio..."
echo ""

# Run Studio in background
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
