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
  # Clean up bun shim if we left one behind
  rm -f /usr/local/bin/bun 2>/dev/null || true
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
  echo "  Studio UI:     http://localhost:8000"
  echo "  Inference API: http://localhost:8001/v1"
  # Keep alive while Studio is running
  while curl -sf http://localhost:8000/ > /dev/null 2>&1; do
    sleep 30
  done
  exit 0
fi

# ── First-run setup ──
# unsloth studio setup (setup.sh) does three things:
#   1. Builds the React frontend (needs Node + npm/bun)
#   2. Installs Python backend deps (needs venv OR Colab mode)
#   3. Downloads/builds llama.cpp (needs cmake + CUDA)
#
# Two workarounds are needed:
#
# A) Bun shim: bun install hangs in containers (oven-sh/bun#22846).
#    A shim at /usr/local/bin/bun makes `bun install` fail fast so
#    setup.sh falls back to npm. Other bun commands (--version, pm cache rm)
#    return success so set -euo pipefail doesn't abort prematurely.
#
# B) Colab env var: setup.sh checks for a venv at ~/.unsloth/studio/
#    that only install.sh creates. Without it, setup.sh exits 1.
#    Setting a COLAB_* env var triggers Colab mode, which installs
#    backend deps into system Python instead — exactly what we want
#    since the base image already has torch + CUDA.

echo "Running Unsloth Studio setup (first run may take a few minutes)..."

# Workaround A: bun shim
cat > /usr/local/bin/bun << 'SHIM'
#!/bin/sh
case "$1" in
  install) exit 1 ;;
  *) exit 0 ;;
esac
SHIM
chmod +x /usr/local/bin/bun

# Workaround B: trigger Colab mode for system Python deps
export COLAB_GPU_CLI=1

unsloth studio setup < /dev/null || true

# Clean up
rm -f /usr/local/bin/bun
unset COLAB_GPU_CLI

echo ""
echo "Launching Unsloth Studio..."

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
    echo "Warning: Studio UI did not respond within 240 seconds"
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
echo "========================================"

# Wait for Studio process — keeps container running
wait $STUDIO_PID
