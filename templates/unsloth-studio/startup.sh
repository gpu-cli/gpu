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
  echo "  Studio UI:     http://localhost:8000"
  echo "  Inference API: http://localhost:8001/v1"
  while curl -sf http://localhost:8000/ > /dev/null 2>&1; do
    sleep 30
  done
  exit 0
fi

# ── Resolve unsloth binary ──
# pip3 install creates the binary at /usr/local/bin/unsloth with the correct
# Python shebang. Verify it exists before proceeding.
if ! command -v unsloth &>/dev/null; then
  echo "ERROR: 'unsloth' command not found."
  echo "Checking Python can import unsloth_cli..."
  if python3 -c "from unsloth_cli import app; print('OK')" 2>/dev/null; then
    echo "Module found — creating wrapper..."
    python3 -c "
import sys, os
wrapper = '#!' + sys.executable + '\nfrom unsloth_cli import app\napp()\n'
with open('/usr/local/bin/unsloth', 'w') as f:
    f.write(wrapper)
os.chmod('/usr/local/bin/unsloth', 0o755)
"
  else
    echo "FATAL: unsloth package not installed correctly."
    exit 1
  fi
fi

echo "Using: $(which unsloth)"

# ── First-run setup ──
# unsloth studio setup (setup.sh) builds the React frontend, installs
# Python backend deps, and downloads/builds llama.cpp.
#
# Two workarounds needed for containerized environments:
#
# 1. Bun shim: bun install hangs in containers (oven-sh/bun#22846).
#    Shim makes `bun install` fail fast → npm fallback.
#    Other bun commands succeed so set -euo pipefail doesn't abort.
#
# 2. COLAB_ env var: setup.sh requires a venv at ~/.unsloth/studio/
#    (only created by install.sh which uses Python 3.13 → xformers fails).
#    Setting any COLAB_* env var triggers Colab mode → installs backend
#    deps into system Python instead. Our base image already has torch+CUDA.
echo "Running Unsloth Studio setup (first run may take a few minutes)..."

cat > /usr/local/bin/bun << 'SHIM'
#!/bin/sh
case "$1" in
  install) exit 1 ;;
  *) exit 0 ;;
esac
SHIM
chmod +x /usr/local/bin/bun

export COLAB_GPU_CLI=1
unsloth studio setup < /dev/null || true
rm -f /usr/local/bin/bun
unset COLAB_GPU_CLI

echo ""
echo "Launching Unsloth Studio..."

unsloth studio -H 0.0.0.0 -p 8000 &
STUDIO_PID=$!

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

wait $STUDIO_PID
