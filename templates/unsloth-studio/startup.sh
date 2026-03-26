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

# Resolve unsloth binary — prefer the official venv location, fall back to PATH.
UNSLOTH_BIN=""
UNSLOTH_VENV="$HOME/.unsloth/studio/unsloth_studio"
if [ -x "$UNSLOTH_VENV/bin/unsloth" ]; then
  UNSLOTH_BIN="$UNSLOTH_VENV/bin/unsloth"
elif command -v unsloth &>/dev/null; then
  UNSLOTH_BIN="$(command -v unsloth)"
fi

# First-run: install via the official installer if unsloth isn't available.
# The official install.sh handles headless environments (detects non-TTY),
# pre-installs system deps, sets up its own venv, builds frontend + llama.cpp.
# First run takes 15-35 min; subsequent runs skip if already installed.
if [ -z "$UNSLOTH_BIN" ]; then
  echo "Installing Unsloth Studio (first run, may take 15-35 minutes)..."

  # Pre-install system deps so the installer doesn't prompt for sudo
  apt-get update -qq && apt-get install -y -qq \
    cmake git build-essential libcurl4-openssl-dev pciutils curl nodejs npm \
    2>&1 | tail -1

  # Force npm: bun install hangs in containers (oven-sh/bun#22846).
  # This shim makes bun fail fast so unsloth's setup falls back to npm.
  printf '#!/bin/sh\nexit 1\n' > /usr/local/bin/bun && chmod +x /usr/local/bin/bun

  curl -fsSL https://unsloth.ai/install.sh | sh

  rm -f /usr/local/bin/bun

  # Re-resolve after install
  if [ -x "$UNSLOTH_VENV/bin/unsloth" ]; then
    UNSLOTH_BIN="$UNSLOTH_VENV/bin/unsloth"
  elif command -v unsloth &>/dev/null; then
    UNSLOTH_BIN="$(command -v unsloth)"
  else
    echo "ERROR: unsloth binary not found after installation"
    exit 1
  fi
else
  echo "Unsloth Studio already installed."
fi

echo ""
echo "Launching Unsloth Studio..."
echo ""

# Run Studio in background to keep SSH connection alive
"$UNSLOTH_BIN" studio -H 0.0.0.0 -p 8000 &
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
