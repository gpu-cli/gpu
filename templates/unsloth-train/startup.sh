#!/bin/bash
# Unsloth Training Template - Startup Script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1
export HF_HOME="$SCRIPT_DIR/.hf_cache"
export HUGGINGFACE_HUB_CACHE="$SCRIPT_DIR/.hf_cache/hub"
export TRANSFORMERS_CACHE="$SCRIPT_DIR/.hf_cache/transformers"
export WANDB_DIR="$SCRIPT_DIR/logs/wandb"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p logs outputs checkpoints/latest checkpoints/final exports data

TRAIN_PID=""
TB_PID=""

cleanup() {
  echo ""
  echo "Shutting down training workflow..."

  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "Forwarding SIGTERM to training process..."
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" || true
  fi

  if [ -n "$TB_PID" ] && kill -0 "$TB_PID" 2>/dev/null; then
    kill "$TB_PID" 2>/dev/null || true
    wait "$TB_PID" || true
  fi

  echo "Shutdown complete."
  exit 0
}
trap cleanup SIGINT SIGTERM

echo "=== Unsloth Training Template ==="
echo "Working directory: $SCRIPT_DIR"
echo ""

if [ ! -f "./training.json" ]; then
  echo "Error: training.json not found"
  exit 1
fi

TB_ENABLED=$(python - <<'PY'
import json, sys
from pathlib import Path

cfg = json.loads(Path("training.json").read_text())
runtime = cfg.get("runtime", {})
dataset = cfg.get("dataset", {})
model = cfg.get("model", {})

print("Configuration summary:", file=sys.stderr)
print(f"  Run name: {runtime.get('run_name', 'unsloth-run')}", file=sys.stderr)
print(f"  Base model: {model.get('name', '<missing>')}", file=sys.stderr)
print(f"  Dataset source: {dataset.get('source', 'local')}", file=sys.stderr)
print(f"  Dataset path: {dataset.get('path', '<none>')}", file=sys.stderr)
print(f"  Output dir: {runtime.get('output_dir', 'outputs/current')}", file=sys.stderr)

tb = runtime.get("tensorboard", {}).get("enabled", True)
print(f"  TensorBoard: {tb}", file=sys.stderr)
print("true" if tb else "false")
PY
)

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HuggingFace token detected. Gated model access enabled."
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "Weights & Biases token detected."
fi

if [ "$TB_ENABLED" = "true" ]; then
  if curl -sf http://localhost:6006/ >/dev/null 2>&1; then
    echo "TensorBoard already running on port 6006 (reusing existing server)"
  else
    echo "Starting TensorBoard on port 6006..."
    tensorboard --logdir "$SCRIPT_DIR/logs" --host 0.0.0.0 --port 6006 >/tmp/unsloth-tensorboard.log 2>&1 &
    TB_PID=$!
  fi
fi

RESUME_ARGS=()
if [ -d "checkpoints/latest" ] && [ -n "$(ls -A checkpoints/latest 2>/dev/null)" ]; then
  echo "Found existing checkpoint in checkpoints/latest - resuming from latest state"
  RESUME_ARGS+=(--resume-from checkpoints/latest)
else
  echo "No existing checkpoint detected - starting a fresh training run"
fi

echo ""
echo "Launching training..."
echo "Command: python train.py --config training.json ${RESUME_ARGS[*]}"
echo ""

python train.py --config training.json "${RESUME_ARGS[@]}" 2>&1 | tee -a logs/train.log &
TRAIN_PID=$!

wait "$TRAIN_PID"
TRAIN_EXIT=$?

if [ "$TRAIN_EXIT" -ne 0 ]; then
  echo "Training exited with code $TRAIN_EXIT"
  exit "$TRAIN_EXIT"
fi

echo ""
echo "========================================"
echo "       UNSLOTH TRAINING COMPLETE"
echo "========================================"
echo ""
echo "  Logs:         logs/train.log"
echo "  Latest:       checkpoints/latest"
echo "  Final:        checkpoints/final"
echo "  Outputs:      outputs/"
echo "  Exports:      exports/"
echo "  TensorBoard:  http://localhost:6006"
echo ""
echo "To export merged weights later:"
echo "  gpu run python merge_and_export.py --config training.json"
echo ""
echo "========================================"

while true; do
  sleep 30
done
