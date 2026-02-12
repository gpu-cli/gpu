#!/bin/bash
# InvokeAI + BiRefNet Background Removal - Startup Script
# Runs setup (idempotently) and starts InvokeAI server
set -e

# Force unbuffered output from all Python processes (InvokeAI, pip, setup script)
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INVOKEAI_PID=""

cleanup() {
  echo ""
  echo "Shutting down..."
  [ -n "$INVOKEAI_PID" ] && kill "$INVOKEAI_PID" 2>/dev/null
  wait
  echo "Shutdown complete."
  exit 0
}
trap cleanup SIGTERM SIGINT

# Ensure dependencies are available
# (env_spec shell steps may be skipped due to only_once markers from a prior pod session)
# NOTE: Do NOT use -q flag - output keeps SSH connection alive during long installs
echo "Installing dependencies..."
pip install 'invokeai[xformers]' timm kornia 2>&1

# Run setup (idempotent - skips if already done)
echo "Checking setup..."
python install_invokeai.py

# Check if InvokeAI is already running (from previous job on same pod)
if curl -sf http://localhost:9090/api/v1/app/version > /dev/null 2>&1; then
  echo "InvokeAI is already running (reusing existing server)"
  INVOKEAI_PID=""
  # Keep the script running to maintain the job
  while curl -sf http://localhost:9090/api/v1/app/version > /dev/null 2>&1; do
    sleep 30
  done
else
  echo "Starting InvokeAI server..."
  export INVOKEAI_HOST=0.0.0.0
  export INVOKEAI_PORT=9090
  stdbuf -oL invokeai-web --root ./invokeai 2>&1 &
  INVOKEAI_PID=$!

  # Readiness hook (defined in gpu.jsonc) handles waiting for InvokeAI to become ready.
  # This script just needs to keep running while InvokeAI loads.
  echo "InvokeAI loading... (readiness hook will detect when ready)"

  # Wait for InvokeAI (main process) - keeps container running
  wait $INVOKEAI_PID
fi
