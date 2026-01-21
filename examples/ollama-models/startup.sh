#!/bin/bash
# Ollama Models Template - Startup Script
# Starts Ollama server, pre-pulls configured models, and launches Web UI
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Store models on the workspace volume (has more space than container disk)
export OLLAMA_MODELS="$SCRIPT_DIR/.ollama/models"
mkdir -p "$OLLAMA_MODELS"

# Graceful shutdown handler
cleanup() {
  echo ""
  echo "Shutting down..."
  [ -n "$WEB_PID" ] && kill "$WEB_PID" 2>/dev/null
  [ -n "$OLLAMA_PID" ] && kill "$OLLAMA_PID" 2>/dev/null
  wait
  echo "Shutdown complete."
  exit 0
}
trap cleanup SIGTERM SIGINT

# Validate model name (alphanumeric, colons, dots, hyphens, slashes only)
validate_model_name() {
  local name="$1"
  if [[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9._:/-]*$ ]]; then
    return 0
  fi
  return 1
}

echo "=== Ollama Models Template ==="
echo "Working directory: $SCRIPT_DIR"
echo ""

# Check if Ollama is already running (from previous job)
if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "Ollama is already running (reusing existing server)"
  OLLAMA_PID=""
else
  # Start Ollama server in background
  echo "Starting Ollama server..."
  ollama serve &
  OLLAMA_PID=$!

  # Wait for Ollama to be ready
  echo "Waiting for Ollama API..."
  for i in {1..60}; do
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
      echo "Ollama is ready!"
      break
    fi
    if [ $i -eq 60 ]; then
      echo "Error: Ollama failed to start within 60 seconds"
      exit 1
    fi
    sleep 1
  done
fi

# Pre-pull configured models from models.json
if [ -f "./models.json" ]; then
  echo ""
  echo "Pre-pulling configured models..."

  # Read model names from models.json array
  MODELS=$(jq -r '.models[]' ./models.json 2>/dev/null || echo "")

  if [ -n "$MODELS" ]; then
    while IFS= read -r model; do
      if [ -n "$model" ]; then
        if validate_model_name "$model"; then
          echo "  Pulling: $model"
          ollama pull "$model" || echo "  Warning: Failed to pull $model"
        else
          echo "  Skipping invalid model name: $model"
        fi
      fi
    done <<< "$MODELS"
    echo "Model pre-pull complete!"
  else
    echo "  No models configured in models.json"
  fi
else
  echo "Warning: models.json not found, skipping model pre-pull"
fi

# Start Web UI server (Python's built-in HTTP server)
echo ""
# Copy models.json to ui/ so the web server can serve it
cp -f ./models.json ./ui/models.json 2>/dev/null || true

# Check if Web UI is already running (from previous job)
if curl -sf http://localhost:8080/ > /dev/null 2>&1; then
  echo "Web UI is already running on port 8080 (reusing existing server)"
  WEB_PID=""
else
  echo "Starting Web UI on port 8080..."
  cd ui && python -m http.server 8080 --bind 0.0.0.0 &
  WEB_PID=$!
  cd "$SCRIPT_DIR"
fi

echo ""
echo "========================================"
echo "         OLLAMA MODELS READY"
echo "========================================"
echo ""
echo "  Ollama API:  http://localhost:11434"
echo "  Web UI:      http://localhost:8080"
echo ""
echo "  API Examples:"
echo "    # List models"
echo "    curl http://localhost:11434/api/tags"
echo ""
echo "    # Chat (OpenAI-compatible)"
echo "    curl http://localhost:11434/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\":\"glm-4.7-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'"
echo ""
echo "  Pull more models:"
echo "    ollama pull codellama:7b"
echo "    ollama pull deepseek-r1:7b"
echo ""
echo "========================================"

# Wait for Ollama (main process) - keeps container running
wait $OLLAMA_PID
