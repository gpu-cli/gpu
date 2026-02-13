#!/bin/bash
# CrewAI Stock Analysis - Startup Script
# Starts Ollama server, downloads model, launches web API server
set -e

# Force unbuffered output from all Python processes
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Store models on the workspace volume (persists across pod restarts)
export OLLAMA_MODELS="$SCRIPT_DIR/.ollama/models"
mkdir -p "$OLLAMA_MODELS"

OLLAMA_PID=""
WEB_PID=""

# Graceful shutdown handler
cleanup() {
  echo ""
  echo "Shutting down..."
  [ -n "$WEB_PID" ] && kill "$WEB_PID" 2>/dev/null
  if command -v ollama &>/dev/null; then
    ollama stop 2>/dev/null || true
  fi
  [ -n "$OLLAMA_PID" ] && kill "$OLLAMA_PID" 2>/dev/null
  wait 2>/dev/null
  echo "Shutdown complete."
  exit 0
}
trap cleanup SIGTERM SIGINT

echo "=== CrewAI Stock Analysis ==="
echo "Working directory: $SCRIPT_DIR"
echo ""

# Verify dependencies are available (installed by gpu.jsonc python.pip_global)
echo "Checking dependencies..."
python -c "import crewai; import litellm; import flask" 2>/dev/null || {
    echo "Dependencies not found, installing..."
    if command -v uv > /dev/null 2>&1; then
        uv pip install --system 'crewai[tools]' litellm 'ddgs>=7.0.0' 'beautifulsoup4>=4.12.0' 'requests>=2.31.0' 'pyyaml>=6.0.0' 'flask>=3.0.0' 2>&1
    else
        pip install --ignore-installed 'crewai[tools]' litellm 'ddgs>=7.0.0' 'beautifulsoup4>=4.12.0' 'requests>=2.31.0' 'pyyaml>=6.0.0' 'flask>=3.0.0' 2>&1
    fi
}

# Ensure Ollama is installed (defense-in-depth if env_spec shell step failed)
command -v ollama > /dev/null 2>&1 || {
    echo "Ollama not found, installing..."
    curl -fsSL https://ollama.com/install.sh | sh
}

# Check if Ollama is already running (from previous job on same pod)
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

# Run Ollama init (idempotent - selects model based on VRAM and pulls in foreground)
# Running in foreground ensures continuous output, preventing SSH proxy timeouts
# during large model downloads (e.g., qwen2.5:32b at ~19GB can take 10-30+ minutes)
echo ""
echo "Initializing model..."
python init_ollama.py

# Check if web server is already running (from previous job)
if curl -sf http://localhost:8501/health > /dev/null 2>&1; then
  echo "Web server is already running on port 8501 (reusing existing server)"
  WEB_PID=""
else
  echo ""
  echo "Starting CrewAI web server on port 8501..."
  python web_server.py &
  WEB_PID=$!
fi

echo ""
echo "========================================"
echo "   CREWAI STOCK ANALYSIS READY"
echo "========================================"
echo ""
echo "  Web API:       http://localhost:8501"
echo "  Ollama API:    http://localhost:11434"
echo ""
echo "  Analyze a stock:"
echo "    curl http://localhost:8501/analyze/NVDA"
echo ""
echo "  List reports:"
echo "    curl http://localhost:8501/reports"
echo ""
echo "  Or run directly:"
echo "    gpu run python main.py NVDA"
echo ""
echo "========================================"

# Wait for main process - keeps container running
if [ -n "$OLLAMA_PID" ]; then
  wait $OLLAMA_PID
else
  # If reusing existing Ollama, wait on web server instead
  if [ -n "$WEB_PID" ]; then
    wait $WEB_PID
  else
    # Both servers reused - keep script alive by polling
    while curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
      sleep 30
    done
  fi
fi
