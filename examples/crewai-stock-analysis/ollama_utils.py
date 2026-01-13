"""Ollama server management utilities"""
import subprocess
import time
import requests


def get_vram_gb() -> int:
    """Detect available GPU VRAM in GB"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        # Memory is reported in MiB, convert to GB
        return int(result.stdout.strip().split("\n")[0]) // 1024
    except Exception:
        return 24  # Default assumption


def select_model(vram_gb: int) -> str:
    """Select appropriate model based on available VRAM"""
    if vram_gb >= 40:
        return "qwen2.5:32b"  # 19GB model, excellent reasoning
    return "qwen2.5:14b"  # 9GB model, good balance


def wait_for_ollama(timeout: int = 30) -> bool:
    """Wait for Ollama server to be ready"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False


def ensure_ollama_running(model: str) -> str:
    """Start Ollama server and pull model if needed"""
    # Check if server is already running
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        print("Ollama server already running")
    except requests.exceptions.RequestException:
        print("Starting Ollama server...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not wait_for_ollama():
            raise RuntimeError("Failed to start Ollama server")
        print("Ollama server started")

    # Pull model if not available
    print(f"Ensuring model {model} is available (this may take a few minutes on first run)...")
    result = subprocess.run(["ollama", "pull", model], check=True)
    print(f"Model {model} ready")

    return f"ollama/{model}"
