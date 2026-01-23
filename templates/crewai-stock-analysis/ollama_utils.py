"""Ollama server management utilities"""
import subprocess
import sys
import time
import requests


def log(msg: str) -> None:
    """Print with flush for real-time output."""
    print(msg, flush=True)


def get_vram_gb() -> int:
    """Detect available GPU VRAM in GB"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        # Memory is reported in MiB, convert to GB
        # Only take first GPU if multiple
        mib = int(result.stdout.strip().split("\n")[0])
        gb = mib // 1024
        return gb
    except (subprocess.SubprocessError, FileNotFoundError, ValueError, IndexError) as e:
        log(f"   ⚠️  Could not detect VRAM: {e}")
        return 24  # Default assumption


def select_model(vram_gb: int) -> str:
    """Select appropriate model based on available VRAM

    Model sizes on disk:
    - qwen2.5:7b  = ~4.5GB
    - qwen2.5:14b = ~9GB
    - qwen2.5:32b = ~19GB

    We use 14b as default for better compatibility (fits on 24GB GPUs with room to spare).
    Only use 32b for GPUs with 48GB+ VRAM.
    """
    if vram_gb >= 48:
        return "qwen2.5:32b"  # 19GB model, excellent reasoning
    elif vram_gb >= 16:
        return "qwen2.5:14b"  # 9GB model, good balance
    else:
        return "qwen2.5:7b"   # 4.5GB model, basic capability


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
        log("   ✓ Ollama server already running")
    except requests.exceptions.RequestException:
        log("   → Starting Ollama server...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not wait_for_ollama():
            log("   ❌ Failed to start Ollama server")
            raise RuntimeError("Failed to start Ollama server")
        log("   ✓ Ollama server started")

    # Pull model - this streams progress automatically
    log(f"   → Downloading model: {model}")
    log("     (This may take 5-10 minutes on first run...)")

    # Run with inherited stdout/stderr so progress shows
    result = subprocess.run(
        ["ollama", "pull", model],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if result.returncode != 0:
        log(f"   ❌ Failed to pull model {model}")
        raise RuntimeError(f"Failed to pull model {model}")

    log(f"   ✓ Model {model} ready")

    return f"ollama/{model}"
