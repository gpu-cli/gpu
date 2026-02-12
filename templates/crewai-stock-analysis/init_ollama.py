#!/usr/bin/env python3
"""Initialize Ollama server with model

Idempotent setup - verifies Ollama server and pulls the appropriate model.
Called automatically by startup.sh on each pod start.
"""
import os

os.environ["PYTHONUNBUFFERED"] = "1"

from ollama_utils import get_vram_gb, select_model, ensure_ollama_running


def main():
    print("=" * 60, flush=True)
    print("Setting up CrewAI Stock Analysis", flush=True)
    print("=" * 60, flush=True)

    # Detect GPU and select model
    print("\n[1/3] Detecting GPU...", flush=True)
    vram = get_vram_gb()
    model = select_model(vram)
    print(f"      VRAM: {vram}GB | Model: {model}", flush=True)

    # Setup Ollama and download model
    print("\n[2/3] Downloading model (this may take 5-10 minutes)...", flush=True)
    ollama_model = ensure_ollama_running(model)

    # Write model config for web server to use
    print("\n[3/3] Saving configuration...", flush=True)
    with open(".ollama_model", "w") as f:
        f.write(ollama_model)

    print("\n" + "=" * 60, flush=True)
    print("Setup complete! Ollama server is running.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
