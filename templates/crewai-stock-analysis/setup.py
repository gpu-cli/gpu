#!/usr/bin/env python3
"""Setup script - Initialize Ollama server with model

Run once with: gpu run python setup.py
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"

from ollama_utils import get_vram_gb, select_model, ensure_ollama_running


def main():
    print("=" * 60)
    print("Setting up CrewAI Stock Analysis Server")
    print("=" * 60)

    # Detect GPU and select model
    print("\n[1/3] Detecting GPU...")
    vram = get_vram_gb()
    model = select_model(vram)
    print(f"      VRAM: {vram}GB | Model: {model}")

    # Setup Ollama and download model
    print("\n[2/3] Initializing Ollama...")
    ollama_model = ensure_ollama_running(model)

    # Write model config for main.py to use
    print("\n[3/3] Saving configuration...")
    with open(".ollama_model", "w") as f:
        f.write(ollama_model)

    print("\n" + "=" * 60)
    print("Setup complete! Ollama server is running.")
    print("=" * 60)
    print("\nRun queries with: gpu run python main.py")
    print("Example: gpu run python main.py NVDA")


if __name__ == "__main__":
    main()
