#!/usr/bin/env python3
"""
Start ComfyUI server.

Run install_comfyui.py first to install ComfyUI and download models,
or use 'gpu use .' which handles setup automatically via startup.sh.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
COMFYUI_DIR = SCRIPT_DIR / "ComfyUI"


def check_setup() -> bool:
    """Check if setup has been completed."""
    main_py = COMFYUI_DIR / "main.py"

    if not main_py.exists():
        print("Error: ComfyUI not installed.")
        print(f"Expected: {main_py}")
        print()
        print("Run 'gpu use .' to start the automated setup,")
        print("or run manually: gpu run python install_comfyui.py")
        return False

    models_dir = COMFYUI_DIR / "models" / "diffusion_models"
    if not models_dir.exists() or not any(models_dir.glob("*.safetensors")):
        print("Error: No models found in ComfyUI/models/diffusion_models/")
        print("Run 'gpu use .' to start the automated setup,")
        print("or run manually: gpu run python install_comfyui.py")
        return False

    return True


def main() -> None:
    """Start ComfyUI server."""
    if not check_setup():
        sys.exit(1)

    print("=" * 60)
    print("Starting ComfyUI...")
    print("=" * 60)
    print()
    print("Once started, access ComfyUI at the URL shown below.")
    print("Your workflow is available in: Workflows menu")
    print()

    subprocess.run(
        [sys.executable, "main.py", "--listen", "0.0.0.0"],
        cwd=COMFYUI_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
