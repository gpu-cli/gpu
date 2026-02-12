#!/usr/bin/env python3
"""
Install script for InvokeAI with BiRefNet background removal.

This script:
1. Creates InvokeAI directory structure
2. Clones BiRefNet custom node into nodes folder
3. Pre-downloads BiRefNet model (~500MB)

Called by startup.sh. Idempotent - safe to run multiple times.
Dependencies (invokeai, timm, kornia) are installed by startup.sh.
"""

import subprocess
from pathlib import Path


def log(msg: str) -> None:
    """Print with flush for real-time output."""
    print(msg, flush=True)


def print_step(step: int, total: int, message: str) -> None:
    """Print a formatted step message."""
    print(f"\n{'='*60}", flush=True)
    print(f"[{step}/{total}] {message}", flush=True)
    print("=" * 60, flush=True)


SCRIPT_DIR = Path(__file__).parent.absolute()
INVOKEAI_DIR = SCRIPT_DIR / "invokeai"
NODES_DIR = INVOKEAI_DIR / "nodes"
BIREFNET_DIR = NODES_DIR / "invoke_birefnet"


def main() -> None:
    """Main setup function."""
    total_steps = 3

    # Step 1: Create directory structure
    print_step(1, total_steps, "Creating directory structure")
    INVOKEAI_DIR.mkdir(exist_ok=True)
    (INVOKEAI_DIR / "outputs").mkdir(exist_ok=True)
    NODES_DIR.mkdir(exist_ok=True)
    log(f"Created: {INVOKEAI_DIR}")

    # Step 2: Clone BiRefNet custom node
    print_step(2, total_steps, "Installing BiRefNet background removal node")
    if BIREFNET_DIR.exists():
        log("BiRefNet node already installed, updating...")
        subprocess.run(["git", "-C", str(BIREFNET_DIR), "pull"], check=True)
    else:
        log("Cloning BiRefNet node...")
        subprocess.run(
            ["git", "clone", "https://github.com/veeliks/invoke_birefnet.git", str(BIREFNET_DIR)],
            check=True,
        )
    log("BiRefNet node installed")

    # Step 3: Pre-download BiRefNet model
    print_step(3, total_steps, "Downloading BiRefNet model (~500MB)")
    log("This ensures the model is ready for immediate use...")
    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(
        'ZhengPeng7/BiRefNet',
        trust_remote_code=True
    )
    del model  # Free memory after download
    log("BiRefNet model downloaded successfully")

    # Success
    log("\n" + "=" * 60)
    log("Setup complete! BiRefNet model pre-downloaded and ready to use.")
    log("=" * 60)


if __name__ == "__main__":
    main()
