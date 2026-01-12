#!/usr/bin/env python3
"""
Setup script for InvokeAI with BiRefNet background removal.

This script:
1. Creates InvokeAI directory structure
2. Installs InvokeAI with xformers
3. Clones BiRefNet custom node into nodes folder
4. Installs BiRefNet dependencies (timm, kornia)
5. Pre-downloads BiRefNet model (~500MB)

Run once before using run.py. Setup takes ~5-10 minutes.
"""

import subprocess
import sys
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
    total_steps = 5

    # Step 1: Create directory structure
    print_step(1, total_steps, "Creating directory structure")
    INVOKEAI_DIR.mkdir(exist_ok=True)
    (INVOKEAI_DIR / "outputs").mkdir(exist_ok=True)
    NODES_DIR.mkdir(exist_ok=True)
    log(f"Created: {INVOKEAI_DIR}")

    # Step 2: Install InvokeAI
    print_step(2, total_steps, "Installing InvokeAI with xformers")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "invokeai[xformers]", "--upgrade", "-q"],
        check=True,
    )
    log("InvokeAI installed successfully")

    # Step 3: Clone BiRefNet custom node
    print_step(3, total_steps, "Installing BiRefNet background removal node")
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

    # Step 4: Install BiRefNet dependencies
    print_step(4, total_steps, "Installing BiRefNet dependencies")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "timm", "kornia", "-q"],
        check=True,
    )
    log("Dependencies installed: timm, kornia")

    # Step 5: Pre-download BiRefNet model
    print_step(5, total_steps, "Downloading BiRefNet model (~500MB)")
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
    log("Setup complete!")
    log("=" * 60)
    log("\nBiRefNet model pre-downloaded and ready to use.")
    log("\nNext steps:")
    log("  1. Start InvokeAI:")
    log("     gpu run --publish 9090:9090 python run.py")
    log("  2. Open the URL shown in terminal")
    log("  3. The BiRefNet node is available under 'Remove Background (BiRefNet)'")
    log("\nNote: Add SD models via Model Manager if you need image generation.")


if __name__ == "__main__":
    main()
