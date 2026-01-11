#!/usr/bin/env python3
"""
Start InvokeAI server with BiRefNet background removal.

Run setup.py first to install InvokeAI and the BiRefNet node.
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
INVOKEAI_DIR = SCRIPT_DIR / "invokeai"
BIREFNET_DIR = INVOKEAI_DIR / "nodes" / "invoke_birefnet"


def check_setup() -> bool:
    """Check if setup has been completed."""
    if not INVOKEAI_DIR.exists():
        print("Error: InvokeAI not installed.")
        print("Please run setup first:")
        print("  gpu run python setup.py")
        return False

    if not BIREFNET_DIR.exists():
        print("Error: BiRefNet node not installed.")
        print("Please run setup first:")
        print("  gpu run python setup.py")
        return False

    return True


def main() -> None:
    """Start InvokeAI server."""
    if not check_setup():
        sys.exit(1)

    print("=" * 60)
    print("Starting InvokeAI with BiRefNet...")
    print("=" * 60)
    print()
    print("BiRefNet model pre-installed for background removal.")
    print("Add SD models via Model Manager if you need image generation.")
    print()
    print("BiRefNet node available under: Image > Remove Background (BiRefNet)")
    print()

    # Set InvokeAI environment
    os.environ["INVOKEAI_ROOT"] = str(INVOKEAI_DIR)
    os.environ["INVOKEAI_HOST"] = "0.0.0.0"
    os.environ["INVOKEAI_PORT"] = "9090"

    subprocess.run(
        ["invokeai-web", "--root", str(INVOKEAI_DIR)],
        check=True,
    )


if __name__ == "__main__":
    main()
