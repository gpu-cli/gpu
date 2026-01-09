#!/usr/bin/env python3
"""
Setup script for InvokeAI.

Installs InvokeAI and creates required directories.

Usage:
    gpu run python setup_invokeai.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Set InvokeAI root directory
    script_dir = Path(__file__).parent.absolute()
    invokeai_root = script_dir / "invokeai"

    # Create data directories
    invokeai_root.mkdir(exist_ok=True)
    (invokeai_root / "outputs").mkdir(exist_ok=True)

    # Install InvokeAI
    print("Installing InvokeAI...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "invokeai[xformers]", "--upgrade"],
        check=True,
    )

    print()
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print()
    print(f"INVOKEAI_ROOT: {invokeai_root}")
    print()
    print("Run InvokeAI with:")
    print("  gpu run --publish 9090:9090 python run_invokeai.py")
    print()


if __name__ == "__main__":
    main()
