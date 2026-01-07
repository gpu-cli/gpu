#!/usr/bin/env python3
"""Hello GPU - A minimal test template for GPU CLI."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("Hello from GPU CLI template!")
    print("=" * 60)
    print()

    # Show system info
    print("System Information:")
    print(f"  Python: {sys.version}")
    print(f"  Working directory: {os.getcwd()}")
    print()

    # Check for NVIDIA GPU
    print("GPU Information:")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                print(f"  {line}")
        else:
            print("  nvidia-smi failed (no GPU?)")
    except FileNotFoundError:
        print("  nvidia-smi not found (no NVIDIA driver?)")
    except Exception as e:
        print(f"  Error checking GPU: {e}")

    print()

    # Create output
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    output_file = outputs_dir / "hello.txt"
    output_file.write_text("Hello from GPU CLI!\n\nTemplate test successful.\n")
    print(f"Created output: {output_file}")

    print()
    print("=" * 60)
    print("Template test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
