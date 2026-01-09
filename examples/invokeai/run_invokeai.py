#!/usr/bin/env python3
"""
Run script for InvokeAI server.

Usage:
    gpu run --publish 9090:9090 python run_invokeai.py
    gpu run --publish 9090:9090 python run_invokeai.py --port 8080
"""

import argparse
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run InvokeAI server")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=9090, help="Listen port")
    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent.absolute()
    invokeai_root = script_dir / "invokeai"

    # Create directories
    invokeai_root.mkdir(exist_ok=True)
    (invokeai_root / "outputs").mkdir(exist_ok=True)

    # Set environment variables for InvokeAI
    # Note: invokeai-web uses env vars, not CLI args for host/port
    os.environ["INVOKEAI_ROOT"] = str(invokeai_root)
    os.environ["INVOKEAI_HOST"] = args.host
    os.environ["INVOKEAI_PORT"] = str(args.port)

    print(f"INVOKEAI_ROOT: {invokeai_root}")
    print(f"Starting InvokeAI on {args.host}:{args.port}")

    # Run InvokeAI
    subprocess.run(["invokeai-web", "--root", str(invokeai_root)], check=True)


if __name__ == "__main__":
    main()
