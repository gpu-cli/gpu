#!/usr/bin/env python3
"""Setup script for InvokeAI."""

import subprocess
import os

# Set InvokeAI root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
invokeai_root = os.path.join(script_dir, "invokeai")
os.environ["INVOKEAI_ROOT"] = invokeai_root

# Create data directories
os.makedirs(invokeai_root, exist_ok=True)
os.makedirs(os.path.join(invokeai_root, "outputs"), exist_ok=True)

# Install InvokeAI with CUDA support
print("Installing InvokeAI...")
subprocess.run(
    ["pip", "install", "invokeai[xformers]", "--upgrade"],
    check=True,
)

print("\nSetup complete! Run InvokeAI with:")
print("  gpu run -p 9090:9090 python run_invokeai.py")
