#!/usr/bin/env python3
"""Run script for InvokeAI server."""

import sys
import os

# Set InvokeAI root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
invokeai_root = os.path.join(script_dir, "invokeai")
os.environ["INVOKEAI_ROOT"] = invokeai_root

# Ensure directories exist
os.makedirs(invokeai_root, exist_ok=True)
os.makedirs(os.path.join(invokeai_root, "outputs"), exist_ok=True)

print(f"INVOKEAI_ROOT: {invokeai_root}")
print("Starting InvokeAI server...")

# Import and run InvokeAI directly (keeps process alive)
from invokeai.app.run_app import run_app
run_app()
