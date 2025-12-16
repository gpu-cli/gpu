#!/usr/bin/env python3
"""Setup script for ComfyUI with essential custom nodes."""

import subprocess
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
comfyui_dir = os.path.join(script_dir, "ComfyUI")
custom_nodes_dir = os.path.join(comfyui_dir, "custom_nodes")

# Clone ComfyUI
if not os.path.exists(comfyui_dir):
    print("Cloning ComfyUI...")
    subprocess.run(
        ["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", comfyui_dir],
        check=True,
    )

# Install dependencies
print("Installing dependencies...")
subprocess.run(
    ["pip", "install", "-r", os.path.join(comfyui_dir, "requirements.txt")],
    check=True,
)
subprocess.run(["pip", "install", "--force-reinstall", "torchaudio"], check=True)

# Essential custom nodes
CUSTOM_NODES = [
    ("ComfyUI-Manager", "https://github.com/ltdrdata/ComfyUI-Manager.git"),
    ("ComfyUI-Impact-Pack", "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git"),
    ("was-node-suite-comfyui", "https://github.com/ltdrdata/was-node-suite-comfyui.git"),
    ("ComfyUI_IPAdapter_plus", "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git"),
    ("ComfyUI-Advanced-ControlNet", "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git"),
    ("ComfyUI_Comfyroll_CustomNodes", "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git"),
    ("sdxl_prompt_styler", "https://github.com/twri/sdxl_prompt_styler.git"),
]

# Install custom nodes
os.makedirs(custom_nodes_dir, exist_ok=True)
print("Installing custom nodes...")

for name, url in CUSTOM_NODES:
    node_path = os.path.join(custom_nodes_dir, name)
    if os.path.exists(node_path):
        print(f"  {name} (already installed)")
        continue

    print(f"  {name}...")
    subprocess.run(["git", "clone", url, node_path], check=True)

    node_requirements = os.path.join(node_path, "requirements.txt")
    if os.path.exists(node_requirements):
        subprocess.run(["pip", "install", "-r", node_requirements], check=True)

print("\nSetup complete! Run ComfyUI with:")
print("  gpu run --publish 8188:8188 python ComfyUI/main.py --listen 0.0.0.0")
