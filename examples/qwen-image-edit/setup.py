#!/usr/bin/env python3
"""
Setup script for ComfyUI with workflow preloading.

This script:
1. Reads the workflow URL from gpu.jsonc
2. Downloads the workflow and parses it for required models
3. Installs ComfyUI and required custom nodes
4. Downloads all required models with progress bars
5. Places the workflow in ComfyUI for easy access

Run once before using run.py.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


def log(msg: str) -> None:
    """Print with flush for real-time output."""
    print(msg, flush=True)

# Mapping of model filenames to their HuggingFace locations
# Format: "filename": "repo_id:path_in_repo"
MODEL_SOURCES: dict[str, str] = {
    # Qwen Image Edit diffusion models
    "qwen_image_edit_fp8_e4m3fn.safetensors": "Comfy-Org/Qwen-Image-Edit_ComfyUI:split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors",
    "qwen_image_edit_2511_fp8mixed.safetensors": "Comfy-Org/Qwen-Image-Edit_ComfyUI:split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
    "qwen_image_edit_bf16.safetensors": "Comfy-Org/Qwen-Image-Edit_ComfyUI:split_files/diffusion_models/qwen_image_edit_bf16.safetensors",
    # Qwen text encoders
    "qwen_2.5_vl_7b_fp8_scaled.safetensors": "Comfy-Org/Qwen-Image_ComfyUI:split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    # Qwen VAE
    "qwen_image_vae.safetensors": "Comfy-Org/Qwen-Image_ComfyUI:split_files/vae/qwen_image_vae.safetensors",
    # Qwen LoRAs (from lightx2v repo)
    "Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors": "lightx2v/Qwen-Image-Lightning:Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors",
}

# Custom nodes that may be required by workflows
CUSTOM_NODES: dict[str, str] = {
    "ComfyUI-Manager": "https://github.com/ltdrdata/ComfyUI-Manager.git",
    "Comfyui-QwenEditUtils": "https://github.com/lrzjason/Comfyui-QwenEditUtils.git",
}

SCRIPT_DIR = Path(__file__).parent.absolute()
COMFYUI_DIR = SCRIPT_DIR / "ComfyUI"


def print_step(step: int, total: int, message: str) -> None:
    """Print a formatted step message."""
    print(f"\n{'='*60}", flush=True)
    print(f"[{step}/{total}] {message}", flush=True)
    print("=" * 60, flush=True)


def load_config() -> dict[str, Any]:
    """Load configuration (hardcoded for now)."""
    return {
        "workflow": "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/image_qwen_image_edit.json"
    }


def download_workflow(url: str) -> dict[str, Any]:
    """Download workflow JSON from URL."""
    log(f"Downloading workflow from:\n  {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_models_from_workflow(workflow: dict[str, Any]) -> dict[str, list[str]]:
    """
    Parse ComfyUI workflow JSON to find required models.

    Returns dict with keys: diffusion_models, text_encoders, vae, loras
    """
    models: dict[str, list[str]] = {
        "diffusion_models": [],
        "text_encoders": [],
        "vae": [],
        "loras": [],
    }

    def process_nodes(nodes: list[dict[str, Any]]) -> None:
        """Extract model names from a list of nodes."""
        for node in nodes:
            node_type = node.get("type", "")
            widgets = node.get("widgets_values", [])

            if not widgets or not isinstance(widgets, list):
                continue

            first_widget = widgets[0] if widgets else None
            if not isinstance(first_widget, str):
                continue

            # Map node types to model directories
            if "UNETLoader" in node_type or "DiffusionLoader" in node_type:
                models["diffusion_models"].append(first_widget)
            elif "CLIPLoader" in node_type:
                models["text_encoders"].append(first_widget)
            elif "VAELoader" in node_type:
                models["vae"].append(first_widget)
            elif "LoraLoader" in node_type:
                models["loras"].append(first_widget)

    # Process top-level nodes
    nodes = workflow.get("nodes", [])
    if isinstance(nodes, list):
        process_nodes(nodes)

    # Also check subgraphs (ComfyUI can nest nodes in subgraphs)
    subgraphs = workflow.get("definitions", {}).get("subgraphs", [])
    if isinstance(subgraphs, list):
        for subgraph in subgraphs:
            sub_nodes = subgraph.get("nodes", [])
            if isinstance(sub_nodes, list):
                process_nodes(sub_nodes)

    # Remove duplicates while preserving order
    for key in models:
        models[key] = list(dict.fromkeys(models[key]))

    return models


def clone_comfyui() -> None:
    """Clone ComfyUI repository if not exists."""
    if COMFYUI_DIR.exists():
        log("ComfyUI already cloned, skipping...")
        return

    log("Cloning ComfyUI...")
    subprocess.run(
        ["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", str(COMFYUI_DIR)],
        check=True,
    )


def install_comfyui_requirements() -> None:
    """Install ComfyUI Python requirements."""
    requirements_file = COMFYUI_DIR / "requirements.txt"
    if not requirements_file.exists():
        log("Warning: ComfyUI requirements.txt not found")
        return

    log("Installing ComfyUI requirements...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "-q"],
        check=True,
    )


def install_custom_node(name: str, url: str) -> None:
    """Install a custom node from git URL."""
    nodes_dir = COMFYUI_DIR / "custom_nodes"
    nodes_dir.mkdir(exist_ok=True)

    node_dir = nodes_dir / name
    if node_dir.exists():
        log(f"  {name} already installed")
        return

    log(f"  Installing {name}...")
    subprocess.run(
        ["git", "clone", url, str(node_dir)],
        check=True,
        capture_output=True,
    )


def create_model_directories() -> None:
    """Create ComfyUI model directories."""
    models_dir = COMFYUI_DIR / "models"
    dirs = ["diffusion_models", "text_encoders", "vae", "loras", "checkpoints", "clip"]

    for d in dirs:
        (models_dir / d).mkdir(parents=True, exist_ok=True)


def download_model(filename: str, target_dir: Path) -> bool:
    """
    Download a model file using huggingface-cli.

    Returns True if successful, False otherwise.
    """
    target_path = target_dir / filename

    if target_path.exists():
        log(f"  {filename} already exists, skipping...")
        return True

    if filename not in MODEL_SOURCES:
        log(f"  Warning: Unknown model '{filename}' - please download manually")
        return False

    source = MODEL_SOURCES[filename]
    repo_id, file_path = source.split(":", 1)

    log(f"  Downloading {filename}...")
    log(f"    From: {repo_id}")

    try:
        # Use huggingface-cli for reliable downloads with progress
        result = subprocess.run(
            [
                "huggingface-cli",
                "download",
                repo_id,
                file_path,
                "--local-dir",
                str(target_dir.parent),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Move from split_files subdirectory to target directory
        downloaded_path = target_dir.parent / file_path
        if downloaded_path.exists() and downloaded_path != target_path:
            downloaded_path.rename(target_path)

        # Clean up split_files directory if empty
        split_files_dir = target_dir.parent / "split_files"
        if split_files_dir.exists():
            try:
                # Remove empty directories
                for subdir in split_files_dir.iterdir():
                    if subdir.is_dir() and not any(subdir.iterdir()):
                        subdir.rmdir()
                if not any(split_files_dir.iterdir()):
                    split_files_dir.rmdir()
            except OSError:
                pass

        return target_path.exists()

    except subprocess.CalledProcessError as e:
        log(f"    Error downloading: {e.stderr}")
        return False


def download_models(required_models: dict[str, list[str]]) -> tuple[int, int]:
    """
    Download all required models.

    Returns (success_count, total_count).
    """
    models_dir = COMFYUI_DIR / "models"
    success = 0
    total = 0

    dir_mapping = {
        "diffusion_models": models_dir / "diffusion_models",
        "text_encoders": models_dir / "text_encoders",
        "vae": models_dir / "vae",
        "loras": models_dir / "loras",
    }

    for category, model_list in required_models.items():
        if not model_list:
            continue

        target_dir = dir_mapping.get(category)
        if not target_dir:
            continue

        log(f"\n{category.replace('_', ' ').title()}:")
        for model_name in model_list:
            total += 1
            if download_model(model_name, target_dir):
                success += 1

    return success, total


def save_workflow(workflow: dict[str, Any], name: str) -> None:
    """Save workflow to ComfyUI user directory."""
    workflows_dir = COMFYUI_DIR / "user" / "default" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)

    workflow_path = workflows_dir / f"{name}.json"
    workflow_path.write_text(json.dumps(workflow, indent=2))
    log(f"Workflow saved to: {workflow_path.relative_to(SCRIPT_DIR)}")


def validate_setup(required_models: dict[str, list[str]]) -> bool:
    """Validate that all required models are present."""
    models_dir = COMFYUI_DIR / "models"
    missing = []

    dir_mapping = {
        "diffusion_models": models_dir / "diffusion_models",
        "text_encoders": models_dir / "text_encoders",
        "vae": models_dir / "vae",
        "loras": models_dir / "loras",
    }

    for category, model_list in required_models.items():
        target_dir = dir_mapping.get(category)
        if not target_dir:
            continue

        for model_name in model_list:
            if not (target_dir / model_name).exists():
                missing.append(f"{category}/{model_name}")

    if missing:
        log("\nWarning: Missing models:")
        for m in missing:
            log(f"  - {m}")
        return False

    return True


def main() -> None:
    """Main setup function."""
    total_steps = 7

    # Step 1: Load configuration
    print_step(1, total_steps, "Loading configuration")
    config = load_config()
    workflow_url = config.get("workflow")

    if not workflow_url:
        log("Error: No 'workflow' URL specified in gpu.jsonc")
        sys.exit(1)

    # Step 2: Download and parse workflow
    print_step(2, total_steps, "Downloading workflow")
    try:
        workflow = download_workflow(workflow_url)
    except requests.RequestException as e:
        log(f"Error downloading workflow: {e}")
        sys.exit(1)

    # Extract workflow name from URL
    workflow_name = Path(workflow_url).stem

    required_models = extract_models_from_workflow(workflow)
    log("\nRequired models detected:")
    for category, models in required_models.items():
        if models:
            log(f"  {category}: {', '.join(models)}")

    # Step 3: Clone ComfyUI
    print_step(3, total_steps, "Setting up ComfyUI")
    clone_comfyui()
    install_comfyui_requirements()

    # Step 4: Install custom nodes
    print_step(4, total_steps, "Installing custom nodes")
    for name, url in CUSTOM_NODES.items():
        install_custom_node(name, url)

    # Step 5: Create model directories
    print_step(5, total_steps, "Creating model directories")
    create_model_directories()

    # Step 6: Download models
    print_step(6, total_steps, "Downloading models (this may take a while)")
    success, total = download_models(required_models)
    log(f"\nDownloaded {success}/{total} models")

    # Save workflow
    save_workflow(workflow, workflow_name)

    # Step 7: Validate setup
    print_step(7, total_steps, "Validating setup")
    if validate_setup(required_models):
        log("\nSetup complete!")
        log("\nNext steps:")
        log("  1. Run ComfyUI:")
        log("     gpu run --publish 8188:8188 python run.py")
        log("  2. Open the URL shown in terminal")
        log(f"  3. Load workflow from: Workflows > {workflow_name}")
    else:
        log("\nSetup completed with warnings. Some models may need manual download.")
        sys.exit(1)


if __name__ == "__main__":
    main()
