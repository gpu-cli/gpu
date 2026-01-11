#!/usr/bin/env python3
"""
Setup script for HuggingFace Gradio Space.

This script:
1. Validates the Space exists on HuggingFace
2. Downloads the Space using huggingface_hub.snapshot_download()
3. Installs system dependencies from packages.txt (if present)
4. Installs Python dependencies from requirements.txt
5. Validates the download and creates a setup marker

Run once before using run.py. Re-running is idempotent.

Usage:
    python setup.py owner/space-name
    python setup.py https://huggingface.co/spaces/owner/space-name
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Force unbuffered output for real-time feedback
os.environ["PYTHONUNBUFFERED"] = "1"

from huggingface_hub import snapshot_download, HfApi

SCRIPT_DIR = Path(__file__).parent.absolute()
SPACES_DIR = SCRIPT_DIR / "spaces"
SETUP_MARKER = ".setup_complete"


def log(msg: str) -> None:
    """Print with flush for real-time output."""
    print(msg, flush=True)


def print_step(step: int, total: int, message: str) -> None:
    """Print a formatted step message."""
    print(f"\n{'='*60}", flush=True)
    print(f"[{step}/{total}] {message}", flush=True)
    print("=" * 60, flush=True)


def parse_space_id(space_arg: str) -> str:
    """
    Parse Space ID from argument.

    Handles:
    - owner/space-name
    - https://huggingface.co/spaces/owner/space-name
    """
    if "huggingface.co/spaces/" in space_arg:
        space_arg = space_arg.split("huggingface.co/spaces/")[-1].rstrip("/")

    if "/" not in space_arg:
        log(f"Error: Invalid Space ID: {space_arg}")
        log("Expected format: owner/space-name")
        sys.exit(1)

    return space_arg


def validate_space_exists(space_id: str) -> dict:
    """
    Validate the Space exists and get its metadata.

    Returns Space info dict or exits with error.
    """
    try:
        api = HfApi()
        space_info = api.space_info(space_id)
        return {
            "id": space_info.id,
            "author": space_info.author,
            "sdk": space_info.sdk,
            "private": space_info.private,
        }
    except Exception as e:
        log(f"Error: Could not find Space '{space_id}'")
        log(f"Details: {e}")
        log("")
        log("Check that:")
        log("  1. The Space ID is correct (owner/space-name)")
        log("  2. The Space is public (or set HF_TOKEN for private Spaces)")
        log("  3. Your network can reach huggingface.co")
        sys.exit(1)


def download_space(space_id: str) -> Path:
    """
    Download Space using huggingface_hub.

    Returns path to downloaded Space directory.
    """
    space_name = space_id.split("/")[-1]
    space_path = SPACES_DIR / space_name

    # Check if already downloaded with setup complete
    marker_path = space_path / SETUP_MARKER
    if marker_path.exists():
        log(f"Space already set up at: {space_path}")
        return space_path

    # Download (will use cache if partially downloaded)
    SPACES_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Downloading Space: {space_id}")
    log(f"Destination: {space_path}")
    log("")

    try:
        snapshot_download(
            repo_id=space_id,
            repo_type="space",
            local_dir=str(space_path),
            local_dir_use_symlinks=False,  # Full copy for reliability
        )
    except Exception as e:
        log(f"Error downloading Space: {e}")
        sys.exit(1)

    log(f"Download complete: {space_path}")
    return space_path


def install_system_packages(space_path: Path) -> None:
    """
    Install system packages from packages.txt (if present).

    This is the HuggingFace Spaces convention for apt packages.
    """
    packages_file = space_path / "packages.txt"
    if not packages_file.exists():
        log("No packages.txt found, skipping system dependencies")
        return

    # Read and filter empty lines and comments
    packages = []
    for line in packages_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            packages.append(line)

    if not packages:
        log("packages.txt is empty, skipping system dependencies")
        return

    log(f"Installing system packages: {', '.join(packages)}")

    try:
        # Update apt first
        subprocess.run(
            ["apt-get", "update", "-qq"],
            check=True,
            capture_output=True,
        )

        # Install packages
        subprocess.run(
            ["apt-get", "install", "-y", "-qq"] + packages,
            check=True,
        )
        log("System packages installed successfully")
    except subprocess.CalledProcessError as e:
        log("Warning: Failed to install some system packages")
        log(f"Error: {e}")
        log("The Space may still work if these are optional dependencies")


def install_python_requirements(space_path: Path) -> bool:
    """
    Install Python requirements from requirements.txt.

    Returns True if successful, False otherwise.
    """
    req_file = space_path / "requirements.txt"
    if not req_file.exists():
        log("No requirements.txt found in Space")
        return True

    log("Installing Python dependencies...")
    log(f"Requirements file: {req_file}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            log("Warning: Some dependencies may have failed to install")
            if result.stderr:
                # Show first few lines of error
                error_lines = result.stderr.strip().split("\n")[:5]
                for line in error_lines:
                    log(f"  {line}")
            return False

        log("Python dependencies installed successfully")
        return True
    except Exception as e:
        log(f"Error installing dependencies: {e}")
        return False


def find_entry_point(space_path: Path) -> Optional[Path]:
    """
    Find the Gradio app entry point.

    Returns the entry point path or None if not found.
    """
    # Common entry point names in order of preference
    candidates = ["app.py", "gradio_app.py", "main.py", "demo.py", "run.py", "interface.py"]

    for name in candidates:
        path = space_path / name
        if path.exists():
            return path

    # Search for files with gradio.launch() call
    for py_file in space_path.glob("*.py"):
        try:
            content = py_file.read_text()
            if "gradio" in content.lower() and ".launch(" in content:
                return py_file
        except Exception:
            continue

    return None


def validate_setup(space_path: Path) -> bool:
    """
    Validate the setup is complete and functional.

    Returns True if valid, False otherwise.
    """
    issues = []

    # Check Space directory exists
    if not space_path.exists():
        issues.append(f"Space directory not found: {space_path}")

    # Check entry point exists
    entry_point = find_entry_point(space_path)
    if not entry_point:
        issues.append("No Gradio entry point found (app.py, main.py, etc.)")
        # List available files to help debugging
        py_files = list(space_path.glob("*.py"))
        if py_files:
            log("Available Python files:")
            for f in py_files[:10]:
                log(f"  - {f.name}")
    else:
        log(f"Entry point found: {entry_point.name}")

    if issues:
        log("\nSetup validation failed:")
        for issue in issues:
            log(f"  - {issue}")
        return False

    return True


def create_setup_marker(space_path: Path, space_info: dict) -> None:
    """Create marker file indicating setup is complete."""
    marker_path = space_path / SETUP_MARKER
    marker_data = {
        "space_id": space_info["id"],
        "sdk": space_info.get("sdk", "unknown"),
        "setup_complete": True,
    }
    marker_path.write_text(json.dumps(marker_data, indent=2))
    log(f"Setup marker created: {marker_path}")


def main() -> None:
    """Main setup function."""
    if len(sys.argv) < 2:
        log("Usage: python setup.py <space_id>")
        log("Example: python setup.py Lightricks/ltx-video-distilled")
        sys.exit(1)

    space_id = parse_space_id(sys.argv[1])
    total_steps = 5

    log(f"Setting up HuggingFace Space: {space_id}")
    log(f"URL: https://huggingface.co/spaces/{space_id}")
    log("")

    # Step 1: Validate Space exists
    print_step(1, total_steps, f"Validating Space: {space_id}")
    space_info = validate_space_exists(space_id)
    log(f"Space found: {space_info['id']}")
    log(f"SDK: {space_info.get('sdk', 'unknown')}")

    if space_info.get("sdk") != "gradio":
        log(f"Warning: Space SDK is '{space_info.get('sdk')}', not 'gradio'")
        log("This template is designed for Gradio Spaces")

    # Step 2: Download Space
    print_step(2, total_steps, "Downloading Space")
    space_path = download_space(space_id)

    # Step 3: Install system packages
    print_step(3, total_steps, "Installing system dependencies")
    install_system_packages(space_path)

    # Step 4: Install Python requirements
    print_step(4, total_steps, "Installing Python dependencies")
    install_python_requirements(space_path)

    # Step 5: Validate setup
    print_step(5, total_steps, "Validating setup")
    if validate_setup(space_path):
        create_setup_marker(space_path, space_info)
        log("\n" + "=" * 60)
        log("Setup complete!")
        log("=" * 60)
        log("")
        log("Next steps:")
        log("  1. Run the Space:")
        log(f"     python run.py {space_id}")
        log("")
        log("  Or with GPU CLI:")
        log(f"     gpu run python run.py {space_id}")
        log("")
        log(f"Space downloaded to: {space_path}")
    else:
        log("\nSetup completed with warnings.")
        log("The Space may still work - try running it:")
        log(f"  python run.py {space_id}")
        sys.exit(1)


if __name__ == "__main__":
    main()
