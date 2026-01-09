#!/usr/bin/env python3
"""
Setup script for HuggingFace Spaces.

Clones any HuggingFace Space and installs its dependencies,
preparing it to run on a remote GPU via GPU CLI.

Usage:
    gpu run python setup_space.py tencent/Hunyuan3D-2.1
    gpu run python setup_space.py https://huggingface.co/spaces/stabilityai/stable-diffusion
    gpu run python setup_space.py owner/space-name --branch main
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_space_identifier(identifier: str) -> tuple[str, str]:
    """Parse a Space identifier into owner and space name.

    Accepts formats:
    - owner/space-name
    - https://huggingface.co/spaces/owner/space-name
    - huggingface.co/spaces/owner/space-name
    """
    # Remove trailing slashes
    identifier = identifier.rstrip("/")

    # Handle full URLs
    url_pattern = r"(?:https?://)?(?:www\.)?huggingface\.co/spaces/([^/]+)/([^/]+)"
    match = re.match(url_pattern, identifier)
    if match:
        return match.group(1), match.group(2)

    # Handle owner/space format
    if "/" in identifier:
        parts = identifier.split("/")
        if len(parts) == 2:
            return parts[0], parts[1]

    raise ValueError(
        f"Invalid Space identifier: {identifier}\n"
        "Expected format: owner/space-name or https://huggingface.co/spaces/owner/space-name"
    )


def pull_lfs_files(space_path: Path) -> None:
    """Pull Git LFS files, installing git-lfs if needed."""
    print("Pulling Git LFS files...")

    # Try git lfs pull first
    result = subprocess.run(
        ["git", "-C", str(space_path), "lfs", "pull"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("Git LFS files pulled successfully")
        return

    # git-lfs might not be installed, try installing it
    print("git-lfs not found, installing...")
    subprocess.run(
        ["apt-get", "update", "-qq"],
        capture_output=True,
    )
    install_result = subprocess.run(
        ["apt-get", "install", "-y", "-qq", "git-lfs"],
        capture_output=True,
    )

    if install_result.returncode != 0:
        print("Warning: Could not install git-lfs. Example images may not load.")
        return

    subprocess.run(["git", "lfs", "install"], capture_output=True)

    # Try again
    result = subprocess.run(
        ["git", "-C", str(space_path), "lfs", "pull"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("Git LFS files pulled successfully")
    else:
        print(f"Warning: Could not pull Git LFS files. Example images may not load.")


def clone_space(owner: str, space: str, branch: str, spaces_dir: Path) -> Path:
    """Clone a HuggingFace Space repository."""
    space_path = spaces_dir / space

    if space_path.exists():
        print(f"Space already cloned at: {space_path}")
        print("Pulling latest changes...")
        subprocess.run(["git", "-C", str(space_path), "pull"], check=True)
        pull_lfs_files(space_path)
        return space_path

    print(f"Cloning {owner}/{space}...")
    clone_url = f"https://huggingface.co/spaces/{owner}/{space}"

    cmd = ["git", "clone"]
    if branch:
        cmd.extend(["-b", branch])
    cmd.extend([clone_url, str(space_path)])

    subprocess.run(cmd, check=True)
    print(f"Cloned to: {space_path}")

    pull_lfs_files(space_path)

    return space_path


def install_requirements(space_path: Path) -> None:
    """Install Python requirements for the Space."""
    requirements_file = space_path / "requirements.txt"

    if not requirements_file.exists():
        print("No requirements.txt found, skipping dependency installation")
        return

    print("Installing dependencies from requirements.txt...")

    # First, try bulk install (fast path) - stream output so user sees progress
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
    )

    if result.returncode == 0:
        print("Dependencies installed successfully")
        return

    # Bulk install failed - fall back to one-by-one installation
    print("\nBulk install failed, trying packages individually...")

    # Packages with known version issues and their fallbacks
    fallback_versions = {
        "open3d": ["open3d"],  # Try latest if specific version fails
        "bpy": [],  # bpy 4.0 doesn't exist, skip entirely
    }

    failed_packages = []
    with open(requirements_file) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#") or line.startswith("-"):
                continue

            # Extract package name from version specifier
            pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0].strip()

            pkg_result = subprocess.run(
                [sys.executable, "-m", "pip", "install", line],
                capture_output=True,
                text=True,
            )

            if pkg_result.returncode != 0:
                # Try fallback versions if available
                installed = False
                if pkg_name in fallback_versions:
                    for fallback in fallback_versions[pkg_name]:
                        fallback_result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", fallback],
                            capture_output=True,
                            text=True,
                        )
                        if fallback_result.returncode == 0:
                            print(f"  Installed: {fallback} (fallback for {line})")
                            installed = True
                            break

                if not installed:
                    failed_packages.append(line)
                    print(f"  Warning: Failed to install {line}")
            else:
                print(f"  Installed: {line}")

    if failed_packages:
        print(f"\nWarning: {len(failed_packages)} package(s) failed to install:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        print("The Space may still work if these are optional dependencies.\n")
    else:
        print("Dependencies installed successfully")


def run_setup_scripts(space_path: Path) -> None:
    """Run common setup scripts if they exist."""
    setup_scripts = [
        "setup.py",
        "setup.sh",
        "install.py",
        "install.sh",
    ]

    for script in setup_scripts:
        script_path = space_path / script
        if script_path.exists():
            print(f"Found {script}, running...")
            if script.endswith(".py"):
                subprocess.run([sys.executable, str(script_path)], cwd=space_path)
            elif script.endswith(".sh"):
                subprocess.run(["bash", str(script_path)], cwd=space_path)


def build_custom_extensions(space_path: Path) -> None:
    """Build custom CUDA extensions from source if detected.

    This handles Spaces like Hunyuan3D that include custom CUDA extensions
    which need to be compiled against the local PyTorch installation.
    """
    # Hunyuan3D: custom_rasterizer
    custom_rasterizer_paths = [
        space_path / "hy3dpaint" / "custom_rasterizer",
        space_path / "hy3dpaint" / "packages" / "custom_rasterizer",
    ]

    for rasterizer_path in custom_rasterizer_paths:
        if rasterizer_path.exists() and (rasterizer_path / "setup.py").exists():
            print(f"Building custom_rasterizer from source at {rasterizer_path}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                cwd=rasterizer_path,
            )
            if result.returncode == 0:
                print("custom_rasterizer built successfully")
            else:
                print("Warning: custom_rasterizer build failed, some features may be limited")
            break

    # Hunyuan3D: DifferentiableRenderer
    diff_renderer_path = space_path / "hy3dpaint" / "DifferentiableRenderer"
    compile_script = diff_renderer_path / "compile_mesh_painter.sh"
    if compile_script.exists():
        print("Compiling DifferentiableRenderer...")
        result = subprocess.run(
            ["bash", str(compile_script)],
            cwd=diff_renderer_path,
        )
        if result.returncode == 0:
            print("DifferentiableRenderer compiled successfully")
        else:
            print("Warning: DifferentiableRenderer compilation failed")


def detect_entry_point(space_path: Path) -> str | None:
    """Detect the main entry point for the Gradio app."""
    entry_points = ["app.py", "gradio_app.py", "main.py", "demo.py"]

    for entry in entry_points:
        if (space_path / entry).exists():
            return entry

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Clone and setup a HuggingFace Space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python setup_space.py tencent/Hunyuan3D-2.1
    python setup_space.py stabilityai/stable-diffusion-3.5-large
    python setup_space.py https://huggingface.co/spaces/owner/space --branch dev
        """,
    )
    parser.add_argument(
        "space",
        type=str,
        help="Space identifier (owner/name or full URL)",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="",
        help="Git branch to clone (default: main)",
    )
    parser.add_argument(
        "--spaces-dir",
        type=str,
        default="spaces",
        help="Directory to store cloned Spaces (default: spaces)",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip installing Python dependencies",
    )
    args = parser.parse_args()

    # Parse the Space identifier
    try:
        owner, space = parse_space_identifier(args.space)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("HuggingFace Space Setup")
    print("=" * 60)
    print(f"Space: {owner}/{space}")
    print(f"URL: https://huggingface.co/spaces/{owner}/{space}")
    print("=" * 60)
    print()

    # Setup paths
    script_dir = Path(__file__).parent.absolute()
    spaces_dir = script_dir / args.spaces_dir
    spaces_dir.mkdir(exist_ok=True)

    # Clone the Space
    space_path = clone_space(owner, space, args.branch, spaces_dir)

    # Install requirements
    if not args.skip_deps:
        install_requirements(space_path)

    # Run any setup scripts
    run_setup_scripts(space_path)

    # Build custom CUDA extensions if present (e.g., Hunyuan3D)
    build_custom_extensions(space_path)

    # Detect entry point
    entry_point = detect_entry_point(space_path)

    print()
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print()
    print(f"Space directory: {space_path}")
    if entry_point:
        print(f"Entry point: {entry_point}")
        print()
        print("To run the Space:")
        print(f"  gpu run --publish 7860:7860 python run_space.py")
        print()
        print("Or run directly:")
        print(f"  gpu run --publish 7860:7860 python spaces/{space}/{entry_point}")
    else:
        print("Could not detect entry point. Check the Space for app.py or similar.")
    print()


if __name__ == "__main__":
    main()
