#!/usr/bin/env python3
"""
Run script for HuggingFace Spaces.

Automatically detects and launches the Gradio app from a cloned Space.
Handles common entry points and configures for remote access.

Usage:
    gpu run --publish 7860:7860 python run_space.py
    gpu run --publish 7860:7860 python run_space.py --space Hunyuan3D-2.1
    gpu run --publish 7860:7860 python run_space.py --space my-space --port 8080
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def create_hf_path_symlink(space_path: Path) -> None:
    """
    Create /home/user/app symlink to support HuggingFace hardcoded paths.

    Many HuggingFace Spaces have hardcoded paths like /home/user/app/ which is
    the standard path on HuggingFace infrastructure. This symlink allows those
    paths to work when running locally.
    """
    hf_app_path = Path("/home/user/app")

    # Skip if symlink already exists and points to the right place
    if hf_app_path.is_symlink():
        if hf_app_path.resolve() == space_path.resolve():
            return
        # Remove incorrect symlink
        try:
            hf_app_path.unlink()
        except PermissionError:
            print(f"  Warning: Cannot update symlink at {hf_app_path}")
            return

    # Skip if path exists as a real directory
    if hf_app_path.exists():
        print(f"  Warning: {hf_app_path} exists and is not a symlink")
        return

    # Create parent directories and symlink
    try:
        hf_app_path.parent.mkdir(parents=True, exist_ok=True)
        hf_app_path.symlink_to(space_path)
        print(f"  Created symlink: {hf_app_path} -> {space_path}")
    except PermissionError:
        print(f"  Warning: Cannot create symlink at {hf_app_path} (permission denied)")
    except Exception as e:
        print(f"  Warning: Cannot create symlink: {e}")


def patch_huggingface_env(space_path: Path) -> None:
    """
    Patch Space files to handle HuggingFace-specific setup gracefully.

    Makes import failures non-fatal so Spaces can run with reduced functionality
    rather than crashing when custom CUDA extensions aren't available.
    """
    # Patterns to patch - skip or make failures non-fatal for graceful degradation
    patterns = [
        # Skip wheel install entirely - we build from source during setup
        # This avoids "wheel is not a supported wheel on this platform" errors
        ('subprocess.run(shlex.split("pip install custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl"), check=True)',
         'print("Skipping wheel install - custom_rasterizer built from source")'),
        # Also handle check=False variant
        ('subprocess.run(shlex.split("pip install custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl"), check=False)',
         'print("Skipping wheel install - custom_rasterizer built from source")'),
    ]

    # Find Python files that might have HuggingFace-specific code
    for py_file in space_path.glob("*.py"):
        try:
            content = py_file.read_text()
            original_content = content

            for old, new in patterns:
                if old in content:
                    content = content.replace(old, new)
                    print(f"  Patched {py_file.name}: {old[:50]}...")

            if content != original_content:
                py_file.write_text(content)
        except Exception as e:
            print(f"  Warning: Could not patch {py_file.name}: {e}")


def create_spaces_mock_file(directory: Path) -> Path:
    """
    Create a mock 'spaces' module file to replace HuggingFace's infrastructure-only module.

    The 'spaces' module provides @spaces.GPU decorator on HuggingFace Spaces.
    This mock makes it a no-op so Spaces can run outside HuggingFace infrastructure.
    """
    mock_content = '''"""
Mock 'spaces' module for running HuggingFace Spaces outside their infrastructure.

Provides no-op implementations of:
- spaces.GPU / spaces.gpu - GPU decorator
- spaces.zero - ZeroGPU module
- spaces.zero.GPU - ZeroGPU decorator
- spaces.zero.torch - Torch utilities
"""
import types

def GPU(func=None, duration=None, **kwargs):
    """Mock @spaces.GPU decorator - just returns the function unchanged."""
    if func is not None:
        return func
    def decorator(fn):
        return fn
    return decorator

# Lowercase alias
gpu = GPU

# Create zero submodule for ZeroGPU compatibility
zero = types.ModuleType("spaces.zero")
zero.GPU = GPU
zero.gpu = GPU

# ZeroGPU startup function - called to initialize the Space
def _startup():
    pass
zero.startup = _startup

# Create zero.torch submodule
zero_torch = types.ModuleType("spaces.zero.torch")
zero_torch.GPU = GPU
zero.torch = zero_torch

# Register submodules so "from spaces import zero" works
import sys
sys.modules["spaces.zero"] = zero
sys.modules["spaces.zero.torch"] = zero_torch
'''
    mock_path = directory / "spaces.py"
    mock_path.write_text(mock_content)
    return mock_path


# Common Gradio entry point files in order of priority
ENTRY_POINTS = [
    "app.py",
    "gradio_app.py",
    "main.py",
    "demo.py",
    "run.py",
    "server.py",
]

# Default Gradio port
DEFAULT_PORT = 7860


def find_spaces(spaces_dir: Path) -> list[Path]:
    """Find all cloned Spaces in the spaces directory."""
    if not spaces_dir.exists():
        return []

    spaces = []
    for item in spaces_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            spaces.append(item)

    return sorted(spaces, key=lambda p: p.stat().st_mtime, reverse=True)


def detect_entry_point(space_path: Path) -> Path | None:
    """Detect the main entry point for a Space."""
    for entry in ENTRY_POINTS:
        entry_path = space_path / entry
        if entry_path.exists():
            return entry_path

    # Fallback: look for any Python file with 'gradio' import
    for py_file in space_path.glob("*.py"):
        try:
            content = py_file.read_text()
            if "import gradio" in content or "from gradio" in content:
                return py_file
        except Exception:
            continue

    return None


def get_launch_command(entry_point: Path, port: int, extra_args: list[str]) -> list[str]:
    """Build the command to launch the Gradio app."""
    # Check if file uses gradio launch with server_name
    try:
        content = entry_point.read_text()
        uses_launch = ".launch(" in content
    except Exception:
        uses_launch = False

    cmd = [sys.executable, str(entry_point)]

    # Add extra arguments passed by user
    cmd.extend(extra_args)

    return cmd


def patch_gradio_launch(entry_point: Path) -> None:
    """
    Provide guidance on making the app accessible remotely.
    We don't patch automatically to avoid breaking Spaces.
    """
    try:
        content = entry_point.read_text()
        if ".launch(" in content and "server_name" not in content:
            print()
            print("NOTE: For remote access, the app should use:")
            print('  demo.launch(server_name="0.0.0.0")')
            print()
            print("If the app isn't accessible, edit the launch() call in:")
            print(f"  {entry_point}")
            print()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Run a cloned HuggingFace Space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_space.py
    python run_space.py --space Hunyuan3D-2.1
    python run_space.py --space my-space --port 8080
    python run_space.py -- --low_vram_mode  # Pass args to the Space
        """,
    )
    parser.add_argument(
        "--space",
        type=str,
        default="",
        help="Name of the Space to run (defaults to most recently cloned)",
    )
    parser.add_argument(
        "--spaces-dir",
        type=str,
        default="spaces",
        help="Directory containing cloned Spaces (default: spaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to run Gradio on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--entry-point",
        type=str,
        default="",
        help="Specific entry point file to run (e.g., gradio_app.py)",
    )

    # Allow passing extra arguments to the Space
    args, extra_args = parser.parse_known_args()

    # Setup paths
    script_dir = Path(__file__).parent.absolute()
    spaces_dir = script_dir / args.spaces_dir

    # Find available Spaces
    spaces = find_spaces(spaces_dir)

    if not spaces:
        print("No Spaces found. Run setup_space.py first:")
        print("  gpu run python setup_space.py owner/space-name")
        sys.exit(1)

    # Select Space to run
    if args.space:
        # Find by name
        space_path = None
        for sp in spaces:
            if sp.name == args.space or sp.name.lower() == args.space.lower():
                space_path = sp
                break
        if not space_path:
            print(f"Space '{args.space}' not found. Available Spaces:")
            for sp in spaces:
                print(f"  - {sp.name}")
            sys.exit(1)
    else:
        # Use most recently modified
        space_path = spaces[0]
        if len(spaces) > 1:
            print(f"Multiple Spaces found, using most recent: {space_path.name}")
            print(f"Use --space <name> to select a specific Space")
            print()

    # Detect or use specified entry point
    if args.entry_point:
        entry_point = space_path / args.entry_point
        if not entry_point.exists():
            print(f"Entry point not found: {entry_point}")
            sys.exit(1)
    else:
        entry_point = detect_entry_point(space_path)
        if not entry_point:
            print(f"Could not detect entry point in {space_path}")
            print("Available Python files:")
            for py_file in space_path.glob("*.py"):
                print(f"  - {py_file.name}")
            print()
            print("Use --entry-point to specify the file to run")
            sys.exit(1)

    print("=" * 60)
    print("HuggingFace Space Runner")
    print("=" * 60)
    print(f"Space: {space_path.name}")
    print(f"Entry point: {entry_point.name}")
    print(f"Port: {args.port}")
    if extra_args:
        print(f"Extra args: {' '.join(extra_args)}")
    print("=" * 60)
    print()

    # Set environment for remote access
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    os.environ["GRADIO_SERVER_PORT"] = str(args.port)

    # Create symlink for HuggingFace hardcoded paths
    print("Setting up HuggingFace path compatibility...")
    create_hf_path_symlink(space_path)

    # Patch HuggingFace-specific environment checks
    # This disables HF-only setup code that won't work locally
    print("Patching HuggingFace-specific code...")
    patch_huggingface_env(space_path)

    # Fix NumPy version - many Spaces require NumPy 1.x but other packages may upgrade to 2.x
    # This causes "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x" errors
    print("Ensuring NumPy 1.x compatibility...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "numpy<2", "-q"],
        capture_output=True,
    )

    # Create mock 'spaces' module in the space directory
    # This replaces HuggingFace's infrastructure-only module
    create_spaces_mock_file(space_path)

    # Add space directory to PYTHONPATH so the mock 'spaces' module is found first
    python_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(space_path) + (os.pathsep + python_path if python_path else "")

    # Change to space directory
    os.chdir(space_path)

    # Provide guidance on remote access
    patch_gradio_launch(entry_point)

    # Build and run command
    cmd = get_launch_command(entry_point, args.port, extra_args)

    print(f"Running: {' '.join(cmd)}")
    print()
    print(f"Access the app at: http://localhost:{args.port}")
    print("(Use the remapped port shown in GPU CLI output if different)")
    print()
    print("-" * 60)
    print()

    # Run the Space
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except subprocess.CalledProcessError as e:
        print(f"\nSpace exited with error code: {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
