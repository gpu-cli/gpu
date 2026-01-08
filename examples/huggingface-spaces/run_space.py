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


def create_spaces_mock_file(directory: Path) -> Path:
    """
    Create a mock 'spaces' module file to replace HuggingFace's infrastructure-only module.

    The 'spaces' module provides @spaces.GPU decorator on HuggingFace Spaces.
    This mock makes it a no-op so Spaces can run outside HuggingFace infrastructure.
    """
    mock_content = '''"""
Mock 'spaces' module for running HuggingFace Spaces outside their infrastructure.
"""

def GPU(func=None, duration=None):
    """Mock @spaces.GPU decorator - just returns the function unchanged."""
    if func is not None:
        return func
    def decorator(fn):
        return fn
    return decorator

# Lowercase alias
gpu = GPU
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
