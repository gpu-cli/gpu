#!/usr/bin/env python3
"""
Launch a downloaded HuggingFace Gradio Space.

Run setup.py first to download and install dependencies.

Usage:
    python run.py owner/space-name
    python run.py https://huggingface.co/spaces/owner/space-name
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

SCRIPT_DIR = Path(__file__).parent.absolute()
SPACES_DIR = SCRIPT_DIR / "spaces"
SETUP_MARKER = ".setup_complete"


def log(msg: str) -> None:
    """Print with flush for real-time output."""
    print(msg, flush=True)


def parse_space_id(space_arg: str) -> str:
    """Parse Space ID from argument."""
    if "huggingface.co/spaces/" in space_arg:
        space_arg = space_arg.split("huggingface.co/spaces/")[-1].rstrip("/")
    return space_arg


def get_space_path(space_id: str) -> Path:
    """Get path to downloaded Space."""
    space_name = space_id.split("/")[-1]
    return SPACES_DIR / space_name


def check_setup(space_path: Path, space_id: str) -> bool:
    """Check if setup has been completed."""
    marker_path = space_path / SETUP_MARKER

    if not space_path.exists():
        log(f"Error: Space not downloaded at {space_path}")
        log("")
        log("Please run setup first:")
        log(f"  python setup.py {space_id}")
        return False

    if not marker_path.exists():
        log("Warning: Setup may be incomplete (no marker file)")
        log("Consider re-running setup:")
        log(f"  python setup.py {space_id}")
        # Continue anyway - setup marker is optional

    return True


def find_entry_point(space_path: Path) -> Optional[Path]:
    """Find the Gradio app entry point."""
    candidates = ["app.py", "gradio_app.py", "main.py", "demo.py", "run.py", "interface.py"]

    for name in candidates:
        path = space_path / name
        if path.exists():
            return path

    # Search for files with gradio.launch()
    for py_file in space_path.glob("*.py"):
        try:
            content = py_file.read_text()
            if "gradio" in content.lower() and ".launch(" in content:
                return py_file
        except Exception:
            continue

    return None


def setup_environment(space_path: Path) -> None:
    """Set up environment variables for Gradio."""
    # Gradio configuration for remote access
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    os.environ["GRADIO_SERVER_PORT"] = "7860"

    # HuggingFace cache directory (use Space's cache)
    cache_dir = space_path / ".cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")

    # Some Spaces check for HF Space environment
    os.environ.setdefault("SPACE_ID", "")

    # Disable Gradio analytics
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


def launch_app(space_path: Path, entry_point: Path) -> None:
    """Launch the Gradio application."""
    log("=" * 60)
    log(f"Launching: {entry_point.name}")
    log(f"Space: {space_path.name}")
    log("=" * 60)
    log("")
    log("Access URL: http://localhost:7860")
    log("")
    log("Note: Run with port forwarding flag:")
    log("  gpu run -p 7860:7860 python run.py <space_id>")
    log("")
    log("Press Ctrl+C to stop")
    log("")

    # Change to Space directory (many Spaces expect this)
    os.chdir(space_path)

    # Add Space directory to Python path for imports
    sys.path.insert(0, str(space_path))

    # Launch with unbuffered output
    try:
        subprocess.run(
            [sys.executable, "-u", str(entry_point)],
            check=True,
        )
    except KeyboardInterrupt:
        log("\nShutting down...")
    except subprocess.CalledProcessError as e:
        log(f"\nSpace exited with error code: {e.returncode}")
        sys.exit(e.returncode)


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        log("Usage: python run.py <space_id>")
        log("Example: python run.py Lightricks/ltx-video-distilled")
        sys.exit(1)

    space_id = parse_space_id(sys.argv[1])
    space_path = get_space_path(space_id)

    log(f"Space ID: {space_id}")
    log(f"Space path: {space_path}")
    log("")

    # Validate setup
    if not check_setup(space_path, space_id):
        sys.exit(1)

    # Find entry point
    entry_point = find_entry_point(space_path)
    if not entry_point:
        log(f"Error: No Gradio entry point found in {space_path}")
        log("")
        log("Available Python files:")
        for f in space_path.glob("*.py"):
            log(f"  - {f.name}")
        log("")
        log("Try running the Space manually:")
        log(f"  cd {space_path}")
        log("  python <entry_point>.py")
        sys.exit(1)

    log(f"Entry point: {entry_point.name}")

    # Set up environment
    setup_environment(space_path)

    # Launch
    launch_app(space_path, entry_point)


if __name__ == "__main__":
    main()
