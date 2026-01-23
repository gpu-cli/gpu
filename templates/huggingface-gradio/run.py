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
import re
import subprocess
import sys
import tempfile
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


def create_spaces_mock(space_path: Path) -> None:
    """
    Create a mock 'spaces' module to replace HuggingFace's infrastructure-only module.

    The real 'spaces' module provides @spaces.GPU decorator on HuggingFace Spaces.
    This mock makes it a no-op so Spaces can run outside HuggingFace infrastructure.
    """
    mock_content = '''"""
Mock 'spaces' module for running HuggingFace Spaces outside their infrastructure.

Provides no-op implementations of:
- spaces.GPU / spaces.gpu - GPU decorator (with duration parameter support)
- spaces.zero - ZeroGPU module
- spaces.zero.GPU - ZeroGPU decorator
- spaces.zero.torch - Torch utilities
- spaces.config - Space configuration
- spaces.utils - Utility functions
"""
import functools
import sys
import types
from typing import Any, Callable, Optional, Union


def GPU(
    func: Optional[Callable] = None,
    duration: Optional[Union[int, Callable]] = None,
    **kwargs: Any,
) -> Callable:
    """
    Mock @spaces.GPU decorator.

    Supports all real decorator patterns:
    - @spaces.GPU
    - @spaces.GPU()
    - @spaces.GPU(duration=120)
    - @spaces.GPU(duration=lambda *args: compute_duration(*args))
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)
        return wrapper

    # @spaces.GPU - called with function directly
    if func is not None:
        return decorator(func)

    # @spaces.GPU() or @spaces.GPU(duration=120) - returns decorator
    return decorator


# Lowercase alias (some Spaces use @spaces.gpu)
gpu = GPU


# Create zero submodule for ZeroGPU compatibility
zero = types.ModuleType("spaces.zero")
zero.GPU = GPU
zero.gpu = GPU


def _startup() -> None:
    """ZeroGPU startup function - no-op locally."""
    pass

zero.startup = _startup


# Create zero.torch submodule
zero_torch = types.ModuleType("spaces.zero.torch")
zero_torch.GPU = GPU
zero_torch.gpu = GPU
zero.torch = zero_torch


# Config module for Space configuration
class SpaceConfig:
    """Mock Space configuration."""

    def __init__(self):
        self.hardware = "local"
        self.space_id = None
        self.repo_id = None

    @property
    def in_space(self) -> bool:
        """Returns False since we're running locally."""
        return False


config = SpaceConfig()


# Utils module
utils = types.ModuleType("spaces.utils")


def _is_zero_gpu() -> bool:
    """Check if running on ZeroGPU - always False locally."""
    return False


def _is_space() -> bool:
    """Check if running in a Space - always False locally."""
    return False


utils.is_zero_gpu = _is_zero_gpu
utils.is_space = _is_space


# Register all submodules so imports work
sys.modules["spaces"] = sys.modules[__name__]
sys.modules["spaces.zero"] = zero
sys.modules["spaces.zero.torch"] = zero_torch
sys.modules["spaces.utils"] = utils
sys.modules["spaces.config"] = types.ModuleType("spaces.config")
sys.modules["spaces.config"].Config = SpaceConfig


# Additional exports
__all__ = [
    "GPU",
    "gpu",
    "zero",
    "config",
    "utils",
]
'''
    mock_path = space_path / "spaces.py"
    mock_path.write_text(mock_content)
    log(f"Created spaces mock: {mock_path.name}")

    # Uninstall real 'spaces' package if installed (it expects HF infrastructure)
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "spaces"],
        capture_output=True,
    )


def patch_entry_point_for_share(entry_point: Path) -> Path:
    """Patch the entry point to enable share=True for public URL access.

    Returns path to patched file (or original if no patching needed).
    """
    content = entry_point.read_text()

    # Check if already has share=True
    if "share=True" in content or "share = True" in content:
        log("Entry point already has share=True, no patching needed")
        return entry_point

    patched = False

    # First, replace any existing share=False with share=True
    if "share=False" in content:
        content = content.replace("share=False", "share=True")
        patched = True
        log("Replaced share=False with share=True")
    elif "share = False" in content:
        content = content.replace("share = False", "share = True")
        patched = True
        log("Replaced share = False with share = True")
    else:
        # No existing share argument, need to add it
        # Replace .launch() with .launch(share=True)
        if ".launch()" in content:
            content = content.replace(".launch()", ".launch(share=True)")
            patched = True
        # Replace .launch(args...) with .launch(share=True, args...)
        # Match .launch( followed by anything that's not a closing paren
        elif ".launch(" in content:
            content = re.sub(r'\.launch\(', '.launch(share=True, ', content)
            patched = True

    if not patched:
        log("Warning: Could not find .launch() call to patch")
        return entry_point

    # Write patched content to a temp file in the same directory
    patched_path = entry_point.parent / f"_patched_{entry_point.name}"
    patched_path.write_text(content)
    log(f"Patched entry point for share=True: {patched_path.name}")
    return patched_path


def launch_app(space_path: Path, entry_point: Path) -> None:
    """Launch the Gradio application."""
    # Create spaces mock module for ZeroGPU compatibility
    create_spaces_mock(space_path)

    # Patch entry point to enable share=True for public URL
    patched_entry = patch_entry_point_for_share(entry_point)

    log("=" * 60)
    log(f"Launching: {entry_point.name}")
    log(f"Space: {space_path.name}")
    log("=" * 60)
    log("")
    log("Access URL: Look for 'Running on public URL: https://xxxxx.gradio.live' below")
    log("")
    log("The gradio.live URL provides much faster performance than localhost.")
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
            [sys.executable, "-u", str(patched_entry)],
            check=True,
        )
    except KeyboardInterrupt:
        log("\nShutting down...")
    except subprocess.CalledProcessError as e:
        log(f"\nSpace exited with error code: {e.returncode}")
        sys.exit(e.returncode)
    finally:
        # Clean up patched file if we created one
        if patched_entry != entry_point and patched_entry.exists():
            patched_entry.unlink()


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
