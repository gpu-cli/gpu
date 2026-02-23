#!/bin/bash
# Hunyuan3D-2.1 Startup Script
#
# This script:
# 1. Downloads the Hunyuan3D-2.1 space from HuggingFace
# 2. Installs Python dependencies (with numpy==1.24.4 pinned)
# 3. Compiles custom CUDA extensions (custom_rasterizer, DifferentiableRenderer)
# 4. Creates a mock 'spaces' module (replaces HF ZeroGPU infrastructure)
# 5. Patches ENV to skip HF-specific setup, launches Gradio UI via uvicorn
#
# Designed for RunPod with CUDA 12.4 devel images (Python 3.11).
#
# NOTE: We do NOT use `set -e` because non-critical failures (bpy, texture gen
# compilation) should not kill the pod. Critical failures use `die()` explicitly.

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACE_DIR="${SCRIPT_DIR}/space"
SPACE_ID="tencent/Hunyuan3D-2.1"
OUTPUTS_DIR="${SCRIPT_DIR}/outputs"
PIP="python3 -m pip"

log() {
    echo "[hunyuan3d] $(date '+%H:%M:%S') $*"
}

die() {
    log "FATAL: $*"
    exit 1
}

# ==============================================================================
# Step 1: Download the HuggingFace Space
# ==============================================================================
download_space() {
    if [ -f "${SPACE_DIR}/.download_complete" ]; then
        log "Space already downloaded at ${SPACE_DIR}"
        return 0
    fi

    log "Downloading HuggingFace Space: ${SPACE_ID}"

    # Ensure huggingface_hub is available
    if ! python3 -c "import huggingface_hub" 2>/dev/null; then
        log "huggingface_hub not found; installing..."
        ${PIP} install "huggingface_hub>=0.30.0" --no-cache-dir || die "Failed to install huggingface_hub"
    fi

    mkdir -p "${SPACE_DIR}"
    python3 -u -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${SPACE_ID}',
    repo_type='space',
    local_dir='${SPACE_DIR}',
    local_dir_use_symlinks=False,
)
print('Download complete')
" || die "Failed to download Space"

    touch "${SPACE_DIR}/.download_complete"
    log "Space downloaded successfully"
}

# ==============================================================================
# Step 2: Install Space-specific Python requirements
#
# Strategy:
# - Build a constraints file pinning torch* to base image versions
# - Install Space's requirements.txt (which pins numpy==1.24.4) with constraints
# - Skip bpy (no wheels) and --extra-index-url (wrong torch builds)
# - Verify numpy<2 at the end; force-reinstall if needed
# ==============================================================================
install_space_deps() {
    # Smart marker: check if deps are installed AND numpy is correct
    if [ -f "${SPACE_DIR}/.deps_complete" ]; then
        NUMPY_VER=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "missing")
        if [[ "${NUMPY_VER}" == 1.* ]]; then
            log "Space dependencies already installed (numpy=${NUMPY_VER})"
            return 0
        fi
        log "Deps marker exists but numpy=${NUMPY_VER} (need 1.x). Reinstalling..."
        rm -f "${SPACE_DIR}/.deps_complete"
    fi

    log "Installing Space-specific Python dependencies..."

    # Build constraints: pin torch ecosystem to base image versions
    CONSTRAINTS="${SPACE_DIR}/_constraints.txt"
    python3 -c "
import importlib.metadata as md
constraints = []
for dist in ['torch', 'torchvision', 'torchaudio', 'triton', 'xformers']:
    try:
        ver = md.version(dist)
        constraints.append(f'{dist}=={ver}')
    except Exception:
        pass
with open('${CONSTRAINTS}', 'w') as f:
    f.write('\n'.join(constraints) + '\n')
print('Constraints:', constraints)
"

    SPACE_REQS="${SPACE_DIR}/requirements.txt"
    if [ -f "${SPACE_REQS}" ]; then
        # Filter out problematic packages:
        # - bpy: no wheels for cloud images (Blender-only, affects export formats not core gen)
        # - --extra-index-url: points to blender pypi, can pull wrong torch builds
        grep -v -i '^bpy' "${SPACE_REQS}" | \
        grep -v '^--extra-index-url' | \
        grep -v '^#' | \
        grep -v '^$' > "${SPACE_DIR}/_filtered_reqs.txt" || true

        log "Installing from Space requirements.txt (with torch constraints)..."
        ${PIP} install -r "${SPACE_DIR}/_filtered_reqs.txt" \
            -c "${CONSTRAINTS}" \
            --no-cache-dir 2>&1 || {
            log "Bulk install had errors; falling back to per-package install..."
            while IFS= read -r pkg; do
                pkg=$(echo "$pkg" | xargs)
                [ -z "$pkg" ] && continue
                [[ "$pkg" == \#* ]] && continue
                log "  Installing: ${pkg}"
                ${PIP} install "${pkg}" -c "${CONSTRAINTS}" --no-cache-dir 2>&1 || \
                    log "  WARNING: Failed to install ${pkg} (continuing)"
            done < "${SPACE_DIR}/_filtered_reqs.txt"
        }
    fi

    # Verify numpy<2. The Space pins numpy==1.24.4, but transitive deps can override.
    NUMPY_VER=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "missing")
    if [[ "${NUMPY_VER}" != 1.* ]]; then
        log "numpy is ${NUMPY_VER} (need 1.x). Force-installing numpy==1.24.4..."
        ${PIP} install "numpy==1.24.4" --force-reinstall --no-cache-dir 2>&1 || \
            die "Failed to install numpy 1.x - onnxruntime/rembg will crash"
        NUMPY_VER=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "FAILED")
    fi
    log "numpy version: ${NUMPY_VER}"

    touch "${SPACE_DIR}/.deps_complete"
    log "Space dependencies installed"
}

# ==============================================================================
# Step 3: Compile custom CUDA extensions
# ==============================================================================
compile_extensions() {
    # Smart marker: verify the extension actually imports, not just that we ran
    if [ -f "${SPACE_DIR}/.compile_complete" ]; then
        if python3 -c "import custom_rasterizer" 2>/dev/null; then
            log "Custom extensions already compiled and importable"
            return 0
        fi
        log "Compile marker exists but custom_rasterizer not importable. Recompiling..."
        rm -f "${SPACE_DIR}/.compile_complete"
    fi

    log "Compiling custom CUDA extensions..."

    # Set CUDA environment
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"

    # Install custom_rasterizer
    # The Space ships a cp310 wheel; on Python 3.11 we build from source
    log "Installing custom_rasterizer..."
    WHEEL="${SPACE_DIR}/custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl"
    INSTALLED=0

    if [ -f "${WHEEL}" ]; then
        ${PIP} install "${WHEEL}" 2>/dev/null && INSTALLED=1
    fi

    if [ "${INSTALLED}" -eq 0 ]; then
        log "Pre-built wheel unavailable or incompatible; building from source..."
        if [ -d "${SPACE_DIR}/hy3dpaint/packages/custom_rasterizer" ]; then
            (cd "${SPACE_DIR}/hy3dpaint/packages/custom_rasterizer" && ${PIP} install -e .) && INSTALLED=1
        elif [ -d "${SPACE_DIR}/hy3dpaint/custom_rasterizer" ]; then
            (cd "${SPACE_DIR}/hy3dpaint/custom_rasterizer" && ${PIP} install -e .) && INSTALLED=1
        fi
    fi

    if [ "${INSTALLED}" -eq 0 ]; then
        log "WARNING: Could not install custom_rasterizer - texture generation will not work"
    else
        log "custom_rasterizer installed successfully"
    fi

    # Compile DifferentiableRenderer (mesh painter for texture gen)
    DR_DIR="${SPACE_DIR}/hy3dpaint/DifferentiableRenderer"
    if [ -d "${DR_DIR}" ] && [ -f "${DR_DIR}/compile_mesh_painter.sh" ]; then
        log "Compiling DifferentiableRenderer mesh painter..."
        (cd "${DR_DIR}" && bash compile_mesh_painter.sh) || \
            log "WARNING: DifferentiableRenderer compilation failed - texture gen may not work"
    else
        log "WARNING: DifferentiableRenderer not found at ${DR_DIR}"
    fi

    touch "${SPACE_DIR}/.compile_complete"
    log "Custom extensions step complete"
}

# ==============================================================================
# Step 4: Create mock 'spaces' module for ZeroGPU compatibility
# ==============================================================================
create_spaces_mock() {
    log "Creating spaces module mock for ZeroGPU compatibility..."

    cat > "${SPACE_DIR}/spaces.py" << 'MOCK_EOF'
"""
Mock 'spaces' module for running HuggingFace Spaces outside their infrastructure.
Provides no-op implementations of spaces.GPU, spaces.zero, etc.
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
    """Mock @spaces.GPU decorator - no-op on real GPU hardware."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kw: Any) -> Any:
            return fn(*args, **kw)
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


gpu = GPU

zero = types.ModuleType("spaces.zero")
zero.GPU = GPU
zero.gpu = GPU

def _startup() -> None:
    pass

zero.startup = _startup

zero_torch = types.ModuleType("spaces.zero.torch")
zero_torch.GPU = GPU
zero_torch.gpu = GPU
zero.torch = zero_torch

class SpaceConfig:
    def __init__(self):
        self.hardware = "local"
        self.space_id = None
    @property
    def in_space(self) -> bool:
        return False

config = SpaceConfig()

utils = types.ModuleType("spaces.utils")
utils.is_zero_gpu = lambda: False
utils.is_space = lambda: False

sys.modules["spaces"] = sys.modules[__name__]
sys.modules["spaces.zero"] = zero
sys.modules["spaces.zero.torch"] = zero_torch
sys.modules["spaces.utils"] = utils
sys.modules["spaces.config"] = types.ModuleType("spaces.config")
sys.modules["spaces.config"].Config = SpaceConfig

__all__ = ["GPU", "gpu", "zero", "config", "utils"]
MOCK_EOF

    # Uninstall real 'spaces' package if present (it expects HF infrastructure)
    ${PIP} uninstall -y spaces 2>/dev/null || true

    log "Spaces mock created"
}

# ==============================================================================
# Step 5: Patch and launch the Gradio app
# ==============================================================================
launch_app() {
    log "Preparing to launch Hunyuan3D-2.1 Gradio app..."

    mkdir -p "${OUTPUTS_DIR}"

    # Gradio/uvicorn environment
    export GRADIO_SERVER_NAME="0.0.0.0"
    export GRADIO_SERVER_PORT="7860"
    export GRADIO_ANALYTICS_ENABLED="False"

    # HuggingFace cache in workspace (persists across pod restarts with network volume)
    export HF_HOME="${SCRIPT_DIR}/.cache/huggingface"
    mkdir -p "${HF_HOME}"

    cd "${SPACE_DIR}"

    # Python path: Space root + subpackages
    export PYTHONPATH="${SPACE_DIR}:${SPACE_DIR}/hy3dshape:${SPACE_DIR}/hy3dpaint:${PYTHONPATH:-}"

    # Patch the entry point:
    # - Set ENV to non-"Huggingface" so it skips:
    #   * CUDA toolkit download (we have CUDA 12.4 devel)
    #   * custom_rasterizer wheel install (we already compiled from source)
    #   * DifferentiableRenderer compile (already done in step 3)
    # - The app uses uvicorn.run() (not demo.launch()), already binds 0.0.0.0:7860
    ENTRY_POINT="${SPACE_DIR}/gradio_app.py"
    PATCHED="${SPACE_DIR}/_patched_gradio_app.py"

    if [ ! -f "${ENTRY_POINT}" ]; then
        die "gradio_app.py not found at ${ENTRY_POINT}. Space download may be corrupt. Try: rm -f ${SPACE_DIR}/.download_complete && bash startup.sh"
    fi

    python3 -u -c "
with open('${ENTRY_POINT}', 'r') as f:
    content = f.read()

# Verify the expected string exists before patching
if 'ENV = \"Huggingface\"' not in content:
    # Maybe already patched or upstream changed format
    if 'ENV = \"RunPod\"' in content:
        print('Already patched')
    else:
        print('WARNING: Could not find ENV = \"Huggingface\" in gradio_app.py')
        print('Upstream may have changed. Proceeding with unpatched file.')
else:
    content = content.replace('ENV = \"Huggingface\"', 'ENV = \"RunPod\"')
    print('Patched: ENV set to RunPod')

with open('${PATCHED}', 'w') as f:
    f.write(content)
" || die "Failed to patch entry point"

    log "============================================================"
    log "Launching Hunyuan3D-2.1 Gradio UI"
    log "============================================================"
    log ""
    log "Access: http://localhost:7860 (via gpu-cli port forwarding)"
    log ""
    log "First run downloads models (~15GB) - takes 2-5 minutes."
    log "Subsequent runs use cached models."
    log ""

    # Launch via exec (replaces shell process, signals propagate correctly)
    # --low_vram_mode: offloads models to CPU when idle (saves VRAM)
    # --cache-path: store generated meshes for output sync
    exec python3 -u "${PATCHED}" \
        --model_path tencent/Hunyuan3D-2.1 \
        --subfolder hunyuan3d-dit-v2-1 \
        --texgen_model_path tencent/Hunyuan3D-2.1 \
        --cache-path "${OUTPUTS_DIR}" \
        --host 0.0.0.0 \
        --port 7860 \
        --low_vram_mode
}

# ==============================================================================
# Main
# ==============================================================================
main() {
    log "============================================================"
    log "Hunyuan3D-2.1 Setup"
    log "============================================================"

    download_space
    install_space_deps
    compile_extensions
    create_spaces_mock
    launch_app
}

main "$@"
