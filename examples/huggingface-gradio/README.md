# HuggingFace Gradio Space Runner

Run any public Gradio-based HuggingFace Space on cloud GPUs with two simple commands.

## Quick Start

```bash
# 1. Setup (downloads Space and dependencies)
gpu run python setup.py Lightricks/ltx-video-distilled

# 2. Run the Space (with port forwarding to localhost:7860)
gpu run -p 7860:7860 python run.py Lightricks/ltx-video-distilled
```

Open http://localhost:7860 in your browser to access the Space.

## How It Works

This template uses a two-phase architecture:

**Phase 1 - Setup (`setup.py`):**
1. Validates the Space exists on HuggingFace
2. Downloads the Space using `huggingface_hub.snapshot_download()`
3. Installs system packages from `packages.txt` (if present)
4. Installs Python dependencies from the Space's `requirements.txt`
5. Creates a setup marker for validation

**Phase 2 - Run (`run.py`):**
1. Validates setup is complete
2. Finds the Gradio entry point (app.py, main.py, etc.)
3. Sets environment variables for remote access
4. Launches the Gradio application

The `@spaces.GPU` decorator is effect-free outside ZeroGPU environments,
so Spaces run natively without any patching or mocking.

## Tested Spaces

| Space | Description | VRAM |
|-------|-------------|------|
| [Lightricks/ltx-video-distilled](https://huggingface.co/spaces/Lightricks/ltx-video-distilled) | Fast video generation (13B model) | ~40GB |
| [tencent/Hunyuan3D-2.1](https://huggingface.co/spaces/tencent/Hunyuan3D-2.1) | 3D asset generation | ~40GB |
| [multimodalart/FLUX.2-dev-turbo](https://huggingface.co/spaces/multimodalart/FLUX.2-dev-turbo) | 8-step turbo FLUX.2 | ~24GB |

## Private Spaces

For private Spaces, set `HF_TOKEN` before running:

```bash
export HF_TOKEN=hf_xxxx
gpu run python setup.py owner/private-space
gpu run -p 7860:7860 python run.py owner/private-space
```

Or authenticate via CLI:

```bash
huggingface-cli login
gpu run python setup.py owner/private-space
gpu run -p 7860:7860 python run.py owner/private-space
```

## Configuration

Default settings in `gpu.jsonc`:
- **GPU**: NVIDIA A100-SXM4-80GB (80GB VRAM) - required for large models like LTX Video 13B
- **Workspace**: 150GB
- **Port forwarding**: Use `-p 7860:7860` flag (Gradio default port)

For smaller Spaces (<24GB VRAM), edit `gpu.jsonc`:

```jsonc
"gpu_type": "NVIDIA GeForce RTX 4090",
"min_vram": 24
```

For mid-size Spaces (24-40GB VRAM):

```jsonc
"gpu_type": "NVIDIA A100 80GB PCIe",
"min_vram": 40
```

## Re-running Setup

Setup is idempotent - safe to re-run:

```bash
# Re-run to update dependencies
gpu run python setup.py Lightricks/ltx-video-distilled
```

To force a fresh download:

```bash
rm -rf spaces/ltx-video-distilled
gpu run python setup.py Lightricks/ltx-video-distilled
```

## Troubleshooting

### Space not accessible remotely

The scripts set `GRADIO_SERVER_NAME=0.0.0.0` automatically. If you still have issues,
check if the Space hardcodes `demo.launch()` without server_name. You may need to
edit the downloaded Space's entry point:

```python
# Change this:
demo.launch()

# To this:
demo.launch(server_name="0.0.0.0")
```

### Missing system dependencies

Some Spaces require additional system packages. Add them to `gpu.jsonc`:

```jsonc
"environment": {
  "system": {
    "apt": [{ "name": "package-name" }]
  }
}
```

Common packages needed:
- `ffmpeg` - Video processing
- `libgl1-mesa-glx` - OpenGL support
- `libglib2.0-0` - GLib library

### CUDA out of memory

Use a larger GPU or check if the Space has a low-VRAM option:

```jsonc
"gpu_type": "NVIDIA A100-SXM4-80GB"
```

### Entry point not found

Some Spaces use non-standard entry points. Check the Space's README and run manually:

```bash
cd spaces/space-name
python custom_entry.py
```

### Python dependency conflicts

If the Space's requirements conflict with base dependencies, try:

```bash
# Clear the Space and re-run setup
rm -rf spaces/space-name
gpu run python setup.py owner/space-name
```

### Port varies by Space

Different Spaces may use different ports. Common ports are 7860 and 7861.
Check the Space's output for the actual port and adjust the `-p` flag:

```bash
# For port 7861
gpu run -p 7861:7861 python run.py owner/space-name
```

## Files

```
huggingface-gradio/
├── gpu.jsonc         # GPU CLI configuration
├── setup.py          # Phase 1: Download + install
├── run.py            # Phase 2: Validate + launch
├── requirements.txt  # Base Python dependencies
├── README.md         # This file
└── spaces/           # Downloaded Spaces (gitignored)
```

## Network Volume (Recommended)

For faster startup with large models, use a RunPod Network Volume:

1. Create a 200GB volume at [RunPod Storage](https://www.runpod.io/console/user/storage)
2. Add to `gpu.jsonc`: `"network_volume_id": "YOUR_VOLUME_ID"`

Benefits:
- Models persist across sessions
- Faster startup after first run
- Cost savings from reduced download time

## Resources

- [HuggingFace Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://www.gradio.app/docs)
- [ZeroGPU Documentation](https://huggingface.co/docs/hub/spaces-zerogpu)
- [GPU CLI Documentation](https://gpu-cli.sh/docs)
