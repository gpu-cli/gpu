# HuggingFace Gradio Space Runner

Run any public **Gradio-based** HuggingFace Space on cloud GPUs with two simple commands.

> **Note:** This template only supports **Gradio** Spaces. Streamlit, Docker, and Static Spaces are not supported.

## Quick Start

```bash
# 1. Setup (downloads Space and dependencies)
gpu run python setup.py Lightricks/ltx-video-distilled

# 2. Run the Space
gpu run python run.py Lightricks/ltx-video-distilled
```

Look for the `Running on public URL: https://xxxxx.gradio.live` line in the output and open that URL in your browser.

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
3. Creates a `spaces` module mock for ZeroGPU compatibility
4. Sets environment variables for remote access
5. Launches the Gradio application

The `@spaces.GPU` decorator (used by many Spaces for ZeroGPU) is mocked locally
to be a no-op, allowing Spaces to run on standard GPU infrastructure.

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
gpu run python run.py owner/private-space
```

Or authenticate via CLI:

```bash
huggingface-cli login
gpu run python setup.py owner/private-space
gpu run python run.py owner/private-space
```

## Access Options

By default, the template uses Gradio's `share=True` to create a public `gradio.live` URL. This provides the best performance and is the recommended approach.

### Option 1: Public URL via gradio.live (Default, Recommended)

```bash
gpu run python run.py Lightricks/ltx-video-distilled
```

- ✅ **Best performance** - optimized tunnel infrastructure
- ✅ No port forwarding needed
- ✅ Shareable public URL
- ⚠️ 72-hour URL expiration

### Option 2: Direct localhost (Slower)

If you need local-only access or can't use public URLs, you can use GPU CLI's port forwarding instead:

1. Edit `run.py` and comment out the `patch_entry_point_for_share` call in `launch_app()`:
   ```python
   def launch_app(space_path: Path, entry_point: Path) -> None:
       create_spaces_mock(space_path)
       # patched_entry = patch_entry_point_for_share(entry_point)  # Disable share
       patched_entry = entry_point  # Use original entry point
   ```

2. Run with port publishing:
   ```bash
   gpu run --publish 7860:7860 python run.py Lightricks/ltx-video-distilled
   ```

3. Check the terminal for the remapped port:
   ```
   [gpu]> Remote 7860 -> http://localhost:61234 (remapped)
   ```

4. Open the localhost URL shown (e.g., `http://localhost:61234`)

- ✅ Local-only access (no public URL)
- ⚠️ **Slower performance** than gradio.live
- ⚠️ Need to track port remapping

## Configuration

Default settings in `gpu.jsonc`:
- **GPU**: NVIDIA A100-SXM4-80GB (80GB VRAM) - required for large models like LTX Video 13B
- **Workspace**: 150GB
- **Access**: Public `gradio.live` URL (automatic, no port forwarding needed)

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

### "Space uses 'streamlit' SDK, not Gradio"

This template only supports **Gradio** Spaces. If you see this error:

```
Error: Space 'owner/space' uses 'streamlit' SDK, not Gradio.
This template only supports Gradio-based Spaces.
For Streamlit Spaces: streamlit run app.py
```

The Space uses a different SDK. Alternatives:
- **Streamlit**: Clone manually and run `streamlit run app.py`
- **Docker**: Requires container setup, not supported here
- **Static**: HTML-only Spaces, no Python needed

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
The `gradio.live` URL handles port forwarding automatically, so this shouldn't affect you.
Check the Space's output for the actual port if you need to debug locally.

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
