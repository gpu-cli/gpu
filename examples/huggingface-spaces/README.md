# HuggingFace Spaces Development on GPU CLI

Clone, test, and develop [HuggingFace Spaces](https://huggingface.co/spaces) with full GPU support. Run GPU-intensive Spaces like 3D generation, image synthesis, and LLMs locally before publishing.

## Prerequisites

- GPU CLI installed and authenticated (`gpu auth login`)
- RunPod account with API key configured
- Git installed locally

## Quick Start

```bash
# 1. Clone and setup a Space (e.g., Hunyuan3D-2.1)
gpu run python setup_space.py tencent/Hunyuan3D-2.1

# 2. Run the Space with GPU
gpu run --publish 7860:7860 python run_space.py
```

**Important:** Check the terminal output for the actual URL. GPU CLI may remap the port:

```
[gpu]> Remote 7860 -> http://localhost:61234 (remapped)
```

If this occurs, open the remapped URL, **not** `http://localhost:7860`.

## Development Workflow

### Testing Before Publishing

```bash
# 1. Clone your Space
gpu run python setup_space.py your-username/your-space

# 2. Make changes locally in spaces/your-space/

# 3. Test on GPU
gpu run --publish 7860:7860 python run_space.py

# 4. Push changes back to HuggingFace
cd spaces/your-space
git add .
git commit -m "Update app"
git push
```

### Cloning Existing Spaces

You can clone any public Space for testing or remixing:

```bash
# By owner/name
gpu run python setup_space.py stabilityai/stable-diffusion-3.5-large

# By full URL
gpu run python setup_space.py https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev

# Specific branch
gpu run python setup_space.py owner/space --branch dev
```

## Example: Hunyuan3D-2.1

[Hunyuan3D-2.1](https://huggingface.co/spaces/tencent/Hunyuan3D-2.1) generates high-fidelity 3D assets with PBR materials from images.

### VRAM Requirements

| Task | VRAM Required |
|------|---------------|
| Shape generation only | 10 GB |
| Texture generation only | 21 GB |
| Shape + Texture | 29 GB |

### Running Hunyuan3D-2.1

```bash
# Setup (downloads ~50GB of models)
gpu run python setup_space.py tencent/Hunyuan3D-2.1

# Run with low VRAM mode (if needed)
gpu run --publish 7860:7860 python run_space.py -- --low_vram_mode
```

## Network Volume Setup (Highly Recommended)

Many Spaces download large models (10-50GB+). Use a Network Volume to persist them:

1. Go to [RunPod Storage](https://www.runpod.io/console/user/storage)
2. Click **+ New Network Volume**
3. Choose a datacenter with A100 availability
4. Set size to **200GB** (enough for most models)
5. Copy the **Volume ID**
6. Add to your `gpu.jsonc`:

```jsonc
{
  "network_volume_id": "YOUR_VOLUME_ID"
}
```

**Benefits:**
- Models download once, persist across sessions
- Session startup: 1-2 minutes vs 15+ minutes
- Cost savings from faster startup

## Running Multiple Spaces

The template supports multiple cloned Spaces:

```bash
# Clone several Spaces
gpu run python setup_space.py tencent/Hunyuan3D-2.1
gpu run python setup_space.py stabilityai/stable-diffusion-3.5-large
gpu run python setup_space.py black-forest-labs/FLUX.1-dev

# Run a specific Space
gpu run --publish 7860:7860 python run_space.py --space Hunyuan3D-2.1
gpu run --publish 7860:7860 python run_space.py --space stable-diffusion-3.5-large
```

Without `--space`, the most recently cloned Space is used.

## Passing Arguments to Spaces

Pass arguments to the Space's app using `--`:

```bash
# Pass --low_vram_mode to the Space
gpu run --publish 7860:7860 python run_space.py -- --low_vram_mode

# Pass multiple arguments
gpu run --publish 7860:7860 python run_space.py -- --model_path custom/model --steps 50
```

## GPU Options

| GPU | VRAM | Best For | Cost/hr |
|-----|------|----------|---------|
| A100 80GB | 80GB | Large models, 3D generation | ~$1.74 |
| H100 80GB | 80GB | Fastest inference | ~$3.99 |
| A100 40GB | 40GB | Most Spaces | ~$1.19 |
| RTX 4090 | 24GB | Smaller models, SD/SDXL | ~$0.44 |

To use a different GPU, update `gpu.jsonc`:

```jsonc
{
  "gpu_type": "NVIDIA GeForce RTX 4090",
  "min_vram": 24
}
```

## Troubleshooting

### Space not accessible remotely

Most Spaces need `server_name="0.0.0.0"` in their `demo.launch()` call. The script sets `GRADIO_SERVER_NAME=0.0.0.0` environment variable, but some Spaces override this.

Fix: Edit the Space's `app.py`:

```python
# Before
demo.launch()

# After
demo.launch(server_name="0.0.0.0")
```

### "CUDA out of memory"

1. Use `--low_vram_mode` if the Space supports it
2. Use a larger GPU (A100 80GB)
3. Close other GPU processes

### Slow model downloads

Use a Network Volume to cache models. First download is slow (~15-30 min for large models), but subsequent runs are fast.

### Missing dependencies

Some Spaces have complex dependencies. Check for:

- `setup.py` or `install.sh` scripts in the Space
- Custom CUDA extensions that need compilation
- System packages (install via `apt-get`)

```bash
# Example: Install system dependencies
gpu run bash -c "apt-get update && apt-get install -y ffmpeg && python run_space.py"
```

### Entry point not detected

If the script can't find the entry point:

```bash
# List Python files
ls spaces/your-space/*.py

# Run specific file
gpu run --publish 7860:7860 python run_space.py --entry-point gradio_app.py
```

## Configuration

The `gpu.jsonc` file:

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "huggingface-spaces",
  "provider": "runpod",
  "outputs": ["outputs/", "exports/"],
  "gpu_type": "NVIDIA A100 80GB PCIe",
  "min_vram": 40,
  "workspace_size_gb": 150
}
```

## Files

```
huggingface-spaces/
├── gpu.jsonc          # GPU CLI configuration
├── setup_space.py     # Clone and setup any HF Space
├── run_space.py       # Run the Gradio app
├── requirements.txt   # Base Python dependencies
├── README.md          # This file
├── .gitignore         # Git ignore patterns
└── spaces/            # Cloned Spaces (gitignored)
    ├── Hunyuan3D-2.1/
    ├── stable-diffusion-3.5-large/
    └── ...
```

## Popular GPU-Intensive Spaces

| Space | Description | VRAM |
|-------|-------------|------|
| [tencent/Hunyuan3D-2.1](https://huggingface.co/spaces/tencent/Hunyuan3D-2.1) | 3D asset generation | 29GB |
| [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/spaces/stabilityai/stable-diffusion-3.5-large) | SD 3.5 image generation | 24GB |
| [black-forest-labs/FLUX.1-dev](https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev) | FLUX image generation | 24GB |
| [Qwen/Qwen2-VL](https://huggingface.co/spaces/Qwen/Qwen2-VL) | Vision-language model | 40GB |
| [nvidia/Cosmos](https://huggingface.co/spaces/nvidia/Cosmos) | Video generation | 48GB |

## Resources

- [HuggingFace Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://www.gradio.app/docs)
- [Gradio HF Integration Guide](https://www.gradio.app/guides/using-hugging-face-integrations)
- [Hunyuan3D-2.1 GitHub](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
