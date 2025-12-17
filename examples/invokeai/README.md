# InvokeAI on GPU CLI

Run [InvokeAI](https://github.com/invoke-ai/InvokeAI) on remote GPUs with a single command. InvokeAI is a professional-grade Stable Diffusion interface with an intuitive web UI.

## Prerequisites

- GPU CLI installed and authenticated (`gpu auth login`)
- RunPod account with API key configured

## Quick Start

```bash
# 1. Setup InvokeAI (first time only, ~3-5 min)
gpu run python setup_invokeai.py

# 2. Start InvokeAI server
gpu run -p 9090:9090 python run_invokeai.py
```

**Important:** Check the terminal output for the actual URL. GPU CLI may remap the port:

```
[gpu]> Remote 9090 -> http://localhost:61234 (remapped)
```

If this occurs, open the remapped URL, **not** `http://localhost:9090`.

## Features

InvokeAI includes:

- **Unified Canvas** - Inpainting, outpainting, and image editing
- **Node Editor** - Build complex generation workflows
- **Model Manager** - Download and manage models from the UI
- **ControlNet** - Pose, depth, and edge-guided generation
- **LoRA & Embeddings** - Fine-tuned model support
- **Upscaling** - Built-in ESRGAN upscalers

## First Time Setup

### 1. Download Models

On first launch, InvokeAI opens to the Model Manager:

1. Go to **Model Manager** tab
2. Click **Starter Models** to see recommended models
3. Select models to download:
   - `stable-diffusion-xl-base-1.0` - SDXL (recommended)
   - `stable-diffusion-v1-5` - SD 1.5 (faster, less VRAM)
4. Click **Download** and wait for completion

### 2. Generate Images

1. Go to **Text to Image** tab
2. Enter a prompt
3. Select your model from the dropdown
4. Click **Generate** (or press `Ctrl+Enter`)

Generated images sync to your local `invokeai/outputs/` folder automatically.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+Enter` | Generate image |
| `Ctrl+Shift+Enter` | Generate with current seed |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |
| `G` | Toggle gallery |
| `O` | Toggle options panel |

## Troubleshooting

### Installation takes too long

InvokeAI has many dependencies. The first setup may take 5-10 minutes. Subsequent runs reuse the cached installation.

### "CUDA out of memory"

Try these in order:
1. Use a smaller model (SD 1.5 instead of SDXL)
2. Reduce image dimensions
3. Disable xformers: add `--no-xformers` flag to the run command

### Models not appearing

Models are stored in `invokeai/models/`. Ensure the Model Manager download completed successfully.

### Port not accessible

Always use the **remapped port** shown in the terminal output, not 9090.

## Configuration

The `gpu.toml` file configures the remote environment:

```toml
project_id = "invokeai"
provider = "runpod"
outputs = ["invokeai/outputs/"]
gpu_type = "NVIDIA GeForce RTX 4090"

[environment]
base_image = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
```

### GPU Options

| GPU | VRAM | Best For |
|-----|------|----------|
| RTX 4090 | 24GB | SDXL, large batches, high resolution |
| RTX 4080 | 16GB | SDXL, most workflows |
| RTX 3090 | 24GB | SDXL, large batches |
| RTX 3080 | 10GB | SD 1.5, smaller models |

## Files

```
invokeai/
├── gpu.toml            # GPU CLI configuration
├── setup_invokeai.py   # Setup script (run once)
├── run_invokeai.py     # Run script (starts server)
├── README.md           # This file
└── invokeai/           # Created by setup (gitignored)
    ├── outputs/        # Generated images (synced locally)
    ├── models/         # Downloaded models
    └── databases/      # InvokeAI configuration
```

## Testing Your Setup

### Quick Test (Recommended First Model)

For the fastest test, use **Stable Diffusion 1.5** (smaller, ~4GB):

1. Open InvokeAI in your browser (use the remapped URL from terminal)
2. Click **Model Manager** in the left sidebar
3. Click **Starter Bundle** or search for `stable-diffusion-v1-5`
4. Click **Install** - wait 2-3 minutes for download
5. Go to **Text to Image** tab
6. Select `stable-diffusion-v1-5` from the model dropdown
7. Enter a prompt: `a photo of a cat wearing sunglasses, high quality`
8. Click **Invoke** (or press `Ctrl+Enter`)
9. Wait ~5-10 seconds for generation

### For Higher Quality (SDXL)

Once comfortable, try **SDXL** (larger, ~6GB, better quality):

1. In Model Manager, search for `stable-diffusion-xl-base-1.0`
2. Install and select it
3. Use prompts like: `a majestic lion in a forest, cinematic lighting, 8k`
4. SDXL works best at 1024x1024 resolution

### Verify Output Sync

Generated images should automatically sync to your local machine:

```bash
ls invokeai/outputs/
```

## Resources

- [InvokeAI Documentation](https://invoke-ai.github.io/InvokeAI/)
- [InvokeAI Discord](https://discord.gg/invoke-ai)
- [Model Downloads](https://huggingface.co/models?pipeline_tag=text-to-image)
