# ComfyUI on GPU CLI

Run [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on remote GPUs with a single command. This example includes essential custom nodes pre-configured for a production-ready workflow.

## Prerequisites

- GPU CLI installed and authenticated (`gpu auth login`)
- RunPod account with API key configured

## Quick Start

```bash
# 1. Setup ComfyUI + custom nodes (first time only, ~2-3 min)
gpu run python setup_comfyui.py

# 2. Start ComfyUI server
gpu run --publish 8188:8188 python ComfyUI/main.py --listen 0.0.0.0
```

**Important:** Check the terminal output for the actual URL. GPU CLI may remap the port and you'll see something like:

```
[gpu]> Remote 8188 -> http://localhost:60984 (remapped)
```

If this occurs, open the remapped URL (e.g., `http://localhost:60984`), **not** `http://localhost:8188`.

## What's Included

The setup script installs these essential custom nodes:

| Node Pack | Description |
|-----------|-------------|
| **ComfyUI-Manager** | Install/update nodes directly from the UI |
| **Impact Pack** | Face detection, detailer, upscaler |
| **WAS Node Suite** | 220+ utility nodes |
| **IPAdapter Plus** | Use reference images to guide style |
| **Advanced ControlNet** | Extended pose/depth control |
| **Comfyroll Nodes** | 175 efficiency & utility nodes |
| **SDXL Prompt Styler** | Style templates for prompts |

## First Time Setup

### 1. Download a Model

ComfyUI needs a checkpoint model to generate images:

1. Click the **Manager** button (bottom of screen)
2. Click **Model Manager**
3. Search for a model:
   - `sd_xl_base_1.0` - SDXL (high quality, needs 8GB+ VRAM)
   - `v1-5-pruned` - SD 1.5 (faster, less VRAM)
4. Click **Install** and wait for download

### 2. Load a Workflow

**Option A: Default Workflow**
- ComfyUI loads with a basic text-to-image workflow
- Select your downloaded model in the "Load Checkpoint" node

**Option B: Import from Community**
- Download workflow JSON from [Civitai](https://civitai.com) or [OpenArt](https://openart.ai/workflows)
- Drag & drop the file into ComfyUI
- Manager auto-detects and installs missing nodes

**Option C: Workflow Gallery**
- Click **Manager** > **Workflow Gallery**
- Browse and load community workflows

**NOTE:**
- We recommended starting with the "Generate Brand Packaging from a Dieline" template using `sample-dieline.jpg`.

### 3. Generate an Image

1. Enter your prompt in the **CLIP Text Encode** node
2. Click **Queue Prompt** (or press `Ctrl+Enter`)
3. Watch progress on each node
4. Output appears in the **Save Image** node

Generated images sync to your local `ComfyUI/output/` folder automatically.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+Enter` | Queue prompt |
| `Ctrl+S` | Save workflow |
| `Space + drag` | Pan canvas |
| `Scroll` | Zoom |
| `Double-click` | Add node (search) |
| `Right-click` | Context menu |
| `Ctrl+M` | Mute selected node |
| `Ctrl+B` | Bypass selected node |

## Troubleshooting

### "torchaudio" errors on startup

The setup script handles this automatically by reinstalling torchaudio. If you see errors, re-run:

```bash
gpu run python setup_comfyui.py
```

### "Missing model" or red nodes

Use ComfyUI Manager to install missing dependencies:
1. Click **Manager** > **Install Missing Custom Nodes**
2. For models: **Manager** > **Model Manager** > Search & install

### Port not accessible

Always use the **remapped port** shown in the terminal output, not 8188.

### Pod stopped unexpectedly

GPU CLI auto-stops pods after 5 minutes of inactivity to save costs. Just run the start command again - your models and setup persist.

## Configuration

The `gpu.toml` file configures the remote environment:

```toml
project_id = "comfyui"
provider = "runpod"

# Files synced back to your machine
outputs = ["ComfyUI/output/"]

# GPU selection (RTX 4090 recommended for SDXL)
gpu_type = "NVIDIA GeForce RTX 4090"

[environment]
base_image = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
```

### GPU Options

| GPU | VRAM | Best For |
|-----|------|----------|
| RTX 4090 | 24GB | SDXL, Flux, large batches |
| RTX 4080 | 16GB | SDXL, most workflows |
| RTX 3090 | 24GB | SDXL, large batches (older) |
| RTX 3080 | 10GB | SD 1.5, smaller models |

## Files

```
comfyui/
├── gpu.toml           # GPU CLI configuration
├── setup_comfyui.py   # Setup script (run once)
├── README.md          # This file
└── ComfyUI/           # Created by setup (gitignored)
    ├── output/        # Generated images (synced locally)
    ├── models/        # Downloaded models
    └── custom_nodes/  # Installed node packs
```
