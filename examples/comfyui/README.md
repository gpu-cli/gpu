# ComfyUI + Qwen Image Edit on GPU CLI

Run [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with **Qwen Image Edit** pre-configured on remote GPUs. Edit images using natural language prompts like "make the sky overcast" or "add sunglasses to the person".

## Prerequisites

- GPU CLI installed and authenticated (`gpu auth login`)
- RunPod account with API key configured

## Quick Start

```bash
# 1. Setup ComfyUI + Qwen models (first time only, ~5-10 min)
gpu run python setup_comfyui.py

# 2. Start ComfyUI server
gpu run --publish 8188:8188 python ComfyUI/main.py --listen 0.0.0.0
```

**Important:** Check the terminal output for the actual URL. GPU CLI may remap the port:

```
[gpu]> Remote 8188 -> http://localhost:60984 (remapped)
```

## Featured: Qwen Image Edit

This example comes pre-configured with **Qwen Image Edit** - a powerful image editing model that understands natural language instructions.

### Using Qwen Image Edit

1. Open ComfyUI in your browser (use the remapped URL)
2. Click the **Template** icon (grid icon, top-left)
3. Navigate to **Image** → **Qwen Image Edit**
4. Load the workflow
5. Upload an image in the **LoadImage** node
6. Enter your edit instruction (e.g., "make it night time", "add rain")
7. Click **Queue Prompt** (or `Ctrl+Enter`)

### Example Prompts

| Prompt | Effect |
|--------|--------|
| "make the sky overcast and add rain" | Weather change |
| "add sunglasses to the person" | Object addition |
| "change the car color to red" | Color modification |
| "make it look like a painting" | Style transfer |
| "remove the background" | Background removal |

### Performance

- **4 steps** with Lightning LoRA: ~55 seconds (fast preview)
- **8 steps**: ~90 seconds (balanced)
- **50 steps**: ~95 seconds (high quality)

## What's Pre-Installed

### Models (~20GB, auto-downloaded)
- `qwen_image_edit_fp8_e4m3fn.safetensors` - Diffusion model
- `qwen_2.5_vl_7b_fp8_scaled.safetensors` - Vision-Language encoder
- `qwen_image_vae.safetensors` - VAE
- `Qwen-Image-Lightning-4steps-V1.0.safetensors` - Fast generation LoRA

### Custom Nodes
| Node Pack | Description |
|-----------|-------------|
| **ComfyUI-Manager** | Install/update nodes from the UI |
| **Comfyui-QwenEditUtils** | Qwen image edit text encoding |
| **Impact Pack** | Face detection, detailer, upscaler |
| **WAS Node Suite** | 220+ utility nodes |
| **IPAdapter Plus** | Reference image style guidance |
| **Advanced ControlNet** | Pose/depth control |

## Network Volume Setup (Recommended)

For faster startup and persistent model storage:

1. Go to [RunPod Storage](https://www.runpod.io/console/user/storage)
2. Click **+ New Network Volume**
3. Choose a datacenter with RTX 4090 availability
4. Set size to **150GB** (Qwen models are ~20GB)
5. Copy the **Volume ID** and add to `gpu.jsonc`:

```jsonc
{
  "network_volume_id": "YOUR_VOLUME_ID"
}
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+Enter` | Queue prompt |
| `Ctrl+S` | Save workflow |
| `Space + drag` | Pan canvas |
| `Scroll` | Zoom |
| `Double-click` | Add node (search) |
| `Ctrl+M` | Mute selected node |

## Troubleshooting

### Models not loading

The Qwen models (~20GB) are pre-downloaded during setup. If missing:
1. Check that setup completed successfully
2. Models should be in `ComfyUI/models/`
3. Re-run setup if needed: `gpu run python setup_comfyui.py`

### "CUDA out of memory"

Qwen Image Edit requires 20-24GB VRAM. The RTX 4090 (24GB) is configured by default. If you see memory errors:
1. Close other GPU processes
2. Reduce image resolution
3. Use 4-step Lightning LoRA for faster generation

### Port not accessible

Always use the **remapped port** shown in the terminal output, not 8188.

## Configuration

The `gpu.jsonc` file configures the remote environment:

```jsonc
{
  "project_id": "comfyui",
  "provider": "runpod",
  "outputs": ["ComfyUI/output/"],
  "gpu_type": "NVIDIA GeForce RTX 4090",
  "min_vram": 24,
  "workspace_size_gb": 100,
  "download": [
    {
      "strategy": "hf",
      "source": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
      "target": "ComfyUI/models/"
    }
  ]
}
```

## Files

```
comfyui/
├── gpu.jsonc          # GPU CLI configuration
├── setup_comfyui.py   # Setup script (run once)
├── README.md          # This file
└── ComfyUI/           # Created by setup (gitignored)
    ├── output/        # Generated images (synced locally)
    ├── models/        # Qwen + other models
    └── custom_nodes/  # Installed node packs
```

## Resources

- [Qwen Image Edit Tutorial](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit)
- [ComfyUI Documentation](https://docs.comfy.org/)
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
