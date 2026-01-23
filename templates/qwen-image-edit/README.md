# ComfyUI with Qwen Image Edit

GPU CLI template for running ComfyUI with workflow preloading. Specify any ComfyUI workflow URL and the setup script will download it along with all required models.

## Quick Start

```bash
# 1. Setup (downloads ComfyUI, models, and workflow - ~15-20 min)
gpu run python setup.py

# 2. Run ComfyUI server
gpu run --publish 8188:8188 python run.py
```

Open the URL shown in terminal to access ComfyUI.

## Hardware Requirements

- **GPU**: RTX 4090 or equivalent (24GB VRAM minimum)
- **Storage**: ~30GB for models + workspace
- **Default config**: `gpu_type: "NVIDIA GeForce RTX 4090"`

## Using a Different Workflow

Edit `gpu.jsonc` and change the `workflow` URL:

```jsonc
{
  "workflow": "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/image_qwen_image_edit.json"
}
```

The setup script will:
1. Download your workflow
2. Parse it to find required models (UNETLoader, CLIPLoader, VAELoader, LoraLoader nodes)
3. Download all models from HuggingFace
4. Place the workflow in `ComfyUI/user/default/workflows/`

## Finding Workflows

- **Official ComfyUI templates**: https://github.com/Comfy-Org/workflow_templates
- **ComfyUI documentation**: https://docs.comfy.org/tutorials
- **Community workflows**: Export any ComfyUI workflow as JSON

## Default Workflow: Qwen Image Edit

The default workflow uses [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit-2511), a 20B parameter model for precise image editing with natural language.

**Capabilities:**
- Edit images with text prompts ("add sunglasses", "change sky to sunset")
- Precise text editing within images
- Style transfer and appearance changes
- Object addition/removal

**Example prompts:**
- "make the sky overcast and add rain"
- "add sunglasses to the person"
- "change the car color to red"
- "remove the background"

**Models downloaded (~26GB):**
| Model | Size | Purpose |
|-------|------|---------|
| qwen_image_edit_fp8_e4m3fn.safetensors | ~20GB | Main diffusion model |
| qwen_2.5_vl_7b_fp8_scaled.safetensors | ~5GB | Text encoder |
| qwen_image_vae.safetensors | ~1GB | VAE |
| Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors | ~500MB | LoRA for fast 4-step generation |

## ComfyUI Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+Enter` | Queue prompt |
| `Ctrl+Shift+Enter` | Queue prompt (front) |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Space` | Pan canvas |
| `Ctrl+M` | Mute/unmute node |
| `Ctrl+B` | Bypass node |

## Troubleshooting

### "Model not found" error
Run setup again to re-download models:
```bash
gpu run python setup.py
```

### Out of memory (OOM)
The Qwen models require ~24GB VRAM. Ensure you're using an RTX 4090 or equivalent:
```jsonc
{
  "gpu_type": "NVIDIA GeForce RTX 4090",
  "min_vram": 24
}
```

### Port already in use
ComfyUI defaults to port 8188. The GPU CLI will remap it automatically.

### Workflow not loading
Check that your workflow URL returns valid JSON. Test by opening the URL in a browser.

## Files

- `gpu.jsonc` - GPU CLI configuration with workflow URL
- `setup.py` - Downloads ComfyUI, models, and workflow
- `run.py` - Starts the ComfyUI server
- `sample-image.jpg` - Test image for editing workflows

## Output Syncing

Generated images are saved to `ComfyUI/output/` and automatically synced back to your local machine.
