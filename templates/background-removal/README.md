# Background Removal with InvokeAI + BiRefNet

GPU CLI template for InvokeAI with the BiRefNet background removal node and model pre-installed.

## Quick Start

```bash
# Start InvokeAI with BiRefNet (~5-10 min first run, downloads models)
gpu use .
```

Open the URL shown in terminal to access InvokeAI.

## Pre-installed Model

Setup downloads BiRefNet (~500MB) for background removal.

**Optional:** Add Stable Diffusion models via InvokeAI's Model Manager if you need image generation.

## Using BiRefNet Background Removal

**Note:** You'll see "No models installed" on the main screen - this is expected. BiRefNet is accessed through the Workflow Editor, not the main Launchpad.

### Step-by-step:

1. **Open InvokeAI** in your browser (use the URL shown in terminal)

2. **Go to Workflow Editor** - Click the "Workflow Editor" tab at the top (next to "Launchpad")

3. **Add the BiRefNet node:**
   - Right-click on the canvas (the dotted area)
   - Search for "BiRefNet" or "Remove Background"
   - Click to add the node

4. **Load your image:**
   - Click on the dark **"Image"** box inside the node
   - Select an image from your computer

5. **Configure and run:**
   - Check the **"Save To Gallery"** checkbox
   - Press **Ctrl+Enter** or click **Invoke** to process

6. **Get your result:**
   - The output image (with transparent background) appears in the gallery on the right
   - Right-click to download as PNG

## Hardware Requirements

- **GPU**: RTX 6000 Ada or equivalent (24GB VRAM)
- **Storage**: ~10GB for InvokeAI + BiRefNet model
- **Python**: 3.11+ required

## BiRefNet Features

- High-quality foreground/background separation
- Works with complex edges (hair, fur, transparent objects)
- State-of-the-art dichotomous image segmentation
- Model: [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)

## Files

- `gpu.jsonc` - GPU CLI configuration (ports, startup, environment)
- `startup.sh` - Startup script: installs deps, runs setup, starts InvokeAI server
- `install_invokeai.py` - Idempotent setup: downloads InvokeAI + BiRefNet node and model
- `run.py` - Standalone InvokeAI server launcher (used by startup.sh)

## Troubleshooting

### "No models installed" warning
This is normal! The warning refers to Stable Diffusion models, not BiRefNet. BiRefNet is loaded separately and works via the Workflow Editor.

### "Unable to Queue" / "missingInputForField" error
You need to load an image first. Click the dark "Image" box in the BiRefNet node and select an image.

### "Module not found" errors
Run setup again:
```bash
gpu use .
```

### Out of memory
BiRefNet works best with 24GB VRAM. For smaller GPUs, consider using smaller image sizes.

### Port already in use
InvokeAI uses port 9090. GPU CLI will remap it automatically.

## Output Syncing

Generated images are saved to `invokeai/outputs/` and automatically synced back to your local machine.
