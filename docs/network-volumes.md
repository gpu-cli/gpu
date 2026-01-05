# Network Volumes Guide

Network Volumes provide persistent storage that survives pod restarts. They're essential for workflows with large models (50GB+) to avoid re-downloading on every session.

## Why Use Network Volumes?

| Without Volume | With Volume |
|----------------|-------------|
| Download 50GB model every session | Download once, reuse forever |
| 15-30 min startup | 1-2 min startup |
| Lose training checkpoints on preemption | Checkpoints persist |
| Higher bandwidth costs | One-time download cost |

## Creating a Network Volume

### Via RunPod Console (Recommended)

1. Go to [RunPod Storage](https://www.runpod.io/console/user/storage)
2. Click **+ New Network Volume**
3. Choose your settings:

| Setting | Recommendation |
|---------|----------------|
| **Name** | Descriptive (e.g., "ai-models-us") |
| **Datacenter** | Match your GPU availability needs |
| **Size** | See sizing guide below |

4. Click **Create**
5. Copy the **Volume ID** (e.g., `abc123xyz`)

### Datacenter Selection

Choose a datacenter with the GPUs you need:

| Datacenter | RTX 4090 | A100 40GB | A100 80GB | H100 |
|------------|----------|-----------|-----------|------|
| US-KS-2 | ✅ | ✅ | ❌ | ❌ |
| US-OR-1 | ✅ | ✅ | ✅ | ✅ |
| EU-RO-1 | ✅ | ✅ | ✅ | ✅ |
| CA-MTL-3 | ❌ | ✅ | ✅ | ✅ |

> **Tip:** GPU CLI automatically deploys pods in the same datacenter as your volume.

### Sizing Guide

| Use Case | Models | Recommended Size |
|----------|--------|------------------|
| ComfyUI (SDXL) | SDXL, LoRAs, ControlNets | 100GB |
| Wan 2.2 Video | Wan 2.2 14B | 200GB |
| LLM (70B) | Qwen 72B, Llama 70B | 150GB |
| Multi-workflow | Multiple large models | 500GB |

## Configuring GPU CLI

Add the volume ID to your `gpu.jsonc`:

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "my-project",
  "provider": "runpod",

  // Add your Network Volume ID here
  "network_volume_id": "YOUR_VOLUME_ID",

  "gpu_type": "NVIDIA GeForce RTX 4090"
}
```

## Volume Mount Location

Network Volumes are mounted at `/runpod-volume/` on the pod.

```
/runpod-volume/
├── huggingface/          # HuggingFace cache
├── models/               # Custom models
├── checkpoints/          # Training checkpoints
└── ...
```

### Using with HuggingFace

GPU CLI automatically sets `HF_HOME` to use the network volume:

```python
# Models automatically cached to /runpod-volume/huggingface/
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/sdxl-base-1.0")
```

### Using with Custom Models

```python
# Read/write directly to network volume
model_path = "/runpod-volume/models/my-model.safetensors"
```

## Best Practices

### 1. One Volume Per Project Type

Don't mix unrelated models. Create separate volumes for:
- Image generation (SDXL, FLUX, ControlNets)
- Video generation (Wan, Hunyuan)
- LLM inference (Qwen, Llama)

### 2. Use Downloads in gpu.jsonc

Pre-download models to the volume:

```jsonc
{
  "network_volume_id": "abc123",
  "download": [
    { "strategy": "hf", "source": "black-forest-labs/FLUX.1-dev" },
    { "strategy": "civitai", "source": "4384", "target": "/runpod-volume/models/loras/" }
  ]
}
```

### 3. Save Checkpoints to Volume

For training, save checkpoints to the network volume:

```python
# In your training script
checkpoint_dir = "/runpod-volume/checkpoints/my-run"
trainer.save_model(checkpoint_dir)
```

### 4. Match Datacenter to GPU Needs

Before creating a volume, check GPU availability:

```bash
# Check available GPUs in each datacenter
# Via RunPod console: Pods > Deploy > Select GPU > See availability
```

## Troubleshooting

### Volume not mounting

1. Verify the volume ID is correct
2. Ensure the pod is in the same datacenter as the volume
3. Check volume isn't attached to another active pod

### "No space left on device"

1. Check volume usage in RunPod console
2. Delete unused models: `rm -rf /runpod-volume/unused-model/`
3. Resize volume in RunPod console (requires detaching first)

### Slow file access

Network volumes are slower than local SSD for random access:
- **Good for:** Large sequential reads (loading models)
- **Slow for:** Frequent small writes (use local SSD, sync at end)

```python
# Write to local SSD during training
output_dir = "/workspace/outputs"

# Then copy to volume at end
import shutil
shutil.copytree("/workspace/outputs", "/runpod-volume/outputs")
```

### Pod won't start in volume's datacenter

The datacenter may be at capacity. Options:
1. Wait and retry later
2. Create a new volume in a different datacenter
3. Use a different GPU type available in that datacenter

## Pricing

Network Volumes are billed per GB per month:

| Size | Monthly Cost |
|------|--------------|
| 100GB | ~$15 |
| 200GB | ~$30 |
| 500GB | ~$75 |

> **Tip:** The one-time cost is usually less than repeatedly downloading large models.

## Deleting a Volume

1. Stop all pods using the volume
2. Go to [RunPod Storage](https://www.runpod.io/console/user/storage)
3. Click the volume
4. Click **Delete**

> **Warning:** Deletion is permanent. Download any important data first.

## Example Workflows

### ComfyUI with SDXL + LoRAs

```jsonc
{
  "project_id": "comfyui",
  "network_volume_id": "abc123",
  "gpu_type": "NVIDIA GeForce RTX 4090"
}
```

Volume contents after setup:
```
/runpod-volume/
├── ComfyUI/
│   ├── models/checkpoints/sd_xl_base_1.0.safetensors
│   ├── models/loras/...
│   └── custom_nodes/...
```

### Video Generation with Wan 2.2

```jsonc
{
  "project_id": "wan22-video",
  "network_volume_id": "def456",
  "gpu_type": "NVIDIA A100 80GB",
  "download": [
    { "strategy": "hf", "source": "Wan-AI/Wan2.2-T2V-A14B-Diffusers" }
  ]
}
```

### Multi-Agent with vLLM

```jsonc
{
  "project_id": "multi-agent",
  "network_volume_id": "ghi789",
  "gpu_type": "NVIDIA A100 80GB",
  "download": [
    { "strategy": "hf", "source": "Qwen/Qwen2.5-72B-Instruct-AWQ" }
  ]
}
```
