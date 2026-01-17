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

## Managing Volumes via CLI

GPU CLI provides commands to manage network volumes directly, without using the RunPod console.

### List volumes

```bash
# List all volumes
gpu volume list

# Detailed view with usage and attached pods
gpu volume list --detailed

# JSON output for scripting
gpu volume list --json
```

### Create a volume

```bash
# Interactive (prompts for datacenter)
gpu volume create --name my-models --size 200

# Specify datacenter
gpu volume create --name my-models --size 200 --datacenter US-OR-1

# Create and set as global
gpu volume create --name shared-models --size 500 --set-global
```

### Set global volume

The global volume is shared across all projects (unless overridden by `volume_mode`).

```bash
gpu volume set-global vol_abc123xyz
```

### Check volume usage

```bash
# Current global volume
gpu volume status

# Specific volume
gpu volume status --volume vol_abc123xyz
```

### Extend volume

```bash
gpu volume extend vol_abc123xyz --size 300
```

### Delete volume

```bash
gpu volume delete vol_abc123xyz
```

## Volume Modes

Configure how your project uses network volumes in `gpu.jsonc`:

### Global (default)

Uses the shared global volume set via `gpu volume set-global`. All projects share this volume unless overridden.

```jsonc
{
  "volume_mode": "global"
}
```

### Dedicated

Project-specific volume, isolated from other projects.

```jsonc
{
  "volume_mode": "dedicated",
  "dedicated_volume_id": "vol_project_xyz"  // optional, auto-created if omitted
}
```

### None

No network volume (ephemeral storage only). Good for quick experiments.

```jsonc
{
  "volume_mode": "none"
}
```

## Configuring GPU CLI

Add volume configuration to your `gpu.jsonc`:

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "my-project",
  "provider": "runpod",

  // Option 1: Use global volume (default)
  "volume_mode": "global",

  // Option 2: Use specific volume ID
  "network_volume_id": "YOUR_VOLUME_ID",

  // Option 3: Project-dedicated volume
  // "volume_mode": "dedicated",

  "gpu_type": "NVIDIA GeForce RTX 4090"
}
```

## Volume Reconciliation

GPU CLI automatically keeps local configuration in sync with the cloud provider:

- **Deleted volumes**: If you delete a volume via the RunPod console, GPU CLI detects this and clears stale references from your config on next use.
- **Metadata sync**: Volume name and datacenter are cached locally and updated if they change on the provider.
- **When it runs**: On `gpu volume list`, `gpu run`, daemon startup, and periodically (every 15 minutes).

This means you can safely manage volumes via the RunPod web console — GPU CLI will adapt automatically.

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

Pre-download models and clone tool repositories to the volume:

```jsonc
{
  "network_volume_id": "abc123",
  "download": [
    // HuggingFace models
    { "strategy": "hf", "source": "black-forest-labs/FLUX.1-dev" },

    // Civitai models (supports model ID, version ID, or AIR URN)
    { "strategy": "civitai", "source": "4384" },
    { "strategy": "civitai", "source": "urn:air:sdxl:lora:civitai:328553@368189" },

    // Git repositories (auto-updates when clean)
    { "strategy": "git", "source": "https://github.com/comfyanonymous/ComfyUI", "target": "ComfyUI" },

    // HTTP direct downloads
    { "strategy": "http", "source": "https://example.com/model.bin", "target": "models/model.bin" }
  ]
}
```

**Download strategies:**

| Strategy | Use Case | Example Source |
|----------|----------|----------------|
| `hf` | HuggingFace models | `"black-forest-labs/FLUX.1-dev"` |
| `civitai` | Civitai models | `"4384"` or AIR URN |
| `git` | Tool repositories | `"https://github.com/comfyanonymous/ComfyUI"` |
| `git-lfs` | Repos with large files | HuggingFace repos with LFS |
| `http` | Direct URL downloads | Any HTTP/HTTPS URL |
| `script` | Custom download logic | Shell script path |

**Git strategy features:**
- Shallow clone by default (`--depth 1`)
- Auto-pull on clean working tree
- Preserves user modifications (warns if dirty)
- Supports `branch`, `tag`, or `commit` pinning

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
