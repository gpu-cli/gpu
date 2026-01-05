# GPU CLI

Run any code on cloud GPUs with a single command. Just prefix your normal commands with `gpu run`.

```bash
python train.py           # local
gpu run python train.py   # remote GPU
```

## Features

- **Simple** - Prefix commands with `gpu run`, that's it
- **Fast** - Connection pooling, delta sync, real-time output streaming
- **Cost-efficient** - Auto-stops pods when idle (save 60-98% on GPU costs)
- **Multi-cloud** - RunPod, Vast.ai, local Docker
- **Secure** - Zero-trust encryption on supported providers

## Quick Start

```bash
# 1. Install GPU CLI
brew install gpu-cli/tap/gpu

# 2. Authenticate with RunPod
gpu auth login

# 3. Run your code on a remote GPU
gpu run python train.py
```

## Examples

Ready-to-use templates for common AI/ML workflows:

| Example | GPU | Description |
|---------|-----|-------------|
| [ComfyUI](./examples/comfyui/) | RTX 4090 | Node-based Stable Diffusion UI with custom nodes |
| [AI Video Generation](./examples/wan22-video/) | A100 80GB | Generate videos with Wan 2.2 14B |
| [Multi-Agent](./examples/multi-agent/) | A100 80GB | vLLM + CrewAI for agentic workflows |
| [Invoke AI](./examples/invokeai/) | RTX 4090 | Stable Diffusion with a web interface |

## Common Commands

```bash
# Run a command on remote GPU
gpu run python script.py

# Run a server with port forwarding
gpu run -p 8188:8188 python server.py --listen 0.0.0.0

# Open a shell on the remote pod
gpu shell

# View running pods
gpu pods list

# Stop a pod
gpu pods stop <pod-id>

# Interactive dashboard
gpu dashboard
```

## Configuration

Create a `gpu.jsonc` in your project:

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "my-project",
  "provider": "runpod",

  // Sync outputs back to local machine
  "outputs": ["output/", "models/"],

  // GPU selection
  "gpu_type": "NVIDIA GeForce RTX 4090",
  "min_vram": 24,

  // Optional: Persistent storage for models
  // "network_volume_id": "YOUR_VOLUME_ID",

  "environment": {
    "base_image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
  }
}
```

### Network Volumes (Recommended)

For faster startup and persistent model storage, use RunPod Network Volumes. See the [Network Volumes Guide](./docs/network-volumes.md) for setup instructions.

## GPU Options

| GPU | VRAM | Best For | Cost/hr |
|-----|------|----------|---------|
| RTX 4090 | 24GB | Image generation, LoRA training | ~$0.50 |
| RTX 4080 | 16GB | SDXL, most workflows | ~$0.35 |
| A100 40GB | 40GB | 70B models, video generation | ~$1.19 |
| A100 80GB | 80GB | 70B+ models, large batch | ~$1.74 |
| H100 80GB | 80GB | Maximum performance | ~$3.99 |

## Documentation

- [Network Volumes Guide](./docs/network-volumes.md) - Persistent storage for models
- [GPU CLI Docs](https://gpu-cli.sh/docs) - Full documentation
