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

| Example | Description |
|---------|-------------|
| [ComfyUI](./examples/comfyui/) | Node-based Stable Diffusion UI with custom nodes |
| [Invoke AI](./examples/invokeai/) | Stable Diffusion with a web interface |

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

Create a `gpu.toml` in your project:

```toml
project_id = "my-project"
provider = "runpod"

# Sync outputs back to local machine
outputs = ["output/", "models/"]

# GPU selection
gpu_type = "NVIDIA GeForce RTX 4090"

[environment]
base_image = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
```
