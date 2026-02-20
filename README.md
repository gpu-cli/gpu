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
- **Teams** — Organizations with pooled sessions, sub-accounts, and CI/CD service tokens (Team & Enterprise)

## Quick Start

```bash
# 1. Install GPU CLI
curl -fsSL https://gpu-cli.sh/install.sh | sh

# 2. Run your code on a remote GPU
gpu run python train.py
```

---

## Claude Code Plugin

This repo includes a Claude Code plugin that supercharges GPU CLI with AI assistance. Describe what you want in plain English, and Claude generates complete, runnable GPU workflows.

### What's Included

#### Skills (Automatic AI Capabilities)

| Skill | Description |
|-------|-------------|
| **gpu-workflow-creator** | Transform natural language into complete GPU projects |
| **gpu-ml-trainer** | LLM fine-tuning, LoRA training, classifier training |
| **gpu-inference-server** | Set up vLLM, TGI, or custom inference APIs |
| **gpu-media-processor** | Whisper transcription, voice cloning, video generation |
| **gpu-cost-optimizer** | GPU selection advice and cost optimization |
| **gpu-debugger** | Debug failed runs, OOM errors, connectivity issues |

#### Slash Commands

| Command | Description |
|---------|-------------|
| `/gpu-cli:gpu-create` | Create a complete project from a description |
| `/gpu-cli:gpu-optimize` | Analyze and optimize your gpu.jsonc |
| `/gpu-cli:gpu-debug` | Debug a failed GPU run |
| `/gpu-cli:gpu-quick` | Quick-start common workflows |

### Example Conversations

**Create a LoRA training project:**
```
You: I want to train a LoRA on photos of my dog so I can generate images of it

Claude: [Generates complete project with gpu.jsonc, train.py, requirements.txt, README.md]
```

**Set up a private LLM API:**
```
You: Set up Llama 3.1 70B as a private ChatGPT-like API

Claude: [Generates vLLM server config with OpenAI-compatible endpoints]
```

**Debug an error:**
```
You: /gpu-cli:gpu-debug CUDA out of memory when running FLUX

Claude: [Analyzes error, suggests reducing batch size or upgrading to A100]
```

**Optimize costs:**
```
You: /gpu-cli:gpu-optimize

Claude: [Reviews gpu.jsonc, suggests RTX 4090 instead of A100 for your workload, saving 75%]
```

---

## Templates

Ready-to-use templates for common AI/ML workflows:

| Template | GPU | Description |
|----------|-----|-------------|
| [Ollama Models](./templates/ollama-models/) | RTX 4090 | Run LLMs with Ollama - includes Web UI and OpenAI-compatible API |
| [vLLM Models](./templates/vllm-models/) | RTX 4090 | High-performance LLM inference with vLLM |
| [Background Removal](./templates/background-removal/) | RTX 4090 | Remove backgrounds from images using AI |
| [CrewAI Stock Analysis](./templates/crewai-stock-analysis/) | RTX 4090 | Multi-agent stock analysis with CrewAI + Ollama |
| [HuggingFace Gradio](./templates/huggingface-gradio/) | RTX 4090 | Run HuggingFace models with Gradio UI |
| [Qwen Image Edit](./templates/qwen-image-edit/) | RTX 4090 | Edit images using Qwen vision model |

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
gpu stop

# Interactive dashboard
gpu dashboard
```

### Team Management

```bash
# Create an organization
gpu org create "My Team"

# Switch to org context
gpu org switch my-team

# Invite a teammate
gpu org invite alice@example.com --role admin

# Create a CI/CD service account
gpu org service-account create --name "github-actions"
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
  "gpu_type": "RTX 4090",
  "min_vram": 24,

  // Optional: Pre-download models
  "download": [
    { "strategy": "hf", "source": "black-forest-labs/FLUX.1-dev", "allow": "*.safetensors" }
  ],

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
| RTX 4090 | 24GB | Image generation, LoRA training | ~$0.44 |
| RTX 4080 | 16GB | SDXL, most workflows | ~$0.35 |
| A100 40GB | 40GB | 70B models, video generation | ~$1.29 |
| A100 80GB | 80GB | 70B+ models, large batch | ~$1.79 |
| H100 80GB | 80GB | Maximum performance | ~$3.99 |

## Documentation

- [Network Volumes Guide](./docs/network-volumes.md) - Persistent storage for models
- [Organizations Guide](https://gpu-cli.sh/docs/organizations) - Team billing, sub-accounts, and service tokens
- [GPU CLI Docs](https://gpu-cli.sh/docs) - Full documentation

## License

MIT
