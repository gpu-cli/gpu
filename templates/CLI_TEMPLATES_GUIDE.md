# GPU CLI Templates Guide

A comprehensive guide to understanding and creating templates for the GPU CLI tool.

## Overview

Templates in GPU CLI provide pre-configured project scaffolds that can be applied to quickly set up GPU workloads. They support interactive inputs, automatic configuration merging, and session persistence for resumable workflows.

Templates can be sourced from:
- **Local paths** - Your filesystem
- **GitHub repositories** - Any public repo with a valid template
- **Official registry** - Curated templates maintained by the GPU CLI team

---

## Template Structure

A valid template is a directory containing at minimum a configuration file. The typical structure:

```
my-template/
├── gpu.jsonc              # Required: Main configuration file
├── pyproject.toml         # Python dependencies (recommended)
├── Dockerfile             # Custom environment (optional)
├── main.py                # Entry point script(s)
└── README.md              # Documentation (recommended)
```

### Required Files

| File | Purpose |
|------|---------|
| `gpu.jsonc` or `gpu.json` | Main configuration defining project settings, GPU requirements, and outputs |

### Recommended Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Python dependencies using standard packaging format |
| `README.md` | Documentation explaining template purpose and usage |
| `*.py` | Application code |

### Optional Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Custom container environment |
| `gpu.toml` | Alternative TOML configuration format |

---

## Configuration Format

Templates use `gpu.jsonc` (JSON with Comments) as the primary configuration format.

### Schema Reference

All configuration files should include a schema reference for IDE autocomplete:

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  // ... configuration
}
```

### Core Configuration Fields

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",

  // Required
  "project_id": "my-project",           // Unique identifier
  "provider": "runpod",                 // "runpod" or "docker"

  // GPU Requirements
  "gpu_type": "RTX 4090",               // Target GPU model
  "gpu_count": 1,                       // Number of GPUs
  "min_vram": 24,                       // Minimum VRAM in GB

  // Storage
  "outputs": ["results/", "models/"],   // Directories to sync back
  "workspace_size_gb": 50,              // Workspace size
  "network_volume_id": "abc123",        // Persistent volume ID

  // Security
  "encryption": false,                  // Enable data encryption

  // Environment
  "environment": {
    "system": {
      "apt": [{ "name": "ffmpeg" }]     // System packages
    }
  },

  // Downloads
  "download": [
    { "strategy": "hf", "source": "owner/model" },
    { "strategy": "civitai", "source": "12345", "target": "/workspace/models/" }
  ]
}
```

### Template Metadata

Templates can include metadata for display purposes:

```jsonc
{
  "template": {
    "name": "My Template",
    "description": "A description of what this template does",
    "author": "Your Name"
  }
}
```

### Provider Options

| Provider | Description |
|----------|-------------|
| `runpod` | Remote GPU cloud (production) |
| `docker` | Local Docker environment (development/testing) |

### GPU Types

Common GPU types include:
- `RTX 4090`, `RTX 5090`
- `NVIDIA RTX 6000 Ada`
- `NVIDIA H100 PCIe`
- `NVIDIA A100`

---

## Template Sources

### Local Paths

Reference templates from your filesystem:

```bash
gpu use ./my-template           # Relative path
gpu use ../templates/ml-train   # Parent directory
gpu use /absolute/path/template # Absolute path
gpu use ~/templates/my-template # Home directory
```

### GitHub Repositories

Reference templates from GitHub:

```bash
gpu use owner/repo              # Default branch
gpu use owner/repo@v1.0.0       # Specific tag
gpu use owner/repo@main         # Specific branch
gpu use owner/repo@abc123       # Specific commit
gpu use github:owner/repo       # Explicit prefix
gpu use github:owner/repo@ref   # Explicit with ref
```

GitHub templates are fetched via the tarball API and cached locally at `~/.gpu-cli/templates/github/`.

### Official Registry

Use curated templates by name:

```bash
gpu use hello-gpu               # Official template
gpu use hello-gpu@v1.0.0        # With version
```

Currently available official templates:
- `hello-gpu` - Minimal test template

---

## CLI Commands

### `gpu use <template>`

Apply a template to the current directory:

```bash
gpu use <template>              # Apply template
gpu use                         # Resume existing session
```

**Options:**

| Flag | Description |
|------|-------------|
| `--name <name>` | Project/session name (defaults to directory name) |
| `--yes` | Skip interactive prompts, use defaults |
| `--dry-run` | Show configuration without creating files |

**Examples:**

```bash
# Apply a local template
gpu use ./templates/stable-diffusion

# Apply from GitHub with specific tag
gpu use myorg/gpu-templates@v2.0.0 --name my-project

# Resume previous session
gpu use

# Preview without applying
gpu use hello-gpu --dry-run
```

### `gpu template clear-cache`

Clear cached templates:

```bash
gpu template clear-cache        # Interactive confirmation
gpu template clear-cache --force # Skip confirmation
```

### `gpu init`

Initialize a new project with encryption and configuration:

```bash
gpu init
```

Creates a `gpu.jsonc` file and stores LUKS keys in the OS keychain.

---

## Session Management

When you apply a template, GPU CLI creates a session file at `.gpu/template.json` in your project directory.

### Session File Structure

```json
{
  "source": {
    "type": "github",
    "owner": "myorg",
    "repo": "templates",
    "ref": "v1.0.0",
    "subpath": "stable-diffusion"
  },
  "config": {
    "gpu_type": "RTX 4090",
    "outputs": ["outputs/"]
  },
  "input_values": {
    "model": "flux-dev",
    "steps": 50
  },
  "project_id": "generated-project-id",
  "created_at": "2024-01-15T10:30:00Z",
  "last_used_at": "2024-01-15T14:45:00Z"
}
```

### Session Benefits

- **Resume capability**: Run `gpu use` without arguments to resume
- **Input persistence**: User's interactive answers are stored
- **Self-contained**: Session file contains all needed config
- **Audit trail**: Timestamps track creation and usage

---

## Input Specifications

Templates can define interactive inputs that prompt users during `gpu use`. There are 10 input types available:

### 1. Select (Dropdown)

```jsonc
{
  "type": "select",
  "key": "model",
  "label": "Base Model",
  "options": [
    { "value": "flux-dev", "label": "FLUX.1 Dev", "description": "High quality" },
    { "value": "sdxl", "label": "SDXL", "description": "Fast generation" }
  ],
  "default": "flux-dev",
  "required": true
}
```

### 2. Text (Free-form)

```jsonc
{
  "type": "text",
  "key": "prompt",
  "label": "Generation Prompt",
  "placeholder": "A beautiful landscape...",
  "pattern": "^.{10,500}$",
  "required": true
}
```

### 3. Number (Numeric with bounds)

```jsonc
{
  "type": "number",
  "key": "steps",
  "label": "Inference Steps",
  "min": 1,
  "max": 100,
  "default": 50,
  "step": 1
}
```

### 4. Boolean (Toggle)

```jsonc
{
  "type": "boolean",
  "key": "enable_safety",
  "label": "Enable Safety Checker",
  "default": true
}
```

### 5. Model (Model selection)

```jsonc
{
  "type": "model",
  "key": "checkpoint",
  "label": "Model Checkpoint",
  "provider": "huggingface",
  "filter": {
    "pipeline": "text-to-image"
  }
}
```

### 6. GPU (GPU type selection)

```jsonc
{
  "type": "gpu",
  "key": "gpu_type",
  "label": "GPU Type",
  "min_vram": 24,
  "show_pricing": true
}
```

### 7. Price (Budget input)

```jsonc
{
  "type": "price",
  "key": "max_cost",
  "label": "Maximum Cost",
  "currency": "USD",
  "min": 0.10,
  "max": 100.00
}
```

### 8. File (File selection)

```jsonc
{
  "type": "file",
  "key": "workflow",
  "label": "ComfyUI Workflow",
  "extensions": [".json"],
  "description": "Select a workflow file"
}
```

### 9. Secret (API key input)

```jsonc
{
  "type": "secret",
  "key": "hf_token",
  "label": "HuggingFace Token",
  "storage": "keychain",
  "description": "Your HF access token"
}
```

### 10. Multiselect (Multiple choice)

```jsonc
{
  "type": "multiselect",
  "key": "features",
  "label": "Enable Features",
  "options": [
    { "value": "upscale", "label": "Upscaling" },
    { "value": "face_fix", "label": "Face Restoration" },
    { "value": "hires", "label": "Hi-Res Fix" }
  ],
  "default": ["upscale"]
}
```

---

## Merging Strategy

When applying a template to an existing project, GPU CLI uses an **overlay merge** strategy:

### Merge Rules

1. **Existing values are preserved** - Your config values are never overwritten
2. **Template fills gaps** - Only missing fields are added from template
3. **Protected fields never copy** - Certain fields are always skipped

### Protected Fields

These fields are **never** copied from templates:

| Field | Reason |
|-------|--------|
| `project_id` | Machine-generated, unique per project |
| `network_volume_id` | User-specific resource |
| `profile` | User-specific setting |
| `$schema` | Schema reference |
| `template` | Template metadata itself |

### Example Merge

**Template (`gpu.jsonc`):**
```jsonc
{
  "gpu_type": "RTX 4090",
  "min_vram": 24,
  "outputs": ["outputs/"]
}
```

**Existing config:**
```jsonc
{
  "project_id": "my-project",
  "gpu_type": "RTX 3090"  // Already set
}
```

**Result:**
```jsonc
{
  "project_id": "my-project",  // Preserved
  "gpu_type": "RTX 3090",       // Preserved (existing wins)
  "min_vram": 24,               // Added from template
  "outputs": ["outputs/"]       // Added from template
}
```

---

## Creating Templates

### Step 1: Create Directory Structure

```bash
mkdir my-template
cd my-template
```

### Step 2: Create Configuration

Create `gpu.jsonc`:

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "template": {
    "name": "My Custom Template",
    "description": "A template for...",
    "author": "Your Name"
  },
  "provider": "runpod",
  "gpu_type": "RTX 4090",
  "min_vram": 24,
  "outputs": ["outputs/", "models/"],
  "download": [
    { "strategy": "hf", "source": "stabilityai/stable-diffusion-xl-base-1.0" }
  ]
}
```

### Step 3: Add Dependencies

Create `pyproject.toml`:

```toml
[project]
name = "my-template"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "diffusers",
    "transformers",
    "accelerate",
]
```

### Step 4: Add Application Code

Create your main script (e.g., `generate.py`):

```python
import torch
from diffusers import DiffusionPipeline

def main():
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    ).to("cuda")

    image = pipe("A beautiful sunset").images[0]
    image.save("outputs/result.png")

if __name__ == "__main__":
    main()
```

### Step 5: Add Documentation

Create `README.md` explaining usage.

### Step 6: Test Locally

```bash
# From another directory
gpu use ../my-template --dry-run

# Apply and run
gpu use ../my-template
```

### Step 7: Share (Optional)

Push to GitHub and others can use:

```bash
gpu use yourusername/my-template
```

---

## Examples Reference

The CLI repository includes these example templates:

| Example | Description | GPU | Key Features |
|---------|-------------|-----|--------------|
| `01-hello-world` | Basic introductory example | RTX 4090 | Minimal setup |
| `02-pytorch-mnist` | ML training with PyTorch | - | Training workflow |
| `03-stable-diffusion` | Image generation | RTX 4090 | Diffusers, multiple scripts |
| `04-qwen-image` | Advanced image generation | RTX 5090 | 30GB VRAM, large workspace |
| `04-video-generation` | Video synthesis | - | ffmpeg, system packages |
| `05-flux` | FLUX model generation | RTX 6000 Ada | 40GB VRAM |
| `06-network-volume` | Network volume testing | H100 | Persistent storage |
| `07-downloads` | Download strategies | RTX 4090 | HuggingFace downloads |
| `08-agent-downloads` | Agent-based downloads | RTX 4090 | HF + Civitai downloads |
| `session-autostop-smoke` | Session management | Docker | Local testing, autostop |

### Example Configurations

**Simple (01-hello-world):**
```jsonc
{
  "project_id": "hello-world",
  "outputs": ["results/", "models/"],
  "provider": "runpod",
  "gpu_type": "RTX 4090",
  "encryption": false
}
```

**With Downloads (07-downloads):**
```jsonc
{
  "project_id": "downloads-test",
  "provider": "runpod",
  "outputs": ["results/"],
  "gpu_type": "NVIDIA GeForce RTX 4090",
  "download": [
    { "strategy": "hf", "source": "hf-internal-testing/tiny-random-gpt2" }
  ]
}
```

**With System Packages (04-video-generation):**
```jsonc
{
  "project_id": "video-generation",
  "provider": "runpod",
  "outputs": ["videos/"],
  "environment": {
    "system": {
      "apt": [{ "name": "ffmpeg" }]
    }
  }
}
```

**High VRAM Requirements (05-flux):**
```jsonc
{
  "project_id": "flux-dev",
  "provider": "runpod",
  "outputs": ["outputs/"],
  "gpu_type": "NVIDIA RTX 6000 Ada",
  "min_vram": 40,
  "workspace_size_gb": 60
}
```

---

## Caching

### Cache Location

Templates fetched from GitHub are cached at:
```
~/.gpu-cli/templates/github/{cache_key}/
```

Cache keys are SHA256 hashes of the template specifier.

### Cache Metadata

Each cached template includes `.cache-meta.json`:
```json
{
  "source_spec": "github:owner/repo@v1.0.0",
  "fetched_at": "2024-01-15T10:30:00Z",
  "etag": "abc123..."
}
```

### Managing Cache

```bash
# Clear all cached templates
gpu template clear-cache

# Force clear without confirmation
gpu template clear-cache --force
```

---

## Troubleshooting

### Template Not Found

```
Error: Template not found at ./my-template
```
- Verify the path exists
- Ensure `gpu.jsonc` or `gpu.json` is present

### GitHub Rate Limiting

```
Error: GitHub API rate limit exceeded
```
- Wait for rate limit reset
- Use a GitHub token for higher limits

### Missing Configuration

```
Error: Missing required field 'project_id'
```
- Ensure your `gpu.jsonc` includes all required fields

### Cache Issues

```
Error: Cached template appears corrupted
```
- Clear the cache: `gpu template clear-cache`
- Re-fetch the template

---

## Summary

The GPU CLI template system provides:

- **Multiple sources**: Local, GitHub, and official registry
- **Smart caching**: Efficient re-use of fetched templates
- **Session persistence**: Resume workflows with stored state
- **Interactive inputs**: 10 input types for user configuration
- **Safe merging**: Overlay strategy preserves existing config
- **Flexible configuration**: JSONC format with schema support

For more information, see the [GPU CLI documentation](https://gpu-cli.sh/docs).
