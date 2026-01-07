---
description: Quick-start common GPU workflows with minimal setup
---

# GPU Quick Start

Set up a common GPU workflow with a single command.

**Workflow type:** $ARGUMENTS

## Supported Workflows

If the user specifies one of these, generate the minimal configuration:

### `comfyui` or `comfyui-flux`
ComfyUI with FLUX models for image generation.

### `sdxl` or `stable-diffusion`
Stable Diffusion XL for image generation.

### `llm` or `llama`
Run Llama 3.1 as a private API server.

### `whisper` or `transcribe`
Whisper transcription pipeline.

### `train` or `training`
Basic training setup with checkpoint syncing.

### `vllm` or `api`
vLLM server for OpenAI-compatible API.

## Instructions

1. **Identify the workflow** from user input
2. **Generate minimal gpu.jsonc** - Just the essentials
3. **Provide the run command** - Copy-paste ready
4. **Estimate cost** - For a typical session

## Output Format

```
## [Workflow Name] Quick Start

### 1. Create Configuration

```jsonc
// gpu.jsonc
{
  // Minimal config
}
```

### 2. Run It

```bash
gpu run [command]
```

### Cost

- GPU: [type] @ $X.XX/hr
- Typical session: $X.XX

### Next Steps

- [Optional enhancement 1]
- [Optional enhancement 2]
```
