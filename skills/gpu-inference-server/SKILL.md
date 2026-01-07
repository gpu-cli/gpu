---
name: gpu-inference-server
description: "Set up AI inference servers on cloud GPUs. Create private LLM APIs (vLLM, TGI), image generation endpoints, embedding services, and more. All with OpenAI-compatible interfaces that work with existing tools."
---

# GPU Inference Server

**Run your own AI APIs on cloud GPUs.**

This skill specializes in setting up inference servers: private LLM APIs, image generation endpoints, embedding services, and more. All accessible from your Mac via port forwarding.

## When to Use This Skill

| Request Pattern | This Skill Handles |
|-----------------|-------------------|
| "Run a private ChatGPT" | vLLM/TGI server with Llama/Mistral |
| "Set up an LLM API" | OpenAI-compatible endpoints |
| "Host my own image generation" | ComfyUI API, SD WebUI API |
| "Create an embedding service" | Sentence transformers API |
| "I need a coding assistant API" | Code LLM server |

## Server Architecture Options

```
┌─────────────────────────────────────────────────────┐
│  Your Mac                                            │
│  ┌─────────────────────────────────────────────────┐│
│  │  Applications using localhost:8000              ││
│  │  - Python scripts                               ││
│  │  - Web apps                                     ││
│  │  - CLI tools                                    ││
│  │  - Any OpenAI SDK                               ││
│  └──────────────────────┬──────────────────────────┘│
└─────────────────────────┼───────────────────────────┘
                          │ Port forward (SSH tunnel)
                          ▼
┌─────────────────────────────────────────────────────┐
│  Cloud GPU Pod                                       │
│  ┌─────────────────────────────────────────────────┐│
│  │  Inference Server (vLLM/TGI/ComfyUI)            ││
│  │  - Model loaded in VRAM                          ││
│  │  - OpenAI-compatible API                         ││
│  │  - Auto-scales with requests                     ││
│  └─────────────────────────────────────────────────┘│
│  GPU: RTX 4090 / A100 / H100                        │
└─────────────────────────────────────────────────────┘
```

## LLM Inference Servers

### vLLM (Recommended for Most Cases)

**Best for**: High throughput, production use, OpenAI compatibility

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "vllm-server",
  "gpu_type": "RTX 4090",
  "min_vram": 24,
  "persistent_proxy": true,
  "cooldown_minutes": 15,
  "health_check_paths": ["/health", "/v1/models"],
  "download": [
    { "strategy": "hf", "source": "meta-llama/Llama-3.1-8B-Instruct", "timeout": 7200 }
  ],
  "environment": {
    "python": {
      "pip_global": [
        { "name": "vllm", "version": ">=0.6.0" }
      ]
    }
  }
}
```

**Start command:**
```bash
gpu run -p 8000:8000 vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192
```

**For larger models (70B):**
```jsonc
{
  "gpu_type": "A100 SXM 80GB",
  "gpu_count": 2,
  "min_vram": 80
}
```

```bash
gpu run -p 8000:8000 vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 8192
```

### Text Generation Inference (TGI)

**Best for**: HuggingFace ecosystem, easy setup

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "tgi-server",
  "gpu_type": "RTX 4090",
  "min_vram": 24,
  "persistent_proxy": true,
  "cooldown_minutes": 15,
  "docker_image": "ghcr.io/huggingface/text-generation-inference:latest"
}
```

**Start command:**
```bash
gpu run -p 8080:8080 \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --max-input-length 4096 \
    --max-total-tokens 8192
```

### Ollama

**Best for**: Simple setup, many models, chat interface

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "ollama-server",
  "gpu_type": "RTX 4090",
  "min_vram": 24,
  "persistent_proxy": true,
  "cooldown_minutes": 10,
  "environment": {
    "shell": {
      "steps": [
        { "run": "curl -fsSL https://ollama.ai/install.sh | sh", "only_once": true }
      ]
    }
  }
}
```

**Start command:**
```bash
gpu run -p 11434:11434 ollama serve &
sleep 5
ollama pull llama3.1:8b
```

## Model Selection Guide

### Chat/Instruction Models

| Model | Size | VRAM | GPU Recommendation | Quality |
|-------|------|------|-------------------|---------|
| Llama 3.1 8B | 16GB | 20GB | RTX 4090 | Good |
| Llama 3.1 70B | 140GB | 150GB | 2x A100 80GB | Excellent |
| Llama 3.1 405B | 800GB | 800GB | 8x H100 | Best |
| Mistral 7B | 15GB | 18GB | RTX 4090 | Good |
| Mixtral 8x7B | 95GB | 100GB | 2x A100 80GB | Very Good |
| Qwen2.5 72B | 145GB | 160GB | 2x A100 80GB | Excellent |
| DeepSeek V3 | 685GB | 700GB | 8x H100 | Best |

### Code Models

| Model | Size | VRAM | Best For |
|-------|------|------|----------|
| DeepSeek Coder 33B | 66GB | 80GB | General coding |
| CodeLlama 34B | 68GB | 80GB | Python, general |
| Qwen2.5 Coder 32B | 64GB | 80GB | Multi-language |

### Specialized Models

| Model | Size | VRAM | Use Case |
|-------|------|------|----------|
| Llama 3.2 Vision 11B | 22GB | 24GB | Image understanding |
| Pixtral 12B | 24GB | 30GB | Vision + text |

## Image Generation APIs

### ComfyUI API

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "comfyui-api",
  "gpu_type": "RTX 4090",
  "min_vram": 24,
  "persistent_proxy": true,
  "cooldown_minutes": 10,
  "download": [
    { "strategy": "hf", "source": "black-forest-labs/FLUX.1-schnell", "allow": "*.safetensors", "timeout": 7200 },
    { "strategy": "hf", "source": "comfyanonymous/flux_text_encoders/t5xxl_fp16.safetensors", "timeout": 3600 },
    { "strategy": "hf", "source": "comfyanonymous/flux_text_encoders/clip_l.safetensors" }
  ],
  "environment": {
    "shell": {
      "steps": [
        { "run": "git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI", "only_once": true },
        { "run": "cd /workspace/ComfyUI && pip install -r requirements.txt", "only_once": true }
      ]
    }
  }
}
```

**Start command:**
```bash
gpu run -p 8188:8188 python /workspace/ComfyUI/main.py --listen 0.0.0.0 --port 8188
```

**Python client:**
```python
import requests
import json

def generate_image(prompt: str, workflow: dict) -> bytes:
    """Generate image using ComfyUI API."""
    # Queue prompt
    response = requests.post(
        "http://localhost:8188/prompt",
        json={"prompt": workflow}
    )
    prompt_id = response.json()["prompt_id"]

    # Poll for completion
    while True:
        history = requests.get(f"http://localhost:8188/history/{prompt_id}").json()
        if prompt_id in history:
            break

    # Get output image
    outputs = history[prompt_id]["outputs"]
    # ... extract image from outputs

# Example workflow (simplified)
workflow = {
    "3": {"class_type": "KSampler", ...},
    "4": {"class_type": "CheckpointLoaderSimple", ...},
    ...
}

image = generate_image("a cat astronaut", workflow)
```

### Automatic1111 API

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "sd-webui-api",
  "gpu_type": "RTX 4090",
  "min_vram": 12,
  "persistent_proxy": true,
  "cooldown_minutes": 10,
  "download": [
    { "strategy": "hf", "source": "stabilityai/stable-diffusion-xl-base-1.0", "allow": "*.safetensors", "timeout": 3600 }
  ]
}
```

**Start command:**
```bash
gpu run -p 7860:7860 python launch.py --listen --api --xformers
```

**Python client:**
```python
import requests
import base64

def txt2img(prompt: str, negative_prompt: str = "") -> bytes:
    response = requests.post(
        "http://localhost:7860/sdapi/v1/txt2img",
        json={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": 30,
            "cfg_scale": 7,
            "width": 1024,
            "height": 1024,
        }
    )
    return base64.b64decode(response.json()["images"][0])

image_bytes = txt2img("a beautiful sunset over mountains")
with open("output.png", "wb") as f:
    f.write(image_bytes)
```

## Embedding Services

### Sentence Transformers API

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "embedding-server",
  "gpu_type": "RTX 4090",
  "min_vram": 8,
  "persistent_proxy": true,
  "cooldown_minutes": 5,
  "download": [
    { "strategy": "hf", "source": "BAAI/bge-large-en-v1.5" }
  ],
  "environment": {
    "python": {
      "pip_global": [
        { "name": "sentence-transformers" },
        { "name": "fastapi" },
        { "name": "uvicorn" }
      ]
    }
  }
}
```

**Server script:**
```python
#!/usr/bin/env python3
"""OpenAI-compatible embedding API."""

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

class EmbeddingRequest(BaseModel):
    input: list[str] | str
    model: str = "bge-large-en-v1.5"

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[dict]
    model: str
    usage: dict

@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    texts = [request.input] if isinstance(request.input, str) else request.input
    embeddings = model.encode(texts).tolist()

    return EmbeddingResponse(
        data=[
            {"object": "embedding", "embedding": emb, "index": i}
            for i, emb in enumerate(embeddings)
        ],
        model=request.model,
        usage={"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": sum(len(t.split()) for t in texts)}
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Start:**
```bash
gpu run -p 8000:8000 python embed_server.py
```

**Usage (OpenAI SDK compatible):**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.embeddings.create(
    model="bge-large-en-v1.5",
    input=["Hello world", "How are you?"]
)
print(response.data[0].embedding[:5])  # First 5 dimensions
```

## Audio Processing APIs

### Whisper API

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "whisper-api",
  "gpu_type": "RTX 4090",
  "min_vram": 10,
  "persistent_proxy": true,
  "cooldown_minutes": 5,
  "download": [
    { "strategy": "hf", "source": "openai/whisper-large-v3" }
  ],
  "environment": {
    "python": {
      "pip_global": [
        { "name": "openai-whisper" },
        { "name": "fastapi" },
        { "name": "uvicorn" },
        { "name": "python-multipart" }
      ]
    },
    "system": {
      "apt": [{ "name": "ffmpeg" }]
    }
  }
}
```

**Server script:**
```python
#!/usr/bin/env python3
"""OpenAI-compatible Whisper transcription API."""

from fastapi import FastAPI, UploadFile, File
import whisper
import tempfile
import uvicorn

app = FastAPI()
model = whisper.load_model("large-v3")

@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...)):
    # Save uploaded file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Transcribe
    result = model.transcribe(tmp_path)

    return {"text": result["text"]}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Usage:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f
    )
print(transcript.text)
```

## Configuration Patterns

### Persistent Services (Stay Running)

For APIs that should remain available:

```jsonc
{
  "persistent_proxy": true,      // Keep SSH tunnel alive
  "cooldown_minutes": 30,        // Longer idle timeout
  "health_check_paths": ["/health", "/v1/models"]  // Don't count health checks as activity
}
```

### Auto-Scaling with Queued Requests

For services that can auto-resume:

```jsonc
{
  "persistent_proxy": true,
  "cooldown_minutes": 5,
  "resume_timeout_secs": 180,    // Wait up to 3 min for pod to resume
  "max_queued_requests": 100     // Queue requests while resuming
}
```

### High-Throughput Configuration

For production workloads:

```jsonc
{
  "gpu_count": 2,
  "min_vram": 80,
  "cooldown_minutes": 60,        // Stay warm longer
  "cloud_type": "secure"         // Use secure cloud instances
}
```

## Client Code Templates

### Universal OpenAI Client

```python
from openai import OpenAI

# Point to your GPU CLI server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # No auth required for local access
)

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)
print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
):
    print(chunk.choices[0].delta.content or "", end="")
```

### LangChain Integration

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="meta-llama/Llama-3.1-8B-Instruct"
)

response = llm.invoke("Explain quantum computing")
print(response.content)
```

### Continue.dev (VS Code) Configuration

`.continue/config.json`:
```json
{
  "models": [
    {
      "title": "Llama 3.1 (GPU CLI)",
      "provider": "openai",
      "model": "meta-llama/Llama-3.1-8B-Instruct",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "not-needed"
    }
  ]
}
```

## Cost Comparison

### LLM Inference Costs (per 1M tokens)

| Model | GPU | Cost/hr | Tokens/sec | Cost per 1M tokens |
|-------|-----|---------|------------|-------------------|
| Llama 3.1 8B | RTX 4090 | $0.44 | ~100 | ~$1.22 |
| Llama 3.1 70B | 2x A100 | $3.58 | ~30 | ~$33.15 |
| GPT-4o (OpenAI) | - | - | - | ~$5.00 |
| Claude 3 Sonnet | - | - | - | ~$3.00 |

**Your private API can be cheaper than commercial APIs** for high-volume use, plus you get:
- Complete data privacy
- No rate limits
- Full control

## Output Format

When setting up an inference server:

```markdown
## Private [Model] API Server

I've created an OpenAI-compatible API server for [model].

### Configuration

- **Model**: [model name]
- **GPU**: [type] @ $X.XX/hr
- **Endpoint**: http://localhost:8000/v1

### Start Server

```bash
gpu run -p 8000:8000 [command]
```

### Test It

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "[model]", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Use with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="[model]",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Costs

- Idle: Pod auto-stops after [X] minutes
- Active: ~$X.XX/hr
- Per 1K tokens: ~$X.XXX
```
