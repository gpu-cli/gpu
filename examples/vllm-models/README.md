# vLLM Models - GPU CLI Template

Run Open Source LLMs on remote GPUs with vLLM's high-performance inference engine. Includes a lightweight Web UI and OpenAI-compatible API endpoint.

## Quick Start

```bash
cd examples/vllm-models

# (Optional) Edit models.json to configure which model to run

# Start the pod
gpu run
```

After startup, two endpoints are available on your local machine:

| Endpoint | Port | Description |
|----------|------|-------------|
| **Web UI** | 8080 | Chat interface at http://localhost:8080 |
| **vLLM API** | 8000 | OpenAI-compatible API at http://localhost:8000 |

## Features

- **High Performance**: vLLM provides 2-4x higher throughput than vanilla HuggingFace
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI SDK
- **Streaming**: Real-time response streaming in both UI and API
- **KV Cache Optimization**: Efficient memory management for long contexts
- **Continuous Batching**: Handles multiple concurrent requests efficiently

## vLLM vs Ollama

| Aspect | vLLM | Ollama |
|--------|------|--------|
| **Performance** | Higher throughput (PagedAttention) | Good, optimized for single-user |
| **Model Loading** | Single model at startup | Dynamic pull/switch |
| **Memory** | More VRAM (KV cache) | Less VRAM overhead |
| **API** | OpenAI-compatible only | Native + OpenAI + Anthropic |
| **Best For** | Production workloads, high throughput | Development, model experimentation |

## Configuring Models

Edit `models.json` to specify which model to run:

```json
{
  "model": "zai-org/GLM-4.7-Flash",
  "vllm_args": {
    "gpu_memory_utilization": 0.9,
    "max_model_len": 32768,
    "tensor_parallel_size": 1
  }
}
```

### Configuration Options

| Field | Description | Default |
|-------|-------------|---------|
| `model` | HuggingFace model ID | Required |
| `gpu_memory_utilization` | Fraction of GPU memory to use | 0.9 |
| `max_model_len` | Maximum context length | 32768 |
| `tensor_parallel_size` | Number of GPUs for tensor parallelism | 1 |

### Recommended Models

| Model | Size | VRAM | Auth | Best For |
|-------|------|------|------|----------|
| `zai-org/GLM-4.7-Flash` | 19GB | 24GB | No | **Default** - coding, 200K context |
| `Qwen/Qwen2.5-1.5B-Instruct` | 3GB | 8GB | No | Fast, small GPU |
| `Qwen/Qwen2.5-7B-Instruct` | 14GB | 24GB | No | Good quality |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 14GB | 24GB | No | Code-focused |
| `mistralai/Mistral-7B-Instruct-v0.3` | 14GB | 24GB | No | High quality |
| `meta-llama/Llama-3.2-3B-Instruct` | 6GB | 12GB | Yes | Popular (requires HF token) |
| `meta-llama/Llama-3.1-8B-Instruct` | 16GB | 24GB | Yes | High quality (requires HF token) |

### Gated Models (Llama, etc.)

Some models require a HuggingFace token:

1. Create account at https://huggingface.co
2. Accept model license (e.g., https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
3. Create token at https://huggingface.co/settings/tokens
4. Set token in GPU CLI:

```bash
gpu secret set HF_TOKEN <your-token>
```

## API Usage

### List Models

```bash
curl http://localhost:8000/v1/models
```

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-org/GLM-4.7-Flash",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-org/GLM-4.7-Flash",
    "messages": [{"role": "user", "content": "Write a haiku about GPUs"}],
    "stream": true
  }'
```

### Using with OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="vllm"  # Required but ignored
)

response = client.chat.completions.create(
    model="zai-org/GLM-4.7-Flash",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Using with LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="vllm",
    model="zai-org/GLM-4.7-Flash"
)

response = llm.invoke("Hello!")
print(response.content)
```

## Changing Models

Unlike Ollama, vLLM loads a single model at startup. To change models:

1. Edit `models.json` with the new model ID
2. Restart the pod:

```bash
gpu restart
```

## GPU Selection

The template is configured to use GPUs in this priority order:

1. **RTX 4090** (24GB) - Great for 7B-14B models
2. **A40** (48GB) - Good for 30B models
3. **L40S** (48GB) - Newer, good availability
4. **A100 80GB** - For 70B+ models

Edit `gpu.jsonc` to change GPU preferences:

```jsonc
"gpu_types": [
  { "type": "A100 PCIe 80GB" },
  { "type": "A40" }
],
"min_vram": 48
```

## Configuration Reference

### gpu.jsonc

| Field | Description |
|-------|-------------|
| `ports` | Ports to forward (default: `[8000, 8080]`) |
| `keep_alive_minutes` | Idle time before pod stops (default: `20`) |
| `gpu_types` | Preferred GPUs in priority order |
| `min_vram` | Minimum VRAM in GB (default: `24`) |
| `inputs` | Optional secrets like `hf_token` |

### models.json

| Field | Description |
|-------|-------------|
| `model` | HuggingFace model ID to load |
| `vllm_args` | vLLM server configuration |

## Troubleshooting

### Model fails to load

Check if the model fits in GPU memory:
- 7B models: ~14GB VRAM
- 13B models: ~26GB VRAM
- 70B models: ~140GB VRAM (needs multi-GPU)

Try reducing `max_model_len` or `gpu_memory_utilization` in `models.json`.

### "CUDA out of memory"

Lower the GPU memory utilization:

```json
{
  "vllm_args": {
    "gpu_memory_utilization": 0.8
  }
}
```

### Slow startup

Large models can take 30-120 seconds to load. This is normal - vLLM loads the full model into GPU memory at startup.

### Gated model access denied

1. Verify you've accepted the model license on HuggingFace
2. Check your token is set: `gpu secret list`
3. Ensure token has read access

### Port already in use

If port 8000 is already in use locally, modify `gpu.jsonc`:

```jsonc
"ports": [8001, 8080]
```

Then update `ui/app.js` to use the new port.

## Multi-GPU Support

For large models that don't fit on a single GPU, increase tensor parallelism:

```json
{
  "model": "meta-llama/Llama-3.1-70B-Instruct",
  "vllm_args": {
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.9
  }
}
```

And update `gpu.jsonc` to request multiple GPUs:

```jsonc
"gpu_types": [
  { "type": "A100 PCIe 80GB", "count": 2 }
]
```

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [GLM-4.7-Flash Model](https://huggingface.co/zai-org/GLM-4.7-Flash)
