# Ollama Models - GPU CLI Template

Run Open Source LLMs on remote GPUs with Ollama. Includes a lightweight Web UI and OpenAI/Anthropic-compatible API endpoints.

## Quick Start

```bash
cd examples/ollama-models

# (Optional) Edit models.json to configure which models to pre-pull

# Start the pod
gpu run
```

After startup, two endpoints are available on your local machine:

| Endpoint | Port | Description |
|----------|------|-------------|
| **Web UI** | 8080 | Chat interface at http://localhost:8080 |
| **Ollama API** | 11434 | REST API at http://localhost:11434 |

## Features

- **Dual Endpoints**: Both Web UI and API forwarded to localhost
- **OpenAI-Compatible API**: Use existing OpenAI SDK code with local models
- **Anthropic Messages API**: Supported in Ollama 0.14.0+ (for tools like Claude Code)
- **Model Pre-Pull**: Configure models in `models.json` to pre-download on startup
- **Streaming**: Real-time response streaming in both UI and API

## Configuring Models

Edit `models.json` to specify which models to pre-pull when the pod starts:

```json
{
  "models": [
    "llama3.2:3b",
    "mistral:7b",
    "codellama:7b"
  ],
  "default": "llama3.2:3b"
}
```

### Popular Models

| Model | Size | VRAM | Best For |
|-------|------|------|----------|
| `llama3.2:3b` | 2GB | 4GB | Fast, general purpose |
| `llama3.2:7b` | 4GB | 8GB | Balanced quality/speed |
| `llama3.1:70b` | 40GB | 48GB | High quality, slower |
| `mistral:7b` | 4GB | 8GB | Fast, multilingual |
| `codellama:7b` | 4GB | 8GB | Code generation |
| `codellama:13b` | 8GB | 16GB | Better code quality |
| `deepseek-r1:7b` | 4GB | 8GB | Reasoning tasks |
| `phi3:mini` | 2GB | 4GB | Small, fast |

Browse all models: https://ollama.com/library

## API Usage

### Native Ollama API

```bash
# List available models
curl http://localhost:11434/api/tags

# Chat with a model
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2:3b",
  "messages": [{"role": "user", "content": "Hello!"}]
}'

# Pull a new model
curl http://localhost:11434/api/pull -d '{"name": "codellama:7b"}'
```

### OpenAI-Compatible API

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Using with OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Required but ignored
)

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Using with Anthropic SDK (Ollama 0.14.0+)

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:11434",
    api_key="ollama"  # Required but ignored
)

message = client.messages.create(
    model="llama3.2:3b",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(message.content)
```

## Using with Claude Code

This template can serve as a backend for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) via Ollama's Anthropic API compatibility.

### Configuration

1. Start the pod: `gpu run`
2. Configure Claude Code to use the Ollama endpoint:

```bash
# Set environment variables (add to ~/.zshrc or ~/.bashrc)
export ANTHROPIC_AUTH_TOKEN=ollama
export ANTHROPIC_BASE_URL=http://localhost:11434

# Run Claude Code with your model
claude --model glm-4.7-flash
```

### Recommended Models for Claude Code

| Model | Size | Context | VRAM | Best For |
|-------|------|---------|------|----------|
| `glm-4.7-flash` | 19GB | 198K | 24GB | Excellent quality, huge context (default) |
| `qwen2.5-coder:14b` | 9GB | 32K | 16GB | Great balance of speed/quality |
| `qwen2.5-coder:7b` | 4.7GB | 32K | 8GB | Fast, good for simple tasks |

**Note:** Claude Code benefits from models with large context windows. The `glm-4.7-flash` model offers 198K context and excellent quality.

## Pull Additional Models

While the pod is running, you can pull more models:

```bash
# Via API
curl http://localhost:11434/api/pull -d '{"name": "codellama:7b"}'

# Via the Web UI (click the + button in model selector)

# Via SSH into the pod
gpu exec ollama pull deepseek-r1:7b
```

## GPU Selection

The template is configured to use GPUs in this priority order:

1. **RTX 4090** (24GB) - Great for 7B-13B models
2. **A40** (48GB) - Good for 30B models
3. **L40S** (48GB) - Newer, good availability
4. **A100 80GB** - For 70B+ models

Edit `gpu.jsonc` to change GPU preferences:

```jsonc
"gpu_types": [
  { "type": "RTX 4090" },
  { "type": "A40" }
],
"min_vram": 16
```

## Configuration Reference

### gpu.jsonc

| Field | Description |
|-------|-------------|
| `ports` | Ports to forward (default: `[11434, 8080]`) |
| `keep_alive_minutes` | Idle time before pod stops (default: `5`) |
| `gpu_types` | Preferred GPUs in priority order |
| `min_vram` | Minimum VRAM in GB (default: `16`) |

### models.json

| Field | Description |
|-------|-------------|
| `models` | Array of model names to pre-pull |
| `default` | Default model selected in Web UI |

## Troubleshooting

### "No models available"

Pull a model first:
```bash
curl http://localhost:11434/api/pull -d '{"name": "llama3.2:3b"}'
```

### Slow model loading

Large models (30B+) can take 30-60 seconds to load. The cooldown is set to 5 minutes to account for this.

### API returns errors

Check Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

### Port already in use

If ports 11434 or 8080 are already in use locally, stop other services or modify `gpu.jsonc` to use different ports.

## Future Work

The following features are being developed by other engineers:

- **Activity Proxy**: HTTP/WebSocket monitoring for smarter cooldown management
- **Cooldown Hook**: Automatic cooldown extension during model loading

Current workarounds:
- Use longer `keep_alive_minutes` for large models
- Pre-pull models in `models.json` to reduce first-use latency
