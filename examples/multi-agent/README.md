# Multi-Agent Orchestration with vLLM

Run AI agent teams powered by local LLM inference using [vLLM](https://docs.vllm.ai/) and [CrewAI](https://www.crewai.com/). This template provides an OpenAI-compatible API server that can power multi-agent workflows with fast, private inference.

## Prerequisites

- GPU CLI installed and authenticated (`gpu auth login`)
- RunPod account with API key configured
- Access to A100 80GB or H100 GPU (70B model requires ~40GB VRAM)

## Quick Start

```bash
# 1. Start vLLM server on remote GPU
gpu run --publish 8000:8000 python serve.py

# Wait for "Uvicorn running on http://0.0.0.0:8000" message

# 2. Run agent example (in another terminal)
python agent_example.py --topic "The future of AI agents"
```

The vLLM server exposes an OpenAI-compatible API at `http://localhost:8000/v1`.

## Network Volume Setup (Highly Recommended)

The Qwen 72B model is ~40GB. Without a Network Volume, you'll re-download on every session.

1. Go to [RunPod Storage](https://www.runpod.io/console/user/storage)
2. Click **+ New Network Volume**
3. Choose a datacenter with A100 availability (e.g., EU-RO-1)
4. Set size to **150GB**
5. Copy the **Volume ID**
6. Add to your `gpu.jsonc`:

```jsonc
{
  "network_volume_id": "YOUR_VOLUME_ID"
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Your Application                     │
│         (CrewAI, LangGraph, custom agents)              │
└────────────────────────┬────────────────────────────────┘
                         │ OpenAI API (http://localhost:8000/v1)
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    vLLM Server                          │
│              (Qwen 2.5 72B Instruct AWQ)               │
│                   Running on GPU Pod                    │
└─────────────────────────────────────────────────────────┘
```

## Using with Different Frameworks

### CrewAI (Included)

```python
from crewai import Agent, Crew, Task
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-72B-Instruct-AWQ",
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

agent = Agent(
    role="Researcher",
    goal="Research topics thoroughly",
    llm=llm
)
```

### LangGraph

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-72B-Instruct-AWQ",
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Use in your LangGraph workflow
```

### OpenAI Python SDK (Direct)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct-AWQ",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### cURL

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Server Options

```bash
gpu run --publish 8000:8000 python serve.py \
  --model "Qwen/Qwen2.5-72B-Instruct-AWQ" \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Qwen 72B AWQ | HuggingFace model ID |
| `--port` | 8000 | Server port |
| `--max-model-len` | 32768 | Max context length |
| `--gpu-memory-utilization` | 0.90 | GPU memory to use |

## Alternative Models

| Model | VRAM | Context | Use Case |
|-------|------|---------|----------|
| `Qwen/Qwen2.5-72B-Instruct-AWQ` | ~40GB | 32K | Best quality (default) |
| `Qwen/Qwen2.5-32B-Instruct-AWQ` | ~20GB | 32K | Good quality, faster |
| `meta-llama/Llama-3.3-70B-Instruct` | ~40GB | 128K | Long context |
| `deepseek-ai/DeepSeek-V3` | ~60GB | 64K | Advanced reasoning |

To use a different model, update `gpu.jsonc`:

```jsonc
{
  "download": [
    { "strategy": "hf", "source": "YOUR_MODEL_ID" }
  ]
}
```

## Example: Research Team

The included `agent_example.py` demonstrates a three-agent research team:

1. **Researcher** - Gathers information on the topic
2. **Writer** - Creates a polished report
3. **Critic** - Reviews for accuracy and completeness

```bash
python agent_example.py --topic "Emerging trends in AI safety research"
```

Output is saved to `outputs/report.md`.

## Running Agents Remotely

You can run agents on the same pod as vLLM:

```bash
# Start server in background
gpu run --publish 8000:8000 -d python serve.py

# Run agents on the same pod
gpu run python agent_example.py --topic "Your topic"
```

Or run agents locally while server runs on pod:

```bash
# Terminal 1: Server on GPU pod
gpu run --publish 8000:8000 python serve.py

# Terminal 2: Agents locally (uses port-forwarded API)
pip install crewai langchain-openai
python agent_example.py --topic "Your topic"
```

## Performance

On A100 80GB with Qwen 72B AWQ:
- **First token latency**: ~200ms
- **Generation speed**: ~40 tokens/second
- **Concurrent requests**: Handles multiple agents efficiently

## Troubleshooting

### "Connection refused" error

The vLLM server takes 1-2 minutes to load the model. Wait for:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### "CUDA out of memory"

- Use a smaller model: `Qwen/Qwen2.5-32B-Instruct-AWQ`
- Reduce `--gpu-memory-utilization 0.85`
- Reduce `--max-model-len 16384`

### Slow model loading

- Use Network Volume to persist models
- First load: ~5 min (downloading 40GB)
- With Network Volume: ~1 min (loading from disk)

### Agents timeout

Increase timeout in your agent configuration:

```python
llm = ChatOpenAI(
    ...,
    request_timeout=120  # seconds
)
```

## Configuration

The `gpu.jsonc` file:

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "multi-agent",
  "provider": "runpod",
  "outputs": ["outputs/", "logs/"],
  "gpu_type": "NVIDIA A100 80GB PCIe",
  "min_vram": 40,

  "download": [
    { "strategy": "hf", "source": "Qwen/Qwen2.5-72B-Instruct-AWQ" }
  ]
}
```

## Files

```
multi-agent/
├── gpu.jsonc          # GPU CLI configuration
├── serve.py           # vLLM server script
├── agent_example.py   # CrewAI example
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── outputs/           # Generated reports (gitignored)
└── logs/              # Agent logs (gitignored)
```

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Qwen 2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ)
