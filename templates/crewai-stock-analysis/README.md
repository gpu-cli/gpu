# CrewAI Stock Analysis

Multi-agent AI stock analysis powered by local LLMs on remote GPUs.

## Why GPU CLI?

- **$0 LLM costs**: Run 32B parameter models locally vs $0.03/1K tokens for GPT-4
- **Privacy**: All data stays on your GPU - no API calls to OpenAI/Anthropic
- **Fast**: GPU-accelerated inference is 5-10x faster than CPU
- **Zero setup**: No need to install Ollama, download models, or configure anything

## Quick Start

```bash
gpu use .
```

This single command:
1. Provisions a GPU pod with an NVIDIA RTX 4090/A40/A6000 (24-48GB)
2. Installs Ollama and downloads the appropriate model (first run only)
3. Launches the web API server for stock analysis

Once ready, analyze stocks via the API:

```bash
# Analyze a stock (streams progress in real-time)
curl http://localhost:8501/analyze/NVDA

# List completed reports
curl http://localhost:8501/reports

# View a specific report
curl http://localhost:8501/reports/NVDA
```

**Single query mode** - analyze one stock via CLI:

```bash
gpu run python main.py NVDA
```

## What Happens

### During Setup (automatic on first start)

1. GPU CLI provisions a pod with an NVIDIA RTX 4090/A40/A6000 (24-48GB)
2. Ollama installs automatically during provisioning
3. The appropriate model downloads based on available VRAM:
   - 48GB+ VRAM: Qwen 2.5 32B (excellent reasoning)
   - 24GB VRAM: Qwen 2.5 14B (good balance of speed/quality)
4. Configuration is saved for subsequent runs

### During Analysis

1. Three AI agents collaborate to analyze the stock:
   - **Research Analyst**: Gathers news, market data, and company info
   - **Financial Analyst**: Analyzes valuation, growth trends, and financials
   - **Investment Advisor**: Synthesizes everything into a recommendation
2. The final report syncs back to your local `reports/` folder

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (used by readiness hook) |
| `/analyze/<ticker>` | GET | Run analysis (streaming text response) |
| `/reports` | GET | List completed reports |
| `/reports/<ticker>` | GET | Get a specific report |

## Output

After running, you'll find a comprehensive investment report at:
```
reports/NVDA_analysis.md
```

The report includes:
- Recent news and developments
- Financial metrics and valuation analysis
- Peer comparison
- Clear BUY/HOLD/SELL recommendation with target price
- Key risks to monitor

## Files

| File | Description |
|------|-------------|
| `gpu.jsonc` | GPU CLI configuration (GPU types, ports, readiness hook, environment) |
| `startup.sh` | Startup script - starts Ollama, downloads model, launches web server |
| `web_server.py` | Flask REST API wrapping CrewAI analysis |
| `init_ollama.py` | Idempotent Ollama setup - verifies server and downloads model |
| `main.py` | CLI mode - analyze a single stock via `gpu run` |
| `crew.py` | CrewAI crew definition with agents and tasks |
| `ollama_utils.py` | Ollama server management utilities |
| `config/agents.yaml` | Agent role definitions |
| `config/tasks.yaml` | Task descriptions and workflow |
| `tools/search.py` | Web search and scraping tools |

## Customization

### Change the LLM Model

Edit `ollama_utils.py` to use different models:

```python
def select_model(vram_gb: int) -> str:
    if vram_gb >= 40:
        return "llama3.1:70b"  # Larger model
    return "llama3.1:8b"  # Smaller model
```

### Modify Agent Behavior

Edit `config/agents.yaml` to change agent roles, goals, or backstories.

### Customize Analysis Tasks

Edit `config/tasks.yaml` to modify what each agent focuses on.

### Add More Tools

Add new tools in `tools/search.py` - for example, you could add:
- RSS feed reader for financial news
- Stock price API integration
- Document analysis tools

## Performance

| GPU | Model | Analysis Time |
|-----|-------|---------------|
| A40 (48GB) | Qwen 2.5 32B | ~10-15 min |
| RTX 4090 (24GB) | Qwen 2.5 14B | ~5-10 min |

First run takes longer due to model download (~10-20GB).

## Troubleshooting

### "Model not configured" from health endpoint

The Ollama server is still initializing and downloading the model. Wait for the readiness hook to pass - you'll see "Service Ready" in the terminal.

### "Ollama server failed to start"

The Ollama installation may have failed. Stop the pod and start fresh:
```bash
gpu stop
gpu use .
```

### "Model download failed"

Large models can fail to download on slow connections. Try a smaller model by editing `ollama_utils.py`.

### "No search results"

DuckDuckGo may rate-limit requests. The analysis will continue with available data.

## Requirements

- GPU CLI installed and configured (`gpu auth login`)
- RunPod account with API key
- No additional API keys required!

## Cost Estimate

- Pod cost: ~$0.50-1.00 per analysis (15 min on A40)
- LLM inference: $0 (runs locally)
- Compare to: $5-10+ for equivalent GPT-4 API usage
