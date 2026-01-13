# CrewAI Stock Analysis

Multi-agent AI stock analysis powered by local LLMs on remote GPUs.

## Why GPU CLI?

- **$0 LLM costs**: Run 32B parameter models locally vs $0.03/1K tokens for GPT-4
- **Privacy**: All data stays on your GPU - no API calls to OpenAI/Anthropic
- **Fast**: GPU-accelerated inference is 5-10x faster than CPU
- **Zero setup**: No need to install Ollama, download models, or configure anything

## Quick Start

### Step 1: Setup (one-time)

Initialize the Ollama server and download the model:

```bash
gpu run python setup.py
```

This provisions a GPU pod, installs Ollama, and downloads the appropriate model based on available VRAM. Takes 5-10 minutes on first run.

### Step 2: Run Queries

**Interactive mode** - analyze multiple stocks in one session:

```bash
gpu run python main.py
```

You'll see an interactive prompt:
```
============================================================
CrewAI Stock Analysis - Interactive Mode
============================================================
Enter stock tickers to analyze (e.g., NVDA, AAPL, TSLA)
Type 'quit' or 'exit' to stop
============================================================

Ticker> NVDA
... analysis runs ...

Ticker> AAPL
... analysis runs ...

Ticker> quit
```

**Single query mode** - analyze one stock and exit:

```bash
gpu run python main.py NVDA
```

## What Happens

### During Setup (`setup.py`)

1. GPU CLI provisions a pod with an NVIDIA A40/A6000 (48GB) or RTX 4090 (24GB)
2. Ollama installs automatically during provisioning
3. The appropriate model downloads based on available VRAM:
   - 48GB+ VRAM: Qwen 2.5 32B (excellent reasoning)
   - 24GB VRAM: Qwen 2.5 14B (good balance of speed/quality)
4. Configuration is saved for subsequent runs

### During Analysis (`main.py`)

1. Connects to the running pod with Ollama already configured
2. Three AI agents collaborate to analyze the stock:
   - **Research Analyst**: Gathers news, market data, and company info
   - **Financial Analyst**: Analyzes valuation, growth trends, and financials
   - **Investment Advisor**: Synthesizes everything into a recommendation
3. The final report syncs back to your local `reports/` folder
4. In interactive mode, you can analyze multiple stocks without restarting

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
| `gpu.toml` | GPU CLI configuration (GPU types, outputs, environment) |
| `pyproject.toml` | Python dependencies |
| `setup.py` | One-time setup - initializes Ollama and downloads model |
| `main.py` | Interactive query interface for stock analysis |
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

### "Run 'gpu run python setup.py' first to initialize"

You need to run setup before running queries:
```bash
gpu run python setup.py
```

### "Ollama server failed to start"

The Ollama installation may have failed. Try running setup again:
```bash
gpu run python setup.py
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

## Next Steps

- Try different stocks and compare analyses
- Modify agents to focus on specific sectors (tech, healthcare, etc.)
- Add custom tools for specialized data sources
- Integrate with portfolio management workflows
