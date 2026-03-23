# Unsloth Studio - GPU CLI Template

Run Unsloth Studio on a remote GPU — a no-code web UI for fine-tuning, chatting with, and exporting LLMs. Studio provides 2x training speed and 70% less VRAM usage through the same Unsloth core, but with a browser-based interface instead of Python scripts.

## Quick Start

```bash
gpu use unsloth-studio
```

Once the pod is ready, open the Studio UI at the URL shown in the terminal (port 8000 remapped to localhost). From there you can:

1. Load a model from HuggingFace (e.g., Qwen 2.5 7B, Llama 3, DeepSeek)
2. Chat with it to test baseline behavior
3. Fine-tune it with your own data using visual Data Recipes
4. Export the trained model as GGUF, Safetensors, or push to HuggingFace Hub

## Endpoints

| Port | Service | Description |
|------|---------|-------------|
| 8000 | Studio UI | Main web interface for fine-tuning and inference |
| 8001 | Inference API | OpenAI-compatible `/v1` endpoint for programmatic access |

## Key Features

- **500+ supported models** — Qwen, DeepSeek, Gemma, Llama, and more
- **Data Recipes** — visual node-based pipeline to transform raw files (PDF, DOCX, CSV, JSONL) into training datasets
- **Side-by-side comparison** — compare base vs fine-tuned model outputs
- **One-click export** — GGUF (for Ollama/llama.cpp), Safetensors, LoRA adapters
- **OpenAI-compatible API** — use the fine-tuned model from any OpenAI SDK client
- **Works offline** — no external API calls required after model download

## Secrets

### HuggingFace Token (optional)

Needed for gated models (e.g., Llama 3) and pushing trained models to HuggingFace Hub.

```bash
gpu secret set HF_TOKEN <your-token>
```

## Output Layout

These paths sync back automatically:

- `exports/` — exported model weights (GGUF, Safetensors)
- `trained_models/` — fine-tuned LoRA adapters
- `data_recipes/` — saved Data Recipe configurations

Model caches (`.hf_cache/`) stay remote-only to avoid syncing large files.

## Recommended VRAM

| Model class | Mode | Suggested VRAM |
|-------------|------|----------------|
| 1B-4B | QLoRA | 16GB-24GB |
| 7B-8B | QLoRA | 24GB |
| 13B-14B | QLoRA | 40GB-48GB |
| 7B-8B | LoRA / 16-bit | 40GB+ |
| 70B+ | QLoRA | 80GB |

## Troubleshooting

### Studio UI is slow to load

First launch downloads and compiles dependencies (llama.cpp with CUDA, React frontend). This can take 2-5 minutes. Subsequent launches reuse the workspace volume and start much faster.

### CUDA out of memory during fine-tuning

- Use a smaller model or enable 4-bit quantization (QLoRA)
- Reduce batch size in the training settings
- Move to a larger GPU tier

### Gated model download fails

- Accept the model license on HuggingFace first
- Set `HF_TOKEN` with `gpu secret set`

### Inference API not responding

The inference API on port 8001 starts when you load a model in the Studio UI. It's not available until a model is loaded.
