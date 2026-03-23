# Unsloth Training - GPU CLI Template

Fine-tune open LLMs on a remote GPU with Unsloth, TRL, and LoRA/QLoRA. This template is designed for single-GPU supervised fine-tuning and now ships with a more compelling text-to-SQL demo based on `b-mc2/sql-create-context` and `Qwen2.5-7B`.

The runtime intentionally installs Unsloth with `uv` and `--torch-backend=auto` so the pod keeps a GPU-enabled torch build instead of accidentally downgrading to a CPU-only install.

## Quick Start

```bash
cd templates/unsloth-train

# Optional: edit training.json to change the model and training knobs
# Optional: switch to a local dataset instead of the default Hugging Face SQL dataset

gpu use .
```

The default template exposes TensorBoard on `http://localhost:6006` and syncs the durable training artifacts back to your local machine.

Out of the box, it fine-tunes `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` on `b-mc2/sql-create-context`, which teaches the model to turn natural language questions plus schema context into SQL.

## What This Template Does

- Runs Unsloth Core directly on the remote GPU
- Supports SFT with LoRA or QLoRA
- Saves a resumable checkpoint layout under `checkpoints/`
- Syncs artifacts and logs back through GPU CLI outputs
- Optionally uses `HF_TOKEN` for gated models and `WANDB_API_KEY` for external experiment tracking

## File Layout

| File | Purpose |
|------|---------|
| `gpu.jsonc` | GPU CLI runtime config, outputs, secrets, base image |
| `startup.sh` | Bootstrap, TensorBoard launch, resume detection, training handoff |
| `train.py` | Unsloth SFT training entrypoint |
| `training.json` | Model, dataset, LoRA, training, and runtime knobs |
| `merge_and_export.py` | Optional helper for merged 16-bit export |
| `data/sample-alpaca.jsonl` | Tiny local fallback dataset for smoke testing |

## Dataset Formats

The template supports three initial dataset modes.

### SQL Create Context dataset

This is the default mode and is tuned for the public `b-mc2/sql-create-context` dataset.

```json
{
  "dataset": {
    "source": "huggingface",
    "name": "b-mc2/sql-create-context",
    "split": "train",
    "format": "sql_create_context",
    "question_field": "question",
    "context_field": "context",
    "answer_field": "answer"
  }
}
```

Each example is rendered into a prompt with:

- a natural language question
- `CREATE TABLE` schema context
- the target SQL query

That makes the first-run demo feel much more product-like than a generic instruction dataset.

### Alpaca-style JSONL or JSON

Example local mode:

```json
{
  "instruction": "Summarize the goal of LoRA.",
  "input": "",
  "output": "LoRA trains a small adapter instead of updating every model weight."
}
```

Relevant config:

```json
{
  "dataset": {
    "source": "local",
    "path": "data/train.jsonl",
    "format": "alpaca",
    "instruction_field": "instruction",
    "input_field": "input",
    "output_field": "output"
  }
}
```

### Raw text datasets

```json
{
  "dataset": {
    "source": "local",
    "path": "data/train.jsonl",
    "format": "text",
    "text_field": "text"
  }
}
```

## Configuration Guide

### `model`

| Field | Description |
|-------|-------------|
| `name` | Base model ID to fine-tune |
| `max_seq_length` | Training context length |
| `load_in_4bit` | Enable QLoRA-style loading |
| `dtype` | Optional explicit torch dtype |
| `hf_token_env` | Env var to read HuggingFace token from |

### `lora`

| Field | Description |
|-------|-------------|
| `rank` | LoRA rank |
| `alpha` | LoRA alpha |
| `dropout` | LoRA dropout |
| `target_modules` | Modules to adapt |
| `gradient_checkpointing` | `unsloth`, `true`, or `false` |

### `training`

| Field | Description |
|-------|-------------|
| `per_device_train_batch_size` | Per-device batch size |
| `gradient_accumulation_steps` | Effective batch scaling |
| `num_train_epochs` | Full epochs to train |
| `max_steps` | Optional step cap for quick runs |
| `save_steps` | Checkpoint frequency |
| `report_to` | `none` or `wandb` |

### `runtime`

| Field | Description |
|-------|-------------|
| `output_dir` | Active trainer output directory |
| `latest_checkpoint_dir` | Stable resume path |
| `final_checkpoint_dir` | Final promoted checkpoint |
| `logs_dir` | TensorBoard and training logs |
| `exports_dir` | Optional merged exports |
| `tensorboard.enabled` | Enable TensorBoard on port 6006 |

## Output Layout

These paths sync back automatically:

- `outputs/`
- `checkpoints/latest/`
- `checkpoints/final/`
- `logs/`
- `exports/`

These paths intentionally stay remote-only by default:

- `.hf_cache/`
- pip caches
- transient Python bytecode

## Resume Behavior

If `checkpoints/latest/` already contains a saved trainer state, `startup.sh` automatically resumes from it on the next launch.

That means you can:

1. stop a training run
2. keep the workspace volume attached
3. launch the template again
4. continue from the latest checkpoint

## Demo Story

The default pairing is intentionally opinionated:

- **Model:** `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- **Dataset:** `b-mc2/sql-create-context`

Why this pairing works well:

- the model is strong, ungated, and practical on a 24GB GPU
- the dataset is public, free, and CC-BY-4.0 licensed
- the before/after is easy to explain: train your own text-to-SQL copilot
- the schema-grounded examples make the output feel more specialized and useful

## Secrets

### HuggingFace Token

Optional for this default demo. Needed for gated models or private datasets.

```bash
gpu secret set HF_TOKEN <your-token>
```

### Weights & Biases

Optional. Set `training.report_to` to `wandb` and provide:

```bash
gpu secret set WANDB_API_KEY <your-token>
```

## Recommended VRAM Tiers

| Model class | Mode | Suggested VRAM |
|-------------|------|----------------|
| 1B-3B | QLoRA | 8GB-16GB |
| 7B-8B | QLoRA | 24GB |
| 13B-14B | QLoRA | 40GB-48GB |
| 7B-8B | LoRA / 16-bit | 40GB+ |

For v1, this template is tuned around 7B-8B QLoRA workloads, especially the default Qwen text-to-SQL demo.

## Exporting a Merged Model

After training finishes:

```bash
gpu run python merge_and_export.py --config training.json
```

This writes a merged 16-bit export to `exports/merged_16bit/`.

## Troubleshooting

### CUDA out of memory

- lower `max_seq_length`
- reduce `per_device_train_batch_size`
- keep `load_in_4bit` enabled
- move to a larger GPU tier

### Training starts from scratch instead of resuming

- check that `checkpoints/latest/` contains files after the previous run
- keep the same workspace volume attached across runs
- verify `training.json` still points to the same checkpoint layout

### Gated model download fails

- accept the model license on HuggingFace first
- set `HF_TOKEN` with `gpu secret set`
- verify the token has the correct scopes

### TensorBoard is blank

- wait for the first logging step
- confirm `runtime.tensorboard.enabled` is still `true`
- inspect `logs/train.log` for trainer startup failures

## Next Steps

1. Run the default SQL demo once to validate your GPU setup.
2. Tune `training.json` for your model size, batch size, and target task.
3. Switch to a local dataset if you want to fine-tune on your own schema/task data.
4. Export merged weights after training if you want downstream inference packaging.
