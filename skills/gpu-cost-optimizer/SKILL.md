---
name: gpu-cost-optimizer
description: "Analyze GPU workloads and recommend optimal configurations. Get cost estimates, compare GPU options, avoid over-provisioning, and find the cheapest GPU that meets your requirements."
---

# GPU Cost Optimizer

**Spend less, compute more.**

This skill analyzes workloads and recommends the most cost-effective GPU configuration. Stop guessing and overpaying.

## When to Use This Skill

| Request Pattern | This Skill Handles |
|-----------------|-------------------|
| "Which GPU should I use?" | GPU recommendation |
| "How much will this cost?" | Cost estimation |
| "Optimize my gpu.jsonc" | Configuration review |
| "Compare GPUs for my task" | GPU comparison |
| "I'm getting OOM errors" | Right-sizing help |
| "Reduce my GPU costs" | Cost optimization |

## GPU Database (January 2025 Prices)

### Consumer/Prosumer GPUs

| GPU | VRAM | Price/hr | Best For |
|-----|------|----------|----------|
| RTX 3090 | 24GB | $0.22 | Budget inference, small training |
| RTX 4090 | 24GB | $0.44 | Best value for 24GB tasks |
| RTX 4080 | 16GB | $0.35 | Light inference |
| RTX 4070 Ti | 12GB | $0.25 | Small models |

### Professional GPUs

| GPU | VRAM | Price/hr | Best For |
|-----|------|----------|----------|
| RTX A4000 | 16GB | $0.30 | Development |
| RTX A5000 | 24GB | $0.40 | Training |
| RTX A6000 | 48GB | $0.80 | Large models |
| L4 | 24GB | $0.35 | Inference |
| L40 | 48GB | $0.90 | Training/inference |
| L40S | 48GB | $0.95 | Fast training |

### Datacenter GPUs

| GPU | VRAM | Price/hr | Best For |
|-----|------|----------|----------|
| A100 40GB | 40GB | $1.29 | Training |
| A100 80GB PCIe | 80GB | $1.79 | Large models |
| A100 80GB SXM | 80GB | $2.09 | Multi-GPU training |
| H100 PCIe | 80GB | $2.49 | Fast training |
| H100 SXM | 80GB | $3.19 | Fastest training |
| H100 NVL | 94GB | $3.99 | Very large models |
| H200 | 141GB | $4.99 | Largest models |

## VRAM Requirements Guide

### LLM Inference (Chat/Completion)

| Model Size | FP16 VRAM | INT8 VRAM | INT4 VRAM |
|------------|-----------|-----------|-----------|
| 7B | 14GB | 8GB | 5GB |
| 13B | 26GB | 14GB | 8GB |
| 34B | 68GB | 36GB | 20GB |
| 70B | 140GB | 75GB | 40GB |
| 405B | 810GB | 420GB | 220GB |

**Formula**: VRAM (GB) = Parameters (B) × Precision (bytes) × 1.2 (overhead)
- FP16: 2 bytes/param
- INT8: 1 byte/param
- INT4: 0.5 bytes/param

### LLM Training

| Method | 7B VRAM | 13B VRAM | 70B VRAM |
|--------|---------|----------|----------|
| Full Fine-tune | 56GB | 104GB | 560GB |
| LoRA | 24GB | 40GB | 160GB |
| QLoRA | 12GB | 20GB | 48GB |

**Formula**: Full FT = Model × 8 (weights + gradients + optimizer states)

### Image Models

| Model | Inference | Training |
|-------|-----------|----------|
| SD 1.5 | 6GB | 12GB |
| SDXL | 10GB | 24GB |
| FLUX.1-schnell | 20GB | 40GB |
| FLUX.1-dev | 24GB | 48GB |

### Video Generation

| Model | VRAM Required |
|-------|--------------|
| CogVideoX | 40GB |
| Mochi-1 | 40GB |
| Hunyuan Video | 80GB |

## Decision Trees

### LLM Inference Selection

```
What model size?
├── 7-8B (Llama 3.1 8B, Mistral 7B)
│   └── INT4 quantization?
│       ├── Yes → RTX 4070 Ti (12GB) @ $0.25/hr
│       └── No → RTX 4090 (24GB) @ $0.44/hr ✓ Best value
│
├── 13B
│   └── INT4 quantization?
│       ├── Yes → RTX 4090 (24GB) @ $0.44/hr
│       └── No → RTX A6000 (48GB) @ $0.80/hr
│
├── 34B (CodeLlama 34B, etc.)
│   └── RTX A6000 or L40 (48GB) @ $0.80-0.90/hr
│
├── 70B (Llama 3.1 70B)
│   └── Quantization?
│       ├── INT4 → A100 40GB @ $1.29/hr
│       └── FP16 → 2× A100 80GB @ $3.58/hr
│
└── 405B (Llama 3.1 405B)
    └── INT4 → 4× A100 80GB or 2× H200 @ $7-10/hr
```

### Training Selection

```
What are you training?
├── Image LoRA (SDXL)
│   └── RTX 4090 (24GB) @ $0.44/hr ✓
│
├── Image LoRA (FLUX)
│   └── A100 80GB @ $1.79/hr
│
├── LLM Fine-tune (7B)
│   └── Method?
│       ├── QLoRA → RTX 4090 @ $0.44/hr ✓
│       ├── LoRA → A100 40GB @ $1.29/hr
│       └── Full FT → A100 80GB @ $1.79/hr
│
├── LLM Fine-tune (70B)
│   └── Method?
│       ├── QLoRA → A100 80GB @ $1.79/hr
│       ├── LoRA → 2× A100 80GB @ $3.58/hr
│       └── Full FT → 8× A100 80GB @ $14.32/hr
│
└── Custom PyTorch
    └── Profile first, then select
```

## Cost Optimization Strategies

### 1. Use the Smallest GPU That Works

**Bad**: Requesting A100 80GB for a task that fits on RTX 4090
```jsonc
// Don't do this
{"gpu_type": "A100 PCIe 80GB"}  // $1.79/hr for SD 1.5? Wasteful!
```

**Good**: Match GPU to workload
```jsonc
// Do this
{"gpu_type": "RTX 4090", "min_vram": 12}  // $0.44/hr, plenty for SD 1.5
```

### 2. Use Quantization

**INT4 quantization** reduces VRAM by 4× with minimal quality loss for inference.

```python
# Instead of loading full precision
model = AutoModelForCausalLM.from_pretrained("model")

# Use 4-bit quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    "model",
    quantization_config=quantization_config
)
```

**Savings**: Llama 70B on 1× A100 ($1.79/hr) instead of 2× A100 ($3.58/hr)

### 3. Enable Memory Optimizations

```python
# Gradient checkpointing (training)
model.gradient_checkpointing_enable()

# CPU offloading (inference)
pipe.enable_model_cpu_offload()

# VAE tiling (image generation)
pipe.vae.enable_tiling()

# Attention slicing
pipe.enable_attention_slicing()
```

### 4. Batch Processing

Process multiple items per GPU session instead of one-at-a-time:

```python
# Bad: Start pod for each image
for image in images:
    gpu_run(f"process {image}")  # Pod startup overhead each time

# Good: Batch process
gpu_run("process_batch images/")  # One pod startup, many images
```

### 5. Use Cooldown Wisely

```jsonc
// For interactive work (many short commands)
{"cooldown_minutes": 10}  // Pod stays warm

// For batch jobs
{"cooldown_minutes": 5}   // Stop quickly when done

// For API servers
{"cooldown_minutes": 30}  // Keep serving longer
```

### 6. Profile Before Committing

Test with small batch first:
```bash
# Test with subset
gpu run python train.py --max-steps 100

# Check VRAM usage
nvidia-smi  # In the output

# Then decide GPU size
```

## Configuration Review Checklist

When reviewing a gpu.jsonc, check:

| Check | Question | Optimization |
|-------|----------|--------------|
| GPU Size | Is VRAM matched to model? | Downsize if <70% utilization |
| GPU Count | Do you need multi-GPU? | Single GPU if model fits |
| Cooldown | Matches usage pattern? | Reduce for batch, increase for interactive |
| Downloads | Are models pre-cached? | Use download spec to avoid re-download |
| Precision | Can you quantize? | INT8/INT4 for inference |

## Cost Estimation Formulas

### Total Session Cost

```
Cost = (GPU $/hr) × (Hours Active) + (Download Time × GPU $/hr)

Examples:
- RTX 4090, 2 hours: $0.44 × 2 = $0.88
- A100 80GB, 4 hours: $1.79 × 4 = $7.16
- 2× H100, 8 hours: $6.38 × 8 = $51.04
```

### Training Cost Estimate

```
Hours = (Dataset Size × Epochs × Time per Sample) / 3600

Examples:
- QLoRA 7B, 10K samples, 3 epochs: ~3 hours on RTX 4090 = $1.32
- Full FT 7B, 10K samples, 3 epochs: ~24 hours on A100 = $42.96
```

### Inference Cost per 1K Tokens

```
Cost per 1K tokens = (GPU $/hr) / (Tokens/hr / 1000)

Examples (vLLM):
- Llama 8B on RTX 4090: $0.44/hr ÷ 360K/hr = $0.0012/1K tokens
- Llama 70B on 2× A100: $3.58/hr ÷ 108K/hr = $0.033/1K tokens
```

## Common Mistakes

### Over-Provisioning

| Mistake | Cost | Fix | Savings |
|---------|------|-----|---------|
| A100 for SD 1.5 | $1.79/hr | RTX 4090 | 75% |
| H100 for 7B inference | $2.49/hr | RTX 4090 | 82% |
| 2× GPU for model that fits on 1 | 2× cost | Single GPU | 50% |

### Under-Provisioning

| Mistake | Symptom | Fix |
|---------|---------|-----|
| RTX 4090 for 70B | OOM error | Use A100 or quantize |
| 24GB for FLUX training | OOM during backprop | Use A100 80GB |
| Single GPU for 405B | Won't load | Use 4× A100 or 2× H200 |

## GPU Comparison Tool

### Compare Options for Your Task

**Task**: Run Llama 3.1 70B for inference

| Option | GPU Setup | $/hr | Tokens/sec | $/1M tokens |
|--------|-----------|------|------------|-------------|
| A | 2× A100 80GB (FP16) | $3.58 | 30 | $33.15 |
| B | 1× A100 80GB (INT4) | $1.79 | 25 | $19.89 |
| C | 1× H100 80GB (INT4) | $2.49 | 40 | $17.29 |

**Recommendation**: Option B for cost, Option C for speed

## Output Format

When analyzing GPU requirements:

```markdown
## GPU Analysis for [Task]

### Requirements

- **Model size**: [X GB]
- **VRAM needed**: [X GB] ([precision])
- **Task type**: [inference/training]

### Recommended Configuration

```jsonc
{
  "gpu_type": "[GPU]",
  "min_vram": [X],
  // ... rest of config
}
```

### Cost Estimate

| Scenario | Duration | Cost |
|----------|----------|------|
| [Scenario 1] | [X hours] | $[X] |
| [Scenario 2] | [X hours] | $[X] |

### Alternatives

| GPU | $/hr | Pro | Con |
|-----|------|-----|-----|
| [GPU 1] | $X | [pro] | [con] |
| [GPU 2] | $X | [pro] | [con] |

### Optimization Tips

1. [Tip 1]
2. [Tip 2]
3. [Tip 3]
```
