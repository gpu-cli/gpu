---
name: gpu-ml-trainer
description: "Specialized skill for ML training workflows on cloud GPUs. Fine-tune LLMs with LoRA/QLoRA, train image LoRAs, build classifiers, and run custom training jobs. Generates production-ready training pipelines with checkpointing, logging, and optimal GPU selection."
---

# GPU ML Trainer

**Train ML models on cloud GPUs from your Mac.**

This skill specializes in training workflows: fine-tuning LLMs, training LoRAs for image generation, building classifiers, and custom PyTorch/JAX training jobs.

## When to Use This Skill

Activate for training-related requests:

| Request Pattern | This Skill Handles |
|-----------------|-------------------|
| "Fine-tune Llama on my data" | LLM fine-tuning with LoRA/QLoRA |
| "Train a LoRA for SDXL" | Image model LoRA training |
| "Build an image classifier" | Vision model training |
| "Train a custom model" | PyTorch/JAX training pipelines |
| "Continue training from checkpoint" | Resume training workflows |

## Training Architecture Decision Tree

```
User wants to train something
          │
          ▼
    ┌─────────────────┐
    │ What type of    │
    │ model/task?     │
    └────────┬────────┘
             │
    ┌────────┼────────┬─────────────┐
    ▼        ▼        ▼             ▼
  LLM    Image Gen  Classifier   Custom
    │        │         │            │
    ▼        ▼         ▼            ▼
LoRA/QLoRA  Kohya    Fine-tune   PyTorch
Full FT   Diffusers  Vision      Training
```

## LLM Fine-Tuning

### Method Selection

| Method | VRAM Required | Use Case | Quality |
|--------|---------------|----------|---------|
| **QLoRA** | 16GB (7B), 48GB (70B) | Budget-friendly, good results | Good |
| **LoRA** | 24GB (7B), 80GB (70B) | Better quality than QLoRA | Better |
| **Full Fine-tune** | 80GB (7B), 640GB (70B) | Maximum quality, expensive | Best |

### QLoRA Configuration (Recommended for Most Users)

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "llm-qlora-finetune",
  "gpu_type": "RTX 4090",  // 24GB enough for 7B QLoRA
  "min_vram": 24,
  "outputs": ["output/", "logs/"],
  "cooldown_minutes": 15,
  "download": [
    { "strategy": "hf", "source": "meta-llama/Llama-3.1-8B-Instruct", "timeout": 7200 }
  ],
  "environment": {
    "python": {
      "requirements": "requirements.txt"
    }
  }
}
```

### QLoRA Training Script Template

```python
#!/usr/bin/env python3
"""QLoRA fine-tuning for LLMs using transformers + peft."""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configuration
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "output"
MAX_SEQ_LENGTH = 2048

# QLoRA config
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# Training config
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

def format_instruction(sample):
    """Format your data into instruction format."""
    return f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""

def main():
    # 4-bit quantization for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"Loading {MODEL_ID} with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load your dataset
    # Option 1: From HuggingFace
    # dataset = load_dataset("your-dataset")

    # Option 2: From local JSONL
    dataset = load_dataset("json", data_files="data/train.jsonl")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",  # or "wandb" if you have it configured
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        formatting_func=format_instruction,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
    )

    # Train!
    print("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(f"{OUTPUT_DIR}/final")
    print(f"Model saved to {OUTPUT_DIR}/final")

if __name__ == "__main__":
    main()
```

### Data Format

**Required format** - JSONL file with instruction/output pairs:

```jsonl
{"instruction": "What is the capital of France?", "output": "The capital of France is Paris."}
{"instruction": "Summarize this text: ...", "output": "The text discusses..."}
{"instruction": "Write Python code to sort a list", "output": "```python\nsorted_list = sorted(my_list)\n```"}
```

### GPU Selection for LLM Training

| Model Size | QLoRA GPU | LoRA GPU | Full FT GPU | Est. Time (1K samples) |
|------------|-----------|----------|-------------|------------------------|
| 7-8B | RTX 4090 (24GB) | A100 40GB | A100 80GB | 30-60 min |
| 13B | A100 40GB | A100 80GB | 2x A100 80GB | 1-2 hours |
| 34B | A100 80GB | 2x A100 80GB | 4x A100 80GB | 2-4 hours |
| 70B | 2x A100 80GB | 4x A100 80GB | 8x H100 | 4-8 hours |

## Image LoRA Training

### SDXL LoRA with Kohya

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "sdxl-lora-training",
  "gpu_type": "RTX 4090",
  "min_vram": 24,
  "outputs": ["output/", "logs/"],
  "cooldown_minutes": 15,
  "download": [
    { "strategy": "hf", "source": "stabilityai/stable-diffusion-xl-base-1.0", "allow": "*.safetensors", "timeout": 3600 }
  ],
  "environment": {
    "shell": {
      "steps": [
        { "run": "git clone https://github.com/kohya-ss/sd-scripts.git /workspace/sd-scripts", "only_once": true },
        { "run": "cd /workspace/sd-scripts && pip install -r requirements.txt", "only_once": true }
      ]
    }
  }
}
```

### FLUX LoRA with AI Toolkit

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "flux-lora-training",
  "gpu_type": "A100 PCIe 80GB",
  "min_vram": 48,
  "outputs": ["output/"],
  "cooldown_minutes": 15,
  "download": [
    { "strategy": "hf", "source": "black-forest-labs/FLUX.1-dev", "allow": "*.safetensors", "timeout": 7200 }
  ],
  "environment": {
    "shell": {
      "steps": [
        { "run": "git clone https://github.com/ostris/ai-toolkit.git /workspace/ai-toolkit", "only_once": true },
        { "run": "cd /workspace/ai-toolkit && pip install -r requirements.txt", "only_once": true }
      ]
    }
  }
}
```

### Image Dataset Preparation

**Folder structure:**
```
dataset/
├── 1_subject/           # Repeats: train this subject 1x per epoch
│   ├── image1.jpg
│   ├── image1.txt       # Caption: "a photo of sks person"
│   ├── image2.jpg
│   └── image2.txt
└── regularization/      # Optional: regularization images
    ├── reg1.jpg
    └── reg1.txt
```

**Caption file format** (image1.txt):
```
a photo of sks person, professional headshot, studio lighting
```

### Training Parameters Guide

| Parameter | Character/Subject | Style | Object |
|-----------|------------------|-------|--------|
| Learning Rate | 1e-4 | 5e-5 | 1e-4 |
| Epochs | 1000-2000 | 500-1000 | 500-1000 |
| Network Dim | 32-64 | 64-128 | 32-64 |
| Network Alpha | 16-32 | 32-64 | 16-32 |
| Min Images | 10-20 | 50-100 | 20-50 |

## Classifier Training

### Vision Classifier with Fine-tuned ViT

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "image-classifier",
  "gpu_type": "RTX 4090",
  "min_vram": 12,
  "outputs": ["checkpoints/", "results/"],
  "download": [
    { "strategy": "hf", "source": "google/vit-base-patch16-224" }
  ],
  "environment": {
    "python": {
      "requirements": "requirements.txt"
    }
  }
}
```

### Classifier Training Script

```python
#!/usr/bin/env python3
"""Fine-tune ViT for image classification."""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from tqdm import tqdm
import json
from pathlib import Path

# Configuration
MODEL_ID = "google/vit-base-patch16-224"
DATA_DIR = "data"
OUTPUT_DIR = Path("checkpoints")
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 5e-5

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load dataset (ImageFolder format)
    # data/train/class1/*.jpg, data/train/class2/*.jpg, etc.
    dataset = load_dataset("imagefolder", data_dir=DATA_DIR)

    # Get class labels
    labels = dataset["train"].features["label"].names
    num_labels = len(labels)
    print(f"Training classifier for {num_labels} classes: {labels}")

    # Save label mapping
    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump({i: label for i, label in enumerate(labels)}, f)

    # Load model
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    model = ViTForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    ).to("cuda")

    # Transforms
    def transform(examples):
        examples["pixel_values"] = [
            processor(image.convert("RGB"), return_tensors="pt")["pixel_values"][0]
            for image in examples["image"]
        ]
        return examples

    dataset = dataset.with_transform(transform)

    # DataLoaders
    train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE) if "validation" in dataset else None

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_accuracy = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            pixel_values = torch.stack(batch["pixel_values"]).to("cuda")
            labels = torch.tensor(batch["label"]).to("cuda")

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        # Validation
        if val_loader:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = torch.stack(batch["pixel_values"]).to("cuda")
                    labels = torch.tensor(batch["label"]).to("cuda")
                    outputs = model(pixel_values=pixel_values)
                    predictions = outputs.logits.argmax(-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            accuracy = correct / total
            print(f"Validation Accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model.save_pretrained(OUTPUT_DIR / "best_model")
                processor.save_pretrained(OUTPUT_DIR / "best_model")

    # Save final model
    model.save_pretrained(OUTPUT_DIR / "final_model")
    processor.save_pretrained(OUTPUT_DIR / "final_model")
    print(f"Training complete! Best accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
```

### Dataset Format for Classifiers

**ImageFolder format:**
```
data/
├── train/
│   ├── cat/
│   │   ├── cat1.jpg
│   │   └── cat2.jpg
│   └── dog/
│       ├── dog1.jpg
│       └── dog2.jpg
└── validation/
    ├── cat/
    └── dog/
```

## Advanced Training Features

### Checkpointing & Resume

```python
# In training script
training_args = TrainingArguments(
    save_steps=100,
    save_total_limit=3,
    resume_from_checkpoint=True,  # Auto-resume from last checkpoint
)

# To resume:
# gpu run python train.py --resume output/checkpoint-500
```

### Weights & Biases Integration

```python
import wandb

wandb.init(project="my-training-run")

training_args = TrainingArguments(
    report_to="wandb",
    run_name="experiment-1",
)
```

### Multi-GPU Training

```jsonc
{
  "gpu_count": 2,  // Request 2 GPUs
  "gpu_type": "A100 SXM 80GB"
}
```

```python
# In training script
training_args = TrainingArguments(
    # Automatically uses all available GPUs
    per_device_train_batch_size=4,  # Per GPU
    # Effective batch = 4 * 2 GPUs * gradient_accumulation
)
```

### Gradient Checkpointing (Save VRAM)

```python
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
```

## Training Cost Estimates

### LLM Fine-tuning

| Task | GPU | Time | Cost |
|------|-----|------|------|
| QLoRA 7B (1K samples) | RTX 4090 | 30 min | ~$0.22 |
| QLoRA 7B (10K samples) | RTX 4090 | 3 hours | ~$1.32 |
| QLoRA 70B (1K samples) | 2x A100 | 2 hours | ~$7.16 |
| Full FT 7B (10K samples) | A100 80GB | 8 hours | ~$14.32 |

### Image LoRA Training

| Task | GPU | Time | Cost |
|------|-----|------|------|
| SDXL LoRA (20 images) | RTX 4090 | 20 min | ~$0.15 |
| FLUX LoRA (20 images) | A100 80GB | 45 min | ~$1.34 |
| Kohya SDXL (100 images) | RTX 4090 | 2 hours | ~$0.88 |

### Classifier Training

| Task | GPU | Time | Cost |
|------|-----|------|------|
| ViT fine-tune (1K images) | RTX 4090 | 10 min | ~$0.07 |
| ResNet fine-tune (10K images) | RTX 4090 | 30 min | ~$0.22 |

## Output Format

When generating training workflows:

```markdown
## Training Pipeline for [Task]

I've created a training pipeline with the following:

### Configuration

- **Model**: [base model]
- **Method**: [QLoRA/LoRA/Full Fine-tune]
- **GPU**: [type] @ $X.XX/hr
- **Estimated time**: [duration]
- **Estimated cost**: $X.XX

### Files Created

1. `gpu.jsonc` - GPU CLI configuration
2. `train.py` - Training script with [features]
3. `requirements.txt` - Dependencies
4. `data/` - Put your training data here

### Data Requirements

[Specific format for their data]

### Usage

```bash
# 1. Add your data
# 2. Start training
gpu run python train.py

# 3. Monitor progress
# Checkpoints saved every N steps
```

### After Training

Your trained model will be at `output/final/`

To use it:
```python
[Example inference code]
```
```
