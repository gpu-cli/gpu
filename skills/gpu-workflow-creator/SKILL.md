---
name: gpu-workflow-creator
description: "Transform natural language requests into complete GPU CLI workflows. The ultimate skill for Mac users who want to run NVIDIA GPU workloads without configuration complexity. Describe what you want, get a working project."
---

# GPU Workflow Creator

**Transform plain English into production GPU workflows.**

This skill is the magic bridge for Mac users: describe what you want to accomplish, and get a complete, runnable GPU CLI project that executes on cloud NVIDIA GPUs.

## The Promise

```
User: "I want to train a LoRA on my photos so I can generate images of myself in different styles"

You generate:
├── gpu.jsonc          # Optimal GPU config (A100 for training)
├── train.py           # Complete training script
├── generate.py        # Inference script for using the trained LoRA
├── requirements.txt   # All dependencies
├── README.md          # Step-by-step usage instructions
└── dataset/           # Placeholder with instructions for user's photos
```

## When to Use This Skill

Activate for ANY request that involves:
- Running ML/AI workloads on GPUs
- "I want to..." + anything requiring GPU compute
- Natural language descriptions of ML tasks
- Requests that mention models, training, generation, inference
- Questions about how to do GPU tasks from a Mac

**Example Triggers:**
- "I want to fine-tune Llama on my company data"
- "Help me run SDXL to generate product photos"
- "Set up a private ChatGPT-like API"
- "Transcribe all my podcast episodes with Whisper"
- "Clone my voice from audio samples"
- "Generate videos from text prompts"
- "Train an image classifier for my app"

## Core Workflow

### Step 1: Understand Intent

Parse the user's request to identify:

| Dimension | Questions to Answer |
|-----------|---------------------|
| **Task Type** | Training? Inference? Processing? Generation? |
| **Domain** | Images? Text/LLM? Audio? Video? Code? |
| **Scale** | One-off? Batch? API service? |
| **Data** | What inputs? What outputs? |
| **Constraints** | Budget? Speed? Quality? Privacy? |

### Step 2: Select Architecture

Based on intent, choose the right architecture:

| Task Pattern | Architecture |
|--------------|--------------|
| Train model once | Batch job with checkpoint sync |
| Fine-tune LLM | LoRA/QLoRA training pipeline |
| Generate images | ComfyUI or diffusers pipeline |
| Run LLM API | vLLM/TGI server with port forwarding |
| Process files | Batch processing with output sync |
| Interactive UI | Web app with persistent proxy |

### Step 3: Calculate Resources

**GPU Selection Matrix:**

| Workload | Min VRAM | Recommended GPU | Est. Cost |
|----------|----------|-----------------|-----------|
| SD 1.5 inference | 8GB | RTX 4090 | $0.44/hr |
| SDXL inference | 12GB | RTX 4090 | $0.44/hr |
| FLUX inference | 24GB | RTX 4090 | $0.44/hr |
| LLM 7B inference | 16GB | RTX 4090 | $0.44/hr |
| LLM 13B inference | 24GB | RTX 4090 | $0.44/hr |
| LLM 70B inference | 140GB | 2x A100 80GB | $3.58/hr |
| LoRA training SDXL | 24GB | RTX 4090 | $0.44/hr |
| LoRA training FLUX | 48GB | A100 80GB | $1.79/hr |
| LLM fine-tuning 7B | 24GB | A100 40GB | $1.29/hr |
| LLM fine-tuning 70B | 160GB | 2x A100 80GB | $3.58/hr |
| Whisper large-v3 | 10GB | RTX 4090 | $0.44/hr |
| Video generation | 80GB | A100 80GB | $1.79/hr |

### Step 4: Generate Project

Create a complete project structure:

```
project/
├── gpu.jsonc           # GPU CLI configuration
├── main.py             # Primary script
├── requirements.txt    # Python dependencies
├── README.md           # Usage documentation
└── [task-specific files]
```

## Configuration Templates

### Training Job Template

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "training-job",
  "gpu_type": "A100 PCIe 80GB",
  "min_vram": 40,
  "outputs": ["checkpoints/", "logs/", "models/"],
  "cooldown_minutes": 10,  // Longer for training
  "download": [
    // Pre-download base models
  ],
  "environment": {
    "python": {
      "requirements": "requirements.txt"
    }
  }
}
```

### Inference Server Template

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "inference-server",
  "gpu_type": "RTX 4090",
  "min_vram": 24,
  "persistent_proxy": true,  // Keep running for API calls
  "cooldown_minutes": 5,
  "download": [
    // Pre-download model weights
  ]
}
```

### Batch Processing Template

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "batch-processor",
  "gpu_type": "RTX 4090",
  "outputs": ["results/"],
  "cooldown_minutes": 5,
  "environment": {
    "python": {
      "requirements": "requirements.txt"
    }
  }
}
```

## Model Presets Database

### Image Generation

| Model | Source | Size | VRAM | Speed |
|-------|--------|------|------|-------|
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | 35GB | 24GB | Medium |
| FLUX.1-schnell | `black-forest-labs/FLUX.1-schnell` | 35GB | 24GB | Fast |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` | 7GB | 12GB | Fast |
| SD 1.5 | `runwayml/stable-diffusion-v1-5` | 5GB | 8GB | Very Fast |

### Large Language Models

| Model | Source | Size | VRAM | Quality |
|-------|--------|------|------|---------|
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | 16GB | 20GB | Good |
| Llama 3.1 70B | `meta-llama/Llama-3.1-70B-Instruct` | 140GB | 140GB | Excellent |
| Mistral 7B | `mistralai/Mistral-7B-Instruct-v0.3` | 15GB | 16GB | Good |
| Qwen2.5 72B | `Qwen/Qwen2.5-72B-Instruct` | 145GB | 150GB | Excellent |

### Audio Models

| Model | Source | Size | VRAM | Use Case |
|-------|--------|------|------|----------|
| Whisper large-v3 | `openai/whisper-large-v3` | 3GB | 10GB | Transcription |
| XTTS-v2 | `coqui/XTTS-v2` | 2GB | 8GB | Voice cloning |

### Video Generation

| Model | Source | Size | VRAM | Quality |
|-------|--------|------|------|---------|
| Hunyuan Video | `tencent/HunyuanVideo` | 50GB | 80GB | High |
| CogVideoX | `THUDM/CogVideoX-5b` | 20GB | 40GB | Medium |
| Mochi-1 | `genmo/mochi-1-preview` | 20GB | 40GB | High |

## Example Workflows

### Example 1: Personal LoRA Training

**User Request:** "Train a LoRA on 20 photos of my cat so I can generate images of it"

**Generated Project:**

`gpu.jsonc`:
```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "cat-lora-training",
  "gpu_type": "RTX 4090",
  "min_vram": 24,
  "outputs": ["output/", "logs/"],
  "cooldown_minutes": 10,
  "download": [
    { "strategy": "hf", "source": "stabilityai/stable-diffusion-xl-base-1.0", "allow": "*.safetensors", "timeout": 3600 }
  ],
  "environment": {
    "system": {
      "apt": [{ "name": "git" }, { "name": "libgl1" }]
    },
    "python": {
      "requirements": "requirements.txt"
    }
  }
}
```

`requirements.txt`:
```
torch>=2.1.0
diffusers>=0.25.0
transformers>=4.36.0
accelerate>=0.25.0
peft>=0.7.0
datasets>=2.16.0
Pillow>=10.0.0
```

`train.py`:
```python
#!/usr/bin/env python3
"""LoRA training script for SDXL using diffusers."""

import os
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from diffusers.training_utils import EMAModel
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Configuration
INSTANCE_PROMPT = "a photo of sks cat"  # Change 'sks cat' to your subject
OUTPUT_DIR = Path("output")
DATASET_DIR = Path("dataset")
NUM_TRAIN_EPOCHS = 100
LEARNING_RATE = 1e-4
LORA_RANK = 32

class LoRADataset(Dataset):
    def __init__(self, data_dir: Path, prompt: str, size: int = 1024):
        self.images = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
        self.prompt = prompt
        self.size = size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = image.resize((self.size, self.size))
        return {"image": image, "prompt": self.prompt}

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load base model
    print("Loading SDXL base model...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.to("cuda")

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
    )

    # Apply LoRA to UNet
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.print_trainable_parameters()

    # Setup dataset
    dataset = LoRADataset(DATASET_DIR, INSTANCE_PROMPT)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Training loop
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=LEARNING_RATE)

    print(f"Training on {len(dataset)} images for {NUM_TRAIN_EPOCHS} epochs...")
    for epoch in range(NUM_TRAIN_EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_TRAIN_EPOCHS}"):
            # Training step (simplified)
            optimizer.zero_grad()
            # ... training logic
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            # Save checkpoint
            checkpoint_path = OUTPUT_DIR / f"checkpoint-{epoch+1}"
            pipe.unet.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_path = OUTPUT_DIR / "final_lora"
    pipe.unet.save_pretrained(final_path)
    print(f"Training complete! LoRA saved to {final_path}")

if __name__ == "__main__":
    main()
```

`generate.py`:
```python
#!/usr/bin/env python3
"""Generate images using your trained LoRA."""

import torch
from diffusers import StableDiffusionXLPipeline
from pathlib import Path

LORA_PATH = Path("output/final_lora")
PROMPT = "a photo of sks cat wearing a space suit, on the moon"
OUTPUT_DIR = Path("generated")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load base model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    ).to("cuda")

    # Load your LoRA
    pipe.load_lora_weights(str(LORA_PATH))

    # Generate!
    image = pipe(
        prompt=PROMPT,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]

    output_path = OUTPUT_DIR / "generated.png"
    image.save(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# Cat LoRA Training

Train a LoRA model on photos of your cat to generate custom images.

## Setup

1. **Add your photos** to the `dataset/` folder:
   - 10-30 photos work best
   - JPG or PNG format
   - Crop to focus on your cat
   - Variety of angles and lighting

2. **Edit the prompt** in `train.py`:
   ```python
   INSTANCE_PROMPT = "a photo of sks cat"  # Keep 'sks' as the trigger word
   ```

## Training

```bash
# Train the LoRA (~15-30 minutes on RTX 4090)
gpu run python train.py

# Monitor training progress - checkpoints saved every 10 epochs
```

## Generate Images

After training completes:

```bash
# Edit generate.py with your desired prompt
gpu run python generate.py
```

Example prompts:
- "a photo of sks cat wearing a crown"
- "a photo of sks cat in a fantasy forest"
- "a photo of sks cat as an astronaut"

## Files

- `output/final_lora/` - Your trained LoRA weights
- `generated/` - Generated images (synced back to your Mac)

## Costs

- Training: ~$0.30 (30 min @ $0.44/hr)
- Generation: ~$0.02 per batch (3 min per 10 images)
```

### Example 2: Private LLM API Server

**User Request:** "Set up a private ChatGPT-like API with Llama 3.1 70B"

**Generated Project:**

`gpu.jsonc`:
```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "llama-api-server",
  "gpu_type": "A100 SXM 80GB",
  "gpu_count": 2,
  "min_vram": 80,
  "persistent_proxy": true,
  "cooldown_minutes": 10,
  "health_check_paths": ["/health", "/v1/models"],
  "download": [
    { "strategy": "hf", "source": "meta-llama/Llama-3.1-70B-Instruct", "timeout": 14400 }
  ],
  "environment": {
    "python": {
      "pip_global": [
        { "name": "vllm", "version": ">=0.4.0" },
        { "name": "openai" }
      ]
    }
  }
}
```

`README.md`:
```markdown
# Private Llama 3.1 70B API Server

OpenAI-compatible API powered by Llama 3.1 70B running on your own cloud GPUs.

## Start the Server

```bash
gpu run -p 8000:8000 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000 \
    --host 0.0.0.0
```

## Use the API

The server is available at `http://localhost:8000` (forwarded from the cloud GPU).

### With curl:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### With Python:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(response.choices[0].message.content)
```

### With any OpenAI-compatible client:
Just point the base URL to `http://localhost:8000/v1`

## Costs

- 2x A100 80GB: ~$3.58/hr
- Auto-stops after 10 minutes idle
- First request may take ~2 minutes (model loading)

## Data Privacy

- Your data never leaves your control
- No logs sent to third parties
- Model runs on your cloud GPU account
```

### Example 3: Batch Transcription Pipeline

**User Request:** "Transcribe all my podcast episodes with speaker diarization"

**Generated Project:**

`gpu.jsonc`:
```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "podcast-transcription",
  "gpu_type": "RTX 4090",
  "min_vram": 12,
  "outputs": ["transcripts/"],
  "cooldown_minutes": 5,
  "download": [
    { "strategy": "hf", "source": "openai/whisper-large-v3" },
    { "strategy": "hf", "source": "pyannote/speaker-diarization-3.1" }
  ],
  "environment": {
    "system": {
      "apt": [{ "name": "ffmpeg" }]
    },
    "python": {
      "requirements": "requirements.txt"
    }
  }
}
```

`requirements.txt`:
```
torch>=2.1.0
openai-whisper>=20231117
pyannote.audio>=3.1.0
pydub>=0.25.1
tqdm>=4.66.0
```

`transcribe.py`:
```python
#!/usr/bin/env python3
"""Batch transcription with speaker diarization."""

import whisper
from pyannote.audio import Pipeline
from pathlib import Path
from tqdm import tqdm
import torch

INPUT_DIR = Path("audio")
OUTPUT_DIR = Path("transcripts")

def transcribe_with_speakers(audio_path: Path, model, diarization_pipeline):
    """Transcribe audio with speaker labels."""
    # Run diarization
    diarization = diarization_pipeline(str(audio_path))

    # Run transcription
    result = model.transcribe(str(audio_path), word_timestamps=True)

    # Merge speaker info with transcription
    output_lines = []
    for segment in result["segments"]:
        # Find speaker for this segment
        speaker = "SPEAKER_?"
        for turn, _, spk in diarization.itertracks(yield_label=True):
            if turn.start <= segment["start"] <= turn.end:
                speaker = spk
                break

        output_lines.append(f"[{speaker}] {segment['text'].strip()}")

    return "\n".join(output_lines)

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load models
    print("Loading Whisper large-v3...")
    model = whisper.load_model("large-v3")

    print("Loading speaker diarization...")
    diarization = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=True  # Needs HF token
    ).to(torch.device("cuda"))

    # Find all audio files
    audio_files = list(INPUT_DIR.glob("*.mp3")) + list(INPUT_DIR.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files")

    for audio_path in tqdm(audio_files, desc="Transcribing"):
        transcript = transcribe_with_speakers(audio_path, model, diarization)

        output_path = OUTPUT_DIR / f"{audio_path.stem}.txt"
        output_path.write_text(transcript)
        print(f"Saved: {output_path}")

    print(f"\nDone! Transcripts saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# Podcast Transcription Pipeline

Transcribe audio files with automatic speaker identification.

## Setup

1. **Add audio files** to the `audio/` folder:
   - Supports MP3, WAV, M4A
   - Any length (podcasts, interviews, meetings)

2. **Set HuggingFace token** (required for speaker diarization):
   ```bash
   export HF_TOKEN=your_token_here
   ```

## Run Transcription

```bash
gpu run python transcribe.py
```

## Output

Transcripts are saved to `transcripts/` with speaker labels:

```
[SPEAKER_00] Welcome to the podcast. Today we're talking about AI.
[SPEAKER_01] Thanks for having me. I'm excited to dive in.
[SPEAKER_00] Let's start with the basics...
```

## Performance

- ~10-15 seconds per minute of audio on RTX 4090
- 1-hour podcast: ~2 minutes processing
- Batch of 10 episodes: ~20 minutes

## Costs

- RTX 4090: $0.44/hr
- 10 one-hour episodes: ~$0.15 total
```

## Quality Standards

1. **Complete, Runnable Projects**: Every generated project must work with `gpu run` immediately
2. **Optimal GPU Selection**: Never over-provision (waste money) or under-provision (OOM errors)
3. **Clear Documentation**: README explains what, why, and how
4. **Production Patterns**: Error handling, checkpointing, progress tracking
5. **Cost Transparency**: Always estimate costs for the user

## Response Format

When generating a workflow:

```
## Understanding Your Request

[Brief summary of what they want to accomplish]

## Generated Project

I've created a complete project for [task]. Here's what's included:

### Files Created

1. **gpu.jsonc** - Configuration for [GPU type] @ $X.XX/hr
2. **main.py** - [Description of what it does]
3. **requirements.txt** - All Python dependencies
4. **README.md** - Usage instructions

### Quick Start

```bash
# [Setup steps if any]
gpu run python main.py
```

### Cost Estimate

- GPU: [type] @ $X.XX/hr
- Estimated time: [duration]
- **Total cost: ~$X.XX**

### What Happens

1. [Step 1]
2. [Step 2]
3. [Step 3]
4. Results sync back to your Mac at `[path]`
```

## Advanced Patterns

### Multi-Stage Pipelines

For complex workflows (preprocess → train → evaluate):

```jsonc
// Stage 1: Preprocessing (cheap GPU)
{ "project_id": "preprocess", "gpu_type": "RTX 4090", ... }

// Stage 2: Training (powerful GPU)
{ "project_id": "train", "gpu_type": "A100 SXM 80GB", ... }

// Stage 3: Evaluation (cheap GPU)
{ "project_id": "eval", "gpu_type": "RTX 4090", ... }
```

### Persistent Services

For APIs that need to stay running:

```jsonc
{
  "persistent_proxy": true,
  "cooldown_minutes": 30,  // Longer idle timeout
  "health_check_paths": ["/health"]
}
```

### Large Dataset Handling

For workflows with big data:

```jsonc
{
  "network_volume_id": "vol_xxx",  // Persistent storage
  "workspace_size_gb": 200
}
```
