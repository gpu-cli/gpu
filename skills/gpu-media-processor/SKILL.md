---
name: gpu-media-processor
description: "Process audio, video, and media on cloud GPUs. Transcribe with Whisper, clone voices, generate videos, upscale images, and run batch media processing. All results sync back to your Mac."
---

# GPU Media Processor

**Transform audio, video, and images on cloud GPUs.**

This skill handles media processing workflows: transcription, voice cloning, video generation, image upscaling, and batch processing.

## When to Use This Skill

| Request Pattern | This Skill Handles |
|-----------------|-------------------|
| "Transcribe my audio/video" | Whisper transcription |
| "Clone my voice" | XTTS, RVC voice cloning |
| "Generate a video from text" | Hunyuan, CogVideo, Mochi |
| "Upscale my images" | Real-ESRGAN, GFPGAN |
| "Process all my files" | Batch media processing |
| "Add subtitles to my video" | Whisper + subtitle generation |
| "Remove background from images" | Segment Anything, rembg |
| "Enhance my photos" | Face restoration, denoising |

## Audio Processing

### Whisper Transcription

**Best for**: Transcribing podcasts, meetings, interviews, videos

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "whisper-transcription",
  "gpu_type": "RTX 4090",
  "min_vram": 10,
  "outputs": ["transcripts/"],
  "cooldown_minutes": 5,
  "download": [
    { "strategy": "hf", "source": "openai/whisper-large-v3" }
  ],
  "environment": {
    "system": {
      "apt": [{ "name": "ffmpeg" }]
    },
    "python": {
      "pip_global": [
        { "name": "openai-whisper" },
        { "name": "tqdm" }
      ]
    }
  }
}
```

**Transcription script:**
```python
#!/usr/bin/env python3
"""Batch transcription with Whisper."""

import whisper
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("audio")
OUTPUT_DIR = Path("transcripts")
MODEL_SIZE = "large-v3"

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loading Whisper {MODEL_SIZE}...")
    model = whisper.load_model(MODEL_SIZE)

    # Find all audio/video files
    extensions = ["*.mp3", "*.wav", "*.m4a", "*.mp4", "*.mkv", "*.webm"]
    files = []
    for ext in extensions:
        files.extend(INPUT_DIR.glob(ext))

    print(f"Found {len(files)} files to transcribe")

    for file_path in tqdm(files, desc="Transcribing"):
        result = model.transcribe(
            str(file_path),
            language="en",  # Or None for auto-detect
            task="transcribe",  # Or "translate" for translation to English
            verbose=False
        )

        # Save transcript
        output_path = OUTPUT_DIR / f"{file_path.stem}.txt"
        output_path.write_text(result["text"])

        # Save with timestamps (SRT format)
        srt_path = OUTPUT_DIR / f"{file_path.stem}.srt"
        write_srt(result["segments"], srt_path)

        print(f"Saved: {output_path}")

def write_srt(segments, output_path):
    """Write SRT subtitle file."""
    with open(output_path, "w") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# 1. Put audio/video files in audio/ folder
# 2. Run transcription
gpu run python transcribe.py

# Results sync back to transcripts/
```

**Performance:**
- ~10x realtime on RTX 4090 (1 hour audio = 6 minutes processing)
- ~$0.04 per hour of audio transcribed

### WhisperX (Better Accuracy + Diarization)

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "whisperx-transcription",
  "gpu_type": "RTX 4090",
  "min_vram": 12,
  "outputs": ["transcripts/"],
  "download": [
    { "strategy": "hf", "source": "openai/whisper-large-v3" },
    { "strategy": "hf", "source": "pyannote/speaker-diarization-3.1" }
  ],
  "environment": {
    "system": {
      "apt": [{ "name": "ffmpeg" }]
    },
    "python": {
      "pip_global": [
        { "name": "whisperx" },
        { "name": "pyannote.audio" }
      ]
    }
  }
}
```

```python
#!/usr/bin/env python3
"""WhisperX transcription with speaker diarization."""

import whisperx
from pathlib import Path

def transcribe_with_speakers(audio_path: str, hf_token: str):
    device = "cuda"

    # 1. Transcribe with Whisper
    model = whisperx.load_model("large-v3", device)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)

    # 3. Speaker diarization
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    return result

# Output includes speaker labels:
# [SPEAKER_00]: "Hello, welcome to the podcast."
# [SPEAKER_01]: "Thanks for having me."
```

### Voice Cloning (XTTS)

**Clone your voice from audio samples**

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "voice-clone",
  "gpu_type": "RTX 4090",
  "min_vram": 8,
  "outputs": ["output/"],
  "download": [
    { "strategy": "hf", "source": "coqui/XTTS-v2" }
  ],
  "environment": {
    "python": {
      "pip_global": [
        { "name": "TTS" }
      ]
    }
  }
}
```

```python
#!/usr/bin/env python3
"""Voice cloning with XTTS-v2."""

from TTS.api import TTS
from pathlib import Path

# Reference audio (your voice samples)
REFERENCE_AUDIO = "reference/my_voice.wav"  # 6-30 seconds of clean speech
OUTPUT_DIR = Path("output")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load XTTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    # Texts to generate
    texts = [
        "Hello! This is my cloned voice speaking.",
        "GPU CLI makes it easy to run AI on cloud GPUs.",
        "Pretty amazing, right?",
    ]

    for i, text in enumerate(texts):
        output_path = OUTPUT_DIR / f"generated_{i}.wav"
        tts.tts_to_file(
            text=text,
            speaker_wav=REFERENCE_AUDIO,
            language="en",
            file_path=str(output_path)
        )
        print(f"Generated: {output_path}")

if __name__ == "__main__":
    main()
```

**Tips for best results:**
- Use 6-30 seconds of clean speech
- Avoid background noise
- Multiple samples improve quality
- Match the speaking style you want

### Voice Conversion (RVC)

**Transform singing/speaking voice to another voice**

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "rvc-voice-conversion",
  "gpu_type": "RTX 4090",
  "min_vram": 8,
  "outputs": ["output/"],
  "environment": {
    "shell": {
      "steps": [
        { "run": "git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git /workspace/RVC", "only_once": true },
        { "run": "cd /workspace/RVC && pip install -r requirements.txt", "only_once": true }
      ]
    }
  }
}
```

## Video Generation

### Hunyuan Video

**Best for**: High quality, longer videos

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "hunyuan-video",
  "gpu_type": "A100 SXM 80GB",
  "min_vram": 80,
  "outputs": ["output/"],
  "cooldown_minutes": 15,
  "download": [
    { "strategy": "hf", "source": "tencent/HunyuanVideo", "timeout": 14400 }
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

```python
#!/usr/bin/env python3
"""Generate videos with Hunyuan Video."""

import torch
from diffusers import HunyuanVideoPipeline
from diffusers.utils import export_to_video

PROMPT = "A golden retriever running on a beach at sunset, cinematic quality"
OUTPUT_PATH = "output/generated.mp4"

def main():
    pipe = HunyuanVideoPipeline.from_pretrained(
        "tencent/HunyuanVideo",
        torch_dtype=torch.float16
    ).to("cuda")

    # Enable memory optimizations
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    video = pipe(
        prompt=PROMPT,
        num_frames=49,  # ~2 seconds at 24fps
        height=720,
        width=1280,
        num_inference_steps=50,
    ).frames[0]

    export_to_video(video, OUTPUT_PATH, fps=24)
    print(f"Video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
```

### CogVideoX

**Best for**: Good balance of quality and speed

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "cogvideox",
  "gpu_type": "A100 PCIe 80GB",
  "min_vram": 40,
  "outputs": ["output/"],
  "download": [
    { "strategy": "hf", "source": "THUDM/CogVideoX-5b", "timeout": 7200 }
  ],
  "environment": {
    "system": {
      "apt": [{ "name": "ffmpeg" }]
    },
    "python": {
      "pip_global": [
        { "name": "diffusers", "version": ">=0.30.0" },
        { "name": "transformers" },
        { "name": "accelerate" }
      ]
    }
  }
}
```

```python
#!/usr/bin/env python3
"""Generate videos with CogVideoX."""

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

PROMPT = "A cat playing piano, 4K, high quality"
OUTPUT_PATH = "output/cogvideo.mp4"

def main():
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    pipe.enable_model_cpu_offload()

    video = pipe(
        prompt=PROMPT,
        num_frames=49,
        guidance_scale=6,
        num_inference_steps=50,
    ).frames[0]

    export_to_video(video, OUTPUT_PATH, fps=8)
    print(f"Video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
```

### Mochi-1

**Best for**: Motion quality, newer model

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "mochi-video",
  "gpu_type": "A100 PCIe 80GB",
  "min_vram": 40,
  "outputs": ["output/"],
  "download": [
    { "strategy": "hf", "source": "genmo/mochi-1-preview", "timeout": 7200 }
  ]
}
```

## Image Processing

### Real-ESRGAN (Image Upscaling)

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "image-upscale",
  "gpu_type": "RTX 4090",
  "min_vram": 8,
  "outputs": ["output/"],
  "environment": {
    "python": {
      "pip_global": [
        { "name": "realesrgan" }
      ]
    }
  }
}
```

```python
#!/usr/bin/env python3
"""Upscale images with Real-ESRGAN."""

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from PIL import Image
import numpy as np
from pathlib import Path

INPUT_DIR = Path("images")
OUTPUT_DIR = Path("output")
SCALE = 4  # 4x upscale

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=SCALE,
        model_path='weights/RealESRGAN_x4plus.pth',
        model=model,
        device='cuda'
    )

    # Process images
    for img_path in INPUT_DIR.glob("*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            img = Image.open(img_path)
            img_array = np.array(img)

            output, _ = upsampler.enhance(img_array, outscale=SCALE)

            output_path = OUTPUT_DIR / f"{img_path.stem}_upscaled{img_path.suffix}"
            Image.fromarray(output).save(output_path)
            print(f"Upscaled: {img_path.name} -> {output_path.name}")

if __name__ == "__main__":
    main()
```

### GFPGAN (Face Restoration)

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "face-restore",
  "gpu_type": "RTX 4090",
  "min_vram": 8,
  "outputs": ["output/"],
  "environment": {
    "python": {
      "pip_global": [
        { "name": "gfpgan" }
      ]
    }
  }
}
```

```python
#!/usr/bin/env python3
"""Restore faces in photos with GFPGAN."""

from gfpgan import GFPGANer
from pathlib import Path
import cv2

INPUT_DIR = Path("images")
OUTPUT_DIR = Path("output")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    restorer = GFPGANer(
        model_path='weights/GFPGANv1.4.pth',
        upscale=2,
        arch='clean',
        device='cuda'
    )

    for img_path in INPUT_DIR.glob("*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img = cv2.imread(str(img_path))

            _, _, output = restorer.enhance(
                img,
                has_aligned=False,
                only_center_face=False
            )

            output_path = OUTPUT_DIR / f"{img_path.stem}_restored{img_path.suffix}"
            cv2.imwrite(str(output_path), output)
            print(f"Restored: {img_path.name}")

if __name__ == "__main__":
    main()
```

### Background Removal (rembg)

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "remove-background",
  "gpu_type": "RTX 4090",
  "min_vram": 4,
  "outputs": ["output/"],
  "environment": {
    "python": {
      "pip_global": [
        { "name": "rembg[gpu]" }
      ]
    }
  }
}
```

```python
#!/usr/bin/env python3
"""Remove backgrounds from images."""

from rembg import remove
from PIL import Image
from pathlib import Path

INPUT_DIR = Path("images")
OUTPUT_DIR = Path("output")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for img_path in INPUT_DIR.glob("*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img = Image.open(img_path)
            output = remove(img)

            output_path = OUTPUT_DIR / f"{img_path.stem}_nobg.png"
            output.save(output_path)
            print(f"Processed: {img_path.name}")

if __name__ == "__main__":
    main()
```

## Batch Processing Patterns

### Parallel Processing Template

```python
#!/usr/bin/env python3
"""Parallel batch processing template."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
MAX_WORKERS = 4  # Adjust based on VRAM

def process_file(file_path: Path):
    """Process a single file. Override this."""
    # Your processing logic here
    pass

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    files = list(INPUT_DIR.glob("*"))
    print(f"Processing {len(files)} files with {MAX_WORKERS} workers")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(
            executor.map(process_file, files),
            total=len(files),
            desc="Processing"
        ))

if __name__ == "__main__":
    main()
```

### Progress Reporting

```python
from tqdm import tqdm
import json

def save_progress(completed: list, output_file: str = "progress.json"):
    """Save progress for resume capability."""
    with open(output_file, "w") as f:
        json.dump({"completed": completed}, f)

def load_progress(output_file: str = "progress.json") -> list:
    """Load progress for resume."""
    try:
        with open(output_file) as f:
            return json.load(f)["completed"]
    except FileNotFoundError:
        return []
```

## Cost Estimates

### Audio Processing

| Task | GPU | Time per Hour of Audio | Cost |
|------|-----|----------------------|------|
| Whisper large-v3 | RTX 4090 | 6 min | ~$0.04 |
| WhisperX + Diarization | RTX 4090 | 10 min | ~$0.07 |
| Voice cloning (XTTS) | RTX 4090 | 30 sec | ~$0.01 |

### Video Generation

| Task | GPU | Time per 5-sec Video | Cost |
|------|-----|---------------------|------|
| Hunyuan Video | A100 80GB | 5-10 min | ~$0.30 |
| CogVideoX | A100 80GB | 3-5 min | ~$0.15 |
| Mochi-1 | A100 80GB | 3-5 min | ~$0.15 |

### Image Processing

| Task | GPU | Images/Hour | Cost per 100 Images |
|------|-----|-------------|-------------------|
| Real-ESRGAN 4x | RTX 4090 | ~500 | ~$0.09 |
| GFPGAN | RTX 4090 | ~1000 | ~$0.04 |
| Background removal | RTX 4090 | ~2000 | ~$0.02 |

## Output Format

When setting up a media processing workflow:

```markdown
## [Task] Pipeline

I've created a [task] pipeline that processes your [media type].

### Configuration

- **GPU**: [type] @ $X.XX/hr
- **Processing speed**: [estimate]
- **Cost estimate**: $X.XX per [unit]

### Setup

1. Add your files to `input/`
2. Run the processing:
   ```bash
   gpu run python process.py
   ```
3. Results sync to `output/`

### Files Created

- `gpu.jsonc` - GPU configuration
- `process.py` - Processing script
- `requirements.txt` - Dependencies

### Output Format

[Describe what output files look like]
```
