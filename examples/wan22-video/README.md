# AI Video Generation with Wan 2.2

Generate high-quality videos from text prompts using [Wan 2.2](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B), the state-of-the-art open-source video generation model with Mixture-of-Experts (MoE) architecture.

## Prerequisites

- GPU CLI installed and authenticated (`gpu auth login`)
- RunPod account with API key configured
- Access to A100 80GB or H100 GPU (Wan 2.2 14B requires ~40GB VRAM)

## Quick Start

```bash
# Generate a video (first run downloads ~50GB model)
gpu run python generate.py --prompt "A cat walking through a sunlit garden"

# With more options
gpu run python generate.py \
  --prompt "Ocean waves crashing on rocks at golden hour" \
  --steps 30 \
  --num-frames 81 \
  --width 1280 \
  --height 720
```

Generated videos sync to your local `outputs/` folder automatically.

## Network Volume Setup (Highly Recommended)

Wan 2.2 models are ~50GB. Without a Network Volume, you'll re-download on every session.

1. Go to [RunPod Storage](https://www.runpod.io/console/user/storage)
2. Click **+ New Network Volume**
3. Choose a datacenter with A100 availability (e.g., EU-RO-1, US-OR-1)
4. Set size to **200GB** (models + workspace)
5. Copy the **Volume ID** (e.g., `abc123xyz`)
6. Add to your `gpu.jsonc`:

```jsonc
{
  "network_volume_id": "YOUR_VOLUME_ID"
}
```

**Benefits:**
- 50GB model downloads once, persists across sessions
- Session startup: 1-2 minutes vs 15+ minutes
- Cost savings from faster startup

## Generation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt` | (required) | Text description of the video |
| `--negative-prompt` | "blurry, low quality..." | What to avoid |
| `--steps` | 25 | Inference steps (higher = better quality, slower) |
| `--guidance-scale` | 7.5 | How closely to follow the prompt |
| `--num-frames` | 81 | Number of frames (~5 seconds at 16fps) |
| `--width` | 1280 | Video width |
| `--height` | 720 | Video height (720p) |
| `--fps` | 16 | Output video framerate |
| `--seed` | random | Seed for reproducibility |
| `--output` | outputs | Output directory |

## Example Prompts

```bash
# Nature scenes
gpu run python generate.py --prompt "A butterfly landing on a flower in slow motion"
gpu run python generate.py --prompt "Northern lights dancing over a snowy mountain"

# Abstract/artistic
gpu run python generate.py --prompt "Colorful paint swirling in water, macro shot"
gpu run python generate.py --prompt "Geometric shapes morphing and transforming"

# Characters/action
gpu run python generate.py --prompt "A robot walking through a futuristic city"
gpu run python generate.py --prompt "Astronaut floating in space with Earth in background"
```

## Performance Tips

### Quality vs Speed

| Steps | Time (A100) | Quality |
|-------|-------------|---------|
| 15 | ~2 min | Draft |
| 25 | ~4 min | Good (default) |
| 30 | ~5 min | High |
| 50 | ~8 min | Maximum |

### Resolution Options

| Resolution | VRAM | Use Case |
|------------|------|----------|
| 480p (854x480) | ~30GB | Fast iteration, previews |
| 720p (1280x720) | ~40GB | Production (default) |
| 1080p (1920x1080) | ~60GB+ | High quality (H100 recommended) |

### Duration

- Default: 81 frames = ~5 seconds at 16fps
- Maximum: ~160 frames (~10 seconds) on A100 80GB
- For longer videos, generate segments and concatenate

## Troubleshooting

### "CUDA out of memory"

Try these in order:
1. Reduce resolution: `--width 854 --height 480`
2. Reduce frames: `--num-frames 49` (~3 seconds)
3. Use a larger GPU (H100 80GB)

### Slow generation

- First run downloads 50GB model - use Network Volume to avoid re-downloading
- Reduce `--steps` for faster iterations
- Use 480p for quick previews, then 720p for final

### Video looks blurry/distorted

- Increase `--steps` to 30 or higher
- Improve your prompt with more detail
- Add specific negative prompts: `--negative-prompt "blurry, distorted, low resolution, watermark, text"`

## Configuration

The `gpu.jsonc` file:

```jsonc
{
  "$schema": "https://gpu-cli.sh/schema/v1/gpu.json",
  "project_id": "wan22-video",
  "provider": "runpod",
  "outputs": ["outputs/"],
  "gpu_type": "NVIDIA A100 80GB PCIe",
  "min_vram": 40,

  // Pre-download model
  "download": [
    { "strategy": "hf", "source": "Wan-AI/Wan2.2-T2V-A14B-Diffusers" }
  ],

  // ffmpeg for video encoding
  "environment": {
    "system": { "apt": [{ "name": "ffmpeg" }] }
  }
}
```

### GPU Options

| GPU | VRAM | Max Resolution | Cost/hr |
|-----|------|----------------|---------|
| A100 80GB | 80GB | 1080p | ~$1.74 |
| H100 80GB | 80GB | 1080p | ~$3.99 |
| A100 40GB | 40GB | 720p | ~$1.19 |

## Files

```
wan22-video/
├── gpu.jsonc          # GPU CLI configuration
├── generate.py        # Video generation script
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── outputs/           # Generated videos (gitignored)
```

## Model Information

**Wan 2.2 T2V-A14B** uses a Mixture-of-Experts architecture:
- 27B total parameters, 14B active per step
- Two experts: high-noise (layout) + low-noise (details)
- Supports 480p and 720p at up to 81 frames
- Released July 2025, outperforms commercial models on Wan-Bench 2.0

**License:** Model has its own license - check [HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) for terms.
