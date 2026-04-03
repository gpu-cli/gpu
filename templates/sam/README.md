# SAM 3 Video Segmentation

Segment and track anything in images and video using text prompts. Type "dog", "person in red shirt", or any visual concept and [SAM 3.1](https://github.com/facebookresearch/sam3) by Meta AI finds, outlines, and tracks them.

## Quick Start

```bash
gpu use .
```

## Prerequisites

SAM 3.1 is a **gated model** on HuggingFace. Before running:

1. Create a HuggingFace account at https://huggingface.co
2. Accept the SAM 3 license at https://huggingface.co/facebook/sam3
3. Generate an access token at https://huggingface.co/settings/tokens
4. Add it to GPU CLI: `gpu auth add hf`

## How It Works

**Video Segmentation** (main feature):
1. Upload an MP4 video (up to 60 seconds)
2. Type what you want to track (e.g. "person", "red car")
3. SAM 3 segments the object in the first frame and tracks it through every subsequent frame
4. Download the annotated video with colored masks and tracking IDs

**Image Segmentation:**
1. Upload any image
2. Describe the objects to find
3. Adjust confidence threshold to filter detections
4. Get the segmented image with colored masks and bounding boxes

## Features

- Open-vocabulary detection: describe any visual concept in natural language
- Multi-object tracking: segment and track multiple objects simultaneously
- Real-time performance: ~4GB VRAM, processes video at interactive speeds
- Confidence filtering: adjustable threshold for precision vs recall

## Hardware Requirements

- **GPU**: 16GB+ VRAM (RTX 4090, A5000, L40S, A40)
- **Storage**: ~50GB (model weights + video workspace)
- **Actual VRAM usage**: ~4GB during inference

## Notes

- Uses the same stable RunPod PyTorch base image as the other official templates.
- First run may still take a few minutes because the model weights are downloaded on demand.

## Files

| File | Purpose |
|------|---------|
| `gpu.jsonc` | GPU CLI configuration |
| `startup.sh` | Dependency install + server launch |
| `server.py` | Gradio web UI with video + image segmentation |

## Output Syncing

All annotated outputs are saved to `outputs/` and automatically synced back to your local machine:
- `seg_*.png` — segmented images
- `tracked_*.mp4` — annotated videos with tracking

## Tips

- **Short prompts work best**: "dog" outperforms "a cute fluffy dog sitting on grass"
- **Use attributes to distinguish**: "person in red shirt" vs "person in blue jacket"
- **Adjust threshold**: Lower (0.2) for more detections, higher (0.7) for confident ones only
- **Video length**: Keep videos under 60s for responsive processing

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Access denied" on model download | Accept license at https://huggingface.co/facebook/sam3 and run `gpu auth add hf` |
| Slow first run | Model download (~3.4GB) happens once; subsequent runs use cached weights |
| No detections found | Try a simpler prompt (single word) or lower the confidence threshold |
| Video too long | Template caps at 60s; trim longer videos before uploading |
