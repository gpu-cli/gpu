# Hunyuan3D-2.1: Image-to-3D with PBR Textures

Run Tencent's [Hunyuan3D-2.1](https://huggingface.co/spaces/tencent/Hunyuan3D-2.1) on cloud GPUs with a single command. Generate high-fidelity 3D meshes with physically-based rendering (PBR) textures from images.

## Quick Start

```bash
# From this template directory:
gpu use .
```

After the pod is ready, open the Gradio URL shown in the terminal to access the web UI.

## What It Does

Hunyuan3D-2.1 is a production-ready 3D asset generation system featuring:

- **Shape Generation** (3.3B params) - Creates high-quality 3D geometry from a single image
- **PBR Texture Synthesis** (2B params) - Generates photorealistic textures with metallic/roughness maps
- **Multiple Export Formats** - GLB, OBJ, PLY, STL
- **Interactive Web UI** - Gradio-based interface with real-time 3D preview

### Generation Modes

| Mode | Description | Time | VRAM |
|------|-------------|------|------|
| **Gen Shape** | Geometry only (white mesh) | ~10-30s | ~10 GB |
| **Gen Textured Shape** | Shape + PBR textures | ~60-120s | ~29 GB |

## Architecture

This template takes a **dedicated approach** rather than a generic HuggingFace Space runner, because Hunyuan3D-2.1 has:

1. **Custom CUDA C++ extensions** that must be compiled (custom_rasterizer, DifferentiableRenderer)
2. **Specific CUDA/PyTorch version requirements** (CUDA 12.4, PyTorch 2.4+)
3. **ZeroGPU decorators** (`@spaces.GPU`) that need mocking for real GPU hardware
4. **Complex dependency chain** with potential conflicts (bpy, cupy, basicsr)

### Startup Flow

```
1. Download HF Space (huggingface_hub snapshot_download)
2. Install Space dependencies (with torch version constraints)
3. Compile custom CUDA extensions (custom_rasterizer + mesh painter)
4. Create 'spaces' mock module (replaces ZeroGPU infrastructure)
5. Patch gradio_app.py for RunPod (skip HF-specific setup)
6. Launch Gradio UI on port 7860
```

## Files

```
image-to-3d/
├── gpu.jsonc         # GPU CLI configuration (GPU reqs, ports, hooks, environment)
├── startup.sh        # Bootstrap: download, compile, patch, launch
├── requirements.txt  # Base Python dependencies (installed during env setup)
├── README.md         # This file
├── space/            # Downloaded HF Space (created at runtime, gitignored)
└── outputs/          # Generated 3D meshes (synced back to local machine)
```

## Configuration

Default settings in `gpu.jsonc`:

| Setting | Value | Notes |
|---------|-------|-------|
| **GPU** | A100 80GB / RTX 6000 Ada | 29GB needed for shape + texture |
| **Min VRAM** | 40 GB | Comfortable headroom for PBR pipeline |
| **Workspace** | 100 GB | Models (~15GB) + generated assets |
| **Port** | 7860 | Gradio web UI |
| **Keep Alive** | 60 min | Model loading takes 2-5 minutes |

### Using a Smaller GPU

For **shape generation only** (no textures), you can use a smaller GPU:

```jsonc
// In gpu.jsonc, change:
"gpu_types": [
  { "type": "NVIDIA GeForce RTX 4090" },
  { "type": "NVIDIA L40S" }
],
"min_vram": 24
```

## Performance

| Phase | First Run | Subsequent Runs |
|-------|-----------|-----------------|
| Space download | ~1-2 min | Cached |
| Dependency install | ~3-5 min | Cached (`only_once`) |
| CUDA compilation | ~2-3 min | Cached |
| Model download | ~2-5 min | Cached in workspace |
| Shape generation | ~10-30s | ~10-30s |
| Texture generation | ~30-90s | ~30-90s |

**Total first cold start**: ~15-20 minutes (downloads + compilation + model loading)
**Subsequent runs**: Pod ready in ~3-5 minutes (model loading only)

## Advanced Options

### Low VRAM Mode

The template runs with `--low_vram_mode` by default, which offloads models to CPU when not actively generating. This allows the full pipeline to work on GPUs with less available VRAM.

### Network Volume (Recommended)

For faster startup with cached models, use a RunPod Network Volume:

1. Create a 200GB volume at [RunPod Storage](https://www.runpod.io/console/user/storage)
2. Add to `gpu.jsonc`: `"network_volume_id": "YOUR_VOLUME_ID"`

Benefits:
- Models persist across pod restarts
- Compiled extensions persist
- Downloaded Space persists
- Saves 5-10 minutes on every cold start

## Troubleshooting

### CUDA compilation errors

The template uses `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` which includes CUDA development headers. If compilation fails:

```bash
# Check CUDA is available
nvidia-smi
nvcc --version

# Try recompiling manually
cd space/hy3dpaint/DifferentiableRenderer
bash compile_mesh_painter.sh
```

### Out of memory

```bash
# Use shape-only mode (Gen Shape button) - requires only ~10GB
# Or use a larger GPU (A100 80GB recommended for full pipeline)
```

### Models not downloading

```bash
# Check HuggingFace connectivity
python3 -c "from huggingface_hub import HfApi; print(HfApi().model_info('tencent/Hunyuan3D-2.1'))"
```

### Texture generation not working

This usually means custom_rasterizer or DifferentiableRenderer failed to compile:

```bash
# Check if custom_rasterizer is installed
python3 -c "import custom_rasterizer; print('OK')"

# Recompile by removing marker and restarting
rm space/.compile_complete
bash startup.sh
```

## Model Details

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Shape Generator | `hunyuan3d-dit-v2-1` | 3.3B params | Image-to-3D geometry |
| Texture Generator | `hunyuan3d-paint-v2-1` | 2B params | PBR material synthesis |
| Background Remover | Built-in (rembg) | ~100MB | Input image preprocessing |
| Face Reducer | Built-in (trimesh) | N/A | Mesh simplification |

## Resources

- [Hunyuan3D-2.1 HuggingFace Space](https://huggingface.co/spaces/tencent/Hunyuan3D-2.1)
- [Hunyuan3D-2 GitHub Repository](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- [Technical Report (arXiv)](https://arxiv.org/abs/2506.15442)
- [GPU CLI Documentation](https://gpu-cli.sh/docs)

## License

Hunyuan3D-2.1 is licensed under the [Tencent Hunyuan Non-Commercial License](https://huggingface.co/spaces/tencent/Hunyuan3D-2.1/blob/main/LICENSE). Please review the license terms before use.
