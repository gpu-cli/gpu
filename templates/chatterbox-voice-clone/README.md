# Chatterbox Voice Clone

GPU CLI template for voice cloning using [Chatterbox-Turbo](https://github.com/resemble-ai/chatterbox) TTS by Resemble AI. Clone any voice from a YouTube URL or uploaded audio, then generate speech in that voice.

## Quick Start

```bash
# Start the voice cloning UI (~2-3 min first run, downloads model)
gpu use .
```

Open the URL shown in terminal to access the Gradio web UI.

## How It Works

1. **Paste a YouTube URL** containing the voice you want to clone (or upload/record audio)
2. **Enter text** you want spoken in that voice
3. **Click "Clone Voice"** to generate speech

The first 60 seconds of the YouTube audio are extracted as a voice reference. Chatterbox-Turbo then generates new speech matching that voice.

## Features

- **YouTube voice extraction** - Paste any YouTube URL to clone the speaker's voice
- **Audio upload/recording** - Alternatively upload a WAV/MP3 or record via microphone
- **Paralinguistic tags** - Add `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]`, `[gasp]`, `[groan]`, `[sniff]`, `[shush]`, `[clear throat]` for natural expression
- **Advanced controls** - Temperature, top-p, top-k, repetition penalty, seed for reproducibility
- **Loudness normalization** - Output normalized to -27 LUFS by default
- **Auto-sync** - Generated audio files automatically sync to your local `outputs/` directory

## Model Details

- **Chatterbox-Turbo** - 350M parameter TTS model by Resemble AI
- **Zero-shot voice cloning** - No fine-tuning needed, works from a single audio reference
- **Fast inference** - Distilled speech-token-to-mel decoder (1 step vs 10)
- **PerTh watermarking** - All outputs include imperceptible watermarks for responsible AI

## Hardware Requirements

- **GPU**: RTX 4090 or equivalent (16GB+ VRAM recommended)
- **Storage**: ~10GB for model weights + dependencies
- **Python**: 3.11+ required

## Files

| File | Purpose |
|------|---------|
| `gpu.jsonc` | GPU CLI configuration (ports, startup, environment) |
| `startup.sh` | Startup script: installs deps, launches Gradio server |
| `server.py` | Gradio web UI with YouTube extraction + Chatterbox-Turbo inference |
| `README.md` | This documentation |

## Output Syncing

Generated audio files are saved to `outputs/` and automatically synced back to your local machine.

## Tips

- **Short references work best** - 10-30 seconds of clear speech is ideal
- **Clean audio matters** - References with minimal background noise produce better clones
- **Use tags for realism** - Adding `[chuckle]` or `[sigh]` makes output sound more natural
- **Adjust temperature** - Lower (0.3-0.5) for consistent output, higher (0.8-1.2) for variety
- **Set a seed** - Use a non-zero seed to reproduce the exact same generation

## Troubleshooting

### "Failed to download YouTube audio"
- Ensure the YouTube URL is valid and the video is publicly accessible
- Some videos may have download restrictions

### Model loading is slow
First run downloads ~2GB of model weights. Subsequent runs use the cached weights from the workspace volume.

### Out of memory
Chatterbox-Turbo needs ~4-6GB VRAM. If you see OOM errors, try a GPU with more VRAM.

### Audio quality issues
- Try a cleaner voice reference (less background noise)
- Adjust temperature and repetition penalty
- Use a longer reference clip (10-30s of speech)

## Responsible Use

All Chatterbox outputs include [PerTh watermarks](https://github.com/resemble-ai/perth) - imperceptible neural watermarks that survive compression and editing. Use voice cloning responsibly and ethically.
