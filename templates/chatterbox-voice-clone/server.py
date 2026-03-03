#!/usr/bin/env python3
"""Chatterbox Voice Clone - Gradio Web UI

Clone any voice from YouTube using Chatterbox-Turbo TTS.
Paste a YouTube URL, enter text, and generate speech in the cloned voice.

Built on:
- Chatterbox-Turbo (350M params) by Resemble AI
- yt-dlp for YouTube audio extraction
- Gradio for the web interface

Supports paralinguistic tags: [laugh], [chuckle], [cough], [sigh], [gasp],
[groan], [sniff], [shush], [clear throat]
"""

import os
import random
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchaudio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = Path(__file__).parent.absolute()
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
TEMP_DIR = SCRIPT_DIR / ".tmp_audio"

# Maximum duration (seconds) to extract from YouTube videos
MAX_YT_DURATION = 60

# Paralinguistic event tags supported by Chatterbox-Turbo
EVENT_TAGS = [
    "[clear throat]",
    "[sigh]",
    "[shush]",
    "[cough]",
    "[groan]",
    "[sniff]",
    "[gasp]",
    "[chuckle]",
    "[laugh]",
]

# ============================================================
# Custom CSS - GPU CLI Terminal Noir Theme
# ============================================================
CUSTOM_CSS = """
.tag-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
    border: none !important;
    background: transparent !important;
}
.tag-btn {
    min-width: fit-content !important;
    width: auto !important;
    height: 32px !important;
    font-size: 13px !important;
    background: #1a1a2e !important;
    border: 1px solid #2d2d44 !important;
    color: #33cc99 !important;
    border-radius: 6px !important;
    padding: 0 10px !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.tag-btn:hover {
    background: #2d2d44 !important;
    transform: translateY(-1px);
}
.status-banner {
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 14px;
}
.status-ready {
    background: #0d2818;
    border: 1px solid #1a5c30;
    color: #33cc99;
}
.status-loading {
    background: #1a1a2e;
    border: 1px solid #2d2d44;
    color: #8888aa;
}
"""

# JavaScript to insert tags at cursor position in textarea
INSERT_TAG_JS = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#text-input textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""


def set_seed(seed: int) -> None:
    """Set random seed for reproducible generation."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    """Load Chatterbox-Turbo model to GPU."""
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
    except RuntimeError as e:
        if "torchvision" in str(e):
            print(
                "\n[ERROR] torchvision version mismatch! "
                "Run: pip install 'torchvision==0.21.0' to match torch 2.6.0\n",
                flush=True,
            )
        raise

    # Chatterbox's from_pretrained() hardcodes `token=os.getenv("HF_TOKEN") or True`
    # which forces token=True when HF_TOKEN is unset, causing an auth error even
    # though the model is public (MIT). Fix: download the snapshot ourselves with
    # token=False, then load from the local path.
    print(f"Loading Chatterbox-Turbo on {DEVICE}...", flush=True)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        # User has a real HF token set — let Chatterbox use it normally
        model = ChatterboxTurboTTS.from_pretrained(DEVICE)
    else:
        # No token: download manually with token=False (model is public/MIT)
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(
            repo_id="ResembleAI/chatterbox-turbo",
            token=False,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
        )
        model = ChatterboxTurboTTS.from_local(local_path, DEVICE)

    print("Model loaded successfully!", flush=True)
    return model


def download_youtube_audio(
    url: str, output_dir: str, duration_limit: int = MAX_YT_DURATION
) -> str:
    """Download audio from YouTube URL using yt-dlp.

    Args:
        url: YouTube video URL
        output_dir: Directory to save the audio file
        duration_limit: Maximum seconds of audio to extract (default 60)

    Returns:
        Path to the downloaded WAV file.
    """
    import yt_dlp

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"yt_{uuid.uuid4().hex[:8]}")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_path}.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": [
            "-ar",
            "24000",  # Chatterbox expects 24kHz
        ],
        "prefer_ffmpeg": True,
        "quiet": True,
        "no_warnings": True,
    }

    # Trim to duration limit
    if duration_limit:
        ydl_opts["postprocessor_args"].extend(
            [
                "-t",
                str(duration_limit),
            ]
        )

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_duration = info.get("duration", 0)

        print(f"  Title: {info.get('title', 'Unknown')}", flush=True)
        print(f"  Duration: {video_duration}s", flush=True)
        if duration_limit:
            actual_duration = min(duration_limit, video_duration)
            print(f"  Extracting first {actual_duration}s of audio", flush=True)

        ydl.download([url])

    wav_path = f"{output_path}.wav"
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Expected WAV file not found at {wav_path}")

    return wav_path


def generate_voice_clone(
    model,
    text: str,
    yt_url: str,
    audio_upload,
    temperature: float,
    seed_num: int,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    norm_loudness: bool,
):
    """Generate speech cloning a voice from YouTube or uploaded audio.

    Args:
        model: Loaded Chatterbox-Turbo model
        text: Text to synthesize
        yt_url: YouTube URL for voice reference
        audio_upload: Uploaded audio file path (alternative to YouTube)
        temperature: Sampling temperature (0.05-2.0)
        seed_num: Random seed (0 for random)
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
        norm_loudness: Whether to normalize output loudness

    Returns:
        Tuple of (sample_rate, audio_numpy_array) for Gradio Audio component
    """
    if model is None:
        raise gr.Error("Model not loaded yet. Please wait for initialization to complete.")

    if not text or not text.strip():
        raise gr.Error("Please enter some text to synthesize.")

    # Determine audio source: YouTube URL or uploaded file
    audio_prompt_path = None
    temp_files = []

    try:
        if yt_url and yt_url.strip():
            # Download from YouTube
            print(f"\nDownloading audio from YouTube: {yt_url}", flush=True)
            try:
                wav_path = download_youtube_audio(yt_url.strip(), str(TEMP_DIR))
                audio_prompt_path = wav_path
                temp_files.append(wav_path)
            except Exception as e:
                raise gr.Error(f"Failed to download YouTube audio: {e}")
        elif audio_upload is not None:
            # Use uploaded audio file
            audio_prompt_path = audio_upload
        else:
            raise gr.Error(
                "Please provide a YouTube URL or upload an audio file as voice reference."
            )

        # Set seed for reproducibility
        if seed_num != 0:
            set_seed(int(seed_num))

        # Generate speech with Chatterbox-Turbo
        print(f"\nGenerating speech: '{text[:80]}...'", flush=True)
        wav = model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            temperature=temperature,
            min_p=0.0,
            top_p=top_p,
            top_k=int(top_k),
            repetition_penalty=repetition_penalty,
            norm_loudness=norm_loudness,
        )

        # Save output to outputs/ for sync back to local machine
        output_filename = f"clone_{uuid.uuid4().hex[:8]}.wav"
        output_path = OUTPUTS_DIR / output_filename
        torchaudio.save(str(output_path), wav, model.sr)
        print(f"Saved output: {output_path}", flush=True)

        return model, (model.sr, wav.squeeze(0).numpy())

    finally:
        # Clean up temporary files
        for f in temp_files:
            try:
                os.remove(f)
            except OSError:
                pass


# ============================================================
# Build Gradio UI
# ============================================================
def build_ui():
    """Build and return the Gradio Blocks interface."""

    with gr.Blocks(
        title="Chatterbox Voice Clone - GPU CLI",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="emerald",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
    ) as demo:
        gr.Markdown(
            "# Chatterbox Voice Clone\n"
            "Clone any voice from YouTube using **Chatterbox-Turbo** TTS. "
            "Paste a YouTube URL or upload audio, enter text, and generate speech.\n\n"
            "*Powered by [GPU CLI](https://gpu-cli.sh) + [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI. "
            "All outputs include [PerTh watermarks](https://github.com/resemble-ai/perth) for responsible AI.*"
        )

        # Model state (loaded once, reused across generations)
        model_state = gr.State(None)

        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                gr.Markdown("### Voice Reference")
                yt_url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    info="Paste a YouTube URL containing the voice you want to clone (first 60s used)",
                )
                audio_upload = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Or Upload / Record Audio",
                    elem_id="audio-upload",
                )

                gr.Markdown("### Text to Speak")
                text = gr.Textbox(
                    value="Hi there! This is a voice clone generated by Chatterbox Turbo. "
                    "Pretty impressive, right? [chuckle] "
                    "You can use paralinguistic tags to add realism to the speech.",
                    label="Text to synthesize",
                    placeholder="Enter the text you want spoken in the cloned voice...",
                    max_lines=5,
                    elem_id="text-input",
                    info="Max ~300 characters. Use tags like [laugh], [cough], [sigh] for expression.",
                )

                # Paralinguistic tag buttons
                gr.Markdown("**Insert expression tag:**")
                with gr.Row(elem_classes=["tag-container"]):
                    for tag in EVENT_TAGS:
                        btn = gr.Button(tag, elem_classes=["tag-btn"])
                        btn.click(
                            fn=None,
                            inputs=[btn, text],
                            outputs=text,
                            js=INSERT_TAG_JS,
                        )

                run_btn = gr.Button("Clone Voice", variant="primary", size="lg")

            # Right column - Output + Settings
            with gr.Column(scale=1):
                gr.Markdown("### Generated Audio")
                audio_output = gr.Audio(label="Output Audio", type="numpy")

                with gr.Accordion("Advanced Settings", open=False):
                    seed_num = gr.Number(
                        value=0,
                        label="Random Seed",
                        info="0 for random, set a number for reproducible results",
                    )
                    temp = gr.Slider(
                        0.05,
                        2.0,
                        step=0.05,
                        value=0.8,
                        label="Temperature",
                        info="Higher = more varied, lower = more consistent",
                    )
                    top_p = gr.Slider(
                        0.0,
                        1.0,
                        step=0.01,
                        value=0.95,
                        label="Top P",
                    )
                    top_k = gr.Slider(
                        0,
                        1000,
                        step=10,
                        value=1000,
                        label="Top K",
                    )
                    repetition_penalty = gr.Slider(
                        1.0,
                        2.0,
                        step=0.05,
                        value=1.2,
                        label="Repetition Penalty",
                    )
                    norm_loudness = gr.Checkbox(
                        value=True,
                        label="Normalize Loudness (-27 LUFS)",
                    )

        # Load model on startup
        demo.load(fn=load_model, inputs=[], outputs=model_state)

        # Generate button click
        run_btn.click(
            fn=generate_voice_clone,
            inputs=[
                model_state,
                text,
                yt_url,
                audio_upload,
                temp,
                seed_num,
                top_p,
                top_k,
                repetition_penalty,
                norm_loudness,
            ],
            outputs=[model_state, audio_output],
        )

    return demo


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Ensure directories exist
    OUTPUTS_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

    print("=" * 60, flush=True)
    print("  Chatterbox Voice Clone - GPU CLI Template", flush=True)
    print("=" * 60, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Outputs: {OUTPUTS_DIR}", flush=True)
    print("", flush=True)

    demo = build_ui()
    demo.queue(
        max_size=20,
        default_concurrency_limit=1,
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
