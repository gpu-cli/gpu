#!/usr/bin/env python3
"""
AI Video Generation with Wan 2.2

Generates high-quality videos from text prompts using the Wan 2.2 14B model.
Requires ~40GB VRAM (A100 80GB or H100 recommended).

Usage:
    gpu run python generate.py --prompt "A cat walking in a garden"
    gpu run python generate.py --prompt "Ocean waves at sunset" --steps 30
"""

import argparse
import os
from pathlib import Path

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video


def main():
    parser = argparse.ArgumentParser(description="Generate videos with Wan 2.2")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text description of the video to generate",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, distorted, watermark",
        help="Negative prompt to avoid unwanted features",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of inference steps (default: 25, higher = better quality)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale (default: 7.5)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=81,
        help="Number of frames to generate (default: 81 = ~5 seconds at 16fps)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Video height (default: 720)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Output video FPS (default: 16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("WARNING: Running on CPU will be extremely slow. Use a GPU pod.")

    # Load pipeline
    print("Loading Wan 2.2 pipeline...")
    pipe = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)

    # Enable memory optimizations
    pipe.enable_model_cpu_offload()

    # Set seed for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        print(f"Using seed: {args.seed}")

    # Generate video
    print(f"Generating video: '{args.prompt}'")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Frames: {args.num_frames} ({args.num_frames / args.fps:.1f} seconds)")
    print(f"  Steps: {args.steps}")

    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        generator=generator,
    )

    # Export to video file
    # Create filename from prompt (sanitized)
    safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in args.prompt)
    safe_prompt = safe_prompt[:50].strip().replace(" ", "_")
    output_path = output_dir / f"{safe_prompt}.mp4"

    print(f"Saving video to: {output_path}")
    export_to_video(output.frames[0], str(output_path), fps=args.fps)

    print(f"Done! Video saved to: {output_path}")
    print("The video will sync to your local machine automatically.")


if __name__ == "__main__":
    main()
