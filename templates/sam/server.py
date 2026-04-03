#!/usr/bin/env python3
"""SAM 3 Video Segmentation - Gradio Web UI

Segment and track anything in images and video using text prompts.
Type "dog", "person in red shirt", or any visual concept and SAM 3
finds, segments, and tracks objects across frames.

Built on:
- SAM 3.1 (848M params) by Meta AI Research
- Gradio for the web interface
- OpenCV for video I/O and visualization
"""

import os
import uuid
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
SCRIPT_DIR = Path(__file__).parent.absolute()
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

# Maximum video duration in seconds (keeps processing time reasonable)
MAX_VIDEO_SECONDS = 60

# Color palette for object masks (distinct, high-contrast colors)
COLORS = [
    (0, 200, 83),  # green
    (0, 150, 255),  # blue
    (255, 80, 80),  # red
    (255, 200, 0),  # yellow
    (180, 80, 255),  # purple
    (0, 220, 220),  # cyan
    (255, 128, 0),  # orange
    (255, 100, 200),  # pink
    (128, 255, 128),  # light green
    (100, 180, 255),  # light blue
]

# ============================================================
# Custom CSS
# ============================================================
CUSTOM_CSS = """
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


# ============================================================
# Model Loading
# ============================================================
def load_image_model():
    """Load SAM 3 image segmentation model."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print(f"Loading SAM 3 image model on {DEVICE} ({DTYPE})...", flush=True)
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("Image model loaded.", flush=True)
    return model, processor


def load_video_model():
    """Load SAM 3 video predictor."""
    from sam3.model_builder import build_sam3_video_predictor

    print(f"Loading SAM 3 video predictor on {DEVICE}...", flush=True)
    predictor = build_sam3_video_predictor()
    print("Video predictor loaded.", flush=True)
    return predictor


# ============================================================
# Visualization
# ============================================================
def overlay_masks(image, masks, boxes=None, scores=None, object_ids=None, alpha=0.45):
    """Draw colored segmentation masks and bounding boxes on an image.

    Args:
        image: numpy array (H, W, 3) in BGR or RGB
        masks: numpy array of boolean masks (N, H, W)
        boxes: optional numpy array of bounding boxes (N, 4) as [x1, y1, x2, y2]
        scores: optional numpy array of confidence scores (N,)
        object_ids: optional numpy array of tracking IDs (N,)
        alpha: mask transparency (0=invisible, 1=opaque)

    Returns:
        Annotated image as numpy array
    """
    result = image.copy()
    h, w = result.shape[:2]

    if masks is None or len(masks) == 0:
        return result

    for i, mask in enumerate(masks):
        color = np.array(COLORS[i % len(COLORS)], dtype=np.uint8)

        # Resize mask if needed
        if mask.shape[-2:] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h)) > 0

        # Apply colored mask overlay
        colored = np.zeros_like(result)
        colored[mask] = color
        result = cv2.addWeighted(result, 1.0, colored, alpha, 0)

        # Draw mask contour for crisp edges
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color.tolist(), 2)

        # Draw bounding box
        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(result, (x1, y1), (x2, y2), color.tolist(), 2)

            # Build label text
            parts = []
            if object_ids is not None and i < len(object_ids):
                parts.append(f"ID {int(object_ids[i])}")
            if scores is not None and i < len(scores):
                parts.append(f"{scores[i]:.2f}")
            label = " | ".join(parts)

            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    result,
                    (x1, max(y1 - th - 8, 0)),
                    (x1 + tw + 4, max(y1, th + 8)),
                    color.tolist(),
                    -1,
                )
                cv2.putText(
                    result,
                    label,
                    (x1 + 2, max(y1 - 4, th + 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    return result


def normalize_video_output(output, frame_shape):
    """Normalize SAM 3 video outputs to the overlay format used by this app."""
    h, w = frame_shape[:2]

    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")
    object_ids = output.get("object_ids")

    if masks is None and "out_binary_masks" in output:
        masks = output.get("out_binary_masks")
        scores = output.get("out_probs")
        object_ids = output.get("out_obj_ids")

        boxes_xywh = output.get("out_boxes_xywh")
        if boxes_xywh is not None:
            if torch.is_tensor(boxes_xywh):
                boxes_xywh = boxes_xywh.detach().cpu().numpy()
            else:
                boxes_xywh = np.asarray(boxes_xywh)

            if boxes_xywh.size > 0:
                boxes = np.zeros((len(boxes_xywh), 4), dtype=np.float32)
                boxes[:, 0] = boxes_xywh[:, 0] * w
                boxes[:, 1] = boxes_xywh[:, 1] * h
                boxes[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2]) * w
                boxes[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3]) * h

    return masks, boxes, scores, object_ids


# ============================================================
# Image Segmentation
# ============================================================
def segment_image(image, text_prompt, threshold, state):
    """Segment objects in a single image using a text prompt.

    Args:
        image: input image (numpy array from Gradio)
        text_prompt: text description of objects to find
        threshold: confidence threshold for filtering detections
        state: tuple of (model, processor) or None

    Returns:
        Tuple of (annotated_image, state)
    """
    if image is None:
        raise gr.Error("Please upload an image.")
    if not text_prompt or not text_prompt.strip():
        raise gr.Error("Please enter a text prompt describing what to segment.")

    # Load model on first use
    if state is None:
        state = load_image_model()

    _, processor = state

    from PIL import Image as PILImage

    # Convert numpy to PIL
    if isinstance(image, np.ndarray):
        pil_image = PILImage.fromarray(image)
    else:
        pil_image = image

    print(f"Segmenting image with prompt: '{text_prompt}'", flush=True)

    # Run inference
    inference_state = processor.set_image(pil_image)
    output = processor.set_text_prompt(
        state=inference_state,
        prompt=text_prompt.strip(),
    )

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    # Convert tensors to numpy
    if torch.is_tensor(masks):
        masks = masks.cpu().numpy()
    if torch.is_tensor(boxes):
        boxes = boxes.cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()

    # Filter by confidence threshold
    if scores is not None and len(scores) > 0:
        keep = scores >= threshold
        masks = masks[keep]
        boxes = boxes[keep] if boxes is not None else None
        scores = scores[keep]

    n_detections = len(masks) if masks is not None else 0
    print(f"Found {n_detections} objects above threshold {threshold}", flush=True)

    # Ensure image is numpy for overlay
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Overlay masks on image
    result = overlay_masks(image, masks, boxes, scores)

    # Save output
    output_filename = f"seg_{uuid.uuid4().hex[:8]}.png"
    output_path = OUTPUTS_DIR / output_filename
    cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}", flush=True)

    return result, state


# ============================================================
# Video Segmentation
# ============================================================
def segment_video(video_path, text_prompt, video_state, progress=gr.Progress()):
    """Segment and track objects through a video using a text prompt.

    Args:
        video_path: path to uploaded video file
        text_prompt: text description of objects to track
        video_state: video predictor model or None
        progress: Gradio progress tracker

    Returns:
        Tuple of (output_video_path, download_path, video_state)
    """
    if video_path is None:
        raise gr.Error("Please upload a video.")
    if not text_prompt or not text_prompt.strip():
        raise gr.Error("Please enter a text prompt describing what to track.")

    # Load model on first use
    if video_state is None:
        video_state = load_video_model()

    predictor = video_state

    print(f"Processing video with prompt: '{text_prompt}'", flush=True)
    progress(0.0, desc="Reading video...")

    # Read video frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Enforce max duration
    max_frames = int(MAX_VIDEO_SECONDS * fps)
    if total_frames > max_frames:
        print(
            f"Video has {total_frames} frames ({total_frames / fps:.1f}s). "
            f"Trimming to {max_frames} frames ({MAX_VIDEO_SECONDS}s).",
            flush=True,
        )
        total_frames = max_frames

    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise gr.Error("Could not read any frames from the video.")

    print(
        f"Read {len(frames)} frames ({len(frames) / fps:.1f}s at {fps:.0f} FPS)",
        flush=True,
    )
    progress(0.1, desc="Starting segmentation session...")

    # Save frames to temp directory for video predictor
    import tempfile
    import shutil

    frame_dir = tempfile.mkdtemp(prefix="sam3_frames_")
    try:
        for i, frame in enumerate(frames):
            frame_path = os.path.join(frame_dir, f"{i:06d}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Start video session
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=frame_dir,
            )
        )
        session_id = response["session_id"]

        progress(0.2, desc="Adding text prompt and propagating...")

        # Add text prompt on first frame
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=text_prompt.strip(),
            )
        )

        # Collect per-frame outputs.
        # SAM 3 video propagation uses a stream of per-frame responses.
        outputs_per_frame = {}
        if "outputs" in response:
            outputs_per_frame[0] = response["outputs"]

        for stream_response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            frame_index = stream_response.get("frame_index")
            frame_output = stream_response.get("outputs")
            if frame_index is not None and frame_output is not None:
                outputs_per_frame[int(frame_index)] = frame_output

        progress(0.5, desc="Rendering annotated video...")

        # Write output video
        output_filename = f"tracked_{uuid.uuid4().hex[:8]}.mp4"
        output_path = str(OUTPUTS_DIR / output_filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for idx, frame in enumerate(frames):
            out = outputs_per_frame.get(idx)
            if out is not None:
                masks, boxes, scores, object_ids = normalize_video_output(
                    out, frame.shape
                )

                # Convert tensors to numpy
                if masks is not None and torch.is_tensor(masks):
                    masks = masks.cpu().numpy()
                if boxes is not None and torch.is_tensor(boxes):
                    boxes = boxes.cpu().numpy()
                if scores is not None and torch.is_tensor(scores):
                    scores = scores.cpu().numpy()
                if object_ids is not None and torch.is_tensor(object_ids):
                    object_ids = object_ids.cpu().numpy()

                frame = overlay_masks(frame, masks, boxes, scores, object_ids)

            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if idx % max(1, len(frames) // 20) == 0:
                pct = 0.5 + 0.5 * (idx / len(frames))
                progress(pct, desc=f"Rendering frame {idx + 1}/{len(frames)}")

        writer.release()

    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    progress(1.0, desc="Done!")
    n_annotated = len(outputs_per_frame)
    print(
        f"Saved annotated video: {output_path} ({n_annotated}/{len(frames)} frames annotated)",
        flush=True,
    )

    return output_path, output_path, video_state


# ============================================================
# Build Gradio UI
# ============================================================
def build_ui():
    """Build and return the Gradio Blocks interface."""

    with gr.Blocks(
        title="SAM 3 Video Segmentation - GPU CLI",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="emerald",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
    ) as demo:
        gr.Markdown(
            "# SAM 3: Segment Anything\n"
            "Detect, segment, and track objects in **images and video** using text prompts. "
            'Type any visual concept like "dog", "red car", or "person holding umbrella".\n\n'
            "*Powered by [GPU CLI](https://gpu-cli.sh) + [SAM 3.1](https://github.com/facebookresearch/sam3) by Meta AI Research.*"
        )

        # Shared model state (loaded once per tab, reused across runs)
        image_state = gr.State(None)
        video_state = gr.State(None)

        with gr.Tabs():
            # ===== Video Segmentation Tab =====
            with gr.TabItem("Video Segmentation"):
                gr.Markdown(
                    "Upload a video and describe the objects you want to track. "
                    "SAM 3 will segment and track them across all frames."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Upload Video (MP4)")
                        video_prompt = gr.Textbox(
                            label="Text Prompt",
                            placeholder="e.g. person, dog, red car, tennis ball",
                            value="person",
                            info=f"Describe what to track. Short, specific prompts work best. Max {MAX_VIDEO_SECONDS}s video.",
                        )
                        video_btn = gr.Button(
                            "Segment & Track", variant="primary", size="lg"
                        )

                    with gr.Column(scale=1):
                        video_output = gr.Video(label="Annotated Output")
                        video_download = gr.File(label="Download Video")

                video_btn.click(
                    fn=segment_video,
                    inputs=[video_input, video_prompt, video_state],
                    outputs=[video_output, video_download, video_state],
                )

            # ===== Image Segmentation Tab =====
            with gr.TabItem("Image Segmentation"):
                gr.Markdown(
                    "Upload an image and describe the objects you want to find. "
                    "SAM 3 will detect and segment every matching instance."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload Image",
                            type="numpy",
                            sources=["upload"],
                        )
                        image_prompt = gr.Textbox(
                            label="Text Prompt",
                            placeholder="e.g. cat, blue bicycle, traffic light",
                            value="person",
                            info="Describe the objects to segment. Try specific descriptions for precise results.",
                        )
                        image_threshold = gr.Slider(
                            0.1,
                            0.9,
                            step=0.05,
                            value=0.4,
                            label="Confidence Threshold",
                            info="Higher = fewer but more confident detections",
                        )
                        image_btn = gr.Button(
                            "Segment Image", variant="primary", size="lg"
                        )

                    with gr.Column(scale=1):
                        image_output = gr.Image(label="Segmented Output", type="numpy")

                image_btn.click(
                    fn=segment_image,
                    inputs=[image_input, image_prompt, image_threshold, image_state],
                    outputs=[image_output, image_state],
                )

        gr.Markdown(
            "---\n"
            "**Tips:** Short prompts work best (e.g. 'dog' not 'a cute fluffy dog'). "
            "Use specific attributes to distinguish objects (e.g. 'person in red shirt'). "
            "All outputs are saved to `outputs/` and synced back to your local machine."
        )

    return demo


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print("=" * 60, flush=True)
    print("  SAM 3 Video Segmentation - GPU CLI Template", flush=True)
    print("=" * 60, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  Dtype:  {DTYPE}", flush=True)
    print(f"  Outputs: {OUTPUTS_DIR}", flush=True)
    print("", flush=True)

    demo = build_ui()
    demo.queue(
        max_size=10,
        default_concurrency_limit=1,
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
