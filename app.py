#@title MonoFace - Advanced Face Swapping & Processing Interface
import os
import sys
import cv2
import numpy as np
import gradio as gr
import subprocess
import time
import shutil
import glob
from typing import List, Optional, Tuple, Dict, Any

from face_analyser import FaceAnalyser, get_many_faces, get_average_face
from Face.modules import FaceSwapper, FaceDetector, FaceLandmarker, FaceMasker
from Face.typing import Face, VisionFrame, Padding
from downloads import is_model_downloaded, download_model

# -----------------------------------------
# 1) System Helpers & Audio/Video FFmpeg
# -----------------------------------------
VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def safe_filename(path: Optional[str]) -> Tuple[str, str]:
    if not path:
        return "output", ".mp4"
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    return name, ext.lower()


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image from path: {path}")
    return img


def get_video_info(video_path: str) -> Tuple[int, float, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 1, 0.0, 0, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return max(total, 1), fps, w, h


def frame_to_sec(frame: int, fps: float) -> float:
    return 0.0 if fps <= 0 else frame / fps


def sec_to_frame(sec: float, fps: float) -> int:
    return int(sec * fps)


def extract_frames_ffmpeg(
    video_path: str,
    output_dir: str,
    trim_mode: str,
    start_val: float,
    end_val: float,
    fps_override: float,
    quality_0_100: int
) -> float:
    """
    Extracts frames using FFmpeg with high performance and quality mapping.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    total_frames, fps, w, h = get_video_info(video_path)

    if trim_mode == "Frames":
        start_sec = frame_to_sec(int(start_val), fps)
        end_sec = frame_to_sec(int(end_val), fps) if end_val > 0 else frame_to_sec(total_frames, fps)
    else:
        start_sec = float(start_val)
        end_sec = float(end_val) if end_val > 0 else frame_to_sec(total_frames, fps)

    duration = max(0.1, end_sec - start_sec)
    out_fps = fps if fps_override <= 0 else fps_override

    # Map quality (0-100) to FFmpeg q:v (2-31)
    ffmpeg_q = int(31 - (quality_0_100 * 0.29))
    ffmpeg_q = max(2, min(31, ffmpeg_q))

    print(f"✂️ Extracting frames: {start_sec:.2f}s -> {end_sec:.2f}s (Duration: {duration:.2f}s, FPS: {out_fps})")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", video_path,
        "-vf", f"fps={out_fps}",
        "-q:v", str(ffmpeg_q),
        os.path.join(output_dir, "frame_%06d.jpg")
    ]

    subprocess.run(cmd, check=True)
    return out_fps


def frames_to_video_ffmpeg(frame_dir: str, output_path: str, fps: float) -> str:
    print("🎬 Stitching video with FFmpeg...")
    input_pattern = os.path.join(frame_dir, "frame_%06d.jpg")
    temp_vid = output_path.replace(".mp4", "_silent.mp4")

    cmd_vid = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        temp_vid
    ]
    subprocess.run(cmd_vid, check=True)
    return temp_vid


# -----------------------------------------
# 2) Pipeline Instantiation Helper
# -----------------------------------------
def create_pipeline(
    swapper_model: str,
    swapper_weight: float,
    pixel_boost: str,
    detector_model: str,
    detector_size_str: str,
    detector_score: float,
    detector_angles: List[int],
    margin_top: int,
    margin_right: int,
    margin_bottom: int,
    margin_left: int,
    landmarker_model: str,
    landmarker_score: float,
    mask_types: List[str],
    mask_blur: float,
    mask_padding_top: int,
    mask_padding_right: int,
    mask_padding_bottom: int,
    mask_padding_left: int,
    occluder_model: str,
    mask_regions: Optional[List[str]] = None,
    mask_areas: Optional[List[str]] = None
) -> Tuple[FaceAnalyser, FaceSwapper]:
    # Parse detector resolution
    det_w, det_h = (int(x) for x in detector_size_str.split('x'))

    analyser = FaceAnalyser(
        detector_model=detector_model,
        detector_score=detector_score,
        detector_size=(det_w, det_h),
        detector_angles=[int(a) for a in detector_angles] if detector_angles else [0],
        landmarker_model=landmarker_model,
        landmarker_score=landmarker_score
    )
    # Set custom detector margins
    analyser.detector.margin = (margin_top, margin_right, margin_bottom, margin_left)

    swapper = FaceSwapper(
        model_name=swapper_model,
        weight=swapper_weight,
        pixel_boost=pixel_boost,
        mask_types=mask_types,
        mask_blur=mask_blur,
        mask_padding=(mask_padding_top, mask_padding_right, mask_padding_bottom, mask_padding_left)
    )
    swapper.masker.occluder_model = occluder_model

    return analyser, swapper


def get_source_face_from_paths(analyser: FaceAnalyser, source_paths: List[str]) -> Face:
    source_faces = []
    for p in source_paths:
        img = read_image(p)
        faces = analyser.get_many_faces([img])
        if faces:
            # Pick largest detected face
            largest_face = sorted(
                faces,
                key=lambda f: (f.bounding_box[2] - f.bounding_box[0]) * (f.bounding_box[3] - f.bounding_box[1]),
                reverse=True
            )[0]
            source_faces.append(largest_face)

    if not source_faces:
        raise gr.Error("No face detected in any of the uploaded source images!")

    avg_face = analyser.get_average_face(source_faces)
    return avg_face or source_faces[0]


# -----------------------------------------
# 3) Interactive Preview Function
# -----------------------------------------
def preview_swap_frame(
    source_files: List[str],
    target_file: str,
    frame_index: int,
    swapper_model: str,
    swapper_weight: float,
    pixel_boost: str,
    detector_model: str,
    detector_size: str,
    detector_score: float,
    detector_angles: List[int],
    margin_top: int,
    margin_right: int,
    margin_bottom: int,
    margin_left: int,
    landmarker_model: str,
    landmarker_score: float,
    mask_types: List[str],
    mask_blur: float,
    mask_padding_top: int,
    mask_padding_right: int,
    mask_padding_bottom: int,
    mask_padding_left: int,
    occluder_model: str,
    mask_regions: List[str]
) -> np.ndarray:
    if not source_files or not target_file:
        raise gr.Error("Please upload both source image(s) and a target file first.")

    _, ext = safe_filename(target_file)

    analyser, swapper = create_pipeline(
        swapper_model=swapper_model,
        swapper_weight=swapper_weight,
        pixel_boost=pixel_boost,
        detector_model=detector_model,
        detector_size_str=detector_size,
        detector_score=detector_score,
        detector_angles=detector_angles,
        margin_top=margin_top,
        margin_right=margin_right,
        margin_bottom=margin_bottom,
        margin_left=margin_left,
        landmarker_model=landmarker_model,
        landmarker_score=landmarker_score,
        mask_types=mask_types,
        mask_blur=mask_blur,
        mask_padding_top=mask_padding_top,
        mask_padding_right=mask_padding_right,
        mask_padding_bottom=mask_padding_bottom,
        mask_padding_left=mask_padding_left,
        occluder_model=occluder_model,
        mask_regions=mask_regions
    )

    source_face = get_source_face_from_paths(analyser, source_files)

    if ext in IMAGE_EXTS:
        target_frame = read_image(target_file)
    else:
        cap = cv2.VideoCapture(target_file)
        if not cap.isOpened():
            raise gr.Error("Failed to open target video file.")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = int(np.clip(frame_index, 0, max(0, total_frames - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, target_frame = cap.read()
        cap.release()
        if not ok or target_frame is None:
            raise gr.Error("Failed to read video frame at requested index.")

    target_faces = analyser.get_many_faces([target_frame])
    if not target_faces:
        # Return original frame with a notification
        return cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)

    result_frame = target_frame.copy()
    for target_face in target_faces:
        result_frame = swapper.swap_face(
            source_face=source_face,
            target_face=target_face,
            target_vision_frame=result_frame,
            pixel_boost=pixel_boost
        )

    return cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)


# -----------------------------------------
# 4) Main Batch Execution Function
# -----------------------------------------
def run_batch_swap(
    source_files: List[str],
    target_file: str,
    trim_mode: str,
    start_sec: float,
    end_sec: float,
    start_frame: int,
    end_frame: int,
    fps_override: float,
    frame_quality: int,
    swapper_model: str,
    swapper_weight: float,
    pixel_boost: str,
    detector_model: str,
    detector_size: str,
    detector_score: float,
    detector_angles: List[int],
    margin_top: int,
    margin_right: int,
    margin_bottom: int,
    margin_left: int,
    landmarker_model: str,
    landmarker_score: float,
    mask_types: List[str],
    mask_blur: float,
    mask_padding_top: int,
    mask_padding_right: int,
    mask_padding_bottom: int,
    mask_padding_left: int,
    occluder_model: str,
    mask_regions: List[str],
    progress=gr.Progress()
) -> Tuple[Optional[str], Optional[str], str]:
    if not source_files or not target_file:
        raise gr.Error("Missing source face images or target file.")

    analyser, swapper = create_pipeline(
        swapper_model=swapper_model,
        swapper_weight=swapper_weight,
        pixel_boost=pixel_boost,
        detector_model=detector_model,
        detector_size_str=detector_size,
        detector_score=detector_score,
        detector_angles=detector_angles,
        margin_top=margin_top,
        margin_right=margin_right,
        margin_bottom=margin_bottom,
        margin_left=margin_left,
        landmarker_model=landmarker_model,
        landmarker_score=landmarker_score,
        mask_types=mask_types,
        mask_blur=mask_blur,
        mask_padding_top=mask_padding_top,
        mask_padding_right=mask_padding_right,
        mask_padding_bottom=mask_padding_bottom,
        mask_padding_left=mask_padding_left,
        occluder_model=occluder_model,
        mask_regions=mask_regions
    )

    progress(0.05, desc="Extracting source identity...")
    source_face = get_source_face_from_paths(analyser, source_files)

    target_name, ext = safe_filename(target_file)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------
    # Image Target Mode
    # ------------------
    if ext in IMAGE_EXTS:
        progress(0.4, desc="Swapping faces in image...")
        img = read_image(target_file)
        target_faces = analyser.get_many_faces([img])

        if not target_faces:
            raise gr.Error("No faces detected in target image with current detector settings.")

        res = img.copy()
        for target_face in target_faces:
            res = swapper.swap_face(source_face, target_face, res, pixel_boost=pixel_boost)

        out_img_path = os.path.join(output_dir, f"swapped_{target_name}.png")
        cv2.imwrite(out_img_path, res)
        progress(1.0, desc="Completed!")
        return out_img_path, None, f"✅ Image swap completed ({len(target_faces)} face(s) swapped)."

    # ------------------
    # Video Target Mode
    # ------------------
    temp_dir = os.path.join(output_dir, "temp_frames")
    swapped_dir = os.path.join(output_dir, "temp_swapped_frames")

    progress(0.1, desc="Extracting video frames via FFmpeg...")
    try:
        real_fps = extract_frames_ffmpeg(
            video_path=target_file,
            output_dir=temp_dir,
            trim_mode=trim_mode,
            start_val=start_frame if trim_mode == "Frames" else start_sec,
            end_val=end_frame if trim_mode == "Frames" else end_sec,
            fps_override=fps_override,
            quality_0_100=frame_quality
        )
    except Exception as e:
        raise gr.Error(f"Video extraction error: {e}")

    if os.path.exists(swapped_dir):
        shutil.rmtree(swapped_dir)
    os.makedirs(swapped_dir, exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(temp_dir, "*.jpg")))
    total_frames = len(frame_files)

    if total_frames == 0:
        raise gr.Error("No frames extracted! Please check your trim settings.")

    print(f"🚀 Processing {total_frames} frames using {swapper_model} (Pixel Boost: {pixel_boost})...")
    start_time = time.time()

    for idx, frame_path in enumerate(frame_files):
        frame = cv2.imread(frame_path)
        target_faces = analyser.get_many_faces([frame])

        if target_faces:
            for target_face in target_faces:
                frame = swapper.swap_face(source_face, target_face, frame, pixel_boost=pixel_boost)

        out_name = os.path.basename(frame_path)
        cv2.imwrite(os.path.join(swapped_dir, out_name), frame)

        if idx % 5 == 0 or idx == total_frames - 1:
            elapsed = time.time() - start_time
            current_fps = (idx + 1) / max(elapsed, 0.001)
            ratio = (idx + 1) / total_frames
            progress(0.15 + (0.75 * ratio), desc=f"Swapping: {idx+1}/{total_frames} [{current_fps:.1f} FPS]")
            print(f"⚡ Frame {idx+1}/{total_frames} | Speed: {current_fps:.2f} FPS")

    # Stitch video back
    progress(0.92, desc="Stitching frames with FFmpeg...")
    final_video = os.path.join(output_dir, f"swapped_{target_name}.mp4")
    temp_vid = frames_to_video_ffmpeg(swapped_dir, final_video, real_fps)

    # Sync Audio
    progress(0.97, desc="Synchronizing audio track...")
    total_orig_frames, orig_fps, _, _ = get_video_info(target_file)
    s_sec = frame_to_sec(start_frame, orig_fps) if trim_mode == "Frames" else start_sec

    cmd_audio = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", temp_vid,
        "-ss", str(s_sec), "-i", target_file,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        final_video
    ]
    subprocess.run(cmd_audio)

    # Cleanup temp frames
    try:
        shutil.rmtree(temp_dir)
        shutil.rmtree(swapped_dir)
        if os.path.exists(temp_vid):
            os.remove(temp_vid)
    except Exception:
        pass

    total_time = time.time() - start_time
    avg_fps = total_frames / max(total_time, 0.001)
    status_msg = f"✅ Done! Processed {total_frames} frames in {total_time:.1f}s (Average: {avg_fps:.2f} FPS)."
    print(status_msg)

    return None, final_video, status_msg


# -----------------------------------------
# 5) UI Dynamic Visibility & Info Handlers
# -----------------------------------------
def update_ui_for_target(target_file: Optional[str]):
    if not target_file:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(value="0.0"),
            gr.update(value="0.0"),
            gr.update(visible=False),
            gr.update(maximum=1, value=0)
        )

    name, ext = safe_filename(target_file)
    if ext in VIDEO_EXTS:
        total_frames, fps, w, h = get_video_info(target_file)
        total_sec = frame_to_sec(total_frames, fps)
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=str(round(fps, 3))),
            gr.update(value=str(round(total_sec, 2))),
            gr.update(visible=True),
            gr.update(maximum=max(total_frames - 1, 1), value=0)
        )

    if ext in IMAGE_EXTS:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="0.0"),
            gr.update(value="0.0"),
            gr.update(visible=False),
            gr.update(maximum=1, value=0)
        )

    return (
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(value="0.0"),
        gr.update(value="0.0"),
        gr.update(visible=False),
        gr.update(maximum=1, value=0)
    )


# -----------------------------------------
# 6) Gradio Interface Layout
# -----------------------------------------
custom_css = """
.gradio-container { max-width: 1100px !important; margin: auto; }
.header-badge { font-size: 0.85rem; color: #10b981; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
.section-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 0.5rem; }
footer { display: none !important; }
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate"
)

with gr.Blocks(title="MonoFace Pro", css=custom_css, theme=theme) as demo:
    gr.Markdown(
        """
        # ⚡ MonoFace Studio: Next-Gen Face Swapping Pipeline
        <span class='header-badge'>High Performance • Multi-Model Architecture • Precision Masking • Pixel Boost</span>
        """
    )

    with gr.Row():
        src_files = gr.File(
            label="Source Face Image(s)",
            file_count="multiple",
            type="filepath",
            file_types=["image"]
        )
        tgt_file = gr.File(
            label="Target Image or Video",
            file_count="single",
            type="filepath",
            file_types=["image", "video"]
        )

    # Info & Video Statistics
    with gr.Row():
        orig_fps = gr.Textbox(label="Original Video FPS", value="0.0", interactive=False)
        total_duration = gr.Textbox(label="Total Duration (seconds)", value="0.0", interactive=False)

    # -------------------------------------------------------------
    # Model & Processor Controls
    # -------------------------------------------------------------
    with gr.Tab("⚙️ Swapper & Model Options"):
        with gr.Row():
            swapper_model = gr.Dropdown(
                choices=[
                    "inswapper_128",
                    "inswapper_128_fp16",
                    "hyperswap_1a_256",
                    "hyperswap_1b_256",
                    "hyperswap_1c_256",
                    "simswap_256",
                    "simswap_512_unofficial"
                ],
                value="inswapper_128",
                label="Face Swapper Model"
            )
            pixel_boost = gr.Dropdown(
                choices=["none", "128x128", "256x256", "384x384", "512x512", "768x768", "1024x1024"],
                value="none",
                label="Pixel Boost (Super Resolution Tiling)"
            )
            swapper_weight = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="Swapper Identity Balance (0=Target, 1=Source)"
            )

    # -------------------------------------------------------------
    # Detection & Landmarking Controls
    # -------------------------------------------------------------
    with gr.Tab("🎯 Detection & Landmarker"):
        with gr.Row():
            detector_model = gr.Dropdown(
                choices=["yolo_face", "scrfd", "retinaface", "yunet"],
                value="yolo_face",
                label="Face Detector Model"
            )
            detector_size = gr.Dropdown(
                choices=["320x320", "480x480", "512x512", "640x640", "768x768", "960x960", "1024x1024"],
                value="640x640",
                label="Detector Input Size"
            )
            detector_score = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="Detector Score Threshold"
            )

        with gr.Row():
            detector_angles = gr.CheckboxGroup(
                choices=[0, 90, 180, 270],
                value=[0],
                label="Detector Angles (Rotational Search)"
            )
            landmarker_model = gr.Dropdown(
                choices=["2dfan4", "peppa_wutz"],
                value="2dfan4",
                label="Face Landmarker Model"
            )
            landmarker_score = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="Landmarker Score Threshold"
            )

        # Margin expansion
        with gr.Accordion("📐 Face Detector Margins (% Expansion)", open=False):
            with gr.Row():
                margin_top = gr.Slider(0, 100, value=0, step=1, label="Margin Top (%)")
                margin_right = gr.Slider(0, 100, value=0, step=1, label="Margin Right (%)")
                margin_bottom = gr.Slider(0, 100, value=0, step=1, label="Margin Bottom (%)")
                margin_left = gr.Slider(0, 100, value=0, step=1, label="Margin Left (%)")

    # -------------------------------------------------------------
    # Masking & Blending Controls
    # -------------------------------------------------------------
    with gr.Tab("🎭 Face Masking & Occlusion"):
        with gr.Row():
            mask_types = gr.CheckboxGroup(
                choices=["box", "occlusion", "region", "area"],
                value=["box", "occlusion"],
                label="Active Mask Types"
            )
            occluder_model = gr.Dropdown(
                choices=["face_occluder", "xseg_1", "xseg_2", "xseg_3"],
                value="face_occluder",
                label="Occlusion Segmentation Model"
            )
            mask_blur = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.3,
                step=0.05,
                label="Mask Feather / Box Blur"
            )

        with gr.Row():
            mask_regions = gr.CheckboxGroup(
                choices=["skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye", "nose", "mouth", "upper-lip", "lower-lip"],
                value=["skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye", "nose", "mouth", "upper-lip", "lower-lip"],
                label="Semantic Mask Regions (When 'region' mask is enabled)"
            )

        with gr.Accordion("✂️ Face Mask Padding (% Cut)", open=False):
            with gr.Row():
                mask_padding_top = gr.Slider(0, 100, value=0, step=1, label="Mask Padding Top (%)")
                mask_padding_right = gr.Slider(0, 100, value=0, step=1, label="Mask Padding Right (%)")
                mask_padding_bottom = gr.Slider(0, 100, value=0, step=1, label="Mask Padding Bottom (%)")
                mask_padding_left = gr.Slider(0, 100, value=0, step=1, label="Mask Padding Left (%)")

    # -------------------------------------------------------------
    # Video Trimming & Output Controls
    # -------------------------------------------------------------
    with gr.Tab("✂️ Video Trimming & Encoding"):
        trim_mode = gr.Radio(["Seconds", "Frames"], value="Seconds", label="Trim Mode")
        with gr.Row():
            start_sec = gr.Number(label="Start Time (seconds)", value=0.0)
            end_sec = gr.Number(label="End Time (seconds, 0=Full)", value=0.0)
        with gr.Row():
            start_frame = gr.Number(label="Start Frame", value=0)
            end_frame = gr.Number(label="End Frame (0=Full)", value=0)
        with gr.Row():
            frame_quality = gr.Slider(0, 100, value=80, step=1, label="Output Frame Quality (0-100)")
            fps_override = gr.Slider(0, 120, value=0, step=1, label="FPS Override (0=Same as Target)")

    # -------------------------------------------------------------
    # Interactive Frame Preview Section
    # -------------------------------------------------------------
    with gr.Accordion("🎞️ Interactive Frame Preview", open=True, visible=False) as preview_box:
        with gr.Row():
            frame_slider = gr.Slider(minimum=0, maximum=1, step=1, value=0, label="Preview Frame Index")
            preview_btn = gr.Button("👁️ Generate Frame Preview", variant="secondary")
        preview_output = gr.Image(label="Live Preview Result", type="numpy")

    # Run Button & Outputs
    run_btn = gr.Button("🚀 RUN MONOFACE BATCH SWAP", variant="primary", size="lg")
    status_box = gr.Textbox(label="Status & Execution Log", interactive=False)

    with gr.Row():
        out_image = gr.Image(label="Swapped Output Image")
        out_video = gr.Video(label="Swapped Output Video")

    # -------------------------------------------------------------
    # Events & Callbacks
    # -------------------------------------------------------------
    tgt_file.change(
        fn=update_ui_for_target,
        inputs=[tgt_file],
        outputs=[out_image, out_video, orig_fps, total_duration, preview_box, frame_slider]
    )

    all_config_inputs = [
        swapper_model,
        swapper_weight,
        pixel_boost,
        detector_model,
        detector_size,
        detector_score,
        detector_angles,
        margin_top,
        margin_right,
        margin_bottom,
        margin_left,
        landmarker_model,
        landmarker_score,
        mask_types,
        mask_blur,
        mask_padding_top,
        mask_padding_right,
        mask_padding_bottom,
        mask_padding_left,
        occluder_model,
        mask_regions
    ]

    preview_btn.click(
        fn=preview_swap_frame,
        inputs=[src_files, tgt_file, frame_slider] + all_config_inputs,
        outputs=[preview_output]
    )

    run_btn.click(
        fn=run_batch_swap,
        inputs=[
            src_files,
            tgt_file,
            trim_mode,
            start_sec,
            end_sec,
            start_frame,
            end_frame,
            fps_override,
            frame_quality
        ] + all_config_inputs,
        outputs=[out_image, out_video, status_box]
    )

if __name__ == "__main__":
    demo.launch(share=True, inbrowser=True)