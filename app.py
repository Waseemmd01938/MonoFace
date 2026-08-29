#@title MonoFace - Advanced Face Swapping & Processing Interface
import glob
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

from Face.modules import FaceSwapper, free_memory
from Face.typing import Face
from face_analyser import FaceAnalyser, clear_face_cache, preload_face_analyser

# Cached pipeline instance to avoid repeated object recreation
_CACHED_PIPELINE: Dict[str, Any] = {}

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
    mask_areas: Optional[List[str]] = None,
    mask_regions: Optional[List[str]] = None
) -> Tuple[FaceAnalyser, FaceSwapper]:
    global _CACHED_PIPELINE
    det_w, det_h = (int(x) for x in detector_size_str.split('x'))
    angles_tuple = tuple(int(a) for a in detector_angles) if detector_angles else (0,)

    analyser_key = (detector_model, landmarker_model, det_w, det_h, angles_tuple)
    if _CACHED_PIPELINE.get('analyser_key') == analyser_key:
        analyser: FaceAnalyser = _CACHED_PIPELINE['analyser']
        analyser.detector_score = detector_score
        analyser.detector.score_threshold = detector_score
        analyser.landmarker_score = landmarker_score
        analyser.detector.margin = (margin_top, margin_right, margin_bottom, margin_left)
    else:
        analyser = FaceAnalyser(
            detector_model=detector_model,
            detector_score=detector_score,
            detector_size=(det_w, det_h),
            detector_angles=list(angles_tuple),
            landmarker_model=landmarker_model,
            landmarker_score=landmarker_score
        )
        analyser.detector.margin = (margin_top, margin_right, margin_bottom, margin_left)
        _CACHED_PIPELINE['analyser_key'] = analyser_key
        _CACHED_PIPELINE['analyser'] = analyser

    swapper_key = (swapper_model, pixel_boost)
    if _CACHED_PIPELINE.get('swapper_key') == swapper_key:
        swapper: FaceSwapper = _CACHED_PIPELINE['swapper']
        swapper.weight = swapper_weight
        swapper.mask_types = mask_types
        swapper.mask_blur = mask_blur
        swapper.mask_padding = (mask_padding_top, mask_padding_right, mask_padding_bottom, mask_padding_left)
        swapper.mask_areas = mask_areas
        swapper.mask_regions = mask_regions
    else:
        swapper = FaceSwapper(
            model_name=swapper_model,
            weight=swapper_weight,
            pixel_boost=pixel_boost,
            mask_types=mask_types,
            mask_blur=mask_blur,
            mask_padding=(mask_padding_top, mask_padding_right, mask_padding_bottom, mask_padding_left),
            mask_areas=mask_areas,
            mask_regions=mask_regions
        )
        _CACHED_PIPELINE['swapper_key'] = swapper_key
        _CACHED_PIPELINE['swapper'] = swapper

    swapper.masker.occluder_model = occluder_model

    return analyser, swapper


def get_source_face_from_paths(analyser: FaceAnalyser, source_paths: List[str]) -> Face:
    source_faces = []
    for p in source_paths:
        img = read_image(p)
        faces = analyser.get_many_faces([img])
        if faces:
            # Pick largest detected face
            largest_face = max(
                faces,
                key=lambda f: (f.bounding_box[2] - f.bounding_box[0]) * (f.bounding_box[3] - f.bounding_box[1])
            )
            source_faces.append(largest_face)

    if not source_faces:
        raise gr.Error("No face detected in any of the uploaded source images!")

    avg_face = analyser.get_average_face(source_faces)
    return avg_face or source_faces[0]


def crop_face_avatar(vision_frame: np.ndarray, face: Face, size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    """Crops a face with balanced margin padding for gallery display."""
    start_x, start_y, end_x, end_y = map(int, face.bounding_box)
    padding_x = int((end_x - start_x) * 0.25)
    padding_y = int((end_y - start_y) * 0.25)
    h, w = vision_frame.shape[:2]
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = min(w, end_x + padding_x)
    end_y = min(h, end_y + padding_y)
    if end_x <= start_x or end_y <= start_y:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    crop = vision_frame[start_y:end_y, start_x:end_x]
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def update_reference_face_gallery(
    target_file: Optional[str],
    frame_index: int,
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
    face_selector_order: str,
    face_selector_position: int
) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    """Extracts all detected faces from the selected target / preview frame for visual inspection and selection."""
    if not target_file:
        return [], "Upload a target image or video to inspect detected faces."

    try:
        _, ext = safe_filename(target_file)
        if ext in IMAGE_EXTS:
            target_frame = read_image(target_file)
        else:
            cap = cv2.VideoCapture(target_file)
            if not cap.isOpened():
                return [], "Failed to open target video."
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = int(np.clip(frame_index, 0, max(0, total_frames - 1)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, target_frame = cap.read()
            cap.release()
            if not ok or target_frame is None:
                return [], "Failed to read target frame."

        det_w, det_h = (int(x) for x in detector_size_str.split('x'))
        analyser = FaceAnalyser(
            detector_model=detector_model,
            detector_score=detector_score,
            detector_size=(det_w, det_h),
            detector_angles=[int(a) for a in detector_angles] if detector_angles else [0],
            landmarker_model=landmarker_model,
            landmarker_score=landmarker_score
        )
        analyser.detector.margin = (margin_top, margin_right, margin_bottom, margin_left)

        faces = analyser.get_many_faces([target_frame], extract_embedding=True)
        if not faces:
            return [], "⚠️ No faces detected in the current target frame with active detector settings."

        sorted_faces = analyser.sort_faces(faces, face_selector_order)
        gallery_items = []
        for idx, f in enumerate(sorted_faces):
            avatar = crop_face_avatar(target_frame, f)
            score = f.score_set.get('detector', 0.0) if isinstance(f.score_set, dict) else 0.0
            caption = f"Face #{idx} (Score: {score:.2f})"
            gallery_items.append((avatar, caption))

        sel_pos = int(face_selector_position) if isinstance(face_selector_position, (int, float, str)) else 0
        sel_idx = min(max(0, sel_pos), len(sorted_faces) - 1)
        status = f"✅ Detected {len(sorted_faces)} face(s). Currently selected: Face #{sel_idx}."

        return gallery_items, status
    except Exception as e:
        return [], f"⚠️ Face detection error: {str(e)}"


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
    face_selector_mode: str,
    face_selector_order: str,
    face_selector_position: int,
    reference_face_distance: float,
    mask_types: List[str],
    mask_blur: float,
    mask_padding_top: int,
    mask_padding_right: int,
    mask_padding_bottom: int,
    mask_padding_left: int,
    occluder_model: str,
    mask_areas: List[str],
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
        mask_areas=mask_areas,
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

    # In preview mode, always extract embeddings to support reference matching
    target_faces = analyser.get_many_faces([target_frame], extract_embedding=True)
    if not target_faces:
        return cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)

    reference_face = None
    if face_selector_mode == 'reference':
        sorted_all = analyser.sort_faces(target_faces, face_selector_order)
        ref_idx = min(max(0, int(face_selector_position)), len(sorted_all) - 1)
        reference_face = sorted_all[ref_idx]

    selected_faces = analyser.select_faces(
        target_faces=target_faces,
        mode=face_selector_mode,
        order=face_selector_order,
        position=int(face_selector_position),
        reference_face=reference_face,
        reference_distance=float(reference_face_distance)
    )

    result_frame = target_frame.copy()
    for target_face in selected_faces:
        result_frame = swapper.swap_face(
            source_face=source_face,
            target_face=target_face,
            target_vision_frame=result_frame,
            pixel_boost=pixel_boost,
            mask_areas=mask_areas,
            mask_regions=mask_regions
        )

    return cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)


# -----------------------------------------
# 4) Main Batch Execution Function
# -----------------------------------------
def run_batch_swap(
    source_files: List[str],
    target_file: str,
    preview_frame_index: int,
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
    face_selector_mode: str,
    face_selector_order: str,
    face_selector_position: int,
    reference_face_distance: float,
    mask_types: List[str],
    mask_blur: float,
    mask_padding_top: int,
    mask_padding_right: int,
    mask_padding_bottom: int,
    mask_padding_left: int,
    occluder_model: str,
    mask_areas: List[str],
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
        mask_areas=mask_areas,
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
        target_faces = analyser.get_many_faces([img], extract_embedding=True)

        if not target_faces:
            raise gr.Error("No faces detected in target image with current detector settings.")

        reference_face = None
        if face_selector_mode == 'reference':
            sorted_all = analyser.sort_faces(target_faces, face_selector_order)
            ref_idx = min(max(0, int(face_selector_position)), len(sorted_all) - 1)
            reference_face = sorted_all[ref_idx]

        selected_faces = analyser.select_faces(
            target_faces=target_faces,
            mode=face_selector_mode,
            order=face_selector_order,
            position=int(face_selector_position),
            reference_face=reference_face,
            reference_distance=float(reference_face_distance)
        )

        if not selected_faces:
            raise gr.Error("No matching faces found for the specified selector criteria.")

        res = img.copy()
        for target_face in selected_faces:
            res = swapper.swap_face(
                source_face,
                target_face,
                res,
                pixel_boost=pixel_boost,
                mask_areas=mask_areas,
                mask_regions=mask_regions
            )

        out_img_path = os.path.join(output_dir, f"swapped_{target_name}.png")
        cv2.imwrite(out_img_path, res)
        progress(1.0, desc="Completed!")
        return out_img_path, None, f"✅ Image swap completed ({len(selected_faces)} face(s) swapped)."

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

    print(f"🚀 Processing {total_frames} frames using {swapper_model} (Mode: {face_selector_mode}, Order: {face_selector_order}, Pixel Boost: {pixel_boost})...")
    start_time = time.time()
    prepared_source_embedding = swapper.prepare_source_embedding(source_face)

    # Establish reference face identity from the user's selected Preview Target Frame
    reference_face = None
    if face_selector_mode == 'reference' and total_frames > 0:
        ref_frame_num = int(np.clip(preview_frame_index, 0, total_frames - 1))
        ref_frame_path = frame_files[ref_frame_num] if ref_frame_num < len(frame_files) else frame_files[0]
        ref_frame = cv2.imread(ref_frame_path)
        ref_faces = analyser.get_many_faces([ref_frame], extract_embedding=True)
        if ref_faces:
            sorted_ref = analyser.sort_faces(ref_faces, face_selector_order)
            ref_idx = min(max(0, int(face_selector_position)), len(sorted_ref) - 1)
            reference_face = sorted_ref[ref_idx]
            print(f"🎯 Reference face identity established from Preview Target Frame #{ref_frame_num} (Face #{ref_idx}, Order: {face_selector_order})")

    need_embeddings = (face_selector_mode == 'reference')

    for idx, frame_path in enumerate(frame_files):
        frame = cv2.imread(frame_path)
        target_faces = analyser.get_many_faces([frame], extract_embedding=need_embeddings)

        if target_faces:
            selected_faces = analyser.select_faces(
                target_faces=target_faces,
                mode=face_selector_mode,
                order=face_selector_order,
                position=int(face_selector_position),
                reference_face=reference_face,
                reference_distance=float(reference_face_distance)
            )
            for target_face in selected_faces:
                frame = swapper.swap_face(
                    source_face,
                    target_face,
                    frame,
                    pixel_boost=pixel_boost,
                    prepared_source_embedding=prepared_source_embedding,
                    mask_areas=mask_areas,
                    mask_regions=mask_regions
                )

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

    # Cleanup temp frames and clear memory
    try:
        shutil.rmtree(temp_dir)
        shutil.rmtree(swapped_dir)
        if os.path.exists(temp_vid):
            os.remove(temp_vid)
    except Exception:
        pass

    clear_face_cache()
    free_memory()

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
# 6) Gradio Interface Layout & Modern Studio Theme
# -----------------------------------------
import onnxruntime

def get_hw_info_badge() -> str:
    providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return '<span class="hw-badge hw-gpu">⚡ GPU ACCELERATION: CUDA ACTIVE</span>'
    return '<span class="hw-badge hw-cpu">💻 CPU INFERENCE MODE</span>'

custom_css = """
/* ==========================================================================
   MonoFace Pro Studio - Modern Glassmorphism & Responsive Design System
   ========================================================================== */

:root {
    --mf-primary: #6366f1;
    --mf-primary-hover: #4f46e5;
    --mf-accent: #8b5cf6;
    --mf-cyan: #06b6d4;
    --mf-emerald: #10b981;
    --mf-dark-bg: #0b0f19;
    --mf-card-bg: rgba(17, 24, 39, 0.75);
    --mf-card-border: rgba(99, 102, 241, 0.2);
    --mf-card-border-hover: rgba(139, 92, 246, 0.4);
    --mf-text-main: #f3f4f6;
    --mf-text-muted: #9ca3af;
}

/* Global Container & Typography */
.gradio-container {
    max-width: 1500px !important;
    width: 96% !important;
    margin: 0 auto !important;
    padding: 1rem !important;
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: var(--mf-text-main) !important;
}

/* Hide default gradio footer */
footer { display: none !important; }

/* Studio Header Styling */
.studio-header {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 0 30px -10px rgba(99, 102, 241, 0.2);
    position: relative;
    overflow: hidden;
}

.studio-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4, #10b981);
}

.studio-header-content {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
}

.studio-title-group h1 {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #ffffff 30%, #c7d2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.studio-subtitle {
    color: var(--mf-text-muted);
    font-size: 0.88rem;
    margin-top: 0.35rem;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.75rem;
}

.header-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
}

.feature-pill {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.hw-badge {
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    letter-spacing: 0.5px;
}

.hw-gpu {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid rgba(16, 185, 129, 0.5);
    color: #34d399;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.25);
}

.hw-cpu {
    background: rgba(245, 158, 11, 0.2);
    border: 1px solid rgba(245, 158, 11, 0.5);
    color: #fbbf24;
}

/* Glassmorphic Card Containers */
.studio-card {
    background: var(--mf-card-bg) !important;
    border: 1px solid var(--mf-card-border) !important;
    border-radius: 18px !important;
    padding: 1.25rem !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.25) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 1rem !important;
}

.studio-card:hover {
    border-color: var(--mf-card-border-hover) !important;
}

.card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f3f4f6;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Tab Bar Styling */
.tab-nav {
    border-bottom: 1px solid rgba(99, 102, 241, 0.2) !important;
    background: transparent !important;
    gap: 0.35rem !important;
    overflow-x: auto !important;
    flex-wrap: nowrap !important;
    scrollbar-width: thin;
}

.tab-nav button {
    border-radius: 10px 10px 0 0 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1rem !important;
    color: var(--mf-text-muted) !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
}

.tab-nav button.selected {
    color: #ffffff !important;
    background: rgba(99, 102, 241, 0.15) !important;
    border-bottom: 2px solid var(--mf-primary) !important;
}

/* File Uploaders */
.file-upload-compact {
    min-height: 140px !important;
}

/* Hero Run CTA Button */
.run-btn-hero {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #4f46e5 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.5px !important;
    padding: 0.9rem 1.5rem !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    width: 100% !important;
}

.run-btn-hero:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5) !important;
}

.run-btn-hero:active {
    transform: scale(0.98) !important;
}

/* Preview CTA Button */
.preview-btn {
    background: linear-gradient(135deg, #374151 0%, #1f2937 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    color: #e0e7ff !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}

.preview-btn:hover {
    background: rgba(99, 102, 241, 0.25) !important;
    border-color: #8b5cf6 !important;
}

/* Cancel Runtime CTA Button */
.cancel-btn {
    background: rgba(239, 68, 68, 0.15) !important;
    border: 1px solid rgba(239, 68, 68, 0.4) !important;
    color: #f87171 !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border-radius: 14px !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}

.cancel-btn:hover {
    background: rgba(239, 68, 68, 0.3) !important;
    border-color: #ef4444 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.35) !important;
}

.cancel-btn:active {
    transform: scale(0.98) !important;
}

/* Terminal-like Status Log */
.status-log textarea {
    font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace !important;
    font-size: 0.85rem !important;
    background: #060913 !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 12px !important;
}

/* Sliders & Interactive Elements */
input[type="range"] {
    accent-color: #6366f1 !important;
}

/* Reference Face Gallery */
.ref-face-gallery {
    border-radius: 12px !important;
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
}

/* ==========================================================================
   Responsive Breakpoints: Desktop (2-Col Studio) vs Mobile (Adaptive Stack)
   ========================================================================== */

@media (min-width: 1024px) {
    .studio-grid {
        display: grid !important;
        grid-template-columns: 1.15fr 0.85fr !important;
        gap: 1.25rem !important;
        align-items: start !important;
    }
    
    .sticky-preview-deck {
        position: sticky !important;
        top: 1rem !important;
        z-index: 10 !important;
    }
}

@media (max-width: 1023px) {
    .gradio-container {
        width: 100% !important;
        padding: 0.5rem !important;
    }
    
    .studio-header {
        padding: 1.25rem 1rem !important;
        border-radius: 14px !important;
    }
    
    .studio-title-group h1 {
        font-size: 1.45rem !important;
    }
    
    .studio-card {
        padding: 1rem !important;
        border-radius: 14px !important;
    }
    
    .tab-nav button {
        padding: 0.5rem 0.75rem !important;
        font-size: 0.8rem !important;
    }
    
    .run-btn-hero {
        font-size: 0.95rem !important;
        padding: 0.8rem 1.2rem !important;
    }
}

@media (max-width: 640px) {
    .studio-header-content {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .header-badges {
        width: 100%;
        justify-content: flex-start;
    }
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
).set(
    body_background_fill="#0b0f19",
    body_background_fill_dark="#0b0f19",
    block_background_fill="rgba(17, 24, 39, 0.75)",
    block_background_fill_dark="rgba(17, 24, 39, 0.75)",
    block_border_color="rgba(99, 102, 241, 0.2)",
    block_border_color_dark="rgba(99, 102, 241, 0.2)",
    button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
    button_primary_background_fill_dark="linear-gradient(135deg, #6366f1, #8b5cf6)",
    button_primary_text_color="#ffffff"
)

with gr.Blocks(title="MonoFace Studio Pro") as demo:
    # -------------------------------------------------------------
    # Studio Header
    # -------------------------------------------------------------
    gr.HTML(
        f"""
        <div class="studio-header">
            <div class="studio-header-content">
                <div class="studio-title-group">
                    <h1>⚡ MonoFace Studio Pro</h1>
                    <div class="studio-subtitle">
                        <span>High-Fidelity AI Face Swapping & Video Pipeline</span>
                        <div class="header-badges">
                            <span class="feature-pill">🎯 Multi-Detector</span>
                            <span class="feature-pill">✨ Super-Res Boost</span>
                            <span class="feature-pill">🎭 Precision Masking</span>
                            <span class="feature-pill">🎞️ Real-time Preview</span>
                        </div>
                    </div>
                </div>
                <div>
                    {get_hw_info_badge()}
                </div>
            </div>
        </div>
        """
    )

    # -------------------------------------------------------------
    # Main Responsive Studio Layout
    # -------------------------------------------------------------
    with gr.Row(elem_classes=["studio-grid"]):
        # =========================================================
        # LEFT COLUMN: Media Inputs & Pipeline Configuration
        # =========================================================
        with gr.Column():
            # 1. Media Upload Hub Card
            with gr.Group(elem_classes=["studio-card"]):
                gr.HTML("<div class='card-title'>📁 Step 1: Input Media Assets</div>")
                with gr.Row():
                    src_files = gr.File(
                        label="Source Face Image(s)",
                        file_count="multiple",
                        type="filepath",
                        file_types=["image"],
                        elem_classes=["file-upload-compact"]
                    )
                    tgt_file = gr.File(
                        label="Target Image or Video",
                        file_count="single",
                        type="filepath",
                        file_types=["image", "video"],
                        elem_classes=["file-upload-compact"]
                    )
                
                # Video Statistics Bar
                with gr.Row():
                    orig_fps = gr.Textbox(label="Original Video FPS", value="0.0", interactive=False, scale=1)
                    total_duration = gr.Textbox(label="Total Duration (seconds)", value="0.0", interactive=False, scale=1)

            # 2. Pipeline Controls Card (Organized in Modular Tabs)
            with gr.Group(elem_classes=["studio-card"]):
                gr.HTML("<div class='card-title'>🎛️ Step 2: Advanced Pipeline Configuration</div>")
                
                with gr.Tabs(elem_classes=["tab-nav"]):
                    # Tab A: Swapper & Model Options
                    with gr.Tab("⚙️ Swapper & Model"):
                        with gr.Row():
                            swapper_model = gr.Dropdown(
                                choices=[
                                    "inswapper_128_fp16",
                                    "inswapper_128",
                                    "hyperswap_1a_256",
                                    "hyperswap_1b_256",
                                    "hyperswap_1c_256",
                                    "simswap_256",
                                    "simswap_512_unofficial"
                                ],
                                value="inswapper_128_fp16",
                                label="Face Swapper Model"
                            )
                            pixel_boost = gr.Dropdown(
                                choices=["none", "128x128", "256x256", "384x384", "512x512", "768x768", "1024x1024"],
                                value="none",
                                label="Pixel Boost (Super-Resolution)"
                            )
                        swapper_weight = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Swapper Identity Balance (0=Target, 1=Source)"
                        )

                    # Tab B: Detection & Landmarker
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
                                value=0.0,
                                step=0.05,
                                label="Landmarker Threshold (0.0=Fast 5-pt)"
                            )

                        # Margin expansion accordion
                        with gr.Accordion("📐 Face Margins / Detector Expansion (%)", open=False):
                            with gr.Row():
                                margin_top = gr.Slider(0, 100, value=0, step=1, label="Margin Top (%)")
                                margin_right = gr.Slider(0, 100, value=0, step=1, label="Margin Right (%)")
                            with gr.Row():
                                margin_bottom = gr.Slider(0, 100, value=0, step=1, label="Margin Bottom (%)")
                                margin_left = gr.Slider(0, 100, value=0, step=1, label="Margin Left (%)")

                    # Tab C: Face Selector & Reference
                    with gr.Tab("👤 Face Selector & Reference"):
                        with gr.Row():
                            face_selector_mode = gr.Dropdown(
                                choices=["many", "reference"],
                                value="many",
                                label="Selector Mode (many=All faces, reference=Match target)"
                            )
                            face_selector_order = gr.Dropdown(
                                choices=[
                                    "large-small",
                                    "small-large",
                                    "left-right",
                                    "right-left",
                                    "top-bottom",
                                    "bottom-top",
                                    "best-worst",
                                    "worst-best"
                                ],
                                value="large-small",
                                label="Face Sorting Order"
                            )
                        
                        reference_face_distance = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.3,
                            label="Reference Face Distance Threshold",
                            visible=False
                        )

                        # Internal state for selected target face position index
                        face_selector_position = gr.State(value=0)

                        with gr.Group(visible=False) as reference_container:
                            gr.Markdown("#### 🔍 Detected Faces in Frame (Click to select Reference Face):")
                            ref_gallery = gr.Gallery(
                                label="Detected Faces",
                                columns=4,
                                height=180,
                                allow_preview=False,
                                object_fit="cover",
                                elem_classes=["ref-face-gallery"]
                            )
                            ref_status = gr.Markdown("Load a target image/video or adjust preview slider to detect faces.")

                    # Tab D: Masking & Occlusion
                    with gr.Tab("🎭 Masking & Occlusion"):
                        with gr.Row():
                            mask_types = gr.CheckboxGroup(
                                choices=["box", "occlusion", "region", "area"],
                                value=["box"],
                                label="Active Mask Types"
                            )
                            occluder_model = gr.Dropdown(
                                choices=["xseg_1", "xseg_2", "xseg_3"],
                                value="xseg_1",
                                label="Occlusion Model"
                            )

                        mask_blur = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.05,
                            label="Mask Feather / Box Blur"
                        )

                        with gr.Row():
                            mask_areas = gr.CheckboxGroup(
                                choices=["upper-face", "lower-face", "mouth", "eyes", "nose"],
                                value=["upper-face", "lower-face", "mouth", "eyes", "nose"],
                                label="Landmark Areas ('area' mask)"
                            )
                        with gr.Row():
                            mask_regions = gr.CheckboxGroup(
                                choices=["skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye", "nose", "mouth", "upper-lip", "lower-lip"],
                                value=["skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye", "nose", "mouth", "upper-lip", "lower-lip"],
                                label="Semantic Regions ('region' mask)"
                            )

                        with gr.Accordion("✂️ Mask Padding Margins (%)", open=False):
                            with gr.Row():
                                mask_padding_top = gr.Slider(0, 100, value=0, step=1, label="Padding Top (%)")
                                mask_padding_right = gr.Slider(0, 100, value=0, step=1, label="Padding Right (%)")
                            with gr.Row():
                                mask_padding_bottom = gr.Slider(0, 100, value=0, step=1, label="Padding Bottom (%)")
                                mask_padding_left = gr.Slider(0, 100, value=0, step=1, label="Padding Left (%)")

                    # Tab E: Video Trimming & Encoding
                    with gr.Tab("✂️ Trimming & Encoding"):
                        trim_mode = gr.Radio(["Seconds", "Frames"], value="Seconds", label="Trim Mode")
                        with gr.Row():
                            start_sec = gr.Number(label="Start Time (sec)", value=0.0)
                            end_sec = gr.Number(label="End Time (sec, 0=Full)", value=0.0)
                        with gr.Row():
                            start_frame = gr.Number(label="Start Frame", value=0)
                            end_frame = gr.Number(label="End Frame (0=Full)", value=0)
                        with gr.Row():
                            frame_quality = gr.Slider(0, 100, value=80, step=1, label="Output Frame Quality (0-100)")
                            fps_override = gr.Slider(0, 120, value=0, step=1, label="FPS Override (0=Same as Target)")

        # =========================================================
        # RIGHT COLUMN: Interactive Live Preview & Execution Studio
        # =========================================================
        with gr.Column(elem_classes=["sticky-preview-deck"]):
            # 3. Interactive Frame Preview Inspector
            with gr.Group(visible=False, elem_classes=["studio-card"]) as preview_box:
                gr.HTML("<div class='card-title'>🎞️ Step 3: Interactive Frame Inspector</div>")
                with gr.Row():
                    frame_slider = gr.Slider(minimum=0, maximum=1, step=1, value=0, label="Target Video Frame Index", scale=3)
                    preview_btn = gr.Button("👁️ Preview Frame", variant="secondary", scale=2, elem_classes=["preview-btn"])
                preview_output = gr.Image(label="Live Single-Frame Swap Preview", type="numpy")

            # 4. Primary Run Action & Live Status Deck
            with gr.Group(elem_classes=["studio-card"]):
                gr.HTML("<div class='card-title'>🚀 Step 4: Execution & Status</div>")
                with gr.Row():
                    run_btn = gr.Button("⚡ RUN MONOFACE", variant="primary", scale=3, elem_classes=["run-btn-hero"])
                    cancel_btn = gr.Button("⏹️ CANCEL RUNTIME", variant="stop", scale=2, elem_classes=["cancel-btn"])
                status_box = gr.Textbox(
                    label="Status & Execution Log",
                    value="Ready to process. Upload source & target files to begin.",
                    interactive=False,
                    lines=3,
                    elem_classes=["status-log"]
                )

            # 5. Output Showcase
            with gr.Group(elem_classes=["studio-card"]):
                gr.HTML("<div class='card-title'>🎬 Step 5: Processed Results</div>")
                with gr.Row():
                    out_image = gr.Image(label="Swapped Output Image", visible=True)
                    out_video = gr.Video(label="Swapped Output Video", visible=True)

    # -------------------------------------------------------------
    # Dynamic Events & Callbacks
    # -------------------------------------------------------------
    tgt_file.change(
        fn=update_ui_for_target,
        inputs=[tgt_file],
        outputs=[out_image, out_video, orig_fps, total_duration, preview_box, frame_slider]
    )

    ref_detector_inputs = [
        tgt_file,
        frame_slider,
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
        face_selector_order,
        face_selector_position
    ]

    def on_reference_gallery_select(evt: gr.SelectData):
        try:
            raw_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            selected_idx = int(raw_idx)
            return selected_idx, f"🎯 Selected Face #{selected_idx} as active target/reference face."
        except Exception:
            return 0, "🎯 Selected Face #0 as active target/reference face."

    ref_gallery.select(
        fn=on_reference_gallery_select,
        inputs=None,
        outputs=[face_selector_position, ref_status]
    )

    def on_face_selector_mode_change(
        mode: str,
        target_file: Optional[str],
        frame_index: int,
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
        face_selector_order: str,
        face_selector_position: int
    ):
        is_ref = (mode == "reference")
        if is_ref:
            gallery_items, status = update_reference_face_gallery(
                target_file, frame_index, detector_model, detector_size_str, detector_score, detector_angles,
                margin_top, margin_right, margin_bottom, margin_left, landmarker_model, landmarker_score,
                face_selector_order, face_selector_position
            )
            return gr.update(visible=True), gr.update(visible=True), gallery_items, status
        return gr.update(visible=False), gr.update(visible=False), [], "Switch to 'reference' mode to select a reference face."

    face_selector_mode.change(
        fn=on_face_selector_mode_change,
        inputs=[face_selector_mode] + ref_detector_inputs,
        outputs=[reference_face_distance, reference_container, ref_gallery, ref_status]
    )

    # Automatically refresh detected target faces gallery on parameter updates
    for comp in [tgt_file, face_selector_order]:
        comp.change(
            fn=update_reference_face_gallery,
            inputs=ref_detector_inputs,
            outputs=[ref_gallery, ref_status]
        )

    for comp in [detector_model, detector_size, detector_angles, landmarker_model]:
        comp.change(
            fn=update_reference_face_gallery,
            inputs=ref_detector_inputs,
            outputs=[ref_gallery, ref_status]
        )

    for comp in [frame_slider, detector_score, landmarker_score, margin_top, margin_right, margin_bottom, margin_left]:
        comp.release(
            fn=update_reference_face_gallery,
            inputs=ref_detector_inputs,
            outputs=[ref_gallery, ref_status]
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
        face_selector_mode,
        face_selector_order,
        face_selector_position,
        reference_face_distance,
        mask_types,
        mask_blur,
        mask_padding_top,
        mask_padding_right,
        mask_padding_bottom,
        mask_padding_left,
        occluder_model,
        mask_areas,
        mask_regions
    ]

    preview_btn.click(
        fn=preview_swap_frame,
        inputs=[src_files, tgt_file, frame_slider] + all_config_inputs,
        outputs=[preview_output]
    )

    run_event = run_btn.click(
        fn=run_batch_swap,
        inputs=[
            src_files,
            tgt_file,
            frame_slider,
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

    cancel_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[run_event]
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MonoFace Studio")
    parser.add_argument("--share", type=bool, default=True, help="Share the app")
    parser.add_argument("--inbrowser", type=bool, default=True, help="Open in browser")
    # parser.add_argument("--server-name", type=str, default="[IP_ADDRESS]", help="Server name")
    args = parser.parse_args()
    
    providers = onnxruntime.get_available_providers()
    print(f"🔥 Active ONNX Runtime Execution Providers: {providers}")
    if "CUDAExecutionProvider" not in providers:
        print("⚠️ Warning: CUDAExecutionProvider not detected! Running on CPU. For 10-30x faster GPU inference, install: pip install onnxruntime-gpu")
    
    print("⏳ Preloading face analyser models (Detector, Landmarker, Fan 68/5, Recognizer)...")
    try:
        preload_face_analyser()
        print("✅ Face analyser models preloaded successfully!")
    except Exception as e:
        print(f"⚠️ Note: Face analyser models will load on first inference ({e})")

    try:
        gr.close_all()
    except Exception:
        pass

    try:
        demo.launch(
            share=args.share,
            inbrowser=args.inbrowser,
            server_name="0.0.0.0",
            server_port=7860,
            theme=theme,
            css=custom_css
        )
    except OSError:
        print("⚠️ Port 7860 occupied. Falling back to automatically selected open port...")
        demo.launch(
            share=args.share,
            inbrowser=args.inbrowser,
            server_name="0.0.0.0",
            theme=theme,
            css=custom_css
        )

