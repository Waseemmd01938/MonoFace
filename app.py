#@title MonoFace - Advanced Face Swapping & Processing Interface
import glob
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import gradio as gr
from gradio.themes import Size
import numpy as np
import onnxruntime
from tqdm import tqdm

from Face.modules import FaceSwapper, free_memory
from Face.typing import Face
from face_analyser import FaceAnalyser, clear_face_cache, preload_face_analyser
from downloads import set_download_callback

# Cached pipeline instance to avoid repeated object recreation
_CACHED_PIPELINE: Dict[str, Any] = {}

# Process execution state: 'pending', 'processing', 'stopping'
PROCESS_STATE: str = 'pending'


def get_process_state() -> str:
    global PROCESS_STATE
    return PROCESS_STATE


def is_processing() -> bool:
    return get_process_state() == 'processing'


def is_stopping() -> bool:
    return get_process_state() == 'stopping'


def start_process() -> None:
    global PROCESS_STATE
    PROCESS_STATE = 'processing'


def stop_process() -> None:
    global PROCESS_STATE
    PROCESS_STATE = 'stopping'


def end_process() -> None:
    global PROCESS_STATE
    PROCESS_STATE = 'pending'


# Active status logs buffer for terminal display
_RECENT_STATUS_LOGS: List[str] = [
    "[MONOFACE.CORE] Initialized MonoFace pipeline engine.",
    "Ready to process. Upload source & target files."
]

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
    start_frame: int,
    end_frame: int,
    fps_override: float,
    quality_0_100: int
) -> float:
    """Extracts frames using FFmpeg with high performance and quality mapping."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    total_frames, fps, w, h = get_video_info(video_path)
    start_sec = frame_to_sec(int(start_frame), fps)
    end_sec = frame_to_sec(int(end_frame), fps) if end_frame > 0 else frame_to_sec(total_frames, fps)

    duration = max(0.1, end_sec - start_sec)
    out_fps = fps if fps_override <= 0 else fps_override

    ffmpeg_q = int(31 - (quality_0_100 * 0.29))
    ffmpeg_q = max(2, min(31, ffmpeg_q))

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", video_path,
        "-vf", f"fps={out_fps}",
        "-q:v", str(ffmpeg_q),
        os.path.join(output_dir, "frame_%06d.jpg")
    ]

    subprocess.run(cmd, check=True)
    return out_fps


def frames_to_video_ffmpeg(
    frame_dir: str,
    output_path: str,
    fps: float,
    video_encoder: str = "libx264",
    video_preset: str = "veryfast",
    video_quality: int = 80
) -> str:
    """Stitches frames into video with configurable encoder and quality."""
    input_pattern = os.path.join(frame_dir, "frame_%06d.jpg")
    temp_vid = output_path.replace(".mp4", "_silent.mp4")

    crf_val = int(51 - (video_quality * 0.51))
    crf_val = max(1, min(51, crf_val))

    cmd_vid = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", video_encoder if video_encoder else "libx264",
        "-preset", video_preset if video_preset else "veryfast",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf_val),
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
    """Crops a face with balanced margin padding for reference gallery display."""
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
    """Extracts detected faces from current target/frame for interactive visual selection."""
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
            return [], "No faces detected in the target frame with current detector settings."

        sorted_faces = analyser.sort_faces(faces, face_selector_order)
        gallery_items = []
        for idx, f in enumerate(sorted_faces):
            avatar = crop_face_avatar(target_frame, f)
            score = f.score_set.get('detector', 0.0) if isinstance(f.score_set, dict) else 0.0
            caption = f"Face #{idx} ({score:.2f})"
            gallery_items.append((avatar, caption))

        sel_pos = int(face_selector_position) if isinstance(face_selector_position, (int, float, str)) else 0
        sel_idx = min(max(0, sel_pos), len(sorted_faces) - 1)
        status = f"Detected {len(sorted_faces)} face(s). Active Selection: Face #{sel_idx}."

        return gallery_items, status
    except Exception as e:
        return [], f"Face detection error: {str(e)}"


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
    mask_regions: List[str],
    preview_mode: str = "default"
) -> np.ndarray:
    if not source_files or not target_file:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    src_list = source_files if isinstance(source_files, list) else [source_files]
    if len(src_list) == 0 or not src_list[0]:
        return np.zeros((480, 640, 3), dtype=np.uint8)

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

    source_face = get_source_face_from_paths(analyser, src_list)

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

    if preview_mode == "side-by-side":
        combined = np.hstack([target_frame, result_frame])
        return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

    return cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)


# -----------------------------------------
# 4) Main Batch Execution Function
# -----------------------------------------
def run_batch_swap(
    source_files: Any,
    target_file: str,
    preview_frame_index: int,
    output_custom_path: str,
    output_video_fps: float,
    output_video_quality: int,
    output_video_encoder: str,
    output_video_preset: str,
    trim_start: float,
    trim_end: float,
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
    progress=gr.Progress(track_tqdm=True)
) -> Tuple[Optional[str], Optional[str], str]:
    start_process()
    if not source_files or not target_file:
        end_process()
        raise gr.Error("Please upload both source image(s) and a target file.")

    src_list = source_files if isinstance(source_files, list) else [source_files]
    if len(src_list) == 0 or not src_list[0]:
        end_process()
        raise gr.Error("Please upload at least one valid source face image.")

    target_name, ext = safe_filename(target_file)
    print(f"[TARGET] Target File: {target_file} (Format: {ext})")

    download_logs: List[str] = []

    def runtime_download_tracker(model_name: str, percent: float, downloaded: int, total_size: int, speed_mb: float):
        done_mb = downloaded / (1024 * 1024)
        tot_mb = total_size / (1024 * 1024)
        ratio = min(1.0, downloaded / max(1, total_size))
        progress(ratio * 0.05, desc=f"Downloading {model_name}: {percent:.0f}% ({done_mb:.1f}/{tot_mb:.1f} MB @ {speed_mb:.1f} MB/s)")
        if percent >= 100.0 or downloaded >= total_size:
            download_logs.append(f"[MONOFACE.DOWNLOAD] Model ready: {model_name} ({tot_mb:.1f} MB)")

    set_download_callback(runtime_download_tracker)

    try:
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
    finally:
        set_download_callback(None)

    if is_stopping():
        end_process()
        return None, None, "Processing cancelled by user."

    progress(0.05, desc="Extracting Source Face Embeddings...")
    source_face = get_source_face_from_paths(analyser, src_list)

    if is_stopping():
        end_process()
        return None, None, "Processing cancelled by user."

    # Output directory
    output_dir = output_custom_path.strip() if output_custom_path and output_custom_path.strip() else os.path.join(os.path.dirname(os.path.abspath(__file__)), "Outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------
    # Image Target Mode
    # ------------------
    if ext in IMAGE_EXTS:
        start_time = time.time()
        print(f"[PROCESS] Processing image: '{target_name}'...")
        progress(0.4, desc="Swapping face in image...")
        img = read_image(target_file)
        target_faces = analyser.get_many_faces([img], extract_embedding=True)

        if not target_faces:
            end_process()
            raise gr.Error("No faces detected in target image with current detector settings.")

        if is_stopping():
            end_process()
            return None, None, "Processing cancelled by user."

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
            end_process()
            raise gr.Error("No matching faces found for the specified selector criteria.")

        res = img.copy()
        for target_face in selected_faces:
            if is_stopping():
                end_process()
                return None, None, "Processing cancelled by user."
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
        end_process()

        elapsed = time.time() - start_time
        out_basename = os.path.basename(out_img_path)
        print(f"[PROCESS] Processing to image succeeded: '{out_basename}' in {elapsed:.2f} seconds")
        dl_prefix = "\n".join(download_logs) + "\n" if download_logs else ""
        log_text = dl_prefix + f"Processing to image succeeded in {elapsed:.2f} seconds.\nSaved: {out_basename}"
        return out_img_path, None, log_text

    # ------------------
    # Video Target Mode
    # ------------------
    temp_dir = os.path.join(output_dir, "temp_frames")
    swapped_dir = os.path.join(output_dir, "temp_swapped_frames")

    if is_stopping():
        end_process()
        return None, None, "Processing cancelled by user."

    progress(0.1, desc="Extracting video frames via FFmpeg...")
    try:
        real_fps = extract_frames_ffmpeg(
            video_path=target_file,
            output_dir=temp_dir,
            start_frame=int(trim_start),
            end_frame=int(trim_end),
            fps_override=output_video_fps,
            quality_0_100=output_video_quality
        )
    except Exception as e:
        end_process()
        raise gr.Error(f"Video extraction error: {e}")

    if is_stopping():
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
        end_process()
        return None, None, "Processing cancelled by user."

    if os.path.exists(swapped_dir):
        shutil.rmtree(swapped_dir)
    os.makedirs(swapped_dir, exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(temp_dir, "*.jpg")))
    total_frames = len(frame_files)

    if total_frames == 0:
        end_process()
        raise gr.Error("No frames extracted! Please check your trim settings.")

    print(f"[PROCESS] Processing {total_frames} frames using {swapper_model} (Mode: {face_selector_mode}, Order: {face_selector_order})...")
    start_time = time.time()
    prepared_source_embedding = swapper.prepare_source_embedding(source_face)

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
            print(f"[REFERENCE] Reference face identity established from Preview Frame #{ref_frame_num}")

    need_embeddings = (face_selector_mode == 'reference')

    # Terminal progress bar using tqdm
    with tqdm(total=total_frames, desc="Processing", unit="frame", ascii=" =", file=sys.stdout, dynamic_ncols=True, mininterval=0.1) as pbar:
        for idx, frame_path in enumerate(frame_files):
            # Check cancellation signal on every frame
            if is_stopping():
                print(f"\n[PROCESS] Processing stopped by user at frame {idx}/{total_frames}.")
                try:
                    shutil.rmtree(temp_dir)
                    shutil.rmtree(swapped_dir)
                except Exception:
                    pass
                clear_face_cache()
                free_memory()
                end_process()
                return None, None, f"Processing stopped by user at frame {idx}/{total_frames}."

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
            pbar.update(1)
            sys.stdout.flush()

            if idx % 2 == 0 or idx == total_frames - 1:
                elapsed = time.time() - start_time
                current_fps = (idx + 1) / max(elapsed, 0.001)
                ratio = (idx + 1) / total_frames
                progress(0.15 + (0.75 * ratio), desc=f"Swapping: {idx+1}/{total_frames} [{current_fps:.1f} FPS]")

    if is_stopping():
        try:
            shutil.rmtree(temp_dir)
            shutil.rmtree(swapped_dir)
        except Exception:
            pass
        clear_face_cache()
        free_memory()
        end_process()
        return None, None, "Processing cancelled by user."

    # Stitch video back
    progress(0.92, desc="Stitching frames with FFmpeg...")
    final_video = os.path.join(output_dir, f"swapped_{target_name}.mp4")
    temp_vid = frames_to_video_ffmpeg(
        swapped_dir,
        final_video,
        real_fps,
        video_encoder=output_video_encoder,
        video_preset=output_video_preset,
        video_quality=output_video_quality
    )

    # Sync Audio
    progress(0.97, desc="Synchronizing audio track...")
    total_orig_frames, orig_fps, _, _ = get_video_info(target_file)
    s_sec = frame_to_sec(int(trim_start), orig_fps)

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

    # Cleanup
    try:
        shutil.rmtree(temp_dir)
        shutil.rmtree(swapped_dir)
        if os.path.exists(temp_vid):
            os.remove(temp_vid)
    except Exception:
        pass

    clear_face_cache()
    free_memory()
    end_process()

    total_time = time.time() - start_time
    avg_fps = total_frames / max(total_time, 0.001)
    out_vid_name = os.path.basename(final_video)
    print(f"[PROCESS] Processing to video succeeded: '{out_vid_name}' in {total_time:.2f} seconds (Average: {avg_fps:.2f} FPS)")
    dl_prefix = "\n".join(download_logs) + "\n" if download_logs else ""
    log_text = dl_prefix + f"Processing to video succeeded in {total_time:.2f} seconds (Average: {avg_fps:.2f} FPS)\nSaved: {out_vid_name}"

    return None, final_video, log_text


# -----------------------------------------
# 5) UI Target Info Handlers
# -----------------------------------------
def on_target_change(target_file: Optional[str]):
    if not target_file:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(maximum=100, value=0),
            gr.update(maximum=100, value=0)
        )
    _, ext = safe_filename(target_file)
    if ext in VIDEO_EXTS:
        total_frames, fps, _, _ = get_video_info(target_file)
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(maximum=max(total_frames - 1, 1), value=0),
            gr.update(maximum=max(total_frames - 1, 1), value=max(total_frames - 1, 1))
        )
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(maximum=1, value=0),
        gr.update(maximum=1, value=1)
    )


# -----------------------------------------
# 6) FaceFusion Blue Theme & CSS Overrides
# -----------------------------------------
def get_facefusion_theme() -> gr.Theme:
    """Blue-accent FaceFusion theme."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.Color(
            name='neutral',
            c50='#fafafa',
            c100='#f5f5f5',
            c200='#e5e5e5',
            c300='#d4d4d4',
            c400='#a3a3a3',
            c500='#737373',
            c600='#525252',
            c700='#404040',
            c800='#262626',
            c900='#212121',
            c950='#171717',
        ),
        radius_size=Size(
            xxs='0.375rem',
            xs='0.375rem',
            sm='0.375rem',
            md='0.375rem',
            lg='0.375rem',
            xl='0.375rem',
            xxl='0.375rem',
        ),
        font=gr.themes.GoogleFont('Open Sans')
    ).set(
        color_accent='transparent',
        color_accent_soft='transparent',
        color_accent_soft_dark='transparent',
        background_fill_primary='*neutral_100',
        background_fill_primary_dark='*neutral_950',
        background_fill_secondary='*neutral_50',
        background_fill_secondary_dark='*neutral_800',
        block_background_fill='white',
        block_background_fill_dark='*neutral_900',
        block_border_width='0',
        block_label_background_fill='*neutral_100',
        block_label_background_fill_dark='*neutral_800',
        block_label_border_width='none',
        block_label_margin='0.5rem',
        block_label_radius='*radius_md',
        block_label_text_color='*neutral_700',
        block_label_text_size='*text_sm',
        block_label_text_color_dark='white',
        block_label_text_weight='600',
        block_title_background_fill='*neutral_100',
        block_title_background_fill_dark='*neutral_800',
        block_title_padding='*block_label_padding',
        block_title_radius='*block_label_radius',
        block_title_text_color='*neutral_700',
        block_title_text_size='*text_sm',
        block_title_text_weight='600',
        block_padding='0.5rem',
        border_color_accent='transparent',
        border_color_accent_dark='transparent',
        border_color_accent_subdued='transparent',
        border_color_accent_subdued_dark='transparent',
        border_color_primary='transparent',
        border_color_primary_dark='transparent',
        button_large_padding='2rem 0.5rem',
        button_large_text_weight='normal',
        button_primary_background_fill='*primary_600',
        button_primary_background_fill_dark='*primary_600',
        button_primary_text_color='white',
        button_secondary_background_fill='white',
        button_secondary_background_fill_dark='*neutral_800',
        button_secondary_background_fill_hover='white',
        button_secondary_background_fill_hover_dark='*neutral_800',
        button_secondary_text_color='*neutral_800',
        button_small_padding='0.75rem',
        button_small_text_size='0.875rem',
        checkbox_background_color='*neutral_200',
        checkbox_background_color_dark='*neutral_900',
        checkbox_background_color_selected='*primary_600',
        checkbox_background_color_selected_dark='*primary_700',
        checkbox_label_background_fill='*neutral_50',
        checkbox_label_background_fill_dark='*neutral_800',
        checkbox_label_background_fill_hover='*neutral_50',
        checkbox_label_background_fill_hover_dark='*neutral_800',
        checkbox_label_background_fill_selected='*primary_600',
        checkbox_label_background_fill_selected_dark='*primary_600',
        checkbox_label_text_color_selected='white',
        error_background_fill='white',
        error_background_fill_dark='*neutral_900',
        error_text_color='*primary_600',
        error_text_color_dark='*primary_600',
        input_background_fill='*neutral_50',
        input_background_fill_dark='*neutral_800',
        shadow_drop='none',
        slider_color='*primary_600',
        slider_color_dark='*primary_600'
    )


def get_facefusion_css() -> str:
    local_css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overrides.css")
    if os.path.exists(local_css_path):
        with open(local_css_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    :root:root:root:root .gradio-container { overflow: unset; }
    :root:root:root:root main { max-width: 110em; }
    :root:root:root:root footer { display: none; }
    """


with gr.Blocks(title="MonoFace Studio Pro", fill_width=True) as demo:

    # 3-Column Layout following FaceFusion architecture (scale=4, scale=4, scale=7)
    with gr.Row():

        # =========================================================
        # COLUMN 1 (scale=4): Models & System Configuration
        # =========================================================
        with gr.Column(scale=4):
            
            # About Banner
            with gr.Blocks():
                gr.Button(value="MonoFace Studio Pro", variant="primary")
                active_providers = onnxruntime.get_available_providers()
                hw_text = "⚡ GPU ACCELERATION: CUDA ACTIVE" if "CUDAExecutionProvider" in active_providers else "💻 CPU INFERENCE MODE"
                gr.Button(value=hw_text, size="sm")

            # Face Swapper Model Options
            with gr.Blocks():
                swapper_model = gr.Dropdown(
                    label="FACE SWAPPER MODEL",
                    choices=[
                        "hyperswap_1a_256",
                        "inswapper_128_fp16",
                        "inswapper_128",
                        "hyperswap_1b_256",
                        "hyperswap_1c_256",
                        "simswap_256",
                        "simswap_512_unofficial"
                    ],
                    value="hyperswap_1a_256"
                )
                pixel_boost = gr.Dropdown(
                    label="FACE SWAPPER PIXEL BOOST",
                    choices=["none", "128x128", "256x256", "384x384", "512x512", "768x768", "1024x1024"],
                    value="256x256"
                )
                swapper_weight = gr.Slider(
                    label="FACE SWAPPER WEIGHT",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05
                )

            # Output Options
            with gr.Blocks():
                output_video_encoder = gr.Dropdown(
                    label="OUTPUT VIDEO ENCODER",
                    choices=["libx264", "libx265", "h264_nvenc", "hevc_nvenc"],
                    value="libx264"
                )
                output_video_preset = gr.Dropdown(
                    label="OUTPUT VIDEO PRESET",
                    choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
                    value="veryfast"
                )
                output_video_quality = gr.Slider(
                    label="OUTPUT VIDEO QUALITY",
                    minimum=0,
                    maximum=100,
                    value=80,
                    step=1
                )
                output_video_fps = gr.Slider(
                    label="OUTPUT VIDEO FPS",
                    minimum=0,
                    maximum=120,
                    value=0,
                    step=0.01
                )
                output_path_input = gr.Textbox(
                    label="OUTPUT PATH",
                    value=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Outputs")
                )

        # =========================================================
        # COLUMN 2 (scale=4): Source, Target, Output, Terminal & Runner
        # =========================================================
        with gr.Column(scale=4):
            
            # Source Box
            with gr.Blocks():
                src_files = gr.File(
                    label="SOURCE",
                    file_count="multiple",
                    type="filepath",
                    file_types=["image"]
                )
                src_image_preview = gr.Image(
                    show_label=False,
                    visible=False,
                    interactive=False
                )

            # Target Box
            with gr.Blocks():
                tgt_file = gr.File(
                    label="TARGET",
                    file_count="single",
                    type="filepath",
                    file_types=["image", "video"]
                )
                tgt_image_preview = gr.Image(
                    show_label=False,
                    visible=False,
                    interactive=False
                )
                tgt_video_preview = gr.Video(
                    show_label=False,
                    visible=False,
                    interactive=False
                )

            # Output Box
            with gr.Blocks():
                out_image = gr.Image(label="OUTPUT", visible=True)
                out_video = gr.Video(label="OUTPUT", visible=False)

            # Terminal Box
            with gr.Blocks():
                status_box = gr.Textbox(
                    label="TERMINAL",
                    value="Initialized MonoFace pipeline engine.\nReady to process. Upload source & target files.",
                    lines=5,
                    interactive=False
                )

            # Instant Runner
            with gr.Blocks():
                with gr.Row():
                    run_btn = gr.Button(value="START", variant="primary", size="sm")
                    clear_btn = gr.Button(value="CLEAR", size="sm")

        # =========================================================
        # COLUMN 3 (scale=7): Preview, Selector, Masking, Detector, Landmarker
        # =========================================================
        with gr.Column(scale=7):
            
            # Preview Box
            with gr.Blocks():
                preview_output = gr.Image(label="PREVIEW", type="numpy")
                preview_frame_slider = gr.Slider(
                    label="PREVIEW FRAME",
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    visible=False
                )
                preview_mode = gr.Dropdown(
                    label="PREVIEW MODE",
                    choices=["default", "side-by-side"],
                    value="default"
                )

            # Trim Frame Box
            with gr.Blocks():
                with gr.Row():
                    trim_start = gr.Slider(label="TRIM FRAME START", minimum=0, maximum=100, value=0, step=1, visible=False)
                    trim_end = gr.Slider(label="TRIM FRAME END", minimum=0, maximum=100, value=100, step=1, visible=False)

            # Face Selector Box
            with gr.Blocks():
                face_selector_mode = gr.Dropdown(
                    label="FACE SELECTOR MODE",
                    choices=["reference", "many", "one"],
                    value="reference"
                )
                with gr.Group(visible=True) as reference_container:
                    ref_gallery = gr.Gallery(
                        label="REFERENCE FACE (Click to select target face)",
                        columns=4,
                        height=100,
                        allow_preview=False,
                        object_fit="cover"
                    )
                    ref_status = gr.Markdown("Detected reference faces will appear here.")
                face_selector_position = gr.State(value=0)
                with gr.Row():
                    face_selector_order = gr.Dropdown(
                        label="FACE SELECTOR ORDER",
                        choices=["large-small", "small-large", "left-right", "right-left", "top-bottom", "bottom-top", "best-worst", "worst-best"],
                        value="large-small"
                    )
                    reference_face_distance = gr.Slider(
                        label="REFERENCE FACE DISTANCE",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                        visible=True
                    )

            # Face Masker Box
            with gr.Blocks():
                with gr.Row():
                    occluder_model = gr.Dropdown(
                        label="FACE OCCLUDER MODEL",
                        choices=["xseg_1", "xseg_2", "xseg_3"],
                        value="xseg_1",
                        visible=False
                    )
                    mask_blur = gr.Slider(
                        label="FACE MASK BLUR",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                        visible=True
                    )
                mask_types = gr.CheckboxGroup(
                    label="FACE MASK TYPES",
                    choices=["box", "occlusion", "area", "region"],
                    value=["box"]
                )
                
                # Dynamic Mask Areas Checkbox Group (Visible when 'area' is checked in mask_types)
                mask_areas = gr.CheckboxGroup(
                    label="FACE MASK AREAS (Area Mode)",
                    choices=["upper-face", "lower-face", "mouth", "eyes", "nose"],
                    value=["upper-face", "lower-face", "mouth", "eyes", "nose"],
                    visible=False
                )

                # Dynamic Mask Regions Checkbox Group (Visible when 'region' is checked in mask_types)
                mask_regions = gr.CheckboxGroup(
                    label="FACE MASK REGIONS (Semantic Region Mode)",
                    choices=["skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye", "nose", "mouth", "upper-lip", "lower-lip"],
                    value=["skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye", "nose", "mouth", "upper-lip", "lower-lip"],
                    visible=False
                )

                # Box Padding Controls (Visible when 'box' is checked)
                with gr.Group(visible=True) as mask_padding_group:
                    with gr.Row():
                        mask_padding_top = gr.Slider(label="FACE MASK PADDING TOP", minimum=0, maximum=100, value=0, step=1)
                        mask_padding_right = gr.Slider(label="FACE MASK PADDING RIGHT", minimum=0, maximum=100, value=0, step=1)
                    with gr.Row():
                        mask_padding_bottom = gr.Slider(label="FACE MASK PADDING BOTTOM", minimum=0, maximum=100, value=0, step=1)
                        mask_padding_left = gr.Slider(label="FACE MASK PADDING LEFT", minimum=0, maximum=100, value=0, step=1)

            # Face Detector Box
            with gr.Blocks():
                with gr.Row():
                    detector_model = gr.Dropdown(
                        label="FACE DETECTOR MODEL",
                        choices=["yolo_face", "scrfd", "retinaface", "yunet"],
                        value="yolo_face"
                    )
                    detector_size = gr.Dropdown(
                        label="FACE DETECTOR SIZE",
                        choices=["320x320", "480x480", "512x512", "640x640", "768x768", "960x960", "1024x1024"],
                        value="640x640"
                    )
                with gr.Row():
                    margin_top = gr.Slider(label="FACE DETECTOR MARGIN TOP", minimum=0, maximum=100, value=0, step=1)
                    margin_right = gr.Slider(label="FACE DETECTOR MARGIN RIGHT", minimum=0, maximum=100, value=0, step=1)
                with gr.Row():
                    margin_bottom = gr.Slider(label="FACE DETECTOR MARGIN BOTTOM", minimum=0, maximum=100, value=0, step=1)
                    margin_left = gr.Slider(label="FACE DETECTOR MARGIN LEFT", minimum=0, maximum=100, value=0, step=1)
                detector_angles = gr.CheckboxGroup(
                    label="FACE DETECTOR ANGLES",
                    choices=[0, 90, 180, 270],
                    value=[0]
                )
                detector_score = gr.Slider(
                    label="FACE DETECTOR SCORE",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05
                )

            # Face Landmarker Box
            with gr.Blocks():
                landmarker_model = gr.Dropdown(
                    label="FACE LANDMARKER MODEL",
                    choices=["2dfan4", "peppa_wutz"],
                    value="2dfan4"
                )
                landmarker_score = gr.Slider(
                    label="FACE LANDMARKER SCORE",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05
                )

    # -------------------------------------------------------------
    # Dynamic Events & Callbacks
    # -------------------------------------------------------------
    tgt_file.change(
        fn=on_target_change,
        inputs=[tgt_file],
        outputs=[out_image, out_video, preview_frame_slider, trim_end]
    )

    # Dynamic Mask Types handler: reveals area / region / occlusion / box padding options on click!
    def update_mask_types_visibility(active_mask_types: List[str]):
        types = active_mask_types or []
        has_box = "box" in types
        has_occlusion = "occlusion" in types
        has_area = "area" in types
        has_region = "region" in types
        return (
            gr.update(visible=has_occlusion),  # occluder_model
            gr.update(visible=has_box),        # mask_blur
            gr.update(visible=has_area),       # mask_areas
            gr.update(visible=has_region),     # mask_regions
            gr.update(visible=has_box)         # mask_padding_group
        )

    mask_types.change(
        fn=update_mask_types_visibility,
        inputs=[mask_types],
        outputs=[occluder_model, mask_blur, mask_areas, mask_regions, mask_padding_group]
    )

    # Dynamic Face Selector Mode handler: reveals reference face gallery on reference mode
    def update_face_selector_mode_visibility(mode: str):
        is_ref = (mode == "reference")
        return (
            gr.update(visible=is_ref),  # reference_container
            gr.update(visible=is_ref)   # reference_face_distance
        )

    face_selector_mode.change(
        fn=update_face_selector_mode_visibility,
        inputs=[face_selector_mode],
        outputs=[reference_container, reference_face_distance]
    )

    ref_detector_inputs = [
        tgt_file,
        preview_frame_slider,
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
            return selected_idx, f"🎯 Selected Face #{selected_idx} as active reference face."
        except Exception:
            return 0, "🎯 Selected Face #0 as active reference face."

    ref_gallery.select(
        fn=on_reference_gallery_select,
        inputs=None,
        outputs=[face_selector_position, ref_status]
    )

    # Update gallery on target / order / detector / landmarker change
    for comp in [tgt_file, face_selector_order, detector_model, detector_size, detector_angles, landmarker_model]:
        comp.change(
            fn=update_reference_face_gallery,
            inputs=ref_detector_inputs,
            outputs=[ref_gallery, ref_status]
        )

    for comp in [preview_frame_slider, detector_score, landmarker_score, margin_top, margin_right, margin_bottom, margin_left]:
        comp.release(
            fn=update_reference_face_gallery,
            inputs=ref_detector_inputs,
            outputs=[ref_gallery, ref_status]
        )

    preview_inputs = [
        src_files,
        tgt_file,
        preview_frame_slider,
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
        mask_regions,
        preview_mode
    ]

    # Live preview triggers
    for comp in [src_files, tgt_file, preview_mode, swapper_model, pixel_boost, detector_model, detector_size, detector_angles, landmarker_model, face_selector_mode, face_selector_order, mask_types, mask_areas, mask_regions, occluder_model]:
        comp.change(
            fn=preview_swap_frame,
            inputs=preview_inputs,
            outputs=[preview_output]
        )

    for comp in [preview_frame_slider, swapper_weight, detector_score, landmarker_score, reference_face_distance, mask_blur, mask_padding_top, mask_padding_right, mask_padding_bottom, mask_padding_left, margin_top, margin_right, margin_bottom, margin_left]:
        comp.release(
            fn=preview_swap_frame,
            inputs=preview_inputs,
            outputs=[preview_output]
        )

    ref_gallery.select(
        fn=preview_swap_frame,
        inputs=preview_inputs,
        outputs=[preview_output]
    )

    all_batch_inputs = [
        src_files,
        tgt_file,
        preview_frame_slider,
        output_path_input,
        output_video_fps,
        output_video_quality,
        output_video_encoder,
        output_video_preset,
        trim_start,
        trim_end,
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

    def on_source_change(files):
        if not files:
            return gr.update(value=None, visible=False)
        p = files[0] if isinstance(files, list) else files
        if isinstance(p, dict):
            p = p.get('name') or p.get('path')
        if p and os.path.isfile(str(p)):
            return gr.update(value=str(p), visible=True)
        return gr.update(value=None, visible=False)

    def on_target_change(file_path):
        if not file_path:
            return (
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(visible=False, value=0, maximum=100),
                gr.update(visible=False, value=0, maximum=100),
                gr.update(visible=False, value=100, maximum=100),
                gr.update(value=0)
            )
        p = file_path.get('name') if isinstance(file_path, dict) else file_path
        if not p or not os.path.isfile(str(p)):
            return (
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(visible=False, value=0, maximum=100),
                gr.update(visible=False, value=0, maximum=100),
                gr.update(visible=False, value=100, maximum=100),
                gr.update(value=0)
            )
        _, ext = safe_filename(str(p))
        if ext in IMAGE_EXTS:
            return (
                gr.update(value=str(p), visible=True),
                gr.update(value=None, visible=False),
                gr.update(visible=False, value=0, maximum=1),
                gr.update(visible=False, value=0, maximum=1),
                gr.update(visible=False, value=1, maximum=1),
                gr.update(value=0)
            )
        elif ext in VIDEO_EXTS:
            total_frames, fps, _, _ = get_video_info(str(p))
            max_f = max(0, total_frames - 1)
            detected_fps = round(fps, 2) if fps > 0 else 30.0
            return (
                gr.update(value=None, visible=False),
                gr.update(value=str(p), visible=True),
                gr.update(visible=True, value=0, minimum=0, maximum=max_f, step=1),
                gr.update(visible=True, value=0, minimum=0, maximum=max_f, step=1),
                gr.update(visible=True, value=max_f, minimum=0, maximum=max_f, step=1),
                gr.update(value=detected_fps)
            )
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(visible=False, value=0, maximum=100),
            gr.update(visible=False, value=0, maximum=100),
            gr.update(visible=False, value=100, maximum=100),
            gr.update(value=0)
        )

    src_files.change(
        fn=on_source_change,
        inputs=[src_files],
        outputs=[src_image_preview]
    )

    tgt_file.change(
        fn=on_target_change,
        inputs=[tgt_file],
        outputs=[tgt_image_preview, tgt_video_preview, preview_frame_slider, trim_start, trim_end, output_video_fps]
    )

    run_event = run_btn.click(
        fn=run_batch_swap,
        inputs=all_batch_inputs,
        outputs=[out_image, out_video, status_box]
    )

    def on_clear():
        stop_process()
        return None, None, "Processing cancelled / outputs cleared."

    clear_btn.click(
        fn=on_clear,
        inputs=None,
        outputs=[out_image, out_video, status_box],
        cancels=[run_event]
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MonoFace Studio")
    parser.add_argument("--share", type=bool, default=False, help="Share the app")
    parser.add_argument("--inbrowser", type=bool, default=False, help="Open in browser")
    args = parser.parse_args()
    
    providers = onnxruntime.get_available_providers()
    print(f"[ONNX] Active ONNX Runtime Execution Providers: {providers}")
    if "CUDAExecutionProvider" not in providers:
        print("[WARNING] CUDAExecutionProvider not detected! Running on CPU. For 10-30x faster GPU inference, install: pip install onnxruntime-gpu")
    
    print("[INIT] Preloading face analyser models (Detector, Landmarker, Fan 68/5, Recognizer)...")
    try:
        preload_face_analyser()
        print("[INIT] Face analyser models preloaded successfully!")
    except Exception as e:
        print(f"[INIT] Note: Face analyser models will load on first inference ({e})")

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
            theme=get_facefusion_theme(),
            css=get_facefusion_css()
        )
    except OSError:
        print("[LAUNCH] Port 7860 occupied. Falling back to automatically selected open port...")
        demo.launch(
            share=args.share,
            inbrowser=args.inbrowser,
            theme=get_facefusion_theme(),
            css=get_facefusion_css()
        )
