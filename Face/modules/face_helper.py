from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple, Union
import os
import urllib.request
import cv2
import numpy as np

# Type aliases
BoundingBox = np.ndarray  # [x1, y1, x2, y2]
FaceLandmark5 = np.ndarray  # (5, 2)
FaceLandmark68 = np.ndarray  # (68, 2)
Score = float
Angle = int
Matrix = np.ndarray
Points = np.ndarray
VisionFrame = np.ndarray
Mask = np.ndarray

# Canonical warp templates for face alignment
WARP_TEMPLATES: Dict[str, np.ndarray] = {
    'arcface_112_v1': np.array([
        [0.35473214, 0.45658929],
        [0.64526786, 0.45658929],
        [0.50000000, 0.61154464],
        [0.37913393, 0.77687500],
        [0.62086607, 0.77687500]
    ], dtype=np.float32),
    'arcface_112_v2': np.array([
        [0.34191607, 0.46157411],
        [0.65653393, 0.45983393],
        [0.50022500, 0.64050536],
        [0.37097589, 0.82469196],
        [0.63151696, 0.82325089]
    ], dtype=np.float32),
    'arcface_128': np.array([
        [0.36167656, 0.40387734],
        [0.63696719, 0.40235469],
        [0.50019687, 0.56044219],
        [0.38710391, 0.72160547],
        [0.61507734, 0.72034453]
    ], dtype=np.float32),
    'dfl_whole_face': np.array([
        [0.35342266, 0.39285716],
        [0.62797622, 0.39285716],
        [0.48660713, 0.54017860],
        [0.38839287, 0.68750011],
        [0.59821427, 0.68750011]
    ], dtype=np.float32),
    'ffhq_512': np.array([
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465]
    ], dtype=np.float32),
    'mtcnn_512': np.array([
        [0.36562865, 0.46733799],
        [0.63305391, 0.46585885],
        [0.50019127, 0.61942959],
        [0.39032951, 0.77598822],
        [0.61178945, 0.77476328]
    ], dtype=np.float32),
    'styleganex_384': np.array([
        [0.42353745, 0.52289879],
        [0.57725008, 0.52319972],
        [0.50123859, 0.61331904],
        [0.43364461, 0.68337652],
        [0.57015325, 0.68306005]
    ], dtype=np.float32)
}

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Models')


def ensure_model_exists(file_name: str, download_tag: str = 'models-3.0.0', models_dir: Optional[str] = None) -> str:
    """Ensures that the requested ONNX model is available locally, downloading if necessary."""
    target_dir = models_dir or DEFAULT_MODELS_DIR
    os.makedirs(target_dir, exist_ok=True)
    model_path = os.path.join(target_dir, file_name)

    if os.path.isfile(model_path) and os.path.getsize(model_path) > 1024:
        return model_path

    urls = [
        f"https://github.com/facefusion/facefusion-assets/releases/download/{download_tag}/{file_name}",
        f"https://huggingface.co/facefusion/facefusion-assets/resolve/main/{download_tag}/{file_name}"
    ]

    for url in urls:
        try:
            print(f"Downloading model {file_name} from {url}...")
            urllib.request.urlretrieve(url, model_path)
            if os.path.isfile(model_path) and os.path.getsize(model_path) > 1024:
                print(f"Successfully downloaded {file_name}")
                return model_path
        except Exception as e:
            print(f"Failed to download from {url}: {e}")

    if not (os.path.isfile(model_path) and os.path.getsize(model_path) > 1024):
        # Also check relative facefusion repository if available locally
        possible_local = os.path.join("D:\\waseem\\ML\\facefusion\\.assets\\models", file_name)
        if os.path.isfile(possible_local):
            return possible_local
        raise FileNotFoundError(f"Could not locate or download model: {file_name}")

    return model_path


def estimate_matrix_by_face_landmark_5(face_landmark_5: FaceLandmark5, warp_template: str, crop_size: Tuple[int, int]) -> Matrix:
    """Estimates affine transformation matrix from 5 face landmarks to a template."""
    template = WARP_TEMPLATES[warp_template] * np.array(crop_size, dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(face_landmark_5.astype(np.float32), template, method=cv2.RANSAC, ransacReprojThreshold=100)
    return matrix


def warp_face_by_face_landmark_5(temp_vision_frame: VisionFrame, face_landmark_5: FaceLandmark5, warp_template: str, crop_size: Tuple[int, int]) -> Tuple[VisionFrame, Matrix]:
    """Warps a face crop using 5 landmarks to canonical alignment."""
    affine_matrix = estimate_matrix_by_face_landmark_5(face_landmark_5, warp_template, crop_size)
    crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size, borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    return crop_vision_frame, affine_matrix


def warp_face_by_bounding_box(temp_vision_frame: VisionFrame, bounding_box: BoundingBox, crop_size: Tuple[int, int]) -> Tuple[VisionFrame, Matrix]:
    """Warps a face region given its bounding box."""
    source_points = np.array([[bounding_box[0], bounding_box[1]], [bounding_box[2], bounding_box[1]], [bounding_box[0], bounding_box[3]]], dtype=np.float32)
    target_points = np.array([[0, 0], [crop_size[0], 0], [0, crop_size[1]]], dtype=np.float32)
    affine_matrix = cv2.getAffineTransform(source_points, target_points)
    interpolation = cv2.INTER_AREA if (bounding_box[2] - bounding_box[0] > crop_size[0] or bounding_box[3] - bounding_box[1] > crop_size[1]) else cv2.INTER_LINEAR
    crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size, flags=interpolation)
    return crop_vision_frame, affine_matrix


def warp_face_by_translation(temp_vision_frame: VisionFrame, translation: np.ndarray, scale: float, crop_size: Tuple[int, int]) -> Tuple[VisionFrame, Matrix]:
    """Warps a face by scale and translation."""
    affine_matrix = np.array([[scale, 0, translation[0]], [0, scale, translation[1]]], dtype=np.float32)
    crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size)
    return crop_vision_frame, affine_matrix


def paste_back(temp_vision_frame: VisionFrame, crop_vision_frame: VisionFrame, crop_vision_mask: Mask, affine_matrix: Matrix) -> VisionFrame:
    """Pastes an aligned processed face crop back onto the original frame."""
    paste_bounding_box, paste_matrix = calculate_paste_area(temp_vision_frame, crop_vision_frame, affine_matrix)
    x1, y1, x2, y2 = paste_bounding_box
    paste_width = x2 - x1
    paste_height = y2 - y1

    if paste_width <= 0 or paste_height <= 0:
        return temp_vision_frame

    inverse_vision_mask = cv2.warpAffine(crop_vision_mask, paste_matrix, (paste_width, paste_height)).clip(0, 1)
    if len(inverse_vision_mask.shape) == 2:
        inverse_vision_mask = np.expand_dims(inverse_vision_mask, axis=-1)

    inverse_vision_frame = cv2.warpAffine(crop_vision_frame, paste_matrix, (paste_width, paste_height), borderMode=cv2.BORDER_REPLICATE)
    out_frame = temp_vision_frame.copy()
    paste_vision_frame = out_frame[y1:y2, x1:x2]
    paste_vision_frame = paste_vision_frame * (1 - inverse_vision_mask) + inverse_vision_frame * inverse_vision_mask
    out_frame[y1:y2, x1:x2] = paste_vision_frame.astype(temp_vision_frame.dtype)
    return out_frame


def calculate_paste_area(temp_vision_frame: VisionFrame, crop_vision_frame: VisionFrame, affine_matrix: Matrix) -> Tuple[BoundingBox, Matrix]:
    temp_height, temp_width = temp_vision_frame.shape[:2]
    crop_height, crop_width = crop_vision_frame.shape[:2]
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)
    crop_points = np.array([[0, 0], [crop_width, 0], [crop_width, crop_height], [0, crop_height]], dtype=np.float32)
    paste_region_points = transform_points(crop_points, inverse_matrix)
    paste_region_point_min = np.floor(paste_region_points.min(axis=0)).astype(int)
    paste_region_point_max = np.ceil(paste_region_points.max(axis=0)).astype(int)
    x1, y1 = np.clip(paste_region_point_min, 0, [temp_width, temp_height])
    x2, y2 = np.clip(paste_region_point_max, 0, [temp_width, temp_height])
    paste_bounding_box = np.array([x1, y1, x2, y2])
    paste_matrix = inverse_matrix.copy()
    paste_matrix[0, 2] -= x1
    paste_matrix[1, 2] -= y1
    return paste_bounding_box, paste_matrix


@lru_cache()
def create_static_anchors(feature_stride: int, anchor_total: int, stride_height: int, stride_width: int) -> np.ndarray:
    """Generates detection anchor coordinates for feature maps."""
    x, y = np.mgrid[:stride_width, :stride_height]
    anchors = np.stack((y, x), axis=-1)
    anchors = (anchors * feature_stride).reshape((-1, 2))
    anchors = np.stack([anchors] * anchor_total, axis=1).reshape((-1, 2))
    return anchors


def create_rotation_matrix_and_size(angle: Angle, size: Tuple[int, int]) -> Tuple[Matrix, Tuple[int, int]]:
    """Creates a 2D rotation matrix and calculated bounding dimensions."""
    rotation_matrix = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
    rotation_size = np.dot(np.abs(rotation_matrix[:, :2]), size)
    rotation_matrix[:, -1] += (rotation_size - size) * 0.5
    rotation_size = (int(rotation_size[0]), int(rotation_size[1]))
    return rotation_matrix, rotation_size


def normalize_bounding_box(bounding_box: BoundingBox) -> BoundingBox:
    x1, y1, x2, y2 = bounding_box
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return np.array([x1, y1, x2, y2])


def create_bounding_box(face_landmark_68: FaceLandmark68) -> BoundingBox:
    x1, y1 = np.min(face_landmark_68, axis=0)
    x2, y2 = np.max(face_landmark_68, axis=0)
    return normalize_bounding_box(np.array([x1, y1, x2, y2]))


def transform_points(points: Points, matrix: Matrix) -> Points:
    points = points.reshape(-1, 1, 2).astype(np.float32)
    transformed = cv2.transform(points, matrix)
    return transformed.reshape(-1, 2)


def transform_bounding_box(bounding_box: BoundingBox, matrix: Matrix) -> BoundingBox:
    points = np.array([
        [bounding_box[0], bounding_box[1]],
        [bounding_box[2], bounding_box[1]],
        [bounding_box[2], bounding_box[3]],
        [bounding_box[0], bounding_box[3]]
    ], dtype=np.float32)
    transformed = transform_points(points, matrix)
    x1, y1 = np.min(transformed, axis=0)
    x2, y2 = np.max(transformed, axis=0)
    return normalize_bounding_box(np.array([x1, y1, x2, y2]))


def distance_to_bounding_box(points: Points, distance: np.ndarray) -> BoundingBox:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.column_stack([x1, y1, x2, y2])


def distance_to_face_landmark_5(points: Points, distance: np.ndarray) -> FaceLandmark5:
    x = points[:, 0::2] + distance[:, 0::2]
    y = points[:, 1::2] + distance[:, 1::2]
    return np.stack((x, y), axis=-1)


def scale_face_landmark_5(face_landmark_5: FaceLandmark5, scale: float) -> FaceLandmark5:
    face_landmark_5_scale = face_landmark_5 - face_landmark_5[2]
    face_landmark_5_scale *= scale
    face_landmark_5_scale += face_landmark_5[2]
    return face_landmark_5_scale


def convert_to_face_landmark_5(face_landmark_68: FaceLandmark68) -> FaceLandmark5:
    """Converts 68-point landmarks to standard 5-point facial landmarks."""
    return np.array([
        np.mean(face_landmark_68[36:42], axis=0),  # left eye
        np.mean(face_landmark_68[42:48], axis=0),  # right eye
        face_landmark_68[30],                     # nose tip
        face_landmark_68[48],                     # left mouth corner
        face_landmark_68[54]                      # right mouth corner
    ], dtype=np.float32)


def estimate_face_angle(face_landmark_68: FaceLandmark68) -> Angle:
    """Estimates face rotation angle (0, 90, 180, 270) based on landmarks."""
    x1, y1 = face_landmark_68[0]
    x2, y2 = face_landmark_68[16]
    theta = np.arctan2(y2 - y1, x2 - x1)
    theta = np.degrees(theta) % 360
    angles = np.linspace(0, 360, 5)
    index = np.argmin(np.abs(angles - theta))
    return int(angles[index] % 360)


def apply_nms(bounding_boxes: List[BoundingBox], scores: List[Score], score_threshold: float, nms_threshold: float) -> Sequence[int]:
    """Applies Non-Maximum Suppression to filter overlapping detections."""
    if not bounding_boxes:
        return []
    bounding_boxes_norm = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for (x1, y1, x2, y2) in bounding_boxes]
    keep_indices = cv2.dnn.NMSBoxes(bounding_boxes_norm, [float(s) for s in scores], score_threshold=score_threshold, nms_threshold=nms_threshold)
    if isinstance(keep_indices, tuple) or isinstance(keep_indices, list):
        return [idx if isinstance(idx, int) else idx[0] for idx in keep_indices]
    if hasattr(keep_indices, 'flatten'):
        return keep_indices.flatten().tolist()
    return []


def conditional_optimize_contrast(crop_vision_frame: VisionFrame) -> VisionFrame:
    """Applies adaptive contrast enhancement CLAHE in low light conditions."""
    lab = cv2.cvtColor(crop_vision_frame, cv2.COLOR_BGR2Lab)
    if np.mean(lab[:, :, 0]) < 30:
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return crop_vision_frame


def implode_pixel_boost(crop_vision_frame: VisionFrame, pixel_boost_total: int, model_size: Tuple[int, int]) -> np.ndarray:
    """Deconstructs high-resolution crop frame into tiles matching the model input size."""
    pixel_boost_vision_frame = crop_vision_frame.reshape(model_size[0], pixel_boost_total, model_size[1], pixel_boost_total, 3)
    pixel_boost_vision_frame = pixel_boost_vision_frame.transpose(1, 3, 0, 2, 4).reshape(pixel_boost_total ** 2, model_size[0], model_size[1], 3)
    return pixel_boost_vision_frame


def explode_pixel_boost(temp_vision_frames: List[VisionFrame], pixel_boost_total: int, model_size: Tuple[int, int], pixel_boost_size: Tuple[int, int]) -> VisionFrame:
    """Reassembles swapped model tiles into high-resolution swapped face frame."""
    crop_vision_frame = np.stack(temp_vision_frames).reshape(pixel_boost_total, pixel_boost_total, model_size[0], model_size[1], 3)
    crop_vision_frame = crop_vision_frame.transpose(2, 0, 3, 1, 4).reshape(pixel_boost_size[0], pixel_boost_size[1], 3)
    return crop_vision_frame

