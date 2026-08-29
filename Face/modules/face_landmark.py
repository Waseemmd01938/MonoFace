from typing import List, Optional, Tuple, Dict, Any, Union
import os
import cv2
import numpy as np
import onnxruntime

from Face.modules.face_helper import (
    BoundingBox,
    FaceLandmark5,
    FaceLandmark68,
    Score,
    Angle,
    VisionFrame,
    Matrix,
    create_rotation_matrix_and_size,
    warp_face_by_translation,
    transform_points,
    conditional_optimize_contrast,
    estimate_matrix_by_face_landmark_5,
    convert_to_face_landmark_5,
    estimate_face_angle,
    ensure_model_exists
)
from Face.modules.model_store import get_inference_session, get_default_providers

# Canonical MediaPipe 468-mesh to 68-point iBUG/Dlib landmark indices mapping
MEDIAPIPE_TO_68_INDICES = [
    # Jawline (0-16)
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    # Right Eyebrow (17-21)
    70, 63, 105, 66, 107,
    # Left Eyebrow (22-26)
    336, 296, 334, 293, 300,
    # Nose Bridge & Tip (27-35)
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    # Right Eye (36-41)
    33, 160, 158, 133, 153, 144,
    # Left Eye (42-47)
    362, 385, 387, 263, 373, 380,
    # Outer Mouth (48-59)
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # Inner Mouth (60-67)
    78, 81, 13, 311, 308, 402, 14, 178
]

LANDMARK_MODELS = {

    '2dfan4': {
        'file': '2dfan4.onnx',
        'tag': 'models-3.0.0',
        'size': (256, 256)
    },
    'peppa_wutz': {
        'file': 'peppa_wutz.onnx',
        'tag': 'models-3.0.0',
        'size': (256, 256)
    },
    'fan_68_5': {
        'file': 'fan_68_5.onnx',
        'tag': 'models-3.0.0'
    },
    'mediapipe': {
        'file': None,
        'tag': None,
        'size': None
    }
}


class FaceLandmarker:
    def __init__(
        self,
        model_name: str = '2dfan4',
        providers: Optional[List[str]] = None,
        model_path: Optional[str] = None
    ):
        self.model_name = model_name.lower()
        if self.model_name not in LANDMARK_MODELS:
            raise ValueError(f"Unsupported landmarker model: {model_name}. Supported: {list(LANDMARK_MODELS.keys())}")

        self.providers = providers if providers is not None else get_default_providers()
        self._mp_mesh = None

        if self.model_name == 'mediapipe':
            self.model_file = None
            self.session = None
        else:
            cfg = LANDMARK_MODELS[self.model_name]
            self.model_file = model_path or ensure_model_exists(cfg['file'], cfg['tag'])
            self.session = get_inference_session(self.model_file, providers=self.providers)

        # Preload fan_68_5 session upfront for instant landmark conversion
        fan_cfg = LANDMARK_MODELS['fan_68_5']
        fan_path = ensure_model_exists(fan_cfg['file'], fan_cfg['tag'])
        self._fan_68_5_session: onnxruntime.InferenceSession = get_inference_session(fan_path, providers=self.providers)

    def preload(self) -> None:
        """Warms up landmark and fan_68_5 sessions."""
        if self.model_name == 'mediapipe':
            try:
                import mediapipe as mp
                if self._mp_mesh is None:
                    self._mp_mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5
                    )
            except Exception:
                pass
        elif self.session is not None:
            _ = self.session.get_inputs()
        if self._fan_68_5_session is not None:
            _ = self._fan_68_5_session.get_inputs()


    def detect_landmarks(
        self,
        vision_frame: VisionFrame,
        bounding_box: BoundingBox,
        face_angle: Angle = 0
    ) -> Tuple[FaceLandmark68, Score]:
        """
        Detects 68 facial landmarks from an image frame and face bounding box.
        Returns:
            (landmark_68: np.ndarray shape (68, 2), score: float)
        """
        if self.model_name == '2dfan4':
            return self._detect_with_2dfan4(vision_frame, bounding_box, face_angle)
        elif self.model_name == 'peppa_wutz':
            return self._detect_with_peppa_wutz(vision_frame, bounding_box, face_angle)
        elif self.model_name == 'mediapipe':
            return self._detect_with_mediapipe(vision_frame, bounding_box, face_angle)
        elif self.model_name == 'fan_68_5':
            # Needs 5 landmarks first
            raise ValueError("fan_68_5 converts from 5-point landmarks. Use estimate_landmark_68_from_5(face_landmark_5)")
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _detect_with_2dfan4(
        self,
        temp_vision_frame: VisionFrame,
        bounding_box: BoundingBox,
        face_angle: Angle
    ) -> Tuple[FaceLandmark68, Score]:
        model_size = LANDMARK_MODELS['2dfan4']['size']
        box_size = np.subtract(bounding_box[2:], bounding_box[:2]).max()
        scale = 195.0 / max(float(box_size), 1.0)
        box_center = np.add(bounding_box[2:], bounding_box[:2])
        translation = (model_size[0] - box_center * scale) * 0.5

        rotation_matrix, rotation_size = create_rotation_matrix_and_size(face_angle, model_size)
        crop_vision_frame, affine_matrix = warp_face_by_translation(temp_vision_frame, translation, scale, model_size)
        crop_vision_frame = cv2.warpAffine(crop_vision_frame, rotation_matrix, rotation_size)
        crop_vision_frame = conditional_optimize_contrast(crop_vision_frame)

        input_tensor = crop_vision_frame.transpose(2, 0, 1).astype(np.float32) / 255.0

        output = self.session.run(None, {'input': [input_tensor]})
        face_landmark_68_raw, face_heatmap = output[0], output[1]

        # Decode coordinates (from 64x64 grid to 256x256 model size)
        face_landmark_68 = face_landmark_68_raw[:, :, :2][0] / 64.0 * model_size[0]
        # Invert rotation and translation
        face_landmark_68 = transform_points(face_landmark_68, cv2.invertAffineTransform(rotation_matrix))
        face_landmark_68 = transform_points(face_landmark_68, cv2.invertAffineTransform(affine_matrix))

        # Score calculation from heatmap peaks
        score_68 = np.mean(np.amax(face_heatmap, axis=(2, 3)))
        score_normalized = float(np.interp(score_68, [0, 0.9], [0, 1]))
        return face_landmark_68, score_normalized

    def _detect_with_peppa_wutz(
        self,
        temp_vision_frame: VisionFrame,
        bounding_box: BoundingBox,
        face_angle: Angle
    ) -> Tuple[FaceLandmark68, Score]:
        model_size = LANDMARK_MODELS['peppa_wutz']['size']
        box_size = np.subtract(bounding_box[2:], bounding_box[:2]).max()
        scale = 195.0 / max(float(box_size), 1.0)
        box_center = np.add(bounding_box[2:], bounding_box[:2])
        translation = (model_size[0] - box_center * scale) * 0.5

        rotation_matrix, rotation_size = create_rotation_matrix_and_size(face_angle, model_size)
        crop_vision_frame, affine_matrix = warp_face_by_translation(temp_vision_frame, translation, scale, model_size)
        crop_vision_frame = cv2.warpAffine(crop_vision_frame, rotation_matrix, rotation_size)
        crop_vision_frame = conditional_optimize_contrast(crop_vision_frame)

        input_tensor = crop_vision_frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        prediction = self.session.run(None, {'input': input_tensor})[0]
        pts_with_conf = prediction.reshape(-1, 3)

        face_landmark_68 = pts_with_conf[:, :2] / 64.0 * model_size[0]
        face_landmark_68 = transform_points(face_landmark_68, cv2.invertAffineTransform(rotation_matrix))
        face_landmark_68 = transform_points(face_landmark_68, cv2.invertAffineTransform(affine_matrix))

        score_68 = pts_with_conf[:, 2].mean()
        score_normalized = float(np.interp(score_68, [0, 0.95], [0, 1]))
        return face_landmark_68, score_normalized

    def _detect_with_mediapipe(
        self,
        temp_vision_frame: VisionFrame,
        bounding_box: BoundingBox,
        face_angle: Angle
    ) -> Tuple[FaceLandmark68, Score]:
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError("MediaPipe is required for 'mediapipe' landmarker. Please install via: pip install mediapipe")

        if self._mp_mesh is None:
            self._mp_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )

        h, w = temp_vision_frame.shape[:2]
        x1, y1, x2, y2 = bounding_box
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        pad_x = bw * 0.25
        pad_y = bh * 0.25
        crop_x1 = max(0, int(x1 - pad_x))
        crop_y1 = max(0, int(y1 - pad_y))
        crop_x2 = min(w, int(x2 + pad_x))
        crop_y2 = min(h, int(y2 + pad_y))

        crop = temp_vision_frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return np.zeros((68, 2), dtype=np.float32), 0.0

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ch, cw = crop.shape[:2]
        results = self._mp_mesh.process(crop_rgb)

        if not results.multi_face_landmarks:
            # Fallback to full frame if crop detection misses
            full_rgb = cv2.cvtColor(temp_vision_frame, cv2.COLOR_BGR2RGB)
            results = self._mp_mesh.process(full_rgb)
            if not results.multi_face_landmarks:
                return np.zeros((68, 2), dtype=np.float32), 0.0
            raw_lms = results.multi_face_landmarks[0].landmark
            pts_468 = np.array([[p.x * w, p.y * h] for p in raw_lms], dtype=np.float32)
        else:
            raw_lms = results.multi_face_landmarks[0].landmark
            pts_468 = np.array([[p.x * cw + crop_x1, p.y * ch + crop_y1] for p in raw_lms], dtype=np.float32)

        landmark_68 = pts_468[MEDIAPIPE_TO_68_INDICES]
        return landmark_68.astype(np.float32), 0.99

    def estimate_landmark_68_from_5(self, face_landmark_5: FaceLandmark5) -> FaceLandmark68:
        """Estimates full 68-point landmarks given 5-point face landmarks using preloaded fan_68_5 model."""


        affine_matrix = estimate_matrix_by_face_landmark_5(face_landmark_5, 'ffhq_512', (1, 1))
        norm_landmark_5 = cv2.transform(face_landmark_5.reshape(1, -1, 2).astype(np.float32), affine_matrix).reshape(-1, 2)

        out = self._fan_68_5_session.run(None, {'input': [norm_landmark_5.astype(np.float32)]})[0][0]
        out_transformed = cv2.transform(out.reshape(1, -1, 2).astype(np.float32), cv2.invertAffineTransform(affine_matrix)).reshape(-1, 2)
        return out_transformed

