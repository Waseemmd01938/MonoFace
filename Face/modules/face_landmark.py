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

        cfg = LANDMARK_MODELS[self.model_name]
        self.model_file = model_path or ensure_model_exists(cfg['file'], cfg['tag'])
        self.session = get_inference_session(self.model_file, providers=self.providers)

        # Preload fan_68_5 session upfront for instant landmark conversion
        fan_cfg = LANDMARK_MODELS['fan_68_5']
        fan_path = ensure_model_exists(fan_cfg['file'], fan_cfg['tag'])
        self._fan_68_5_session: onnxruntime.InferenceSession = get_inference_session(fan_path, providers=self.providers)

    def preload(self) -> None:
        """Warms up landmark and fan_68_5 sessions."""
        _ = self.session.get_inputs()
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

    def estimate_landmark_68_from_5(self, face_landmark_5: FaceLandmark5) -> FaceLandmark68:
        """Estimates full 68-point landmarks given 5-point face landmarks using preloaded fan_68_5 model."""


        affine_matrix = estimate_matrix_by_face_landmark_5(face_landmark_5, 'ffhq_512', (1, 1))
        norm_landmark_5 = cv2.transform(face_landmark_5.reshape(1, -1, 2).astype(np.float32), affine_matrix).reshape(-1, 2)

        out = self._fan_68_5_session.run(None, {'input': [norm_landmark_5.astype(np.float32)]})[0][0]
        out_transformed = cv2.transform(out.reshape(1, -1, 2).astype(np.float32), cv2.invertAffineTransform(affine_matrix)).reshape(-1, 2)
        return out_transformed


def detect_face_landmark(
    vision_frame: VisionFrame,
    bounding_box: BoundingBox,
    face_angle: Angle = 0,
    model_name: str = '2dfan4'
) -> Tuple[FaceLandmark68, Score]:
    """Helper function to extract 68-point face landmarks."""
    landmarker = FaceLandmarker(model_name=model_name)
    return landmarker.detect_landmarks(vision_frame, bounding_box, face_angle)
