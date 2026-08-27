from typing import List, Optional, Sequence, Tuple, Union, Dict, Any
import os
import cv2
import numpy as np
import onnxruntime

from Face.modules.face_helper import (
    BoundingBox,
    FaceLandmark5,
    Score,
    Angle,
    VisionFrame,
    create_rotation_matrix_and_size,
    create_static_anchors,
    distance_to_bounding_box,
    distance_to_face_landmark_5,
    normalize_bounding_box,
    transform_bounding_box,
    transform_points,
    apply_nms,
    ensure_model_exists
)

from Face.modules.model_store import get_inference_session, get_default_providers

MODEL_CONFIGS = {

    'yolo_face': {
        'file': 'yoloface_8n.onnx',
        'tag': 'models-3.0.0',
        'input_size': (640, 640),
        'normalize_range': [0, 1]
    },
    'scrfd': {
        'file': 'scrfd_2.5g.onnx',
        'tag': 'models-3.0.0',
        'input_size': (640, 640),
        'normalize_range': [-1, 1]
    },
    'retinaface': {
        'file': 'retinaface_10g.onnx',
        'tag': 'models-3.0.0',
        'input_size': (640, 640),
        'normalize_range': [-1, 1]
    },
    'yunet': {
        'file': 'yunet_2023_mar.onnx',
        'tag': 'models-3.4.0',
        'input_size': (640, 640),
        'normalize_range': [0, 255]
    }
}


class FaceDetector:
    def __init__(
        self,
        model_name: str = 'yolo_face',
        score_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        angles: Optional[List[Angle]] = None,
        margin: Tuple[int, int, int, int] = (0, 0, 0, 0),
        providers: Optional[List[str]] = None,
        model_path: Optional[str] = None
    ):
        self.model_name = model_name.lower()
        if self.model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported face detector model: {model_name}. Supported: {list(MODEL_CONFIGS.keys())}")

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.angles = angles or [0]
        self.margin = margin

        self.providers = providers if providers is not None else get_default_providers()

        cfg = MODEL_CONFIGS[self.model_name]
        self.model_file = model_path or ensure_model_exists(cfg['file'], cfg['tag'])
        self.session = get_inference_session(self.model_file, providers=self.providers)

    def preload(self) -> None:
        """Warms up the detector session."""
        _ = self.session.get_inputs()


    def detect(self, vision_frame: VisionFrame) -> List[Dict[str, Any]]:
        """
        Runs face detection on an input image (BGR).
        Returns a list of dicts with keys:
          - 'bbox': [x1, y1, x2, y2]
          - 'score': float confidence score
          - 'landmark_5': np.ndarray of shape (5, 2)
        """
        all_bboxes: List[BoundingBox] = []
        all_scores: List[Score] = []
        all_landmarks: List[FaceLandmark5] = []

        for angle in self.angles:
            if angle == 0:
                bboxes, scores, landmarks = self._detect_single_frame(vision_frame)
            else:
                bboxes, scores, landmarks = self._detect_by_angle(vision_frame, angle)

            all_bboxes.extend(bboxes)
            all_scores.extend(scores)
            all_landmarks.extend(landmarks)

        if not all_bboxes:
            return []

        # Filter by NMS if multiple detections or angles
        keep_indices = apply_nms(all_bboxes, all_scores, self.score_threshold, self.nms_threshold)
        results = []
        for idx in keep_indices:
            results.append({
                'bbox': all_bboxes[idx],
                'score': float(all_scores[idx]),
                'landmark_5': all_landmarks[idx]
            })

        return results

    def _prepare_detect_frame(self, temp_vision_frame: VisionFrame) -> np.ndarray:
        h, w = temp_vision_frame.shape[:2]
        ratio = min(self.input_size[0] / w, self.input_size[1] / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized_frame = cv2.resize(temp_vision_frame, (new_w, new_h))
        detect_frame = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        detect_frame[:new_h, :new_w] = resized_frame
        return detect_frame

    def _normalize_detect_frame(self, detect_frame: np.ndarray, norm_range: List[float]) -> np.ndarray:
        frame = detect_frame.astype(np.float32)
        if norm_range == [-1, 1]:
            frame = (frame - 127.5) / 128.0
        elif norm_range == [0, 1]:
            frame = frame / 255.0
        # Transpose to NCHW
        frame = frame.transpose(2, 0, 1)
        return np.expand_dims(frame, axis=0)

    def _detect_single_frame(self, vision_frame: VisionFrame) -> Tuple[List[BoundingBox], List[Score], List[FaceLandmark5]]:
        margin_top, margin_right, margin_bottom, margin_left = self._prepare_margin(vision_frame)
        padded_frame = np.pad(vision_frame, ((margin_top, margin_bottom), (margin_left, margin_right), (0, 0)))

        if self.model_name == 'yolo_face':
            bboxes, scores, landmarks = self._detect_with_yolo_face(padded_frame)
        elif self.model_name == 'scrfd':
            bboxes, scores, landmarks = self._detect_with_scrfd(padded_frame)
        elif self.model_name == 'retinaface':
            bboxes, scores, landmarks = self._detect_with_retinaface(padded_frame)
        elif self.model_name == 'yunet':
            bboxes, scores, landmarks = self._detect_with_yunet(padded_frame)
        else:
            bboxes, scores, landmarks = [], [], []

        # Remove margin offset
        offset_box = np.array([margin_left, margin_top, margin_left, margin_top])
        offset_lm = np.array([margin_left, margin_top])

        bboxes = [normalize_bounding_box(b) - offset_box for b in bboxes]
        landmarks = [lm - offset_lm for lm in landmarks]
        return bboxes, scores, landmarks

    def _prepare_margin(self, vision_frame: VisionFrame) -> Tuple[int, int, int, int]:
        h, w = vision_frame.shape[:2]
        margin_top = int(h * np.interp(self.margin[0], [0, 100], [0, 0.5]))
        margin_right = int(w * np.interp(self.margin[1], [0, 100], [0, 0.5]))
        margin_bottom = int(h * np.interp(self.margin[2], [0, 100], [0, 0.5]))
        margin_left = int(w * np.interp(self.margin[3], [0, 100], [0, 0.5]))
        return margin_top, margin_right, margin_bottom, margin_left

    def _detect_by_angle(self, vision_frame: VisionFrame, face_angle: Angle) -> Tuple[List[BoundingBox], List[Score], List[FaceLandmark5]]:
        rotation_matrix, rotation_size = create_rotation_matrix_and_size(face_angle, (vision_frame.shape[1], vision_frame.shape[0]))
        rotated_frame = cv2.warpAffine(vision_frame, rotation_matrix, rotation_size)
        rotation_inv = cv2.invertAffineTransform(rotation_matrix)

        bboxes, scores, landmarks = self._detect_single_frame(rotated_frame)
        bboxes = [transform_bounding_box(b, rotation_inv) for b in bboxes]
        landmarks = [transform_points(lm, rotation_inv) for lm in landmarks]
        return bboxes, scores, landmarks

    def _detect_with_yolo_face(self, vision_frame: VisionFrame) -> Tuple[List[BoundingBox], List[Score], List[FaceLandmark5]]:
        h, w = vision_frame.shape[:2]
        dw, dh = self.input_size
        detect_frame = self._prepare_detect_frame(vision_frame)
        norm_frame = self._normalize_detect_frame(detect_frame, [0, 1])

        out = self.session.run(None, {'input': norm_frame})[0]
        out = np.squeeze(out).T

        bboxes_raw, scores_raw, landmarks_raw = np.split(out, [4, 5], axis=1)
        keep = np.where(scores_raw.ravel() > self.score_threshold)[0]

        if not np.any(keep):
            return [], [], []

        bboxes_raw = bboxes_raw[keep]
        scores_raw = scores_raw[keep]
        landmarks_raw = landmarks_raw[keep]

        ratio_x = w / min(w, dw * (w / max(w, h))) if max(w, h) > 0 else 1.0
        ratio_w = w / min(dw, w) if w < dw else w / dw
        ratio_h = h / min(dh, h) if h < dh else h / dh

        scale = max(w / dw, h / dh)
        bounding_boxes = []
        scores = scores_raw.ravel().tolist()
        landmarks_5 = []

        for b in bboxes_raw:
            x_c, y_c, bw, bh = b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale
            bounding_boxes.append(np.array([x_c - bw / 2, y_c - bh / 2, x_c + bw / 2, y_c + bh / 2]))

        for lm in landmarks_raw:
            lm_pts = lm.reshape(-1, 3)[:, :2] * scale
            landmarks_5.append(lm_pts.astype(np.float32))

        return bounding_boxes, scores, landmarks_5

    def _detect_with_scrfd(self, vision_frame: VisionFrame) -> Tuple[List[BoundingBox], List[Score], List[FaceLandmark5]]:
        feature_strides = [8, 16, 32]
        feature_map_channel = 3
        anchor_total = 2

        h, w = vision_frame.shape[:2]
        dw, dh = self.input_size
        scale_x, scale_y = w / dw, h / dh

        detect_frame = cv2.resize(vision_frame, (dw, dh))
        norm_frame = self._normalize_detect_frame(detect_frame, [-1, 1])
        detection = self.session.run(None, {'input': norm_frame})

        bounding_boxes, scores, landmarks_5 = [], [], []

        for idx, stride in enumerate(feature_strides):
            scores_raw = detection[idx]
            keep = np.where(scores_raw >= self.score_threshold)[0]

            if np.any(keep):
                stride_h, stride_w = dh // stride, dw // stride
                anchors = create_static_anchors(stride, anchor_total, stride_h, stride_w)
                bboxes_raw = detection[idx + feature_map_channel] * stride
                landmarks_raw = detection[idx + feature_map_channel * 2] * stride

                decoded_boxes = distance_to_bounding_box(anchors, bboxes_raw)[keep]
                decoded_lms = distance_to_face_landmark_5(anchors, landmarks_raw)[keep]

                for box in decoded_boxes:
                    bounding_boxes.append(np.array([box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y]))

                for sc in scores_raw[keep]:
                    scores.append(float(sc[0]))

                for lm in decoded_lms:
                    landmarks_5.append(lm * [scale_x, scale_y])

        return bounding_boxes, scores, landmarks_5

    def _detect_with_retinaface(self, vision_frame: VisionFrame) -> Tuple[List[BoundingBox], List[Score], List[FaceLandmark5]]:
        feature_strides = [8, 16, 32]
        feature_map_channel = 3
        anchor_total = 2

        h, w = vision_frame.shape[:2]
        dw, dh = self.input_size
        scale_x, scale_y = w / dw, h / dh

        detect_frame = cv2.resize(vision_frame, (dw, dh))
        norm_frame = self._normalize_detect_frame(detect_frame, [-1, 1])
        detection = self.session.run(None, {'input': norm_frame})

        bounding_boxes, scores, landmarks_5 = [], [], []

        for idx, stride in enumerate(feature_strides):
            scores_raw = detection[idx]
            keep = np.where(scores_raw >= self.score_threshold)[0]

            if np.any(keep):
                stride_h, stride_w = dh // stride, dw // stride
                anchors = create_static_anchors(stride, anchor_total, stride_h, stride_w)
                bboxes_raw = detection[idx + feature_map_channel] * stride
                landmarks_raw = detection[idx + feature_map_channel * 2] * stride

                decoded_boxes = distance_to_bounding_box(anchors, bboxes_raw)[keep]
                decoded_lms = distance_to_face_landmark_5(anchors, landmarks_raw)[keep]

                for box in decoded_boxes:
                    bounding_boxes.append(np.array([box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y]))

                for sc in scores_raw[keep]:
                    scores.append(float(sc[0]))

                for lm in decoded_lms:
                    landmarks_5.append(lm * [scale_x, scale_y])

        return bounding_boxes, scores, landmarks_5

    def _detect_with_yunet(self, vision_frame: VisionFrame) -> Tuple[List[BoundingBox], List[Score], List[FaceLandmark5]]:
        feature_strides = [8, 16, 32]
        feature_map_channel = 3
        anchor_total = 1

        h, w = vision_frame.shape[:2]
        dw, dh = self.input_size
        scale_x, scale_y = w / dw, h / dh

        detect_frame = cv2.resize(vision_frame, (dw, dh))
        norm_frame = self._normalize_detect_frame(detect_frame, [0, 255])
        detection = self.session.run(None, {'input': norm_frame})

        bounding_boxes, scores, landmarks_5 = [], [], []

        for idx, stride in enumerate(feature_strides):
            scores_raw = (detection[idx] * detection[idx + feature_map_channel]).reshape(-1)
            keep = np.where(scores_raw >= self.score_threshold)[0]

            if np.any(keep):
                stride_h, stride_w = dh // stride, dw // stride
                anchors = create_static_anchors(stride, anchor_total, stride_h, stride_w)
                center = detection[idx + feature_map_channel * 2].squeeze(0)[:, :2] * stride + anchors
                box_size = np.exp(detection[idx + feature_map_channel * 2].squeeze(0)[:, 2:4]) * stride
                lms_raw = detection[idx + feature_map_channel * 3].squeeze(0)

                x1 = (center[:, 0] - box_size[:, 0] / 2) * scale_x
                y1 = (center[:, 1] - box_size[:, 1] / 2) * scale_y
                x2 = (center[:, 0] + box_size[:, 0] / 2) * scale_x
                y2 = (center[:, 1] + box_size[:, 1] / 2) * scale_y

                for i in keep:
                    bounding_boxes.append(np.array([x1[i], y1[i], x2[i], y2[i]]))
                    scores.append(float(scores_raw[i]))
                    lm_5 = (lms_raw[i].reshape(5, 2) * stride + anchors[i]) * [scale_x, scale_y]
                    landmarks_5.append(lm_5)

        return bounding_boxes, scores, landmarks_5


def detect_faces(
    vision_frame: VisionFrame,
    model_name: str = 'yolo_face',
    score_threshold: float = 0.5,
    input_size: Tuple[int, int] = (640, 640)
) -> List[Dict[str, Any]]:
    """Helper functional API to run face detection on an image."""
    detector = FaceDetector(model_name=model_name, score_threshold=score_threshold, input_size=input_size)
    return detector.detect(vision_frame)
