from typing import List, Optional, Tuple, Union, Dict, Any
import os
import cv2
import numpy as np
import onnxruntime

from Face.typing import Mask, VisionFrame, FaceLandmark68, Padding, Face
from downloads import download_model

OCCLUSION_MODELS = {
    'face_occluder': {
        'file': 'face_occluder.onnx',
        'size': (256, 256)
    },
    'xseg_1': {
        'file': 'xseg_1.onnx',
        'size': (256, 256)
    },
    'xseg_2': {
        'file': 'xseg_2.onnx',
        'size': (256, 256)
    },
    'xseg_3': {
        'file': 'xseg_3.onnx',
        'size': (256, 256)
    }
}

PARSER_MODELS = {
    'bisenet_resnet_34': {
        'file': 'bisenet_resnet_34.onnx',
        'size': (512, 512)
    },
    'face_parser': {
        'file': 'face_parser.onnx',
        'size': (512, 512)
    }
}

FACE_MASK_AREA_MAP: Dict[str, List[int]] = {
    'upper-face': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    'lower-face': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
    'mouth': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
    'eyes': [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    'nose': [27, 28, 29, 30, 31, 32, 33, 34, 35]
}

FACE_MASK_REGION_MAP: Dict[str, int] = {
    'skin': 1,
    'left-eyebrow': 2,
    'right-eyebrow': 3,
    'left-eye': 4,
    'right-eye': 5,
    'glasses': 6,
    'left-ear': 7,
    'right-ear': 8,
    'earring': 9,
    'nose': 10,
    'mouth': 11,
    'upper-lip': 12,
    'lower-lip': 13,
    'neck': 14,
    'necklace': 15,
    'cloth': 16,
    'hair': 17,
    'hat': 18
}


def create_box_mask(
    crop_vision_frame: VisionFrame,
    face_mask_blur: float = 0.3,
    face_mask_padding: Padding = (0, 0, 0, 0)
) -> Mask:
    """
    Generates a feathered rectangular boundary mask for the face crop.
    """
    ch, cw = crop_vision_frame.shape[:2]
    blur_amount = int(cw * 0.5 * face_mask_blur)
    blur_area = max(blur_amount // 2, 1)

    box_mask = np.ones((ch, cw), dtype=np.float32)

    top = max(blur_area, int(ch * face_mask_padding[0] / 100))
    right = max(blur_area, int(cw * face_mask_padding[1] / 100))
    bottom = max(blur_area, int(ch * face_mask_padding[2] / 100))
    left = max(blur_area, int(cw * face_mask_padding[3] / 100))

    if top > 0:
        box_mask[:top, :] = 0
    if bottom > 0:
        box_mask[-bottom:, :] = 0
    if left > 0:
        box_mask[:, :left] = 0
    if right > 0:
        box_mask[:, -right:] = 0

    if blur_amount > 0:
        box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)

    return box_mask.clip(0, 1)


class FaceMasker:
    """
    Unified Face Masking Engine supporting Box, Occlusion, Landmark-Area, and Semantic Region masks.
    """
    def __init__(
        self,
        occluder_model: str = 'face_occluder',
        parser_model: str = 'bisenet_resnet_34',
        providers: Optional[List[str]] = None
    ):
        if providers is None:
            available = onnxruntime.get_available_providers()
            self.providers = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if p in available] or ['CPUExecutionProvider']
        else:
            self.providers = providers

        self.occluder_model = occluder_model
        self.parser_model = parser_model

        self._occluder_session: Optional[onnxruntime.InferenceSession] = None
        self._parser_session: Optional[onnxruntime.InferenceSession] = None

    def _get_occluder_session(self) -> onnxruntime.InferenceSession:
        if self._occluder_session is None:
            cfg = OCCLUSION_MODELS.get(self.occluder_model, OCCLUSION_MODELS['face_occluder'])
            model_path = download_model(self.occluder_model)
            self._occluder_session = onnxruntime.InferenceSession(model_path, providers=self.providers)
        return self._occluder_session

    def _get_parser_session(self) -> onnxruntime.InferenceSession:
        if self._parser_session is None:
            cfg = PARSER_MODELS.get(self.parser_model, PARSER_MODELS['bisenet_resnet_34'])
            model_path = download_model(self.parser_model)
            self._parser_session = onnxruntime.InferenceSession(model_path, providers=self.providers)
        return self._parser_session

    def create_occlusion_mask(self, crop_vision_frame: VisionFrame) -> Mask:
        """
        Creates an occlusion mask segmenting hands, glasses, hair, microphones, or objects in front of the face.
        """
        session = self._get_occluder_session()
        cfg = OCCLUSION_MODELS.get(self.occluder_model, OCCLUSION_MODELS['face_occluder'])
        target_size = cfg['size']

        h, w = crop_vision_frame.shape[:2]
        prep = cv2.resize(crop_vision_frame, target_size)
        prep = np.expand_dims(prep, axis=0).astype(np.float32) / 255.0

        raw_mask = session.run(None, {'input': prep})[0][0]
        raw_mask = raw_mask.transpose(0, 1).clip(0, 1).astype(np.float32) if len(raw_mask.shape) == 2 else raw_mask[0].clip(0, 1).astype(np.float32)
        resized_mask = cv2.resize(raw_mask, (w, h))

        feathered_mask = (cv2.GaussianBlur(resized_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        return feathered_mask.clip(0, 1)

    def create_area_mask(
        self,
        crop_vision_frame: VisionFrame,
        face_landmark_68: FaceLandmark68,
        face_mask_areas: List[str]
    ) -> Mask:
        """
        Creates a mask based on landmark areas (convex hulls over facial landmarks).
        """
        h, w = crop_vision_frame.shape[:2]
        landmark_points = []

        for area in face_mask_areas:
            if area in FACE_MASK_AREA_MAP:
                landmark_points.extend(FACE_MASK_AREA_MAP[area])

        if not landmark_points:
            return np.ones((h, w), dtype=np.float32)

        selected_landmarks = face_landmark_68[landmark_points].astype(np.int32)
        convex_hull = cv2.convexHull(selected_landmarks)

        area_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillConvexPoly(area_mask, convex_hull, 1.0)

        area_mask = (cv2.GaussianBlur(area_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        return area_mask.clip(0, 1)

    def create_region_mask(
        self,
        crop_vision_frame: VisionFrame,
        face_mask_regions: List[str]
    ) -> Mask:
        """
        Creates a semantic facial region mask using BiSeNet face parsing.
        """
        session = self._get_parser_session()
        cfg = PARSER_MODELS.get(self.parser_model, PARSER_MODELS['bisenet_resnet_34'])
        target_size = cfg['size']

        h, w = crop_vision_frame.shape[:2]
        prep = cv2.resize(crop_vision_frame, target_size)
        prep = prep[:, :, ::-1].astype(np.float32) / 255.0
        prep -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        prep /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        prep = prep.transpose(2, 0, 1)
        prep = np.expand_dims(prep, axis=0)

        parsed = session.run(None, {'input': prep})[0][0]
        parsed_labels = parsed.argmax(axis=0)

        target_ids = [FACE_MASK_REGION_MAP[r] for r in face_mask_regions if r in FACE_MASK_REGION_MAP]
        if not target_ids:
            return np.ones((h, w), dtype=np.float32)

        region_mask = np.isin(parsed_labels, target_ids).astype(np.float32)
        resized_mask = cv2.resize(region_mask, (w, h))

        feathered_mask = (cv2.GaussianBlur(resized_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        return feathered_mask.clip(0, 1)

    def create_mask(
        self,
        crop_vision_frame: VisionFrame,
        target_face: Optional[Face] = None,
        affine_matrix: Optional[np.ndarray] = None,
        mask_types: Optional[List[str]] = None,
        mask_blur: float = 0.3,
        mask_padding: Padding = (0, 0, 0, 0),
        mask_areas: Optional[List[str]] = None,
        mask_regions: Optional[List[str]] = None
    ) -> Mask:
        """
        Combines requested mask types (box, occlusion, area, region) into a composite blending mask.
        """
        types = mask_types or ['box', 'occlusion']
        masks: List[Mask] = []

        if 'box' in types:
            masks.append(create_box_mask(crop_vision_frame, mask_blur, mask_padding))

        if 'occlusion' in types:
            try:
                masks.append(self.create_occlusion_mask(crop_vision_frame))
            except Exception:
                pass

        if 'area' in types and target_face is not None and affine_matrix is not None and mask_areas:
            try:
                landmark_68 = cv2.transform(target_face.landmark_set['68'].reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
                masks.append(self.create_area_mask(crop_vision_frame, landmark_68, mask_areas))
            except Exception:
                pass

        if 'region' in types and mask_regions:
            try:
                masks.append(self.create_region_mask(crop_vision_frame, mask_regions))
            except Exception:
                pass

        if not masks:
            return np.ones(crop_vision_frame.shape[:2], dtype=np.float32)

        # Intersect all masks by taking elementwise minimum
        final_mask = np.minimum.reduce(masks)
        return final_mask.clip(0, 1)
