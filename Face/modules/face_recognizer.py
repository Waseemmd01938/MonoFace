from typing import List, Optional, Tuple, Union, Dict, Any
import os
import cv2
import numpy as np
import onnxruntime

from Face.modules.face_helper import (
    FaceLandmark5,
    VisionFrame,
    Matrix,
    warp_face_by_face_landmark_5,
    ensure_model_exists
)

Embedding = np.ndarray

RECOGNIZER_CONFIG = {
    'arcface': {
        'file': 'arcface_w600k_r50.onnx',
        'tag': 'models-3.0.0',
        'template': 'arcface_112_v2',
        'size': (112, 112)
    }
}


class FaceRecognizer:
    def __init__(
        self,
        model_name: str = 'arcface',
        providers: Optional[List[str]] = None,
        model_path: Optional[str] = None
    ):
        self.model_name = model_name.lower()
        if self.model_name not in RECOGNIZER_CONFIG:
            raise ValueError(f"Unsupported recognizer model: {model_name}. Supported: {list(RECOGNIZER_CONFIG.keys())}")

        self.cfg = RECOGNIZER_CONFIG[self.model_name]
        self.template = self.cfg['template']
        self.size = self.cfg['size']

        if providers is None:
            available = onnxruntime.get_available_providers()
            self.providers = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if p in available] or ['CPUExecutionProvider']
        else:
            self.providers = providers

        self.model_file = model_path or ensure_model_exists(self.cfg['file'], self.cfg['tag'])
        self.session = onnxruntime.InferenceSession(self.model_file, providers=self.providers)

    def get_embedding(
        self,
        vision_frame: VisionFrame,
        face_landmark_5: FaceLandmark5
    ) -> Tuple[Embedding, Embedding, VisionFrame, Matrix]:
        """
        Extracts ArcFace embedding from an input frame and 5 facial landmarks.
        Returns:
            (raw_embedding: np.ndarray (512,), normalized_embedding: np.ndarray (512,), aligned_crop: np.ndarray (112, 112, 3), affine_matrix: np.ndarray (2, 3))
        """
        crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
            vision_frame,
            face_landmark_5,
            self.template,
            self.size
        )

        # Normalize to [-1, 1], convert BGR to RGB and transpose to NCHW
        normalized_crop = crop_vision_frame / 127.5 - 1.0
        rgb_crop = normalized_crop[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        input_tensor = np.expand_dims(rgb_crop, axis=0)

        raw_embedding = self.session.run(None, {'input': input_tensor})[0].ravel()
        norm = np.linalg.norm(raw_embedding)
        normalized_embedding = raw_embedding / max(norm, 1e-6)

        return raw_embedding, normalized_embedding, crop_vision_frame, affine_matrix

    @staticmethod
    def compute_similarity(embedding_a: Embedding, embedding_b: Embedding) -> float:
        """Computes cosine similarity between two face embeddings (range: -1 to 1, typically 0 to 1)."""
        norm_a = embedding_a / max(np.linalg.norm(embedding_a), 1e-6)
        norm_b = embedding_b / max(np.linalg.norm(embedding_b), 1e-6)
        return float(np.dot(norm_a, norm_b))

    def is_match(self, embedding_a: Embedding, embedding_b: Embedding, threshold: float = 0.6) -> bool:
        """Determines if two embeddings belong to the same identity based on cosine similarity threshold."""
        return self.compute_similarity(embedding_a, embedding_b) >= threshold


def calculate_face_embedding(
    vision_frame: VisionFrame,
    face_landmark_5: FaceLandmark5,
    model_name: str = 'arcface'
) -> Tuple[Embedding, Embedding]:
    """Helper function to calculate raw and normalized face embeddings."""
    recognizer = FaceRecognizer(model_name=model_name)
    raw_emb, norm_emb, _, _ = recognizer.get_embedding(vision_frame, face_landmark_5)
    return raw_emb, norm_emb


def compare_faces(embedding_a: Embedding, embedding_b: Embedding) -> float:
    """Helper function to compare similarity between two face embeddings."""
    return FaceRecognizer.compute_similarity(embedding_a, embedding_b)
