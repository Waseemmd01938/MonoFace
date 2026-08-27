"""
Face Analyser Module for MonoFace
Coordinates Face Detection, Landmarking, and Recognition pipelines.
Preloads models upfront and optimizes memory management for maximum performance.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import cv2

from Face.typing import (
    Face,
    BoundingBox,
    FaceLandmark5,
    FaceLandmark68,
    FaceLandmarkSet,
    FaceScoreSet,
    Embedding,
    Gender,
    Age,
    Race,
    VisionFrame,
    Score,
    Angle
)
from Face.modules.face_helper import (
    apply_nms,
    convert_to_face_landmark_5,
    estimate_face_angle
)
from Face.modules.face_detector import FaceDetector
from Face.modules.face_landmark import FaceLandmarker
from Face.modules.face_recognizer import FaceRecognizer
from Face.modules.model_store import clear_session_cache, free_memory

# In-memory static cache for detected faces (capped for memory bounds)
_FACE_CACHE: Dict[int, List[Face]] = {}
_MAX_FACE_CACHE_ENTRIES = 512


def get_static_faces(vision_frame: VisionFrame) -> Optional[List[Face]]:
    """Retrieves cached faces for a given frame."""
    if vision_frame is None or vision_frame.size == 0:
        return None
    frame_hash = hash(vision_frame.tobytes()[::1000])
    return _FACE_CACHE.get(frame_hash)


def set_static_faces(vision_frame: VisionFrame, faces: List[Face]) -> None:
    """Caches extracted faces for a given frame with bounded capacity."""
    if vision_frame is None or vision_frame.size == 0:
        return
    if len(_FACE_CACHE) >= _MAX_FACE_CACHE_ENTRIES:
        _FACE_CACHE.clear()
    frame_hash = hash(vision_frame.tobytes()[::1000])
    _FACE_CACHE[frame_hash] = faces


def clear_face_cache() -> None:
    """Clears the static faces cache."""
    _FACE_CACHE.clear()


def clear_analyser_memory() -> None:
    """Clears all analyser caches, model sessions, and invokes memory reclamation."""
    clear_face_cache()
    clear_session_cache()
    free_memory()


class FaceAnalyser:
    """
    Unified high-level Face Analyser pipeline.
    Preloads all sub-module models (Detector, Landmarker, Fan 68/5, Recognizer) upfront.
    """
    def __init__(
        self,
        detector_model: str = 'yolo_face',
        detector_score: float = 0.5,
        landmarker_model: str = '2dfan4',
        landmarker_score: float = 0.5,
        recognizer_model: str = 'arcface',
        detector_size: Tuple[int, int] = (640, 640),
        detector_angles: Optional[List[Angle]] = None,
        providers: Optional[List[str]] = None,
        preload_models: bool = True
    ):
        self.detector_score = detector_score
        self.landmarker_score = landmarker_score
        self.detector_angles = detector_angles or [0]
        self.providers = providers

        # Initialize sub-modules (all models use shared session caching)
        self.detector = FaceDetector(
            model_name=detector_model,
            score_threshold=detector_score,
            input_size=detector_size,
            angles=self.detector_angles,
            providers=providers
        )
        self.landmarker = FaceLandmarker(
            model_name=landmarker_model,
            providers=providers
        )
        self.recognizer = FaceRecognizer(
            model_name=recognizer_model,
            providers=providers
        )

        # Preload all face analyser models upfront for instant inference
        if preload_models:
            self.preload()

    def preload(self) -> None:
        """Preloads and warms up all face analyser models (Detector, Landmarker, Fan 68/5, Recognizer)."""
        self.detector.preload()
        self.landmarker.preload()
        self.recognizer.preload()

    def get_many_faces(self, vision_frames: List[VisionFrame], extract_embedding: bool = True) -> List[Face]:
        """Analyzes a list of image frames and extracts all detected and aligned faces."""
        many_faces: List[Face] = []

        for vision_frame in vision_frames:
            if vision_frame is None or vision_frame.size == 0:
                continue

            cached = get_static_faces(vision_frame)
            if cached is not None:
                many_faces.extend(cached)
                continue

            detections = self.detector.detect(vision_frame)
            if not detections:
                continue

            bounding_boxes = [d['bbox'] for d in detections]
            face_scores = [d['score'] for d in detections]
            face_landmarks_5 = [d['landmark_5'] for d in detections]

            faces = self.create_faces(vision_frame, bounding_boxes, face_scores, face_landmarks_5, extract_embedding=extract_embedding)
            if faces:
                many_faces.extend(faces)
                set_static_faces(vision_frame, faces)

        return many_faces

    def create_faces(
        self,
        vision_frame: VisionFrame,
        bounding_boxes: List[BoundingBox],
        face_scores: List[Score],
        face_landmarks_5: List[FaceLandmark5],
        extract_embedding: bool = True,
        classify: bool = False
    ) -> List[Face]:
        """
        Creates structured Face objects with complete 5-point, 68-point landmarks,
        and optional ArcFace identity embeddings.
        """
        faces: List[Face] = []

        for bbox, score, lm_5 in zip(bounding_boxes, face_scores, face_landmarks_5):
            lm_score_68 = 0.0
            face_angle = 0

            # Use 68-point dense landmarker if requested, else use fast 5-to-68 estimator
            if self.landmarker_score > 0:
                try:
                    detected_68, score_68 = self.landmarker.detect_landmarks(vision_frame, bbox, face_angle)
                    if score_68 >= self.landmarker_score:
                        lm_68 = detected_68
                        lm_score_68 = score_68
                        lm_5_68 = convert_to_face_landmark_5(lm_68)
                        lm_68_5 = lm_68
                    else:
                        lm_5_68 = lm_5.copy()
                        lm_68 = self.landmarker.estimate_landmark_68_from_5(lm_5_68)
                        lm_68_5 = lm_68
                except Exception:
                    lm_5_68 = lm_5.copy()
                    lm_68 = self.landmarker.estimate_landmark_68_from_5(lm_5_68)
                    lm_68_5 = lm_68
            else:
                lm_5_68 = lm_5.copy()
                lm_68 = self.landmarker.estimate_landmark_68_from_5(lm_5_68)
                lm_68_5 = lm_68

            landmark_set: FaceLandmarkSet = {
                '5': lm_5,
                '5/68': lm_5_68,
                '68': lm_68,
                '68/5': lm_68_5
            }


            score_set: FaceScoreSet = {
                'detector': float(score),
                'landmarker': float(lm_score_68)
            }

            # Extract 512-D ArcFace embedding only when requested
            if extract_embedding:
                raw_emb, norm_emb, _, _ = self.recognizer.get_embedding(vision_frame, landmark_set['5/68'])
            else:
                raw_emb, norm_emb = None, None

            faces.append(Face(
                bounding_box=bbox,
                score_set=score_set,
                landmark_set=landmark_set,
                angle=face_angle,
                embedding=raw_emb,
                embedding_norm=norm_emb,
                gender=None,
                age=None,
                race=None
            ))

        return faces


    def get_one_face(self, faces: List[Face], position: int = 0) -> Optional[Face]:
        """Retrieves a single face by index from a list of faces."""
        if faces:
            pos = min(max(0, position), len(faces) - 1)
            return faces[pos]
        return None

    def get_average_face(self, faces: List[Face]) -> Optional[Face]:
        """Computes average identity embedding across multiple face instances."""
        if not faces:
            return None

        first_face = faces[0]
        raw_embs = [f.embedding for f in faces]
        norm_embs = [f.embedding_norm for f in faces]

        avg_raw = np.mean(raw_embs, axis=0)
        avg_norm = np.mean(norm_embs, axis=0)
        avg_norm = avg_norm / max(np.linalg.norm(avg_norm), 1e-6)

        return Face(
            bounding_box=first_face.bounding_box,
            score_set=first_face.score_set,
            landmark_set=first_face.landmark_set,
            angle=first_face.angle,
            embedding=avg_raw.astype(np.float32),
            embedding_norm=avg_norm.astype(np.float32),
            gender=None,
            age=None,
            race=None
        )

    def find_similar_faces(
        self,
        faces: List[Face],
        reference_face: Face,
        similarity_threshold: float = 0.6
    ) -> List[Tuple[Face, float]]:
        """Filters and ranks faces that match a reference face identity."""
        matches = []
        for face in faces:
            sim = self.recognizer.compute_similarity(reference_face.embedding_norm, face.embedding_norm)
            if sim >= similarity_threshold:
                matches.append((face, sim))
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches


# Functional API and singleton/caching for compatibility
_GLOBAL_ANALYSER: Optional[FaceAnalyser] = None


def get_analyser() -> FaceAnalyser:
    """Retrieves or initializes global singleton FaceAnalyser."""
    global _GLOBAL_ANALYSER
    if _GLOBAL_ANALYSER is None:
        _GLOBAL_ANALYSER = FaceAnalyser()
    return _GLOBAL_ANALYSER


def preload_face_analyser(
    detector_model: str = 'yolo_face',
    landmarker_model: str = '2dfan4',
    recognizer_model: str = 'arcface',
    providers: Optional[List[str]] = None
) -> FaceAnalyser:
    """Preloads all face analyser models upfront for instant readiness."""
    global _GLOBAL_ANALYSER
    _GLOBAL_ANALYSER = FaceAnalyser(
        detector_model=detector_model,
        landmarker_model=landmarker_model,
        recognizer_model=recognizer_model,
        providers=providers,
        preload_models=True
    )
    return _GLOBAL_ANALYSER


def create_faces(
    vision_frame: VisionFrame,
    bounding_boxes: List[BoundingBox],
    face_scores: List[Score],
    face_landmarks_5: List[FaceLandmark5]
) -> List[Face]:
    return get_analyser().create_faces(vision_frame, bounding_boxes, face_scores, face_landmarks_5)


def get_one_face(faces: List[Face], position: int = 0) -> Optional[Face]:
    return get_analyser().get_one_face(faces, position)


def get_average_face(faces: List[Face]) -> Optional[Face]:
    return get_analyser().get_average_face(faces)


def get_many_faces(vision_frames: List[VisionFrame]) -> List[Face]:
    return get_analyser().get_many_faces(vision_frames)


def scale_face(target_face: Face, target_vision_frame: VisionFrame, temp_vision_frame: VisionFrame) -> Face:
    """Scales bounding boxes and landmarks between differing frame dimensions."""
    scale_x = temp_vision_frame.shape[1] / target_vision_frame.shape[1]
    scale_y = temp_vision_frame.shape[0] / target_vision_frame.shape[0]

    bounding_box = target_face.bounding_box * np.array([scale_x, scale_y, scale_x, scale_y])
    landmark_set: FaceLandmarkSet = {
        '5': target_face.landmark_set['5'] * np.array([scale_x, scale_y]),
        '5/68': target_face.landmark_set['5/68'] * np.array([scale_x, scale_y]),
        '68': target_face.landmark_set['68'] * np.array([scale_x, scale_y]),
        '68/5': target_face.landmark_set['68/5'] * np.array([scale_x, scale_y])
    }

    return target_face._replace(
        bounding_box=bounding_box,
        landmark_set=landmark_set
    )
