from typing import List, Optional, Tuple
import os
import cv2
import numpy as np
import onnxruntime

from Face.typing import Gender, Age, Race, VisionFrame, FaceLandmark5
from Face.modules.face_helper import warp_face_by_face_landmark_5
from downloads import download_model

CLASSIFIER_CONFIG = {
    'fairface': {
        'file': 'fairface.onnx',
        'template': 'arcface_112_v2',
        'size': (224, 224),
        'mean': np.array([0.485, 0.456, 0.406], dtype=np.float32),
        'std': np.array([0.229, 0.224, 0.225], dtype=np.float32)
    }
}


def categorize_gender(gender_id: int) -> Gender:
    return 'female' if gender_id == 1 else 'male'


def categorize_age(age_id: int) -> Age:
    age_ranges = [
        range(0, 3),    # 0: 0-2
        range(3, 10),   # 1: 3-9
        range(10, 20),  # 2: 10-19
        range(20, 30),  # 3: 20-29
        range(30, 40),  # 4: 30-39
        range(40, 50),  # 5: 40-49
        range(50, 60),  # 6: 50-59
        range(60, 70),  # 7: 60-69
        range(70, 100)  # 8: 70+
    ]
    if 0 <= age_id < len(age_ranges):
        return age_ranges[age_id]
    return range(20, 30)


def categorize_race(race_id: int) -> Race:
    race_map = {
        0: 'white',
        1: 'black',
        2: 'latino',
        3: 'asian',
        4: 'asian',
        5: 'indian',
        6: 'arabic'
    }
    return race_map.get(race_id, 'white')


class FaceClassifier:
    def __init__(
        self,
        model_name: str = 'fairface',
        providers: Optional[List[str]] = None,
        model_path: Optional[str] = None
    ):
        self.model_name = model_name.lower()
        self.cfg = CLASSIFIER_CONFIG.get(self.model_name, CLASSIFIER_CONFIG['fairface'])

        if providers is None:
            available = onnxruntime.get_available_providers()
            self.providers = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if p in available] or ['CPUExecutionProvider']
        else:
            self.providers = providers

        self.model_file = model_path or download_model('fairface')
        self.session = onnxruntime.InferenceSession(self.model_file, providers=self.providers)

    def classify(self, vision_frame: VisionFrame, face_landmark_5: FaceLandmark5) -> Tuple[Gender, Age, Race]:
        """Classifies gender, age range, and race using FairFace."""
        crop_vision_frame, _ = warp_face_by_face_landmark_5(
            vision_frame,
            face_landmark_5,
            self.cfg['template'],
            self.cfg['size']
        )
        # BGR to RGB, normalize
        crop = crop_vision_frame.astype(np.float32)[:, :, ::-1] / 255.0
        crop = (crop - self.cfg['mean']) / self.cfg['std']
        input_tensor = crop.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        outputs = self.session.run(None, {'input': input_tensor})
        race_id, gender_id, age_id = outputs[0][0], outputs[1][0], outputs[2][0]

        gender = categorize_gender(int(gender_id))
        age = categorize_age(int(age_id))
        race = categorize_race(int(race_id))

        return gender, age, race


def classify_face(vision_frame: VisionFrame, face_landmark_5: FaceLandmark5) -> Tuple[Gender, Age, Race]:
    """Helper function to classify face attributes."""
    try:
        classifier = FaceClassifier()
        return classifier.classify(vision_frame, face_landmark_5)
    except Exception:
        # Graceful fallback if classifier model is unavailable
        return 'male', range(20, 30), 'white'
