from collections import namedtuple
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypeAlias, TypedDict
import numpy as np
from numpy.typing import NDArray

Scale: TypeAlias = float
Score: TypeAlias = float
Angle: TypeAlias = int

Detection: TypeAlias = NDArray[Any]
Prediction: TypeAlias = NDArray[Any]

BoundingBox: TypeAlias = NDArray[Any]  # [x1, y1, x2, y2]
FaceLandmark5: TypeAlias = NDArray[Any]  # shape (5, 2)
FaceLandmark68: TypeAlias = NDArray[Any]  # shape (68, 2)

FaceLandmarkSet = TypedDict('FaceLandmarkSet', {
    '5': FaceLandmark5,
    '5/68': FaceLandmark5,
    '68': FaceLandmark68,
    '68/5': FaceLandmark68
})

FaceScoreSet = TypedDict('FaceScoreSet', {
    'detector': Score,
    'landmarker': Score
})

Embedding: TypeAlias = NDArray[np.float32]
Gender = Literal['female', 'male']
Age: TypeAlias = range
Race = Literal['white', 'black', 'latino', 'asian', 'indian', 'arabic']

Face = namedtuple('Face', [
    'bounding_box',
    'score_set',
    'landmark_set',
    'angle',
    'embedding',
    'embedding_norm',
    'gender',
    'age',
    'race'
], defaults=(None, None, None))


FaceSet: TypeAlias = Dict[str, List[Face]]
FaceStore = TypedDict('FaceStore', {
    'static_faces': FaceSet
})

VisionFrame: TypeAlias = NDArray[Any]
Mask: TypeAlias = NDArray[Any]
Points: TypeAlias = NDArray[Any]
Distance: TypeAlias = NDArray[Any]
Matrix: TypeAlias = NDArray[Any]
Anchors: TypeAlias = NDArray[Any]
Translation: TypeAlias = NDArray[Any]

Fps: TypeAlias = float
Duration: TypeAlias = float
Color: TypeAlias = Tuple[int, int, int, int]
Padding: TypeAlias = Tuple[int, int, int, int]
Margin: TypeAlias = Tuple[int, int, int, int]
Resolution: TypeAlias = Tuple[int, int]
