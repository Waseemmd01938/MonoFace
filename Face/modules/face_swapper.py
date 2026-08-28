from typing import Dict, List, Optional, Tuple, Union, Any
import os
import cv2
import numpy as np
import onnx
import onnxruntime

from Face.typing import Face, VisionFrame, Mask, Embedding, Padding
from Face.modules.face_helper import (
    warp_face_by_face_landmark_5,
    paste_back,
    implode_pixel_boost,
    explode_pixel_boost
)
from Face.modules.face_masker import FaceMasker, create_box_mask
from Face.modules.model_store import get_inference_session, get_default_providers
from downloads import download_model

_INITIALIZER_CACHE: Dict[str, np.ndarray] = {}

SWAPPER_CONFIGS: Dict[str, Dict[str, Any]] = {

    'inswapper_128': {
        'file': 'inswapper_128.onnx',
        'type': 'inswapper',
        'template': 'arcface_128',
        'size': (128, 128),
        'mean': [0.0, 0.0, 0.0],
        'std': [1.0, 1.0, 1.0]
    },
    'inswapper_128_fp16': {
        'file': 'inswapper_128_fp16.onnx',
        'type': 'inswapper',
        'template': 'arcface_128',
        'size': (128, 128),
        'mean': [0.0, 0.0, 0.0],
        'std': [1.0, 1.0, 1.0]
    },
    'hyperswap_1a_256': {
        'file': 'hyperswap_1a_256.onnx',
        'type': 'hyperswap',
        'template': 'arcface_128',
        'size': (256, 256),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]
    },
    'hyperswap_1b_256': {
        'file': 'hyperswap_1b_256.onnx',
        'type': 'hyperswap',
        'template': 'arcface_128',
        'size': (256, 256),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]
    },
    'hyperswap_1c_256': {
        'file': 'hyperswap_1c_256.onnx',
        'type': 'hyperswap',
        'template': 'arcface_128',
        'size': (256, 256),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]
    },
    'simswap_256': {
        'file': 'simswap_256.onnx',
        'type': 'simswap',
        'template': 'arcface_112_v1',
        'size': (256, 256),
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'converter': 'crossface_simswap'
    },
    'simswap_512_unofficial': {
        'file': 'simswap_unofficial_512.onnx',
        'type': 'simswap',
        'template': 'arcface_112_v1',
        'size': (512, 512),
        'mean': [0.0, 0.0, 0.0],
        'std': [1.0, 1.0, 1.0],
        'converter': 'crossface_simswap'
    },
    'simswap_unofficial_512': {
        'file': 'simswap_unofficial_512.onnx',
        'type': 'simswap',
        'template': 'arcface_112_v1',
        'size': (512, 512),
        'mean': [0.0, 0.0, 0.0],
        'std': [1.0, 1.0, 1.0],
        'converter': 'crossface_simswap'
    }
}


class FaceSwapper:
    """
    High performance face swapper supporting Inswapper, HyperSwap, and SimSwap models with complete masking and Pixel Boost.
    """
    def __init__(
        self,
        model_name: str = 'inswapper_128',
        weight: float = 0.5,
        mask_types: Optional[List[str]] = None,
        mask_blur: float = 0.3,
        mask_padding: Padding = (0, 0, 0, 0),
        mask_areas: Optional[List[str]] = None,
        mask_regions: Optional[List[str]] = None,
        pixel_boost: Optional[str] = None,
        providers: Optional[List[str]] = None,
        model_path: Optional[str] = None
    ):
        self.model_name = model_name.lower()
        if self.model_name not in SWAPPER_CONFIGS:
            raise ValueError(f"Unsupported swapper model '{model_name}'. Supported: {list(SWAPPER_CONFIGS.keys())}")

        self.cfg = SWAPPER_CONFIGS[self.model_name]
        self.weight = weight
        self.mask_types = mask_types or ['box', 'occlusion']
        self.mask_blur = mask_blur
        self.mask_padding = mask_padding
        self.mask_areas = mask_areas
        self.mask_regions = mask_regions
        self.pixel_boost = pixel_boost

        self.providers = providers if providers is not None else get_default_providers()
        self.model_file = model_path or download_model(self.model_name)
        self.session = get_inference_session(self.model_file, providers=self.providers)

        # Inswapper projection matrix initializer
        self.initializer: Optional[np.ndarray] = None
        if self.cfg['type'] == 'inswapper':
            if self.model_file in _INITIALIZER_CACHE:
                self.initializer = _INITIALIZER_CACHE[self.model_file]
            else:
                try:
                    onnx_model = onnx.load(self.model_file)
                    self.initializer = onnx.numpy_helper.to_array(onnx_model.graph.initializer[-1])
                    _INITIALIZER_CACHE[self.model_file] = self.initializer
                except Exception:
                    pass

        # Embedding converter session for SimSwap / Ghost / HifiFace
        self.embedding_converter_session: Optional[onnxruntime.InferenceSession] = None
        if self.cfg.get('converter'):
            converter_file = download_model(self.cfg['converter'])
            self.embedding_converter_session = get_inference_session(converter_file, providers=self.providers)

        # Face Masker instance
        self.masker = FaceMasker(providers=self.providers)


    def prepare_source_embedding(self, source_face: Face) -> Embedding:
        """Transforms source face embedding according to swapper model requirements."""
        model_type = self.cfg['type']

        if model_type == 'hyperswap':
            return source_face.embedding_norm.reshape(1, -1)

        if model_type == 'inswapper':
            if self.initializer is not None:
                source_emb = source_face.embedding.reshape(1, -1)
                projected = np.dot(source_emb, self.initializer)
                return projected / max(np.linalg.norm(source_emb), 1e-6)
            return source_face.embedding_norm.reshape(1, -1)

        if model_type == 'simswap':
            if self.embedding_converter_session is not None:
                source_emb = source_face.embedding.reshape(-1, 512).astype(np.float32)
                converted_emb = self.embedding_converter_session.run(None, {'input': source_emb})[0].ravel()
                converted_norm = converted_emb / max(np.linalg.norm(converted_emb), 1e-6)
                return converted_norm.reshape(1, -1)
            return source_face.embedding_norm.reshape(1, -1)

        return source_face.embedding_norm.reshape(1, -1)

    def balance_embedding(self, source_embedding: Embedding, target_face: Face) -> Embedding:
        """Balances source identity with target face features based on weight."""
        if self.weight == 0.5 or target_face.embedding_norm is None:
            return source_embedding

        weight_factor = float(np.interp(self.weight, [0.0, 1.0], [0.35, -0.35]))
        target_norm = target_face.embedding_norm.reshape(1, -1)
        source_emb = source_embedding.reshape(1, -1)

        balanced = source_emb * (1.0 - weight_factor) + target_norm * weight_factor
        return balanced / max(np.linalg.norm(balanced), 1e-6)

    def _forward_single_crop(self, prep_crop: np.ndarray, source_embedding: np.ndarray) -> np.ndarray:
        inputs = {}
        for inp in self.session.get_inputs():
            if inp.name in ['source', 'emb', 'embedding']:
                inputs[inp.name] = source_embedding.astype(np.float32)
            elif inp.name in ['target', 'img', 'input']:
                inputs[inp.name] = prep_crop.astype(np.float32)
            else:
                if len(inp.shape) == 2:
                    inputs[inp.name] = source_embedding.astype(np.float32)
                else:
                    inputs[inp.name] = prep_crop.astype(np.float32)

        return self.session.run(None, inputs)[0][0]

    def swap_face(
        self,
        source_face: Face,
        target_face: Face,
        target_vision_frame: VisionFrame,
        mask_types: Optional[List[str]] = None,
        mask_blur: Optional[float] = None,
        mask_padding: Optional[Padding] = None,
        mask_areas: Optional[List[str]] = None,
        mask_regions: Optional[List[str]] = None,
        pixel_boost: Optional[str] = None,
        prepared_source_embedding: Optional[Embedding] = None
    ) -> VisionFrame:
        """
        Swaps a target face in the frame with identity from source_face.
        Applies pixel boost (if enabled), masking, and seamless affine paste-back.
        """
        template = self.cfg['template']
        model_size = self.cfg['size']
        model_type = self.cfg['type']
        mean = np.array(self.cfg['mean'], dtype=np.float32)
        std = np.array(self.cfg['std'], dtype=np.float32)

        pb_choice = pixel_boost or self.pixel_boost or 'none'
        if pb_choice not in ['none', None] and 'x' in pb_choice:
            pb_dim = int(pb_choice.split('x')[0])
            pixel_boost_size = (pb_dim, pb_dim)
        else:
            pixel_boost_size = model_size

        pixel_boost_total = max(1, pixel_boost_size[0] // model_size[0])

        # 1. Warp target face to canonical orientation (at pixel_boost_size)
        crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
            target_vision_frame,
            target_face.landmark_set['5/68'],
            template,
            pixel_boost_size
        )

        # 2. Prepare source embedding (reuse precomputed embedding for video performance)
        source_embedding = prepared_source_embedding if prepared_source_embedding is not None else self.prepare_source_embedding(source_face)
        if self.weight != 0.5 and target_face.embedding_norm is not None:
            source_embedding = self.balance_embedding(source_embedding, target_face)


        # 3. Swap inference (with Pixel Boost if total > 1)
        if pixel_boost_total > 1:
            tiles = implode_pixel_boost(crop_vision_frame, pixel_boost_total, model_size)
            swapped_tiles = []
            for tile in tiles:
                tile_prep = tile[:, :, ::-1].astype(np.float32) / 255.0
                tile_prep = (tile_prep - mean) / std
                tile_prep = np.expand_dims(tile_prep.transpose(2, 0, 1), axis=0)

                swapped_tile = self._forward_single_crop(tile_prep, source_embedding)
                swapped_tile = swapped_tile.transpose(1, 2, 0)
                if model_type in ['hyperswap', 'ghost', 'hififace', 'uniface']:
                    swapped_tile = swapped_tile * std + mean
                swapped_tile = (swapped_tile.clip(0, 1)[:, :, ::-1] * 255.0).astype(np.uint8)
                swapped_tiles.append(swapped_tile)

            swapped_crop = explode_pixel_boost(swapped_tiles, pixel_boost_total, model_size, pixel_boost_size)
        else:
            prep_crop = crop_vision_frame[:, :, ::-1].astype(np.float32) / 255.0
            prep_crop = (prep_crop - mean) / std
            prep_crop = np.expand_dims(prep_crop.transpose(2, 0, 1), axis=0)

            swapped_crop = self._forward_single_crop(prep_crop, source_embedding)
            swapped_crop = swapped_crop.transpose(1, 2, 0)
            if model_type in ['hyperswap', 'ghost', 'hififace', 'uniface']:
                swapped_crop = swapped_crop * std + mean
            swapped_crop = (swapped_crop.clip(0, 1)[:, :, ::-1] * 255.0).astype(np.uint8)

        # 4. Generate composite mask
        m_types = mask_types if mask_types is not None else self.mask_types
        m_blur = mask_blur if mask_blur is not None else self.mask_blur
        m_pad = mask_padding if mask_padding is not None else self.mask_padding

        if m_types == ['box'] or m_types == ('box',):
            mask = create_box_mask(swapped_crop, m_blur, m_pad)
        else:
            m_areas = mask_areas if mask_areas is not None else self.mask_areas
            m_regions = mask_regions if mask_regions is not None else self.mask_regions
            mask = self.masker.create_mask(
                swapped_crop,
                target_face=target_face,
                affine_matrix=affine_matrix,
                mask_types=m_types,
                mask_blur=m_blur,
                mask_padding=m_pad,
                mask_areas=m_areas,
                mask_regions=m_regions
            )

        # 5. Paste swapped crop seamlessly back onto the frame
        result_frame = paste_back(
            target_vision_frame,
            swapped_crop,
            mask,
            affine_matrix
        )

        return result_frame


