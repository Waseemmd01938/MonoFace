"""
Model URLs and Download Source Registry for MonoFace
Derived from FaceFusion asset distribution infrastructure.
"""

from typing import Dict, List, Optional, Tuple, Any

DOWNLOAD_PROVIDERS: Dict[str, Dict[str, Any]] = {
    'github': {
        'urls': [
            'https://github.com/facefusion/facefusion-assets/releases/download/{tag}/{file_name}'
        ]
    },
    'huggingface': {
        'urls': [
            'https://huggingface.co/facefusion/facefusion-assets/resolve/main/{tag}/{file_name}'
        ]
    },
    'hf_mirror': {
        'urls': [
            'https://hf-mirror.com/facefusion/facefusion-assets/resolve/main/{tag}/{file_name}'
        ]
    }
}


def resolve_download_url(tag: str, file_name: str, provider: str = 'github') -> str:
    """Resolves download URL for a specific provider and release tag."""
    if provider not in DOWNLOAD_PROVIDERS:
        provider = 'github'
    template = DOWNLOAD_PROVIDERS[provider]['urls'][0]
    return template.format(tag=tag, file_name=file_name)


def get_all_candidate_urls(tag: str, file_name: str) -> List[str]:
    """Returns candidate download URLs across all known mirror providers."""
    return [
        resolve_download_url(tag, file_name, 'github'),
        resolve_download_url(tag, file_name, 'huggingface'),
        resolve_download_url(tag, file_name, 'hf_mirror')
    ]


# Complete Catalog of Models, Release Tags, Hash Files, and URLs
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------
    # 1. Face Detectors
    # -------------------------------------------------------------
    'retinaface_10g': {
        'category': 'face_detector',
        'file': 'retinaface_10g.onnx',
        'hash_file': 'retinaface_10g.hash',
        'tag': 'models-3.0.0',
        'size': (640, 640),
        'vendor': 'InsightFace'
    },
    'scrfd_2.5g': {
        'category': 'face_detector',
        'file': 'scrfd_2.5g.onnx',
        'hash_file': 'scrfd_2.5g.hash',
        'tag': 'models-3.0.0',
        'size': (640, 640),
        'vendor': 'InsightFace'
    },
    'yoloface_8n': {
        'category': 'face_detector',
        'file': 'yoloface_8n.onnx',
        'hash_file': 'yoloface_8n.hash',
        'tag': 'models-3.0.0',
        'size': (640, 640),
        'vendor': 'derronqi'
    },
    'yunet_2023_mar': {
        'category': 'face_detector',
        'file': 'yunet_2023_mar.onnx',
        'hash_file': 'yunet_2023_mar.hash',
        'tag': 'models-3.4.0',
        'size': (640, 640),
        'vendor': 'OpenCV'
    },

    # -------------------------------------------------------------
    # 2. Face Landmarkers
    # -------------------------------------------------------------
    '2dfan4': {
        'category': 'face_landmarker',
        'file': '2dfan4.onnx',
        'hash_file': '2dfan4.hash',
        'tag': 'models-3.0.0',
        'size': (256, 256),
        'vendor': 'breadbread1984'
    },
    'peppa_wutz': {
        'category': 'face_landmarker',
        'file': 'peppa_wutz.onnx',
        'hash_file': 'peppa_wutz.hash',
        'tag': 'models-3.0.0',
        'size': (256, 256),
        'vendor': 'Unknown'
    },
    'fan_68_5': {
        'category': 'face_landmarker',
        'file': 'fan_68_5.onnx',
        'hash_file': 'fan_68_5.hash',
        'tag': 'models-3.0.0',
        'size': None,
        'vendor': 'FaceFusion'
    },

    # -------------------------------------------------------------
    # 3. Face Recognizers
    # -------------------------------------------------------------
    'arcface_w600k_r50': {
        'category': 'face_recognizer',
        'file': 'arcface_w600k_r50.onnx',
        'hash_file': 'arcface_w600k_r50.hash',
        'tag': 'models-3.0.0',
        'template': 'arcface_112_v2',
        'size': (112, 112),
        'vendor': 'InsightFace'
    },

    # -------------------------------------------------------------
    # 4. Face Classifiers (Gender, Age, Race)
    # -------------------------------------------------------------
    'fairface': {
        'category': 'face_classifier',
        'file': 'fairface.onnx',
        'hash_file': 'fairface.hash',
        'tag': 'models-3.0.0',
        'template': 'arcface_112_v2',
        'size': (224, 224),
        'vendor': 'dchen236'
    },

    # -------------------------------------------------------------
    # 5. Face Maskers & Occlusion
    # -------------------------------------------------------------
    'face_occluder': {
        'category': 'face_masker',
        'file': 'face_occluder.onnx',
        'hash_file': 'face_occluder.hash',
        'tag': 'models-3.0.0',
        'size': (256, 256),
        'vendor': 'FaceFusion'
    },
    'face_parser': {
        'category': 'face_masker',
        'file': 'face_parser.onnx',
        'hash_file': 'face_parser.hash',
        'tag': 'models-3.0.0',
        'size': (512, 512),
        'vendor': 'FaceFusion'
    },
    'bisenet_resnet_34': {
        'category': 'face_masker',
        'file': 'bisenet_resnet_34.onnx',
        'hash_file': 'bisenet_resnet_34.hash',
        'tag': 'models-3.0.0',
        'size': (512, 512),
        'vendor': 'zllrunning'
    },
    'birefnet_general': {
        'category': 'face_masker',
        'file': 'birefnet_general.onnx',
        'hash_file': 'birefnet_general.hash',
        'tag': 'models-3.3.0',
        'size': (1024, 1024),
        'vendor': 'ZhengPeng7'
    },
    'rmbg': {
        'category': 'face_masker',
        'file': 'rmbg.onnx',
        'hash_file': 'rmbg.hash',
        'tag': 'models-3.2.0',
        'size': (1024, 1024),
        'vendor': 'briaai'
    },

    # -------------------------------------------------------------
    # 6. Face Swappers
    # -------------------------------------------------------------
    'inswapper_128': {
        'category': 'face_swapper',
        'file': 'inswapper_128.onnx',
        'hash_file': 'inswapper_128.hash',
        'tag': 'models-3.0.0',
        'template': 'arcface_128',
        'size': (128, 128),
        'vendor': 'InsightFace'
    },
    'inswapper_128_fp16': {
        'category': 'face_swapper',
        'file': 'inswapper_128_fp16.onnx',
        'hash_file': 'inswapper_128_fp16.hash',
        'tag': 'models-3.0.0',
        'template': 'arcface_128',
        'size': (128, 128),
        'vendor': 'InsightFace'
    },
    'hyperswap_1a_256': {
        'category': 'face_swapper',
        'file': 'hyperswap_1a_256.onnx',
        'hash_file': 'hyperswap_1a_256.hash',
        'tag': 'models-3.3.0',
        'template': 'arcface_128',
        'size': (256, 256),
        'vendor': 'FaceFusion'
    },
    'hyperswap_1b_256': {
        'category': 'face_swapper',
        'file': 'hyperswap_1b_256.onnx',
        'hash_file': 'hyperswap_1b_256.hash',
        'tag': 'models-3.3.0',
        'template': 'arcface_128',
        'size': (256, 256),
        'vendor': 'FaceFusion'
    },
    'hyperswap_1c_256': {
        'category': 'face_swapper',
        'file': 'hyperswap_1c_256.onnx',
        'hash_file': 'hyperswap_1c_256.hash',
        'tag': 'models-3.3.0',
        'template': 'arcface_128',
        'size': (256, 256),
        'vendor': 'FaceFusion'
    },
    'simswap_256': {
        'category': 'face_swapper',
        'file': 'simswap_256.onnx',
        'hash_file': 'simswap_256.hash',
        'tag': 'models-3.0.0',
        'template': 'arcface_112_v1',
        'size': (256, 256),
        'vendor': 'neuralchen'
    },
    'simswap_512_unofficial': {
        'category': 'face_swapper',
        'file': 'simswap_512_unofficial.onnx',
        'hash_file': 'simswap_512_unofficial.hash',
        'tag': 'models-3.0.0',
        'template': 'arcface_112_v1',
        'size': (512, 512),
        'vendor': 'neuralchen'
    },

    # -------------------------------------------------------------
    # 7. Face Enhancers / Restorers
    # -------------------------------------------------------------
    'codeformer': {
        'category': 'face_enhancer',
        'file': 'codeformer.onnx',
        'hash_file': 'codeformer.hash',
        'tag': 'models-3.0.0',
        'template': 'ffhq_512',
        'size': (512, 512),
        'vendor': 'sczhou'
    },
    'gfpgan_1.2': {
        'category': 'face_enhancer',
        'file': 'gfpgan_1.2.onnx',
        'hash_file': 'gfpgan_1.2.hash',
        'tag': 'models-3.0.0',
        'template': 'ffhq_512',
        'size': (512, 512),
        'vendor': 'TencentARC'
    },
    'gfpgan_1.3': {
        'category': 'face_enhancer',
        'file': 'gfpgan_1.3.onnx',
        'hash_file': 'gfpgan_1.3.hash',
        'tag': 'models-3.0.0',
        'template': 'ffhq_512',
        'size': (512, 512),
        'vendor': 'TencentARC'
    },
    'gfpgan_1.4': {
        'category': 'face_enhancer',
        'file': 'gfpgan_1.4.onnx',
        'hash_file': 'gfpgan_1.4.hash',
        'tag': 'models-3.0.0',
        'template': 'ffhq_512',
        'size': (512, 512),
        'vendor': 'TencentARC'
    },
    'gpen_bfr_256': {
        'category': 'face_enhancer',
        'file': 'gpen_bfr_256.onnx',
        'hash_file': 'gpen_bfr_256.hash',
        'tag': 'models-3.0.0',
        'template': 'ffhq_512',
        'size': (256, 256),
        'vendor': 'yangxy'
    },
    'gpen_bfr_512': {
        'category': 'face_enhancer',
        'file': 'gpen_bfr_512.onnx',
        'hash_file': 'gpen_bfr_512.hash',
        'tag': 'models-3.0.0',
        'template': 'ffhq_512',
        'size': (512, 512),
        'vendor': 'yangxy'
    },
    'gpen_bfr_1024': {
        'category': 'face_enhancer',
        'file': 'gpen_bfr_1024.onnx',
        'hash_file': 'gpen_bfr_1024.hash',
        'tag': 'models-3.0.0',
        'template': 'ffhq_512',
        'size': (1024, 1024),
        'vendor': 'yangxy'
    },
    'gpen_bfr_2048': {
        'category': 'face_enhancer',
        'file': 'gpen_bfr_2048.onnx',
        'hash_file': 'gpen_bfr_2048.hash',
        'tag': 'models-3.0.0',
        'template': 'ffhq_512',
        'size': (2048, 2048),
        'vendor': 'yangxy'
    },
    'restoreformer_plus_plus': {
        'category': 'face_enhancer',
        'file': 'restoreformer_plus_plus.onnx',
        'hash_file': 'restoreformer_plus_plus.hash',
        'tag': 'models-3.0.0',
        'template': 'ffhq_512',
        'size': (512, 512),
        'vendor': 'wzhouxidian'
    },

    # -------------------------------------------------------------
    # 8. Frame Enhancers / Super-Resolution
    # -------------------------------------------------------------
    'real_esrgan_x2': {
        'category': 'frame_enhancer',
        'file': 'real_esrgan_x2.onnx',
        'hash_file': 'real_esrgan_x2.hash',
        'tag': 'models-3.0.0',
        'scale': 2
    },
    'real_esrgan_x4': {
        'category': 'frame_enhancer',
        'file': 'real_esrgan_x4.onnx',
        'hash_file': 'real_esrgan_x4.hash',
        'tag': 'models-3.0.0',
        'scale': 4
    },
    'real_esrgan_x8': {
        'category': 'frame_enhancer',
        'file': 'real_esrgan_x8.onnx',
        'hash_file': 'real_esrgan_x8.hash',
        'tag': 'models-3.0.0',
        'scale': 8
    },
    'real_esrnet_x4': {
        'category': 'frame_enhancer',
        'file': 'real_esrnet_x4.onnx',
        'hash_file': 'real_esrnet_x4.hash',
        'tag': 'models-3.0.0',
        'scale': 4
    },
    'real_hatgan_x4': {
        'category': 'frame_enhancer',
        'file': 'real_hatgan_x4.onnx',
        'hash_file': 'real_hatgan_x4.hash',
        'tag': 'models-3.0.0',
        'scale': 4
    },
    'clear_reality_x4': {
        'category': 'frame_enhancer',
        'file': 'clear_reality_x4.onnx',
        'hash_file': 'clear_reality_x4.hash',
        'tag': 'models-3.0.0',
        'scale': 4
    },
    'ultra_sharp_x4': {
        'category': 'frame_enhancer',
        'file': 'ultra_sharp_x4.onnx',
        'hash_file': 'ultra_sharp_x4.hash',
        'tag': 'models-3.0.0',
        'scale': 4
    },
    'lsdir_x4': {
        'category': 'frame_enhancer',
        'file': 'lsdir_x4.onnx',
        'hash_file': 'lsdir_x4.hash',
        'tag': 'models-3.0.0',
        'scale': 4
    },
    'nomos8k_sc_x4': {
        'category': 'frame_enhancer',
        'file': 'nomos8k_sc_x4.onnx',
        'hash_file': 'nomos8k_sc_x4.hash',
        'tag': 'models-3.0.0',
        'scale': 4
    },
    'span_kendata_x4': {
        'category': 'frame_enhancer',
        'file': 'span_kendata_x4.onnx',
        'hash_file': 'span_kendata_x4.hash',
        'tag': 'models-3.0.0',
        'scale': 4
    },

    # -------------------------------------------------------------
    # 11. LivePortrait / Face Editor / Expression Restorer
    # -------------------------------------------------------------
    'live_portrait_appearance_feature_extractor': {
        'category': 'live_portrait',
        'file': 'live_portrait_appearance_feature_extractor.onnx',
        'hash_file': 'live_portrait_appearance_feature_extractor.hash',
        'tag': 'models-3.2.0'
    },
    'live_portrait_motion_extractor': {
        'category': 'live_portrait',
        'file': 'live_portrait_motion_extractor.onnx',
        'hash_file': 'live_portrait_motion_extractor.hash',
        'tag': 'models-3.2.0'
    },
    'live_portrait_spade_generator': {
        'category': 'live_portrait',
        'file': 'live_portrait_spade_generator.onnx',
        'hash_file': 'live_portrait_spade_generator.hash',
        'tag': 'models-3.2.0'
    },
    'live_portrait_warping_spade_generator': {
        'category': 'live_portrait',
        'file': 'live_portrait_warping_spade_generator.onnx',
        'hash_file': 'live_portrait_warping_spade_generator.hash',
        'tag': 'models-3.2.0'
    },
    'live_portrait_stitching': {
        'category': 'live_portrait',
        'file': 'live_portrait_stitching.onnx',
        'hash_file': 'live_portrait_stitching.hash',
        'tag': 'models-3.2.0'
    },
    'live_portrait_eye_retargeting': {
        'category': 'live_portrait',
        'file': 'live_portrait_eye_retargeting.onnx',
        'hash_file': 'live_portrait_eye_retargeting.hash',
        'tag': 'models-3.2.0'
    },
    'live_portrait_lip_retargeting': {
        'category': 'live_portrait',
        'file': 'live_portrait_lip_retargeting.onnx',
        'hash_file': 'live_portrait_lip_retargeting.hash',
        'tag': 'models-3.2.0'
    },

}


def get_model_entry(model_name_or_file: str) -> Optional[Dict[str, Any]]:
    """Fetches model info dictionary by model key or file name."""
    if model_name_or_file in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name_or_file]
    for key, info in MODEL_REGISTRY.items():
        if info.get('file') == model_name_or_file:
            return info
    return None


def get_model_download_urls(model_key_or_file: str) -> List[str]:
    """Returns candidate download URLs for the given model key or file name."""
    entry = get_model_entry(model_key_or_file)
    if not entry:
        raise ValueError(f"Unknown model: {model_key_or_file}")
    return get_all_candidate_urls(entry['tag'], entry['file'])


def get_hash_download_urls(model_key_or_file: str) -> List[str]:
    """Returns candidate download URLs for the corresponding model hash file."""
    entry = get_model_entry(model_key_or_file)
    if not entry or not entry.get('hash_file'):
        return []
    return get_all_candidate_urls(entry['tag'], entry['hash_file'])
