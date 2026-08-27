"""
Model Store & Memory Management Module for MonoFace
Provides centralized ONNX Runtime session pooling, optimal memory configurations,
and cache deallocation helpers.
"""

from typing import Any, Dict, List, Optional, Tuple
import gc
import os
import onnxruntime

# Global session cache: (model_path, providers_tuple) -> InferenceSession
_SESSION_CACHE: Dict[Tuple[str, Tuple[str, ...]], onnxruntime.InferenceSession] = {}


_VERIFIED_PROVIDERS: Optional[List[str]] = None


def get_default_providers() -> List[str]:
    """Returns verified and working execution providers in priority order."""
    global _VERIFIED_PROVIDERS
    if _VERIFIED_PROVIDERS is not None:
        return _VERIFIED_PROVIDERS

    available = onnxruntime.get_available_providers()
    providers: List[str] = []

    if 'CUDAExecutionProvider' in available:
        providers.append('CUDAExecutionProvider')
    if 'CPUExecutionProvider' in available or not providers:
        providers.append('CPUExecutionProvider')

    _VERIFIED_PROVIDERS = providers
    return _VERIFIED_PROVIDERS


CUDA_PROVIDER_OPTIONS = {
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'cudnn_conv_algo_search': 'DEFAULT',
    'do_copy_in_default_stream': True,
}


def create_optimized_session_options() -> onnxruntime.SessionOptions:
    """Creates ONNX Runtime SessionOptions configured for optimal memory and speed."""
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    opts.log_severity_level = 3  # Suppress internal memcpy/graph warnings
    return opts


def _format_providers(prov_list: List[str]) -> List[Any]:
    """Formats provider list with high-performance CUDA provider options."""
    formatted = []
    for p in prov_list:
        if p == 'CUDAExecutionProvider':
            formatted.append(('CUDAExecutionProvider', CUDA_PROVIDER_OPTIONS))
        else:
            formatted.append(p)
    return formatted


def get_inference_session(
    model_path: str,
    providers: Optional[List[str]] = None,
    session_options: Optional[onnxruntime.SessionOptions] = None
) -> onnxruntime.InferenceSession:
    """
    Retrieves a cached InferenceSession or creates and registers a new one.
    Prevents redundant model re-instantiation and memory bloat.
    Gracefully falls back to CPU if GPU providers lack system CUDA shared libraries.
    """
    global _VERIFIED_PROVIDERS
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    prov_list = providers if providers is not None else get_default_providers()
    prov_key = tuple(prov_list)
    cache_key = (os.path.abspath(model_path), prov_key)

    if cache_key in _SESSION_CACHE:
        return _SESSION_CACHE[cache_key]

    opts = session_options or create_optimized_session_options()
    formatted_providers = _format_providers(prov_list)

    try:
        session = onnxruntime.InferenceSession(model_path, sess_options=opts, providers=formatted_providers)
    except Exception as e:
        # If CUDA library failed to load (e.g. missing libcublasLt or CUDA mismatch), fallback to CPU
        if 'CUDAExecutionProvider' in prov_list:
            _VERIFIED_PROVIDERS = ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        else:
            raise e

    _SESSION_CACHE[cache_key] = session
    return session




def clear_session_cache() -> None:
    """Clears all cached ONNX sessions and frees allocated memory."""
    global _SESSION_CACHE
    _SESSION_CACHE.clear()
    gc.collect()


def free_memory() -> None:
    """Triggers Python garbage collection and clears caches."""
    gc.collect()
