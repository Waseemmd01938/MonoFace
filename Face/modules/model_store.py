"""
Model Store & Memory Management Module for MonoFace
Provides centralized ONNX Runtime session pooling, optimal memory configurations,
and cache deallocation helpers.
"""

from typing import Dict, List, Optional, Tuple
import gc
import os
import onnxruntime

# Global session cache: (model_path, providers_tuple) -> InferenceSession
_SESSION_CACHE: Dict[Tuple[str, Tuple[str, ...]], onnxruntime.InferenceSession] = {}


def get_default_providers() -> List[str]:
    """Returns available execution providers in priority order."""
    available = onnxruntime.get_available_providers()
    providers = []
    if 'CUDAExecutionProvider' in available:
        providers.append('CUDAExecutionProvider')
    if 'CPUExecutionProvider' in available:
        providers.append('CPUExecutionProvider')
    return providers if providers else ['CPUExecutionProvider']


def create_optimized_session_options() -> onnxruntime.SessionOptions:
    """Creates ONNX Runtime SessionOptions configured for optimal memory and speed."""
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    return opts


def get_inference_session(
    model_path: str,
    providers: Optional[List[str]] = None,
    session_options: Optional[onnxruntime.SessionOptions] = None
) -> onnxruntime.InferenceSession:
    """
    Retrieves a cached InferenceSession or creates and registers a new one.
    Prevents redundant model re-instantiation and memory bloat.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    prov_list = providers if providers is not None else get_default_providers()
    prov_key = tuple(prov_list)
    cache_key = (os.path.abspath(model_path), prov_key)

    if cache_key in _SESSION_CACHE:
        return _SESSION_CACHE[cache_key]

    opts = session_options or create_optimized_session_options()
    session = onnxruntime.InferenceSession(model_path, sess_options=opts, providers=prov_list)
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
