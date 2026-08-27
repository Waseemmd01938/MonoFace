"""
Robust File & Model Download Module for MonoFace
Supports mirror fallback (GitHub, HuggingFace, HF Mirror), hash validation, and progress tracking.
"""

from typing import Dict, List, Optional, Tuple, Union, Callable
import os
import sys
import hashlib
import urllib.request
import urllib.error
import time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from urls import (
    MODEL_REGISTRY,
    get_model_entry,
    get_model_download_urls,
    get_hash_download_urls,
    resolve_download_url
)

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Models')
SHARED_CACHE_DIR = os.path.join("D:\\waseem\\ML\\facefusion\\.assets\\models")


def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """Calculates hash string of a local file in chunks."""
    h = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def validate_file_hash(file_path: str, hash_value_or_path: str) -> bool:
    """
    Validates file against an expected hash string or a .hash file.
    FaceFusion hash files typically contain 64-char sha256 or 40-char sha1 hex.
    """
    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        return False

    expected_hash = hash_value_or_path.strip().lower()
    if os.path.isfile(hash_value_or_path):
        try:
            with open(hash_value_or_path, 'r', encoding='utf-8') as f:
                expected_hash = f.read().strip().split()[0].lower()
        except Exception:
            return False

    algo = 'sha256' if len(expected_hash) == 64 else 'sha1' if len(expected_hash) == 40 else 'md5'
    actual_hash = calculate_file_hash(file_path, algorithm=algo)
    return actual_hash == expected_hash


class DownloadProgressBar:
    """Tqdm-based or basic stdout download progress reporter."""
    def __init__(self, description: str = "Downloading"):
        self.pbar = None
        self.description = description

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if total_size <= 0:
            return
        if self.pbar is None and tqdm is not None:
            self.pbar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=self.description,
                ascii=' ='
            )
        if self.pbar is not None:
            downloaded = block_num * block_size
            self.pbar.n = min(downloaded, total_size)
            self.pbar.refresh()

    def close(self):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


def download_file_from_urls(
    urls: List[str],
    destination_path: str,
    show_progress: bool = True,
    timeout: int = 20,
    retries: int = 3
) -> bool:
    """
    Downloads a file with automatic failover across multiple candidate URLs.
    Uses atomic write to prevent corrupted files.
    """
    os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)
    temp_destination = destination_path + ".tmp"
    filename = os.path.basename(destination_path)

    for url in urls:
        for attempt in range(1, retries + 1):
            progress_bar = DownloadProgressBar(description=f"[{filename}]") if show_progress else None
            try:
                # Open request with a user-agent
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=timeout) as response, open(temp_destination, 'wb') as out_file:
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    block_size = 65536

                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        out_file.write(buffer)
                        if progress_bar:
                            progress_bar(downloaded // block_size, block_size, total_size)

                if progress_bar:
                    progress_bar.close()

                if os.path.isfile(temp_destination) and os.path.getsize(temp_destination) > 0:
                    if os.path.isfile(destination_path):
                        os.remove(destination_path)
                    os.rename(temp_destination, destination_path)
                    return True

            except Exception as e:
                if progress_bar:
                    progress_bar.close()
                if os.path.isfile(temp_destination):
                    try:
                        os.remove(temp_destination)
                    except OSError:
                        pass
                if attempt < retries:
                    time.sleep(1.0)

    return False


def get_model_path(model_key_or_file: str, models_dir: Optional[str] = None) -> str:
    """Returns the local absolute path where the model file should reside."""
    entry = get_model_entry(model_key_or_file)
    file_name = entry['file'] if entry else model_key_or_file
    target_dir = models_dir or DEFAULT_MODELS_DIR
    return os.path.join(target_dir, file_name)


def is_model_downloaded(model_key_or_file: str, models_dir: Optional[str] = None) -> bool:
    """Checks if a model exists locally and has valid non-zero size."""
    path = get_model_path(model_key_or_file, models_dir)
    if os.path.isfile(path) and os.path.getsize(path) > 1024:
        return True
    # Also check shared cache directory if available
    entry = get_model_entry(model_key_or_file)
    file_name = entry['file'] if entry else model_key_or_file
    shared_path = os.path.join(SHARED_CACHE_DIR, file_name)
    return os.path.isfile(shared_path) and os.path.getsize(shared_path) > 1024


def download_model(
    model_key_or_file: str,
    models_dir: Optional[str] = None,
    force_download: bool = False,
    validate_hash: bool = False,
    show_progress: bool = True
) -> str:
    """
    Downloads model weights (.onnx) and optional hash file.
    Returns the resolved path to the local model file.
    """
    entry = get_model_entry(model_key_or_file)
    if not entry:
        raise ValueError(f"Model '{model_key_or_file}' is not registered in urls.py.")

    target_dir = models_dir or DEFAULT_MODELS_DIR
    os.makedirs(target_dir, exist_ok=True)

    file_name = entry['file']
    target_path = os.path.join(target_dir, file_name)

    # 1. Check local target path
    if not force_download and os.path.isfile(target_path) and os.path.getsize(target_path) > 1024:
        return target_path

    # 2. Check local shared cache from FaceFusion
    shared_path = os.path.join(SHARED_CACHE_DIR, file_name)
    if not force_download and os.path.isfile(shared_path) and os.path.getsize(shared_path) > 1024:
        return shared_path

    # 3. Download hash file if required
    hash_file_path = None
    if validate_hash and entry.get('hash_file'):
        hash_file_name = entry['hash_file']
        hash_file_path = os.path.join(target_dir, hash_file_name)
        if not os.path.isfile(hash_file_path):
            hash_urls = get_hash_download_urls(model_key_or_file)
            download_file_from_urls(hash_urls, hash_file_path, show_progress=False)

    # 4. Download ONNX model file from candidate URLs
    model_urls = get_model_download_urls(model_key_or_file)
    success = download_file_from_urls(model_urls, target_path, show_progress=show_progress)

    if not success or not (os.path.isfile(target_path) and os.path.getsize(target_path) > 1024):
        raise RuntimeError(f"Failed to download model weights for '{model_key_or_file}' from all mirror sources.")

    # 5. Validate hash if requested
    if validate_hash and hash_file_path and os.path.isfile(hash_file_path):
        if not validate_file_hash(target_path, hash_file_path):
            os.remove(target_path)
            raise ValueError(f"Hash validation failed for downloaded model: {file_name}")

    return target_path


def download_multiple_models(
    model_keys_or_files: List[str],
    models_dir: Optional[str] = None,
    show_progress: bool = True
) -> Dict[str, str]:
    """Batch downloads a list of models and returns a dictionary of {model_key: resolved_local_path}."""
    results = {}
    for key in model_keys_or_files:
        path = download_model(key, models_dir=models_dir, show_progress=show_progress)
        results[key] = path
    return results
