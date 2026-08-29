"""
Robust File & Model Download Module for MonoFace
Supports mirror fallback (GitHub, HuggingFace, HF Mirror), hash validation, and real-time progress tracking.
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

# Global download callback: fn(model_name: str, percent: float, downloaded_bytes: int, total_bytes: int, speed_mb: float)
_DOWNLOAD_CALLBACK: Optional[Callable[[str, float, int, int, float], None]] = None


def set_download_callback(callback: Optional[Callable[[str, float, int, int, float], None]]) -> None:
    """Sets a global callback for tracking download progress across the application."""
    global _DOWNLOAD_CALLBACK
    _DOWNLOAD_CALLBACK = callback


def get_download_callback() -> Optional[Callable[[str, float, int, int, float], None]]:
    """Gets the active global download progress callback."""
    return _DOWNLOAD_CALLBACK


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
    """High-accuracy download progress reporter supporting tqdm, stdout, and live UI callbacks."""
    def __init__(self, model_name: str, description: Optional[str] = None):
        self.model_name = model_name
        self.description = description or f"[{model_name}]"
        self.pbar = None
        self.start_time = time.time()
        self.last_update_time = 0.0

    def __call__(self, downloaded: int, total_size: int, custom_callback: Optional[Callable] = None):
        if total_size <= 0:
            return

        now = time.time()
        elapsed = max(now - self.start_time, 0.001)
        speed_mb = (downloaded / (1024 * 1024)) / elapsed
        percent = min(100.0, (downloaded / total_size) * 100.0)

        # Tqdm bar for console (single in-place updating line)
        if self.pbar is None and tqdm is not None:
            self.pbar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=self.description,
                ascii=' =',
                file=sys.stdout,
                dynamic_ncols=True,
                leave=False
            )
        if self.pbar is not None:
            self.pbar.n = min(downloaded, total_size)
            self.pbar.refresh()

        # Update global & custom callback at throttled intervals
        if now - self.last_update_time >= 0.1 or downloaded >= total_size:
            self.last_update_time = now
            if custom_callback:
                try:
                    custom_callback(self.model_name, percent, downloaded, total_size, speed_mb)
                except Exception:
                    pass
            if _DOWNLOAD_CALLBACK and _DOWNLOAD_CALLBACK != custom_callback:
                try:
                    _DOWNLOAD_CALLBACK(self.model_name, percent, downloaded, total_size, speed_mb)
                except Exception:
                    pass

    def close(self):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


def download_file_from_urls(
    urls: List[str],
    destination_path: str,
    model_name: Optional[str] = None,
    show_progress: bool = True,
    progress_callback: Optional[Callable[[str, float, int, int, float], None]] = None,
    timeout: int = 30,
    retries: int = 3
) -> bool:
    """
    Downloads a file with automatic failover across candidate mirror URLs.
    Uses atomic write and live progress reporting with model name.
    """
    os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)
    temp_destination = destination_path + ".tmp"
    filename = os.path.basename(destination_path)
    display_name = model_name or filename

    print(f"[DOWNLOAD] Starting download for '{display_name}' -> {destination_path}")

    for url in urls:
        for attempt in range(1, retries + 1):
            progress_bar = DownloadProgressBar(model_name=display_name) if show_progress else None
            try:
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
                            progress_bar(downloaded, total_size, custom_callback=progress_callback)

                if progress_bar:
                    progress_bar.close()

                if os.path.isfile(temp_destination) and os.path.getsize(temp_destination) > 0:
                    if os.path.isfile(destination_path):
                        os.remove(destination_path)
                    os.rename(temp_destination, destination_path)
                    print(f"[DOWNLOAD] Successfully downloaded '{display_name}' ({os.path.getsize(destination_path) / (1024*1024):.1f} MB)")
                    return True

            except Exception as e:
                if progress_bar:
                    progress_bar.close()
                if os.path.isfile(temp_destination):
                    try:
                        os.remove(temp_destination)
                    except OSError:
                        pass
                print(f"[DOWNLOAD] Attempt {attempt} failed for {url} ({e})")
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
    """Checks if a model exists locally in the Models directory and has valid non-zero size."""
    path = get_model_path(model_key_or_file, models_dir)
    return os.path.isfile(path) and os.path.getsize(path) > 1024


def download_model(
    model_key_or_file: str,
    models_dir: Optional[str] = None,
    force_download: bool = False,
    validate_hash: bool = False,
    show_progress: bool = True,
    progress_callback: Optional[Callable[[str, float, int, int, float], None]] = None
) -> str:
    """
    Downloads model weights (.onnx) and optional hash file directly to local Models directory.
    Provides live progress bar and callbacks with the model name.
    Returns the resolved path to the local model file.
    """
    entry = get_model_entry(model_key_or_file)
    target_dir = models_dir or DEFAULT_MODELS_DIR
    os.makedirs(target_dir, exist_ok=True)

    if entry:
        file_name = entry['file']
        model_name = model_key_or_file
    else:
        file_name = model_key_or_file
        model_name = os.path.splitext(model_key_or_file)[0]

    target_path = os.path.join(target_dir, file_name)

    # 1. Check local target path
    if not force_download and os.path.isfile(target_path) and os.path.getsize(target_path) > 1024:
        return target_path

    # 2. Download hash file if required
    hash_file_path = None
    if validate_hash and entry and entry.get('hash_file'):
        hash_file_name = entry['hash_file']
        hash_file_path = os.path.join(target_dir, hash_file_name)
        if not os.path.isfile(hash_file_path):
            hash_urls = get_hash_download_urls(model_key_or_file)
            download_file_from_urls(hash_urls, hash_file_path, model_name=f"{model_name}.hash", show_progress=False)

    # 3. Download ONNX model file from candidate URLs
    if entry:
        model_urls = get_model_download_urls(model_key_or_file)
    else:
        from urls import get_all_candidate_urls
        model_urls = get_all_candidate_urls('models-3.0.0', file_name)

    success = download_file_from_urls(
        urls=model_urls,
        destination_path=target_path,
        model_name=model_name,
        show_progress=show_progress,
        progress_callback=progress_callback
    )

    if not success or not (os.path.isfile(target_path) and os.path.getsize(target_path) > 1024):
        raise RuntimeError(f"Failed to download model weights for '{model_name}' from all mirror sources.")

    # 5. Validate hash if requested
    if validate_hash and hash_file_path and os.path.isfile(hash_file_path):
        if not validate_file_hash(target_path, hash_file_path):
            os.remove(target_path)
            raise ValueError(f"Hash validation failed for downloaded model: {file_name}")

    return target_path


def download_multiple_models(
    model_keys_or_files: List[str],
    models_dir: Optional[str] = None,
    show_progress: bool = True,
    progress_callback: Optional[Callable[[str, float, int, int, float], None]] = None
) -> Dict[str, str]:
    """Batch downloads a list of models and returns a dictionary of {model_key: resolved_local_path}."""
    results = {}
    for key in model_keys_or_files:
        path = download_model(key, models_dir=models_dir, show_progress=show_progress, progress_callback=progress_callback)
        results[key] = path
    return results
