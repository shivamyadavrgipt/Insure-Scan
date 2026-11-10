"""First DIP module: grayscale conversion and contrast enhancement.

All functions accept/return OpenCV images (NumPy arrays). The module deliberately
keeps processing separate from file I/O so it can be tested and reused by FastAPI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np


def read_image(image_path: str | Path) -> np.ndarray:
    """Read an image from disk; raise a useful error if it cannot be decoded."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR or grayscale image to an 8-bit grayscale image."""
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("image must be a valid NumPy array")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def equalize_histogram(gray: np.ndarray) -> np.ndarray:
    """Apply global histogram equalization to a grayscale image."""
    gray = to_grayscale(gray)
    return cv2.equalizeHist(gray)


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE for local contrast enhancement, useful for uneven scans."""
    gray = to_grayscale(gray)
    if clip_limit <= 0:
        raise ValueError("clip_limit must be greater than zero")
    if len(tile_grid_size) != 2 or min(tile_grid_size) < 1:
        raise ValueError("tile_grid_size must contain two positive integers")
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=tuple(map(int, tile_grid_size))
    )
    return clahe.apply(gray)


def preprocess_contrast(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Return comparable grayscale, global equalization and CLAHE variants."""
    gray = to_grayscale(image)
    return {
        "grayscale": gray,
        "histogram_equalized": equalize_histogram(gray),
        "clahe": apply_clahe(gray),
    }


def save_preprocessing_results(
    image: np.ndarray, output_dir: str | Path, prefix: str = "document"
) -> Dict[str, Path]:
    """Save original and intermediate outputs; return their exact file paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}

    original_path = output_path / f"{prefix}_original.png"
    if not cv2.imwrite(str(original_path), image):
        raise OSError(f"Failed to save image: {original_path}")
    outputs["original"] = original_path

    for name, processed in preprocess_contrast(image).items():
        path = output_path / f"{prefix}_{name}.png"
        if not cv2.imwrite(str(path), processed):
            raise OSError(f"Failed to save image: {path}")
        outputs[name] = path
    return outputs
