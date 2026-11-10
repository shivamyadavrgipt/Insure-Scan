"""Phase 2 DIP operations: denoising and document binarization."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from .preprocessing import to_grayscale, apply_clahe


def _validate_kernel(kernel_size: int) -> None:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")


def gaussian_denoise(gray: np.ndarray, kernel_size: int = 3, sigma: float = 0) -> np.ndarray:
    """Reduce approximately Gaussian noise; may soften fine character strokes."""
    gray = to_grayscale(gray)
    _validate_kernel(kernel_size)
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigmaX=sigma)


def median_denoise(gray: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Reduce impulse/salt-and-pepper noise while reasonably preserving edges."""
    gray = to_grayscale(gray)
    _validate_kernel(kernel_size)
    return cv2.medianBlur(gray, kernel_size)


def bilateral_denoise(gray: np.ndarray, diameter: int = 5,
                      sigma_color: float = 50, sigma_space: float = 50) -> np.ndarray:
    """Smooth noise while attempting to preserve edges; slower than Gaussian/median."""
    gray = to_grayscale(gray)
    if diameter < 1 or sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("diameter and bilateral sigmas must be positive")
    return cv2.bilateralFilter(gray, diameter, sigma_color, sigma_space)


def global_threshold(gray: np.ndarray, threshold: int = 127,
                     max_value: int = 255) -> np.ndarray:
    """Apply one fixed intensity cutoff to the complete image."""
    gray = to_grayscale(gray)
    if not 0 <= threshold <= 255 or not 1 <= max_value <= 255:
        raise ValueError("threshold must be 0..255 and max_value must be 1..255")
    return cv2.threshold(gray, threshold, max_value, cv2.THRESH_BINARY)[1]


def otsu_threshold(gray: np.ndarray, max_value: int = 255) -> tuple[int, np.ndarray]:
    """Automatically select a global threshold using Otsu's method."""
    gray = to_grayscale(gray)
    if not 1 <= max_value <= 255:
        raise ValueError("max_value must be 1..255")
    threshold, result = cv2.threshold(
        gray, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return int(round(threshold)), result


def _validate_adaptive(block_size: int, max_value: int) -> None:
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer >= 3")
    if not 1 <= max_value <= 255:
        raise ValueError("max_value must be 1..255")


def adaptive_mean_threshold(gray: np.ndarray, block_size: int = 31,
                            constant: int = 10, max_value: int = 255) -> np.ndarray:
    """Use the local neighbourhood mean as the threshold reference."""
    gray = to_grayscale(gray)
    _validate_adaptive(block_size, max_value)
    return cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, block_size, constant)


def adaptive_gaussian_threshold(gray: np.ndarray, block_size: int = 31,
                                 constant: int = 10, max_value: int = 255) -> np.ndarray:
    """Use a Gaussian-weighted local neighbourhood as threshold reference."""
    gray = to_grayscale(gray)
    _validate_adaptive(block_size, max_value)
    return cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, constant)


def run_phase2(gray: np.ndarray) -> Dict[str, np.ndarray]:
    """Return denoising and threshold variants for controlled comparison.

    Threshold methods operate on CLAHE-enhanced median-denoised grayscale.
    Denoising outputs are also retained independently for visual comparison.
    """
    gray = to_grayscale(gray)
    gaussian = gaussian_denoise(gray)
    median = median_denoise(gray)
    bilateral = bilateral_denoise(gray)
    enhanced = apply_clahe(median)
    _, otsu = otsu_threshold(enhanced)
    return {
        "gaussian_denoised": gaussian,
        "median_denoised": median,
        "bilateral_denoised": bilateral,
        "threshold_global": global_threshold(enhanced),
        "threshold_otsu": otsu,
        "threshold_adaptive_mean": adaptive_mean_threshold(enhanced),
        "threshold_adaptive_gaussian": adaptive_gaussian_threshold(enhanced),
    }


def save_phase2_results(gray: np.ndarray, output_dir: str | Path,
                        prefix: str = "document") -> Dict[str, Path]:
    """Save Phase 2 intermediate images and return exact output paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}
    for name, image in run_phase2(gray).items():
        path = output_path / f"{prefix}_{name}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Failed to save image: {path}")
        saved[name] = path
    return saved
