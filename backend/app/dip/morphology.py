"""Phase 3: morphological operations for binary document images."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np


def _as_binary(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty array")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError("image must be a 2-D grayscale/binary image")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    # Ensure a consistent binary representation (background=255, ink=0
    # is common for documents; OpenCV morphology operates on white foreground).
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary


def structuring_element(size: int = 3, shape: str = "rect") -> np.ndarray:
    """Create a square structuring element: rect, ellipse, or cross."""
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    shapes = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    if shape not in shapes:
        raise ValueError("shape must be 'rect', 'ellipse', or 'cross'")
    return cv2.getStructuringElement(shapes[shape], (size, size))


def erosion(binary: np.ndarray, size: int = 3, shape: str = "rect") -> np.ndarray:
    return cv2.erode(_as_binary(binary), structuring_element(size, shape), iterations=1)


def dilation(binary: np.ndarray, size: int = 3, shape: str = "rect") -> np.ndarray:
    return cv2.dilate(_as_binary(binary), structuring_element(size, shape), iterations=1)


def opening(binary: np.ndarray, size: int = 3, shape: str = "rect") -> np.ndarray:
    """Erosion followed by dilation; removes small white foreground objects."""
    return cv2.morphologyEx(
        _as_binary(binary), cv2.MORPH_OPEN, structuring_element(size, shape)
    )


def closing(binary: np.ndarray, size: int = 3, shape: str = "rect") -> np.ndarray:
    """Dilation followed by erosion; fills small gaps in white foreground."""
    return cv2.morphologyEx(
        _as_binary(binary), cv2.MORPH_CLOSE, structuring_element(size, shape)
    )


def morphological_gradient(binary: np.ndarray, size: int = 3, shape: str = "rect") -> np.ndarray:
    """Difference between dilation and erosion; highlights object boundaries."""
    return cv2.morphologyEx(
        _as_binary(binary), cv2.MORPH_GRADIENT, structuring_element(size, shape)
    )


def run_morphology_experiment(
    binary: np.ndarray, sizes: tuple[int, ...] = (3, 5, 7), shape: str = "rect"
) -> Dict[str, np.ndarray]:
    """Generate comparable morphology outputs for each requested kernel size."""
    source = _as_binary(binary)
    outputs: Dict[str, np.ndarray] = {"input_binary": source}
    for size in sizes:
        if size < 1 or size % 2 == 0:
            raise ValueError("all structuring-element sizes must be positive odd integers")
        outputs[f"erosion_{size}x{size}"] = erosion(source, size, shape)
        outputs[f"dilation_{size}x{size}"] = dilation(source, size, shape)
        outputs[f"opening_{size}x{size}"] = opening(source, size, shape)
        outputs[f"closing_{size}x{size}"] = closing(source, size, shape)
        outputs[f"gradient_{size}x{size}"] = morphological_gradient(source, size, shape)
    return outputs


def foreground_statistics(binary: np.ndarray) -> dict[str, float | int]:
    """Return simple binary-image measurements (white pixels are foreground).

    For black-ink/white-paper document images, invert before calling this
    function if ink should be treated as foreground.
    """
    image = _as_binary(binary)
    foreground = image == 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    component_areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([])
    return {
        "foreground_pixels": int(foreground.sum()),
        "foreground_ratio": float(foreground.mean()),
        "connected_components": int(max(count - 1, 0)),
        "small_components_le_10px": int(np.sum(component_areas <= 10)),
    }


def save_morphology_results(
    binary: np.ndarray, output_dir: str | Path, prefix: str = "phase3",
    sizes: tuple[int, ...] = (3, 5, 7), shape: str = "rect"
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    images = run_morphology_experiment(binary, sizes=sizes, shape=shape)
    saved: dict[str, Path] = {}
    for name, image in images.items():
        path = output / f"{prefix}_{name}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write {path}")
        saved[name] = path
    return saved
