import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.dip.preprocessing import (  # noqa: E402
    apply_clahe,
    equalize_histogram,
    preprocess_contrast,
    to_grayscale,
)


def sample_bgr_image():
    image = np.full((120, 200, 3), 220, dtype=np.uint8)
    image[30:90, 30:170] = 80
    return image


def test_grayscale_output_shape_and_type():
    gray = to_grayscale(sample_bgr_image())
    assert gray.shape == (120, 200)
    assert gray.dtype == np.uint8


def test_equalization_and_clahe_preserve_dimensions():
    image = sample_bgr_image()
    gray = to_grayscale(image)
    assert equalize_histogram(gray).shape == gray.shape
    assert apply_clahe(gray).shape == gray.shape


def test_pipeline_returns_expected_stages():
    result = preprocess_contrast(sample_bgr_image())
    assert set(result) == {"grayscale", "histogram_equalized", "clahe"}
    assert all(value.dtype == np.uint8 for value in result.values())
