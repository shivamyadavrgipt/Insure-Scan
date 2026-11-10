import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.dip.morphology import (  # noqa: E402
    closing, dilation, erosion, foreground_statistics, opening,
    run_morphology_experiment, structuring_element,
)


@pytest.fixture
def binary_image():
    image = np.zeros((100, 140), dtype=np.uint8)
    image[20:80, 30:110] = 255
    image[5, 5] = 255  # isolated foreground speck
    return image


def test_structuring_element_shapes():
    for shape in ("rect", "ellipse", "cross"):
        element = structuring_element(3, shape)
        assert element.shape == (3, 3)
    with pytest.raises(ValueError):
        structuring_element(4)
    with pytest.raises(ValueError):
        structuring_element(3, "triangle")


def test_erosion_reduces_and_dilation_expands_white_foreground(binary_image):
    original = np.count_nonzero(binary_image == 255)
    assert np.count_nonzero(erosion(binary_image, 3) == 255) < original
    assert np.count_nonzero(dilation(binary_image, 3) == 255) > original


def test_opening_removes_isolated_speck(binary_image):
    result = opening(binary_image, 3)
    assert result[5, 5] == 0


def test_closing_fills_small_gap():
    image = np.zeros((30, 30), dtype=np.uint8)
    image[10:20, 5:12] = 255
    image[10:20, 13:20] = 255
    result = closing(image, 3)
    assert result[15, 12] == 255


def test_experiment_generates_expected_outputs(binary_image):
    results = run_morphology_experiment(binary_image, sizes=(3, 5))
    assert "input_binary" in results
    assert "opening_3x3" in results
    assert "closing_5x5" in results
    assert all(img.shape == binary_image.shape for img in results.values())


def test_statistics_are_consistent(binary_image):
    stats = foreground_statistics(binary_image)
    assert stats["foreground_pixels"] == int(np.count_nonzero(binary_image == 255))
    assert 0 <= stats["foreground_ratio"] <= 1
