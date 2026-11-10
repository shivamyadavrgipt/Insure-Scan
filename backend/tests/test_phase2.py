import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.dip.phase2 import (  # noqa: E402
    adaptive_gaussian_threshold, adaptive_mean_threshold, bilateral_denoise,
    gaussian_denoise, global_threshold, median_denoise, otsu_threshold, run_phase2,
)


def sample_gray():
    image = np.full((160, 240), 235, dtype=np.uint8)
    image[35:125, 35:205] = 35
    image[55:65, 55:185] = 245
    return image


def test_denoising_preserves_shape_and_dtype():
    image = sample_gray()
    for output in (gaussian_denoise(image), median_denoise(image), bilateral_denoise(image)):
        assert output.shape == image.shape
        assert output.dtype == np.uint8


def test_threshold_methods_return_binary_images():
    image = sample_gray()
    outputs = [global_threshold(image), otsu_threshold(image)[1],
               adaptive_mean_threshold(image), adaptive_gaussian_threshold(image)]
    for output in outputs:
        assert output.shape == image.shape
        assert output.dtype == np.uint8
        assert set(np.unique(output)).issubset({0, 255})


def test_otsu_returns_valid_threshold():
    threshold, output = otsu_threshold(sample_gray())
    assert 0 <= threshold <= 255
    assert output.shape == sample_gray().shape


def test_phase2_pipeline_has_expected_stages():
    outputs = run_phase2(sample_gray())
    assert set(outputs) == {
        "gaussian_denoised", "median_denoised", "bilateral_denoised",
        "threshold_global", "threshold_otsu", "threshold_adaptive_mean",
        "threshold_adaptive_gaussian",
    }


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        gaussian_denoise(sample_gray(), kernel_size=4)
    with pytest.raises(ValueError):
        adaptive_mean_threshold(sample_gray(), block_size=10)
