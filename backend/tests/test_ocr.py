import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.ocr.metrics import (  # noqa: E402
    character_error_rate, word_error_rate, edit_distance, evaluate_ocr
)
from app.ocr.benchmark import build_ocr_variants  # noqa: E402


def test_edit_distance():
    assert edit_distance(list("kitten"), list("sitting")) == 3


def test_cer_and_wer_perfect_match():
    assert character_error_rate("Policy Number 123", "Policy Number 123") == 0
    assert word_error_rate("Policy Number 123", "Policy Number 123") == 0


def test_cer_detects_character_error():
    assert character_error_rate("claim", "clalm") > 0


def test_wer_detects_word_error():
    assert word_error_rate("claim form received", "claim form missing") > 0


def test_empty_reference_handling():
    assert character_error_rate("", "") == 0
    assert word_error_rate("", "extra words") == 1


def test_evaluation_accuracy_is_bounded():
    metrics = evaluate_ocr("invoice total 500", "invoice total 500")
    assert metrics["character_accuracy"] == 1
    assert metrics["word_accuracy"] == 1


def test_variant_builder_has_expected_methods():
    image = np.full((100, 120, 3), 255, dtype=np.uint8)
    image[30:40, 20:80] = 0
    variants = build_ocr_variants(image)
    assert set(variants) == {
        "original", "grayscale", "clahe", "otsu",
        "adaptive_mean", "adaptive_gaussian", "morphology_opening"
    }
    assert all(v.ndim == 2 for v in variants.values())
