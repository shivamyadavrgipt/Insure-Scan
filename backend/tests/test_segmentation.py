import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.dip.segmentation import (  # noqa: E402
    connected_components, draw_boxes, find_text_regions, segment_characters,
    segment_lines, segment_words,
)


@pytest.fixture
def two_line_document():
    # White page, black glyph-like blocks with a clear inter-line gap.
    image = np.full((100, 220), 255, dtype=np.uint8)
    for x in (20, 35, 50, 95, 110):
        image[20:32, x:x+8] = 0
    for x in (25, 40, 55, 100):
        image[60:72, x:x+8] = 0
    return image


def test_connected_components_detects_glyph_candidates(two_line_document):
    boxes = connected_components(two_line_document, min_area=5)
    assert len(boxes) == 9
    assert all(b["width"] > 0 and b["height"] > 0 for b in boxes)


def test_line_segmentation_detects_two_lines(two_line_document):
    lines = segment_lines(two_line_document)
    assert len(lines) == 2
    assert lines[0]["y"] < lines[1]["y"]


def test_word_segmentation_returns_boxes(two_line_document):
    lines = segment_lines(two_line_document)
    words = segment_words(two_line_document, lines, min_gap=8)
    assert len(words) >= 2
    assert all(w["width"] > 0 and w["height"] > 0 for w in words)


def test_region_detection_groups_nearby_glyphs(two_line_document):
    regions = find_text_regions(two_line_document, horizontal_kernel=12)
    assert len(regions) >= 1


def test_character_segmentation_with_word_filter(two_line_document):
    words = segment_words(two_line_document, segment_lines(two_line_document), min_gap=8)
    chars = segment_characters(two_line_document, words, min_area=5)
    assert len(chars) == 9


def test_draw_boxes_does_not_mutate_input(two_line_document):
    before = two_line_document.copy()
    result = draw_boxes(two_line_document, [{"x": 10, "y": 10, "width": 20, "height": 20}])
    assert result.shape == (100, 220, 3)
    assert np.array_equal(two_line_document, before)


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        connected_components(np.array([], dtype=np.uint8))
