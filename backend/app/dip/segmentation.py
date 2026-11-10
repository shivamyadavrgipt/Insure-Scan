"""Phase 4: document region, line, word and character segmentation.

The segmentation functions expect dark text on a light background or a
binary image. Internally, text is normalized to white foreground on black.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _text_foreground(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty array")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError("image must be a 2-D or 3-channel image")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    # Binarize then invert: text is white foreground, paper is black.
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(binary)


def _remove_page_rules(foreground: np.ndarray) -> np.ndarray:
    """Suppress long table/page rules that otherwise join separate text lines."""
    h, w = foreground.shape
    horizontal_len = max(30, w // 25)
    vertical_len = max(30, h // 25)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_len, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_len))
    horizontal_rules = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_rules = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, vertical_kernel)
    rules = cv2.bitwise_or(horizontal_rules, vertical_rules)
    return cv2.subtract(foreground, rules)


def connected_components(
    image: np.ndarray, min_area: int = 3, connectivity: int = 8
) -> list[dict[str, int]]:
    """Return component boxes and areas, filtering tiny specks by min_area."""
    if min_area < 1:
        raise ValueError("min_area must be >= 1")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    fg = _text_foreground(image)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        fg, connectivity=connectivity
    )
    items: list[dict[str, int]] = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if int(area) >= min_area:
            cx, cy = centroids[label]
            items.append({
                "x": int(x), "y": int(y), "width": int(w), "height": int(h),
                "area": int(area), "centroid_x": int(round(cx)),
                "centroid_y": int(round(cy)),
            })
    return sorted(items, key=lambda item: (item["y"], item["x"]))


def find_text_regions(
    image: np.ndarray, min_area: int = 20, min_width: int = 8,
    min_height: int = 5, horizontal_kernel: int = 25, vertical_kernel: int = 1
) -> list[dict[str, int]]:
    """Group nearby character components into approximate text regions/lines."""
    fg = _remove_page_rules(_text_foreground(image))
    if horizontal_kernel < 1 or vertical_kernel < 1:
        raise ValueError("morphology kernel dimensions must be positive")
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_kernel, vertical_kernel)
    )
    joined = cv2.dilate(fg, kernel, iterations=1)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = int(cv2.countNonZero(fg[y:y+h, x:x+w]))
        if area >= min_area and w >= min_width and h >= min_height:
            regions.append({"x": x, "y": y, "width": w, "height": h, "area": area})
    return sorted(regions, key=lambda item: (item["y"], item["x"]))


def segment_lines(image: np.ndarray, min_line_height: int = 3) -> list[dict[str, Any]]:
    """Segment text lines using horizontal projection profiles."""
    if min_line_height < 1:
        raise ValueError("min_line_height must be >= 1")
    fg = _remove_page_rules(_text_foreground(image))
    projection = np.count_nonzero(fg, axis=1)
    # Ignore thin vertical page borders: a genuine text row contains more
    # foreground pixels than the handful contributed by border remnants.
    active = projection > max(2, int(fg.shape[1] * 0.003))
    lines: list[dict[str, Any]] = []
    start = None
    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            end = idx
            if end - start >= min_line_height:
                lines.append({
                    "x": 0, "y": int(start), "width": int(fg.shape[1]),
                    "height": int(end-start),
                    "foreground_pixels": int(projection[start:end].sum()),
                })
            start = None
    if start is not None and len(active) - start >= min_line_height:
        end = len(active)
        lines.append({
            "x": 0, "y": int(start), "width": int(fg.shape[1]),
            "height": int(end-start),
            "foreground_pixels": int(projection[start:end].sum()),
        })
    return lines


def segment_words(
    image: np.ndarray, line_boxes: list[dict[str, Any]] | None = None,
    min_gap: int = 5, min_word_width: int = 2
) -> list[dict[str, int]]:
    """Approximate word boxes by vertical projection inside each detected line."""
    if min_gap < 1 or min_word_width < 1:
        raise ValueError("min_gap and min_word_width must be positive")
    fg = _remove_page_rules(_text_foreground(image))
    if line_boxes is None:
        line_boxes = segment_lines(image)
    words: list[dict[str, int]] = []
    for line in line_boxes:
        y, h = int(line["y"]), int(line["height"])
        x0, x1 = max(0, int(line.get("x", 0))), min(fg.shape[1], int(line.get("x", 0)) + int(line.get("width", fg.shape[1])))
        strip = fg[y:y+h, x0:x1]
        if strip.size == 0:
            continue
        projection = np.count_nonzero(strip, axis=0)
        active = projection > 0
        start = None
        for idx, is_active in enumerate(active):
            if is_active and start is None:
                start = idx
            elif not is_active and start is not None:
                end = idx
                if end-start >= min_word_width:
                    words.append({"x": x0+start, "y": y, "width": end-start, "height": h})
                start = None
        if start is not None and len(active)-start >= min_word_width:
            words.append({"x": x0+start, "y": y, "width": len(active)-start, "height": h})
    # Merge neighbouring character runs separated by gaps smaller than min_gap.
    # The initial projection runs are individual glyphs; merge into word groups.
    merged: list[dict[str, int]] = []
    for box in words:
        if merged and box["y"] == merged[-1]["y"] and box["x"] - (merged[-1]["x"]+merged[-1]["width"]) < min_gap:
            right = max(merged[-1]["x"]+merged[-1]["width"], box["x"]+box["width"])
            merged[-1]["width"] = right-merged[-1]["x"]
        else:
            merged.append(box.copy())
    return merged


def segment_characters(
    image: np.ndarray, word_boxes: list[dict[str, int]] | None = None,
    min_area: int = 2
) -> list[dict[str, int]]:
    """Return connected-component character candidates, sorted top-to-bottom."""
    if min_area < 1:
        raise ValueError("min_area must be >= 1")
    fg = _remove_page_rules(_text_foreground(image))
    components = connected_components(cv2.bitwise_not(fg), min_area=min_area, connectivity=8)
    if word_boxes is not None:
        kept = []
        for comp in components:
            cx, cy = comp["centroid_x"], comp["centroid_y"]
            if any(
                box["x"] <= cx < box["x"]+box["width"] and
                box["y"] <= cy < box["y"]+box["height"]
                for box in word_boxes
            ):
                kept.append(comp)
        components = kept
    return components


def draw_boxes(
    image: np.ndarray, boxes: list[dict[str, int]], color: tuple[int, int, int] = (0, 180, 0),
    thickness: int = 2
) -> np.ndarray:
    """Draw segmentation boxes on a copy of the original image."""
    if image.ndim == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output = image.copy()
    for box in boxes:
        x, y = box["x"], box["y"]
        w, h = box["width"], box["height"]
        cv2.rectangle(output, (x, y), (x+w, y+h), color, thickness)
    return output


def save_segmentation_outputs(image: np.ndarray, output_dir: str | Path, prefix: str = "phase4") -> dict[str, Path]:
    """Run segmentation stages and save overlays plus cropped region examples."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    regions = find_text_regions(image)
    lines = segment_lines(image)
    words = segment_words(image, lines)
    characters = segment_characters(image, words)
    saved: dict[str, Path] = {}
    stages = {
        "regions": (regions, draw_boxes(image, regions)),
        "lines": (lines, draw_boxes(image, lines, (255, 0, 0))),
        "words": (words, draw_boxes(image, words, (0, 0, 255))),
        "characters": (characters, draw_boxes(image, characters, (0, 180, 0), 1)),
    }
    for name, (boxes, overlay) in stages.items():
        path = output / f"{prefix}_{name}.png"
        if not cv2.imwrite(str(path), overlay):
            raise OSError(f"Could not save {path}")
        saved[name] = path
        # Save coordinate metadata as CSV for reproducible inspection.
        csv_path = output / f"{prefix}_{name}.csv"
        if boxes:
            import csv
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(boxes[0].keys()))
                writer.writeheader()
                writer.writerows(boxes)
        else:
            csv_path.write_text("x,y,width,height\n", encoding="utf-8")
        saved[f"{name}_csv"] = csv_path
    return saved
