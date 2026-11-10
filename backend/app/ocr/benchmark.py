"""Run identical Tesseract OCR settings over alternative DIP variants."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.dip.preprocessing import apply_clahe, to_grayscale
from app.dip.phase2 import (
    global_threshold, otsu_threshold,
    adaptive_mean_threshold, adaptive_gaussian_threshold,
)
from app.ocr.tesseract_engine import ocr_image, save_ocr_result
from app.ocr.metrics import evaluate_ocr


def build_ocr_variants(image: np.ndarray) -> dict[str, np.ndarray]:
    """Create comparable inputs. All variants retain dark text on light paper."""
    gray = to_grayscale(image)
    clahe = apply_clahe(gray)
    _, otsu = otsu_threshold(clahe)
    adaptive_mean = adaptive_mean_threshold(clahe)
    adaptive_gaussian = adaptive_gaussian_threshold(clahe)
    # Phase 3 morphology operates on white text foreground; invert back for OCR.
    text_fg = cv2.bitwise_not(adaptive_gaussian)
    opened = cv2.morphologyEx(
        text_fg, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    morphology_for_ocr = cv2.bitwise_not(opened)
    return {
        "original": image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        "grayscale": gray,
        "clahe": clahe,
        "otsu": otsu,
        "adaptive_mean": adaptive_mean,
        "adaptive_gaussian": adaptive_gaussian,
        "morphology_opening": morphology_for_ocr,
    }


def run_ocr_benchmark(
    image: np.ndarray, output_dir: str | Path,
    ground_truth: str | None = None, lang: str = "eng",
    config: str = "--oem 3 --psm 6",
) -> list[dict[str, Any]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, variant in build_ocr_variants(image).items():
        result = ocr_image(variant, lang=lang, config=config)
        save_ocr_result(result, output, f"ocr_{method}")
        row: dict[str, Any] = {
            "method": method,
            "mean_confidence": result["mean_confidence"],
            "word_count": result["word_count"],
            "recognized_characters": len(result["text"]),
        }
        if ground_truth is not None:
            row.update(evaluate_ocr(ground_truth, result["text"]))
        rows.append(row)
    csv_path = output / "ocr_benchmark.csv"
    fields = list(rows[0].keys()) if rows else ["method"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "ocr_benchmark.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return rows
