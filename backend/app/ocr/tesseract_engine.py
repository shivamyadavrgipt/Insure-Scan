"""Tesseract OCR wrapper with word-level confidence and TSV export."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import pytesseract
    from pytesseract import Output
except ImportError:  # Allows the DIP modules/tests to work before OCR setup.
    pytesseract = None
    Output = None


def _require_tesseract() -> None:
    if pytesseract is None:
        raise RuntimeError(
            "pytesseract is not installed. Install it with "
            "`pip install pytesseract` and install the Tesseract system binary."
        )
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError(
            "Tesseract executable was not found. On Ubuntu run "
            "`sudo apt install tesseract-ocr`."
        ) from exc


def ocr_image(
    image: np.ndarray,
    lang: str = "eng",
    config: str = "--oem 3 --psm 6",
) -> dict[str, Any]:
    """Recognize text and return text, mean word confidence and word boxes."""
    _require_tesseract()
    if image is None or image.size == 0:
        raise ValueError("image must be non-empty")
    if image.ndim == 2:
        prepared = image
    elif image.ndim == 3:
        prepared = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("image must be grayscale or a colour image")

    data = pytesseract.image_to_data(
        prepared, lang=lang, config=config, output_type=Output.DICT
    )
    words = []
    confidences = []
    for i, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            confidence = -1.0
        if text and confidence >= 0:
            words.append({
                "text": text,
                "confidence": confidence,
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
            })
            confidences.append(confidence)

    recognized = " ".join(item["text"] for item in words)
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    return {
        "text": recognized,
        "mean_confidence": round(mean_confidence, 2),
        "words": words,
        "word_count": len(words),
        "language": lang,
        "config": config,
    }


def save_ocr_result(result: dict[str, Any], output_dir: str | Path, prefix: str) -> dict[str, Path]:
    """Save recognized text and word-level confidence/coordinates."""
    import csv
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    txt_path = output / f"{prefix}.txt"
    txt_path.write_text(result["text"] + "\\n", encoding="utf-8")
    csv_path = output / f"{prefix}_words.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["text", "confidence", "x", "y", "width", "height"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["words"])
    return {"text": txt_path, "words_csv": csv_path}
