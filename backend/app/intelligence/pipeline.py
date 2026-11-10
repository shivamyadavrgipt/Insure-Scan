"""End-to-end OCR -> classification -> structured extraction pipeline."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import json
from app.ocr.tesseract_engine import ocr_image
from app.intelligence.classification import classify_document
from app.intelligence.extraction import extract_fields
from app.evaluation.validation import validate_fields

def analyze_text(text: str) -> dict[str, Any]:
    classification = classify_document(text)
    fields = extract_fields(text)
    return {
        "document_type": classification["label"],
        "classification": classification,
        "fields": fields,
        "validation": validate_fields(fields),
        "ocr_text": text,
        "notes": [
            "Rule-based baseline; classification confidence is heuristic, not calibrated.",
            "Extracted values are OCR-derived and require human verification before claim processing.",
            "Missing fields are returned as null; no government identity or policy database is queried.",
        ],
    }

def analyze_image(image_path: str | Path, lang: str = "eng") -> dict[str, Any]:
    import cv2
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    ocr = ocr_image(image, lang=lang)
    result = analyze_text(ocr["text"])
    result["ocr"] = {
        "mean_confidence": ocr["mean_confidence"],
        "word_count": ocr["word_count"],
        "language": ocr["language"],
        "config": ocr["config"],
    }
    return result

def save_analysis(result: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
