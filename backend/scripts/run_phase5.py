"""Run OCR and compare results across DIP preprocessing variants."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.ocr.benchmark import run_ocr_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: OCR benchmark")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/ocr")
    parser.add_argument("--ground-truth", default=None,
                        help="UTF-8 text file containing manually verified exact document text")
    parser.add_argument("--lang", default="eng", help="Tesseract language code")
    parser.add_argument("--config", default="--oem 3 --psm 6")
    args = parser.parse_args()

    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")
    ground_truth = None
    if args.ground_truth:
        ground_truth = Path(args.ground_truth).read_text(encoding="utf-8")
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    rows = run_ocr_benchmark(
        image, output, ground_truth=ground_truth,
        lang=args.lang, config=args.config
    )
    print("Phase 5 OCR benchmark completed.")
    print(f"Image dimensions: {image.shape[1]} x {image.shape[0]}")
    print(f"Methods compared: {len(rows)}")
    for row in rows:
        print(
            f"{row['method']:22s} confidence={row['mean_confidence']:6.2f} "
            f"words={row['word_count']:4d}"
        )
        if "cer" in row:
            print(
                f"  CER={row['cer']:.4f} WER={row['wer']:.4f} "
                f"character_accuracy={row['character_accuracy']:.4f}"
            )
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
