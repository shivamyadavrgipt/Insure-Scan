"""Run Phase 4 document segmentation on a Phase 2 binary output."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.dip.segmentation import save_segmentation_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: document segmentation")
    parser.add_argument("--input", required=True, help="Binary or grayscale document image")
    parser.add_argument("--output", default="results/images")
    parser.add_argument("--prefix", default="phase4")
    args = parser.parse_args()
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read input image: {args.input}")
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    saved = save_segmentation_outputs(image, output, args.prefix)
    print("Phase 4 completed.")
    print(f"Input dimensions: {image.shape[1]} x {image.shape[0]}")
    for name, path in saved.items():
        print(f"{name:20s} -> {path}")
    print("\\nSegmentation counts:")
    for name in ("regions", "lines", "words", "characters"):
        csv_path = saved[f"{name}_csv"]
        with csv_path.open(encoding="utf-8") as handle:
            count = max(sum(1 for _ in handle)-1, 0)
        print(f"{name:12s}: {count}")


if __name__ == "__main__":
    main()
