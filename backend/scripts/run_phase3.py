"""Run Phase 3 morphology on a binary image from Phase 2."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.dip.morphology import save_morphology_results, foreground_statistics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: morphology experiment")
    parser.add_argument("--input", required=True, help="Binary image from Phase 2")
    parser.add_argument("--output", default="results/images")
    parser.add_argument("--prefix", default="phase3")
    parser.add_argument("--shape", choices=["rect", "ellipse", "cross"], default="rect")
    parser.add_argument(
        "--foreground", choices=["dark", "light"], default="dark",
        help="Which pixels represent the document content in the input (default: dark text on light paper)"
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[3, 5, 7])
    args = parser.parse_args()

    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read input image: {args.input}")
    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    # OpenCV morphology operates on white foreground. Most scanned documents
    # contain dark ink on a light page, so invert by default to process the ink.
    morphology_input = cv2.bitwise_not(image) if args.foreground == "dark" else image
    saved = save_morphology_results(
        morphology_input, out_dir, args.prefix, tuple(args.sizes), args.shape
    )
    print("Phase 3 completed.")
    print(f"Input: {args.input} ({image.shape[1]} x {image.shape[0]})")
    print(f"Structuring element: {args.shape}; sizes: {args.sizes}")
    print("Input statistics (white foreground after polarity handling):",
          foreground_statistics(morphology_input))
    for name, path in saved.items():
        print(f"{name:24s} -> {path}")
    print("Saved morphology images use white ink/content on a black background.")


if __name__ == "__main__":
    main()
