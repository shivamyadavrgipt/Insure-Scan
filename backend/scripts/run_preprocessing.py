"""Run the first DIP preprocessing experiment from the project root.

Usage:
  python backend/scripts/run_preprocessing.py --input dataset/original/sample.png
  python backend/scripts/run_preprocessing.py --input /path/to/scan.jpg --output results/images
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.dip.preprocessing import read_image, save_preprocessing_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Insurance document DIP preprocessing")
    parser.add_argument("--input", required=True, help="Path to an input document image")
    parser.add_argument("--output", default="results/images", help="Output directory")
    parser.add_argument("--prefix", default="document", help="Output filename prefix")
    args = parser.parse_args()

    image = read_image(args.input)
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    saved = save_preprocessing_results(image, output_dir, args.prefix)

    print("\nPreprocessing completed successfully.")
    for stage, path in saved.items():
        print(f"{stage:22s} -> {path}")
    print(f"\nInput dimensions: {image.shape[1]} x {image.shape[0]}")


if __name__ == "__main__":
    main()
