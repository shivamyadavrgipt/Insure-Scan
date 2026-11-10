"""Run Phase 2 denoising and thresholding comparison from project root."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.dip.preprocessing import read_image, to_grayscale  # noqa: E402
from app.dip.phase2 import save_phase2_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: denoising and thresholding")
    parser.add_argument("--input", required=True, help="Input document image")
    parser.add_argument("--output", default="results/images", help="Output directory")
    parser.add_argument("--prefix", default="phase2", help="Output filename prefix")
    args = parser.parse_args()
    image = to_grayscale(read_image(args.input))
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    saved = save_phase2_results(image, output_dir, args.prefix)
    print("\nPhase 2 completed successfully.")
    for stage, path in saved.items():
        print(f"{stage:30s} -> {path}")
    print(f"\nInput dimensions: {image.shape[1]} x {image.shape[0]}")


if __name__ == "__main__":
    main()
