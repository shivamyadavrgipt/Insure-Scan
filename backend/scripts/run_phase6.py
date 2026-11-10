#!/usr/bin/env python3
"""Run OCR, document classification and field extraction on an image."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.intelligence.pipeline import analyze_image, save_analysis

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to document image")
    parser.add_argument("--output", default="results/intelligence/analysis.json")
    parser.add_argument("--lang", default="eng")
    args = parser.parse_args()
    result = analyze_image(args.input, lang=args.lang)
    output = save_analysis(result, args.output)
    print(f"Document type: {result['document_type']}")
    print(f"Classification evidence: {', '.join(result['classification']['evidence']) or 'none'}")
    print("\\nExtracted fields:")
    for name, item in result["fields"].items():
        print(f"  {name}: {item['value'] if item['value'] is not None else 'NOT FOUND'}")
    print(f"\\nSaved JSON: {output}")

if __name__ == "__main__":
    main()
