#!/usr/bin/env python3
"""Classify a new document image using the trained Phase 8 model."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, joblib
from app.ocr.tesseract_engine import ocr_image

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image",required=True)
    ap.add_argument("--model",default="results/ml/document_classifier.joblib")
    args=ap.parse_args()
    image=cv2.imread(args.image)
    if image is None: raise SystemExit(f"Cannot read image: {args.image}")
    ocr=ocr_image(image)
    model=joblib.load(args.model)
    label=model.predict([ocr["text"]])[0]
    print(json.dumps({"document_type":label,"ocr_mean_confidence":ocr["mean_confidence"],
                      "ocr_text":ocr["text"],
                      "warning":"Prediction is a synthetic-data-trained demo. Verify manually."},indent=2))
if __name__=="__main__": main()
