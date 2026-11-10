#!/usr/bin/env python3
"""Evaluate classification and extraction from a JSONL annotation file."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation.metrics import classification_metrics, extraction_metrics

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="JSONL file: one object per document")
    parser.add_argument("--output", default="results/evaluation/phase7_metrics.json")
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.annotations).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit("No annotation records found.")
    truth_types = [r["ground_truth"]["document_type"] for r in records]
    pred_types = [r["prediction"]["document_type"] for r in records]
    truth_fields = [r["ground_truth"].get("fields", {}) for r in records]
    pred_fields = [r["prediction"].get("fields", {}) for r in records]
    metrics = {"classification": classification_metrics(truth_types, pred_types),
               "extraction": extraction_metrics(truth_fields, pred_fields),
               "annotation_count": len(records),
               "note": "Metrics are meaningful only when ground truth is manually verified and representative."}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\\nSaved: {output}")

if __name__ == "__main__":
    main()
