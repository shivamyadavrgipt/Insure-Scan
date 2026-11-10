"""Classification and field-extraction evaluation metrics."""
from __future__ import annotations
import re
from typing import Any

def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())

def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        raise ValueError("At least one labelled example is required")
    labels = sorted(set(y_true) | set(y_pred))
    per_class = {}
    f1s = []
    correct = 0
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                            "f1": round(f1, 4), "support": sum(x == label for x in y_true)}
        f1s.append(f1)
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    return {"accuracy": round(correct / len(y_true), 4),
            "macro_f1": round(sum(f1s) / len(f1s), 4),
            "total": len(y_true), "per_class": per_class}

def extraction_metrics(references: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have equal length")
    if not references:
        raise ValueError("At least one annotated example is required")
    fields = sorted(set().union(*(x.keys() for x in references), *(x.keys() for x in predictions)))
    per_field = {}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for field in fields:
        tp = fp = fn = 0
        for ref, pred in zip(references, predictions):
            expected, actual = ref.get(field), pred.get(field)
            expected_present, actual_present = expected not in (None, ""), actual not in (None, "")
            if expected_present and actual_present and _norm(expected) == _norm(actual):
                tp += 1
            elif expected_present and actual_present:
                fp += 1; fn += 1
            elif actual_present:
                fp += 1
            elif expected_present:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_field[field] = {"precision": round(precision, 4), "recall": round(recall, 4),
                            "f1": round(f1, 4), "exact_matches": tp,
                            "false_positives": fp, "false_negatives": fn}
        totals["tp"] += tp; totals["fp"] += fp; totals["fn"] += fn
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"micro_precision": round(precision, 4), "micro_recall": round(recall, 4),
            "micro_f1": round(f1, 4), "total_documents": len(references),
            "per_field": per_field, "counts": totals}
