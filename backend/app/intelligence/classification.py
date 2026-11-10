"""Transparent rule-based insurance document classification baseline."""
from __future__ import annotations
import re
from typing import Any

KEYWORDS = {
    "motor_claim_form": ["claim form", "claim reference", "claim number", "incident date", "claim amount"],
    "policy_document": ["insurance policy", "policy schedule", "policy period", "premium", "sum insured"],
    "repair_invoice": ["repair invoice", "invoice number", "workshop", "labour charges", "parts"],
    "vehicle_registration": ["registration certificate", "vehicle registration", "chassis number", "engine number", "owner name"],
    "medical_bill": ["hospital", "patient", "consultation", "pharmacy", "medical bill"],
}

def classify_document(text: str) -> dict[str, Any]:
    """Classify from explicit keyword evidence; return unknown for weak evidence."""
    normalized = re.sub(r"\s+", " ", (text or "").lower())
    scores = {
        label: sum(1 for phrase in phrases if phrase in normalized)
        for label, phrases in KEYWORDS.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1]
    total_hits = sum(scores.values())
    if best_score < 2 or best_score == second_score:
        return {"label": "unknown", "confidence": 0.0, "scores": scores,
                "evidence": [], "method": "keyword_rule_baseline"}
    evidence = [phrase for phrase in KEYWORDS[best_label] if phrase in normalized]
    # A transparent heuristic score, not a calibrated probability.
    confidence = min(0.99, round(0.55 + 0.10 * best_score + 0.04 * (best_score-second_score), 2))
    return {"label": best_label, "confidence": confidence, "scores": scores,
            "evidence": evidence, "method": "keyword_rule_baseline"}
