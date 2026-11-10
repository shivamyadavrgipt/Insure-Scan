"""Evidence-backed extraction of common motor-insurance intake fields."""
from __future__ import annotations
import re
from typing import Any

PATTERNS = {
    "claim_reference": [r"(?:claim\s*(?:reference|ref(?:erence)?|number|no\.?))\s*[:#-]?\s*([A-Z0-9][A-Z0-9/_\-—]{3,})"],
    "policy_number": [r"(?:policy\s*(?:number|no\.?))\s*[:#-]?\s*([A-Z0-9][A-Z0-9/_\-—]{3,})"],
    "policy_holder": [r"(?:policy\s*holder|insured\s*name|customer\s*name)\s*[:#-]?\s*([A-Z][A-Z .'-]{2,}?)(?=\s+(?:Vehicle\s+Registration|Incident\s+Date|Claim\s+Amount|Repair\s+Workshop|Invoice\s+Number)\b|$)"],
    "vehicle_registration": [r"(?:vehicle\s*registration|registration\s*(?:number|no\.?))\s*[:#-]?\s*([A-Z]{2}\s*[- ]?\d{1,2}\s*[- ]?[A-Z]{1,3}\s*[- ]?\d{1,4})"],
    "incident_date": [r"(?:incident\s*date|date\s*of\s*incident|accident\s*date)\s*[:#-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"],
    "claim_amount": [r"(?:claim\s*amount|claimed\s*amount|amount\s*claimed)\s*[:#-]?\s*(?:(?:INR|Rs\.?|₹)\s*)?([\d,]+(?:\.\d{1,2})?)"],
    "repair_workshop": [r"(?:repair\s*workshop|workshop\s*name|garage\s*name)\s*[:#-]?\s*([A-Z][A-Z0-9 &.'-]{2,}?)(?=\s+(?:Invoice\s+Number|Claim\s+Amount|Incident\s+Date)\b|$)"],
    "invoice_number": [r"(?:invoice\s*(?:number|no\.?))\s*[:#-]?\s*([A-Z0-9][A-Z0-9/_\-—]{2,})"],
}

def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).strip(" \t\r\n:;,.|")

def extract_fields(text: str) -> dict[str, Any]:
    """Return value, matched evidence, and heuristic confidence for each field."""
    source = text or ""
    fields: dict[str, Any] = {}
    for name, patterns in PATTERNS.items():
        found = None
        for pattern in patterns:
            match = re.search(pattern, source, flags=re.IGNORECASE)
            if match:
                value = _clean(match.group(1))
                if value:
                    found = {"value": value, "evidence": match.group(0).strip(),
                             "confidence": 0.90, "status": "extracted"}
                    break
        fields[name] = found or {"value": None, "evidence": None,
                                 "confidence": 0.0, "status": "missing"}
    return fields
