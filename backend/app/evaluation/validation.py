"""Syntactic sanity checks and human-review routing (not official verification)."""
from __future__ import annotations
import re
from datetime import datetime
from typing import Any

def validate_fields(fields: dict[str, Any], low_confidence_threshold: float = 0.80) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    review_reasons: list[str] = []
    for name, item in fields.items():
        value = item.get("value") if isinstance(item, dict) else None
        confidence = float(item.get("confidence", 0.0)) if isinstance(item, dict) else 0.0
        errors: list[str] = []
        if value is None or str(value).strip() == "":
            errors.append("missing_value")
        else:
            v = str(value).strip()
            if name == "vehicle_registration":
                compact = re.sub(r"[\s-]", "", v.upper())
                if not re.fullmatch(r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}", compact):
                    errors.append("registration_format_unrecognized")
            elif name == "incident_date":
                parsed = None
                for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y"):
                    try:
                        parsed = datetime.strptime(v, fmt)
                        break
                    except ValueError:
                        pass
                if parsed is None:
                    errors.append("date_format_unrecognized")
            elif name == "claim_amount":
                try:
                    amount = float(v.replace(",", ""))
                    if amount <= 0:
                        errors.append("amount_must_be_positive")
                except ValueError:
                    errors.append("amount_not_numeric")
        if value is not None and confidence < low_confidence_threshold:
            errors.append("low_extraction_confidence")
        status = "pass" if not errors else "review"
        checks[name] = {"status": status, "errors": errors}
        if errors:
            review_reasons.extend(f"{name}:{error}" for error in errors)
    return {
        "checks": checks,
        "requires_human_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "disclaimer": "Format checks do not verify identity, ownership, policy validity or claim legitimacy.",
    }
