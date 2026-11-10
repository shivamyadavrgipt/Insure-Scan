import pytest
from app.evaluation.metrics import classification_metrics, extraction_metrics
from app.evaluation.validation import validate_fields
from app.intelligence.pipeline import analyze_text

def test_classification_metrics():
    result = classification_metrics(["claim", "policy", "claim"], ["claim", "claim", "claim"])
    assert result["accuracy"] == pytest.approx(2/3, abs=0.0001)
    assert "macro_f1" in result

def test_classification_length_mismatch():
    with pytest.raises(ValueError):
        classification_metrics(["a"], [])

def test_extraction_metrics_exact_and_wrong():
    result = extraction_metrics(
        [{"policy_number": "POL-123", "claim_amount": "500"}],
        [{"policy_number": "POL-123", "claim_amount": "600"}],
    )
    assert result["per_field"]["policy_number"]["f1"] == 1.0
    assert result["per_field"]["claim_amount"]["false_negatives"] == 1

def test_validation_routes_missing_fields():
    result = validate_fields({"policy_number": {"value": None, "confidence": 0}})
    assert result["requires_human_review"] is True
    assert "missing_value" in result["checks"]["policy_number"]["errors"]

def test_validation_accepts_syntactically_valid_values():
    result = validate_fields({
        "vehicle_registration": {"value": "UP 00 AB 1234", "confidence": .95},
        "incident_date": {"value": "08-11-2026", "confidence": .95},
        "claim_amount": {"value": "48500", "confidence": .95},
    })
    assert result["requires_human_review"] is False

def test_pipeline_contains_validation():
    result = analyze_text("Claim Form Claim Reference: CLM-2026-1234 Claim Amount: INR 500")
    assert "validation" in result
    assert result["validation"]["requires_human_review"] is True
