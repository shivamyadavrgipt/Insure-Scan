from app.intelligence.classification import classify_document
from app.intelligence.extraction import extract_fields
from app.intelligence.pipeline import analyze_text

SAMPLE = (
    "SAMPLE MOTOR INSURANCE CLAIM FORM Claim Reference: DEMO-2026-0042 "
    "Policy Number: POL-SAMPLE-908172 Policy Holder: SAMPLE CUSTOMER "
    "Vehicle Registration: UP 00 AB 1234 Incident Date: 08-11-2026 "
    "Claim Amount: INR 48,500 Repair Workshop: SAMPLE AUTO REPAIRS "
    "Invoice Number: INV-DEMO-7781"
)

def test_claim_form_classification():
    result = classify_document(SAMPLE)
    assert result["label"] == "motor_claim_form"
    assert "claim form" in result["evidence"]

def test_unknown_for_weak_text():
    assert classify_document("hello world")["label"] == "unknown"

def test_extract_core_fields():
    fields = extract_fields(SAMPLE)
    assert fields["policy_number"]["value"] == "POL-SAMPLE-908172"
    assert fields["vehicle_registration"]["value"] == "UP 00 AB 1234"
    assert fields["incident_date"]["value"] == "08-11-2026"
    assert fields["claim_amount"]["value"] == "48,500"
    assert fields["invoice_number"]["value"] == "INV-DEMO-7781"

def test_missing_fields_are_null():
    fields = extract_fields("This document has no claim identifiers.")
    assert fields["policy_number"]["value"] is None
    assert fields["policy_number"]["status"] == "missing"

def test_evidence_is_returned():
    field = extract_fields(SAMPLE)["policy_number"]
    assert "Policy Number" in field["evidence"]
    assert field["confidence"] > 0

def test_pipeline_keeps_ocr_text():
    result = analyze_text(SAMPLE)
    assert result["document_type"] == "motor_claim_form"
    assert result["ocr_text"] == SAMPLE
    assert "human verification" in " ".join(result["notes"])
