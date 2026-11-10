"""Text OCR evaluation: Character Error Rate and Word Error Rate."""
from __future__ import annotations

import re
from typing import Sequence


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance using O(min(n,m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ref_item != hyp_item),
            ))
        previous = current
    return previous[-1]


def normalize_text(text: str, lowercase: bool = True) -> str:
    text = text.strip()
    if lowercase:
        text = text.lower()
    return re.sub(r"\\s+", " ", text)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER = character edit distance / reference character count."""
    ref = normalize_text(reference).replace(" ", "")
    hyp = normalize_text(hypothesis).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(list(ref), list(hyp)) / len(ref)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = word edit distance / reference word count."""
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def evaluate_ocr(reference: str, hypothesis: str) -> dict[str, float]:
    cer = character_error_rate(reference, hypothesis)
    wer = word_error_rate(reference, hypothesis)
    return {
        "cer": round(cer, 6),
        "character_accuracy": round(max(0.0, 1.0 - cer), 6),
        "wer": round(wer, 6),
        "word_accuracy": round(max(0.0, 1.0 - wer), 6),
    }
