"""Qualitative score extraction helpers from GPT/Claude rating notebooks."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping


def should_skip_row(row: Mapping[str, object], column_name: str) -> bool:
    """Skips already-rated rows to avoid duplicate API calls during bulk evaluation."""
    if column_name not in row:
        return False
    value = row[column_name]
    if _is_missing(value):
        return False
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return bool(value.strip())
    return False


def extract_faithfulness_score(response_text: str) -> str:
    """Parses a 1-5 faithfulness score from noisy natural-language outputs."""
    patterns = _score_patterns("faithfulness")
    return _extract_score_from_patterns(response_text, patterns)


def extract_coverage_score(response_text: str) -> str:
    """Parses a 1-5 coverage score from noisy natural-language outputs."""
    patterns = _score_patterns("coverage")
    return _extract_score_from_patterns(response_text, patterns)


def _score_patterns(metric: str) -> list[str]:
    return [
        rf"{metric}(?:\s+score)?(?:\s+of)?(?:\s+is)?(?:\s*:)?\s*(\d)(?:\.0*)?(?:/5)?",
        rf"(?:score|rating|give|assign)(?:\s+a)?(?:\s+{metric})?(?:\s+score)?(?:\s+of)?\s*:?\s*(\d)(?:\.0*)?(?:/5)?",
        rf"(\d)(?:\.0*)?(?:/5)?(?:\s+out\s+of\s+5)?(?:\s+for\s+{metric})",
    ]


def _extract_score_from_patterns(response_text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return str(match.group(1))
    out_of_five = re.search(r"(\d)(?:\s+out\s+of\s+5)", response_text)
    if out_of_five:
        return str(out_of_five.group(1))
    digits = re.findall(r"\b(\d)\b", response_text)
    for digit in digits:
        value = str(digit)
        if 1 <= int(value) <= 5:
            return value
    return ""


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False
