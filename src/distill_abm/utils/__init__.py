"""Shared utility functions for validation and debugging."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


# Placeholder tokens to detect in text inputs
PLACEHOLDER_TOKENS = ("placeholder", "todo", "tbd", "dummy", "lorem ipsum")


def detect_placeholder_signals(text: str) -> list[str]:
    """Detect placeholder-like tokens in text.

    Returns a list of tokens that were found in the text (case-insensitive).
    """
    lowered = text.lower()
    return [token for token in PLACEHOLDER_TOKENS if token in lowered]


def validate_file_for_placeholder_signals(path: Path | None) -> list[str]:
    """Check a file for placeholder-like content.

    Returns a list of detected placeholder signals, or empty list if file
    doesn't exist or has no placeholder signals.
    """
    if path is None or not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return detect_placeholder_signals(text)
