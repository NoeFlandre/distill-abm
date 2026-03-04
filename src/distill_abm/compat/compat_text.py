"""Text cleaning utilities kept for compatibility parity."""

from __future__ import annotations

import re
from typing import cast

from distill_abm.compat.compat_callables import _call_notebook_first
from distill_abm.summarize.reference_text import clean_context_response as _clean_context_response_refactored
from distill_abm.summarize.reference_text import clean_symbols as _clean_symbols_refactored


def clean_context_response(text: str) -> str:
    return cast(str, _call_notebook_first("clean_context_response", _clean_context_response_refactored, text))


def clean_symbols(text: str | float) -> str | float:
    return cast(str | float, _call_notebook_first("clean_symbols", _clean_symbols_refactored, text))


def capitalize(match: re.Match[str]) -> str:
    return match.group(1) + match.group(2).upper()


def should_skip_row(row: dict[str, object], column_name: str) -> bool:
    value = row.get(column_name)
    if value is None:
        return False
    if isinstance(value, int | float):
        return value > 0
    return isinstance(value, str) and bool(value.strip())
