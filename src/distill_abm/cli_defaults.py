"""Centralized CLI defaults and typed helpers for command option expansion."""

from __future__ import annotations

from typing import cast

from distill_abm.configs.models import SummarizerId
from distill_abm.configs.runtime_defaults import get_runtime_defaults
from distill_abm.pipeline.run import EvidenceMode, TextSourceMode

RUNTIME_DEFAULTS = get_runtime_defaults()
DEFAULT_EVIDENCE_MODE: EvidenceMode = RUNTIME_DEFAULTS.run.evidence_mode
DEFAULT_TEXT_SOURCE_MODE: TextSourceMode = RUNTIME_DEFAULTS.run.text_source_mode
DEFAULT_SUMMARIZERS: tuple[SummarizerId, ...] = RUNTIME_DEFAULTS.run.summarizers

DEFAULT_FULL_CASE_MATRIX_EVIDENCE_MODES: tuple[EvidenceMode, ...] = ("plot", "table", "plot+table")
DEFAULT_FULL_CASE_MATRIX_PROMPT_VARIANTS: tuple[str, ...] = (
    "none",
    "role",
    "insights",
    "example",
    "role+example",
    "role+insights",
    "insights+example",
    "all_three",
)
DEFAULT_FULL_CASE_MATRIX_REPETITIONS: tuple[int, ...] = (1, 2, 3)


def resolve_full_case_matrix_evidence_modes(values: list[str] | None) -> tuple[EvidenceMode, ...]:
    """Normalize CLI evidence-mode values into the typed full-case matrix tuple."""
    return tuple(cast(EvidenceMode, value) for value in (values or DEFAULT_FULL_CASE_MATRIX_EVIDENCE_MODES))


def resolve_full_case_matrix_prompt_variants(values: list[str] | None) -> tuple[str, ...]:
    """Normalize CLI prompt-variant values into the canonical full-case matrix tuple."""
    return tuple(values or DEFAULT_FULL_CASE_MATRIX_PROMPT_VARIANTS)


def resolve_full_case_matrix_repetitions(values: list[int] | None) -> tuple[int, ...]:
    """Normalize CLI repetition values into the canonical full-case matrix tuple."""
    return tuple(values or DEFAULT_FULL_CASE_MATRIX_REPETITIONS)
