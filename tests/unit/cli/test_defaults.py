from __future__ import annotations

from distill_abm.cli_defaults import (
    DEFAULT_FULL_CASE_MATRIX_EVIDENCE_MODES,
    DEFAULT_FULL_CASE_MATRIX_PROMPT_VARIANTS,
    DEFAULT_FULL_CASE_MATRIX_REPETITIONS,
    resolve_full_case_matrix_evidence_modes,
    resolve_full_case_matrix_prompt_variants,
    resolve_full_case_matrix_repetitions,
)


def test_cli_defaults_expose_full_case_matrix_defaults() -> None:
    assert DEFAULT_FULL_CASE_MATRIX_EVIDENCE_MODES == ("plot", "table", "plot+table")
    assert DEFAULT_FULL_CASE_MATRIX_PROMPT_VARIANTS[0] == "none"
    assert DEFAULT_FULL_CASE_MATRIX_REPETITIONS == (1, 2, 3)


def test_resolve_full_case_matrix_values_use_defaults_when_missing() -> None:
    assert resolve_full_case_matrix_evidence_modes(None) == DEFAULT_FULL_CASE_MATRIX_EVIDENCE_MODES
    assert resolve_full_case_matrix_prompt_variants(None) == DEFAULT_FULL_CASE_MATRIX_PROMPT_VARIANTS
    assert resolve_full_case_matrix_repetitions(None) == DEFAULT_FULL_CASE_MATRIX_REPETITIONS


def test_resolve_full_case_matrix_values_preserve_explicit_inputs() -> None:
    assert resolve_full_case_matrix_evidence_modes(["table", "plot"]) == ("table", "plot")
    assert resolve_full_case_matrix_prompt_variants(["role", "example"]) == ("role", "example")
    assert resolve_full_case_matrix_repetitions([2, 4]) == (2, 4)
