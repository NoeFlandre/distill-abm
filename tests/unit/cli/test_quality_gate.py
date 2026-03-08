from __future__ import annotations

from distill_abm.cli_quality_gate import resolve_quality_gate_selection


def test_resolve_quality_gate_selection_uses_static_scope_checks() -> None:
    selection = resolve_quality_gate_selection(scope="static", explicit_checks=None, explicit_profile=None)

    assert selection.profile == "quick"
    assert selection.checks == ["ruff", "mypy"]


def test_resolve_quality_gate_selection_uses_pre_llm_scope_defaults() -> None:
    selection = resolve_quality_gate_selection(scope="pre-llm", explicit_checks=None, explicit_profile=None)

    assert selection.profile == "quick"
    assert selection.checks is None


def test_resolve_quality_gate_selection_prefers_explicit_checks() -> None:
    selection = resolve_quality_gate_selection(
        scope="full",
        explicit_checks=["pytest"],
        explicit_profile=None,
    )

    assert selection.profile == "default"
    assert selection.checks == ["pytest"]


def test_resolve_quality_gate_selection_prefers_explicit_profile() -> None:
    selection = resolve_quality_gate_selection(
        scope="static",
        explicit_checks=None,
        explicit_profile="full",
    )

    assert selection.profile == "full"
    assert selection.checks == ["ruff", "mypy"]
