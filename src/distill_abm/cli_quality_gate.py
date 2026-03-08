"""Convenience helpers for the canonical validation command."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from distill_abm.agent_validation import ValidationProfile

QualityGateScope = Literal["static", "pre-llm", "full"]


class QualityGateSelection(BaseModel):
    """Resolved validation plan for the convenience quality-gate command."""

    profile: ValidationProfile
    checks: list[str] | None


def resolve_quality_gate_selection(
    *,
    scope: QualityGateScope,
    explicit_checks: list[str] | None,
    explicit_profile: ValidationProfile | None,
) -> QualityGateSelection:
    """Resolve the effective validation profile and optional check filter."""
    default_profile: ValidationProfile
    default_checks: list[str] | None

    if scope == "static":
        default_profile = "quick"
        default_checks = ["ruff", "mypy"]
    elif scope == "pre-llm":
        default_profile = "quick"
        default_checks = None
    else:
        default_profile = "default"
        default_checks = None

    return QualityGateSelection(
        profile=explicit_profile or default_profile,
        checks=explicit_checks or default_checks,
    )
