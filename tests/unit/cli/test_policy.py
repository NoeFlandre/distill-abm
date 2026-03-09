from __future__ import annotations

import pytest
import typer

from distill_abm.cli_policy import validate_benchmark_model_policy


def test_validate_benchmark_model_policy_accepts_allowed_openrouter_model() -> None:
    validate_benchmark_model_policy(
        provider="openrouter",
        model="google/gemini-3.1-pro-preview",
        allow_debug_model=False,
        benchmark_models={
            ("openrouter", "google/gemini-3.1-pro-preview"),
            ("openrouter", "moonshotai/kimi-k2.5"),
        },
    )


def test_validate_benchmark_model_policy_rejects_unknown_model() -> None:
    with pytest.raises(typer.BadParameter, match="unsupported benchmark model"):
        validate_benchmark_model_policy(
            provider="openrouter",
            model="unknown/model",
            allow_debug_model=False,
            benchmark_models={("openrouter", "google/gemini-3.1-pro-preview")},
        )


def test_validate_benchmark_model_policy_allows_debug_model_with_flag() -> None:
    validate_benchmark_model_policy(
        provider="mistral",
        model="mistral-medium-latest",
        allow_debug_model=True,
        benchmark_models={("openrouter", "google/gemini-3.1-pro-preview")},
    )
