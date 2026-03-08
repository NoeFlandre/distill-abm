from __future__ import annotations

from unittest.mock import Mock

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
        assert_ollama_model_available=Mock(),
    )


def test_validate_benchmark_model_policy_rejects_unknown_model() -> None:
    with pytest.raises(typer.BadParameter, match="unsupported benchmark model"):
        validate_benchmark_model_policy(
            provider="openrouter",
            model="unknown/model",
            allow_debug_model=False,
            benchmark_models={("openrouter", "google/gemini-3.1-pro-preview")},
            assert_ollama_model_available=Mock(),
        )


def test_validate_benchmark_model_policy_checks_local_ollama_model() -> None:
    checker = Mock()

    validate_benchmark_model_policy(
        provider="ollama",
        model="qwen3.5:0.8b",
        allow_debug_model=False,
        benchmark_models={("ollama", "qwen3.5:0.8b")},
        assert_ollama_model_available=checker,
    )

    checker.assert_called_once_with("qwen3.5:0.8b")
