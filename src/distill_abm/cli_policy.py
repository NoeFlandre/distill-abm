"""CLI-facing benchmark model policy helpers."""

from __future__ import annotations

from collections.abc import Callable

import typer


def validate_benchmark_model_policy(
    *,
    provider: str,
    model: str,
    allow_debug_model: bool,
    benchmark_models: set[tuple[str, str]],
    assert_ollama_model_available: Callable[[str], None],
) -> None:
    """Validate the configured benchmark model against the repository policy."""
    _ = allow_debug_model
    key = (provider.strip().lower(), model.strip())
    if key not in benchmark_models:
        allowed = ", ".join(f"{item_provider}:{item_model}" for item_provider, item_model in sorted(benchmark_models))
        raise typer.BadParameter(
            f"unsupported benchmark model '{provider}:{model}'. Allowed benchmark models: {allowed}."
        )
    if key == ("ollama", "qwen3.5:0.8b"):
        assert_ollama_model_available(model)
