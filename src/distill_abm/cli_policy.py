"""CLI-facing benchmark model policy helpers."""

from __future__ import annotations

import typer


def validate_benchmark_model_policy(
    *,
    provider: str,
    model: str,
    allow_debug_model: bool,
    benchmark_models: set[tuple[str, str]],
) -> None:
    """Validate the configured benchmark model against the repository policy."""
    key = (provider.strip().lower(), model.strip())
    if allow_debug_model and key not in benchmark_models:
        return
    if key not in benchmark_models:
        allowed = ", ".join(f"{item_provider}:{item_model}" for item_provider, item_model in sorted(benchmark_models))
        raise typer.BadParameter(
            f"unsupported benchmark model '{provider}:{model}'. Allowed benchmark models: {allowed}."
        )
