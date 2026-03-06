"""Shared CLI output helpers for structured smoke-style commands."""

from __future__ import annotations

import typer

from distill_abm.cli_models import SmokeCommandResult


def ensure_required_stage_ids(*, selected_stage_ids: list[str], required_stage_ids: list[str], label: str) -> None:
    """Fail with a consistent CLI error when required stage ids are absent."""
    missing_required = [item for item in required_stage_ids if item not in selected_stage_ids]
    if missing_required:
        raise typer.BadParameter(f"required stage(s) missing from {label} selection: {', '.join(missing_required)}")


def emit_smoke_command_result(
    *,
    command_result: SmokeCommandResult,
    json_output: bool,
    markdown_label: str,
    json_label: str,
    failure_label: str,
) -> None:
    """Render one smoke-like command result consistently across the CLI."""
    if json_output:
        typer.echo(command_result.model_dump_json(indent=2))
        if not command_result.success:
            raise typer.Exit(code=1)
        return

    typer.echo(f"{markdown_label}: {command_result.report_markdown_path}")
    typer.echo(f"{json_label}: {command_result.report_json_path}")
    if not command_result.success:
        typer.echo(f"{failure_label}: {', '.join(command_result.failed_items)}")
        raise typer.Exit(code=1)
