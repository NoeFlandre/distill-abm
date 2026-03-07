"""Canonical local validation suite for agent-friendly verification."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from distill_abm.ingest.ingest_smoke import run_ingest_smoke_suite

ValidationStatus = Literal["ok", "failed", "skipped"]
ValidationExecutionMode = Literal["fresh", "skipped"]
ValidationProfile = Literal["quick", "default", "full"]
ValidationErrorCode = Literal[
    "command_failed",
    "ingest_smoke_failed",
    "unknown_check",
]


class ValidationCheck(BaseModel):
    """One validation step exposed by the agent-facing CLI."""

    check_id: str
    description: str
    command: list[str] = Field(default_factory=list)


class ValidationCheckResult(BaseModel):
    """Structured result for one validation check."""

    check_id: str
    description: str
    status: ValidationStatus
    execution_mode: ValidationExecutionMode = "fresh"
    command: list[str] = Field(default_factory=list)
    exit_code: int | None = None
    stdout_preview: str = ""
    stderr_preview: str = ""
    artifact_paths: list[Path] = Field(default_factory=list)
    error_code: ValidationErrorCode | None = None
    error: str | None = None


class ValidationSuiteResult(BaseModel):
    """Top-level validation report written for agents and humans."""

    started_at_utc: str
    finished_at_utc: str
    output_root: Path
    success: bool
    profile: ValidationProfile
    selected_checks: list[str]
    failed_checks: list[str] = Field(default_factory=list)
    check_results: list[ValidationCheckResult] = Field(default_factory=list)
    ingest_smoke_report_json_path: Path | None = None
    ingest_smoke_report_markdown_path: Path | None = None
    report_json_path: Path
    report_markdown_path: Path


def default_validation_checks() -> list[ValidationCheck]:
    """Return the canonical local checks an agent should use after code changes."""
    return [
        ValidationCheck(
            check_id="pytest",
            description="Run the full pytest suite.",
            command=["uv", "run", "pytest"],
        ),
        ValidationCheck(
            check_id="ruff",
            description="Run Ruff lint checks.",
            command=["uv", "run", "ruff", "check", "."],
        ),
        ValidationCheck(
            check_id="mypy",
            description="Run mypy on src and tests.",
            command=["uv", "run", "mypy", "src", "tests"],
        ),
        ValidationCheck(
            check_id="build",
            description="Build the package.",
            command=["uv", "build"],
        ),
        ValidationCheck(
            check_id="smoke-ingest-netlogo",
            description="Run pre-LLM NetLogo ingest smoke checks.",
        ),
    ]


def validation_checks_for_profile(profile: ValidationProfile) -> list[str]:
    """Return the default check roster for a named validation profile."""
    if profile == "quick":
        return ["ruff", "mypy", "smoke-ingest-netlogo"]
    return ["pytest", "ruff", "mypy", "build", "smoke-ingest-netlogo"]


def run_validation_suite(
    *,
    output_root: Path,
    abm_models: dict[str, Path],
    checks: list[str] | None = None,
    ingest_stage_ids: list[str] | None = None,
    profile: ValidationProfile = "default",
) -> ValidationSuiteResult:
    """Run the canonical local validation suite and persist structured reports."""
    started_at = datetime.now(UTC)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_checks = _select_checks(checks, profile=profile)
    selected_ids = {item.check_id for item in selected_checks}
    all_checks = default_validation_checks()

    check_results: list[ValidationCheckResult] = []
    failed_checks: list[str] = []
    ingest_smoke_report_json_path: Path | None = None
    ingest_smoke_report_markdown_path: Path | None = None

    for check in selected_checks:
        if check.check_id == "smoke-ingest-netlogo":
            smoke_output_root = output_root / "ingest_smoke"
            smoke_result = run_ingest_smoke_suite(
                abm_models=abm_models,
                output_root=smoke_output_root,
                stage_ids=ingest_stage_ids,
            )
            status: ValidationStatus = "ok" if smoke_result.success else "failed"
            if status == "failed":
                failed_checks.append(check.check_id)
            ingest_smoke_report_json_path = smoke_result.report_json_path
            ingest_smoke_report_markdown_path = smoke_result.report_markdown_path
            check_results.append(
                ValidationCheckResult(
                    check_id=check.check_id,
                    description=check.description,
                    status=status,
                    artifact_paths=[smoke_result.report_json_path, smoke_result.report_markdown_path],
                    error_code=None if smoke_result.success else "ingest_smoke_failed",
                    error=None if smoke_result.success else f"failed_abms: {', '.join(smoke_result.failed_abms)}",
                )
            )
            continue

        try:
            completed = subprocess.run(check.command, capture_output=True, text=True)
        except OSError as exc:
            failed_checks.append(check.check_id)
            check_results.append(
                ValidationCheckResult(
                    check_id=check.check_id,
                    description=check.description,
                    status="failed",
                    command=check.command,
                    error_code="command_failed",
                    error=str(exc),
                )
            )
            continue

        status = "ok" if completed.returncode == 0 else "failed"
        if status == "failed":
            failed_checks.append(check.check_id)
        check_results.append(
            ValidationCheckResult(
                check_id=check.check_id,
                description=check.description,
                status=status,
                command=check.command,
                exit_code=completed.returncode,
                stdout_preview=completed.stdout[:2000],
                stderr_preview=completed.stderr[:2000],
                error_code=None if completed.returncode == 0 else "command_failed",
            )
        )

    for check in all_checks:
        if check.check_id in selected_ids:
            continue
        check_results.append(
            ValidationCheckResult(
                check_id=check.check_id,
                description=check.description,
                status="skipped",
                execution_mode="skipped",
                command=check.command,
            )
        )

    finished_at = datetime.now(UTC)
    report_json_path = output_root / "validation_report.json"
    report_markdown_path = output_root / "validation_report.md"
    result = ValidationSuiteResult(
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        output_root=output_root,
        success=not failed_checks,
        profile=profile,
        selected_checks=[item.check_id for item in selected_checks],
        failed_checks=failed_checks,
        check_results=check_results,
        ingest_smoke_report_json_path=ingest_smoke_report_json_path,
        ingest_smoke_report_markdown_path=ingest_smoke_report_markdown_path,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
    )
    report_json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    report_markdown_path.write_text(_render_markdown_report(result), encoding="utf-8")
    return result


def _select_checks(checks: list[str] | None, profile: ValidationProfile) -> list[ValidationCheck]:
    available = {item.check_id: item for item in default_validation_checks()}
    requested = checks if checks is not None else validation_checks_for_profile(profile)
    if not requested:
        return []
    unknown = [item for item in requested if item not in available]
    if unknown:
        known = ", ".join(sorted(available))
        raise ValueError(f"unknown validation check(s): {', '.join(unknown)}. Known checks: {known}")
    return [available[item] for item in requested]


def _render_markdown_report(result: ValidationSuiteResult) -> str:
    lines = [
        "# Validation Report",
        "",
        f"- success: `{result.success}`",
        f"- profile: `{result.profile}`",
        f"- selected_checks: `{', '.join(result.selected_checks)}`",
        f"- failed_checks: `{', '.join(result.failed_checks) if result.failed_checks else 'none'}`",
        "",
    ]
    for check in result.check_results:
        lines.append(f"## {check.check_id}")
        lines.append(f"- status: `{check.status}`")
        lines.append(f"- execution_mode: `{check.execution_mode}`")
        if check.command:
            lines.append(f"- command: `{' '.join(check.command)}`")
        if check.exit_code is not None:
            lines.append(f"- exit_code: `{check.exit_code}`")
        if check.artifact_paths:
            lines.append(f"- artifacts: `{', '.join(str(path) for path in check.artifact_paths)}`")
        if check.error_code:
            lines.append(f"- error_code: `{check.error_code}`")
        if check.error:
            lines.append(f"- error: `{check.error}`")
        lines.append("")
    return "\n".join(lines)
