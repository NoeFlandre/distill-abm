from __future__ import annotations

from pathlib import Path

import pytest
import typer

from distill_abm.cli_models import SmokeCommandResult
from distill_abm.cli_output import emit_smoke_command_result, ensure_required_stage_ids


def test_ensure_required_stage_ids_accepts_present_items() -> None:
    ensure_required_stage_ids(
        selected_stage_ids=["documentation", "code"],
        required_stage_ids=["code"],
        label="ingest smoke",
    )


def test_ensure_required_stage_ids_rejects_missing_items() -> None:
    with pytest.raises(typer.BadParameter, match="required stage\\(s\\) missing"):
        ensure_required_stage_ids(
            selected_stage_ids=["documentation"],
            required_stage_ids=["code"],
            label="ingest smoke",
        )


def test_emit_smoke_command_result_exits_for_failed_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    result = SmokeCommandResult(
        command="smoke-viz",
        success=False,
        report_json_path=Path("report.json"),
        report_markdown_path=Path("report.md"),
        failed_items=["fauna"],
    )

    with pytest.raises(typer.Exit) as exc:
        emit_smoke_command_result(
            command_result=result,
            json_output=True,
            markdown_label="markdown",
            json_label="json",
            failure_label="failure",
        )

    assert exc.value.exit_code == 1
    assert '"success": false' in capsys.readouterr().out


def test_emit_smoke_command_result_prints_paths_for_success(capsys: pytest.CaptureFixture[str]) -> None:
    result = SmokeCommandResult(
        command="smoke-viz",
        success=True,
        report_json_path=Path("report.json"),
        report_markdown_path=Path("report.md"),
    )

    emit_smoke_command_result(
        command_result=result,
        json_output=False,
        markdown_label="markdown",
        json_label="json",
        failure_label="failure",
    )

    output = capsys.readouterr().out
    assert "markdown: report.md" in output
    assert "json: report.json" in output


def test_emit_smoke_command_result_prints_failure_and_exits(capsys: pytest.CaptureFixture[str]) -> None:
    result = SmokeCommandResult(
        command="smoke-viz",
        success=False,
        report_json_path=Path("report.json"),
        report_markdown_path=Path("report.md"),
        failed_items=["fauna", "grazing"],
    )

    with pytest.raises(typer.Exit) as exc:
        emit_smoke_command_result(
            command_result=result,
            json_output=False,
            markdown_label="markdown",
            json_label="json",
            failure_label="failure",
        )

    assert exc.value.exit_code == 1
    output = capsys.readouterr().out
    assert "failure: fauna, grazing" in output
