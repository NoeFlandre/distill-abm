from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from distill_abm.agent_validation import default_validation_checks, run_validation_suite, validation_checks_for_profile


def test_default_validation_checks_expose_canonical_agent_workflow() -> None:
    assert [item.check_id for item in default_validation_checks()] == [
        "pytest",
        "ruff",
        "mypy",
        "build",
        "smoke-ingest-netlogo",
    ]


def test_validation_profiles_expose_expected_check_sets() -> None:
    assert validation_checks_for_profile("quick") == ["ruff", "mypy", "smoke-ingest-netlogo"]
    assert validation_checks_for_profile("default") == ["pytest", "ruff", "mypy", "build", "smoke-ingest-netlogo"]


def test_run_validation_suite_writes_structured_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], capture_output: bool, text: bool) -> SimpleNamespace:
        assert capture_output is True
        assert text is True
        return SimpleNamespace(returncode=0, stdout=f"ok {' '.join(command)}", stderr="")

    def fake_run_ingest_smoke_suite(*, abm_models, output_root, stage_ids):  # type: ignore[no-untyped-def]
        output_root.mkdir(parents=True, exist_ok=True)
        report_json = output_root / "ingest_smoke_report.json"
        report_md = output_root / "ingest_smoke_report.md"
        report_json.write_text("{}", encoding="utf-8")
        report_md.write_text("# report", encoding="utf-8")
        return SimpleNamespace(
            success=True,
            failed_abms=[],
            report_json_path=report_json,
            report_markdown_path=report_md,
        )

    monkeypatch.setattr("distill_abm.agent_validation.subprocess.run", fake_run)
    monkeypatch.setattr("distill_abm.agent_validation.run_ingest_smoke_suite", fake_run_ingest_smoke_suite)

    result = run_validation_suite(
        output_root=tmp_path / "validation",
        abm_models={"fauna": Path("data/fauna_abm/fauna.nlogo")},
    )

    assert result.success is True
    assert result.report_json_path.exists()
    assert result.report_markdown_path.exists()
    assert result.ingest_smoke_report_json_path is not None
    assert result.ingest_smoke_report_markdown_path is not None
    assert [item.check_id for item in result.check_results] == [
        "pytest",
        "ruff",
        "mypy",
        "build",
        "smoke-ingest-netlogo",
    ]
    assert result.profile == "default"
    assert all(item.execution_mode == "fresh" for item in result.check_results)


def test_run_validation_suite_marks_non_selected_checks_as_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], capture_output: bool, text: bool) -> SimpleNamespace:
        _ = capture_output, text
        return SimpleNamespace(returncode=0, stdout=f"ok {' '.join(command)}", stderr="")

    monkeypatch.setattr("distill_abm.agent_validation.subprocess.run", fake_run)

    result = run_validation_suite(
        output_root=tmp_path / "validation",
        abm_models={"fauna": Path("data/fauna_abm/fauna.nlogo")},
        checks=["ruff"],
    )

    by_id = {item.check_id: item for item in result.check_results}
    assert by_id["ruff"].status == "ok"
    assert by_id["pytest"].status == "skipped"
    assert by_id["pytest"].execution_mode == "skipped"


def test_run_validation_suite_records_command_launch_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], capture_output: bool, text: bool) -> SimpleNamespace:
        _ = capture_output, text
        raise OSError(f"cannot execute {' '.join(command)}")

    monkeypatch.setattr("distill_abm.agent_validation.subprocess.run", fake_run)

    result = run_validation_suite(
        output_root=tmp_path / "validation",
        abm_models={"fauna": Path("data/fauna_abm/fauna.nlogo")},
        checks=["ruff"],
    )

    assert result.success is False
    assert result.failed_checks == ["ruff"]
    by_id = {item.check_id: item for item in result.check_results}
    assert by_id["ruff"].status == "failed"
    assert by_id["ruff"].error_code == "command_failed"
    assert "cannot execute uv run ruff check ." in (by_id["ruff"].error or "")
