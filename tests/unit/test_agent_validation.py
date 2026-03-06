from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from distill_abm.agent_validation import default_validation_checks, run_validation_suite


def test_default_validation_checks_expose_canonical_agent_workflow() -> None:
    assert [item.check_id for item in default_validation_checks()] == [
        "pytest",
        "ruff",
        "mypy",
        "build",
        "smoke-ingest-netlogo",
    ]


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
