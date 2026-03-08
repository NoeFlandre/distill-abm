from __future__ import annotations

from pathlib import Path

from distill_abm.pipeline.smoke_io import write_csv_rows
from distill_abm.pipeline.smoke_reporting import (
    render_markdown_report,
    write_global_master_csv,
    write_run_master_csv,
)
from distill_abm.pipeline.smoke_types import (
    RESPONSE_BUNDLE_COLUMNS,
    SmokeCase,
    SmokeCaseResult,
    SmokeSuiteInputs,
    SmokeSuiteResult,
)


def test_write_run_master_csv_merges_case_response_rows(tmp_path: Path) -> None:
    output_root = tmp_path / "smoke"
    case_dir = output_root / "cases" / "case-1"
    case_dir.mkdir(parents=True)
    case_rows_path = case_dir / "case_responses.csv"
    write_csv_rows(
        case_rows_path,
        [
            {
                "run_output_dir": str(output_root),
                "case_id": "case-1",
                "response_kind": "context",
                "prompt_signature": "abc",
                "response_path": "response.txt",
            }
        ],
        RESPONSE_BUNDLE_COLUMNS,
    )

    case_result = SmokeCaseResult(
        case=SmokeCase(case_id="case-1", evidence_mode="plot", text_source_mode="summary_only"),
        status="ok",
        output_dir=case_dir,
        case_rows_csv_path=case_rows_path,
    )

    run_master_csv = write_run_master_csv(output_root, [case_result])

    contents = run_master_csv.read_text(encoding="utf-8")
    assert "case-1" in contents


def test_write_global_master_csv_merges_new_rows(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    results_root = tmp_path / "results"
    run_master_csv = results_root / "smoke" / "master_responses.csv"
    run_master_csv.parent.mkdir(parents=True)
    write_csv_rows(
        run_master_csv,
        [
            {
                "run_output_dir": str(run_master_csv.parent),
                "case_id": "case-1",
                "response_kind": "context",
                "prompt_signature": "abc",
                "response_path": "response.txt",
            }
        ],
        RESPONSE_BUNDLE_COLUMNS,
    )
    monkeypatch.chdir(tmp_path)

    global_master = write_global_master_csv(run_master_csv)

    assert global_master == Path("results") / "master_responses.csv"
    assert global_master.exists()
    assert "case-1" in global_master.read_text(encoding="utf-8")


def test_render_markdown_report_includes_key_summary_fields(tmp_path: Path) -> None:
    result = SmokeSuiteResult(
        provider="openrouter",
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        started_at_utc="2026-03-08T00:00:00Z",
        finished_at_utc="2026-03-08T00:01:00Z",
        inputs=SmokeSuiteInputs(
            csv_path=tmp_path / "simulation.csv",
            parameters_path=tmp_path / "parameters.txt",
            documentation_path=tmp_path / "documentation.txt",
            output_dir=tmp_path / "smoke",
            model="nvidia/nemotron-nano-12b-v2-vl:free",
            metric_pattern="metric",
            metric_description="description",
        ),
        success=True,
        cases=[],
        report_markdown_path=tmp_path / "report.md",
        report_json_path=tmp_path / "report.json",
    )

    markdown = render_markdown_report(result)

    assert "# Qwen Smoke Suite Report" in markdown
    assert "Provider: `openrouter`" in markdown
    assert "Success: `True`" in markdown
