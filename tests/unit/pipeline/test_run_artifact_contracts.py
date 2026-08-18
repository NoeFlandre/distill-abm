from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from distill_abm.pipeline.run_artifact_contracts import (
    CASE_SUMMARY_FILENAME,
    DOE_SMOKE_REPORT_FILENAME,
    INGEST_SMOKE_REPORT_FILENAME,
    RUN_LOG_FILENAME,
    VALIDATION_STATE_FILENAME,
    VIZ_SMOKE_REPORT_FILENAME,
    case_summary_path,
    create_run_root,
    doe_smoke_report_path,
    find_previous_run_root,
    ingest_smoke_report_path,
    latest_report_pointer_path,
    latest_run_pointer_path,
    resolve_run_root,
    run_log_path,
    runs_root_path,
    sampled_smoke_report_path,
    validation_state_path,
    viewer_html_path,
    viz_smoke_report_path,
    write_latest_run_pointer,
)


def test_run_artifact_contracts_expose_stable_filenames() -> None:
    assert CASE_SUMMARY_FILENAME == "00_case_summary.json"
    assert VALIDATION_STATE_FILENAME == "validation_state.json"
    assert RUN_LOG_FILENAME == "run.log.jsonl"


def test_run_artifact_contracts_build_standard_paths(tmp_path: Path) -> None:
    output_root = tmp_path / "results"
    run_root = output_root / "runs" / "run_1"
    case_dir = run_root / "cases" / "case_1"

    assert latest_run_pointer_path(output_root) == output_root / "latest_run.txt"
    assert latest_report_pointer_path(output_root) == output_root / "latest_report_path.txt"
    assert runs_root_path(output_root) == output_root / "runs"
    assert run_log_path(run_root) == run_root / "run.log.jsonl"
    assert viewer_html_path(run_root) == run_root / "review.html"
    assert sampled_smoke_report_path(run_root) == run_root / "smoke_local_qwen_report.json"
    assert ingest_smoke_report_path(run_root) == run_root / INGEST_SMOKE_REPORT_FILENAME
    assert viz_smoke_report_path(run_root) == run_root / VIZ_SMOKE_REPORT_FILENAME
    assert doe_smoke_report_path(run_root) == run_root / DOE_SMOKE_REPORT_FILENAME
    assert case_summary_path(case_dir) == case_dir / "00_case_summary.json"
    assert validation_state_path(case_dir) == case_dir / "validation_state.json"


def test_run_artifact_contracts_create_run_root_and_update_latest_pointer(tmp_path: Path) -> None:
    started_at = datetime(2026, 8, 18, 12, 34, 56, 789123, tzinfo=UTC)

    run_id, run_root = create_run_root(output_root=tmp_path, started_at=started_at)
    write_latest_run_pointer(output_root=tmp_path, run_root=run_root)

    assert run_id == "run_20260818_123456_789123"
    assert run_root == tmp_path / "runs" / run_id
    assert run_root.is_dir()
    assert latest_run_pointer_path(tmp_path).read_text(encoding="utf-8") == str(run_root)


def test_resolve_run_root_uses_latest_run_pointer(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "run_1"
    run_root.mkdir(parents=True)
    latest_run_pointer_path(tmp_path).write_text(str(run_root), encoding="utf-8")

    assert resolve_run_root(tmp_path) == run_root


def test_resolve_run_root_returns_candidate_when_no_pointer(tmp_path: Path) -> None:
    assert resolve_run_root(tmp_path) == tmp_path


def test_find_previous_run_root_returns_newest_directory_excluding_current(tmp_path: Path) -> None:
    older_run_root = tmp_path / "runs" / "run_20260818_100000_000000"
    newer_run_root = tmp_path / "runs" / "run_20260818_110000_000000"
    current_run_root = tmp_path / "runs" / "run_20260818_120000_000000"
    older_run_root.mkdir(parents=True)
    newer_run_root.mkdir()
    current_run_root.mkdir()

    assert find_previous_run_root(output_root=tmp_path, current_run_id=current_run_root.name) == newer_run_root


def test_find_previous_run_root_returns_none_without_prior_runs(tmp_path: Path) -> None:
    assert find_previous_run_root(output_root=tmp_path, current_run_id="run_current") is None
