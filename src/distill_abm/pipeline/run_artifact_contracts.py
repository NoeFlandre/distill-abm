"""Shared filename and path contracts for run-oriented smoke workflows."""

from __future__ import annotations

from pathlib import Path

CASE_SUMMARY_FILENAME = "00_case_summary.json"
VALIDATION_STATE_FILENAME = "validation_state.json"
RUN_LOG_FILENAME = "run.log.jsonl"
LATEST_RUN_POINTER_FILENAME = "latest_run.txt"
LATEST_REPORT_POINTER_FILENAME = "latest_report_path.txt"
VIEWER_HTML_FILENAME = "review.html"
SAMPLED_SMOKE_REPORT_FILENAME = "smoke_local_qwen_report.json"
FULL_CASE_MATRIX_REPORT_FILENAME = "smoke_full_case_matrix_report.json"
INGEST_SMOKE_REPORT_FILENAME = "ingest_smoke_report.json"
VIZ_SMOKE_REPORT_FILENAME = "viz_smoke_report.json"
DOE_SMOKE_REPORT_FILENAME = "doe_smoke_report.json"


def runs_root_path(output_root: Path) -> Path:
    return output_root / "runs"


def latest_run_pointer_path(output_root: Path) -> Path:
    return output_root / LATEST_RUN_POINTER_FILENAME


def latest_report_pointer_path(output_root: Path) -> Path:
    return output_root / LATEST_REPORT_POINTER_FILENAME


def run_log_path(run_root: Path) -> Path:
    return run_root / RUN_LOG_FILENAME


def viewer_html_path(run_root: Path) -> Path:
    return run_root / VIEWER_HTML_FILENAME


def sampled_smoke_report_path(run_root: Path) -> Path:
    return run_root / SAMPLED_SMOKE_REPORT_FILENAME


def full_case_matrix_report_path(run_root: Path) -> Path:
    return run_root / FULL_CASE_MATRIX_REPORT_FILENAME


def ingest_smoke_report_path(run_root: Path) -> Path:
    return run_root / INGEST_SMOKE_REPORT_FILENAME


def viz_smoke_report_path(run_root: Path) -> Path:
    return run_root / VIZ_SMOKE_REPORT_FILENAME


def doe_smoke_report_path(run_root: Path) -> Path:
    return run_root / DOE_SMOKE_REPORT_FILENAME


def case_summary_path(case_dir: Path) -> Path:
    return case_dir / CASE_SUMMARY_FILENAME


def validation_state_path(case_dir: Path) -> Path:
    return case_dir / VALIDATION_STATE_FILENAME


def resolve_run_root(path: Path) -> Path:
    """Resolve either a concrete run directory or a root containing a latest-run pointer."""
    latest_run_path = latest_run_pointer_path(path)
    if latest_run_path.exists():
        latest_text = latest_run_path.read_text(encoding="utf-8").strip()
        if latest_text:
            return Path(latest_text)
    return path
