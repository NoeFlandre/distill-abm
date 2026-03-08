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


def case_summary_path(case_dir: Path) -> Path:
    return case_dir / CASE_SUMMARY_FILENAME


def validation_state_path(case_dir: Path) -> Path:
    return case_dir / VALIDATION_STATE_FILENAME
