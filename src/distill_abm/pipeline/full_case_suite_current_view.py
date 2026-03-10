"""Helpers for syncing stable nested-ABM current-view artifacts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from distill_abm.pipeline.run_artifact_contracts import latest_run_pointer_path, run_log_path


@dataclass(frozen=True)
class StableAbmCurrentViewPaths:
    """Stable nested-ABM artifact paths copied into the `current/` view."""

    run_root: Path
    run_log_path: Path
    report_json_path: Path
    report_markdown_path: Path
    review_csv_path: Path


def sync_stable_abm_current_view(*, abm_output_root: Path, run_root: Path) -> StableAbmCurrentViewPaths:
    """Copy the latest nested ABM artifacts into a stable `current/` view."""

    current_root = abm_output_root / "current"
    current_root.mkdir(parents=True, exist_ok=True)
    stable_paths = StableAbmCurrentViewPaths(
        run_root=run_root,
        run_log_path=current_root / "run.log.jsonl",
        report_json_path=current_root / "smoke_full_case_matrix_report.json",
        report_markdown_path=current_root / "smoke_full_case_matrix_report.md",
        review_csv_path=current_root / "request_review.csv",
    )
    source_paths = (
        (run_log_path(run_root), stable_paths.run_log_path),
        (run_root / "smoke_full_case_matrix_report.json", stable_paths.report_json_path),
        (run_root / "smoke_full_case_matrix_report.md", stable_paths.report_markdown_path),
        (run_root / "request_review.csv", stable_paths.review_csv_path),
    )
    for source_path, destination_path in source_paths:
        if source_path.exists():
            shutil.copy2(source_path, destination_path)
    latest_run_pointer_path(abm_output_root).write_text(str(run_root), encoding="utf-8")
    return stable_paths
