"""Progress models and stable-view helpers for the full-case suite smoke."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from distill_abm.pipeline.full_case_suite_current_view import sync_stable_abm_current_view
from distill_abm.pipeline.local_qwen_monitor import collect_local_qwen_monitor_snapshot


class FullCaseSuiteProgressAbm(BaseModel):
    """Live progress snapshot for one ABM inside a suite run."""

    abm: str
    status: str
    attempt: int | None = None
    planned_case_count: int
    completed_case_count: int = 0
    failed_case_count: int = 0
    run_root: Path | None = None
    run_log_path: Path | None = None
    report_json_path: Path | None = None
    running_case_id: str | None = None
    running_case_status: str | None = None
    running_case_detail: str | None = None
    last_error: str | None = None


class FullCaseSuiteProgress(BaseModel):
    """Stable suite-level progress contract for live monitoring."""

    run_id: str
    run_root: Path
    output_root: Path
    model: str
    started_at_utc: str
    finished_at_utc: str | None = None
    status: str
    current_abm: str | None = None
    current_attempt: int | None = None
    total_abms: int
    completed_abm_count: int
    failed_abm_count: int
    planned_case_count: int
    completed_case_count: int
    failed_case_count: int
    current_case_id: str | None = None
    current_case_status: str | None = None
    current_case_detail: str | None = None
    remaining_abms: list[str] = Field(default_factory=list)
    abms: list[FullCaseSuiteProgressAbm] = Field(default_factory=list)


def build_suite_progress(
    *,
    run_id: str,
    run_root: Path,
    output_root: Path,
    model: str,
    started_at: datetime,
    status: str,
    current_abm: str | None,
    current_attempt: int | None,
    remaining_abms: list[str],
    progress_by_name: dict[str, FullCaseSuiteProgressAbm],
    finished_at: datetime | None = None,
) -> FullCaseSuiteProgress:
    """Build the current suite-level progress view from per-ABM progress."""

    abm_progress = [
        refresh_progress_abm_snapshot(output_root=output_root, progress=progress_by_name[abm])
        for abm in progress_by_name
    ]
    completed_abm_count = sum(1 for item in abm_progress if item.status == "completed")
    failed_abm_count = sum(1 for item in abm_progress if item.status == "failed")
    planned_case_count = sum(item.planned_case_count for item in abm_progress)
    completed_case_count = sum(item.completed_case_count for item in abm_progress)
    failed_case_count = sum(item.failed_case_count for item in abm_progress)
    current_abm_progress = next((item for item in abm_progress if item.abm == current_abm), None)
    return FullCaseSuiteProgress(
        run_id=run_id,
        run_root=run_root,
        output_root=output_root,
        model=model,
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat() if finished_at is not None else None,
        status=status,
        current_abm=current_abm,
        current_attempt=current_attempt,
        total_abms=len(abm_progress),
        completed_abm_count=completed_abm_count,
        failed_abm_count=failed_abm_count,
        planned_case_count=planned_case_count,
        completed_case_count=completed_case_count,
        failed_case_count=failed_case_count,
        current_case_id=current_abm_progress.running_case_id if current_abm_progress is not None else None,
        current_case_status=current_abm_progress.running_case_status if current_abm_progress is not None else None,
        current_case_detail=current_abm_progress.running_case_detail if current_abm_progress is not None else None,
        remaining_abms=list(remaining_abms),
        abms=abm_progress,
    )


def refresh_progress_abm_snapshot(
    *,
    output_root: Path,
    progress: FullCaseSuiteProgressAbm,
) -> FullCaseSuiteProgressAbm:
    """Refresh one ABM progress item from the latest nested run state."""

    abm_output_root = output_root / "abms" / progress.abm
    if not abm_output_root.exists():
        return progress
    try:
        snapshot = collect_local_qwen_monitor_snapshot(abm_output_root)
    except Exception:
        return progress
    if not snapshot.exists:
        return progress
    stable_paths = sync_stable_abm_current_view(abm_output_root=abm_output_root, run_root=snapshot.output_root)
    running_case = next((case for case in snapshot.cases if case.case_id == snapshot.running_case_id), None)
    return progress.model_copy(
        update={
            "run_root": stable_paths.run_root,
            "run_log_path": stable_paths.run_log_path,
            "report_json_path": stable_paths.report_json_path,
            "completed_case_count": snapshot.completed_cases,
            "failed_case_count": snapshot.failed_cases,
            "running_case_id": snapshot.running_case_id,
            "running_case_status": running_case.status if running_case is not None else None,
            "running_case_detail": running_case.progress_detail if running_case is not None else None,
        }
    )
