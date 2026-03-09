"""Shared filename, path, and run-lock contracts for run-oriented smoke workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

CASE_SUMMARY_FILENAME = "00_case_summary.json"
VALIDATION_STATE_FILENAME = "validation_state.json"
RUN_LOG_FILENAME = "run.log.jsonl"
LATEST_RUN_POINTER_FILENAME = "latest_run.txt"
LATEST_REPORT_POINTER_FILENAME = "latest_report_path.txt"
VIEWER_HTML_FILENAME = "review.html"
ACTIVE_RUN_LOCK_FILENAME = ".active_run_lock.json"
SAMPLED_SMOKE_REPORT_FILENAME = "smoke_local_qwen_report.json"
FULL_CASE_MATRIX_REPORT_FILENAME = "smoke_full_case_matrix_report.json"
INGEST_SMOKE_REPORT_FILENAME = "ingest_smoke_report.json"
VIZ_SMOKE_REPORT_FILENAME = "viz_smoke_report.json"
DOE_SMOKE_REPORT_FILENAME = "doe_smoke_report.json"


class ActiveRunLock(BaseModel):
    """Persistent lock metadata for one active smoke run."""

    pid: int
    run_id: str
    run_root: Path


def runs_root_path(output_root: Path) -> Path:
    return output_root / "runs"


def latest_run_pointer_path(output_root: Path) -> Path:
    return output_root / LATEST_RUN_POINTER_FILENAME


def latest_report_pointer_path(output_root: Path) -> Path:
    return output_root / LATEST_REPORT_POINTER_FILENAME


def active_run_lock_path(output_root: Path) -> Path:
    return output_root / ACTIVE_RUN_LOCK_FILENAME


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


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_active_run_lock(output_root: Path) -> ActiveRunLock | None:
    """Read the current active-run lock if it exists and is well formed."""
    lock_path = active_run_lock_path(output_root)
    if not lock_path.exists():
        return None
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return ActiveRunLock.model_validate(payload)
    except Exception:
        return None


def acquire_active_run_lock(*, output_root: Path, run_id: str, run_root: Path) -> ActiveRunLock:
    """Acquire the active-run lock for an output root, rejecting concurrent runners."""
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = active_run_lock_path(output_root)
    existing = read_active_run_lock(output_root)
    if existing is not None and _process_exists(existing.pid):
        raise RuntimeError(
            f"another run is already active for {output_root}: pid={existing.pid}, "
            f"run_id={existing.run_id}, run_root={existing.run_root}"
        )
    if existing is not None and not _process_exists(existing.pid):
        try:
            lock_path.unlink()
        except OSError:
            pass
    lock = ActiveRunLock(pid=os.getpid(), run_id=run_id, run_root=run_root)
    lock_path.write_text(json.dumps(lock.model_dump(mode="json"), sort_keys=True), encoding="utf-8")
    return lock


def release_active_run_lock(*, output_root: Path, run_id: str) -> None:
    """Release the active-run lock if it belongs to the current run."""
    lock_path = active_run_lock_path(output_root)
    existing = read_active_run_lock(output_root)
    if existing is None or existing.run_id != run_id:
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


def resolve_run_root(path: Path) -> Path:
    """Resolve either a concrete run directory or a root containing a latest-run pointer."""
    latest_run_path = latest_run_pointer_path(path)
    if latest_run_path.exists():
        latest_text = latest_run_path.read_text(encoding="utf-8").strip()
        if latest_text:
            return Path(latest_text)
    return path
