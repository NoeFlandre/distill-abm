"""Filesystem snapshot collection for the local-Qwen monitor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distill_abm.pipeline.run_artifact_contracts import resolve_run_root


@dataclass(frozen=True)
class LocalQwenCaseSnapshot:
    """One item status snapshot for the live monitor."""

    case_id: str
    status: str
    label: str | None
    num_ctx: int | None
    max_tokens: int | None
    context_prompt_length: int | None
    trend_prompt_length: int | None
    context_total_tokens: int | None
    trend_total_tokens: int | None
    error: str | None
    progress_detail: str | None = None
    completed_steps: int | None = None
    total_steps: int | None = None


@dataclass(frozen=True)
class LocalQwenMonitorSnapshot:
    """Top-level local-Qwen monitor snapshot."""

    output_root: Path
    exists: bool
    mode: str
    total_cases: int
    completed_cases: int
    failed_cases: int
    running_case_id: str | None
    cases: tuple[LocalQwenCaseSnapshot, ...]

    @property
    def terminal(self) -> bool:
        return self.exists and self.total_cases > 0 and self.completed_cases + self.failed_cases >= self.total_cases


def collect_local_qwen_monitor_snapshot(output_root: Path) -> LocalQwenMonitorSnapshot:
    """Collect the current smoke or tuning progress from the output directory."""
    suite_progress = _read_json(output_root / "suite_progress.json")
    if suite_progress is not None:
        return _collect_suite_snapshot(output_root, suite_progress)
    resolved_root = resolve_run_root(output_root)
    trials_root = resolved_root / "trials"
    if trials_root.exists():
        return _collect_tuning_snapshot(resolved_root, trials_root)

    cases_root = resolved_root / "cases"
    if not cases_root.exists():
        return LocalQwenMonitorSnapshot(
            output_root=resolved_root,
            exists=False,
            mode="smoke",
            total_cases=0,
            completed_cases=0,
            failed_cases=0,
            running_case_id=None,
            cases=(),
        )

    case_snapshots = tuple(
        _collect_case_snapshot(case_dir)
        for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir())
    )
    running_case_id = next((case.case_id for case in case_snapshots if case.status.startswith("running")), None)
    completed_cases = sum(1 for case in case_snapshots if case.status == "completed")
    failed_cases = sum(1 for case in case_snapshots if case.status == "failed")
    return LocalQwenMonitorSnapshot(
        output_root=resolved_root,
        exists=True,
        mode="smoke",
        total_cases=len(case_snapshots),
        completed_cases=completed_cases,
        failed_cases=failed_cases,
        running_case_id=running_case_id,
        cases=case_snapshots,
    )


def _collect_case_snapshot(case_dir: Path) -> LocalQwenCaseSnapshot:
    if (case_dir / "02_requests").exists() or (case_dir / "03_outputs").exists():
        return _collect_sample_case_snapshot(case_dir)
    if (case_dir / "03_trends").exists():
        return _collect_full_case_snapshot(case_dir)
    return LocalQwenCaseSnapshot(
        case_id=case_dir.name,
        status="pending",
        label=case_dir.name,
        num_ctx=None,
        max_tokens=None,
        context_prompt_length=None,
        trend_prompt_length=None,
        context_total_tokens=None,
        trend_total_tokens=None,
        error=None,
    )


def _collect_sample_case_snapshot(case_dir: Path) -> LocalQwenCaseSnapshot:
    requests_dir = case_dir / "02_requests"
    outputs_dir = case_dir / "03_outputs"
    context_request = _read_json(requests_dir / "context_request.json")
    trend_request = _read_json(requests_dir / "trend_request.json")
    context_trace = _read_json(outputs_dir / "context_trace.json")
    trend_trace = _read_json(outputs_dir / "trend_trace.json")
    error_text = _read_text(outputs_dir / "error.txt")

    if error_text is not None:
        status = "failed"
    elif trend_trace is not None:
        status = "completed"
    elif trend_request is not None or context_trace is not None:
        status = "running"
    elif context_request is not None:
        status = "running"
    else:
        status = "pending"

    request_for_limits = trend_request or context_request
    return LocalQwenCaseSnapshot(
        case_id=case_dir.name,
        status=status,
        label=case_dir.name,
        num_ctx=_extract_num_ctx(request_for_limits),
        max_tokens=_extract_max_tokens(request_for_limits),
        context_prompt_length=_extract_prompt_length(context_request),
        trend_prompt_length=_extract_prompt_length(trend_request),
        context_total_tokens=_extract_total_tokens(context_trace),
        trend_total_tokens=_extract_total_tokens(trend_trace),
        error=error_text,
        progress_detail="trend" if trend_request is not None else ("context" if context_request is not None else None),
        completed_steps=1 if trend_trace is not None else (0 if context_trace is None else 1),
        total_steps=2,
    )


def _collect_full_case_snapshot(case_dir: Path) -> LocalQwenCaseSnapshot:
    context_dir = case_dir / "02_context"
    trends_root = case_dir / "03_trends"
    trend_dirs = sorted(path for path in trends_root.iterdir() if path.is_dir())
    validation_state = _read_json(case_dir / "validation_state.json")
    context_request = _read_json(context_dir / "context_request.json")
    context_trace = _read_json(context_dir / "context_trace.json")
    context_error = _read_text(context_dir / "error.txt")
    trend_requests = [_read_json(path / "trend_request.json") for path in trend_dirs]
    trend_traces = [_read_json(path / "trend_trace.json") for path in trend_dirs]
    representative_trend_trace = next((trace for trace in trend_traces if trace is not None), None)
    failed_trends = _failed_trends_from_validation_state(validation_state, trend_dirs)
    accepted_trends = _accepted_trend_count_from_validation_state(validation_state)
    completed_trends = [trace for trace in trend_traces if trace is not None]
    completed_trend_count = max(accepted_trends, len(completed_trends))
    any_started_trend = any(
        (trend_dir / "trend_request.json").exists() or (trend_dir / "trend_trace.json").exists()
        for trend_dir in trend_dirs
    )
    context_accepted = _validation_context_status(validation_state) == "accepted"
    if context_error or failed_trends:
        status = "failed"
    elif (context_accepted and accepted_trends == len(trend_dirs) and trend_dirs) or (
        context_trace is not None and len(completed_trends) == len(trend_dirs) and trend_dirs
    ):
        status = "completed"
    elif context_request is not None or any_started_trend:
        status = "running"
    else:
        status = "pending"

    representative_request = (
        next((request for request in trend_requests if request is not None), None) or context_request
    )
    error = context_error
    if not error and failed_trends:
        error = f"failed_trends={','.join(failed_trends)}"
    progress_detail: str | None = None
    if status == "running":
        if context_request is not None and context_trace is None and not any_started_trend:
            progress_detail = "context"
        else:
            active_trend = next(
                (
                    trend_dir.name
                    for trend_dir, trend_trace in zip(trend_dirs, trend_traces, strict=False)
                    if (trend_dir / "trend_request.json").exists() and trend_trace is None
                ),
                None,
            )
            progress_detail = f"trend {active_trend}" if active_trend is not None else "trend"
    elif status == "completed":
        progress_detail = "done"
    elif status == "failed" and failed_trends:
        progress_detail = f"failed trend {failed_trends[0]}"
    return LocalQwenCaseSnapshot(
        case_id=case_dir.name,
        status=status,
        label=case_dir.name,
        num_ctx=_extract_num_ctx(representative_request),
        max_tokens=_extract_max_tokens(representative_request),
        context_prompt_length=_extract_prompt_length(context_request),
        trend_prompt_length=_max_or_none([_extract_prompt_length(request) for request in trend_requests]),
        context_total_tokens=_extract_total_tokens(context_trace),
        trend_total_tokens=_extract_total_tokens(representative_trend_trace),
        error=error,
        progress_detail=progress_detail,
        completed_steps=(1 if context_accepted or context_trace is not None else 0) + completed_trend_count,
        total_steps=1 + len(trend_dirs),
    )


def _collect_tuning_snapshot(output_root: Path, trials_root: Path) -> LocalQwenMonitorSnapshot:
    trial_snapshots = tuple(
        _collect_trial_snapshot(trial_dir)
        for trial_dir in sorted(trials_root.glob("*/*/*"))
        if trial_dir.is_dir() and trial_dir.name.startswith("max_tokens_")
    )
    running_case_id = next((trial.case_id for trial in trial_snapshots if trial.status.startswith("running")), None)
    completed_cases = sum(1 for trial in trial_snapshots if trial.status == "completed")
    failed_cases = sum(1 for trial in trial_snapshots if trial.status == "failed")
    return LocalQwenMonitorSnapshot(
        output_root=output_root,
        exists=True,
        mode="tuning",
        total_cases=len(trial_snapshots),
        completed_cases=completed_cases,
        failed_cases=failed_cases,
        running_case_id=running_case_id,
        cases=trial_snapshots,
    )


def _collect_suite_snapshot(output_root: Path, payload: dict[str, Any]) -> LocalQwenMonitorSnapshot:
    abm_payloads = payload.get("abms")
    if not isinstance(abm_payloads, list):
        abm_payloads = []
    case_snapshots: list[LocalQwenCaseSnapshot] = []
    for item in abm_payloads:
        if not isinstance(item, dict):
            continue
        label = item.get("abm")
        status = item.get("status")
        if not isinstance(label, str) or not isinstance(status, str):
            continue
        error = item.get("last_error")
        abm_output_root = output_root / "abms" / label
        nested_snapshot = _try_collect_nested_suite_abm_snapshot(abm_output_root)
        if nested_snapshot is None:
            case_snapshots.append(
                LocalQwenCaseSnapshot(
                    case_id=label,
                    status=status,
                    label=label,
                    num_ctx=None,
                    max_tokens=None,
                    context_prompt_length=None,
                    trend_prompt_length=None,
                    context_total_tokens=None,
                    trend_total_tokens=None,
                    error=error if isinstance(error, str) else None,
                )
            )
            continue
        running_case = next(
            (case for case in nested_snapshot.cases if case.case_id == nested_snapshot.running_case_id),
            None,
        )
        case_snapshots.append(
            LocalQwenCaseSnapshot(
                case_id=label,
                status=(
                    nested_snapshot.cases[0].status
                    if len(nested_snapshot.cases) == 1
                    else (
                        "failed"
                        if nested_snapshot.failed_cases
                        else (
                            "completed"
                            if nested_snapshot.completed_cases == nested_snapshot.total_cases
                            else "running"
                        )
                    )
                ),
                label=label,
                num_ctx=running_case.num_ctx if running_case is not None else None,
                max_tokens=running_case.max_tokens if running_case is not None else None,
                context_prompt_length=running_case.context_prompt_length if running_case is not None else None,
                trend_prompt_length=running_case.trend_prompt_length if running_case is not None else None,
                context_total_tokens=running_case.context_total_tokens if running_case is not None else None,
                trend_total_tokens=running_case.trend_total_tokens if running_case is not None else None,
                error=(
                    (error if isinstance(error, str) else None)
                    or (running_case.error if running_case is not None else None)
                ),
                progress_detail=(
                    f"{nested_snapshot.running_case_id}: {running_case.progress_detail}"
                    if running_case is not None and nested_snapshot.running_case_id is not None
                    else None
                ),
                completed_steps=nested_snapshot.completed_cases,
                total_steps=nested_snapshot.total_cases,
            )
        )
    total_abms = payload.get("total_abms")
    completed_abm_count = payload.get("completed_abm_count")
    failed_abm_count = payload.get("failed_abm_count")
    running_case_id = payload.get("current_abm")
    return LocalQwenMonitorSnapshot(
        output_root=output_root,
        exists=True,
        mode="suite",
        total_cases=total_abms if isinstance(total_abms, int) else len(case_snapshots),
        completed_cases=completed_abm_count if isinstance(completed_abm_count, int) else 0,
        failed_cases=failed_abm_count if isinstance(failed_abm_count, int) else 0,
        running_case_id=running_case_id if isinstance(running_case_id, str) else None,
        cases=tuple(case_snapshots),
    )


def _try_collect_nested_suite_abm_snapshot(output_root: Path) -> LocalQwenMonitorSnapshot | None:
    if not output_root.exists():
        return None
    try:
        return collect_local_qwen_monitor_snapshot(output_root)
    except Exception:
        return None


def _collect_trial_snapshot(trial_dir: Path) -> LocalQwenCaseSnapshot:
    cases_root = trial_dir / "cases"
    case_snapshots = (
        tuple(
            _collect_case_snapshot(case_dir)
            for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir())
        )
        if cases_root.exists()
        else ()
    )

    if any(case.status == "failed" for case in case_snapshots):
        status = "failed"
    elif case_snapshots and all(case.status == "completed" for case in case_snapshots):
        status = "completed"
    elif case_snapshots:
        status = "running"
    else:
        status = "pending"

    representative = next((case for case in case_snapshots if case.status != "pending"), None)
    if representative is None and case_snapshots:
        representative = case_snapshots[0]

    failed_case_ids = [case.case_id for case in case_snapshots if case.status == "failed"]
    running_case_ids = [case.case_id for case in case_snapshots if case.status == "running"]
    error = None
    if failed_case_ids:
        error = f"failed_cases={','.join(failed_case_ids)}"
    elif running_case_ids:
        error = f"running_cases={','.join(running_case_ids)}"

    fallback_num_ctx = _extract_int_from_name(trial_dir.parent.name, "num_ctx_")
    fallback_max_tokens = _extract_int_from_name(trial_dir.name, "max_tokens_")
    return LocalQwenCaseSnapshot(
        case_id=str(trial_dir.relative_to(trial_dir.parents[2])),
        status=status,
        label=f"{trial_dir.parent.name}/{trial_dir.name}",
        num_ctx=(
            representative.num_ctx if representative and representative.num_ctx is not None else fallback_num_ctx
        ),
        max_tokens=(
            representative.max_tokens
            if representative and representative.max_tokens is not None
            else fallback_max_tokens
        ),
        context_prompt_length=representative.context_prompt_length if representative else None,
        trend_prompt_length=representative.trend_prompt_length if representative else None,
        context_total_tokens=_max_or_none([case.context_total_tokens for case in case_snapshots]),
        trend_total_tokens=_max_or_none([case.trend_total_tokens for case in case_snapshots]),
        error=error,
    )


def _extract_num_ctx(request: dict[str, Any] | None) -> int | None:
    if not isinstance(request, dict):
        return None
    metadata = request.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("ollama_num_ctx")
    return value if isinstance(value, int) else None


def _extract_max_tokens(request: dict[str, Any] | None) -> int | None:
    if not isinstance(request, dict):
        return None
    value = request.get("max_tokens")
    return value if isinstance(value, int) else None


def _extract_prompt_length(request: dict[str, Any] | None) -> int | None:
    if not isinstance(request, dict):
        return None
    value = request.get("prompt_length")
    return value if isinstance(value, int) else None


def _extract_total_tokens(trace: dict[str, Any] | None) -> int | None:
    if not isinstance(trace, dict):
        return None
    response = trace.get("response")
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("total_tokens")
    return value if isinstance(value, int) else None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _validation_context_status(validation_state: dict[str, Any] | None) -> str | None:
    if not isinstance(validation_state, dict):
        return None
    context = validation_state.get("context")
    if not isinstance(context, dict):
        return None
    status = context.get("status")
    return status if isinstance(status, str) else None


def _accepted_trend_count_from_validation_state(validation_state: dict[str, Any] | None) -> int:
    if not isinstance(validation_state, dict):
        return 0
    trends = validation_state.get("trends")
    if not isinstance(trends, dict):
        return 0
    return sum(1 for payload in trends.values() if isinstance(payload, dict) and payload.get("status") == "accepted")


def _failed_trends_from_validation_state(validation_state: dict[str, Any] | None, trend_dirs: list[Path]) -> list[str]:
    if not isinstance(validation_state, dict):
        return [path.name for path in trend_dirs if _read_text(path / "error.txt")]
    trends = validation_state.get("trends")
    if not isinstance(trends, dict):
        return [path.name for path in trend_dirs if _read_text(path / "error.txt")]
    failed: list[str] = []
    for path in trend_dirs:
        plot_key = path.name.removeprefix("plot_").lstrip("0") or "0"
        payload = trends.get(plot_key)
        if isinstance(payload, dict) and payload.get("status") == "retry":
            failed.append(path.name)
            continue
        if _read_text(path / "error.txt"):
            failed.append(path.name)
    return failed


def _extract_int_from_name(value: str, prefix: str) -> int | None:
    if not value.startswith(prefix):
        return None
    suffix = value.removeprefix(prefix)
    return int(suffix) if suffix.isdigit() else None


def _max_or_none(values: list[int | None]) -> int | None:
    concrete = [value for value in values if value is not None]
    return max(concrete) if concrete else None
