"""Live monitoring helpers for local-Qwen smoke and tuning workflows."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distill_abm.run_viewer import resolve_run_root


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

    case_snapshots: list[LocalQwenCaseSnapshot] = []
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        case_snapshots.append(_collect_case_snapshot(case_dir))

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
        cases=tuple(case_snapshots),
    )


def render_local_qwen_monitor(snapshot: LocalQwenMonitorSnapshot) -> str:
    """Render a compact text dashboard for the smoke or tuning status."""
    lines = [
        f"Local Qwen {snapshot.mode}: {snapshot.output_root}",
        (
            f"{'Trials' if snapshot.mode == 'tuning' else 'Cases'}: "
            f"{snapshot.completed_cases} completed / "
            f"{snapshot.failed_cases} failed / "
            f"{snapshot.total_cases} discovered"
        ),
        f"Running: {snapshot.running_case_id or '-'}",
        "",
        "Status  Item                                   num_ctx  max_tok  ctx_tok  tr_tok  ctx_len  tr_len",
        "------  -------------------------------------  -------  -------  -------  ------  -------  ------",
    ]
    for case in snapshot.cases:
        lines.append(
            "  ".join(
                [
                    f"{case.status:<6}",
                    f"{(case.label or case.case_id):<37}",
                    _fmt(case.num_ctx, 7),
                    _fmt(case.max_tokens, 7),
                    _fmt(case.context_total_tokens, 7),
                    _fmt(case.trend_total_tokens, 6),
                    _fmt(case.context_prompt_length, 7),
                    _fmt(case.trend_prompt_length, 6),
                ]
            )
        )
        if case.error:
            lines.append(f"        error: {case.error}")
    return "\n".join(lines)


def stream_local_qwen_monitor(
    *,
    output_root: Path,
    interval_seconds: float,
    clear_screen: bool = True,
) -> None:
    """Continuously render the dashboard until the smoke reaches a terminal state."""
    while True:
        snapshot = collect_local_qwen_monitor_snapshot(output_root)
        rendered = render_local_qwen_monitor(snapshot)
        if clear_screen:
            print("\033[2J\033[H", end="")
        print(rendered, flush=True)
        if snapshot.terminal:
            return
        time.sleep(interval_seconds)


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
    num_ctx = _extract_num_ctx(request_for_limits)
    max_tokens = _extract_max_tokens(request_for_limits)
    return LocalQwenCaseSnapshot(
        case_id=case_dir.name,
        status=status,
        label=case_dir.name,
        num_ctx=num_ctx,
        max_tokens=max_tokens,
        context_prompt_length=_extract_prompt_length(context_request),
        trend_prompt_length=_extract_prompt_length(trend_request),
        context_total_tokens=_extract_total_tokens(context_trace),
        trend_total_tokens=_extract_total_tokens(trend_trace),
        error=error_text,
    )


def _collect_full_case_snapshot(case_dir: Path) -> LocalQwenCaseSnapshot:
    context_dir = case_dir / "02_context"
    trends_root = case_dir / "03_trends"
    trend_dirs = sorted(path for path in trends_root.iterdir() if path.is_dir())
    context_request = _read_json(context_dir / "context_request.json")
    context_trace = _read_json(context_dir / "context_trace.json")
    context_error = _read_text(context_dir / "error.txt")
    trend_requests = [_read_json(path / "trend_request.json") for path in trend_dirs]
    trend_traces = [_read_json(path / "trend_trace.json") for path in trend_dirs]
    trend_errors = {path.name: _read_text(path / "error.txt") for path in trend_dirs}
    failed_trends = [name for name, error in trend_errors.items() if error]
    completed_trends = [trace for trace in trend_traces if trace is not None]
    started_trends = [
        directory
        for directory, request, trace in zip(trend_dirs, trend_requests, trend_traces, strict=False)
        if request is not None or trace is not None
    ]
    if context_error or failed_trends:
        status = "failed"
    elif context_trace is not None and len(completed_trends) == len(trend_dirs) and trend_dirs:
        status = "completed"
    elif context_request is not None or started_trends:
        status = "running"
    else:
        status = "pending"

    representative_request = (
        next((request for request in trend_requests if request is not None), None) or context_request
    )
    error = context_error
    if not error and failed_trends:
        error = f"failed_trends={','.join(failed_trends)}"
    return LocalQwenCaseSnapshot(
        case_id=case_dir.name,
        status=status,
        label=case_dir.name,
        num_ctx=_extract_num_ctx(representative_request),
        max_tokens=_extract_max_tokens(representative_request),
        context_prompt_length=_extract_prompt_length(context_request),
        trend_prompt_length=_max_or_none([_extract_prompt_length(request) for request in trend_requests]),
        context_total_tokens=_extract_total_tokens(context_trace),
        trend_total_tokens=_max_or_none([_extract_total_tokens(trace) for trace in trend_traces]),
        error=error,
    )


def _collect_tuning_snapshot(output_root: Path, trials_root: Path) -> LocalQwenMonitorSnapshot:
    trial_snapshots: list[LocalQwenCaseSnapshot] = []
    for trial_dir in sorted(trials_root.glob("*/*/*")):
        if not trial_dir.is_dir() or not trial_dir.name.startswith("max_tokens_"):
            continue
        trial_snapshots.append(_collect_trial_snapshot(trial_dir))

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
        cases=tuple(trial_snapshots),
    )


def _collect_trial_snapshot(trial_dir: Path) -> LocalQwenCaseSnapshot:
    cases_root = trial_dir / "cases"
    case_snapshots = [
        _collect_case_snapshot(case_dir)
        for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir())
    ] if cases_root.exists() else []

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

    label = f"{trial_dir.parent.name}/{trial_dir.name}"
    fallback_num_ctx = _extract_int_from_name(trial_dir.parent.name, "num_ctx_")
    fallback_max_tokens = _extract_int_from_name(trial_dir.name, "max_tokens_")
    return LocalQwenCaseSnapshot(
        case_id=str(trial_dir.relative_to(trial_dir.parents[2])),
        status=status,
        label=label,
        num_ctx=representative.num_ctx if representative and representative.num_ctx is not None else fallback_num_ctx,
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


def _fmt(value: int | None, width: int) -> str:
    return f"{('-' if value is None else value):>{width}}"


def _extract_int_from_name(value: str, prefix: str) -> int | None:
    if not value.startswith(prefix):
        return None
    suffix = value.removeprefix(prefix)
    return int(suffix) if suffix.isdigit() else None


def _max_or_none(values: list[int | None]) -> int | None:
    concrete = [value for value in values if value is not None]
    return max(concrete) if concrete else None
