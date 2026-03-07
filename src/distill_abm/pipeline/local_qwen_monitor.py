"""Live monitoring helpers for the sampled local-Qwen smoke."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LocalQwenCaseSnapshot:
    """One case status snapshot for the live monitor."""

    case_id: str
    status: str
    num_ctx: int | None
    max_tokens: int | None
    context_prompt_length: int | None
    trend_prompt_length: int | None
    context_total_tokens: int | None
    trend_total_tokens: int | None
    error: str | None


@dataclass(frozen=True)
class LocalQwenMonitorSnapshot:
    """Top-level local-Qwen smoke monitor snapshot."""

    output_root: Path
    exists: bool
    total_cases: int
    completed_cases: int
    failed_cases: int
    running_case_id: str | None
    cases: tuple[LocalQwenCaseSnapshot, ...]

    @property
    def terminal(self) -> bool:
        return self.exists and self.total_cases > 0 and self.completed_cases + self.failed_cases >= self.total_cases


def collect_local_qwen_monitor_snapshot(output_root: Path) -> LocalQwenMonitorSnapshot:
    """Collect the current smoke progress from the output directory."""
    cases_root = output_root / "cases"
    if not cases_root.exists():
        return LocalQwenMonitorSnapshot(
            output_root=output_root,
            exists=False,
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
        output_root=output_root,
        exists=True,
        total_cases=len(case_snapshots),
        completed_cases=completed_cases,
        failed_cases=failed_cases,
        running_case_id=running_case_id,
        cases=tuple(case_snapshots),
    )


def render_local_qwen_monitor(snapshot: LocalQwenMonitorSnapshot) -> str:
    """Render a compact text dashboard for the smoke status."""
    lines = [
        f"Local Qwen smoke: {snapshot.output_root}",
        (
            "Cases: "
            f"{snapshot.completed_cases} completed / "
            f"{snapshot.failed_cases} failed / "
            f"{snapshot.total_cases} discovered"
        ),
        f"Running: {snapshot.running_case_id or '-'}",
        "",
        "Status  Case                                   num_ctx  max_tok  ctx_tok  tr_tok  ctx_len  tr_len",
        "------  -------------------------------------  -------  -------  -------  ------  -------  ------",
    ]
    for case in snapshot.cases:
        lines.append(
            "  ".join(
                [
                    f"{case.status:<6}",
                    f"{case.case_id:<37}",
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
        num_ctx=num_ctx,
        max_tokens=max_tokens,
        context_prompt_length=_extract_prompt_length(context_request),
        trend_prompt_length=_extract_prompt_length(trend_request),
        context_total_tokens=_extract_total_tokens(context_trace),
        trend_total_tokens=_extract_total_tokens(trend_trace),
        error=error_text,
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
