"""Live monitoring helpers for local-Qwen smoke and tuning workflows."""

from __future__ import annotations

import json
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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


@dataclass(frozen=True)
class MonitorViewState:
    """Client-side TUI state for selection and scrolling."""

    selected_index: int = 0
    scroll_offset: int = 0
    visible_rows: int = 12


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
    title = "Run monitor" if snapshot.mode == "suite" else f"Local Qwen {snapshot.mode}"
    lines = [
        f"{title}: {snapshot.output_root}",
        (
            f"{'Trials' if snapshot.mode == 'tuning' else ('ABMs' if snapshot.mode == 'suite' else 'Cases')}: "
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


def render_local_qwen_monitor_rich(snapshot: LocalQwenMonitorSnapshot) -> RenderableType:
    """Render a richer TUI-friendly dashboard for live monitoring."""
    return render_local_qwen_monitor_rich_with_state(snapshot=snapshot, state=MonitorViewState())


def render_local_qwen_monitor_rich_with_state(
    *,
    snapshot: LocalQwenMonitorSnapshot,
    state: MonitorViewState,
) -> RenderableType:
    """Render a richer TUI-friendly dashboard for live monitoring with selection state."""
    summary = Table.grid(expand=True)
    summary.add_column(justify="left", ratio=2)
    summary.add_column(justify="left", ratio=3)
    summary.add_row("Run root", str(snapshot.output_root))
    summary.add_row("Mode", snapshot.mode)
    summary.add_row(
        "Progress",
        f"{snapshot.completed_cases} completed / {snapshot.failed_cases} failed / {snapshot.total_cases} discovered",
    )
    summary.add_row("Running", snapshot.running_case_id or "-")
    summary.add_row("Terminal", "yes" if snapshot.terminal else "no")

    cases_table = Table(expand=True, box=None, show_header=True)
    cases_table.add_column("Status", style="bold")
    cases_table.add_column("Item", overflow="fold", ratio=3)
    cases_table.add_column("num_ctx", justify="right")
    cases_table.add_column("max_tok", justify="right")
    cases_table.add_column("ctx_tok", justify="right")
    cases_table.add_column("tr_tok", justify="right")
    cases_table.add_column("ctx_len", justify="right")
    cases_table.add_column("tr_len", justify="right")
    visible_cases = visible_monitor_cases(cases=snapshot.cases, state=state)
    for row_index, case in enumerate(visible_cases):
        absolute_index = state.scroll_offset + row_index
        selected = absolute_index == state.selected_index
        cases_table.add_row(
            _style_status(case.status, selected=selected),
            _selected_label(case.label or case.case_id, selected=selected),
            _fmt(case.num_ctx, 7).strip(),
            _fmt(case.max_tokens, 7).strip(),
            _fmt(case.context_total_tokens, 7).strip(),
            _fmt(case.trend_total_tokens, 6).strip(),
            _fmt(case.context_prompt_length, 7).strip(),
            _fmt(case.trend_prompt_length, 6).strip(),
            style=_row_style(case.status, selected=selected),
        )

    failures = [case for case in snapshot.cases if case.error]
    failure_panel: RenderableType
    if failures:
        failure_table = Table(expand=True, box=None, show_header=True)
        failure_table.add_column("Case", ratio=2)
        failure_table.add_column("Error", ratio=5, overflow="fold")
        for case in failures[-8:]:
            failure_table.add_row(case.label or case.case_id, case.error or "")
        failure_panel = Panel(failure_table, title="Recent Failures", border_style="red")
    else:
        failure_panel = Panel(Text("No recorded failures", style="green"), title="Recent Failures")

    title_style = "green" if snapshot.exists and not snapshot.failed_cases else "yellow"
    if snapshot.failed_cases:
        title_style = "red"
    selected_case = _selected_case(snapshot=snapshot, state=state)
    details_panel = Panel(
        _render_selected_case_details(selected_case),
        title="Selected Case",
        border_style="cyan" if selected_case is not None else "dim",
    )
    footer = Panel(
        Text("Up/Down: move  PgUp/PgDn: page  Home/End: jump  q: quit", style="dim"),
        border_style="dim",
    )
    return Group(
        Panel(summary, title="Run Monitor", border_style=title_style),
        Panel(cases_table, title="Cases" if snapshot.mode == "smoke" else "Trials"),
        details_panel,
        failure_panel,
        footer,
    )


def stream_local_qwen_monitor(
    *,
    output_root: Path,
    interval_seconds: float,
    clear_screen: bool = True,
    exit_when_terminal: bool = False,
    max_refreshes: int | None = None,
) -> None:
    """Continuously render the dashboard until interrupted or explicitly told to exit."""
    console = Console()
    refresh_count = 0
    view_state = MonitorViewState()
    snapshot = collect_local_qwen_monitor_snapshot(output_root)
    last_snapshot_at = time.monotonic()
    with _MonitorKeyboardReader() as keyboard_reader:
        with Live(console=console, refresh_per_second=4, screen=clear_screen, transient=False) as live:
            while True:
                key = keyboard_reader()
                if key == "quit":
                    return
                if key is not None:
                    view_state = apply_monitor_keypress(view_state, key, total_cases=len(snapshot.cases))
                now = time.monotonic()
                if (
                    next_snapshot_due(
                        last_snapshot_at=last_snapshot_at,
                        now=now,
                        interval_seconds=interval_seconds,
                    )
                    == 0.0
                ):
                    snapshot = collect_local_qwen_monitor_snapshot(output_root)
                    last_snapshot_at = now
                    refresh_count += 1
                    view_state = view_state_for_snapshot(view_state=view_state, snapshot=snapshot)
                live.update(render_local_qwen_monitor_rich_with_state(snapshot=snapshot, state=view_state))
                if max_refreshes is not None and refresh_count >= max_refreshes:
                    return
                if snapshot.terminal and exit_when_terminal:
                    return
                sleep_for = min(
                    next_snapshot_due(
                        last_snapshot_at=last_snapshot_at,
                        now=time.monotonic(),
                        interval_seconds=interval_seconds,
                    ),
                    0.05,
                )
                time.sleep(sleep_for)


def visible_monitor_cases(
    *,
    cases: tuple[LocalQwenCaseSnapshot, ...],
    state: MonitorViewState,
) -> tuple[LocalQwenCaseSnapshot, ...]:
    """Return the visible case window for the current scroll offset."""
    start = max(state.scroll_offset, 0)
    end = max(start + state.visible_rows, start)
    return cases[start:end]


def next_snapshot_due(*, last_snapshot_at: float, now: float, interval_seconds: float) -> float:
    """Return seconds remaining until the next filesystem snapshot refresh is due."""
    if interval_seconds <= 0:
        return 0.0
    elapsed = now - last_snapshot_at
    if elapsed >= interval_seconds:
        return 0.0
    return interval_seconds - elapsed


def view_state_for_snapshot(
    *,
    view_state: MonitorViewState,
    snapshot: LocalQwenMonitorSnapshot,
) -> MonitorViewState:
    """Clamp selection state to the current snapshot size."""
    if not snapshot.cases:
        return MonitorViewState(selected_index=0, scroll_offset=0, visible_rows=view_state.visible_rows)
    max_index = len(snapshot.cases) - 1
    selected_index = min(max(view_state.selected_index, 0), max_index)
    max_scroll = max(len(snapshot.cases) - view_state.visible_rows, 0)
    scroll_offset = min(max(view_state.scroll_offset, 0), max_scroll)
    if selected_index < scroll_offset:
        scroll_offset = selected_index
    elif selected_index >= scroll_offset + view_state.visible_rows:
        scroll_offset = max(selected_index - view_state.visible_rows + 1, 0)
    return MonitorViewState(
        selected_index=selected_index,
        scroll_offset=scroll_offset,
        visible_rows=view_state.visible_rows,
    )


def apply_monitor_keypress(state: MonitorViewState, key: str, *, total_cases: int) -> MonitorViewState:
    """Update the monitor view state for one navigation keypress."""
    if total_cases <= 0:
        return state
    selected_index = state.selected_index
    if key == "down":
        selected_index = min(selected_index + 1, total_cases - 1)
    elif key == "up":
        selected_index = max(selected_index - 1, 0)
    elif key == "page_down":
        selected_index = min(selected_index + state.visible_rows, total_cases - 1)
    elif key == "page_up":
        selected_index = max(selected_index - state.visible_rows, 0)
    elif key == "home":
        selected_index = 0
    elif key == "end":
        selected_index = total_cases - 1
    max_scroll = max(total_cases - state.visible_rows, 0)
    scroll_offset = state.scroll_offset
    if selected_index < scroll_offset:
        scroll_offset = selected_index
    elif selected_index >= scroll_offset + state.visible_rows:
        scroll_offset = selected_index - state.visible_rows + 1
    scroll_offset = min(max(scroll_offset, 0), max_scroll)
    return MonitorViewState(
        selected_index=selected_index,
        scroll_offset=scroll_offset,
        visible_rows=state.visible_rows,
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


def _collect_trial_snapshot(trial_dir: Path) -> LocalQwenCaseSnapshot:
    cases_root = trial_dir / "cases"
    case_snapshots = (
        [
            _collect_case_snapshot(case_dir)
            for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir())
        ]
        if cases_root.exists()
        else []
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


def _fmt(value: int | None, width: int) -> str:
    return f"{('-' if value is None else value):>{width}}"


def _style_status(status: str, *, selected: bool = False) -> Text:
    style = {
        "completed": "green",
        "failed": "red",
        "running": "yellow",
        "pending": "dim",
    }.get(status, "white")
    if selected:
        style = f"bold {style}"
    return Text(status, style=style)


def _row_style(status: str, *, selected: bool = False) -> str:
    style = {
        "completed": "green",
        "failed": "red",
        "running": "yellow",
        "pending": "dim",
    }.get(status, "")
    if selected:
        return f"reverse {style}".strip()
    return style


def _selected_label(label: str, *, selected: bool) -> str:
    return f"> {label}" if selected else f"  {label}"


def _selected_case(
    *,
    snapshot: LocalQwenMonitorSnapshot,
    state: MonitorViewState,
) -> LocalQwenCaseSnapshot | None:
    if not snapshot.cases:
        return None
    index = min(max(state.selected_index, 0), len(snapshot.cases) - 1)
    return snapshot.cases[index]


def _render_selected_case_details(case: LocalQwenCaseSnapshot | None) -> RenderableType:
    if case is None:
        return Text("No case selected", style="dim")
    details = Table.grid(expand=True)
    details.add_column(justify="left", ratio=2)
    details.add_column(justify="left", ratio=5)
    details.add_row("Case", case.label or case.case_id)
    details.add_row("Status", case.status)
    details.add_row("num_ctx", "-" if case.num_ctx is None else str(case.num_ctx))
    details.add_row("max_tokens", "-" if case.max_tokens is None else str(case.max_tokens))
    details.add_row("context tokens", "-" if case.context_total_tokens is None else str(case.context_total_tokens))
    details.add_row("trend tokens", "-" if case.trend_total_tokens is None else str(case.trend_total_tokens))
    details.add_row(
        "context length",
        "-" if case.context_prompt_length is None else str(case.context_prompt_length),
    )
    details.add_row(
        "trend length",
        "-" if case.trend_prompt_length is None else str(case.trend_prompt_length),
    )
    details.add_row("error", case.error or "-")
    return details


class _MonitorKeyboardReader:
    """POSIX keyboard reader for non-blocking arrow-key monitor navigation."""

    def __enter__(self) -> Any:
        if not sys.stdin.isatty():
            return lambda: None
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self._read_key

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if hasattr(self, "_old_settings"):
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def _read_key(self) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        first = os.read(self._fd, 1)
        if not first:
            return None
        if first == b"q":
            return "quit"
        if first == b"\x1b":
            remainder = os.read(self._fd, 2)
            sequence = first + remainder
            return {
                b"\x1b[A": "up",
                b"\x1b[B": "down",
                b"\x1b[H": "home",
                b"\x1b[F": "end",
            }.get(sequence)
        if first == b"j":
            return "down"
        if first == b"k":
            return "up"
        return None


def _extract_int_from_name(value: str, prefix: str) -> int | None:
    if not value.startswith(prefix):
        return None
    suffix = value.removeprefix(prefix)
    return int(suffix) if suffix.isdigit() else None


def _max_or_none(values: list[int | None]) -> int | None:
    concrete = [value for value in values if value is not None]
    return max(concrete) if concrete else None
