"""Live monitoring helpers for local-Qwen smoke and tuning workflows."""

from __future__ import annotations

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

from distill_abm.pipeline.local_qwen_monitor_snapshots import (
    LocalQwenCaseSnapshot,
    LocalQwenMonitorSnapshot,
    collect_local_qwen_monitor_snapshot,
)

__all__ = [
    "LocalQwenCaseSnapshot",
    "LocalQwenMonitorSnapshot",
    "MonitorViewState",
    "apply_monitor_keypress",
    "collect_local_qwen_monitor_snapshot",
    "next_snapshot_due",
    "render_local_qwen_monitor",
    "render_local_qwen_monitor_rich",
    "render_local_qwen_monitor_rich_with_state",
    "stream_local_qwen_monitor",
    "view_state_for_snapshot",
    "visible_monitor_cases",
]


@dataclass(frozen=True)
class MonitorViewState:
    """Client-side TUI state for selection and scrolling."""

    selected_index: int = 0
    scroll_offset: int = 0
    visible_rows: int = 12


def render_local_qwen_monitor(snapshot: LocalQwenMonitorSnapshot) -> str:
    """Render a compact text dashboard for the smoke or tuning status."""
    title = "Run monitor" if snapshot.mode == "suite" else f"Local Qwen {snapshot.mode}"
    progress_label = "ABMs" if snapshot.mode == "suite" else ("Trials" if snapshot.mode == "tuning" else "Cases")
    running_case = next((case for case in snapshot.cases if case.case_id == snapshot.running_case_id), None)
    lines = [
        f"{title}: {snapshot.output_root}",
        (
            f"{progress_label}: {snapshot.completed_cases} completed / "
            f"{snapshot.failed_cases} failed / {snapshot.total_cases} discovered"
        ),
        f"Running: {snapshot.running_case_id or '-'}",
        f"Current work: {running_case.progress_detail if running_case and running_case.progress_detail else '-'}",
        "",
        "Status  Item                                   Work / progress",
        "------  -------------------------------------  --------------------------------------------",
    ]
    for case in snapshot.cases:
        progress_text = case.progress_detail or "-"
        if case.completed_steps is not None and case.total_steps is not None:
            progress_text = f"{progress_text} [{case.completed_steps}/{case.total_steps}]"
        lines.append(f"{case.status:<6}  {(case.label or case.case_id):<37}  {progress_text}")
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
    active_case = next((case for case in snapshot.cases if case.case_id == snapshot.running_case_id), None)
    summary.add_row("Current work", active_case.progress_detail if active_case and active_case.progress_detail else "-")
    summary.add_row("Terminal", "yes" if snapshot.terminal else "no")

    cases_table = Table(expand=True, box=None, show_header=True)
    cases_table.add_column("Status", style="bold")
    cases_table.add_column("Item", overflow="fold", ratio=3)
    cases_table.add_column("Work", overflow="fold", ratio=3)
    cases_table.add_column("Done", justify="right")
    visible_cases = visible_monitor_cases(cases=snapshot.cases, state=state)
    for row_index, case in enumerate(visible_cases):
        absolute_index = state.scroll_offset + row_index
        selected = absolute_index == state.selected_index
        progress_text = case.progress_detail or "-"
        done_text = (
            "-"
            if case.completed_steps is None or case.total_steps is None
            else f"{case.completed_steps}/{case.total_steps}"
        )
        cases_table.add_row(
            _style_status(case.status, selected=selected),
            _selected_label(case.label or case.case_id, selected=selected),
            progress_text,
            done_text,
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
    details.add_row("progress", case.progress_detail or "-")
    details.add_row(
        "steps",
        (
            "-"
            if case.completed_steps is None or case.total_steps is None
            else f"{case.completed_steps}/{case.total_steps}"
        ),
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
