from __future__ import annotations

import json
import math
from pathlib import Path

from rich.console import Console

import distill_abm.pipeline.local_qwen_monitor as monitor_module
from distill_abm.pipeline.local_qwen_monitor import (
    LocalQwenCaseSnapshot,
    LocalQwenMonitorSnapshot,
    MonitorViewState,
    apply_monitor_keypress,
    collect_local_qwen_monitor_snapshot,
    next_snapshot_due,
    render_local_qwen_monitor,
    render_local_qwen_monitor_rich,
    stream_local_qwen_monitor,
    visible_monitor_cases,
)


def test_collect_local_qwen_monitor_snapshot_reads_case_progress(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / "01_case"
    requests_dir = case_dir / "02_requests"
    outputs_dir = case_dir / "03_outputs"
    requests_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    (requests_dir / "context_request.json").write_text(
        json.dumps(
            {
                "max_tokens": 32768,
                "prompt_length": 1200,
                "metadata": {"ollama_num_ctx": 131072},
            }
        ),
        encoding="utf-8",
    )
    (requests_dir / "trend_request.json").write_text(
        json.dumps(
            {
                "max_tokens": 32768,
                "prompt_length": 4800,
                "metadata": {"ollama_num_ctx": 131072},
            }
        ),
        encoding="utf-8",
    )
    (outputs_dir / "context_trace.json").write_text(
        json.dumps({"response": {"usage": {"total_tokens": 222}}}),
        encoding="utf-8",
    )
    (outputs_dir / "trend_trace.json").write_text(
        json.dumps({"response": {"usage": {"total_tokens": 333}}}),
        encoding="utf-8",
    )

    snapshot = collect_local_qwen_monitor_snapshot(tmp_path)
    assert snapshot.exists is True
    assert snapshot.mode == "smoke"
    assert snapshot.total_cases == 1
    assert snapshot.completed_cases == 1
    assert snapshot.failed_cases == 0
    case = snapshot.cases[0]
    assert case.status == "completed"
    assert case.num_ctx == 131072
    assert case.max_tokens == 32768
    assert case.context_total_tokens == 222
    assert case.trend_total_tokens == 333


def test_render_local_qwen_monitor_includes_main_metrics(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / "02_case"
    outputs_dir = case_dir / "03_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "error.txt").write_text("boom", encoding="utf-8")

    rendered = render_local_qwen_monitor(collect_local_qwen_monitor_snapshot(tmp_path))
    assert "Local Qwen smoke" in rendered
    assert "02_case" in rendered
    assert "boom" in rendered


def test_collect_local_qwen_monitor_snapshot_reads_tuning_trials(tmp_path: Path) -> None:
    case_dir = tmp_path / "trials" / "plot" / "num_ctx_8192" / "max_tokens_1024" / "cases" / "01_case"
    requests_dir = case_dir / "02_requests"
    outputs_dir = case_dir / "03_outputs"
    requests_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    (requests_dir / "trend_request.json").write_text(
        json.dumps(
            {
                "max_tokens": 1024,
                "prompt_length": 4800,
                "metadata": {"ollama_num_ctx": 8192},
            }
        ),
        encoding="utf-8",
    )
    (outputs_dir / "trend_trace.json").write_text(
        json.dumps({"response": {"usage": {"total_tokens": 555}}}),
        encoding="utf-8",
    )

    snapshot = collect_local_qwen_monitor_snapshot(tmp_path)
    assert snapshot.exists is True
    assert snapshot.mode == "tuning"
    assert snapshot.total_cases == 1
    assert snapshot.completed_cases == 1
    trial = snapshot.cases[0]
    assert trial.label == "num_ctx_8192/max_tokens_1024"
    assert trial.num_ctx == 8192
    assert trial.max_tokens == 1024
    assert trial.trend_total_tokens == 555


def test_collect_local_qwen_monitor_snapshot_resolves_latest_run_and_full_case_layout(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "run_1"
    (tmp_path / "latest_run.txt").write_text(str(run_root), encoding="utf-8")
    case_dir = run_root / "cases" / "01_case"
    context_dir = case_dir / "02_context"
    trend_dir = case_dir / "03_trends" / "plot_01"
    context_dir.mkdir(parents=True, exist_ok=True)
    trend_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "context_request.json").write_text(
        json.dumps({"max_tokens": 2048, "prompt_length": 500, "metadata": {"ollama_num_ctx": 0}}),
        encoding="utf-8",
    )
    (context_dir / "context_trace.json").write_text(
        json.dumps({"response": {"usage": {"total_tokens": 111}}}),
        encoding="utf-8",
    )
    (trend_dir / "trend_request.json").write_text(
        json.dumps({"max_tokens": 2048, "prompt_length": 900, "metadata": {"ollama_num_ctx": 0}}),
        encoding="utf-8",
    )
    (trend_dir / "trend_trace.json").write_text(
        json.dumps({"response": {"usage": {"total_tokens": 222}}}),
        encoding="utf-8",
    )

    snapshot = collect_local_qwen_monitor_snapshot(tmp_path)
    assert snapshot.exists is True
    assert snapshot.total_cases == 1
    case = snapshot.cases[0]
    assert case.status == "completed"
    assert case.context_prompt_length == 500
    assert case.trend_prompt_length == 900
    assert case.context_total_tokens == 111
    assert case.trend_total_tokens == 222
    assert case.progress_detail == "done"
    assert case.completed_steps == 2
    assert case.total_steps == 2


def test_collect_local_qwen_monitor_snapshot_uses_validation_state_for_full_case_status(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "run_1"
    (tmp_path / "latest_run.txt").write_text(str(run_root), encoding="utf-8")
    case_dir = run_root / "cases" / "01_case"
    context_dir = case_dir / "02_context"
    trend_dir = case_dir / "03_trends" / "plot_01"
    context_dir.mkdir(parents=True, exist_ok=True)
    trend_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "context_request.json").write_text(
        json.dumps({"max_tokens": 2048, "prompt_length": 500, "metadata": {"ollama_num_ctx": 0}}),
        encoding="utf-8",
    )
    (context_dir / "context_output.txt").write_text("ok", encoding="utf-8")
    (context_dir / "context_trace.json").write_text(
        json.dumps({"response": {"usage": {"total_tokens": 111}}}),
        encoding="utf-8",
    )
    (case_dir / "validation_state.json").write_text(
        json.dumps(
            {
                "context": {"status": "accepted", "error": None},
                "trends": {"1": {"status": "accepted", "error": None}},
            }
        ),
        encoding="utf-8",
    )

    snapshot = collect_local_qwen_monitor_snapshot(tmp_path)

    assert snapshot.total_cases == 1
    assert snapshot.cases[0].status == "completed"


def test_collect_local_qwen_monitor_snapshot_reports_running_full_case_detail(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "run_1"
    (tmp_path / "latest_run.txt").write_text(str(run_root), encoding="utf-8")
    case_dir = run_root / "cases" / "01_case"
    context_dir = case_dir / "02_context"
    trend_dir = case_dir / "03_trends" / "plot_01"
    context_dir.mkdir(parents=True, exist_ok=True)
    trend_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "context_trace.json").write_text(
        json.dumps({"response": {"usage": {"total_tokens": 111}}}),
        encoding="utf-8",
    )
    (trend_dir / "trend_request.json").write_text(
        json.dumps({"max_tokens": 2048, "prompt_length": 900, "metadata": {"ollama_num_ctx": 0}}),
        encoding="utf-8",
    )
    (case_dir / "validation_state.json").write_text(
        json.dumps(
            {
                "context": {"status": "accepted", "error": None},
                "trends": {},
            }
        ),
        encoding="utf-8",
    )

    snapshot = collect_local_qwen_monitor_snapshot(tmp_path)

    case = snapshot.cases[0]
    assert case.status == "running"
    assert case.progress_detail == "trend plot_01"
    assert case.completed_steps == 1
    assert case.total_steps == 2


def test_collect_local_qwen_monitor_snapshot_reads_suite_progress(tmp_path: Path) -> None:
    abm_run_root = tmp_path / "abms" / "fauna" / "runs" / "run_1"
    (tmp_path / "abms" / "fauna").mkdir(parents=True, exist_ok=True)
    (tmp_path / "abms" / "fauna" / "latest_run.txt").write_text(str(abm_run_root), encoding="utf-8")
    case_dir = abm_run_root / "cases" / "01_case"
    context_dir = case_dir / "02_context"
    trend_dir = case_dir / "03_trends" / "plot_01"
    context_dir.mkdir(parents=True, exist_ok=True)
    trend_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "context_trace.json").write_text(
        json.dumps({"response": {"usage": {"total_tokens": 111}}}),
        encoding="utf-8",
    )
    (trend_dir / "trend_request.json").write_text(
        json.dumps({"max_tokens": 2048, "prompt_length": 900, "metadata": {"ollama_num_ctx": 0}}),
        encoding="utf-8",
    )
    (case_dir / "validation_state.json").write_text(
        json.dumps({"context": {"status": "accepted", "error": None}, "trends": {}}),
        encoding="utf-8",
    )

    (tmp_path / "suite_progress.json").write_text(
        json.dumps(
            {
                "run_id": "run_1",
                "status": "running",
                "current_abm": "fauna",
                "total_abms": 3,
                "completed_abm_count": 1,
                "failed_abm_count": 0,
                "abms": [
                    {"abm": "fauna", "status": "running", "last_error": None},
                    {"abm": "grazing", "status": "pending", "last_error": None},
                    {"abm": "milk_consumption", "status": "pending", "last_error": "waiting"},
                ],
            }
        ),
        encoding="utf-8",
    )

    snapshot = collect_local_qwen_monitor_snapshot(tmp_path)

    assert snapshot.exists is True
    assert snapshot.mode == "suite"
    assert snapshot.total_cases == 3
    assert snapshot.completed_cases == 1
    assert snapshot.failed_cases == 0
    assert snapshot.running_case_id == "fauna"
    assert snapshot.cases[0].progress_detail == "01_case: trend plot_01"
    assert snapshot.cases[0].completed_steps == 0
    assert snapshot.cases[0].total_steps == 1
    assert snapshot.cases[2].error == "waiting"


def test_render_local_qwen_monitor_rich_includes_summary_and_failures() -> None:
    snapshot = LocalQwenMonitorSnapshot(
        output_root=Path("results/run"),
        exists=True,
        mode="smoke",
        total_cases=3,
        completed_cases=1,
        failed_cases=1,
        running_case_id="02_case",
        cases=(
            LocalQwenCaseSnapshot(
                case_id="01_case",
                status="completed",
                label="01_case",
                num_ctx=8192,
                max_tokens=2048,
                context_prompt_length=100,
                trend_prompt_length=200,
                context_total_tokens=50,
                trend_total_tokens=75,
                error=None,
            ),
            LocalQwenCaseSnapshot(
                case_id="02_case",
                status="failed",
                label="02_case",
                num_ctx=8192,
                max_tokens=2048,
                context_prompt_length=100,
                trend_prompt_length=200,
                context_total_tokens=50,
                trend_total_tokens=75,
                error="boom",
            ),
        ),
    )

    console = Console(record=True, width=140)
    console.print(render_local_qwen_monitor_rich(snapshot))
    rendered = console.export_text()

    assert "Run Monitor" in rendered
    assert "02_case" in rendered
    assert "boom" in rendered
    assert "running" in rendered.lower()


def test_stream_local_qwen_monitor_can_stay_open_after_terminal(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls = {"count": 0}
    snapshot = LocalQwenMonitorSnapshot(
        output_root=Path("results/run"),
        exists=True,
        mode="smoke",
        total_cases=1,
        completed_cases=1,
        failed_cases=0,
        running_case_id=None,
        cases=(),
    )

    def fake_collect(_output_root: Path) -> LocalQwenMonitorSnapshot:
        calls["count"] += 1
        return snapshot

    monkeypatch.setattr(monitor_module, "collect_local_qwen_monitor_snapshot", fake_collect)

    stream_local_qwen_monitor(
        output_root=Path("results/run"),
        interval_seconds=0.0,
        exit_when_terminal=False,
        max_refreshes=2,
    )

    assert calls["count"] == 3


def test_apply_monitor_keypress_moves_selection_and_scroll_window() -> None:
    state = MonitorViewState(selected_index=0, scroll_offset=0, visible_rows=2)

    state = apply_monitor_keypress(state, "down", total_cases=5)
    assert state.selected_index == 1
    assert state.scroll_offset == 0

    state = apply_monitor_keypress(state, "down", total_cases=5)
    assert state.selected_index == 2
    assert state.scroll_offset == 1

    state = apply_monitor_keypress(state, "up", total_cases=5)
    assert state.selected_index == 1
    assert state.scroll_offset == 1


def test_visible_monitor_cases_returns_selected_window() -> None:
    cases = tuple(
        LocalQwenCaseSnapshot(
            case_id=f"{index:02d}",
            status="completed",
            label=f"case-{index}",
            num_ctx=None,
            max_tokens=None,
            context_prompt_length=None,
            trend_prompt_length=None,
            context_total_tokens=None,
            trend_total_tokens=None,
            error=None,
        )
        for index in range(5)
    )
    state = MonitorViewState(selected_index=3, scroll_offset=2, visible_rows=2)

    visible = visible_monitor_cases(cases=cases, state=state)

    assert [case.case_id for case in visible] == ["02", "03"]


def test_next_snapshot_due_decouples_fast_input_polling_from_refresh_interval() -> None:
    due = next_snapshot_due(last_snapshot_at=10.0, now=10.05, interval_seconds=2.0)
    assert math.isclose(due, 1.95)

    due = next_snapshot_due(last_snapshot_at=10.0, now=12.5, interval_seconds=2.0)
    assert due == 0.0
