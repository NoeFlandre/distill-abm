from __future__ import annotations

import json
from pathlib import Path

from distill_abm.pipeline.local_qwen_monitor import (
    collect_local_qwen_monitor_snapshot,
    render_local_qwen_monitor,
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
