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
