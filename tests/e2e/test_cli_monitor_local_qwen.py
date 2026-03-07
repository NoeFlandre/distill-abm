from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from distill_abm.cli import app

runner = CliRunner()


def test_cli_monitor_local_qwen_prints_json_snapshot(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / "01_case" / "02_requests"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "context_request.json").write_text(
        json.dumps(
            {
                "max_tokens": 32768,
                "prompt_length": 1200,
                "metadata": {"ollama_num_ctx": 131072},
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "monitor-local-qwen",
            "--output-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"total_cases": 1' in result.output
    assert '"num_ctx": 131072' in result.output
    assert '"mode": "smoke"' in result.output


def test_cli_monitor_local_qwen_detects_tuning_layout(tmp_path: Path) -> None:
    case_dir = tmp_path / "trials" / "plot" / "num_ctx_8192" / "max_tokens_1024" / "cases" / "01_case" / "02_requests"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "trend_request.json").write_text(
        json.dumps(
            {
                "max_tokens": 1024,
                "prompt_length": 2000,
                "metadata": {"ollama_num_ctx": 8192},
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "monitor-local-qwen",
            "--output-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"mode": "tuning"' in result.output
    assert '"num_ctx": 8192' in result.output
