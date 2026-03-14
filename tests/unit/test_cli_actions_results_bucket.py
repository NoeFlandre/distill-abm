from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import typer

from distill_abm.cli_actions import execute_sync_results_bucket_command


def test_execute_sync_results_bucket_command_builds_plan_with_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "results"
    source_root.mkdir()
    plan_path = tmp_path / "plans" / "sync.jsonl"
    captured: dict[str, object] = {}

    def fake_run(command: list[str], capture_output: bool, text: bool):  # type: ignore[no-untyped-def]
        captured["command"] = command
        captured["capture_output"] = capture_output
        captured["text"] = text
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr("distill_abm.cli_actions.shutil.which", lambda _: "/usr/local/bin/hf")
    monkeypatch.setattr("distill_abm.cli_actions.subprocess.run", fake_run)
    monkeypatch.setattr("distill_abm.cli_actions.typer.echo", lambda *args, **kwargs: None)
    monkeypatch.setenv("HF_TOKEN", "secret")

    execute_sync_results_bucket_command(
        source_root=source_root,
        bucket_uri="hf://buckets/NoeFlandre/distill-abms-results",
        dry_run=True,
        delete=True,
        plan_path=plan_path,
        json_output=False,
        token_env_var="HF_TOKEN",
    )

    assert captured["command"] == [
        "/usr/local/bin/hf",
        "sync",
        str(source_root),
        "hf://buckets/NoeFlandre/distill-abms-results",
        "--delete",
        "--plan",
        str(plan_path),
        "--token",
        "secret",
    ]
    assert plan_path.parent.exists()


def test_execute_sync_results_bucket_command_requires_hf_binary(tmp_path: Path) -> None:
    source_root = tmp_path / "results"
    source_root.mkdir()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("distill_abm.cli_actions.shutil.which", lambda _: None)
        with pytest.raises(typer.BadParameter, match="missing `hf` CLI"):
            execute_sync_results_bucket_command(
                source_root=source_root,
                bucket_uri="hf://buckets/NoeFlandre/distill-abms-results",
                dry_run=False,
                delete=True,
                plan_path=None,
                json_output=False,
                token_env_var="HF_TOKEN",
            )
