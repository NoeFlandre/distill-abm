from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import distill_abm.cli as cli_module
from distill_abm.cli import app

runner = CliRunner()


def test_cli_sync_results_bucket_invokes_action(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}
    source_root = tmp_path / "results"
    source_root.mkdir()

    def fake_execute_sync_results_bucket_command(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)

    monkeypatch.setattr(cli_module, "execute_sync_results_bucket_command", fake_execute_sync_results_bucket_command)

    result = runner.invoke(
        app,
        [
            "sync-results-bucket",
            "--source-root",
            str(source_root),
            "--bucket-uri",
            "hf://buckets/NoeFlandre/distill-abms-results",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["source_root"] == source_root
    assert captured["bucket_uri"] == "hf://buckets/NoeFlandre/distill-abms-results"
    assert captured["dry_run"] is True
    assert captured["delete"] is True
    assert captured["json_output"] is True
