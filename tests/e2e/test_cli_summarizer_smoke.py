from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import distill_abm.cli as cli_module
from distill_abm.cli import app

runner = CliRunner()


def test_cli_smoke_summarizers_invokes_runner(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    def fake_run_summarizer_smoke(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        output_root = kwargs["output_root"]
        return SimpleNamespace(
            success=True,
            run_root=Path(output_root) / "runs" / "run_1",
            report_json_path=Path(output_root) / "report.json",
            report_markdown_path=Path(output_root) / "report.md",
            review_csv_path=Path(output_root) / "review.csv",
            validated_sources_path=Path(output_root) / "validated_bundles.json",
            run_log_path=Path(output_root) / "run.log.jsonl",
            failed_bundle_ids=[],
        )

    monkeypatch.setattr(cli_module, "run_summarizer_smoke", fake_run_summarizer_smoke)

    result = runner.invoke(
        app,
        [
            "smoke-summarizers",
            "--source-root",
            str(tmp_path / "source"),
            "--abm",
            "fauna",
            "--abm",
            "grazing",
            "--output-root",
            str(tmp_path / "out"),
            "--resume",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert "report.json" in result.output
    assert captured["source_root"] == tmp_path / "source"
    assert captured["include_abms"] == ("fauna", "grazing")
    assert captured["output_root"] == tmp_path / "out"
    assert captured["resume"] is True
