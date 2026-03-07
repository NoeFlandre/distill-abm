from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import distill_abm.cli as cli_module
from distill_abm.cli import app

runner = CliRunner()


def _write_supporting_results(root: Path) -> None:
    for abm in ("fauna", "grazing", "milk_consumption"):
        ingest_dir = root / "ingest" / abm / "TXT"
        viz_dir = root / "viz" / abm / "plots"
        ingest_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)
        (ingest_dir / "narrative_combined.txt").write_text("params", encoding="utf-8")
        (ingest_dir / "final_documentation.txt").write_text("docs", encoding="utf-8")
        (root / "viz" / abm / "simulation.csv").write_text("[step];metric-a\n0;1\n", encoding="utf-8")
        (viz_dir / "1.png").write_bytes(b"png")


def test_cli_smoke_local_qwen_invokes_sample_smoke(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _write_supporting_results(tmp_path)
    captured: dict[str, object] = {}

    def fake_run_local_qwen_sample_smoke(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        output_root = kwargs["output_root"]
        return SimpleNamespace(
            success=True,
            report_json_path=Path(output_root) / "report.json",
            report_markdown_path=Path(output_root) / "report.md",
            review_csv_path=Path(output_root) / "review.csv",
            failed_case_ids=[],
        )

    monkeypatch.setattr(cli_module, "_assert_ollama_model_available", lambda _model: None)
    monkeypatch.setattr(cli_module, "create_adapter", lambda provider, model: object())
    monkeypatch.setattr(cli_module, "run_local_qwen_sample_smoke", fake_run_local_qwen_sample_smoke)

    result = runner.invoke(
        app,
        [
            "smoke-local-qwen",
            "--ingest-root",
            str(tmp_path / "ingest"),
            "--viz-root",
            str(tmp_path / "viz"),
            "--output-root",
            str(tmp_path / "out"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert "report.json" in result.output
    assert captured["model"] == "qwen3.5:0.8b"
