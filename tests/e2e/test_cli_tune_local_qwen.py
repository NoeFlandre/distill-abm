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


def test_cli_tune_local_qwen_invokes_tuning(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _write_supporting_results(tmp_path)
    captured: dict[str, object] = {}
    adapter_calls: dict[str, object] = {}

    def fake_run_local_qwen_tuning(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        output_root = kwargs["output_root"]
        return SimpleNamespace(
            success=True,
            report_json_path=Path(output_root) / "report.json",
            report_markdown_path=Path(output_root) / "report.md",
            trials_csv_path=Path(output_root) / "trials.csv",
            recommendations=[],
            trials=[],
        )

    monkeypatch.setattr(cli_module, "_assert_ollama_model_available", lambda _model: None)
    def fake_create_adapter(provider, model, **kwargs):  # type: ignore[no-untyped-def]
        adapter_calls.update({"provider": provider, "model": model, **kwargs})
        return object()

    monkeypatch.setattr(cli_module, "create_adapter", fake_create_adapter)
    monkeypatch.setattr(cli_module, "run_local_qwen_tuning", fake_run_local_qwen_tuning)

    result = runner.invoke(
        app,
        [
            "tune-local-qwen",
            "--ingest-root",
            str(tmp_path / "ingest"),
            "--viz-root",
            str(tmp_path / "viz"),
            "--output-root",
            str(tmp_path / "out"),
            "--resume",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert "report.json" in result.output
    assert captured["model"] == "qwen3.5:0.8b"
    assert captured["resume_existing"] is True
    assert captured["max_tokens_candidates"] == (1024, 2048, 4096, 8192, 16384)
    assert adapter_calls["timeout_seconds"] == 900.0
