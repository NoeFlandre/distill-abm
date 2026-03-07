from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import distill_abm.cli as cli_module
from distill_abm.cli import app

runner = CliRunner()


def test_cli_smoke_full_case_invokes_runner(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}
    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        (
            "models:\n"
            "  nemotron_nano_12b_v2_vl_free:\n"
            "    provider: openrouter\n"
            "    model: nvidia/nemotron-nano-12b-v2-vl:free\n"
        ),
        encoding="utf-8",
    )

    class _PlotCfg:
        def __init__(self, reporter_pattern: str) -> None:
            self.reporter_pattern = reporter_pattern

    class _AbmCfg:
        def __init__(self) -> None:
            self.netlogo_viz = SimpleNamespace(plots=[_PlotCfg("metric one")])
            self.plot_descriptions = ["First plot"]

    def fake_run_full_case_smoke(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        output_root = kwargs["output_root"]
        return SimpleNamespace(
            success=True,
            report_json_path=Path(output_root) / "report.json",
            report_markdown_path=Path(output_root) / "report.md",
            review_csv_path=Path(output_root) / "review.csv",
            failed_plot_indices=[],
        )

    monkeypatch.setattr(cli_module, "run_full_case_smoke", fake_run_full_case_smoke)
    monkeypatch.setattr(cli_module, "load_abm_config", lambda _path: _AbmCfg())
    monkeypatch.setattr(cli_module, "resolve_abm_model_path", lambda **_kwargs: tmp_path / "model.nlogo")
    monkeypatch.setattr(cli_module, "create_adapter", lambda _provider, _model, timeout_seconds=None: object())

    result = runner.invoke(
        app,
        [
            "smoke-full-case",
            "--abm",
            "grazing",
            "--models-root",
            str(tmp_path),
            "--ingest-root",
            str(tmp_path / "ingest"),
            "--viz-root",
            str(tmp_path / "viz"),
            "--models-path",
            str(models_path),
            "--output-root",
            str(tmp_path / "out"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert "report.json" in result.output
    assert captured["evidence_mode"] == "table"
    assert captured["prompt_variant"] == "role"
    assert captured["resume_existing"] is True
