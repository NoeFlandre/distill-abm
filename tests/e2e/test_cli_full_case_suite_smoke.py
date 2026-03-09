from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from typer.testing import CliRunner

import distill_abm.cli as cli_module
from distill_abm.cli import app

runner = CliRunner()


def test_cli_smoke_full_case_suite_invokes_runner(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}
    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        (
            "models:\n"
            "  mistral_medium_debug:\n"
            "    provider: mistral\n"
            "    model: mistral-medium-latest\n"
        ),
        encoding="utf-8",
    )

    class _PlotCfg:
        def __init__(self, reporter_pattern: str) -> None:
            self.reporter_pattern = reporter_pattern

    class _AbmCfg:
        def __init__(self, plot_count: int) -> None:
            self.netlogo_viz = SimpleNamespace(plots=[_PlotCfg(f"metric {index}") for index in range(plot_count)])
            self.plot_descriptions = [f"Plot {index}" for index in range(plot_count)]

    def fake_load_abm_config(path: Path):  # type: ignore[no-untyped-def]
        if "fauna" in str(path):
            return _AbmCfg(14)
        if "grazing" in str(path):
            return _AbmCfg(10)
        return _AbmCfg(12)

    def fake_run_full_case_suite_smoke(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        run_root = kwargs["output_root"] / "runs" / "run_1"
        run_root.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            success=True,
            report_json_path=run_root / "report.json",
            report_markdown_path=run_root / "report.md",
            review_csv_path=run_root / "review.csv",
            review_html_path=run_root / "review.html",
            failed_abms=[],
        )

    monkeypatch.setattr(cli_module, "run_full_case_suite_smoke", fake_run_full_case_suite_smoke)
    monkeypatch.setattr(cli_module, "load_abm_config", fake_load_abm_config)
    monkeypatch.setattr(cli_module, "resolve_abm_model_path", lambda *args, **kwargs: tmp_path / "model.nlogo")
    monkeypatch.setattr(cli_module, "create_adapter", lambda _provider, _model, timeout_seconds=None: object())

    result = runner.invoke(
        app,
        [
            "smoke-full-case-suite",
            "--models-root",
            str(tmp_path),
            "--ingest-root",
            str(tmp_path / "ingest"),
            "--viz-root",
            str(tmp_path / "viz"),
            "--models-path",
            str(models_path),
            "--model-id",
            "mistral_medium_debug",
            "--output-root",
            str(tmp_path / "out"),
            "--allow-debug-model",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert "report.json" in result.output
    assert set(cast(dict[str, object], captured["abm_inputs"])) == {"fauna", "grazing", "milk_consumption"}
