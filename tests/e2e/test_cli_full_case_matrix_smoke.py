from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import distill_abm.cli as cli_module
from distill_abm.cli import app

runner = CliRunner()


def test_cli_smoke_full_case_matrix_invokes_runner(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
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
            self.netlogo_viz = SimpleNamespace(plots=[_PlotCfg("metric one"), _PlotCfg("metric two")])
            self.plot_descriptions = ["First plot", "Second plot"]

    def fake_run_full_case_matrix_smoke(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        output_root = kwargs["output_root"] / "runs" / "run_1"
        return SimpleNamespace(
            success=True,
            report_json_path=Path(output_root) / "report.json",
            report_markdown_path=Path(output_root) / "report.md",
            review_csv_path=Path(output_root) / "review.csv",
            viewer_html_path=Path(output_root) / "review.html",
            failed_case_ids=[],
        )

    monkeypatch.setattr(cli_module, "run_full_case_matrix_smoke", fake_run_full_case_matrix_smoke)
    monkeypatch.setattr(cli_module, "load_abm_config", lambda _path: _AbmCfg())
    monkeypatch.setattr(cli_module, "resolve_abm_model_path", lambda **_kwargs: tmp_path / "model.nlogo")
    monkeypatch.setattr(cli_module, "create_adapter", lambda _provider, _model, timeout_seconds=None: object())

    result = runner.invoke(
        app,
        [
            "smoke-full-case-matrix",
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
    cases = captured["cases"]
    assert isinstance(cases, tuple)
    assert len(cases) == 72
    assert cases[0].case_id == "01_grazing_none_plot_rep1"


def test_cli_monitor_run_uses_generic_monitor_command(tmp_path: Path) -> None:
    result = runner.invoke(app, ["monitor-run", "--output-root", str(tmp_path), "--json"])
    assert result.exit_code == 0
    assert '"exists": false' in result.output
