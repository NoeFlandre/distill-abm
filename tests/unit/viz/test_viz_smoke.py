from __future__ import annotations

from pathlib import Path

import pandas as pd

from distill_abm.viz.plots import MetricPlotBundle
from distill_abm.viz.viz_smoke import VizSmokeSpec, default_viz_smoke_stages, run_viz_smoke_suite


class _FakeNetLogoLink:
    def __init__(self) -> None:
        self.tick = 0
        self.loaded_model: str | None = None

    def load_model(self, model_path: str) -> None:
        self.loaded_model = model_path

    def command(self, command: str) -> None:
        if command == "go":
            self.tick += 1

    def report(self, reporter: str) -> float:
        return float(self.tick * 10 + len(reporter))


def test_default_viz_smoke_stages_are_granular() -> None:
    specs = {
        "milk_consumption": VizSmokeSpec(
            abm="milk_consumption",
            model_path=Path("data/milk_consumption_abm/milk_consumption.nlogo"),
            experiment_name="Milk Consumption Trends",
            experiment_parameters={"number-of-agents": 1000},
            num_runs=2,
            max_ticks=4,
            interval=1,
            reporters=["mean-incum", "mean-alt"],
            plots=[
                MetricPlotBundle(
                    include_pattern="mean-incum",
                    title="Mean Whole Milk Consumption",
                    y_label="Mean Whole Milk Consumption",
                    x_label="Steps",
                    show_mean_line=False,
                ),
                MetricPlotBundle(
                    include_pattern="mean-alt",
                    title="Mean Skimmed/Semi-Skimmed Milk Consumption",
                    y_label="Mean Skimmed/Semi-Skimmed Milk Consumption",
                    x_label="Steps",
                    show_mean_line=False,
                ),
            ],
        )
    }

    assert [stage.stage_id for stage in default_viz_smoke_stages(specs)] == [
        "simulation-csv",
        "plot-1",
        "plot-2",
    ]


def test_run_viz_smoke_suite_writes_stage_level_report(tmp_path: Path) -> None:
    model_path = tmp_path / "model.nlogo"
    model_path.write_text("to setup\nend\nto go\ntick\nend\n", encoding="utf-8")
    specs = {
        "milk_consumption": VizSmokeSpec(
            abm="milk_consumption",
            model_path=model_path,
            experiment_name="Milk Consumption Trends",
            experiment_parameters={"number-of-agents": 1000},
            num_runs=2,
            max_ticks=4,
            interval=1,
            reporters=["mean-incum", "mean-alt"],
            plots=[
                MetricPlotBundle(
                    include_pattern="mean-incum",
                    title="Mean Whole Milk Consumption",
                    y_label="Mean Whole Milk Consumption",
                    x_label="Steps",
                    show_mean_line=False,
                ),
                MetricPlotBundle(
                    include_pattern="mean-alt",
                    title="Mean Skimmed/Semi-Skimmed Milk Consumption",
                    y_label="Mean Skimmed/Semi-Skimmed Milk Consumption",
                    x_label="Steps",
                    show_mean_line=False,
                ),
            ],
        )
    }

    result = run_viz_smoke_suite(
        specs=specs,
        netlogo_home="/fake/netlogo",
        output_root=tmp_path / "viz-smoke",
        netlogo_link_factory=lambda **_: _FakeNetLogoLink(),
    )

    assert result.success is True
    assert result.report_json_path.exists()
    assert result.report_markdown_path.exists()
    assert len(result.abms) == 1
    assert result.abms[0].parameters_path.exists()
    assert result.abms[0].artifact_source == "simulated"
    assert [stage.stage.stage_id for stage in result.abms[0].stage_results] == [
        "simulation-csv",
        "plot-1",
        "plot-2",
    ]
    csv_path = tmp_path / "viz-smoke" / "milk_consumption" / "simulation.csv"
    frame = pd.read_csv(csv_path, sep=";")
    assert list(frame.columns) == ["mean-incum", "mean-alt", "tick", "mean-incum.1", "mean-alt.1", "tick.1"]
    assert (tmp_path / "viz-smoke" / "milk_consumption" / "plots" / "1.png").exists()
    assert (tmp_path / "viz-smoke" / "milk_consumption" / "plots" / "2.png").exists()


def test_run_viz_smoke_suite_records_simulation_exceptions_and_continues(tmp_path: Path) -> None:
    model_path = tmp_path / "model.nlogo"
    model_path.write_text("to setup\nend\nto go\ntick\nend\n", encoding="utf-8")
    spec = VizSmokeSpec(
        abm="milk_consumption",
        model_path=model_path,
        experiment_name="Milk Consumption Trends",
        experiment_parameters={"number-of-agents": 1000},
        num_runs=1,
        max_ticks=2,
        interval=1,
        reporters=["mean-incum"],
        plots=[
            MetricPlotBundle(
                include_pattern="mean-incum",
                title="Mean Whole Milk Consumption",
                y_label="Mean Whole Milk Consumption",
                x_label="Steps",
                show_mean_line=False,
            )
        ],
    )

    def _boom(**_: object) -> _FakeNetLogoLink:
        raise ValueError("missing include file")

    result = run_viz_smoke_suite(
        specs={"milk_consumption": spec},
        netlogo_home="/fake/netlogo",
        output_root=tmp_path / "viz-smoke",
        netlogo_link_factory=_boom,
    )

    assert result.success is False
    assert result.failed_abms == ["milk_consumption"]
    assert result.abms[0].artifact_source == "simulated"
    assert result.abms[0].error_code == "simulation_failed"
    assert "missing include file" in (result.abms[0].error or "")


def test_run_viz_smoke_suite_uses_fallback_artifacts_when_configured(tmp_path: Path) -> None:
    model_path = tmp_path / "model.nlogo"
    model_path.write_text("to setup\nend\nto go\ntick\nend\n", encoding="utf-8")
    fallback_csv = tmp_path / "legacy.csv"
    fallback_csv.write_text("mean-incum;tick\n1;0\n2;1\n", encoding="utf-8")
    fallback_plot_dir = tmp_path / "legacy-plots"
    fallback_plot_dir.mkdir()
    (fallback_plot_dir / "1.png").write_bytes(b"plot-one")
    spec = VizSmokeSpec(
        abm="milk_consumption",
        model_path=model_path,
        experiment_name="Milk Consumption Trends",
        experiment_parameters={"number-of-agents": 1000},
        num_runs=1,
        max_ticks=2,
        interval=1,
        fallback_csv=fallback_csv,
        fallback_plot_dir=fallback_plot_dir,
        reporters=["mean-incum"],
        plots=[
            MetricPlotBundle(
                include_pattern="mean-incum",
                title="Mean Whole Milk Consumption",
                y_label="Mean Whole Milk Consumption",
                x_label="Steps",
                show_mean_line=False,
            )
        ],
    )

    def _boom(**_: object) -> _FakeNetLogoLink:
        raise ValueError("runtime blocked")

    result = run_viz_smoke_suite(
        specs={"milk_consumption": spec},
        netlogo_home="/fake/netlogo",
        output_root=tmp_path / "viz-smoke",
        netlogo_link_factory=_boom,
    )

    assert result.success is True
    assert result.abms[0].artifact_source == "fallback"
    assert (tmp_path / "viz-smoke" / "milk_consumption" / "simulation.csv").read_text(encoding="utf-8") == (
        "mean-incum;tick\n1;0\n2;1\n"
    )
    assert (tmp_path / "viz-smoke" / "milk_consumption" / "plots" / "1.png").read_bytes() == b"plot-one"


def test_run_viz_smoke_suite_respects_always_fallback_mode(tmp_path: Path) -> None:
    model_path = tmp_path / "model.nlogo"
    model_path.write_text("to setup\nend\nto go\ntick\nend\n", encoding="utf-8")
    fallback_csv = tmp_path / "legacy.csv"
    fallback_csv.write_text("mean-incum;tick\n9;0\n", encoding="utf-8")
    fallback_plot_dir = tmp_path / "legacy-plots"
    fallback_plot_dir.mkdir()
    (fallback_plot_dir / "1.png").write_bytes(b"fallback-plot")
    spec = VizSmokeSpec(
        abm="milk_consumption",
        model_path=model_path,
        experiment_name="Milk Consumption Trends",
        experiment_parameters={"number-of-agents": 1000},
        num_runs=1,
        max_ticks=2,
        interval=1,
        fallback_mode="always",
        fallback_csv=fallback_csv,
        fallback_plot_dir=fallback_plot_dir,
        reporters=["mean-incum"],
        plots=[
            MetricPlotBundle(
                include_pattern="mean-incum",
                title="Mean Whole Milk Consumption",
                y_label="Mean Whole Milk Consumption",
                x_label="Steps",
                show_mean_line=False,
            )
        ],
    )

    result = run_viz_smoke_suite(
        specs={"milk_consumption": spec},
        netlogo_home="/fake/netlogo",
        output_root=tmp_path / "viz-smoke",
        netlogo_link_factory=lambda **_: _FakeNetLogoLink(),
    )

    assert result.success is True
    assert result.abms[0].artifact_source == "fallback"
    assert (tmp_path / "viz-smoke" / "milk_consumption" / "simulation.csv").read_text(encoding="utf-8") == (
        "mean-incum;tick\n9;0\n"
    )
