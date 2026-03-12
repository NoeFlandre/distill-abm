from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

import distill_abm.cli as cli_module
from distill_abm.cli_abm_inputs import (
    apply_abm_metric_defaults,
    build_doe_plot_inputs,
    build_full_case_smoke_input,
    build_local_qwen_case_input,
    load_abm_config_for_cli,
    require_abm_plot_configuration,
)


def _abm_config() -> SimpleNamespace:
    return SimpleNamespace(
        metric_pattern="metric-a",
        metric_description="metric description",
        plot_descriptions=["First plot", "Second plot"],
        netlogo_viz=SimpleNamespace(
            plots=[
                SimpleNamespace(reporter_pattern="metric-a"),
                SimpleNamespace(reporter_pattern="metric-b"),
            ]
        ),
    )


def test_load_abm_config_for_cli_uses_canonical_config_path() -> None:
    captured: dict[str, Path] = {}

    def fake_load(path: Path) -> object:
        captured["path"] = path
        return _abm_config()

    loaded = load_abm_config_for_cli(abm="grazing", load_abm_config_fn=fake_load)

    assert captured["path"] == Path("configs/abms/grazing.yaml")
    assert loaded.metric_pattern == "metric-a"


def test_apply_abm_metric_defaults_preserves_explicit_plot_description() -> None:
    metric_pattern, metric_description, plot_description = apply_abm_metric_defaults(
        abm_config=_abm_config(),
        metric_pattern="old-pattern",
        metric_description="old-description",
        plot_description="Explicit plot description",
    )

    assert metric_pattern == "metric-a"
    assert metric_description == "metric description"
    assert plot_description == "Explicit plot description"


def test_require_abm_plot_configuration_rejects_missing_plot_config() -> None:
    broken_config = SimpleNamespace(
        metric_pattern="metric-a",
        metric_description="metric description",
        plot_descriptions=["First plot"],
        netlogo_viz=None,
    )

    with pytest.raises(typer.BadParameter, match="missing netlogo_viz plot config"):
        require_abm_plot_configuration(abm="fauna", abm_config=broken_config)


def test_build_local_qwen_case_input_uses_first_plot() -> None:
    built = build_local_qwen_case_input(
        abm="milk_consumption",
        abm_config=_abm_config(),
        ingest_root=Path("results/archive/ingest_smoke_latest"),
        viz_root=Path("results/archive/viz_smoke_latest"),
    )

    assert built.reporter_pattern == "metric-a"
    assert built.plot_description == "First plot"
    assert built.plot_path == Path("results/archive/viz_smoke_latest/milk_consumption/plots/1.png")


def test_build_full_case_smoke_input_builds_all_plot_inputs() -> None:
    built = build_full_case_smoke_input(
        abm="grazing",
        abm_config=_abm_config(),
        ingest_root=Path("results/archive/ingest_smoke_latest"),
        viz_root=Path("results/archive/viz_smoke_latest"),
    )

    assert len(built.plots) == 2
    assert built.plots[0].plot_index == 1
    assert built.plots[1].reporter_pattern == "metric-b"


def test_build_doe_plot_inputs_builds_ordered_plot_specs() -> None:
    built = build_doe_plot_inputs(
        abm="fauna",
        abm_config=_abm_config(),
        viz_root=Path("results/archive/viz_smoke_latest"),
    )

    assert [item.plot_index for item in built] == [1, 2]
    assert built[0].plot_description == "First plot"
    assert built[1].plot_path == Path("results/archive/viz_smoke_latest/fauna/plots/2.png")


def test_cli_archive_root_defaults_keep_results_root_flat() -> None:
    assert cli_module.ARCHIVE_RESULTS_ROOT == Path("results/archive")
    assert inspect.signature(cli_module.smoke_doe).parameters["ingest_root"].default == Path(
        "results/archive/ingest_smoke_latest"
    )
    assert inspect.signature(cli_module.smoke_doe).parameters["viz_root"].default == Path(
        "results/archive/viz_smoke_latest"
    )
    assert inspect.signature(cli_module.smoke_quantitative).parameters["output_root"].default == Path(
        "results/archive/quantitative_smoke_latest"
    )
    assert inspect.signature(cli_module.validate_workspace).parameters["output_root"].default == Path(
        "results/archive/agent_validation/latest"
    )
