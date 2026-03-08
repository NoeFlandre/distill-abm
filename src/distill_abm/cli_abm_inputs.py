"""Typed helpers for turning ABM configs into CLI smoke/run input objects."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast

import typer

from distill_abm.pipeline.doe_smoke import DoESmokePlotInput
from distill_abm.pipeline.full_case_smoke import FullCasePlotInput, FullCaseSmokeInput
from distill_abm.pipeline.local_qwen_sample_smoke import LocalQwenCaseInput


class PlotConfigLike(Protocol):
    """Minimum plot-config surface used by CLI smoke helpers."""

    reporter_pattern: str


class NetlogoVizConfigLike(Protocol):
    """Minimum netlogo-viz surface used by CLI smoke helpers."""

    plots: Sequence[PlotConfigLike]


class AbmConfigLike(Protocol):
    """Minimum ABM-config surface used by CLI smoke helpers."""

    metric_pattern: str
    metric_description: str
    plot_descriptions: Sequence[str]
    netlogo_viz: NetlogoVizConfigLike | None


def load_abm_config_for_cli(*, abm: str, load_abm_config_fn: Callable[[Path], object]) -> AbmConfigLike:
    """Load one ABM config from the canonical repository location."""
    config = load_abm_config_fn(Path("configs/abms") / f"{abm}.yaml")
    return cast(AbmConfigLike, config)


def apply_abm_metric_defaults(
    *,
    abm_config: AbmConfigLike,
    metric_pattern: str,
    metric_description: str,
    plot_description: str | None,
) -> tuple[str, str, str | None]:
    """Apply ABM-config metric defaults while preserving an explicit plot description."""
    resolved_plot_description = plot_description
    if resolved_plot_description is None and abm_config.plot_descriptions:
        resolved_plot_description = str(abm_config.plot_descriptions[0])
    return abm_config.metric_pattern, abm_config.metric_description, resolved_plot_description


def require_abm_plot_configuration(
    *,
    abm: str,
    abm_config: AbmConfigLike,
) -> tuple[Sequence[PlotConfigLike], Sequence[str]]:
    """Validate that an ABM exposes the plot configuration needed by smoke workflows."""
    netlogo_viz = abm_config.netlogo_viz
    if netlogo_viz is None or not netlogo_viz.plots:
        raise typer.BadParameter(f"missing netlogo_viz plot config for ABM '{abm}'")
    if not abm_config.plot_descriptions:
        raise typer.BadParameter(f"missing plot descriptions for ABM '{abm}'")
    plot_descriptions = tuple(str(item) for item in abm_config.plot_descriptions)
    if len(netlogo_viz.plots) != len(plot_descriptions):
        raise typer.BadParameter(f"plot config and plot descriptions count mismatch for ABM '{abm}'")
    return tuple(netlogo_viz.plots), plot_descriptions


def build_doe_plot_inputs(
    *,
    abm: str,
    abm_config: AbmConfigLike,
    viz_root: Path,
) -> list[DoESmokePlotInput]:
    """Build DOE plot inputs from one ABM config and visualization root."""
    plot_configs, plot_descriptions = require_abm_plot_configuration(abm=abm, abm_config=abm_config)
    return [
        DoESmokePlotInput(
            plot_index=index,
            reporter_pattern=plot_config.reporter_pattern,
            plot_description=plot_description,
            plot_path=viz_root / abm / "plots" / f"{index}.png",
        )
        for index, (plot_config, plot_description) in enumerate(
            zip(plot_configs, plot_descriptions, strict=True),
            start=1,
        )
    ]


def build_local_qwen_case_input(
    *,
    abm: str,
    abm_config: AbmConfigLike,
    ingest_root: Path,
    viz_root: Path,
) -> LocalQwenCaseInput:
    """Build the sampled smoke input for one ABM from its first configured plot."""
    plot_configs, plot_descriptions = require_abm_plot_configuration(abm=abm, abm_config=abm_config)
    first_plot = plot_configs[0]
    first_description = plot_descriptions[0]
    return LocalQwenCaseInput(
        abm=abm,
        csv_path=viz_root / abm / "simulation.csv",
        parameters_path=ingest_root / abm / "TXT" / "narrative_combined.txt",
        documentation_path=ingest_root / abm / "TXT" / "final_documentation.txt",
        reporter_pattern=first_plot.reporter_pattern,
        plot_description=first_description,
        plot_path=viz_root / abm / "plots" / "1.png",
    )


def build_full_case_smoke_input(
    *,
    abm: str,
    abm_config: AbmConfigLike,
    ingest_root: Path,
    viz_root: Path,
) -> FullCaseSmokeInput:
    """Build the full ordered plot input set for one ABM full-case smoke."""
    plot_configs, plot_descriptions = require_abm_plot_configuration(abm=abm, abm_config=abm_config)
    plots = tuple(
        FullCasePlotInput(
            plot_index=index,
            reporter_pattern=plot_config.reporter_pattern,
            plot_description=plot_description,
            plot_path=viz_root / abm / "plots" / f"{index}.png",
        )
        for index, (plot_config, plot_description) in enumerate(
            zip(plot_configs, plot_descriptions, strict=True),
            start=1,
        )
    )
    return FullCaseSmokeInput(
        abm=abm,
        csv_path=viz_root / abm / "simulation.csv",
        parameters_path=ingest_root / abm / "TXT" / "narrative_combined.txt",
        documentation_path=ingest_root / abm / "TXT" / "final_documentation.txt",
        plots=plots,
    )
