"""Optional DOE and sweep steps for smoke workflows."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from itertools import cycle
from pathlib import Path
from typing import Any

from distill_abm.configs.models import PromptsConfig
from distill_abm.eval.doe_full import analyze_factorial_anova
from distill_abm.llm.adapters.base import LLMAdapter
from distill_abm.pipeline import run as run_module
from distill_abm.pipeline.run import PipelineInputs
from distill_abm.pipeline.smoke_types import SmokeCaseResult, SmokeStatus, SmokeSuiteInputs


def run_doe_if_requested(
    output_root: Path,
    doe_input_csv: Path | None,
    resume_existing: bool,
) -> tuple[SmokeStatus, Path | None, str | None]:
    """Run the optional DOE post-processing step when an input CSV is provided."""
    if doe_input_csv is None:
        return "skipped", None, None
    output_csv = output_root / "doe" / "anova_factorial_contributions.csv"
    if resume_existing and output_csv.exists():
        return "ok", output_csv, None
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame = analyze_factorial_anova(doe_input_csv, output_csv, max_interaction_order=2)
    except Exception:
        return "failed", output_csv, traceback.format_exc()
    if frame is None:
        return "failed", output_csv, "analyze_factorial_anova returned no result"
    return "ok", output_csv, None


def run_sweep_if_requested(
    output_root: Path,
    inputs: SmokeSuiteInputs,
    prompts: PromptsConfig,
    adapter: LLMAdapter | Any,
    case_results: list[SmokeCaseResult],
    run_sweep: bool,
    resume_existing: bool,
    run_pipeline_sweep_fn: Callable[..., object] = run_module.run_pipeline_sweep,
) -> tuple[SmokeStatus, Path | None, str | None]:
    """Run the optional prompt-sweep step when enabled."""
    if not run_sweep:
        return "skipped", None, None
    sweep_output = output_root / "sweep" / "combinations_report.csv"
    sweep_descriptions = inputs.sweep_plot_descriptions or [inputs.plot_description or inputs.metric_description]
    available_plots = [case.plot_path for case in case_results if case.plot_path is not None]
    if not available_plots:
        if resume_existing and sweep_output.exists():
            return "ok", sweep_output, None
        return "failed", None, "no successful case produced a plot image for sweep execution"
    plot_count = len(sweep_descriptions)
    if plot_count <= 0:
        return "failed", None, "sweep plot descriptions cannot be empty"
    if len(available_plots) >= plot_count:
        sweep_image_paths = available_plots[:plot_count]
    else:
        sweep_image_paths = [plot for _, plot in zip(range(plot_count), cycle(available_plots), strict=False)]
    try:
        run_pipeline_sweep_fn(
            inputs=PipelineInputs(
                csv_path=inputs.csv_path,
                parameters_path=inputs.parameters_path,
                documentation_path=inputs.documentation_path,
                output_dir=output_root / "sweep",
                model=inputs.model,
                metric_pattern=inputs.metric_pattern,
                metric_description=inputs.metric_description,
                plot_description=inputs.plot_description,
                text_source_mode=inputs.text_source_mode,
                evidence_mode=inputs.evidence_mode,
                summarizers=inputs.summarizers,
            ),
            prompts=prompts,
            adapter=adapter,
            image_paths=sweep_image_paths,
            plot_descriptions=sweep_descriptions,
            output_csv=sweep_output,
            resume_existing=resume_existing,
        )
    except Exception:
        return "failed", sweep_output, traceback.format_exc()
    return "ok", sweep_output, None
