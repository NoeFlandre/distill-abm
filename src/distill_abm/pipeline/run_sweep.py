"""Prompt-sweep orchestration extracted from pipeline run orchestration."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

from distill_abm.pipeline import helpers

if TYPE_CHECKING:
    from distill_abm.configs.models import PromptsConfig
    from distill_abm.llm.adapters.base import LLMAdapter
    from distill_abm.pipeline.run import PipelineInputs, SweepCsvColumnStyle, SweepRunResult


def run_pipeline_sweep(
    inputs: PipelineInputs,
    prompts: PromptsConfig,
    adapter: LLMAdapter,
    image_paths: list[Path],
    plot_descriptions: list[str],
    style_feature_keys: list[str] | None = None,
    output_csv: Path | None = None,
    context_adapter: LLMAdapter | None = None,
    trend_adapter: LLMAdapter | None = None,
    context_model: str | None = None,
    trend_model: str | None = None,
    csv_column_style: SweepCsvColumnStyle = "trend",
    resume_existing: bool = False,
) -> Path:
    """Run prompt/combo sweeps and write trend outputs to a wide CSV."""
    if not image_paths:
        raise ValueError("image_paths cannot be empty")
    if len(image_paths) != len(plot_descriptions):
        raise ValueError("image_paths and plot_descriptions must have the same length")

    out = output_csv or (inputs.output_dir / "combinations_report.csv")
    context_client = context_adapter or adapter
    trend_client = trend_adapter or adapter
    context_model_name = context_model or inputs.model
    trend_model_name = trend_model or inputs.model
    combinations_to_run = build_style_feature_combinations(prompts, style_feature_keys)
    if resume_existing and out.exists():
        existing_descriptions = _load_existing_combination_descriptions(out)
        combinations_to_run = [
            (description, enabled_features)
            for description, enabled_features in combinations_to_run
            if description not in existing_descriptions
        ]
        if not combinations_to_run:
            return out

    if resume_existing:
        for description, enabled_features in combinations_to_run:
            row = _run_sweep_combination(
                description=description,
                enabled_features=enabled_features,
                inputs=inputs,
                prompts=prompts,
                context_client=context_client,
                trend_client=trend_client,
                context_model_name=context_model_name,
                trend_model_name=trend_model_name,
                image_paths=image_paths,
                plot_descriptions=plot_descriptions,
            )
            write_combinations_csv(
                out,
                [row],
                csv_column_style=csv_column_style,
                resume_existing=True,
            )
        return out

    rows: list[SweepRunResult] = []
    for description, enabled_features in combinations_to_run:
        rows.append(
            _run_sweep_combination(
                description=description,
                enabled_features=enabled_features,
                inputs=inputs,
                prompts=prompts,
                context_client=context_client,
                trend_client=trend_client,
                context_model_name=context_model_name,
                trend_model_name=trend_model_name,
                image_paths=image_paths,
                plot_descriptions=plot_descriptions,
            )
        )
    return write_combinations_csv(
        out,
        rows,
        csv_column_style=csv_column_style,
        resume_existing=resume_existing,
    )


def write_combinations_csv(
    output_csv: Path,
    rows: list[SweepRunResult],
    csv_column_style: SweepCsvColumnStyle = "trend",
    resume_existing: bool = False,
) -> Path:
    """Write the wide prompt/response matrix used by sweep mode."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    max_items = max((len(row.trend_analysis_prompts) for row in rows), default=0)
    headers = _sweep_headers(max_items=max_items, csv_column_style=csv_column_style)
    if resume_existing:
        return _write_combinations_csv_resume(output_csv=output_csv, rows=rows, headers=headers)
    helpers.write_sweep_rows(output_csv=output_csv, rows=rows, headers=headers)
    return output_csv


def _sweep_headers(max_items: int, csv_column_style: SweepCsvColumnStyle) -> list[str]:
    return helpers.sweep_headers(max_items=max_items, csv_column_style=csv_column_style)


def _load_existing_combination_descriptions(output_csv: Path) -> set[str]:
    try:
        frame = pd.read_csv(output_csv, keep_default_na=False)
    except Exception:
        return set()
    if "Combination Description" not in frame.columns:
        return set()
    values = frame["Combination Description"].dropna().astype(str)
    return {value.strip() for value in values if value.strip()}


def _run_sweep_combination(
    description: str,
    enabled_features: set[str],
    inputs: PipelineInputs,
    prompts: PromptsConfig,
    context_client: LLMAdapter,
    trend_client: LLMAdapter,
    context_model_name: str,
    trend_model_name: str,
    image_paths: list[Path],
    plot_descriptions: list[str],
) -> SweepRunResult:
    context_prompt = _build_context_prompt(
        inputs=inputs,
        prompts=prompts,
        enabled_style_features=enabled_features,
    )
    context_response = _invoke_adapter(context_client, model=context_model_name, prompt=context_prompt)
    trend_prompt_base = _build_trend_prompt(
        prompts=prompts,
        metric_description=inputs.metric_description,
        context=context_response,
        plot_description=None,
        evidence_mode="plot",
        stats_table_csv="",
        enabled_style_features=enabled_features,
    )
    prompts_for_images: list[str] = []
    responses_for_images: list[str] = []
    for image_path, plot_description in zip(image_paths, plot_descriptions, strict=True):
        trend_prompt = _append_plot_description(trend_prompt_base, plot_description)
        response = _invoke_adapter(
            trend_client,
            model=trend_model_name,
            prompt=trend_prompt,
            image_b64=_encode_image(image_path),
        )
        prompts_for_images.append(trend_prompt)
        responses_for_images.append(response)
    from distill_abm.pipeline.run import SweepRunResult

    return SweepRunResult(
        combination_description=description,
        context_prompt=context_prompt,
        context_response=context_response,
        trend_analysis_prompts=prompts_for_images,
        trend_analysis_responses=responses_for_images,
    )


def build_style_feature_combinations(
    prompts: PromptsConfig,
    style_feature_keys: list[str] | None = None,
) -> list[tuple[str, set[str]]]:
    """Build all subsets of available style features for sweep runs."""
    requested = style_feature_keys or ["role", "example", "insights"]
    available = [key for key in requested if prompts.style_features.get(key, "").strip()]
    combinations_to_run: list[tuple[str, set[str]]] = [("None", set())]
    for size in range(1, len(available) + 1):
        for combo in combinations(available, size):
            combinations_to_run.append((" + ".join(combo), set(combo)))
    return combinations_to_run


def _write_combinations_csv_resume(output_csv: Path, rows: list[SweepRunResult], headers: list[str]) -> Path:
    helpers.write_sweep_rows_resume(output_csv=output_csv, rows=rows, headers=headers)
    return output_csv


def _build_context_prompt(
    inputs: PipelineInputs,
    prompts: PromptsConfig,
    enabled_style_features: set[str],
) -> str:
    return helpers.build_context_prompt(
        inputs_csv_path=inputs.parameters_path,
        inputs_doc_path=inputs.documentation_path,
        prompts=prompts,
        enabled=enabled_style_features,
    )


def _build_trend_prompt(
    prompts: PromptsConfig,
    metric_description: str,
    context: str,
    plot_description: str | None,
    evidence_mode: Literal["plot", "table", "plot+table"],
    stats_table_csv: str,
    enabled_style_features: set[str] | None = None,
) -> str:
    return helpers.build_trend_prompt(
        prompts=prompts,
        metric_description=metric_description,
        context=context,
        plot_description=plot_description,
        evidence_mode=evidence_mode,
        stats_table_csv=stats_table_csv,
        enabled=enabled_style_features,
    )


def _encode_image(path: Path) -> str:
    return helpers.encode_image(path)


def _append_plot_description(base_prompt: str, plot_description: str) -> str:
    return helpers.append_plot_description(base_prompt=base_prompt, plot_description=plot_description)


def _invoke_adapter(
    adapter: LLMAdapter,
    model: str,
    prompt: str,
    image_b64: str | None = None,
) -> str:
    return helpers.invoke_adapter(adapter=adapter, model=model, prompt=prompt, image_b64=image_b64)
