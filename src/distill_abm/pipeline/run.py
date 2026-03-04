"""Pipeline orchestration from CSV ingestion to scored report export."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from distill_abm.configs.models import PromptsConfig
from distill_abm.eval.metrics import SummaryScores, score_summary
from distill_abm.ingest.csv_ingest import load_simulation_csv
from distill_abm.llm.adapters.base import LLMAdapter
from distill_abm.pipeline import helpers
from distill_abm.summarize.models import summarize_with_bart, summarize_with_bert
from distill_abm.viz.plots import MetricPlotBundle, plot_metric_bundles

EvidenceMode = Literal["plot", "stats-markdown", "stats-image", "plot+stats"]
SummarizationMode = Literal["full", "summary", "both"]
ScoreMode = Literal["full", "summary", "both"]
SweepCsvColumnStyle = Literal["trend", "plot"]


class PipelineInputs(BaseModel):
    """Defines required inputs so CLI calls stay predictable and testable."""

    csv_path: Path
    parameters_path: Path
    documentation_path: Path
    output_dir: Path
    model: str
    metric_pattern: str
    metric_description: str
    plot_description: str | None = None
    skip_summarization: bool = False
    summarization_mode: SummarizationMode = "both"
    score_on: ScoreMode = "both"
    evidence_mode: EvidenceMode = "plot"


class PipelineResult(BaseModel):
    """Returns generated artifacts to callers without leaking internal state."""

    plot_path: Path
    report_csv: Path
    context_response: str
    trend_response: str
    trend_full_response: str
    trend_summary_response: str | None = None
    full_scores: SummaryScores | None = None
    summary_scores: SummaryScores | None = None
    stats_markdown: str | None = None
    stats_image_path: Path | None = None
    token_f1: float
    bleu: float
    meteor: float
    rouge1: float
    rouge2: float
    rouge_l: float
    flesch_reading_ease: float


class SweepRunResult(BaseModel):
    """Represents one notebook-style combination run across multiple trend prompts."""

    combination_description: str
    context_prompt: str
    context_response: str
    trend_analysis_prompts: list[str] = Field(default_factory=list)
    trend_analysis_responses: list[str] = Field(default_factory=list)


def run_pipeline(inputs: PipelineInputs, prompts: PromptsConfig, adapter: LLMAdapter) -> PipelineResult:
    """Executes the notebook-equivalent workflow through pure Python components."""
    frame = load_simulation_csv(inputs.csv_path)
    inputs.output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plot_metric_bundles(
        frame=frame,
        bundles=[
            MetricPlotBundle(
                include_pattern=inputs.metric_pattern,
                title="Simulation trend",
                y_label="value",
            )
        ],
        output_dir=inputs.output_dir,
    )[0]

    stats_table = helpers.build_stats_table(frame=frame, include_pattern=inputs.metric_pattern)
    stats_markdown = helpers.build_stats_markdown(stats_table)
    stats_image_path = _write_stats_image_if_needed(
        stats_table=stats_table,
        output_dir=inputs.output_dir,
        include_pattern=inputs.metric_pattern,
        evidence_mode=inputs.evidence_mode,
    )
    context_prompt = _context_prompt(inputs, prompts)
    context = _invoke_adapter(adapter, model=inputs.model, prompt=context_prompt)
    trend_prompt = _build_trend_prompt(
        prompts=prompts,
        metric_description=inputs.metric_description,
        context=context,
        plot_description=inputs.plot_description,
        evidence_mode=inputs.evidence_mode,
        stats_markdown=stats_markdown,
    )
    image_b64 = _encode_image_for_evidence(
        evidence_mode=inputs.evidence_mode,
        plot_path=plot_path,
        stats_image_path=stats_image_path,
    )
    trend_raw = _invoke_adapter(adapter, model=inputs.model, prompt=trend_prompt, image_b64=image_b64)
    summarization_mode = _resolve_summarization_mode(
        skip_summarization=inputs.skip_summarization,
        requested_mode=inputs.summarization_mode,
    )
    score_on = _resolve_score_mode(
        requested_score_on=inputs.score_on,
        summarization_mode=summarization_mode,
    )
    trend_full, trend_summary = _summarize_report_text(trend_raw, mode=summarization_mode)

    full_scores = None
    summary_scores = None
    if summarization_mode in {"full", "both"} or score_on == "full":
        full_scores = score_summary(reference=context, candidate=trend_full)
    if trend_summary is not None and summarization_mode in {"summary", "both"} and score_on in {"summary", "both"}:
        summary_scores = score_summary(reference=context, candidate=trend_summary)

    if summarization_mode == "both" and score_on == "both":
        include_extended = True
    else:
        include_extended = False

    report_trend, selected_scores = _select_report_outputs(
        trend_full=trend_full,
        trend_summary=trend_summary,
        score_on=score_on,
        full_scores=full_scores,
        summary_scores=summary_scores,
    )
    report_csv = _write_report(
        output_dir=inputs.output_dir,
        context=context,
        trend_full=trend_full,
        trend_summary=trend_summary,
        selected_scores=selected_scores,
        full_scores=full_scores if include_extended else None,
        summary_scores=summary_scores if include_extended else None,
        include_extended_columns=include_extended,
    )

    return PipelineResult(
        plot_path=plot_path,
        report_csv=report_csv,
        context_response=context,
        trend_response=report_trend,
        trend_full_response=trend_full,
        trend_summary_response=trend_summary,
        full_scores=full_scores if include_extended else None,
        summary_scores=summary_scores if include_extended else None,
        stats_markdown=stats_markdown,
        stats_image_path=stats_image_path,
        token_f1=selected_scores.token_f1,
        bleu=selected_scores.bleu,
        meteor=selected_scores.meteor,
        rouge1=selected_scores.rouge1,
        rouge2=selected_scores.rouge2,
        rouge_l=selected_scores.rouge_l,
        flesch_reading_ease=selected_scores.flesch_reading_ease,
    )


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
    """Runs all style-feature prompt combinations and writes notebook-style wide CSV outputs."""
    if not image_paths:
        raise ValueError("image_paths cannot be empty")
    if len(image_paths) != len(plot_descriptions):
        raise ValueError("image_paths and plot_descriptions must have the same length")

    context_client = context_adapter or adapter
    trend_client = trend_adapter or adapter
    context_model_name = context_model or inputs.model
    trend_model_name = trend_model or inputs.model
    combinations_to_run = build_style_feature_combinations(prompts, style_feature_keys)
    rows: list[SweepRunResult] = []
    for description, enabled_features in combinations_to_run:
        context_prompt = _context_prompt(inputs, prompts, enabled_style_features=enabled_features)
        context_response = _invoke_adapter(context_client, model=context_model_name, prompt=context_prompt)
        trend_prompt_base = _build_trend_prompt(
            prompts=prompts,
            metric_description=inputs.metric_description,
            context=context_response,
            plot_description=None,
            evidence_mode="plot",
            stats_markdown="",
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
        rows.append(
            SweepRunResult(
                combination_description=description,
                context_prompt=context_prompt,
                context_response=context_response,
                trend_analysis_prompts=prompts_for_images,
                trend_analysis_responses=responses_for_images,
            )
        )

    out = output_csv or (inputs.output_dir / "combinations_report.csv")
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
    """Writes notebook-style wide CSV format with one prompt+response pair per trend image."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    max_items = max((len(row.trend_analysis_prompts) for row in rows), default=0)
    headers = _sweep_headers(max_items=max_items, csv_column_style=csv_column_style)
    if resume_existing:
        return _write_combinations_csv_resume(output_csv=output_csv, rows=rows, headers=headers)
    helpers.write_sweep_rows(output_csv=output_csv, rows=rows, headers=headers)
    return output_csv


def _sweep_headers(max_items: int, csv_column_style: SweepCsvColumnStyle) -> list[str]:
    return helpers.sweep_headers(max_items=max_items, csv_column_style=csv_column_style)


def _write_combinations_csv_resume(output_csv: Path, rows: list[SweepRunResult], headers: list[str]) -> Path:
    helpers.write_sweep_rows_resume(output_csv=output_csv, rows=rows, headers=headers)
    return output_csv


def build_style_feature_combinations(
    prompts: PromptsConfig,
    style_feature_keys: list[str] | None = None,
) -> list[tuple[str, set[str]]]:
    """Builds notebook-style all-combination feature sets (role/example/insights by default)."""
    requested = style_feature_keys or ["role", "example", "insights"]
    available = [key for key in requested if prompts.style_features.get(key, "").strip()]
    combinations_to_run: list[tuple[str, set[str]]] = [("None", set())]
    for size in range(1, len(available) + 1):
        for combo in combinations(available, size):
            combinations_to_run.append((" + ".join(combo), set(combo)))
    return combinations_to_run


def _context_prompt(
    inputs: PipelineInputs,
    prompts: PromptsConfig,
    enabled_style_features: set[str] | None = None,
) -> str:
    return helpers.build_context_prompt(
        inputs_csv_path=inputs.parameters_path,
        inputs_doc_path=inputs.documentation_path,
        prompts=prompts,
        enabled=enabled_style_features,
    )


def _invoke_adapter(adapter: LLMAdapter, model: str, prompt: str, image_b64: str | None = None) -> str:
    return helpers.invoke_adapter(adapter=adapter, model=model, prompt=prompt, image_b64=image_b64)


def _encode_image(path: Path) -> str:
    return helpers.encode_image(path)


def _summarize_report_text(text: str, mode: SummarizationMode) -> tuple[str, str | None]:
    if mode == "full":
        return text, None
    _, summary = helpers.summarize_report_text_pair(
        text=text,
        skip_summarization=False,
        summarize_with_bart_fn=summarize_with_bart,
        summarize_with_bert_fn=summarize_with_bert,
    )
    if mode == "summary":
        return text, summary
    return text, summary


def _resolve_summarization_mode(
    skip_summarization: bool,
    requested_mode: SummarizationMode,
) -> SummarizationMode:
    if skip_summarization:
        return "full"
    return requested_mode


def _resolve_score_mode(
    requested_score_on: ScoreMode,
    summarization_mode: SummarizationMode,
) -> ScoreMode:
    if requested_score_on == "summary" and summarization_mode == "full":
        return "full"
    if requested_score_on == "both" and summarization_mode == "full":
        return "full"
    if requested_score_on == "both" and summarization_mode != "both":
        return "summary" if summarization_mode == "summary" else "full"
    return requested_score_on


def _select_report_outputs(
    trend_full: str,
    trend_summary: str | None,
    score_on: ScoreMode,
    full_scores: SummaryScores | None,
    summary_scores: SummaryScores | None,
) -> tuple[str, SummaryScores]:
    if score_on == "full" or summary_scores is None:
        if full_scores is None:
            return trend_full, score_summary(reference=trend_full, candidate=trend_full)
        return trend_full, full_scores
    if score_on == "summary":
        return trend_summary or trend_full, summary_scores
    return trend_summary or trend_full, summary_scores


def _write_report(
    output_dir: Path,
    context: str,
    trend_full: str,
    trend_summary: str | None,
    selected_scores: SummaryScores,
    full_scores: SummaryScores | None = None,
    summary_scores: SummaryScores | None = None,
    include_extended_columns: bool = False,
) -> Path:
    return helpers.write_report(
        output_dir=output_dir,
        context=context,
        trend_full=trend_full,
        trend_summary=trend_summary,
        scores=selected_scores,
        full_scores=full_scores,
        summary_scores=summary_scores,
        include_extended_columns=include_extended_columns,
    )


def _build_trend_prompt(
    prompts: PromptsConfig,
    metric_description: str,
    context: str,
    plot_description: str | None,
    evidence_mode: EvidenceMode,
    stats_markdown: str,
    enabled_style_features: set[str] | None = None,
) -> str:
    return helpers.build_trend_prompt(
        prompts=prompts,
        metric_description=metric_description,
        context=context,
        plot_description=plot_description,
        evidence_mode=evidence_mode,
        stats_markdown=stats_markdown,
        enabled=enabled_style_features,
    )


def _write_stats_image_if_needed(
    stats_table: pd.DataFrame,
    output_dir: Path,
    include_pattern: str,
    evidence_mode: EvidenceMode,
) -> Path | None:
    return helpers.write_stats_image_if_needed(
        stats_table=stats_table,
        output_dir=output_dir,
        include_pattern=include_pattern,
        evidence_mode=evidence_mode,
    )


def _encode_image_for_evidence(
    evidence_mode: EvidenceMode,
    plot_path: Path,
    stats_image_path: Path | None,
) -> str | None:
    return helpers.encode_evidence_image(
        evidence_mode=evidence_mode,
        plot_path=plot_path,
        stats_image_path=stats_image_path,
    )


def _append_plot_description(base_prompt: str, plot_description: str) -> str:
    return helpers.append_plot_description(base_prompt=base_prompt, plot_description=plot_description)
