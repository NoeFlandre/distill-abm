"""Pipeline orchestration from CSV ingestion to scored report export."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from distill_abm.configs.models import PromptsConfig, SummarizerId
from distill_abm.eval.metrics import SummaryScores, score_summary
from distill_abm.ingest.csv_ingest import load_simulation_csv
from distill_abm.llm.adapters.base import LLMAdapter
from distill_abm.pipeline import helpers, run_sweep
from distill_abm.pipeline.run_state import (
    _build_run_signature,
    _load_resumable_pipeline_result,
    _write_run_metadata,
)
from distill_abm.viz.plots import MetricPlotBundle, plot_metric_bundles

EvidenceMode = Literal["plot", "table", "plot+table"]
ResolvedEvidenceMode = Literal["plot", "table", "plot+table"]
TextSourceMode = Literal["summary_only", "full_text_only"]
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
    text_source_mode: TextSourceMode = "summary_only"
    evidence_mode: EvidenceMode = "plot+table"
    summarizers: tuple[SummarizerId, ...] = ("bart", "bert", "t5", "longformer_ext")
    allow_summary_fallback: bool = False
    enabled_style_features: tuple[str, ...] | None = None
    scoring_reference_path: Path | None = None
    additional_scoring_reference_paths: dict[str, Path] = Field(default_factory=dict)
    resume_existing: bool = False


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
    stats_table_csv: str | None = None
    stats_image_path: Path | None = None
    token_f1: float
    bleu: float
    meteor: float
    rouge1: float
    rouge2: float
    rouge_l: float
    flesch_reading_ease: float
    metadata_path: Path | None = None


class SweepRunResult(BaseModel):
    """Represents one style-factor combination run across multiple trend prompts."""

    combination_description: str
    context_prompt: str
    context_response: str
    trend_analysis_prompts: list[str] = Field(default_factory=list)
    trend_analysis_responses: list[str] = Field(default_factory=list)


# Backward-compatible re-exports for integration tests and older call sites.
build_style_feature_combinations = run_sweep.build_style_feature_combinations
run_pipeline_sweep = run_sweep.run_pipeline_sweep
write_combinations_csv = run_sweep.write_combinations_csv
helpers = helpers

__all__ = [
    "EvidenceMode",
    "PipelineInputs",
    "PipelineResult",
    "ResolvedEvidenceMode",
    "SweepCsvColumnStyle",
    "SweepRunResult",
    "TextSourceMode",
    "_build_run_signature",
    "_load_resumable_pipeline_result",
    "_write_run_metadata",
    "build_style_feature_combinations",
    "helpers",
    "run_pipeline",
    "run_pipeline_sweep",
    "write_combinations_csv",
]


def run_pipeline(inputs: PipelineInputs, prompts: PromptsConfig, adapter: LLMAdapter) -> PipelineResult:
    """Execute one end-to-end paper-aligned workflow through pure Python components."""
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    run_signature = _build_run_signature(inputs=inputs, prompts=prompts, adapter=adapter)
    if inputs.resume_existing:
        resumed = _load_resumable_pipeline_result(output_dir=inputs.output_dir, run_signature=run_signature)
        if resumed is not None:
            return resumed

    frame = load_simulation_csv(inputs.csv_path)
    matched_metric_columns = [str(column) for column in frame.columns if inputs.metric_pattern in str(column)]

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

    resolved_evidence_mode = _resolve_evidence_mode(inputs.evidence_mode)
    stats_table = helpers.build_stats_table(frame=frame, include_pattern=inputs.metric_pattern)
    stats_table_csv = helpers.build_stats_csv(stats_table)
    stats_table_csv_path = _write_stats_table_csv(output_dir=inputs.output_dir, stats_table_csv=stats_table_csv)
    stats_image_path = _write_stats_image_if_needed(
        stats_table=stats_table,
        output_dir=inputs.output_dir,
        include_pattern=inputs.metric_pattern,
        evidence_mode=resolved_evidence_mode,
    )
    enabled_style_features = _enabled_style_features(inputs.enabled_style_features)
    context_prompt = _context_prompt(inputs, prompts, enabled_style_features=enabled_style_features)
    context, context_trace = _invoke_adapter_with_trace(adapter, model=inputs.model, prompt=context_prompt)
    trend_prompt = _build_trend_prompt(
        prompts=prompts,
        metric_description=inputs.metric_description,
        context=context,
        plot_description=inputs.plot_description,
        evidence_mode=resolved_evidence_mode,
        stats_table_csv=stats_table_csv,
        enabled_style_features=enabled_style_features,
    )
    image_b64 = _encode_image_for_evidence(
        evidence_mode=resolved_evidence_mode,
        plot_path=plot_path,
        stats_image_path=stats_image_path,
    )
    trend_raw, trend_trace = _invoke_adapter_with_trace(
        adapter,
        model=inputs.model,
        prompt=trend_prompt,
        image_b64=image_b64,
    )
    scoring_reference_text, scoring_reference_source, scoring_reference_path = _resolve_scoring_reference(
        inputs=inputs,
        context=context,
    )
    additional_scoring_references = _resolve_additional_scoring_references(inputs=inputs)
    trend_full, trend_summary = _summarize_report_text(
        text=trend_raw,
        text_source_mode=inputs.text_source_mode,
        summarizers=inputs.summarizers,
        allow_summary_fallback=inputs.allow_summary_fallback,
    )
    selected_text_source: TextSourceMode = "summary_only" if trend_summary is not None else "full_text_only"
    report_trend = trend_summary if trend_summary is not None else trend_full
    summarization_trace: dict[str, object] = {
        "text_source_mode": inputs.text_source_mode,
        "selected_text_source": selected_text_source,
        "allow_summary_fallback": inputs.allow_summary_fallback,
        "requested_summarizers": list(inputs.summarizers),
        "trend_raw_text": trend_raw,
        "trend_raw_length": len(trend_raw),
        "trend_full_text": trend_full,
        "trend_full_length": len(trend_full),
        "trend_summary_text": trend_summary,
        "trend_summary_length": len(trend_summary) if trend_summary is not None else None,
    }

    selected_scores = score_summary(reference=scoring_reference_text, candidate=report_trend)
    full_scores = score_summary(reference=scoring_reference_text, candidate=trend_full)
    summary_scores = (
        score_summary(reference=scoring_reference_text, candidate=trend_summary) if trend_summary is not None else None
    )
    additional_reference_scores = _score_additional_references(
        references=additional_scoring_references,
        selected_candidate=report_trend,
        full_candidate=trend_full,
        summary_candidate=trend_summary,
    )

    include_extended = trend_summary is not None
    report_csv = _write_report(
        output_dir=inputs.output_dir,
        context=context,
        trend_full=trend_full,
        trend_summary=trend_summary,
        selected_scores=selected_scores,
        full_scores=full_scores,
        summary_scores=summary_scores,
        additional_reference_scores=additional_reference_scores,
        include_extended_columns=include_extended,
    )

    # Persist resolved runtime settings, prompt composition, and scoring settings for auditability.
    metadata_path = _write_run_metadata(
        output_dir=inputs.output_dir,
        inputs=inputs,
        context_prompt=context_prompt,
        plot_path=plot_path,
        report_csv=report_csv,
        stats_table_csv_path=stats_table_csv_path,
        stats_image_path=stats_image_path,
        trend_prompt=trend_prompt,
        context_response=context,
        trend_full=trend_full,
        trend_summary=trend_summary,
        selected_text_source=selected_text_source,
        allow_summary_fallback=inputs.allow_summary_fallback,
        full_scores=full_scores,
        summary_scores=summary_scores,
        selected_scores=selected_scores,
        scoring_reference_text=scoring_reference_text,
        scoring_reference_source=scoring_reference_source,
        scoring_reference_path=scoring_reference_path,
        additional_scoring_references=additional_scoring_references,
        additional_reference_scores=additional_reference_scores,
        include_extended=include_extended,
        include_pattern=inputs.metric_pattern,
        evidence_mode=resolved_evidence_mode,
        requested_evidence_mode=inputs.evidence_mode,
        adapter=adapter,
        trend_image_attached=image_b64 is not None,
        run_signature=run_signature,
        context_trace=context_trace,
        trend_trace=trend_trace,
        summarization_trace=summarization_trace,
        frame_summary={
            "row_count": len(frame),
            "column_count": len(frame.columns),
            "columns": [str(column) for column in frame.columns],
            "matched_metric_columns": matched_metric_columns,
        },
    )

    return PipelineResult(
        plot_path=plot_path,
        report_csv=report_csv,
        context_response=context,
        trend_response=report_trend,
        trend_full_response=trend_full,
        trend_summary_response=trend_summary,
        full_scores=full_scores,
        summary_scores=summary_scores,
        stats_table_csv=stats_table_csv,
        stats_image_path=stats_image_path,
        token_f1=selected_scores.token_f1,
        bleu=selected_scores.bleu,
        meteor=selected_scores.meteor,
        rouge1=selected_scores.rouge1,
        rouge2=selected_scores.rouge2,
        rouge_l=selected_scores.rouge_l,
        flesch_reading_ease=selected_scores.flesch_reading_ease,
        metadata_path=metadata_path,
    )


def _write_stats_table_csv(output_dir: Path, stats_table_csv: str) -> Path:
    path = output_dir / "stats_table.csv"
    path.write_text(stats_table_csv, encoding="utf-8")
    return path


def _invoke_adapter(adapter: LLMAdapter, model: str, prompt: str, image_b64: str | None = None) -> str:
    return helpers.invoke_adapter(adapter=adapter, model=model, prompt=prompt, image_b64=image_b64)


def _invoke_adapter_with_trace(
    adapter: LLMAdapter,
    model: str,
    prompt: str,
    image_b64: str | None = None,
) -> tuple[str, dict[str, object]]:
    return helpers.invoke_adapter_with_trace(adapter=adapter, model=model, prompt=prompt, image_b64=image_b64)


def _enabled_style_features(values: tuple[str, ...] | None) -> set[str] | None:
    if values is None:
        return None
    return {value.strip() for value in values if value.strip()}


def _encode_image(path: Path) -> str:
    return helpers.encode_image(path)


def _summarize_report_text(
    text: str,
    text_source_mode: TextSourceMode,
    summarizers: tuple[SummarizerId, ...],
    allow_summary_fallback: bool,
) -> tuple[str, str | None]:
    """Resolve and generate trend text variants based on selected text-source mode."""
    return helpers.summarize_report_text_pair_for_ids(
        text=text,
        skip_summarization=text_source_mode == "full_text_only",
        summarizer_ids=summarizers,
        allow_fallback=allow_summary_fallback,
    )


def _write_report(
    output_dir: Path,
    context: str,
    trend_full: str,
    trend_summary: str | None,
    selected_scores: SummaryScores,
    full_scores: SummaryScores | None = None,
    summary_scores: SummaryScores | None = None,
    include_extended_columns: bool = False,
    additional_reference_scores: Mapping[str, Mapping[str, SummaryScores | None]] | None = None,
) -> Path:
    return helpers.write_report(
        output_dir=output_dir,
        context=context,
        trend_full=trend_full,
        trend_summary=trend_summary,
        scores=selected_scores,
        full_scores=full_scores,
        summary_scores=summary_scores,
        additional_reference_scores=additional_reference_scores,
        include_extended_columns=include_extended_columns,
    )


def _context_prompt(
    inputs: PipelineInputs,
    prompts: PromptsConfig,
    enabled_style_features: set[str] | None = None,
) -> str:
    """Build the context prompt using optional style-feature injection."""
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
    evidence_mode: ResolvedEvidenceMode,
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


def _write_stats_image_if_needed(
    stats_table: pd.DataFrame,
    output_dir: Path,
    include_pattern: str,
    evidence_mode: ResolvedEvidenceMode,
) -> Path | None:
    return helpers.write_stats_image_if_needed(
        stats_table=stats_table,
        output_dir=output_dir,
        include_pattern=include_pattern,
        evidence_mode=evidence_mode,
    )


def _encode_image_for_evidence(
    evidence_mode: ResolvedEvidenceMode,
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


def _resolve_evidence_mode(evidence_mode: EvidenceMode) -> ResolvedEvidenceMode:
    return helpers.resolve_evidence_mode(evidence_mode)


def _resolve_scoring_reference(inputs: PipelineInputs, context: str) -> tuple[str, str, Path | None]:
    if inputs.scoring_reference_path is None:
        return context, "context_response", None
    reference_text = inputs.scoring_reference_path.read_text(encoding="utf-8").strip()
    if not reference_text:
        raise ValueError(f"scoring reference file is empty: {inputs.scoring_reference_path}")
    return reference_text, "human_ground_truth_file", inputs.scoring_reference_path


def _resolve_additional_scoring_references(inputs: PipelineInputs) -> dict[str, tuple[str, str, Path]]:
    references: dict[str, tuple[str, str, Path]] = {}
    for reference_name, reference_path in sorted(inputs.additional_scoring_reference_paths.items()):
        reference_text = reference_path.read_text(encoding="utf-8").strip()
        if not reference_text:
            raise ValueError(f"additional scoring reference file is empty: {reference_path}")
        references[reference_name] = (reference_text, "human_ground_truth_file", reference_path)
    return references


def _score_additional_references(
    references: Mapping[str, tuple[str, str, Path]],
    selected_candidate: str,
    full_candidate: str,
    summary_candidate: str | None,
) -> dict[str, dict[str, SummaryScores | None]]:
    scored_references: dict[str, dict[str, SummaryScores | None]] = {}
    for reference_name, (reference_text, _reference_source, _reference_path) in references.items():
        scored_references[reference_name] = {
            "selected_scores": score_summary(reference=reference_text, candidate=selected_candidate),
            "full_scores": score_summary(reference=reference_text, candidate=full_candidate),
            "summary_scores": (
                score_summary(reference=reference_text, candidate=summary_candidate)
                if summary_candidate is not None
                else None
            ),
        }
    return scored_references
