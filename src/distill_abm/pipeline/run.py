"""Pipeline orchestration from CSV ingestion to scored report export."""

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from distill_abm.configs.models import PromptsConfig
from distill_abm.configs.runtime_defaults import get_runtime_defaults
from distill_abm.eval.metrics import SummaryScores, score_summary
from distill_abm.ingest.csv_ingest import load_simulation_csv
from distill_abm.llm.adapters.base import LLMAdapter
from distill_abm.pipeline import helpers
from distill_abm.summarize.models import (
    summarize_with_bart,
    summarize_with_bert,
    summarize_with_longformer_ext,
    summarize_with_t5,
)
from distill_abm.viz.plots import MetricPlotBundle, plot_metric_bundles

EvidenceMode = Literal["plot", "table-csv", "plot+table", "stats-markdown", "stats-image", "plot+stats"]
ResolvedEvidenceMode = Literal["plot", "table-csv", "plot+table"]
SummarizationMode = Literal["full", "summary", "both"]
ScoreMode = Literal["full", "summary", "both"]
SweepCsvColumnStyle = Literal["trend", "plot"]
AdditionalSummarizer = Literal["t5", "longformer_ext"]


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
    additional_summarizers: tuple[AdditionalSummarizer, ...] = ()
    scoring_reference_path: Path | None = None
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
    """Represents one notebook-style combination run across multiple trend prompts."""

    combination_description: str
    context_prompt: str
    context_response: str
    trend_analysis_prompts: list[str] = Field(default_factory=list)
    trend_analysis_responses: list[str] = Field(default_factory=list)


def run_pipeline(inputs: PipelineInputs, prompts: PromptsConfig, adapter: LLMAdapter) -> PipelineResult:
    """Executes the notebook-equivalent workflow through pure Python components."""
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    run_signature = _build_run_signature(inputs=inputs, prompts=prompts, adapter=adapter)
    if inputs.resume_existing:
        resumed = _load_resumable_pipeline_result(output_dir=inputs.output_dir, run_signature=run_signature)
        if resumed is not None:
            return resumed

    frame = load_simulation_csv(inputs.csv_path)

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
    context_prompt = _context_prompt(inputs, prompts)
    context = _invoke_adapter(adapter, model=inputs.model, prompt=context_prompt)
    trend_prompt = _build_trend_prompt(
        prompts=prompts,
        metric_description=inputs.metric_description,
        context=context,
        plot_description=inputs.plot_description,
        evidence_mode=resolved_evidence_mode,
        stats_table_csv=stats_table_csv,
    )
    image_b64 = _encode_image_for_evidence(
        evidence_mode=resolved_evidence_mode,
        plot_path=plot_path,
        stats_image_path=stats_image_path,
    )
    trend_raw = _invoke_adapter(adapter, model=inputs.model, prompt=trend_prompt, image_b64=image_b64)
    summarization_mode = _resolve_summarization_mode(
        skip_summarization=inputs.skip_summarization,
        requested_mode=inputs.summarization_mode,
    )
    scoring_reference_text, scoring_reference_source, scoring_reference_path = _resolve_scoring_reference(
        inputs=inputs,
        context=context,
    )
    score_on = _resolve_score_mode(
        requested_score_on=inputs.score_on,
        summarization_mode=summarization_mode,
    )

    # Compute full and/or summary trend variants first, then apply scoring policy.
    trend_full, trend_summary = _summarize_report_text(
        trend_raw,
        mode=summarization_mode,
        additional_summarizers=inputs.additional_summarizers,
    )

    full_scores = None
    summary_scores = None
    if summarization_mode in {"full", "both"} or score_on == "full":
        full_scores = score_summary(reference=scoring_reference_text, candidate=trend_full)
    if trend_summary is not None and summarization_mode in {"summary", "both"} and score_on in {"summary", "both"}:
        summary_scores = score_summary(reference=scoring_reference_text, candidate=trend_summary)

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
        summarization_mode=summarization_mode,
        score_on=score_on,
        full_scores=full_scores,
        summary_scores=summary_scores,
        selected_scores=selected_scores,
        scoring_reference_text=scoring_reference_text,
        scoring_reference_source=scoring_reference_source,
        scoring_reference_path=scoring_reference_path,
        include_extended=include_extended,
        include_pattern=inputs.metric_pattern,
        evidence_mode=resolved_evidence_mode,
        requested_evidence_mode=inputs.evidence_mode,
        adapter=adapter,
        trend_image_attached=image_b64 is not None,
        run_signature=run_signature,
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


def _build_run_signature(inputs: PipelineInputs, prompts: PromptsConfig, adapter: LLMAdapter) -> str:
    runtime_defaults = get_runtime_defaults()
    payload = {
        "inputs": {
            "csv_path": str(inputs.csv_path.resolve()),
            "parameters_path": str(inputs.parameters_path.resolve()),
            "documentation_path": str(inputs.documentation_path.resolve()),
            "metric_pattern": inputs.metric_pattern,
            "metric_description": inputs.metric_description,
            "plot_description": inputs.plot_description,
            "skip_summarization": inputs.skip_summarization,
            "summarization_mode": inputs.summarization_mode,
            "score_on": inputs.score_on,
            "evidence_mode": inputs.evidence_mode,
            "additional_summarizers": list(inputs.additional_summarizers),
            "scoring_reference_path": (
                str(inputs.scoring_reference_path.resolve()) if inputs.scoring_reference_path is not None else None
            ),
        },
        "input_file_hashes": {
            "csv": _hash_file(inputs.csv_path),
            "parameters": _hash_file(inputs.parameters_path),
            "documentation": _hash_file(inputs.documentation_path),
            "scoring_reference": _hash_file(inputs.scoring_reference_path),
        },
        "llm": {
            "provider": adapter.provider,
            "model": inputs.model,
            "temperature": runtime_defaults.llm_request.temperature,
            "max_tokens": runtime_defaults.llm_request.max_tokens,
        },
        "prompts": prompts.model_dump(mode="json"),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _hash_file(path: Path | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_stats_table_csv(output_dir: Path, stats_table_csv: str) -> Path:
    path = output_dir / "stats_table.csv"
    path.write_text(stats_table_csv, encoding="utf-8")
    return path


def _load_resumable_pipeline_result(output_dir: Path, run_signature: str) -> PipelineResult | None:
    metadata_path = output_dir / "pipeline_run_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    reproducibility = payload.get("reproducibility", {})
    if str(reproducibility.get("run_signature", "")) != run_signature:
        return None

    artifacts = payload.get("artifacts", {})
    scores_payload = payload.get("scores", {})
    responses = payload.get("responses", {})

    try:
        plot_path = Path(str(artifacts["plot_path"]))
        report_csv = Path(str(artifacts["report_csv"]))
        if not plot_path.exists() or not report_csv.exists():
            return None

        stats_image_raw = artifacts.get("stats_image_path")
        stats_image_path = Path(str(stats_image_raw)) if isinstance(stats_image_raw, str) and stats_image_raw else None
        if stats_image_path is not None and not stats_image_path.exists():
            stats_image_path = None

        stats_table_csv: str | None = None
        stats_table_csv_raw = artifacts.get("stats_table_csv_path")
        if isinstance(stats_table_csv_raw, str) and stats_table_csv_raw:
            stats_table_csv_path = Path(stats_table_csv_raw)
            if stats_table_csv_path.exists():
                stats_table_csv = stats_table_csv_path.read_text(encoding="utf-8")

        selected_scores = SummaryScores.model_validate(scores_payload["selected_scores"])
        full_scores_raw = scores_payload.get("full_scores")
        summary_scores_raw = scores_payload.get("summary_scores")
        full_scores = SummaryScores.model_validate(full_scores_raw) if full_scores_raw else None
        summary_scores = SummaryScores.model_validate(summary_scores_raw) if summary_scores_raw else None

        trend_full = str(responses.get("trend_full_response", ""))
        trend_summary_raw = responses.get("trend_summary_response")
        trend_summary = str(trend_summary_raw) if isinstance(trend_summary_raw, str) else None
        trend_response = trend_summary if trend_summary is not None else trend_full

        return PipelineResult(
            plot_path=plot_path,
            report_csv=report_csv,
            context_response=str(responses.get("context_response", "")),
            trend_response=trend_response,
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
    except (KeyError, TypeError, ValueError):
        return None


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
    context_prompt = _context_prompt(inputs, prompts, enabled_style_features=enabled_features)
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


def _summarize_report_text(
    text: str,
    mode: SummarizationMode,
    additional_summarizers: tuple[AdditionalSummarizer, ...] = (),
) -> tuple[str, str | None]:
    """Resolve and generate trend text variants based on the selected summarization mode."""
    if mode == "full":
        return text, None
    extra = _resolve_additional_summarizer_runners(additional_summarizers)
    _, summary = helpers.summarize_report_text_pair(
        text=text,
        skip_summarization=False,
        summarize_with_bart_fn=summarize_with_bart,
        summarize_with_bert_fn=summarize_with_bert,
        additional_summarizers=extra,
    )
    if mode == "summary":
        return text, summary
    return text, summary


def _resolve_additional_summarizer_runners(
    additional_summarizers: tuple[AdditionalSummarizer, ...],
) -> list[tuple[str, Callable[[str], str]]]:
    runners: list[tuple[str, Callable[[str], str]]] = []
    for name in additional_summarizers:
        if name == "t5":
            runners.append(("t5", summarize_with_t5))
        elif name == "longformer_ext":
            runners.append(("longformer_ext", summarize_with_longformer_ext))
    return runners


def _resolve_summarization_mode(
    skip_summarization: bool,
    requested_mode: SummarizationMode,
) -> SummarizationMode:
    """Skip summarization requests force full-text trend mode."""
    if skip_summarization:
        return "full"
    return requested_mode


def _resolve_score_mode(
    requested_score_on: ScoreMode,
    summarization_mode: SummarizationMode,
) -> ScoreMode:
    """Resolve score mode to a valid combination given unavailable summary path."""
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
    """Select the trend text and score tuple that the report writer should persist."""
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


def _write_run_metadata(
    output_dir: Path,
    inputs: PipelineInputs,
    context_prompt: str,
    plot_path: Path,
    report_csv: Path,
    stats_table_csv_path: Path,
    stats_image_path: Path | None,
    trend_prompt: str,
    context_response: str,
    trend_full: str,
    trend_summary: str | None,
    summarization_mode: SummarizationMode,
    score_on: ScoreMode,
    full_scores: SummaryScores | None,
    summary_scores: SummaryScores | None,
    include_extended: bool,
    scoring_reference_text: str,
    scoring_reference_source: str,
    scoring_reference_path: Path | None,
    include_pattern: str,
    evidence_mode: ResolvedEvidenceMode,
    requested_evidence_mode: EvidenceMode,
    adapter: LLMAdapter,
    selected_scores: SummaryScores,
    trend_image_attached: bool,
    run_signature: str,
) -> Path:
    """Persist deterministic run metadata for reproducibility and auditability."""
    runtime_defaults = get_runtime_defaults()
    default_temperature = runtime_defaults.llm_request.temperature
    default_max_tokens = runtime_defaults.llm_request.max_tokens
    metadata = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "csv_path": str(inputs.csv_path),
            "parameters_path": str(inputs.parameters_path),
            "documentation_path": str(inputs.documentation_path),
            "metric_pattern": inputs.metric_pattern,
            "metric_description": inputs.metric_description,
            "plot_description": inputs.plot_description,
            "evidence_mode": evidence_mode,
            "evidence_mode_requested": requested_evidence_mode,
            "summarization_mode": summarization_mode,
            "score_on": score_on,
            "skip_summarization": inputs.skip_summarization,
            "summarization_mode_requested": inputs.summarization_mode,
            "score_on_requested": inputs.score_on,
            "additional_summarizers": list(inputs.additional_summarizers),
            "output_dir": str(inputs.output_dir),
            "scoring_reference_path": str(scoring_reference_path) if scoring_reference_path is not None else None,
            "resume_existing": inputs.resume_existing,
        },
        "artifacts": {
            "plot_path": str(plot_path),
            "report_csv": str(report_csv),
            "stats_table_csv_path": str(stats_table_csv_path),
            "stats_image_path": str(stats_image_path) if stats_image_path is not None else None,
            "trend_evidence_image_path": (
                str(plot_path)
                if evidence_mode in {"plot", "plot+table"}
                else (str(stats_image_path) if stats_image_path else None)
            ),
        },
        "llm": {
            "provider": adapter.provider,
            "model": inputs.model,
            "request": {
                "temperature": default_temperature,
                "max_tokens": default_max_tokens,
            },
            "requests": {
                "context": {
                    "image_attached": False,
                    "temperature": default_temperature,
                    "max_tokens": default_max_tokens,
                },
                "trend": {
                    "image_attached": trend_image_attached,
                    "temperature": default_temperature,
                    "max_tokens": default_max_tokens,
                },
            },
        },
        "prompts": {
            "context_prompt": context_prompt,
            "trend_prompt": trend_prompt,
        },
        "responses": {
            "context_response": context_response,
            "trend_full_response": trend_full,
            "trend_summary_response": trend_summary,
        },
        "reproducibility": {
            "include_pattern": include_pattern,
            "context_prompt_signature": hashlib.sha256(context_prompt.encode("utf-8")).hexdigest(),
            "trend_prompt_signature": hashlib.sha256(trend_prompt.encode("utf-8")).hexdigest(),
            "context_prompt_length": len(context_prompt),
            "trend_prompt_length": len(trend_prompt),
            "trend_full_length": len(trend_full),
            "trend_summary_present": trend_summary is not None,
            "trend_summary_length": len(trend_summary) if trend_summary is not None else None,
            "include_extended_columns": include_extended,
            "csv_encoding": "utf-8",
            "delimiter": ",",
            "run_signature": run_signature,
        },
        "summarizers": {
            "bart": {
                "model": "sshleifer/distilbart-cnn-12-6",
                "max_input_length": 1024,
                "min_summary_length": 50,
                "max_summary_length": 100,
                "enabled": True,
            },
            "bert": {
                "max_input_length": 512,
                "min_summary_length": 100,
                "max_summary_length": 150,
                "enabled": True,
            },
            "t5": {
                "model": "t5-small",
                "max_input_length": 1024,
                "min_summary_length": 40,
                "max_summary_length": 120,
                "enabled": "t5" in inputs.additional_summarizers,
            },
            "longformer_like": {
                "model": "allenai/led-base-16384",
                "max_input_length": 2048,
                "min_summary_length": 64,
                "max_summary_length": 180,
                "enabled": "longformer_ext" in inputs.additional_summarizers,
            },
        },
        "scores": {
            "selected_scores": selected_scores.model_dump(),
            "full_scores": full_scores.model_dump() if full_scores is not None else None,
            "summary_scores": summary_scores.model_dump() if summary_scores is not None else None,
            "reference": {
                "source": scoring_reference_source,
                "path": str(scoring_reference_path) if scoring_reference_path is not None else None,
                "length": len(scoring_reference_text),
                "signature": hashlib.sha256(scoring_reference_text.encode("utf-8")).hexdigest(),
            },
        },
    }

    metadata_path = output_dir / "pipeline_run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata_path


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
