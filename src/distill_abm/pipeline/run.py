"""Pipeline orchestration from CSV ingestion to scored report export."""

from __future__ import annotations

import base64
import csv
from itertools import combinations
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from distill_abm.configs.models import PromptsConfig
from distill_abm.eval.metrics import SummaryScores, score_summary
from distill_abm.ingest.csv_ingest import load_simulation_csv
from distill_abm.llm.adapters.base import LLMAdapter, LLMMessage, LLMRequest
from distill_abm.summarize.models import summarize_with_bart, summarize_with_bert
from distill_abm.summarize.text import clean_markdown_symbols, strip_think_prefix
from distill_abm.viz.plots import (
    MetricPlotBundle,
    generate_stats_table,
    plot_metric_bundles,
    render_stats_table_image,
    render_stats_table_markdown,
)

EvidenceMode = Literal["plot", "stats-markdown", "stats-image", "plot+stats"]
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
    evidence_mode: EvidenceMode = "plot"


class PipelineResult(BaseModel):
    """Returns generated artifacts to callers without leaking internal state."""

    plot_path: Path
    report_csv: Path
    context_response: str
    trend_response: str
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
    stats_table = generate_stats_table(frame, include_pattern=inputs.metric_pattern)
    stats_markdown = render_stats_table_markdown(stats_table)
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
    trend = _summarize_report_text(trend_raw, skip_summarization=inputs.skip_summarization)
    scores = score_summary(reference=context, candidate=trend)
    report_csv = _write_report(inputs.output_dir, context, trend, scores)
    return PipelineResult(
        plot_path=plot_path,
        report_csv=report_csv,
        context_response=context,
        trend_response=trend,
        stats_markdown=stats_markdown,
        stats_image_path=stats_image_path,
        token_f1=scores.token_f1,
        bleu=scores.bleu,
        meteor=scores.meteor,
        rouge1=scores.rouge1,
        rouge2=scores.rouge2,
        rouge_l=scores.rouge_l,
        flesch_reading_ease=scores.flesch_reading_ease,
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
    """Writes notebook-style wide CSV format with one prompt+response column pair per trend image."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    max_items = max((len(row.trend_analysis_prompts) for row in rows), default=0)
    headers = _sweep_headers(max_items=max_items, csv_column_style=csv_column_style)
    if resume_existing:
        return _write_combinations_csv_resume(output_csv=output_csv, rows=rows, headers=headers)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(_row_to_record(row))
    return output_csv


def _sweep_headers(max_items: int, csv_column_style: SweepCsvColumnStyle) -> list[str]:
    headers = ["Combination Description", "Context Prompt", "Context Response"]
    for index in range(1, max_items + 1):
        if csv_column_style == "plot":
            headers.append(f"Plot {index} Prompt")
            headers.append(f"Plot {index} Analysis")
        else:
            headers.append(f"Trend Analysis Prompt {index}")
            headers.append(f"Trend Analysis Response {index}")
    return headers


def _row_to_record(row: SweepRunResult) -> list[str]:
    record: list[str] = [
        row.combination_description,
        row.context_prompt,
        row.context_response,
    ]
    for prompt, response in zip(row.trend_analysis_prompts, row.trend_analysis_responses, strict=True):
        record.extend([prompt, response])
    return record


def _write_combinations_csv_resume(output_csv: Path, rows: list[SweepRunResult], headers: list[str]) -> Path:
    existing = _load_existing_rows_if_compatible(output_csv, headers)
    for sweep_row in rows:
        new_record = _row_to_record(sweep_row)
        merged = existing.get(sweep_row.combination_description, [""] * len(headers))
        if len(merged) < len(headers):
            merged = merged + [""] * (len(headers) - len(merged))
        merged[0] = sweep_row.combination_description
        if len(new_record) > 1 and not merged[1]:
            merged[1] = new_record[1]
        if len(new_record) > 2 and not merged[2]:
            merged[2] = new_record[2]
        for index in range(3, len(headers), 2):
            next_index = index + 1
            if next_index >= len(headers):
                break
            source_prompt = new_record[index] if index < len(new_record) else ""
            source_response = new_record[next_index] if next_index < len(new_record) else ""
            if (not merged[index]) and (not merged[next_index]) and (source_prompt or source_response):
                merged[index] = source_prompt
                merged[next_index] = source_response
        existing[sweep_row.combination_description] = merged

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for key in existing:
            writer.writerow(existing[key])
    return output_csv


def _load_existing_rows_if_compatible(output_csv: Path, headers: list[str]) -> dict[str, list[str]]:
    if not output_csv.exists():
        return {}
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        file_headers = next(reader, None)
        if file_headers != headers:
            return {}
        records: dict[str, list[str]] = {}
        for row in reader:
            if not row:
                continue
            key = row[0]
            records[key] = row
    return records


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
    parameters = inputs.parameters_path.read_text(encoding="utf-8")
    documentation = inputs.documentation_path.read_text(encoding="utf-8")
    base = prompts.context_prompt.format(parameters=parameters, documentation=documentation)
    role = prompts.style_features.get("role", "").strip()
    include_role = enabled_style_features is None or "role" in enabled_style_features
    if role and include_role:
        return f"{role}\n\n{base}"
    return base


def _invoke_adapter(adapter: LLMAdapter, model: str, prompt: str, image_b64: str | None = None) -> str:
    request = LLMRequest(model=model, messages=[LLMMessage(role="user", content=prompt)], image_b64=image_b64)
    response = adapter.complete(request)
    return clean_markdown_symbols(strip_think_prefix(response.text))


def _encode_image(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _write_report(output_dir: Path, context: str, trend: str, scores: SummaryScores) -> Path:
    report_path = output_dir / "report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "context_response",
                "trend_response",
                "token_f1",
                "bleu",
                "meteor",
                "rouge1",
                "rouge2",
                "rouge_l",
                "flesch_reading_ease",
            ]
        )
        writer.writerow(
            [
                context,
                trend,
                scores.token_f1,
                scores.bleu,
                scores.meteor,
                scores.rouge1,
                scores.rouge2,
                scores.rouge_l,
                scores.flesch_reading_ease,
            ]
        )
    return report_path


def _summarize_report_text(text: str, skip_summarization: bool) -> str:
    if skip_summarization:
        return text
    try:
        bart = summarize_with_bart(text).strip()
        bert = summarize_with_bert(text).strip()
    except Exception:
        # Keep direct-report mode as a robust fallback when summarizers are unavailable.
        return text
    combined = "\n".join(part for part in [bart, bert] if part)
    return combined if combined else text


def _build_trend_prompt(
    prompts: PromptsConfig,
    metric_description: str,
    context: str,
    plot_description: str | None,
    evidence_mode: EvidenceMode,
    stats_markdown: str,
    enabled_style_features: set[str] | None = None,
) -> str:
    parts: list[str] = []
    enabled = enabled_style_features or set()
    role = prompts.style_features.get("role", "").strip()
    include_all = enabled_style_features is None
    if role and (include_all or "role" in enabled):
        parts.append(role)
    parts.append(prompts.trend_prompt.format(description=metric_description, context=context))
    example = prompts.style_features.get("example", "").strip()
    if example and (include_all or "example" in enabled):
        parts.append(example)
    insights = prompts.style_features.get("insights", "").strip()
    if insights and (include_all or "insights" in enabled):
        parts.append(insights)
    parts = _append_plot_description_parts(parts, plot_description)
    if evidence_mode in {"stats-markdown", "plot+stats"}:
        parts.append(f"Stats table:\n{stats_markdown}")
    return "\n\n".join(parts)


def _write_stats_image_if_needed(
    stats_table: pd.DataFrame,
    output_dir: Path,
    include_pattern: str,
    evidence_mode: EvidenceMode,
) -> Path | None:
    if evidence_mode != "stats-image":
        return None
    path = output_dir / f"{_slug(include_pattern)}_stats.png"
    return render_stats_table_image(stats_table, path)


def _encode_image_for_evidence(
    evidence_mode: EvidenceMode,
    plot_path: Path,
    stats_image_path: Path | None,
) -> str | None:
    if evidence_mode in {"plot", "plot+stats"}:
        return _encode_image(plot_path)
    if evidence_mode == "stats-image" and stats_image_path is not None:
        return _encode_image(stats_image_path)
    return None


def _slug(pattern: str) -> str:
    return pattern.strip().replace(" ", "_").replace("/", "_").lower()


def _append_plot_description_parts(parts: list[str], plot_description: str | None) -> list[str]:
    if plot_description:
        stripped = plot_description.strip()
        if stripped:
            parts.append(stripped)
    return parts


def _append_plot_description(base_prompt: str, plot_description: str) -> str:
    stripped = plot_description.strip()
    if not stripped:
        return base_prompt
    return f"{base_prompt}\n\n{stripped}"
