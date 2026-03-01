"""Pipeline orchestration from CSV ingestion to scored report export."""

from __future__ import annotations

import base64
import csv
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from distill_abm.configs.models import PromptsConfig
from distill_abm.eval.metrics import SummaryScores, score_summary
from distill_abm.ingest.csv_ingest import load_simulation_csv
from distill_abm.llm.adapters.base import LLMAdapter, LLMMessage, LLMRequest
from distill_abm.summarize.models import summarize_with_bart, summarize_with_bert
from distill_abm.summarize.text import clean_markdown_symbols, strip_think_prefix
from distill_abm.viz.plots import (
    generate_stats_table,
    plot_metric_bundle,
    render_stats_table_image,
    render_stats_table_markdown,
)

EvidenceMode = Literal["plot", "stats-markdown", "stats-image", "plot+stats"]


class PipelineInputs(BaseModel):
    """Defines required inputs so CLI calls stay predictable and testable."""

    csv_path: Path
    parameters_path: Path
    documentation_path: Path
    output_dir: Path
    model: str
    metric_pattern: str
    metric_description: str
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


def run_pipeline(inputs: PipelineInputs, prompts: PromptsConfig, adapter: LLMAdapter) -> PipelineResult:
    """Executes the notebook-equivalent workflow through pure Python components."""
    frame = load_simulation_csv(inputs.csv_path)
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_metric_bundle(frame, inputs.metric_pattern, inputs.output_dir, "Simulation trend", "value")
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


def _context_prompt(inputs: PipelineInputs, prompts: PromptsConfig) -> str:
    parameters = inputs.parameters_path.read_text(encoding="utf-8")
    documentation = inputs.documentation_path.read_text(encoding="utf-8")
    return prompts.context_prompt.format(parameters=parameters, documentation=documentation)


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
    evidence_mode: EvidenceMode,
    stats_markdown: str,
) -> str:
    base = prompts.trend_prompt.format(description=metric_description, context=context)
    if evidence_mode in {"stats-markdown", "plot+stats"}:
        return f"{base}\n\nStats table:\n{stats_markdown}"
    return base


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
