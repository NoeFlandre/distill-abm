"""Internal pipeline helpers for prompt composition and reporting."""

from __future__ import annotations

import base64
import csv
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol

import pandas as pd

from distill_abm.configs.models import PromptsConfig
from distill_abm.eval.metrics import SummaryScores
from distill_abm.llm.adapters.base import LLMAdapter, LLMMessage, LLMRequest
from distill_abm.summarize.models import summarize_with_bart, summarize_with_bert
from distill_abm.summarize.text import clean_markdown_symbols, strip_think_prefix
from distill_abm.viz.plots import generate_stats_table, render_stats_table_image, render_stats_table_markdown

EvidenceMode = Literal["plot", "stats-markdown", "stats-image", "plot+stats"]


class SweepRow(Protocol):
    combination_description: str
    context_prompt: str
    context_response: str
    trend_analysis_prompts: list[str]
    trend_analysis_responses: list[str]


def build_context_prompt(
    inputs_csv_path: Path,
    inputs_doc_path: Path,
    prompts: PromptsConfig,
    enabled: set[str] | None,
) -> str:
    """Compose context prompt with notebook-style optional role preamble."""
    parameters = inputs_csv_path.read_text(encoding="utf-8")
    documentation = inputs_doc_path.read_text(encoding="utf-8")
    base = prompts.context_prompt.format(parameters=parameters, documentation=documentation)
    role = prompts.style_features.get("role", "").strip()
    if (enabled is None or "role" in enabled) and role:
        return f"{role}\n\n{base}"
    return base


def build_trend_prompt(
    prompts: PromptsConfig,
    metric_description: str,
    context: str,
    plot_description: str | None,
    evidence_mode: EvidenceMode,
    stats_markdown: str,
    enabled: set[str] | None = None,
) -> str:
    """Compose trend prompt with optional notebook style features and stats overlay."""
    parts: list[str] = []
    active = enabled or set()
    include_all = enabled is None

    role = prompts.style_features.get("role", "").strip()
    if role and (include_all or "role" in active):
        parts.append(role)

    parts.append(prompts.trend_prompt.format(description=metric_description, context=context))

    example = prompts.style_features.get("example", "").strip()
    if example and (include_all or "example" in active):
        parts.append(example)

    insights = prompts.style_features.get("insights", "").strip()
    if insights and (include_all or "insights" in active):
        parts.append(insights)

    if plot_description:
        stripped_plot = plot_description.strip()
        if stripped_plot:
            parts.append(stripped_plot)

    if evidence_mode in {"stats-markdown", "plot+stats"}:
        parts.append(f"Stats table:\n{stats_markdown}")

    return "\n\n".join(parts)


def invoke_adapter(adapter: LLMAdapter, model: str, prompt: str, image_b64: str | None = None) -> str:
    """Execute one LLM call and normalize response text."""
    request = LLMRequest(model=model, messages=[LLMMessage(role="user", content=prompt)], image_b64=image_b64)
    response = adapter.complete(request)
    return clean_markdown_symbols(strip_think_prefix(response.text))


def encode_image(path: Path) -> str:
    """Read and base64-encode an image artifact."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def summarize_report_text(
    text: str,
    skip_summarization: bool,
    summarize_with_bart_fn: Callable[[str], str] = summarize_with_bart,
    summarize_with_bert_fn: Callable[[str], str] = summarize_with_bert,
) -> str:
    """Apply dual summarization pass-through unless summarization is disabled."""
    if skip_summarization:
        return text
    try:
        bart = summarize_with_bart_fn(text).strip()
        bert = summarize_with_bert_fn(text).strip()
    except Exception:
        # Keep direct output as a robust fallback when summarization backends are unavailable.
        return text
    return "\n".join(part for part in (bart, bert) if part) or text


def build_stats_table(frame: pd.DataFrame, include_pattern: str) -> pd.DataFrame:
    """Build notebook-style statistics table for one metric pattern."""
    return generate_stats_table(frame, include_pattern=include_pattern)


def build_stats_markdown(stats_table: pd.DataFrame) -> str:
    """Render stats table as Markdown."""
    return render_stats_table_markdown(stats_table)


def write_stats_image_if_needed(
    stats_table: pd.DataFrame, output_dir: Path, include_pattern: str, evidence_mode: EvidenceMode
) -> Path | None:
    """Render stats image only for stats-image mode and return artifact path."""
    if evidence_mode != "stats-image":
        return None
    path = output_dir / f"{_slug(include_pattern)}_stats.png"
    return render_stats_table_image(stats_table, path)


def encode_evidence_image(evidence_mode: EvidenceMode, plot_path: Path, stats_image_path: Path | None) -> str | None:
    """Select an encoded image for trend-stage evidence."""
    if evidence_mode in {"plot", "plot+stats"}:
        return encode_image(plot_path)
    if evidence_mode == "stats-image" and stats_image_path is not None:
        return encode_image(stats_image_path)
    return None


def write_report(output_dir: Path, context: str, trend: str, scores: SummaryScores) -> Path:
    """Persist benchmark metrics and trend/context text for one pipeline run."""
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


def sweep_headers(max_items: int, csv_column_style: str) -> list[str]:
    """Build wide CSV headers for combination-level sweep outputs."""
    headers = ["Combination Description", "Context Prompt", "Context Response"]
    for index in range(1, max_items + 1):
        if csv_column_style == "plot":
            headers.append(f"Plot {index} Prompt")
            headers.append(f"Plot {index} Analysis")
        else:
            headers.append(f"Trend Analysis Prompt {index}")
            headers.append(f"Trend Analysis Response {index}")
    return headers


def row_to_record(row: SweepRow) -> list[str]:
    """Flatten prompt/response pairs into a CSV row."""
    record: list[str] = [row.combination_description, row.context_prompt, row.context_response]
    for prompt, response in zip(row.trend_analysis_prompts, row.trend_analysis_responses, strict=True):
        record.extend([prompt, response])
    return record


def load_existing_rows_if_compatible(output_csv: Path, headers: list[str]) -> dict[str, list[str]]:
    """Load existing rows only if headers exactly match the expected layout."""
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


def write_sweep_rows(output_csv: Path, rows: Sequence[SweepRow], headers: list[str]) -> None:
    """Write full sweep records in one pass."""
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row_to_record(row))


def write_sweep_rows_resume(output_csv: Path, rows: Sequence[SweepRow], headers: list[str]) -> None:
    """Resume/merge rows without overwriting completed prompt/response columns."""
    existing = load_existing_rows_if_compatible(output_csv, headers)
    for sweep_row in rows:
        new_record = row_to_record(sweep_row)
        merged = existing.get(sweep_row.combination_description, [""] * len(headers))
        if len(merged) < len(headers):
            merged = merged + [""] * (len(headers) - len(merged))

        merged[0] = sweep_row.combination_description
        if len(new_record) > 1 and not merged[1]:
            merged[1] = new_record[1]
        if len(new_record) > 2 and not merged[2]:
            merged[2] = new_record[2]

        for index in range(3, len(headers), 2):
            prompt_index = index
            response_index = index + 1
            if response_index >= len(headers):
                break
            if merged[prompt_index] or merged[response_index]:
                continue
            prompt = new_record[prompt_index] if prompt_index < len(new_record) else ""
            response = new_record[response_index] if response_index < len(new_record) else ""
            if prompt or response:
                merged[prompt_index] = prompt
                merged[response_index] = response

        existing[sweep_row.combination_description] = merged

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for key in existing:
            writer.writerow(existing[key])


def append_plot_description(base_prompt: str, plot_description: str) -> str:
    """Append a plot description block to an existing prompt."""
    stripped = plot_description.strip()
    if not stripped:
        return base_prompt
    return f"{base_prompt}\n\n{stripped}"


def _slug(pattern: str) -> str:
    """Normalize table filenames for artifact names."""
    return pattern.strip().replace(" ", "_").replace("/", "_").lower()
