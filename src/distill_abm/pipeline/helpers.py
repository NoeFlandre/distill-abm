"""Internal pipeline helpers for prompt composition and reporting."""

from __future__ import annotations

import base64
import csv
import hashlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, cast

import pandas as pd

from distill_abm.configs.models import PromptsConfig, SummarizerId
from distill_abm.configs.runtime_defaults import get_runtime_defaults
from distill_abm.eval.metrics import SummaryScores
from distill_abm.llm.adapters.base import LLMAdapter, LLMMessage, LLMProviderError, LLMRequest
from distill_abm.llm.request_defaults import resolve_request_temperature
from distill_abm.llm.resilience import ensure_circuit_closed, record_failure, record_success
from distill_abm.pipeline.statistical_evidence import build_statistical_evidence
from distill_abm.structured_logging import get_logger, log_event
from distill_abm.summarize.models import (
    summarize_with_bart,
    summarize_with_bert,
    summarize_with_longformer_ext,
    summarize_with_t5,
)
from distill_abm.summarize.text import clean_markdown_symbols, strip_think_prefix
from distill_abm.viz.plots import generate_stats_table

EvidenceMode = Literal["plot", "table", "plot+table"]
ResolvedEvidenceMode = Literal["plot", "table", "plot+table"]
TextSourceMode = Literal["summary_only", "full_text_only"]
LOGGER = get_logger(__name__)


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
    """Compose context prompt with optional role preamble."""
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
    stats_table_csv: str,
    enabled: set[str] | None = None,
) -> str:
    """Compose trend prompt with optional role/example/insight features and table evidence overlay."""
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

    resolved_mode = resolve_evidence_mode(evidence_mode)
    if resolved_mode in {"table", "plot+table"}:
        parts.append(f"Statistical summary of the relevant simulation output:\n{stats_table_csv}")

    return "\n\n".join(parts)


def invoke_adapter(
    adapter: LLMAdapter,
    model: str,
    prompt: str,
    image_b64: str | None = None,
    max_tokens: int | None = None,
    request_metadata: dict[str, object] | None = None,
    max_retries: int | None = None,
    retry_backoff_seconds: float | None = None,
) -> str:
    """Execute one LLM call with bounded retries and normalize response text."""
    text, _trace = invoke_adapter_with_trace(
        adapter=adapter,
        model=model,
        prompt=prompt,
        image_b64=image_b64,
        max_tokens=max_tokens,
        request_metadata=request_metadata,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    return text


def invoke_adapter_with_trace(
    adapter: LLMAdapter,
    model: str,
    prompt: str,
    image_b64: str | None = None,
    max_tokens: int | None = None,
    request_metadata: dict[str, object] | None = None,
    max_retries: int | None = None,
    retry_backoff_seconds: float | None = None,
) -> tuple[str, dict[str, object]]:
    """Execute one LLM call and return normalized text plus a debug trace payload."""
    request = LLMRequest(
        model=model,
        messages=[LLMMessage(role="user", content=prompt)],
        temperature=resolve_request_temperature(adapter.provider),
        image_b64=image_b64,
        max_tokens=max_tokens,
        metadata=dict(request_metadata or {}),
    )
    defaults = get_runtime_defaults().llm_request
    retries = max(defaults.max_retries if max_retries is None else max_retries, 0)
    backoff = max(defaults.retry_backoff_seconds if retry_backoff_seconds is None else retry_backoff_seconds, 0.0)
    preserve_raw_text = bool(request.metadata.get("preserve_raw_text"))
    request_block = {
        "provider": adapter.provider,
        "model": model,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "max_retries": retries,
        "retry_backoff_seconds": backoff,
        "image_attached": image_b64 is not None,
        "prompt_text": prompt,
        "prompt_length": len(prompt),
        "prompt_signature": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "message_count": len(request.messages),
        "messages": [message.model_dump() for message in request.messages],
        "metadata": request.metadata,
    }
    log_event(
        LOGGER,
        "llm_request_start",
        provider=adapter.provider,
        model=model,
        prompt_signature=request_block["prompt_signature"],
        prompt_length=request_block["prompt_length"],
        image_attached=request_block["image_attached"],
        max_retries=retries,
    )

    errors: list[str] = []
    for attempt in range(retries + 1):
        ensure_circuit_closed(provider=adapter.provider, model=model)
        try:
            response = adapter.complete(request)
            if preserve_raw_text:
                clean_text = response.text
            else:
                clean_text = clean_markdown_symbols(strip_think_prefix(response.text))
            record_success(provider=adapter.provider, model=model)
            log_event(
                LOGGER,
                "llm_request_success",
                provider=adapter.provider,
                model=model,
                attempt=attempt + 1,
                prompt_signature=request_block["prompt_signature"],
                clean_text_signature=hashlib.sha256(clean_text.encode("utf-8")).hexdigest(),
                clean_text_length=len(clean_text),
            )
            return clean_text, {
                "request": request_block,
                "response": {
                    "provider": response.provider,
                    "model": response.model,
                    "raw_text": response.text,
                    "raw_text_length": len(response.text),
                    "raw_text_signature": hashlib.sha256(response.text.encode("utf-8")).hexdigest(),
                    "clean_text": clean_text,
                    "clean_text_length": len(clean_text),
                    "clean_text_signature": hashlib.sha256(clean_text.encode("utf-8")).hexdigest(),
                    "usage": _extract_usage_from_raw(response.raw),
                    "raw": response.raw,
                },
                "attempts_made": attempt + 1,
                "errors": list(errors),
            }
        except Exception as exc:
            wrapped = exc if isinstance(exc, LLMProviderError) else LLMProviderError(str(exc))
            errors.append(str(wrapped))
            record_failure(provider=adapter.provider, model=model, error=str(wrapped))
            log_event(
                LOGGER,
                "llm_request_failure",
                level=logging.WARNING,
                provider=adapter.provider,
                model=model,
                attempt=attempt + 1,
                prompt_signature=request_block["prompt_signature"],
                error=str(wrapped),
            )
            is_last_attempt = attempt >= retries
            if is_last_attempt:
                break
            if backoff > 0:
                time.sleep(backoff * (2**attempt))
    raise LLMProviderError(
        f"{adapter.provider} request failed after {retries + 1} attempt(s): {errors[-1] if errors else 'unknown'}"
    )


def encode_image(path: Path) -> str:
    """Read and base64-encode an image artifact."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _extract_usage_from_raw(raw: object) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    usage = raw.get("usage")
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if all(isinstance(value, int) for value in (prompt, completion, total)):
            prompt_tokens = cast(int, prompt)
            completion_tokens = cast(int, completion)
            total_tokens = cast(int, total)
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
    prompt_eval = raw.get("prompt_eval_count")
    eval_count = raw.get("eval_count")
    if isinstance(prompt_eval, int) and isinstance(eval_count, int):
        return {
            "prompt_tokens": prompt_eval,
            "completion_tokens": eval_count,
            "total_tokens": prompt_eval + eval_count,
        }
    return None


def summarize_report_text(
    text: str,
    skip_summarization: bool,
    summarize_with_bart_fn: Callable[[str], str] = summarize_with_bart,
    summarize_with_bert_fn: Callable[[str], str] = summarize_with_bert,
    additional_summarizers: Sequence[tuple[str, Callable[[str], str]]] = (),
    allow_fallback: bool = True,
) -> str:
    """Apply dual summarization pass-through unless summarization is disabled."""
    _, summary = summarize_report_text_pair(
        text=text,
        skip_summarization=skip_summarization,
        summarize_with_bart_fn=summarize_with_bart_fn,
        summarize_with_bert_fn=summarize_with_bert_fn,
        additional_summarizers=additional_summarizers,
        allow_fallback=allow_fallback,
    )
    if summary is None:
        return text
    return summary


def summarize_report_text_pair(
    text: str,
    skip_summarization: bool,
    summarize_with_bart_fn: Callable[[str], str] = summarize_with_bart,
    summarize_with_bert_fn: Callable[[str], str] = summarize_with_bert,
    additional_summarizers: Sequence[tuple[str, Callable[[str], str]]] = (),
    summarizer_ids: tuple[SummarizerId, ...] | None = None,
    allow_fallback: bool = True,
) -> tuple[str, str | None]:
    """Return raw trend text plus optional summary for dual-path operation."""
    if skip_summarization:
        return text, None

    specs = (
        [("bart", summarize_with_bart_fn), ("bert", summarize_with_bert_fn), *additional_summarizers]
        if summarizer_ids is None
        else list(_summarizer_specs_from_ids(summarizer_ids))
    )

    summary = _collect_summary(text=text, summarizer_specs=specs)
    if summary:
        return text, summary

    if allow_fallback:
        return text, None
    raise RuntimeError("No configured summarizer produced a valid summary for this text mode")


def summarize_report_text_pair_for_ids(
    text: str,
    skip_summarization: bool,
    summarizer_ids: tuple[SummarizerId, ...],
    allow_fallback: bool = True,
) -> tuple[str, str | None]:
    """Return raw trend text plus optional summary for selected summarizer IDs."""
    return summarize_report_text_pair(
        text=text,
        skip_summarization=skip_summarization,
        summarizer_ids=summarizer_ids,
        allow_fallback=allow_fallback,
    )


def _summarizer_specs_from_ids(
    summarizer_ids: tuple[SummarizerId, ...],
) -> tuple[tuple[str, Callable[[str], str]], ...]:
    spec_by_id: dict[SummarizerId, Callable[[str], str]] = {
        "bart": summarize_with_bart,
        "bert": summarize_with_bert,
        "t5": summarize_with_t5,
        "longformer_ext": summarize_with_longformer_ext,
    }
    return tuple((summarizer_id, spec_by_id[summarizer_id]) for summarizer_id in summarizer_ids)


def _collect_summary(text: str, summarizer_specs: Sequence[tuple[str, Callable[[str], str]]]) -> str:
    """Run all configured summarizers and combine non-empty summaries in order."""
    summary_parts: list[str] = []
    for _name, runner in summarizer_specs:
        try:
            value = runner(text).strip()
        except Exception:
            # Keep direct output as a robust fallback when summarization backends are unavailable.
            continue
        if value:
            summary_parts.append(value)
    return "\n".join(summary_parts).strip()


def select_trend_response(trend_full: str, trend_summary: str | None, use_summary: bool) -> str:
    """Select the report trend response according to a mode."""
    if use_summary and trend_summary is not None:
        return trend_summary
    return trend_full


def build_stats_table(frame: pd.DataFrame, include_pattern: str) -> pd.DataFrame:
    """Build statistics table for one metric pattern."""
    return generate_stats_table(frame, include_pattern=include_pattern)


def build_stats_csv(stats_table: pd.DataFrame) -> str:
    """Render stats table as CSV text for text-only prompt evidence."""
    columns = ["time_step", "mean", "std", "min", "max", "median"]
    table = stats_table[columns]
    return table.to_csv(index=False, lineterminator="\n")


def build_statistical_table_evidence(*, frame: pd.DataFrame, reporter_pattern: str, compression_tier: int = 0) -> str:
    """Render statistical evidence text from the plot-relevant simulation slice only."""
    return build_statistical_evidence(
        frame=frame,
        reporter_pattern=reporter_pattern,
        compression_tier=compression_tier,
    ).summary_text


def write_stats_image_if_needed(
    stats_table: pd.DataFrame, output_dir: Path, include_pattern: str, evidence_mode: EvidenceMode
) -> Path | None:
    """Stats-table images are intentionally disabled for reproducible text-table ablations."""
    _ = (stats_table, output_dir, include_pattern, evidence_mode)
    return None


def encode_evidence_image(evidence_mode: EvidenceMode, plot_path: Path, stats_image_path: Path | None) -> str | None:
    """Select an encoded image for trend-stage evidence."""
    _ = stats_image_path
    resolved_mode = resolve_evidence_mode(evidence_mode)
    if resolved_mode in {"plot", "plot+table"}:
        return encode_image(plot_path)
    return None


def resolve_evidence_mode(evidence_mode: EvidenceMode) -> ResolvedEvidenceMode:
    """Validate reviewer-facing evidence ablation modes."""
    if evidence_mode in {"plot", "table", "plot+table"}:
        return evidence_mode
    raise ValueError(f"unsupported evidence mode: {evidence_mode}")


def write_report(
    output_dir: Path,
    context: str,
    trend_full: str,
    trend_summary: str | None,
    scores: SummaryScores,
    full_scores: SummaryScores | None = None,
    summary_scores: SummaryScores | None = None,
    include_extended_columns: bool = False,
    additional_reference_scores: Mapping[str, Mapping[str, SummaryScores | None]] | None = None,
) -> Path:
    """Persist benchmark metrics and trend/context text for one pipeline run."""
    report_path = output_dir / "report.csv"
    has_dual_path = include_extended_columns
    trend_response = select_trend_response(
        trend_full=trend_full,
        trend_summary=trend_summary,
        use_summary=summary_scores is not None and trend_summary is not None,
    )
    headers = [
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
    row: list[str | float | int] = [
        context,
        trend_response,
        scores.token_f1,
        scores.bleu,
        scores.meteor,
        scores.rouge1,
        scores.rouge2,
        scores.rouge_l,
        scores.flesch_reading_ease,
    ]

    if has_dual_path:
        headers.extend(["trend_full_response", "trend_summary_response"])
        row.extend([trend_full, trend_summary or ""])

    if full_scores is not None:
        headers.extend(
            [
                "full_token_f1",
                "full_bleu",
                "full_meteor",
                "full_rouge1",
                "full_rouge2",
                "full_rouge_l",
                "full_flesch_reading_ease",
            ]
        )
        row.extend(
            [
                full_scores.token_f1,
                full_scores.bleu,
                full_scores.meteor,
                full_scores.rouge1,
                full_scores.rouge2,
                full_scores.rouge_l,
                full_scores.flesch_reading_ease,
            ]
        )

    if summary_scores is not None:
        headers.extend(
            [
                "summary_token_f1",
                "summary_bleu",
                "summary_meteor",
                "summary_rouge1",
                "summary_rouge2",
                "summary_rouge_l",
                "summary_flesch_reading_ease",
            ]
        )
        row.extend(
            [
                summary_scores.token_f1,
                summary_scores.bleu,
                summary_scores.meteor,
                summary_scores.rouge1,
                summary_scores.rouge2,
                summary_scores.rouge_l,
                summary_scores.flesch_reading_ease,
            ]
        )

    if additional_reference_scores is not None:
        for reference_name, reference_scores in additional_reference_scores.items():
            _append_reference_score_columns(
                headers=headers,
                row=row,
                prefix=reference_name,
                scores=reference_scores.get("selected_scores"),
            )
            _append_reference_score_columns(
                headers=headers,
                row=row,
                prefix=f"{reference_name}_full",
                scores=reference_scores.get("full_scores"),
            )
            _append_reference_score_columns(
                headers=headers,
                row=row,
                prefix=f"{reference_name}_summary",
                scores=reference_scores.get("summary_scores"),
            )

    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerow(row)
    return report_path


def _append_reference_score_columns(
    *,
    headers: list[str],
    row: list[str | float | int],
    prefix: str,
    scores: SummaryScores | None,
) -> None:
    if scores is None:
        return
    headers.extend(
        [
            f"{prefix}_token_f1",
            f"{prefix}_bleu",
            f"{prefix}_meteor",
            f"{prefix}_rouge1",
            f"{prefix}_rouge2",
            f"{prefix}_rouge_l",
            f"{prefix}_flesch_reading_ease",
        ]
    )
    row.extend(
        [
            scores.token_f1,
            scores.bleu,
            scores.meteor,
            scores.rouge1,
            scores.rouge2,
            scores.rouge_l,
            scores.flesch_reading_ease,
        ]
    )


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
