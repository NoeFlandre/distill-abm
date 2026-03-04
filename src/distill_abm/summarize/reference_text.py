"""Summary helpers for reusable text preprocessing and chunking."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


class TokenizerLike(Protocol):
    """Protocol used by BART/BERT chunkers for deterministic testing."""

    def encode(
        self,
        text: str,
        truncation: bool = False,
        add_special_tokens: bool = True,
    ) -> list[int]: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str: ...


class SummarizerLike(Protocol):
    """Protocol for transformers pipeline objects and injected test doubles."""

    tokenizer: TokenizerLike

    def __call__(
        self,
        text: str,
        min_length: int,
        max_length: int,
        truncation: bool,
    ) -> list[dict[str, str]]: ...


def clean_context_response(text: str) -> str:
    """Drops reasoning traces so only final answer text enters downstream summaries."""
    return text.split("</think>")[-1].strip()


def process_csv_context(input_file: Path, output_file: Path) -> None:
    """CSV pass that cleans the `Context Response` column."""
    with input_file.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        rows = [row for row in reader]
    if fieldnames is None:
        raise ValueError("input CSV is missing a header row")
    with output_file.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if "Context Response" in row:
                row["Context Response"] = clean_context_response(row["Context Response"])
            writer.writerow(row)


def clean_symbols(text: str | float) -> str | float:
    """Removes markdown symbols while preserving NaN semantics."""
    if pd.isna(text):
        return text
    return str(text).replace("#", "").replace("*", "")


def chunk_text(text: str, tokenizer: TokenizerLike, max_input_length: int = 1024) -> list[str]:
    """Splits long text into tokenizer-aligned windows for summarization models."""
    tokens = tokenizer.encode(text, truncation=False)
    chunks: list[str] = []
    for index in range(0, len(tokens), max_input_length):
        chunk_ids = tokens[index : index + max_input_length]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk)
    return chunks


def summarize_text(
    text: str,
    summarizer: SummarizerLike,
    min_summary_length: int = 50,
    max_summary_length: int = 100,
    max_input_length: int = 1024,
) -> str:
    """Run chunked summarization with the configured summarizer."""
    chunks = chunk_text(text, summarizer.tokenizer, max_input_length)
    summaries: list[str] = []
    for chunk in chunks:
        result = summarizer(
            chunk,
            min_length=min_summary_length,
            max_length=max_summary_length,
            truncation=True,
        )
        summaries.append(result[0]["summary_text"])
    return " ".join(summaries)


def clean_dataframe_symbols(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply symbol cleanup to selected dataframe columns."""
    copied = frame.copy()
    for column in columns:
        if column in copied.columns:
            copied[column] = copied[column].apply(clean_symbols)
    return copied


def summarize_csv_batch(
    input_csv: Path,
    output_csv: Path,
    plot_columns: list[str],
    context_column: str = "Context Response",
    bart_output_column: str = "Summary (BART) Reduced",
    bert_output_column: str = "Summary (BERT) Reduced",
    bart_summarize_fn: Callable[[str, int, int], str] | None = None,
    bert_summarize_fn: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Generate summary columns for LLM-response tables."""
    frame = pd.read_csv(input_csv)
    frame = _clean_summary_input_frame(frame, context_column=context_column, plot_columns=plot_columns)

    bart = bart_summarize_fn or _default_bart_summarize
    bert = bert_summarize_fn or _default_bert_summarize

    bart_summaries: list[str] = []
    bert_summaries: list[str] = []
    for _, row in frame.iterrows():
        context_text = _row_text(row, context_column)
        bart_summaries.append(_build_bart_summary(row, context_text, plot_columns, bart))
        bert_summaries.append(_build_bert_summary(row, context_text, plot_columns, bert))

    out = frame.copy()
    out[bart_output_column] = bart_summaries
    out[bert_output_column] = bert_summaries
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def _default_bart_summarize(text: str, min_length: int, max_length: int) -> str:
    from distill_abm.summarize.models import BartSummarizerRunner

    runner = BartSummarizerRunner()
    return runner.summarize(text=text, min_summary_length=min_length, max_summary_length=max_length)


def _default_bert_summarize(text: str) -> str:
    from distill_abm.summarize.models import summarize_with_bert

    return summarize_with_bert(text)


def _clean_summary_input_frame(frame: pd.DataFrame, context_column: str, plot_columns: list[str]) -> pd.DataFrame:
    copied = frame.copy()
    if context_column in copied.columns:
        copied[context_column] = copied[context_column].apply(_clean_context_cell)
    return clean_dataframe_symbols(copied, [context_column, *plot_columns])


def _clean_context_cell(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return clean_context_response(str(value))


def _row_text(row: pd.Series, column: str) -> str:
    if column not in row.index:
        return ""
    value = row[column]
    if pd.isna(value):
        return ""
    return str(value).strip()


def _build_bart_summary(
    row: pd.Series,
    context_text: str,
    plot_columns: list[str],
    bart_summarize_fn: Callable[[str, int, int], str],
) -> str:
    context_summary = bart_summarize_fn(context_text, 200, 350) if context_text else ""
    plot_summaries: list[str] = []
    for column in plot_columns:
        text = _row_text(row, column)
        if text:
            plot_summaries.append(bart_summarize_fn(text, 50, 100))

    summary = context_summary + "\n\n"
    for index in range(0, len(plot_summaries), 4):
        summary += "\n".join(plot_summaries[index : index + 4]) + "\n\n"
    return summary.strip()


def _build_bert_summary(
    row: pd.Series,
    context_text: str,
    plot_columns: list[str],
    bert_summarize_fn: Callable[[str], str],
) -> str:
    summary = ""
    if context_text:
        summary += bert_summarize_fn(context_text) + "\n\n"
    for index in range(0, len(plot_columns), 4):
        block_parts: list[str] = []
        for column in plot_columns[index : index + 4]:
            text = _row_text(row, column)
            if text:
                block_parts.append(bert_summarize_fn(text))
        if block_parts:
            summary += " ".join(block_parts).strip() + "\n\n"
    return summary.strip()
