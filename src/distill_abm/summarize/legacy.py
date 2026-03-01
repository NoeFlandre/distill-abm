"""Legacy summary helpers retained for notebook-equivalent behavior."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol

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
    """Notebook-compatible CSV pass that cleans the `Context Response` column."""
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
    """Runs chunked summarization exactly as done in BART notebook cells."""
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
    """Applies notebook markdown cleanup to selected dataframe columns."""
    copied = frame.copy()
    for column in columns:
        if column in copied.columns:
            copied[column] = copied[column].apply(clean_symbols)
    return copied
