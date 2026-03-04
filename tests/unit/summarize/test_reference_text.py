from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, cast

import pandas as pd

from distill_abm.summarize.reference_text import (
    chunk_text,
    clean_context_response,
    clean_dataframe_symbols,
    clean_symbols,
    process_csv_context,
    summarize_text,
)


class DummyTokenizer:
    def encode(self, text: str, truncation: bool = False, add_special_tokens: bool = True) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(value) for value in ids)


class DummySummarizer:
    tokenizer = DummyTokenizer()

    def __call__(
        self,
        text: str,
        min_length: int,
        max_length: int,
        truncation: bool,
    ) -> list[dict[str, str]]:
        _ = (min_length, max_length, truncation)
        return [{"summary_text": text.upper()}]


def test_clean_context_response() -> None:
    assert clean_context_response("a</think>final") == "final"


def test_process_csv_context(tmp_path: Path) -> None:
    source = tmp_path / "in.csv"
    target = tmp_path / "out.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Context Response"])
        writer.writeheader()
        writer.writerow({"Context Response": "x</think>clean"})
    process_csv_context(source, target)
    output = target.read_text(encoding="utf-8")
    assert "clean" in output


def test_clean_symbols_and_dataframe() -> None:
    assert clean_symbols("#x*y") == "xy"
    assert pd.isna(clean_symbols(float("nan")))
    frame = pd.DataFrame({"a": ["#x", "*y"]})
    cleaned = clean_dataframe_symbols(frame, ["a"])
    assert cleaned["a"].tolist() == ["x", "y"]


def test_chunk_and_summarize_text() -> None:
    parts = chunk_text("abcdef", tokenizer=DummyTokenizer(), max_input_length=2)
    assert parts == ["ab", "cd", "ef"]
    summary = summarize_text(
        "abcdef",
        summarizer=cast(Any, DummySummarizer()),
        max_input_length=3,
    )
    assert summary == "ABC DEF"
