from __future__ import annotations

import csv
from pathlib import Path

from distill_abm.summarize.postprocess import (
    capitalize_sentences,
    clean_non_unicode,
    postprocess_summary,
    remove_hyphens_after_punctuation,
    remove_sentences_with_www,
    remove_space_before_dot,
    remove_unnecessary_punctuation,
    remove_unnecessary_spaces_in_parentheses,
)


def test_regex_cleaners() -> None:
    assert remove_sentences_with_www("Keep. Drop www.x.com") == "Keep."
    assert remove_hyphens_after_punctuation("A. - test") == "A. test"
    assert remove_unnecessary_punctuation("Hi. , there") == "Hi. there"
    assert remove_unnecessary_spaces_in_parentheses("( a )") == "(a)"
    assert remove_space_before_dot("a .") == "a."
    assert capitalize_sentences("hello. world") == "Hello. World"


def test_clean_non_unicode_and_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "in.csv"
    target = tmp_path / "out.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["café"])
    clean_non_unicode(source, target)
    assert target.read_text(encoding="utf-8")
    assert postprocess_summary("hello. world") == "Hello. World"
