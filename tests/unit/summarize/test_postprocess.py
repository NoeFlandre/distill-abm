from __future__ import annotations

import csv
from pathlib import Path

from distill_abm.summarize.postprocess import (
    capitalize_sentences,
    clean_non_unicode,
    postprocess_summary,
    remove_repeated_phrases,
    remove_repeated_sentences,
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
    assert capitalize_sentences("") == ""


def test_clean_non_unicode_and_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "in.csv"
    target = tmp_path / "out.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["café"])
    clean_non_unicode(source, target)
    assert target.read_text(encoding="utf-8")
    assert postprocess_summary("hello. world") == "Hello. World"


def test_remove_repeated_sentences_only_collapses_adjacent_duplicates() -> None:
    text = "Milk increases steadily. Milk increases steadily. Then it plateaus. Milk increases steadily."

    assert remove_repeated_sentences(text) == "Milk increases steadily. Then it plateaus. Milk increases steadily."


def test_remove_repeated_phrases_collapses_obvious_tail_loops() -> None:
    text = (
        "This steady linear increase suggests a consistent rate of decision-making and interaction among the agents "
        "throughout the simulation period. The rate of change in the rate of change in the rate of change in the rate "
        "of change in the rate of change in the rate of change in the rate of change in the rate of change."
    )

    assert remove_repeated_phrases(text) == (
        "This steady linear increase suggests a consistent rate of decision-making and interaction among the agents "
        "throughout the simulation period. The rate of change."
    )


def test_postprocess_summary_preserves_non_loop_repetition_that_may_be_meaningful() -> None:
    text = "Very very high demand can still matter. Very high demand may persist."

    assert postprocess_summary(text) == text
