from __future__ import annotations

from distill_abm.eval.qualitative import (
    extract_coverage_score,
    extract_faithfulness_score,
    should_skip_row,
)


def test_should_skip_row() -> None:
    assert should_skip_row({"score": 3}, "score")
    assert should_skip_row({"score": "done"}, "score")
    assert not should_skip_row({"score": 0}, "score")


def test_extract_scores() -> None:
    assert extract_faithfulness_score("Faithfulness score: 4/5") == "4"
    assert extract_coverage_score("I give coverage 3 out of 5") == "3"
