from __future__ import annotations

from distill_abm.eval.reference_scores import compute_scores


def test_compute_scores_shape() -> None:
    scores = compute_scores("the cat sat", "cat sat")
    assert 0.0 <= scores.bleu <= 1.0
    assert 0.0 <= scores.meteor <= 1.0
    assert isinstance(scores.flesch_reading_ease, float)
