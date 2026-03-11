from __future__ import annotations

import pytest

from distill_abm.eval.reference_scores import ReferenceScores, compute_scores


def test_compute_scores_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "distill_abm.eval.reference_scores._compute_with_external_metrics",
        lambda ground_truth, summary: ReferenceScores(
            bleu=0.1,
            meteor=0.2,
            rouge1=0.3,
            rouge2=0.4,
            rouge_l=0.5,
            flesch_reading_ease=42.0,
        ),
    )
    scores = compute_scores("the cat sat", "cat sat")
    assert 0.0 <= scores.bleu <= 1.0
    assert 0.0 <= scores.meteor <= 1.0
    assert isinstance(scores.flesch_reading_ease, float)


def test_compute_scores_fails_when_metric_dependencies_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_external_metrics(ground_truth: str, summary: str):  # type: ignore[no-untyped-def]
        _ = ground_truth, summary
        raise ImportError("metrics dependency unavailable")

    monkeypatch.setattr("distill_abm.eval.reference_scores._compute_with_external_metrics", fake_external_metrics)

    with pytest.raises(RuntimeError, match="lexical metrics unavailable"):
        compute_scores("the cat sat", "cat sat")


def test_compute_scores_does_not_hide_metric_runtime_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_external_metrics(ground_truth: str, summary: str):  # type: ignore[no-untyped-def]
        _ = ground_truth, summary
        raise ValueError("bad metric state")

    monkeypatch.setattr("distill_abm.eval.reference_scores._compute_with_external_metrics", fake_external_metrics)

    with pytest.raises(ValueError, match="bad metric state"):
        compute_scores("the cat sat", "cat sat")
