from __future__ import annotations

import sys

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


def test_compute_scores_recovers_from_nltk_fraction_normalize_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTextStat:
        @staticmethod
        def flesch_reading_ease(summary: str) -> float:
            _ = summary
            return 42.0

    class _FakeSmoothingFunction:
        def method4(self) -> object:
            return object()

    class _FakeBleuModule:
        def __init__(self) -> None:
            self.Fraction = object()
            self.SmoothingFunction = _FakeSmoothingFunction

        def sentence_bleu(self, *args: object, **kwargs: object) -> float:
            _ = args, kwargs
            fraction = self.Fraction
            if callable(fraction):
                fraction(1, 2, _normalize=False)
                return 0.75
            raise TypeError("Fraction.__new__() got an unexpected keyword argument '_normalize'")

    class _FakeMeteorModule:
        @staticmethod
        def meteor_score(*args: object, **kwargs: object) -> float:
            _ = args, kwargs
            return 0.5

    class _FakeRougeScore:
        def __init__(self, fmeasure: float) -> None:
            self.fmeasure = fmeasure

    class _FakeRougeScorer:
        def __init__(self, metrics: list[str], use_stemmer: bool) -> None:
            _ = metrics, use_stemmer

        def score(self, ground_truth: str, summary: str) -> dict[str, _FakeRougeScore]:
            _ = ground_truth, summary
            return {
                "rouge1": _FakeRougeScore(0.4),
                "rouge2": _FakeRougeScore(0.3),
                "rougeL": _FakeRougeScore(0.2),
            }

    class _FakeRougeModule:
        RougeScorer = _FakeRougeScorer

    monkeypatch.setitem(sys.modules, "textstat", _FakeTextStat())
    monkeypatch.setitem(sys.modules, "nltk.translate.bleu_score", _FakeBleuModule())
    monkeypatch.setitem(sys.modules, "nltk.translate.meteor_score", _FakeMeteorModule())
    monkeypatch.setitem(sys.modules, "rouge_score", type("_RougePackage", (), {"rouge_scorer": _FakeRougeModule()})())

    scores = compute_scores("the cat sat", "cat sat")

    assert scores.bleu == 0.75
    assert scores.meteor == 0.5
    assert scores.rouge1 == 0.4
