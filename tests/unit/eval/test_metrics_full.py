import pytest

from distill_abm.eval.metrics import score_summary


def _fake_compute_scores(reference: str, candidate: str):  # type: ignore[no-untyped-def]
    _ = (reference, candidate)
    from distill_abm.eval.reference_scores import ReferenceScores

    return ReferenceScores(
        bleu=0.1,
        meteor=0.2,
        rouge1=0.3,
        rouge2=0.4,
        rouge_l=0.5,
        flesch_reading_ease=42.0,
    )


def test_score_summary_includes_reference_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("distill_abm.eval.metrics.compute_scores", _fake_compute_scores)
    scores = score_summary(
        reference="the quick brown fox jumps over the lazy dog",
        candidate="the quick fox jumps over dog",
    )

    assert 0.0 <= scores.token_f1 <= 1.0
    assert 0.0 <= scores.bleu <= 1.0
    assert 0.0 <= scores.meteor <= 1.0
    assert 0.0 <= scores.rouge1 <= 1.0
    assert 0.0 <= scores.rouge2 <= 1.0
    assert 0.0 <= scores.rouge_l <= 1.0
    assert isinstance(scores.flesch_reading_ease, float)
