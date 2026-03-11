import pytest

from distill_abm.eval import reference_scores as reference_scores_module
from distill_abm.eval.reference_scores import ReferenceScores
from distill_abm.eval.metrics import score_summary


def _stub_reference_scores() -> ReferenceScores:
    return ReferenceScores(
        bleu=0.1,
        meteor=0.2,
        rouge1=0.3,
        rouge2=0.4,
        rouge_l=0.5,
        flesch_reading_ease=42.5,
    )


def test_score_summary_basic_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that basic token overlap is calculated correctly."""
    monkeypatch.setattr("distill_abm.eval.metrics.compute_scores", lambda reference, candidate: _stub_reference_scores())
    scores = score_summary(reference="the cat sat on the mat", candidate="cat sat on mat")
    assert 0.0 <= scores.token_f1 <= 1.0
    assert scores.reference_length == 6
    assert scores.candidate_length == 4
    # "cat", "sat", "on", "mat" overlap. "the" (x2) doesn't.
    # overlap = 4
    # precision = 4/4 = 1.0
    # recall = 4/6 = 0.666
    # f1 = 2 * 1 * 0.66 / (1 + 0.66) = 1.33 / 1.66 = 0.8
    assert round(scores.precision, 2) == 1.0
    assert round(scores.recall, 2) == 0.67
    assert round(scores.token_f1, 2) == 0.80


def test_score_summary_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that identical strings give perfect scores."""
    monkeypatch.setattr("distill_abm.eval.metrics.compute_scores", lambda reference, candidate: _stub_reference_scores())
    scores = score_summary("test string", "test string")
    assert scores.precision == 1.0
    assert scores.recall == 1.0
    assert scores.token_f1 == 1.0


def test_score_summary_no_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that strings with no overlap give zero scores."""
    monkeypatch.setattr("distill_abm.eval.metrics.compute_scores", lambda reference, candidate: _stub_reference_scores())
    scores = score_summary("abc", "def")
    assert scores.precision == 0.0
    assert scores.recall == 0.0
    assert scores.token_f1 == 0.0


def test_score_summary_empty_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that an empty candidate string gives zero scores."""
    monkeypatch.setattr("distill_abm.eval.metrics.compute_scores", lambda reference, candidate: _stub_reference_scores())
    scores = score_summary("abc", "")
    assert scores.precision == 0.0
    assert scores.recall == 0.0
    assert scores.token_f1 == 0.0
    assert scores.candidate_length == 0


def test_score_summary_empty_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that an empty reference string gives zero scores."""
    monkeypatch.setattr("distill_abm.eval.metrics.compute_scores", lambda reference, candidate: _stub_reference_scores())
    scores = score_summary("", "abc")
    assert scores.precision == 0.0
    assert scores.recall == 0.0
    assert scores.token_f1 == 0.0
    assert scores.reference_length == 0


def test_score_summary_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that scoring is case-insensitive."""
    monkeypatch.setattr("distill_abm.eval.metrics.compute_scores", lambda reference, candidate: _stub_reference_scores())
    scores = score_summary("THE CAT", "the cat")
    assert scores.token_f1 == 1.0


def test_compute_scores_fails_when_nltk_resources_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reference_scores_module,
        "_compute_with_external_metrics",
        lambda ground_truth, summary: (_ for _ in ()).throw(LookupError("wordnet missing")),
    )

    with pytest.raises(RuntimeError, match="lexical metrics unavailable"):
        reference_scores_module.compute_scores("abc", "def")


def test_score_summary_preserves_reading_ease_from_reference_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "distill_abm.eval.metrics.compute_scores",
        lambda reference, candidate: _stub_reference_scores(),
    )

    scores = score_summary("reference", "candidate")

    assert scores.flesch_reading_ease == 42.5
