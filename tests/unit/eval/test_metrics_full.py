from distill_abm.eval.metrics import score_summary


def test_score_summary_includes_legacy_metrics() -> None:
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
