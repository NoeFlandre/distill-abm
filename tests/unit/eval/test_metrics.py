from distill_abm.eval.metrics import score_summary


def test_score_summary_ranges() -> None:
    scores = score_summary(reference="the cat sat on the mat", candidate="cat sat on mat")
    assert 0.0 <= scores.token_f1 <= 1.0
    assert scores.reference_length == 6
    assert scores.candidate_length == 4
