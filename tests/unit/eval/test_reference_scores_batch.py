from __future__ import annotations

from pathlib import Path

import pandas as pd

from distill_abm.eval.reference_scores import ReferenceScores, score_summaries_csv_batch


def test_score_summaries_csv_batch_writes_notebook_metric_columns(tmp_path: Path) -> None:
    input_csv = tmp_path / "in.csv"
    output_csv = tmp_path / "out.csv"
    pd.DataFrame(
        [
            {"Summary (BART) Reduced": "bart one", "Summary (BERT) Reduced": "bert one"},
            {"Summary (BART) Reduced": "bart two", "Summary (BERT) Reduced": "bert two"},
        ]
    ).to_csv(input_csv, index=False)

    calls: list[tuple[str, str]] = []

    def fake_scores(ground_truth: str, summary: str) -> ReferenceScores:
        calls.append((ground_truth, summary))
        base = float(len(summary.split()))
        return ReferenceScores(
            bleu=base,
            meteor=base + 1.0,
            rouge1=base + 2.0,
            rouge2=base + 3.0,
            rouge_l=base + 4.0,
            flesch_reading_ease=base + 5.0,
        )

    result = score_summaries_csv_batch(
        input_csv=input_csv,
        output_csv=output_csv,
        ground_truth_text="ground truth",
        score_fn=fake_scores,
    )

    assert calls == [
        ("ground truth", "bart one"),
        ("ground truth", "bart two"),
        ("ground truth", "bert one"),
        ("ground truth", "bert two"),
    ]
    assert result["BLEU (BART)"].tolist() == [2.0, 2.0]
    assert result["METEOR (BART)"].tolist() == [3.0, 3.0]
    assert result["ROUGE-1 (BART)"].tolist() == [4.0, 4.0]
    assert result["ROUGE-2 (BART)"].tolist() == [5.0, 5.0]
    assert result["ROUGE-L (BART)"].tolist() == [6.0, 6.0]
    assert result["Flesch Reading Ease (BART)"].tolist() == [7.0, 7.0]
    assert result["BLEU (BERT)"].tolist() == [2.0, 2.0]
    assert result["METEOR (BERT)"].tolist() == [3.0, 3.0]
    assert result["ROUGE-1 (BERT)"].tolist() == [4.0, 4.0]
    assert result["ROUGE-2 (BERT)"].tolist() == [5.0, 5.0]
    assert result["ROUGE-L (BERT)"].tolist() == [6.0, 6.0]
    assert result["Flesch Reading Ease (BERT)"].tolist() == [7.0, 7.0]
    persisted = pd.read_csv(output_csv)
    assert persisted.equals(result)


def test_score_summaries_csv_batch_requires_summary_columns(tmp_path: Path) -> None:
    input_csv = tmp_path / "in.csv"
    output_csv = tmp_path / "out.csv"
    pd.DataFrame([{"Other": "x"}]).to_csv(input_csv, index=False)

    try:
        score_summaries_csv_batch(
            input_csv=input_csv,
            output_csv=output_csv,
            ground_truth_text="ground truth",
        )
    except ValueError as exc:
        assert "Summary (BART) Reduced" in str(exc)
        assert "Summary (BERT) Reduced" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing summary columns")
