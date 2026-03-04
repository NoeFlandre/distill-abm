from __future__ import annotations

from pathlib import Path

import pandas as pd

from distill_abm.summarize.postprocess import postprocess_csv_batch


def test_postprocess_csv_batch_matches_stage_order(tmp_path: Path) -> None:
    input_csv = tmp_path / "in.csv"
    output_csv = tmp_path / "out.csv"
    pd.DataFrame(
        [
            {
                "Summary (BART) Reduced": "drop www.bad.com. keep café .",
                "Summary (BERT) Reduced": "hello. , world. - next ( a ) café .",
                "Other": "Météo",
            }
        ]
    ).to_csv(input_csv, index=False)

    result = postprocess_csv_batch(input_csv=input_csv, output_csv=output_csv)

    assert result["Summary (BART) Reduced"].tolist() == ["keep cafe."]
    assert result["Summary (BERT) Reduced"].tolist() == ["Hello. World. Next (a) cafe ."]
    assert result["Other"].tolist() == ["Meteo"]
    persisted = pd.read_csv(output_csv)
    assert persisted.equals(result)


def test_postprocess_csv_batch_tolerates_missing_summary_columns(tmp_path: Path) -> None:
    input_csv = tmp_path / "in.csv"
    output_csv = tmp_path / "out.csv"
    pd.DataFrame([{"Summary (BART) Reduced": "A .", "Other": "é"}]).to_csv(input_csv, index=False)

    result = postprocess_csv_batch(input_csv=input_csv, output_csv=output_csv)

    assert result["Summary (BART) Reduced"].tolist() == ["A."]
    assert result["Other"].tolist() == ["e"]


def test_postprocess_csv_batch_without_bart_column(tmp_path: Path) -> None:
    input_csv = tmp_path / "in.csv"
    output_csv = tmp_path / "out.csv"
    pd.DataFrame([{"Summary (BERT) Reduced": "x. , y.", "Other": "café"}]).to_csv(input_csv, index=False)

    result = postprocess_csv_batch(input_csv=input_csv, output_csv=output_csv, bart_column="Summary (BART) Reduced")

    assert result["Summary (BERT) Reduced"].tolist() == ["X. Y."]
    assert result["Other"].tolist() == ["cafe"]
