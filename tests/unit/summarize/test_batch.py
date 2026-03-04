from __future__ import annotations

from pathlib import Path

import pandas as pd

from distill_abm.summarize.reference_text import summarize_csv_batch


def test_summarize_csv_batch_applies_notebook_lengths_and_formats(tmp_path: Path) -> None:
    input_csv = tmp_path / "responses.csv"
    output_csv = tmp_path / "summaries.csv"
    pd.DataFrame(
        [
            {
                "Context Response": "<think>scratch</think>#Ctx*",
                "Plot 1 Analysis": "#A*",
                "Plot 2 Analysis": "",
                "Plot 3 Analysis": "C",
                "Plot 4 Analysis": "D",
                "Plot 5 Analysis": "E",
            }
        ]
    ).to_csv(input_csv, index=False)

    bart_calls: list[tuple[str, int, int]] = []
    bert_calls: list[str] = []

    def fake_bart(text: str, min_length: int, max_length: int) -> str:
        bart_calls.append((text, min_length, max_length))
        return f"B({min_length}-{max_length}):{text}"

    def fake_bert(text: str) -> str:
        bert_calls.append(text)
        return f"E:{text}"

    summarize_csv_batch(
        input_csv=input_csv,
        output_csv=output_csv,
        plot_columns=[f"Plot {index} Analysis" for index in range(1, 6)],
        bart_summarize_fn=fake_bart,
        bert_summarize_fn=fake_bert,
    )

    result = pd.read_csv(output_csv)
    assert result["Summary (BART) Reduced"].tolist() == [
        "B(200-350):Ctx\n\nB(50-100):A\nB(50-100):C\nB(50-100):D\nB(50-100):E"
    ]
    assert result["Summary (BERT) Reduced"].tolist() == ["E:Ctx\n\nE:A E:C E:D\n\nE:E"]
    assert bart_calls == [
        ("Ctx", 200, 350),
        ("A", 50, 100),
        ("C", 50, 100),
        ("D", 50, 100),
        ("E", 50, 100),
    ]
    assert bert_calls == ["Ctx", "A", "C", "D", "E"]


def test_summarize_csv_batch_ignores_unknown_plot_columns(tmp_path: Path) -> None:
    input_csv = tmp_path / "responses.csv"
    output_csv = tmp_path / "summaries.csv"
    pd.DataFrame(
        [
            {
                "Context Response": "Context",
                "Plot 1 Analysis": "One",
            }
        ]
    ).to_csv(input_csv, index=False)

    summarize_csv_batch(
        input_csv=input_csv,
        output_csv=output_csv,
        plot_columns=["Plot 1 Analysis", "Plot 2 Analysis", "Plot 3 Analysis"],
        bart_summarize_fn=lambda text, _min_length, _max_length: f"B:{text}",
        bert_summarize_fn=lambda text: f"E:{text}",
    )

    result = pd.read_csv(output_csv)
    assert result["Summary (BART) Reduced"].tolist() == ["B:Context\n\nB:One"]
    assert result["Summary (BERT) Reduced"].tolist() == ["E:Context\n\nE:One"]
