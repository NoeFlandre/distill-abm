from __future__ import annotations

from pathlib import Path

import pandas as pd

from distill_abm.viz.plots import generate_stats_table, render_stats_table_image, render_stats_table_markdown


def test_generate_stats_table_computes_row_statistics() -> None:
    frame = pd.DataFrame(
        {
            "tick": [0, 1, 2],
            "mean-incum-run-1": [1.0, 3.0, 5.0],
            "mean-incum-run-2": [3.0, 5.0, 7.0],
        }
    )

    table = generate_stats_table(frame, include_pattern="mean-incum")
    assert list(table.columns) == ["time_step", "mean", "std", "min", "max", "median"]
    assert table["time_step"].tolist() == [0, 1, 2]
    assert table["mean"].tolist() == [2.0, 4.0, 6.0]
    assert table["min"].tolist() == [1.0, 3.0, 5.0]
    assert table["max"].tolist() == [3.0, 5.0, 7.0]
    assert table["median"].tolist() == [2.0, 4.0, 6.0]


def test_render_stats_table_markdown_returns_clean_markdown() -> None:
    table = pd.DataFrame(
        {
            "time_step": [0],
            "mean": [2.0],
            "std": [1.0],
            "min": [1.0],
            "max": [3.0],
            "median": [2.0],
        }
    )
    markdown = render_stats_table_markdown(table)
    assert "| time_step | mean | std | min | max | median |" in markdown
    assert "| --- | --- | --- | --- | --- | --- |" in markdown
    assert "| 0 | 2.0000 | 1.0000 | 1.0000 | 3.0000 | 2.0000 |" in markdown


def test_render_stats_table_image_writes_png(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "time_step": [0, 1],
            "mean": [2.0, 4.0],
            "std": [1.0, 1.0],
            "min": [1.0, 3.0],
            "max": [3.0, 5.0],
            "median": [2.0, 4.0],
        }
    )
    output = tmp_path / "stats.png"
    written = render_stats_table_image(table, output)
    assert written == output
    assert output.exists()
    assert output.suffix == ".png"
