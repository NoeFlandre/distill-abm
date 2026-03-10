from __future__ import annotations

from pathlib import Path

from distill_abm.pipeline.full_case_review_csv import write_case_review_csv


def test_write_case_review_csv_preserves_header_order_and_rows(tmp_path: Path) -> None:
    target = tmp_path / "review.csv"
    rows = [
        {
            "plot_index": "context",
            "reporter_pattern": "",
            "plot_description": "",
            "trend_prompt_path": "context.json",
            "trend_output_path": "context.txt",
            "image_path": "",
            "table_csv_path": "",
            "success": "True",
            "error": "",
            "validation_status": "accepted",
        },
        {
            "plot_index": "1",
            "reporter_pattern": "wolves",
            "plot_description": "Wolf population",
            "trend_prompt_path": "trend.json",
            "trend_output_path": "trend.txt",
            "image_path": "plot.png",
            "table_csv_path": "table.csv",
            "success": "False",
            "error": "boom",
            "validation_status": "retry",
        },
    ]

    write_case_review_csv(target, rows)

    assert target.read_text(encoding="utf-8").splitlines() == [
        (
            "plot_index,reporter_pattern,plot_description,trend_prompt_path,"
            "trend_output_path,image_path,table_csv_path,success,error,validation_status"
        ),
        "context,,,context.json,context.txt,,,True,,accepted",
        "1,wolves,Wolf population,trend.json,trend.txt,plot.png,table.csv,False,boom,retry",
    ]
