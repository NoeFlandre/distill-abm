"""Shared review CSV helpers for full-case smoke workflows."""

from __future__ import annotations

import csv
from pathlib import Path

CASE_REVIEW_FIELDNAMES = [
    "plot_index",
    "reporter_pattern",
    "plot_description",
    "trend_prompt_path",
    "trend_output_path",
    "image_path",
    "table_csv_path",
    "success",
    "error",
    "validation_status",
]


def write_case_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the stable per-case review CSV contract used by full-case smoke workflows."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_REVIEW_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
