"""Helpers for stable run-level review artifacts in full-case matrix smoke."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from distill_abm.pipeline.run_artifact_contracts import case_summary_path, validation_state_path

if TYPE_CHECKING:
    from distill_abm.pipeline.full_case_matrix_smoke import FullCaseMatrixCaseResult

RUN_REVIEW_FIELDNAMES = [
    "case_id",
    "abm",
    "evidence_mode",
    "prompt_variant",
    "repetition",
    "case_dir",
    "case_summary_path",
    "review_csv_path",
    "validation_state_path",
    "success",
    "resumed_from_existing",
    "error",
]


def build_run_review_rows(case_results: list[FullCaseMatrixCaseResult]) -> list[dict[str, str]]:
    """Serialize full-case matrix case results into stable run-level review rows."""

    return [
        {
            "case_id": case.case_id,
            "abm": case.abm,
            "evidence_mode": case.evidence_mode,
            "prompt_variant": case.prompt_variant,
            "repetition": str(case.repetition),
            "case_dir": str(case.case_dir),
            "case_summary_path": str(case_summary_path(case.case_dir)),
            "review_csv_path": str(case.case_dir / "review.csv"),
            "validation_state_path": str(validation_state_path(case.case_dir)),
            "success": str(case.success),
            "resumed_from_existing": str(case.resumed_from_existing),
            "error": case.error or "",
        }
        for case in case_results
    ]


def write_run_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the stable matrix run-level review CSV."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_REVIEW_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
