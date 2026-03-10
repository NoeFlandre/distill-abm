from __future__ import annotations

from pathlib import Path

from distill_abm.pipeline.full_case_matrix_run_review import build_run_review_rows, write_run_review_csv
from distill_abm.pipeline.full_case_matrix_smoke import FullCaseMatrixCaseResult


def test_build_run_review_rows_and_write_csv_preserve_contract(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / "01_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    case_results = [
        FullCaseMatrixCaseResult(
            case_id="01_case",
            abm="grazing",
            evidence_mode="plot",
            prompt_variant="role",
            repetition=1,
            case_dir=case_dir,
            success=False,
            resumed_from_existing=True,
            error="boom",
        )
    ]

    rows = build_run_review_rows(case_results)

    assert rows == [
        {
            "case_id": "01_case",
            "abm": "grazing",
            "evidence_mode": "plot",
            "prompt_variant": "role",
            "repetition": "1",
            "case_dir": str(case_dir),
            "case_summary_path": str(case_dir / "00_case_summary.json"),
            "review_csv_path": str(case_dir / "review.csv"),
            "validation_state_path": str(case_dir / "validation_state.json"),
            "success": "False",
            "resumed_from_existing": "True",
            "error": "boom",
        }
    ]

    target = tmp_path / "request_review.csv"
    write_run_review_csv(target, rows)

    assert target.read_text(encoding="utf-8").splitlines() == [
        (
            "case_id,abm,evidence_mode,prompt_variant,repetition,case_dir,case_summary_path,"
            "review_csv_path,validation_state_path,success,resumed_from_existing,error"
        ),
        (
            f"01_case,grazing,plot,role,1,{case_dir},{case_dir / '00_case_summary.json'},"
            f"{case_dir / 'review.csv'},{case_dir / 'validation_state.json'},False,True,boom"
        ),
    ]
