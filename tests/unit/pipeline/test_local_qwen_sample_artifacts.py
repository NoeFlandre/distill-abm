from __future__ import annotations

import json
from pathlib import Path

from distill_abm.pipeline.local_qwen_sample_artifacts import (
    build_review_row_from_existing,
    copy_resumable_case_result,
    resolve_case_num_ctx,
    resolve_previous_run_root,
)


def _write_case_bundle(case_dir: Path) -> None:
    (case_dir / "01_inputs").mkdir(parents=True)
    (case_dir / "02_requests").mkdir(parents=True)
    (case_dir / "03_outputs").mkdir(parents=True)
    (case_dir / "00_case_summary.json").write_text(
        json.dumps(
            {
                "case_id": "case_01",
                "abm": "fauna",
                "evidence_mode": "plot",
                "prompt_variant": "none",
                "model": "m",
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "01_inputs" / "context_prompt.txt").write_text("ctx", encoding="utf-8")
    (case_dir / "01_inputs" / "trend_prompt.txt").write_text("trend", encoding="utf-8")
    (case_dir / "01_inputs" / "parameters.txt").write_text("params", encoding="utf-8")
    (case_dir / "01_inputs" / "documentation.txt").write_text("docs", encoding="utf-8")
    (case_dir / "01_inputs" / "trend_evidence_plot.png").write_bytes(b"png")
    (case_dir / "02_requests" / "context_request.json").write_text("{}", encoding="utf-8")
    (case_dir / "02_requests" / "trend_request.json").write_text("{}", encoding="utf-8")
    (case_dir / "02_requests" / "hyperparameters.json").write_text("{}", encoding="utf-8")
    (case_dir / "03_outputs" / "context_output.txt").write_text("ctx out", encoding="utf-8")
    (case_dir / "03_outputs" / "context_trace.json").write_text("{}", encoding="utf-8")
    (case_dir / "03_outputs" / "trend_output.txt").write_text("trend out", encoding="utf-8")
    (case_dir / "03_outputs" / "trend_trace.json").write_text("{}", encoding="utf-8")


def test_resolve_previous_run_root_prefers_previous_run_directory(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    older_run = output_root / "runs" / "run_older"
    current_run = output_root / "runs" / "run_current"
    older_run.mkdir(parents=True)
    current_run.mkdir(parents=True)

    resolved = resolve_previous_run_root(output_root=output_root, current_run_id="run_current")

    assert resolved == older_run


def test_resolve_previous_run_root_falls_back_to_legacy_layout(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    (output_root / "cases").mkdir(parents=True)

    resolved = resolve_previous_run_root(output_root=output_root, current_run_id="run_current")

    assert resolved == output_root


def test_copy_resumable_case_result_copies_complete_successful_case(tmp_path: Path) -> None:
    previous_run_root = tmp_path / "prev"
    source_case_dir = previous_run_root / "cases" / "case_01"
    _write_case_bundle(source_case_dir)
    destination_case_dir = tmp_path / "dest" / "case_01"

    copied = copy_resumable_case_result(
        case_id="case_01",
        destination_case_dir=destination_case_dir,
        previous_run_root=previous_run_root,
    )

    assert copied is True
    assert (destination_case_dir / "03_outputs" / "trend_output.txt").read_text(encoding="utf-8") == "trend out"


def test_copy_resumable_case_result_rejects_failed_case(tmp_path: Path) -> None:
    previous_run_root = tmp_path / "prev"
    source_case_dir = previous_run_root / "cases" / "case_01"
    _write_case_bundle(source_case_dir)
    (source_case_dir / "03_outputs" / "error.txt").write_text("boom", encoding="utf-8")
    destination_case_dir = tmp_path / "dest" / "case_01"

    copied = copy_resumable_case_result(
        case_id="case_01",
        destination_case_dir=destination_case_dir,
        previous_run_root=previous_run_root,
    )

    assert copied is False


def test_build_review_row_from_existing_reads_saved_artifacts(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_01"
    _write_case_bundle(case_dir)

    row = build_review_row_from_existing(case_dir)

    assert row["case_id"] == "case_01"
    assert row["image_path"].endswith("trend_evidence_plot.png")
    assert row["trend_output_text"] == "trend out"


def test_resolve_case_num_ctx_uses_mode_override_when_positive() -> None:
    resolved = resolve_case_num_ctx("plot+table", 8192, {"plot": 4096, "plot+table": 16384})

    assert resolved == 16384


def test_resolve_case_num_ctx_ignores_non_positive_override() -> None:
    resolved = resolve_case_num_ctx("table", 8192, {"table": 0})

    assert resolved == 8192
