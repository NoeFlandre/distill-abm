from __future__ import annotations

from pathlib import Path

from distill_abm.pipeline.smoke_manifests import load_resumable_case, write_case_manifest
from distill_abm.pipeline.smoke_types import SmokeCase, SmokeCaseResult


def test_write_case_manifest_sets_manifest_path_and_writes_json(tmp_path: Path) -> None:
    case_result = SmokeCaseResult(
        case=SmokeCase(case_id="case-1", evidence_mode="plot", text_source_mode="summary_only"),
        status="ok",
        output_dir=tmp_path / "case-1",
    )
    case_result.output_dir.mkdir(parents=True)

    written = write_case_manifest(case_result)

    assert written.case_manifest_path == case_result.output_dir / "case_manifest.json"
    assert written.case_manifest_path.exists()


def test_load_resumable_case_returns_none_when_manifest_missing(tmp_path: Path) -> None:
    assert load_resumable_case(tmp_path / "missing.json") is None


def test_load_resumable_case_returns_ok_case(tmp_path: Path) -> None:
    case_dir = tmp_path / "case-1"
    case_dir.mkdir()
    manifest_path = case_dir / "case_manifest.json"
    case_result = SmokeCaseResult(
        case=SmokeCase(case_id="case-1", evidence_mode="plot", text_source_mode="summary_only"),
        status="ok",
        output_dir=case_dir,
    )
    write_case_manifest(case_result)

    loaded = load_resumable_case(manifest_path)

    assert loaded is not None
    assert loaded.status == "ok"
    assert loaded.case_manifest_path == manifest_path


def test_load_resumable_case_rejects_failed_case(tmp_path: Path) -> None:
    case_dir = tmp_path / "case-1"
    case_dir.mkdir()
    case_result = SmokeCaseResult(
        case=SmokeCase(case_id="case-1", evidence_mode="plot", text_source_mode="summary_only"),
        status="failed",
        output_dir=case_dir,
    )
    write_case_manifest(case_result)

    assert load_resumable_case(case_dir / "case_manifest.json") is None
