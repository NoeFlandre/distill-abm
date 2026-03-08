"""Manifest helpers for resumable smoke workflows."""

from __future__ import annotations

from pathlib import Path

from distill_abm.pipeline.smoke_types import SmokeCaseResult


def write_case_manifest(case_result: SmokeCaseResult) -> SmokeCaseResult:
    """Persist one case manifest next to the case artifacts."""
    manifest_path = case_result.output_dir / "case_manifest.json"
    case_result.case_manifest_path = manifest_path
    manifest_path.write_text(case_result.model_dump_json(indent=2), encoding="utf-8")
    return case_result


def load_resumable_case(case_manifest: Path) -> SmokeCaseResult | None:
    """Load one previous successful case manifest for resume."""
    if not case_manifest.exists():
        return None
    try:
        loaded = SmokeCaseResult.model_validate_json(case_manifest.read_text(encoding="utf-8"))
    except Exception:
        return None
    if loaded.status != "ok":
        return None
    loaded.case_manifest_path = case_manifest
    return loaded
