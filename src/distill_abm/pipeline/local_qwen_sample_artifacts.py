"""Artifact and run-state helpers for sampled LLM smoke runs."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from distill_abm.pipeline.run_artifact_contracts import CASE_SUMMARY_FILENAME


def resolve_previous_run_root(*, output_root: Path, current_run_id: str) -> Path | None:
    """Return the most recent previous run root, including legacy flat layouts."""
    runs_root = output_root / "runs"
    if not runs_root.exists():
        legacy_cases_root = output_root / "cases"
        if legacy_cases_root.exists():
            return output_root
        return None
    candidates = sorted(
        (path for path in runs_root.iterdir() if path.is_dir() and path.name != current_run_id),
        reverse=True,
    )
    if candidates:
        return candidates[0]
    legacy_cases_root = output_root / "cases"
    if legacy_cases_root.exists():
        return output_root
    return None


def copy_resumable_case_result(
    *,
    case_id: str,
    destination_case_dir: Path,
    previous_run_root: Path | None,
) -> bool:
    """Copy a complete successful case bundle from a previous run if available."""
    if previous_run_root is None:
        return False
    source_case_dir = previous_run_root / "cases" / case_id
    required_paths = (
        source_case_dir / CASE_SUMMARY_FILENAME,
        source_case_dir / "01_inputs" / "context_prompt.txt",
        source_case_dir / "01_inputs" / "trend_prompt.txt",
        source_case_dir / "02_requests" / "context_request.json",
        source_case_dir / "02_requests" / "trend_request.json",
        source_case_dir / "02_requests" / "hyperparameters.json",
        source_case_dir / "03_outputs" / "context_output.txt",
        source_case_dir / "03_outputs" / "context_trace.json",
        source_case_dir / "03_outputs" / "trend_output.txt",
        source_case_dir / "03_outputs" / "trend_trace.json",
    )
    if any(not path.exists() for path in required_paths):
        return False
    if (source_case_dir / "03_outputs" / "error.txt").exists():
        return False
    if destination_case_dir.exists():
        shutil.rmtree(destination_case_dir)
    shutil.copytree(source_case_dir, destination_case_dir)
    return True


def build_review_row_from_existing(case_dir: Path) -> dict[str, str]:
    """Load a previously written case bundle into one review CSV row."""
    inputs_dir = case_dir / "01_inputs"
    outputs_dir = case_dir / "03_outputs"
    requests_dir = case_dir / "02_requests"
    summary_path = case_dir / CASE_SUMMARY_FILENAME
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    image_path = inputs_dir / "trend_evidence_plot.png"
    table_path = inputs_dir / "trend_evidence_table.csv"
    return {
        "case_id": str(summary_payload["case_id"]),
        "abm": str(summary_payload["abm"]),
        "evidence_mode": str(summary_payload["evidence_mode"]),
        "prompt_variant": str(summary_payload["prompt_variant"]),
        "model": str(summary_payload["model"]),
        "case_summary_path": str(summary_path),
        "context_prompt_path": str(inputs_dir / "context_prompt.txt"),
        "context_prompt_text": (inputs_dir / "context_prompt.txt").read_text(encoding="utf-8"),
        "trend_prompt_path": str(inputs_dir / "trend_prompt.txt"),
        "trend_prompt_text": (inputs_dir / "trend_prompt.txt").read_text(encoding="utf-8"),
        "image_path": str(image_path if image_path.exists() else ""),
        "table_csv_path": str(table_path if table_path.exists() else ""),
        "parameters_path": str(inputs_dir / "parameters.txt"),
        "documentation_path": str(inputs_dir / "documentation.txt"),
        "hyperparameters_path": str(requests_dir / "hyperparameters.json"),
        "context_output_path": str(outputs_dir / "context_output.txt"),
        "context_output_text": (outputs_dir / "context_output.txt").read_text(encoding="utf-8"),
        "trend_output_path": str(outputs_dir / "trend_output.txt"),
        "trend_output_text": (outputs_dir / "trend_output.txt").read_text(encoding="utf-8"),
    }


def resolve_case_num_ctx(
    evidence_mode: str,
    default_num_ctx: int,
    num_ctx_by_mode: Mapping[str, int] | None,
) -> int:
    """Resolve the per-evidence-mode context window with a safe default fallback."""
    if not num_ctx_by_mode:
        return default_num_ctx
    resolved = num_ctx_by_mode.get(evidence_mode, default_num_ctx)
    return resolved if resolved > 0 else default_num_ctx
