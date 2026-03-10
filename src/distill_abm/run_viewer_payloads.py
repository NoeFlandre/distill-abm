"""Payload builders for the static run review HTML viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from distill_abm.pipeline.run_artifact_contracts import (
    CASE_SUMMARY_FILENAME,
    FULL_CASE_MATRIX_REPORT_FILENAME,
    RUN_LOG_FILENAME,
    SAMPLED_SMOKE_REPORT_FILENAME,
    VALIDATION_STATE_FILENAME,
)

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


def build_viewer_payload(run_root: Path) -> JsonObject:
    """Build the serialized payload consumed by the static HTML viewer."""
    report_path = resolve_report_path(run_root)
    report: JsonObject = read_optional_json(report_path) if report_path.exists() else {}
    run_log_path = run_root / RUN_LOG_FILENAME
    cases_root = run_root / "cases"
    cases: list[JsonValue] = []
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        if (case_dir / "03_outputs").exists():
            cases.append(build_sample_case_payload(run_root=run_root, report=report, case_dir=case_dir))
        elif (case_dir / "03_trends").exists():
            cases.append(build_full_case_payload(run_root=run_root, report=report, case_dir=case_dir))
    return {
        "run_root": str(run_root),
        "report_path": str(report_path) if report_path.exists() else "",
        "run_log_path": str(run_log_path) if run_log_path.exists() else "",
        "success": report.get("success"),
        "failed_case_ids": report.get("failed_case_ids", []),
        "started_at_utc": report.get("started_at_utc", ""),
        "finished_at_utc": report.get("finished_at_utc", ""),
        "cases": cast(JsonValue, cases),
    }


def resolve_report_path(run_root: Path) -> Path:
    """Resolve the sampled or full-case report for a run root."""
    for name in (SAMPLED_SMOKE_REPORT_FILENAME, FULL_CASE_MATRIX_REPORT_FILENAME):
        candidate = run_root / name
        if candidate.exists():
            return candidate
    return run_root / SAMPLED_SMOKE_REPORT_FILENAME


def build_sample_case_payload(*, run_root: Path, report: JsonObject, case_dir: Path) -> JsonObject:
    """Build the viewer payload for a sampled smoke case."""
    case_summary = read_optional_json(case_dir / CASE_SUMMARY_FILENAME)
    case_id = _get_string(case_summary, "case_id", default=case_dir.name)
    inputs_dir = case_dir / "01_inputs"
    requests_dir = case_dir / "02_requests"
    outputs_dir = case_dir / "03_outputs"
    return {
        "case_id": case_id,
        "abm": _get_string(case_summary, "abm"),
        "evidence_mode": _get_string(case_summary, "evidence_mode"),
        "prompt_variant": _get_string(case_summary, "prompt_variant"),
        "model": _get_string(case_summary, "model"),
        "resumed_from_existing": lookup_resumed_flag(
            report=report,
            case_id=case_id,
        ),
        "success": not (outputs_dir / "error.txt").exists(),
        "error_text": read_optional_text(outputs_dir / "error.txt"),
        "paths": {
            "case_dir": relative_path(run_root, case_dir),
            "context_prompt": relative_path(run_root, inputs_dir / "context_prompt.txt"),
            "documentation": relative_path(run_root, inputs_dir / "documentation.txt"),
            "parameters": relative_path(run_root, inputs_dir / "parameters.txt"),
            "trend_prompt": relative_path(run_root, inputs_dir / "trend_prompt.txt"),
            "table_csv": relative_path(run_root, inputs_dir / "trend_evidence_table.txt"),
            "table_series_csv": relative_path(run_root, inputs_dir / "trend_evidence_table_series.csv"),
            "table_json": relative_path(run_root, inputs_dir / "trend_evidence_table.json"),
            "image": relative_path(run_root, inputs_dir / "trend_evidence_plot.png"),
            "context_output": relative_path(run_root, outputs_dir / "context_output.txt"),
            "trend_output": relative_path(run_root, outputs_dir / "trend_output.txt"),
            "hyperparameters": relative_path(run_root, requests_dir / "hyperparameters.json"),
            "context_trace": relative_path(run_root, outputs_dir / "context_trace.json"),
            "trend_trace": relative_path(run_root, outputs_dir / "trend_trace.json"),
        },
        "context_prompt_text": read_optional_text(inputs_dir / "context_prompt.txt"),
        "documentation_text": read_optional_text(inputs_dir / "documentation.txt"),
        "parameters_text": read_optional_text(inputs_dir / "parameters.txt"),
        "trend_prompt_text": read_optional_text(inputs_dir / "trend_prompt.txt"),
        "table_csv_text": read_optional_text(inputs_dir / "trend_evidence_table.txt"),
        "context_output_text": read_optional_text(outputs_dir / "context_output.txt"),
        "trend_output_text": read_optional_text(outputs_dir / "trend_output.txt"),
        "hyperparameters_text": read_optional_text(requests_dir / "hyperparameters.json"),
        "trends": cast(JsonValue, []),
    }


def build_full_case_payload(*, run_root: Path, report: JsonObject, case_dir: Path) -> JsonObject:
    """Build the viewer payload for a full-case smoke case."""
    case_summary = read_optional_json(case_dir / CASE_SUMMARY_FILENAME)
    case_id = _get_string(case_summary, "case_id", default=case_dir.name)
    inputs_dir = case_dir / "01_inputs"
    context_dir = case_dir / "02_context"
    trends_root = case_dir / "03_trends"
    trend_entries: list[JsonObject] = []
    for trend_dir in sorted(path for path in trends_root.iterdir() if path.is_dir()):
        trend_entries.append(
            {
                "plot_id": trend_dir.name,
                "trend_prompt_path": relative_path(run_root, trend_dir / "trend_prompt.txt"),
                "trend_prompt_text": read_optional_text(trend_dir / "trend_prompt.txt"),
                "trend_output_path": relative_path(run_root, trend_dir / "trend_output.txt"),
                "trend_output_text": read_optional_text(trend_dir / "trend_output.txt"),
                "table_csv_path": relative_path(run_root, trend_dir / "trend_evidence_table.txt"),
                "table_csv_text": read_optional_text(trend_dir / "trend_evidence_table.txt"),
                "table_series_csv_path": relative_path(run_root, trend_dir / "trend_evidence_table_series.csv"),
                "table_json_path": relative_path(run_root, trend_dir / "trend_evidence_table.json"),
                "image_path": relative_path(run_root, trend_dir / "trend_evidence_plot.png"),
                "trend_trace_path": relative_path(run_root, trend_dir / "trend_trace.json"),
                "error_text": read_optional_text(trend_dir / "error.txt"),
            }
        )
    success = not (context_dir / "error.txt").exists() and all(
        not _get_string(item, "error_text") for item in trend_entries
    )
    return {
        "case_id": case_id,
        "abm": _get_string(case_summary, "abm"),
        "evidence_mode": _get_string(case_summary, "evidence_mode"),
        "prompt_variant": _get_string(case_summary, "prompt_variant"),
        "model": _get_string(case_summary, "model"),
        "resumed_from_existing": lookup_resumed_flag(
            report=report,
            case_id=case_id,
        ),
        "success": success,
        "error_text": read_optional_text(context_dir / "error.txt"),
        "paths": {
            "case_dir": relative_path(run_root, case_dir),
            "context_prompt": relative_path(run_root, inputs_dir / "context_prompt.txt"),
            "documentation": relative_path(run_root, inputs_dir / "documentation.txt"),
            "parameters": relative_path(run_root, inputs_dir / "parameters.txt"),
            "context_output": relative_path(run_root, context_dir / "context_output.txt"),
            "context_trace": relative_path(run_root, context_dir / "context_trace.json"),
            "review_csv": relative_path(run_root, case_dir / "review.csv"),
            "validation_state": relative_path(run_root, case_dir / VALIDATION_STATE_FILENAME),
        },
        "context_prompt_text": read_optional_text(inputs_dir / "context_prompt.txt"),
        "documentation_text": read_optional_text(inputs_dir / "documentation.txt"),
        "parameters_text": read_optional_text(inputs_dir / "parameters.txt"),
        "trend_prompt_text": "",
        "table_csv_text": "",
        "context_output_text": read_optional_text(context_dir / "context_output.txt"),
        "trend_output_text": "",
        "hyperparameters_text": "",
        "trends": cast(JsonValue, trend_entries),
    }


def lookup_resumed_flag(*, report: JsonObject, case_id: str) -> bool:
    """Look up whether a case was reused from a previous run."""
    cases = report.get("cases")
    if not isinstance(cases, list):
        return False
    for case in cases:
        if isinstance(case, dict) and case.get("case_id") == case_id:
            return bool(case.get("resumed_from_existing"))
    return False


def read_optional_text(path: Path) -> str:
    """Return file text or an empty string when absent."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_optional_json(path: Path) -> JsonObject:
    """Return decoded JSON or an empty mapping when absent."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return cast(JsonObject, payload)


def relative_path(run_root: Path, path: Path) -> str:
    """Return a run-relative path when the target exists."""
    if not path.exists():
        return ""
    return path.relative_to(run_root).as_posix()


def _get_string(payload: JsonObject, key: str, *, default: str = "") -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else default
