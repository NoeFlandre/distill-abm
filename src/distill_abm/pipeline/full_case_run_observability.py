"""Additive run-level observability artifacts for full-case smoke workflows."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from distill_abm.pipeline.full_case_matrix_smoke import FullCaseMatrixCaseResult

RUN_OBSERVABILITY_FILENAME = "run_observability.csv"
RUN_OBSERVABILITY_SUMMARY_JSON_FILENAME = "run_observability_summary.json"
RUN_OBSERVABILITY_SUMMARY_MARKDOWN_FILENAME = "run_observability_summary.md"

RUN_OBSERVABILITY_COLUMNS: tuple[str, ...] = (
    "run_root",
    "request_kind",
    "case_id",
    "abm",
    "evidence_mode",
    "prompt_variant",
    "repetition",
    "case_dir",
    "review_csv_path",
    "case_summary_path",
    "validation_state_path",
    "request_json_path",
    "trace_json_path",
    "output_path",
    "thinking_path",
    "plot_index",
    "reporter_pattern",
    "plot_description",
    "image_path",
    "table_csv_path",
    "success",
    "validation_status",
    "error",
    "resumed_from_existing",
    "reused_from_previous_run",
    "reused_from_shared_context_cache",
    "counts_toward_run_totals",
    "context_materialization_source",
    "provider",
    "model",
    "runtime_provider",
    "runtime_precision",
    "temperature",
    "max_tokens",
    "max_retries",
    "retry_backoff_seconds",
    "image_attached",
    "attempts_made",
    "trace_error_count",
    "prompt_length",
    "prompt_signature",
    "usage_prompt_tokens",
    "usage_completion_tokens",
    "usage_total_tokens",
    "table_downsample_stride",
    "compression_tier",
    "prompt_compression_applied",
    "request_metadata_json",
    "runtime_json",
    "usage_json",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_review_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _int_string(value: object) -> str:
    if isinstance(value, bool) or value is None or value == "":
        return ""
    try:
        return str(int(str(value)))
    except (TypeError, ValueError):
        return ""


def _bool_string(value: bool) -> str:
    return "true" if value else "false"


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _row_paths(*, case_dir: Path, plot_index: str, trend_output_path: str) -> tuple[Path, Path, Path, Path]:
    if plot_index == "context":
        base_dir = case_dir / "02_context"
        return (
            base_dir / "context_request.json",
            base_dir / "context_trace.json",
            base_dir / "context_output.txt",
            base_dir / "context_thinking.txt",
        )
    trend_dir = (
        Path(trend_output_path).parent
        if trend_output_path
        else case_dir / "03_trends" / f"plot_{int(plot_index):02d}"
    )
    return (
        trend_dir / "trend_request.json",
        trend_dir / "trend_trace.json",
        trend_dir / "trend_output.txt",
        trend_dir / "trend_thinking.txt",
    )


def _parse_run_started_at(run_root: Path) -> datetime | None:
    try:
        return datetime.strptime(run_root.name, "run_%Y%m%d_%H%M%S_%f").replace(tzinfo=UTC)
    except ValueError:
        return None


def _path_written_in_current_run(path: Path, *, run_started_at: datetime | None) -> bool:
    if not path.exists() or run_started_at is None:
        return False
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    return modified_at >= run_started_at


def _synthetic_context_review_row(*, case_dir: Path, case_error: str | None) -> dict[str, str] | None:
    context_dir = case_dir / "02_context"
    request_json_path = context_dir / "context_request.json"
    trace_json_path = context_dir / "context_trace.json"
    if not request_json_path.exists() and not trace_json_path.exists():
        return None
    error_path = context_dir / "error.txt"
    error_text = error_path.read_text(encoding="utf-8").strip() if error_path.exists() else (case_error or "")
    validation_state = _load_json(case_dir / "validation_state.json")
    context_state = _as_dict(validation_state.get("context"))
    validation_status = _stringify(context_state.get("status") or ("retry" if error_text else "accepted"))
    return {
        "plot_index": "context",
        "reporter_pattern": "",
        "plot_description": "",
        "trend_prompt_path": str(request_json_path if request_json_path.exists() else ""),
        "trend_output_path": str(
            (context_dir / "context_output.txt") if (context_dir / "context_output.txt").exists() else ""
        ),
        "image_path": "",
        "table_csv_path": "",
        "success": "false" if error_text else "true",
        "error": error_text,
        "validation_status": validation_status,
    }


def _build_observability_row(
    *,
    case_result: FullCaseMatrixCaseResult,
    run_root: Path,
    review_csv_path: Path,
    case_summary_path: Path,
    validation_state_path: Path,
    review_row: dict[str, str],
    run_started_at: datetime | None,
) -> dict[str, str]:
    case_dir = case_result.case_dir
    plot_index = review_row.get("plot_index", "")
    request_kind = "context" if plot_index == "context" else "trend"
    request_json_path, trace_json_path, output_path, thinking_path = _row_paths(
        case_dir=case_dir,
        plot_index=plot_index,
        trend_output_path=review_row.get("trend_output_path", ""),
    )
    request_payload = _load_json(request_json_path)
    trace_payload = _load_json(trace_json_path)
    request_block = _as_dict(trace_payload.get("request")) or request_payload
    response_block = _as_dict(trace_payload.get("response"))
    runtime_block = _as_dict(response_block.get("runtime"))
    usage_block = _as_dict(response_block.get("usage"))
    metadata_block = _as_dict(request_block.get("metadata"))
    stride = _int_string(metadata_block.get("table_downsample_stride"))
    compression_tier = str(max(int(stride) - 1, 0)) if stride else ""
    reused_from_previous_run_by_path = not _path_written_in_current_run(
        trace_json_path if trace_json_path.exists() else request_json_path,
        run_started_at=run_started_at,
    )
    context_materialization_source = ""
    reused_from_previous_run = reused_from_previous_run_by_path
    reused_from_shared_context_cache = False
    counts_toward_run_totals = not reused_from_previous_run_by_path
    if request_kind == "context":
        context_materialization_source = _stringify(metadata_block.get("context_materialization_source"))
        if context_materialization_source == "fresh_request":
            reused_from_previous_run = False
            counts_toward_run_totals = True
        elif context_materialization_source == "shared_context_cache":
            reused_from_previous_run = False
            reused_from_shared_context_cache = True
            counts_toward_run_totals = False
        elif context_materialization_source == "resumed_previous_run":
            reused_from_previous_run = True
            counts_toward_run_totals = False
    return {
        "run_root": str(run_root),
        "request_kind": request_kind,
        "case_id": case_result.case_id,
        "abm": case_result.abm,
        "evidence_mode": case_result.evidence_mode,
        "prompt_variant": case_result.prompt_variant,
        "repetition": str(case_result.repetition),
        "case_dir": str(case_dir),
        "review_csv_path": str(review_csv_path),
        "case_summary_path": str(case_summary_path),
        "validation_state_path": str(validation_state_path),
        "request_json_path": str(request_json_path),
        "trace_json_path": str(trace_json_path),
        "output_path": str(output_path),
        "thinking_path": str(thinking_path),
        "plot_index": plot_index,
        "reporter_pattern": review_row.get("reporter_pattern", ""),
        "plot_description": review_row.get("plot_description", ""),
        "image_path": review_row.get("image_path", ""),
        "table_csv_path": review_row.get("table_csv_path", ""),
        "success": review_row.get("success", ""),
        "validation_status": review_row.get("validation_status", ""),
        "error": review_row.get("error", ""),
        "resumed_from_existing": _bool_string(case_result.resumed_from_existing),
        "reused_from_previous_run": _bool_string(reused_from_previous_run),
        "reused_from_shared_context_cache": _bool_string(reused_from_shared_context_cache),
        "counts_toward_run_totals": _bool_string(counts_toward_run_totals),
        "context_materialization_source": context_materialization_source,
        "provider": _stringify(request_block.get("provider")),
        "model": _stringify(request_block.get("model")),
        "runtime_provider": _stringify(runtime_block.get("provider")),
        "runtime_precision": _stringify(runtime_block.get("precision")),
        "temperature": _stringify(request_block.get("temperature")),
        "max_tokens": _stringify(request_block.get("max_tokens")),
        "max_retries": _stringify(request_block.get("max_retries")),
        "retry_backoff_seconds": _stringify(request_block.get("retry_backoff_seconds")),
        "image_attached": _stringify(request_block.get("image_attached")),
        "attempts_made": _stringify(trace_payload.get("attempts_made")),
        "trace_error_count": (
            str(len(trace_payload.get("errors", []))) if isinstance(trace_payload.get("errors"), list) else ""
        ),
        "prompt_length": _stringify(request_block.get("prompt_length")),
        "prompt_signature": _stringify(request_block.get("prompt_signature")),
        "usage_prompt_tokens": _stringify(usage_block.get("prompt_tokens")),
        "usage_completion_tokens": _stringify(usage_block.get("completion_tokens")),
        "usage_total_tokens": _stringify(usage_block.get("total_tokens")),
        "table_downsample_stride": stride,
        "compression_tier": compression_tier,
        "prompt_compression_applied": (
            _bool_string(bool(compression_tier and int(compression_tier) > 0))
            if compression_tier
            else "false"
        ),
        "request_metadata_json": _stringify(metadata_block),
        "runtime_json": _stringify(runtime_block),
        "usage_json": _stringify(usage_block),
    }


def _mark_shared_context_reuse(rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("request_kind") != "context":
            continue
        if row.get("context_materialization_source"):
            continue
        if row.get("reused_from_previous_run") == "true":
            continue
        prompt_signature = row.get("prompt_signature", "")
        model = row.get("model", "")
        run_root = row.get("run_root", "")
        if not prompt_signature:
            continue
        grouped[(run_root, model, prompt_signature)].append(row)
    for grouped_rows in grouped.values():
        if len(grouped_rows) <= 1:
            continue
        grouped_rows.sort(key=lambda row: (row.get("case_id", ""), row.get("case_dir", "")))
        grouped_rows[0]["context_materialization_source"] = "fresh_request"
        for row in grouped_rows[1:]:
            row["reused_from_shared_context_cache"] = "true"
            row["counts_toward_run_totals"] = "false"
            row["context_materialization_source"] = "shared_context_cache"


def build_run_observability_rows(case_results: list[FullCaseMatrixCaseResult]) -> list[dict[str, str]]:
    """Build one additive observability row per context/trend request."""

    rows: list[dict[str, str]] = []
    for case_result in case_results:
        case_dir = case_result.case_dir
        run_root = case_dir.parent.parent
        run_started_at = _parse_run_started_at(run_root)
        review_csv_path = case_dir / "review.csv"
        case_summary_path = case_dir / "00_case_summary.json"
        validation_state_path = case_dir / "validation_state.json"
        review_rows = _read_review_rows(review_csv_path)
        if not any(review_row.get("plot_index") == "context" for review_row in review_rows):
            synthetic_context_row = _synthetic_context_review_row(case_dir=case_dir, case_error=case_result.error)
            if synthetic_context_row is not None:
                review_rows = [synthetic_context_row, *review_rows]
        for review_row in review_rows:
            rows.append(
                _build_observability_row(
                    case_result=case_result,
                    run_root=run_root,
                    review_csv_path=review_csv_path,
                    case_summary_path=case_summary_path,
                    validation_state_path=validation_state_path,
                    review_row=review_row,
                    run_started_at=run_started_at,
                )
            )
    _mark_shared_context_reuse(rows)
    return rows


def write_run_observability_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_OBSERVABILITY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_run_observability_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _counts_toward_run_totals(row: dict[str, str]) -> bool:
    value = row.get("counts_toward_run_totals")
    if value:
        return value == "true"
    if row.get("reused_from_previous_run") == "true":
        return False
    if row.get("reused_from_shared_context_cache") == "true":
        return False
    return True


def build_run_observability_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    counted_rows = [row for row in rows if _counts_toward_run_totals(row)]
    request_counts_by_kind = Counter(row.get("request_kind", "") for row in counted_rows if row.get("request_kind"))
    request_counts_by_abm = Counter(row.get("abm", "") for row in counted_rows if row.get("abm"))
    compression_counts = Counter(row.get("compression_tier", "") for row in counted_rows if row.get("compression_tier"))
    usage_totals: dict[str, int] = defaultdict(int)
    runtime_provider_count = 0
    runtime_precision_count = 0
    prompt_compression_count = 0
    resumed_request_count = 0
    reused_request_count = len(rows) - len(counted_rows)
    for row in counted_rows:
        if row.get("runtime_provider"):
            runtime_provider_count += 1
        if row.get("runtime_precision"):
            runtime_precision_count += 1
        if row.get("prompt_compression_applied") == "true":
            prompt_compression_count += 1
        if row.get("resumed_from_existing") == "true":
            resumed_request_count += 1
        for field, key in (
            ("usage_prompt_tokens", "prompt_tokens"),
            ("usage_completion_tokens", "completion_tokens"),
            ("usage_total_tokens", "total_tokens"),
        ):
            try:
                usage_totals[key] += int(row.get(field, ""))
            except (TypeError, ValueError):
                continue
    return {
        "observed_row_count": len(rows),
        "request_count": len(counted_rows),
        "reused_request_count": reused_request_count,
        "request_counts_by_kind": dict(sorted(request_counts_by_kind.items())),
        "request_counts_by_abm": dict(sorted(request_counts_by_abm.items())),
        "providers": sorted({row.get("provider", "") for row in counted_rows if row.get("provider")}),
        "models": sorted({row.get("model", "") for row in counted_rows if row.get("model")}),
        "runtime_providers": sorted(
            {row.get("runtime_provider", "") for row in counted_rows if row.get("runtime_provider")}
        ),
        "runtime_precisions": sorted(
            {row.get("runtime_precision", "") for row in counted_rows if row.get("runtime_precision")}
        ),
        "temperatures": sorted({row.get("temperature", "") for row in counted_rows if row.get("temperature")}),
        "max_tokens": sorted({row.get("max_tokens", "") for row in counted_rows if row.get("max_tokens")}),
        "retry_settings": {
            "max_retries": sorted({row.get("max_retries", "") for row in counted_rows if row.get("max_retries")}),
            "retry_backoff_seconds": sorted(
                {row.get("retry_backoff_seconds", "") for row in counted_rows if row.get("retry_backoff_seconds")}
            ),
        },
        "compression": {
            "request_count_with_compression": prompt_compression_count,
            "table_downsample_strides_used": sorted(
                {
                    row.get("table_downsample_stride", "")
                    for row in counted_rows
                    if row.get("table_downsample_stride")
                },
                key=int,
            ),
            "compression_tiers_used": sorted(
                {row.get("compression_tier", "") for row in counted_rows if row.get("compression_tier")},
                key=int,
            ),
            "request_counts_by_tier": dict(sorted(compression_counts.items(), key=lambda item: int(item[0]))),
        },
        "usage_totals": dict(usage_totals),
        "resumed_request_count": resumed_request_count,
        "observability_coverage": {
            "requests_with_runtime_provider": runtime_provider_count,
            "requests_with_runtime_precision": runtime_precision_count,
        },
    }


def render_run_observability_summary(summary: dict[str, Any], *, csv_path: Path) -> str:
    compression = summary.get("compression", {})
    retry_settings = summary.get("retry_settings", {})
    usage_totals = summary.get("usage_totals", {})
    coverage = summary.get("observability_coverage", {})
    lines = [
        "# Run Observability Summary",
        "",
        f"- observed_row_count: `{summary.get('observed_row_count', 0)}`",
        f"- request_count: `{summary.get('request_count', 0)}`",
        f"- reused_request_count: `{summary.get('reused_request_count', 0)}`",
        f"- request_counts_by_kind: `{json.dumps(summary.get('request_counts_by_kind', {}), sort_keys=True)}`",
        f"- request_counts_by_abm: `{json.dumps(summary.get('request_counts_by_abm', {}), sort_keys=True)}`",
        f"- providers: `{json.dumps(summary.get('providers', []))}`",
        f"- models: `{json.dumps(summary.get('models', []))}`",
        f"- runtime_providers: `{json.dumps(summary.get('runtime_providers', []))}`",
        f"- runtime_precisions: `{json.dumps(summary.get('runtime_precisions', []))}`",
        f"- temperatures: `{json.dumps(summary.get('temperatures', []))}`",
        f"- max_tokens: `{json.dumps(summary.get('max_tokens', []))}`",
        f"- retry_settings: `{json.dumps(retry_settings, sort_keys=True)}`",
        f"- compression: `{json.dumps(compression, sort_keys=True)}`",
        f"- usage_totals: `{json.dumps(usage_totals, sort_keys=True)}`",
        f"- observability_coverage: `{json.dumps(coverage, sort_keys=True)}`",
        f"- resumed_request_count: `{summary.get('resumed_request_count', 0)}`",
        f"- run_observability_csv: `{csv_path}`",
        "",
    ]
    return "\n".join(lines)


def write_run_observability_artifacts(
    *,
    run_root: Path,
    rows: list[dict[str, str]],
) -> tuple[Path, Path, Path]:
    csv_path = run_root / RUN_OBSERVABILITY_FILENAME
    summary_json_path = run_root / RUN_OBSERVABILITY_SUMMARY_JSON_FILENAME
    summary_markdown_path = run_root / RUN_OBSERVABILITY_SUMMARY_MARKDOWN_FILENAME
    summary = build_run_observability_summary(rows)
    write_run_observability_csv(csv_path, rows)
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary_markdown_path.write_text(
        render_run_observability_summary(summary, csv_path=csv_path),
        encoding="utf-8",
    )
    return csv_path, summary_json_path, summary_markdown_path
