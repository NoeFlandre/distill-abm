"""Reviewer-friendly full-case smoke for one real context plus all plot trends."""

from __future__ import annotations

import csv
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from distill_abm.llm.adapters.base import LLMAdapter
from distill_abm.pipeline.doe_smoke_prompts import (
    build_legacy_doe_context_prompt,
    build_legacy_doe_trend_prompt,
    build_raw_table_csv,
)
from distill_abm.pipeline.helpers import encode_image
from distill_abm.pipeline.local_qwen_sample_response import (
    StructuredSmokeResponseError,
    validate_structured_smoke_text_content,
)
from distill_abm.pipeline.local_qwen_sample_smoke import (
    _invoke_structured_smoke_text,
    _validate_case_inputs,
    _write_json,
    _write_optional_thinking,
    _write_text,
)
from distill_abm.pipeline.run_artifact_contracts import case_summary_path, validation_state_path

EvidenceMode = Literal["plot", "table", "plot+table"]


class FullCasePlotInput(BaseModel):
    """One ordered plot input for a full-case smoke."""

    plot_index: int
    reporter_pattern: str
    plot_description: str
    plot_path: Path


class FullCaseSmokeInput(BaseModel):
    """Resolved per-ABM inputs for one full context-plus-all-trends case."""

    abm: str
    csv_path: Path
    parameters_path: Path
    documentation_path: Path
    plots: tuple[FullCasePlotInput, ...]


class FullCaseTrendResult(BaseModel):
    """One trend call result inside a full case."""

    plot_index: int
    trend_dir: Path
    success: bool
    error: str | None = None


class FullCaseSmokeResult(BaseModel):
    """Top-level result for one full-case smoke."""

    started_at_utc: str
    finished_at_utc: str
    output_root: Path
    case_id: str
    case_dir: Path
    report_json_path: Path
    report_markdown_path: Path
    review_csv_path: Path
    success: bool
    failed_plot_indices: list[int] = Field(default_factory=list)
    trend_results: list[FullCaseTrendResult] = Field(default_factory=list)


class FullCaseValidationState(BaseModel):
    """Explicit resume/validation state persisted alongside one full-case smoke."""

    context: dict[str, object]
    trends: dict[str, dict[str, object]]


def run_full_case_smoke(
    *,
    case_input: FullCaseSmokeInput,
    adapter: LLMAdapter,
    model: str,
    output_root: Path,
    evidence_mode: EvidenceMode,
    prompt_variant: str,
    max_tokens: int = 32768,
    max_retries: int | None = None,
    retry_backoff_seconds: float | None = None,
    resume_existing: bool = True,
) -> FullCaseSmokeResult:
    """Run one real case through one context prompt and all trend prompts for that ABM."""
    started_at = datetime.now(UTC)
    output_root.mkdir(parents=True, exist_ok=True)
    case_id = f"01_{case_input.abm}_{prompt_variant}_{evidence_mode.replace('+', '_plus_')}_full_case"
    case_dir = output_root / "cases" / case_id
    validation_state_file = validation_state_path(case_dir)
    inputs_dir = case_dir / "01_inputs"
    context_dir = case_dir / "02_context"
    trends_dir = case_dir / "03_trends"
    for directory in (inputs_dir, context_dir, trends_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _validate_full_case_inputs(case_input)
    enabled = _enabled_features_from_variant(prompt_variant)
    shutil.copy2(case_input.parameters_path, inputs_dir / "parameters.txt")
    shutil.copy2(case_input.documentation_path, inputs_dir / "documentation.txt")
    context_prompt = build_legacy_doe_context_prompt(
        abm=case_input.abm,
        inputs_csv_path=case_input.parameters_path,
        inputs_doc_path=case_input.documentation_path,
        enabled=enabled,
    )
    _write_text(inputs_dir / "context_prompt.txt", context_prompt)

    review_rows: list[dict[str, str]] = []
    trend_results: list[FullCaseTrendResult] = []
    failed_plot_indices: list[int] = []
    validation_state = _load_validation_state(validation_state_file)
    _backfill_validation_state_from_artifacts(
        validation_state=validation_state,
        context_dir=context_dir,
        trends_dir=trends_dir,
        plots=case_input.plots,
    )
    context_text = ""
    if resume_existing and _is_context_accepted(context_dir=context_dir, validation_state=validation_state):
        context_text = (context_dir / "context_output.txt").read_text(encoding="utf-8")
        _record_context_review_row(review_rows=review_rows, context_dir=context_dir, validation_status="accepted")
    else:
        try:
            context_text, context_trace = _invoke_structured_smoke_text(
                adapter=adapter,
                model=model,
                prompt_with_schema=context_prompt,
                max_tokens=max_tokens,
                ollama_num_ctx=0,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
        except StructuredSmokeResponseError as exc:
            _write_json(context_dir / "context_request.json", exc.trace["request"])
            _write_json(context_dir / "context_trace.json", exc.trace)
            _write_optional_thinking(context_dir / "context_thinking.txt", exc.trace)
            _write_text(context_dir / "error.txt", str(exc))
            validation_state.context = {
                "status": "retry",
                "error": str(exc),
            }
            _write_json(validation_state_file, validation_state.model_dump(mode="json"))
            result = FullCaseSmokeResult(
                started_at_utc=started_at.isoformat(),
                finished_at_utc=datetime.now(UTC).isoformat(),
                output_root=output_root,
                case_id=case_id,
                case_dir=case_dir,
                report_json_path=output_root / "full_case_smoke_report.json",
                report_markdown_path=output_root / "full_case_smoke_report.md",
                review_csv_path=output_root / "review.csv",
                success=False,
                failed_plot_indices=[],
                trend_results=[],
            )
            _finalize_full_case_result(
                result=result,
                case_input=case_input,
                evidence_mode=evidence_mode,
                prompt_variant=prompt_variant,
                model=model,
                review_rows=review_rows,
                validation_state=validation_state,
            )
            return result

        _write_json(context_dir / "context_request.json", context_trace["request"])
        _write_text(context_dir / "context_output.txt", context_text)
        _write_json(context_dir / "context_trace.json", context_trace)
        _write_optional_thinking(context_dir / "context_thinking.txt", context_trace)
        validation_state.context = {
            "status": "accepted",
            "error": None,
        }
        if (context_dir / "error.txt").exists():
            (context_dir / "error.txt").unlink()
        _record_context_review_row(review_rows=review_rows, context_dir=context_dir, validation_status="accepted")

    frame = pd.read_csv(case_input.csv_path, sep=";")
    for plot_input in case_input.plots:
        trend_dir = trends_dir / f"plot_{plot_input.plot_index:02d}"
        trend_dir.mkdir(parents=True, exist_ok=True)
        trend_prompt = build_legacy_doe_trend_prompt(
            abm=case_input.abm,
            context_response=context_text,
            plot_description=plot_input.plot_description,
            evidence_mode=evidence_mode,
            table_csv=_build_trend_table_csv(
                frame=frame,
                evidence_mode=evidence_mode,
                plot_input=plot_input,
                trend_dir=trend_dir,
            ),
            enabled=enabled,
        )
        _write_text(trend_dir / "trend_prompt.txt", trend_prompt)
        image_b64: str | None = None
        image_path: Path | None = None
        if evidence_mode in {"plot", "plot+table"}:
            image_path = trend_dir / "trend_evidence_plot.png"
            shutil.copy2(plot_input.plot_path, image_path)
            image_b64 = encode_image(image_path)
        if resume_existing and _is_trend_accepted(
            plot_index=plot_input.plot_index,
            trend_dir=trend_dir,
            validation_state=validation_state,
        ):
            trend_results.append(
                FullCaseTrendResult(plot_index=plot_input.plot_index, trend_dir=trend_dir, success=True)
            )
            _record_trend_review_row(
                review_rows=review_rows,
                plot_input=plot_input,
                trend_dir=trend_dir,
                image_path=image_path,
                success=True,
                error="",
                validation_status="accepted",
            )
            continue
        try:
            trend_text, trend_trace = _invoke_structured_smoke_text(
                adapter=adapter,
                model=model,
                prompt_with_schema=trend_prompt,
                image_b64=image_b64,
                max_tokens=max_tokens,
                ollama_num_ctx=0,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            _write_json(trend_dir / "trend_request.json", trend_trace["request"])
            _write_text(trend_dir / "trend_output.txt", trend_text)
            _write_json(trend_dir / "trend_trace.json", trend_trace)
            _write_optional_thinking(trend_dir / "trend_thinking.txt", trend_trace)
            if (trend_dir / "error.txt").exists():
                (trend_dir / "error.txt").unlink()
            validation_state.trends[str(plot_input.plot_index)] = {
                "status": "accepted",
                "error": None,
            }
            trend_results.append(
                FullCaseTrendResult(plot_index=plot_input.plot_index, trend_dir=trend_dir, success=True)
            )
            _record_trend_review_row(
                review_rows=review_rows,
                plot_input=plot_input,
                trend_dir=trend_dir,
                image_path=image_path,
                success=True,
                error="",
                validation_status="accepted",
            )
        except StructuredSmokeResponseError as exc:
            _write_json(trend_dir / "trend_request.json", exc.trace["request"])
            _write_json(trend_dir / "trend_trace.json", exc.trace)
            _write_optional_thinking(trend_dir / "trend_thinking.txt", exc.trace)
            _write_text(trend_dir / "error.txt", str(exc))
            failed_plot_indices.append(plot_input.plot_index)
            validation_state.trends[str(plot_input.plot_index)] = {
                "status": "retry",
                "error": str(exc),
            }
            trend_results.append(
                FullCaseTrendResult(
                    plot_index=plot_input.plot_index,
                    trend_dir=trend_dir,
                    success=False,
                    error=str(exc),
                )
            )
            _record_trend_review_row(
                review_rows=review_rows,
                plot_input=plot_input,
                trend_dir=trend_dir,
                image_path=image_path,
                success=False,
                error=str(exc),
                validation_status="retry",
            )

    result = FullCaseSmokeResult(
        started_at_utc=started_at.isoformat(),
        finished_at_utc=datetime.now(UTC).isoformat(),
        output_root=output_root,
        case_id=case_id,
        case_dir=case_dir,
        report_json_path=output_root / "full_case_smoke_report.json",
        report_markdown_path=output_root / "full_case_smoke_report.md",
        review_csv_path=output_root / "review.csv",
        success=not failed_plot_indices,
        failed_plot_indices=failed_plot_indices,
        trend_results=trend_results,
    )
    _finalize_full_case_result(
        result=result,
        case_input=case_input,
        evidence_mode=evidence_mode,
        prompt_variant=prompt_variant,
        model=model,
        review_rows=review_rows,
        validation_state=validation_state,
    )
    return result


def _enabled_features_from_variant(prompt_variant: str) -> set[str]:
    if prompt_variant == "none":
        return set()
    return set(prompt_variant.replace("all_three", "role+insights+example").split("+"))


def _build_trend_table_csv(
    *,
    frame: pd.DataFrame,
    evidence_mode: EvidenceMode,
    plot_input: FullCasePlotInput,
    trend_dir: Path,
) -> str:
    if evidence_mode not in {"table", "plot+table"}:
        return ""
    table_csv = build_raw_table_csv(frame=frame, reporter_pattern=plot_input.reporter_pattern)
    _write_text(trend_dir / "trend_evidence_table.csv", table_csv)
    return table_csv


def _validate_full_case_inputs(case_input: FullCaseSmokeInput) -> None:
    if not case_input.plots:
        raise ValueError("full-case smoke input must include at least one plot")
    first_plot = case_input.plots[0]
    _validate_case_inputs(
        type(
            "_FullCaseValidationInput",
            (),
            {
                "csv_path": case_input.csv_path,
                "parameters_path": case_input.parameters_path,
                "documentation_path": case_input.documentation_path,
                "plot_path": first_plot.plot_path,
                "plot_description": first_plot.plot_description,
            },
        )()
    )
    for plot in case_input.plots[1:]:
        if not plot.plot_path.exists():
            raise FileNotFoundError(f"missing full-case plot: {plot.plot_path}")


def _finalize_full_case_result(
    *,
    result: FullCaseSmokeResult,
    case_input: FullCaseSmokeInput,
    evidence_mode: EvidenceMode,
    prompt_variant: str,
    model: str,
    review_rows: list[dict[str, str]],
    validation_state: FullCaseValidationState,
) -> None:
    summary_payload = {
        "case_id": result.case_id,
        "abm": case_input.abm,
        "evidence_mode": evidence_mode,
        "prompt_variant": prompt_variant,
        "model": model,
        "plot_count": len(case_input.plots),
        "success": result.success,
        "failed_plot_indices": result.failed_plot_indices,
    }
    _write_json(case_summary_path(result.case_dir), summary_payload)
    _write_json(validation_state_path(result.case_dir), validation_state.model_dump(mode="json"))
    _write_review_csv(result.review_csv_path, review_rows)
    _write_json(result.report_json_path, result.model_dump(mode="json"))
    _write_text(result.report_markdown_path, _render_report(result))


def _write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_report(result: FullCaseSmokeResult) -> str:
    lines = [
        "# Full Case Smoke Report",
        "",
        f"- success: `{str(result.success).lower()}`",
        f"- case_id: `{result.case_id}`",
        f"- failed_plot_count: `{len(result.failed_plot_indices)}`",
        f"- review_csv_path: `{result.review_csv_path}`",
        "",
        "| plot_index | success |",
        "| --- | --- |",
    ]
    for trend_result in result.trend_results:
        lines.append(f"| {trend_result.plot_index} | {str(trend_result.success).lower()} |")
    return "\n".join(lines) + "\n"


def _record_context_review_row(*, review_rows: list[dict[str, str]], context_dir: Path, validation_status: str) -> None:
    review_rows.append(
        {
            "plot_index": "context",
            "reporter_pattern": "",
            "plot_description": "",
            "trend_prompt_path": str(context_dir / "context_request.json"),
            "trend_output_path": str(context_dir / "context_output.txt"),
            "image_path": "",
            "table_csv_path": "",
            "success": "True",
            "error": "",
            "validation_status": validation_status,
        }
    )


def _record_trend_review_row(
    *,
    review_rows: list[dict[str, str]],
    plot_input: FullCasePlotInput,
    trend_dir: Path,
    image_path: Path | None,
    success: bool,
    error: str,
    validation_status: str,
) -> None:
    table_path = trend_dir / "trend_evidence_table.csv"
    trend_output_path = trend_dir / "trend_output.txt"
    review_rows.append(
        {
            "plot_index": str(plot_input.plot_index),
            "reporter_pattern": plot_input.reporter_pattern,
            "plot_description": plot_input.plot_description,
            "trend_prompt_path": str(trend_dir / "trend_prompt.txt"),
            "trend_output_path": str(trend_output_path if trend_output_path.exists() else ""),
            "image_path": str(image_path or ""),
            "table_csv_path": str(table_path if table_path.exists() else ""),
            "success": str(success),
            "error": error,
            "validation_status": validation_status,
        }
    )


def _load_validation_state(path: Path) -> FullCaseValidationState:
    if not path.exists():
        return FullCaseValidationState(context={"status": "retry", "error": None}, trends={})
    return FullCaseValidationState.model_validate_json(path.read_text(encoding="utf-8"))


def _backfill_validation_state_from_artifacts(
    *,
    validation_state: FullCaseValidationState,
    context_dir: Path,
    trends_dir: Path,
    plots: tuple[FullCasePlotInput, ...],
) -> None:
    if "status" not in validation_state.context and _context_artifacts_exist(context_dir):
        validation_state.context = {"status": "accepted", "error": None}
    for plot in plots:
        plot_key = str(plot.plot_index)
        if plot_key in validation_state.trends:
            continue
        trend_dir = trends_dir / f"plot_{plot.plot_index:02d}"
        if _trend_artifacts_exist(trend_dir):
            validation_state.trends[plot_key] = {"status": "accepted", "error": None}


def _is_context_accepted(*, context_dir: Path, validation_state: FullCaseValidationState) -> bool:
    if (
        validation_state.context.get("status") != "accepted"
        or not _context_artifacts_exist(context_dir)
        or (context_dir / "error.txt").exists()
    ):
        return False
    try:
        validate_structured_smoke_text_content((context_dir / "context_output.txt").read_text(encoding="utf-8"))
    except ValueError:
        return False
    return True


def _is_trend_accepted(*, plot_index: int, trend_dir: Path, validation_state: FullCaseValidationState) -> bool:
    status_payload = validation_state.trends.get(str(plot_index), {})
    if (
        status_payload.get("status") != "accepted"
        or not _trend_artifacts_exist(trend_dir)
        or (trend_dir / "error.txt").exists()
    ):
        return False
    try:
        validate_structured_smoke_text_content((trend_dir / "trend_output.txt").read_text(encoding="utf-8"))
    except ValueError:
        return False
    return True


def _context_artifacts_exist(context_dir: Path) -> bool:
    return (context_dir / "context_output.txt").exists() and (context_dir / "context_trace.json").exists()


def _trend_artifacts_exist(trend_dir: Path) -> bool:
    return (trend_dir / "trend_output.txt").exists() and (trend_dir / "trend_trace.json").exists()
