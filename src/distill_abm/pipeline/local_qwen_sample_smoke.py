"""Minimal sampled local-Qwen smoke run for end-to-end prompt/evidence inspection."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from distill_abm.configs.runtime_defaults import get_runtime_defaults
from distill_abm.llm.adapters.base import LLMAdapter, LLMProviderError
from distill_abm.pipeline.doe_smoke_prompts import (
    build_legacy_doe_context_prompt,
    build_legacy_doe_trend_prompt,
    build_raw_table_csv,
)
from distill_abm.pipeline.helpers import encode_image, invoke_adapter_with_trace
from distill_abm.run_viewer import render_run_viewer
from distill_abm.structured_logging import attach_json_log_file
from distill_abm.utils import detect_placeholder_signals

EvidenceMode = Literal["plot", "table", "plot+table"]
SMOKE_MAX_TOKENS = 32768
SMOKE_OLLAMA_NUM_CTX = 131072
GENERIC_UNAVAILABLE_PATTERNS = (
    "analysis is currently unavailable",
    "requested information cannot be retrieved",
    "please try again later",
    "consult additional resources",
)


class StructuredSmokeResponseError(ValueError):
    """Structured smoke response failed validation but still produced a trace worth keeping."""

    def __init__(self, message: str, *, trace: dict[str, object], prompt: str) -> None:
        super().__init__(message)
        self.trace = trace
        self.prompt = prompt


class StructuredSmokeText(BaseModel):
    """Structured smoke output schema for one text response."""

    response_text: str


class LocalQwenCaseInput(BaseModel):
    """Resolved per-ABM inputs for sampled local Qwen smoke cases."""

    abm: str
    csv_path: Path
    parameters_path: Path
    documentation_path: Path
    reporter_pattern: str
    plot_description: str
    plot_path: Path


class LocalQwenSampleCase(BaseModel):
    """One sampled case in the local-Qwen smoke matrix."""

    case_id: str
    abm: str
    evidence_mode: EvidenceMode
    prompt_variant: str


class LocalQwenSampleCaseResult(BaseModel):
    """One executed sampled smoke case."""

    case_id: str
    abm: str
    evidence_mode: EvidenceMode
    prompt_variant: str
    case_dir: Path
    success: bool
    error: str | None = None
    resumed_from_existing: bool = False


class LocalQwenSampleSmokeResult(BaseModel):
    """Top-level result for the sampled local-Qwen smoke."""

    started_at_utc: str
    finished_at_utc: str
    output_root: Path
    report_json_path: Path
    report_markdown_path: Path
    review_csv_path: Path
    run_log_path: Path
    viewer_html_path: Path
    success: bool
    failed_case_ids: list[str] = Field(default_factory=list)
    cases: list[LocalQwenSampleCaseResult] = Field(default_factory=list)
    run_id: str


def default_local_qwen_sample_cases() -> tuple[LocalQwenSampleCase, ...]:
    """Return a small stratified sample across ABMs, evidence modes, and prompt variants."""
    return (
        LocalQwenSampleCase(case_id="01_fauna_none_plot", abm="fauna", evidence_mode="plot", prompt_variant="none"),
        LocalQwenSampleCase(
            case_id="02_grazing_role_table",
            abm="grazing",
            evidence_mode="table",
            prompt_variant="role",
        ),
        LocalQwenSampleCase(
            case_id="03_milk_insights_plot_plus_table",
            abm="milk_consumption",
            evidence_mode="plot+table",
            prompt_variant="insights",
        ),
        LocalQwenSampleCase(
            case_id="04_fauna_example_table",
            abm="fauna",
            evidence_mode="table",
            prompt_variant="example",
        ),
        LocalQwenSampleCase(
            case_id="05_grazing_role_plus_example_plot_plus_table",
            abm="grazing",
            evidence_mode="plot+table",
            prompt_variant="role+example",
        ),
        LocalQwenSampleCase(
            case_id="06_milk_role_plus_insights_plot",
            abm="milk_consumption",
            evidence_mode="plot",
            prompt_variant="role+insights",
        ),
        LocalQwenSampleCase(
            case_id="07_fauna_insights_plus_example_plot",
            abm="fauna",
            evidence_mode="plot",
            prompt_variant="insights+example",
        ),
        LocalQwenSampleCase(
            case_id="08_grazing_all_three_table",
            abm="grazing",
            evidence_mode="table",
            prompt_variant="all_three",
        ),
    )


def run_local_qwen_sample_smoke(
    *,
    case_inputs: dict[str, LocalQwenCaseInput],
    adapter: LLMAdapter,
    model: str,
    output_root: Path,
    cases: tuple[LocalQwenSampleCase, ...] | None = None,
    max_tokens: int = SMOKE_MAX_TOKENS,
    ollama_num_ctx: int = SMOKE_OLLAMA_NUM_CTX,
    ollama_num_ctx_by_mode: dict[EvidenceMode, int] | None = None,
    resume_existing: bool = False,
    stop_on_failure: bool = False,
    max_retries: int | None = None,
    retry_backoff_seconds: float | None = None,
) -> LocalQwenSampleSmokeResult:
    """Run a minimal real local-Qwen smoke across sampled DOE-style cases."""
    started_at = datetime.now(UTC)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = started_at.strftime("run_%Y%m%d_%H%M%S_%f")
    run_root = output_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    _write_text(output_root / "latest_run.txt", str(run_root))
    previous_run_root = _resolve_previous_run_root(output_root=output_root, current_run_id=run_id)
    run_log_path = attach_json_log_file(run_root / "run.log.jsonl")
    selected_cases = cases or default_local_qwen_sample_cases()
    review_rows: list[dict[str, str]] = []
    results: list[LocalQwenSampleCaseResult] = []
    failed_case_ids: list[str] = []

    for case in selected_cases:
        case_dir = run_root / "cases" / case.case_id
        inputs_dir = case_dir / "01_inputs"
        requests_dir = case_dir / "02_requests"
        outputs_dir = case_dir / "03_outputs"
        for directory in (inputs_dir, requests_dir, outputs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        input_bundle = case_inputs[case.abm]
        try:
            if resume_existing:
                resumed = _copy_resumable_case_result(
                    case=case,
                    destination_case_dir=case_dir,
                    previous_run_root=previous_run_root,
                )
                if resumed is not None:
                    review_rows.append(_build_review_row_from_existing(case_dir=case_dir))
                    results.append(resumed)
                    continue
            _validate_case_inputs(input_bundle)
            enabled = set(case.prompt_variant.replace("all_three", "role+insights+example").split("+"))
            if case.prompt_variant == "none":
                enabled = set()
            context_prompt_base = build_legacy_doe_context_prompt(
                abm=input_bundle.abm,
                inputs_csv_path=input_bundle.parameters_path,
                inputs_doc_path=input_bundle.documentation_path,
                enabled=enabled,
            )
            context_prompt = context_prompt_base
            shutil.copy2(input_bundle.parameters_path, inputs_dir / "parameters.txt")
            shutil.copy2(input_bundle.documentation_path, inputs_dir / "documentation.txt")
            _write_text(inputs_dir / "context_prompt.txt", context_prompt)
            _write_json(
                requests_dir / "context_request.json",
                _build_request_preview(
                    model=model,
                    prompt_with_schema=context_prompt,
                    image_attached=False,
                    max_tokens=max_tokens,
                    ollama_num_ctx=_resolve_case_num_ctx(case.evidence_mode, ollama_num_ctx, ollama_num_ctx_by_mode),
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                ),
            )

            try:
                context_text, context_trace = _invoke_structured_smoke_text(
                    adapter=adapter,
                    model=model,
                    prompt_with_schema=context_prompt,
                    max_tokens=max_tokens,
                    ollama_num_ctx=_resolve_case_num_ctx(case.evidence_mode, ollama_num_ctx, ollama_num_ctx_by_mode),
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
            except StructuredSmokeResponseError as exc:
                _write_json(requests_dir / "context_request.json", exc.trace["request"])
                _write_json(outputs_dir / "context_trace.json", exc.trace)
                _write_optional_thinking(outputs_dir / "context_thinking.txt", exc.trace)
                raise
            _write_json(requests_dir / "context_request.json", context_trace["request"])
            _write_text(outputs_dir / "context_output.txt", context_text)
            _write_json(outputs_dir / "context_trace.json", context_trace)
            _write_optional_thinking(outputs_dir / "context_thinking.txt", context_trace)

            image_b64: str | None = None
            image_path: Path | None = None
            if case.evidence_mode in {"plot", "plot+table"}:
                image_path = inputs_dir / "trend_evidence_plot.png"
                shutil.copy2(input_bundle.plot_path, image_path)
                image_b64 = encode_image(image_path)
            trend_text, trend_trace = _run_trend_with_fitting_table(
                inputs_dir=inputs_dir,
                requests_dir=requests_dir,
                outputs_dir=outputs_dir,
                adapter=adapter,
                model=model,
                case=case,
                input_bundle=input_bundle,
                context_text=context_text,
                enabled=enabled,
                image_b64=image_b64,
                max_tokens=max_tokens,
                ollama_num_ctx=_resolve_case_num_ctx(case.evidence_mode, ollama_num_ctx, ollama_num_ctx_by_mode),
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            _write_json(requests_dir / "trend_request.json", trend_trace["request"])
            _write_text(outputs_dir / "trend_output.txt", trend_text)
            _write_json(outputs_dir / "trend_trace.json", trend_trace)
            _write_optional_thinking(outputs_dir / "trend_thinking.txt", trend_trace)
            _write_json(
                requests_dir / "hyperparameters.json",
                {
                    "context": context_trace["request"],
                    "trend": trend_trace["request"],
                },
            )
            _write_json(
                case_dir / "00_case_summary.json",
                {
                    "case_id": case.case_id,
                    "abm": case.abm,
                    "evidence_mode": case.evidence_mode,
                    "prompt_variant": case.prompt_variant,
                    "model": model,
                    "inputs_dir": str(inputs_dir),
                    "requests_dir": str(requests_dir),
                    "outputs_dir": str(outputs_dir),
                },
            )

            review_rows.append(
                {
                    "case_id": case.case_id,
                    "abm": case.abm,
                    "evidence_mode": case.evidence_mode,
                    "prompt_variant": case.prompt_variant,
                    "model": model,
                    "case_summary_path": str(case_dir / "00_case_summary.json"),
                    "context_prompt_path": str(inputs_dir / "context_prompt.txt"),
                    "context_prompt_text": context_prompt,
                    "trend_prompt_path": str(inputs_dir / "trend_prompt.txt"),
                    "trend_prompt_text": (inputs_dir / "trend_prompt.txt").read_text(encoding="utf-8"),
                    "image_path": str(image_path or ""),
                    "table_csv_path": str(
                        (inputs_dir / "trend_evidence_table.csv")
                        if (inputs_dir / "trend_evidence_table.csv").exists()
                        else ""
                    ),
                    "parameters_path": str(inputs_dir / "parameters.txt"),
                    "documentation_path": str(inputs_dir / "documentation.txt"),
                    "hyperparameters_path": str(requests_dir / "hyperparameters.json"),
                    "context_output_path": str(outputs_dir / "context_output.txt"),
                    "context_output_text": context_text,
                    "trend_output_path": str(outputs_dir / "trend_output.txt"),
                    "trend_output_text": trend_text,
                }
            )
            results.append(
                LocalQwenSampleCaseResult(
                    case_id=case.case_id,
                    abm=case.abm,
                    evidence_mode=case.evidence_mode,
                    prompt_variant=case.prompt_variant,
                    case_dir=case_dir,
                    success=True,
                )
            )
        except Exception as exc:
            failed_case_ids.append(case.case_id)
            _write_text(outputs_dir / "error.txt", str(exc))
            results.append(
                LocalQwenSampleCaseResult(
                    case_id=case.case_id,
                    abm=case.abm,
                    evidence_mode=case.evidence_mode,
                    prompt_variant=case.prompt_variant,
                    case_dir=case_dir,
                    success=False,
                    error=str(exc),
                )
            )
            if stop_on_failure:
                break

    review_csv_path = run_root / "request_review.csv"
    _write_review_csv(review_csv_path, review_rows)
    finished_at = datetime.now(UTC)
    report_json_path = run_root / "smoke_local_qwen_report.json"
    report_markdown_path = run_root / "smoke_local_qwen_report.md"
    result = LocalQwenSampleSmokeResult(
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        output_root=run_root,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
        review_csv_path=review_csv_path,
        run_log_path=run_log_path,
        viewer_html_path=run_root / "review.html",
        run_id=run_id,
        success=not failed_case_ids,
        failed_case_ids=failed_case_ids,
        cases=results,
    )
    _write_json(report_json_path, result.model_dump(mode="json"))
    _write_text(report_markdown_path, _render_report(result))
    viewer_html_path = render_run_viewer(run_root)
    result.viewer_html_path = viewer_html_path
    _write_json(report_json_path, result.model_dump(mode="json"))
    _write_text(output_root / "latest_report_path.txt", str(report_json_path))
    return result


def _build_case_table(
    *,
    inputs_dir: Path,
    case: LocalQwenSampleCase,
    input_bundle: LocalQwenCaseInput,
) -> str:
    return _build_case_table_for_stride(inputs_dir=inputs_dir, case=case, input_bundle=input_bundle, stride=1)


def _build_case_table_for_stride(
    *,
    inputs_dir: Path,
    case: LocalQwenSampleCase,
    input_bundle: LocalQwenCaseInput,
    stride: int,
) -> str:
    if case.evidence_mode not in {"table", "plot+table"}:
        return ""
    frame = pd.read_csv(input_bundle.csv_path, sep=";")
    if stride > 1:
        step_column = next((str(column) for column in frame.columns if str(column).strip() == "[step]"), None)
        if step_column is not None:
            frame = frame[frame[step_column] % stride == 0]
        else:
            frame = frame.iloc[::stride]
    table_csv = build_raw_table_csv(frame=frame, reporter_pattern=input_bundle.reporter_pattern)
    table_path = inputs_dir / "trend_evidence_table.csv"
    _write_text(table_path, table_csv)
    return table_csv


def _copy_resumable_case_result(
    *,
    case: LocalQwenSampleCase,
    destination_case_dir: Path,
    previous_run_root: Path | None,
) -> LocalQwenSampleCaseResult | None:
    if previous_run_root is None:
        return None
    case_dir = previous_run_root / "cases" / case.case_id
    required_paths = (
        case_dir / "00_case_summary.json",
        case_dir / "01_inputs" / "context_prompt.txt",
        case_dir / "01_inputs" / "trend_prompt.txt",
        case_dir / "02_requests" / "context_request.json",
        case_dir / "02_requests" / "trend_request.json",
        case_dir / "02_requests" / "hyperparameters.json",
        case_dir / "03_outputs" / "context_output.txt",
        case_dir / "03_outputs" / "context_trace.json",
        case_dir / "03_outputs" / "trend_output.txt",
        case_dir / "03_outputs" / "trend_trace.json",
    )
    if any(not path.exists() for path in required_paths):
        return None
    if (case_dir / "03_outputs" / "error.txt").exists():
        return None
    if destination_case_dir.exists():
        shutil.rmtree(destination_case_dir)
    shutil.copytree(case_dir, destination_case_dir)
    return LocalQwenSampleCaseResult(
        case_id=case.case_id,
        abm=case.abm,
        evidence_mode=case.evidence_mode,
        prompt_variant=case.prompt_variant,
        case_dir=destination_case_dir,
        success=True,
        resumed_from_existing=True,
    )


def _resolve_previous_run_root(*, output_root: Path, current_run_id: str) -> Path | None:
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


def _build_review_row_from_existing(case_dir: Path) -> dict[str, str]:
    inputs_dir = case_dir / "01_inputs"
    outputs_dir = case_dir / "03_outputs"
    requests_dir = case_dir / "02_requests"
    summary_path = case_dir / "00_case_summary.json"
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


def _invoke_structured_smoke_text(
    *,
    adapter: LLMAdapter,
    model: str,
    prompt_with_schema: str,
    image_b64: str | None = None,
    max_tokens: int,
    ollama_num_ctx: int,
    max_retries: int | None = None,
    retry_backoff_seconds: float | None = None,
    request_metadata: dict[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    raw_text, trace = invoke_adapter_with_trace(
        adapter=adapter,
        model=model,
        prompt=prompt_with_schema,
        image_b64=image_b64,
        max_tokens=max_tokens,
        request_metadata={
            "structured_output_name": "structured_smoke_text",
            "structured_output_schema": StructuredSmokeText.model_json_schema(),
            "ollama_num_ctx": ollama_num_ctx,
            "ollama_format": StructuredSmokeText.model_json_schema(),
            "preserve_raw_text": True,
            **(request_metadata or {}),
        },
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    response_block = trace.get("response")
    raw_response = response_block if isinstance(response_block, dict) else {}
    raw_payload = raw_response.get("raw")
    raw_payload_dict = raw_payload if isinstance(raw_payload, dict) else {}
    try:
        parsed = StructuredSmokeText.model_validate_json(raw_text)
    except Exception as exc:
        thinking_text = _extract_thinking_text(trace)
        done_reason = raw_payload_dict.get("done_reason")
        eval_count = raw_payload_dict.get("eval_count")
        if thinking_text:
            if done_reason == "length":
                raise StructuredSmokeResponseError(
                    (
                        "model exhausted max_tokens on thinking before emitting a final structured response"
                        + (f" (eval_count={eval_count})" if isinstance(eval_count, int) else "")
                    ),
                    trace=trace,
                    prompt=prompt_with_schema,
                ) from exc
            raise StructuredSmokeResponseError(
                "model returned only thinking without a final structured response",
                trace=trace,
                prompt=prompt_with_schema,
            ) from exc
        raise StructuredSmokeResponseError(
            "model did not return valid structured JSON for the smoke output",
            trace=trace,
            prompt=prompt_with_schema,
        ) from exc
    done_reason = raw_payload_dict.get("done_reason")
    if done_reason == "length":
        raise StructuredSmokeResponseError(
            "model reached max_tokens before completing the structured response",
            trace=trace,
            prompt=prompt_with_schema,
        )
    final_text = parsed.response_text.strip()
    if not final_text:
        raise StructuredSmokeResponseError(
            "structured smoke output contained an empty response_text",
            trace=trace,
            prompt=prompt_with_schema,
        )
    if _is_generic_unavailable_response(final_text):
        raise StructuredSmokeResponseError(
            "generic unavailable response detected in structured smoke output",
            trace=trace,
            prompt=prompt_with_schema,
        )
    if isinstance(response_block, dict):
        response_block["parsed_response"] = parsed.model_dump(mode="json")
    return final_text, trace


def _build_request_preview(
    *,
    model: str,
    prompt_with_schema: str,
    image_attached: bool,
    max_tokens: int,
    ollama_num_ctx: int,
    max_retries: int | None,
    retry_backoff_seconds: float | None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    prompt_signature = hashlib.sha256(prompt_with_schema.encode("utf-8")).hexdigest()
    defaults = get_runtime_defaults().llm_request
    return {
        "provider": "ollama",
        "model": model,
        "temperature": 1.0,
        "max_tokens": max_tokens,
        "max_retries": max(defaults.max_retries if max_retries is None else max_retries, 0),
        "retry_backoff_seconds": max(
            defaults.retry_backoff_seconds if retry_backoff_seconds is None else retry_backoff_seconds,
            0.0,
        ),
        "image_attached": image_attached,
        "prompt_text": prompt_with_schema,
        "prompt_length": len(prompt_with_schema),
        "prompt_signature": prompt_signature,
        "message_count": 1,
        "messages": [{"role": "user", "content": prompt_with_schema}],
        "metadata": {
            "ollama_num_ctx": ollama_num_ctx,
            "ollama_format": StructuredSmokeText.model_json_schema(),
            "preserve_raw_text": True,
            **(metadata or {}),
        },
    }


def _resolve_case_num_ctx(
    evidence_mode: EvidenceMode,
    default_num_ctx: int,
    num_ctx_by_mode: dict[EvidenceMode, int] | None,
) -> int:
    if not num_ctx_by_mode:
        return default_num_ctx
    resolved = num_ctx_by_mode.get(evidence_mode, default_num_ctx)
    return resolved if resolved > 0 else default_num_ctx


def _validate_case_inputs(input_bundle: LocalQwenCaseInput) -> None:
    required_paths = (
        input_bundle.csv_path,
        input_bundle.parameters_path,
        input_bundle.documentation_path,
        input_bundle.plot_path,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"missing required smoke input: {path}")
        if path.is_file() and path.stat().st_size <= 0:
            raise ValueError(f"empty required smoke input: {path}")
    if detect_placeholder_signals(input_bundle.documentation_path.read_text(encoding="utf-8", errors="replace")):
        raise ValueError(f"placeholder-like text detected in documentation: {input_bundle.documentation_path}")
    if detect_placeholder_signals(input_bundle.parameters_path.read_text(encoding="utf-8", errors="replace")):
        raise ValueError(f"placeholder-like text detected in parameters: {input_bundle.parameters_path}")
    if detect_placeholder_signals(input_bundle.plot_description):
        raise ValueError("placeholder-like text detected in plot description")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_optional_thinking(path: Path, trace: dict[str, object]) -> None:
    thinking_text = _extract_thinking_text(trace)
    if thinking_text:
        _write_text(path, thinking_text)


def _extract_thinking_text(trace: dict[str, object]) -> str:
    response_block = trace.get("response")
    if not isinstance(response_block, dict):
        return ""
    raw_block = response_block.get("raw")
    if not isinstance(raw_block, dict):
        return ""
    message_block = raw_block.get("message")
    if not isinstance(message_block, dict):
        return ""
    thinking = message_block.get("thinking")
    return str(thinking).strip() if isinstance(thinking, str) else ""


def _write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "case_id",
        "abm",
        "evidence_mode",
        "prompt_variant",
        "model",
        "case_summary_path",
        "context_prompt_path",
        "context_prompt_text",
        "trend_prompt_path",
        "trend_prompt_text",
        "image_path",
        "table_csv_path",
        "parameters_path",
        "documentation_path",
        "hyperparameters_path",
        "context_output_path",
        "context_output_text",
        "trend_output_path",
        "trend_output_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_report(result: LocalQwenSampleSmokeResult) -> str:
    lines = [
        "# Local Qwen Sample Smoke Report",
        "",
        f"- success: `{str(result.success).lower()}`",
        f"- case_count: `{len(result.cases)}`",
        f"- failed_case_count: `{len(result.failed_case_ids)}`",
        f"- run_log_path: `{result.run_log_path}`",
        f"- viewer_html_path: `{result.viewer_html_path}`",
        f"- review_csv_path: `{result.review_csv_path}`",
        "",
        "| case_id | abm | evidence_mode | prompt_variant | success |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in result.cases:
        lines.append(
            f"| {case.case_id} | {case.abm} | {case.evidence_mode} | {case.prompt_variant} | "
            f"{str(case.success).lower()} |"
        )
    return "\n".join(lines) + "\n"


def _run_trend_with_fitting_table(
    *,
    inputs_dir: Path,
    requests_dir: Path,
    outputs_dir: Path,
    adapter: LLMAdapter,
    model: str,
    case: LocalQwenSampleCase,
    input_bundle: LocalQwenCaseInput,
    context_text: str,
    enabled: set[str],
    image_b64: str | None,
    max_tokens: int,
    ollama_num_ctx: int,
    max_retries: int | None,
    retry_backoff_seconds: float | None,
) -> tuple[str, dict[str, object]]:
    max_stride = 64
    last_exc: StructuredSmokeResponseError | None = None
    for stride in range(1, max_stride + 1):
        table_csv = _build_case_table_for_stride(
            inputs_dir=inputs_dir,
            case=case,
            input_bundle=input_bundle,
            stride=stride,
        )
        trend_prompt = build_legacy_doe_trend_prompt(
            abm=input_bundle.abm,
            context_response=context_text,
            plot_description=input_bundle.plot_description,
            evidence_mode=case.evidence_mode,
            table_csv=table_csv,
            enabled=enabled,
        )
        _write_text(inputs_dir / "trend_prompt.txt", trend_prompt)
        _write_json(
            requests_dir / "trend_request.json",
            _build_request_preview(
                model=model,
                prompt_with_schema=trend_prompt,
                image_attached=image_b64 is not None,
                max_tokens=max_tokens,
                ollama_num_ctx=ollama_num_ctx,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                metadata={"table_downsample_stride": stride},
            ),
        )
        try:
            return _invoke_structured_smoke_text(
                adapter=adapter,
                model=model,
                prompt_with_schema=trend_prompt,
                image_b64=image_b64,
                max_tokens=max_tokens,
                ollama_num_ctx=ollama_num_ctx,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                request_metadata={"table_downsample_stride": stride},
            )
        except StructuredSmokeResponseError as exc:
            _write_json(requests_dir / "trend_request.json", exc.trace["request"])
            _write_json(outputs_dir / "trend_trace.json", exc.trace)
            _write_optional_thinking(outputs_dir / "trend_thinking.txt", exc.trace)
            if case.evidence_mode not in {"table", "plot+table"} or not _looks_like_context_overflow(str(exc)):
                raise
            last_exc = exc
            continue
        except LLMProviderError as exc:
            if case.evidence_mode not in {"table", "plot+table"} or not _looks_like_context_overflow(str(exc)):
                raise
            last_exc = StructuredSmokeResponseError(
                str(exc),
                trace={
                    "request": _build_request_preview(
                        model=model,
                        prompt_with_schema=trend_prompt,
                        image_attached=image_b64 is not None,
                        max_tokens=max_tokens,
                        ollama_num_ctx=ollama_num_ctx,
                        max_retries=max_retries,
                        retry_backoff_seconds=retry_backoff_seconds,
                        metadata={"table_downsample_stride": stride},
                    ),
                    "response": {"raw": None},
                    "attempts_made": 0,
                    "errors": [str(exc)],
                },
                prompt=trend_prompt,
            )
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("table fitting loop terminated without producing a result")


def _looks_like_context_overflow(error: str) -> bool:
    lowered = error.lower()
    return (
        "maximum context length" in lowered
        or "too many input tokens" in lowered
        or "context length" in lowered
        or "input tokens" in lowered
    )


def _is_generic_unavailable_response(text: str) -> bool:
    lowered = text.lower()
    return all(pattern in lowered for pattern in GENERIC_UNAVAILABLE_PATTERNS[:3]) or any(
        pattern in lowered for pattern in GENERIC_UNAVAILABLE_PATTERNS
    )
