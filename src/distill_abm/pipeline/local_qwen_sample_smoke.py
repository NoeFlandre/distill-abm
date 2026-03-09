"""Minimal sampled local-Qwen smoke run for end-to-end prompt/evidence inspection."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from pydantic import BaseModel, Field

from distill_abm.configs.runtime_defaults import get_runtime_defaults
from distill_abm.llm.adapters.base import LLMAdapter, LLMProviderError
from distill_abm.pipeline.doe_smoke_prompts import (
    build_legacy_doe_context_prompt,
    build_legacy_doe_trend_prompt,
)
from distill_abm.pipeline.helpers import encode_image, invoke_adapter_with_trace
from distill_abm.pipeline.local_qwen_sample_artifacts import (
    build_review_row_from_existing,
    copy_resumable_case_result,
    resolve_case_num_ctx,
    resolve_previous_run_root,
)
from distill_abm.pipeline.local_qwen_sample_response import (
    StructuredSmokeResponseError,
    StructuredSmokeText,
    extract_thinking_text,
    looks_like_context_overflow,
    parse_structured_smoke_text,
)
from distill_abm.pipeline.run_artifact_contracts import (
    case_summary_path,
    latest_report_pointer_path,
    latest_run_pointer_path,
    sampled_smoke_report_path,
)
from distill_abm.pipeline.run_artifact_contracts import (
    run_log_path as run_log_contract_path,
)
from distill_abm.pipeline.run_artifact_contracts import (
    viewer_html_path as viewer_html_contract_path,
)
from distill_abm.pipeline.statistical_evidence import build_statistical_evidence, render_evidence_artifacts
from distill_abm.run_viewer import render_run_viewer
from distill_abm.structured_logging import attach_json_log_file
from distill_abm.utils import detect_placeholder_signals

EvidenceMode = Literal["plot", "table", "plot+table"]
SMOKE_MAX_TOKENS = 32768
SMOKE_OLLAMA_NUM_CTX = 131072


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


class _CachedContextResult(BaseModel):
    prompt: str
    text: str
    trace: dict[str, object]


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
    _write_text(latest_run_pointer_path(output_root), str(run_root))
    previous_run_root = resolve_previous_run_root(output_root=output_root, current_run_id=run_id)
    run_log_path = attach_json_log_file(run_log_contract_path(run_root))
    selected_cases = cases or default_local_qwen_sample_cases()
    review_rows: list[dict[str, str]] = []
    results: list[LocalQwenSampleCaseResult] = []
    failed_case_ids: list[str] = []
    context_cache: dict[str, _CachedContextResult] = {}

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
                resumed = copy_resumable_case_result(
                    case_id=case.case_id,
                    destination_case_dir=case_dir,
                    previous_run_root=previous_run_root,
                )
                if resumed:
                    review_rows.append(build_review_row_from_existing(case_dir))
                    results.append(
                        LocalQwenSampleCaseResult(
                            case_id=case.case_id,
                            abm=case.abm,
                            evidence_mode=case.evidence_mode,
                            prompt_variant=case.prompt_variant,
                            case_dir=case_dir,
                            success=True,
                            resumed_from_existing=True,
                        )
                    )
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
            context_cache_key = _context_cache_key(prompt=context_prompt, model=model)
            cached_context = context_cache.get(context_cache_key)
            if cached_context is None:
                case_num_ctx = resolve_case_num_ctx(
                    case.evidence_mode,
                    ollama_num_ctx,
                    cast(dict[str, int] | None, ollama_num_ctx_by_mode),
                )
                _write_json(
                    requests_dir / "context_request.json",
                    _build_request_preview(
                        model=model,
                        prompt_with_schema=context_prompt,
                        image_attached=False,
                        max_tokens=max_tokens,
                        ollama_num_ctx=case_num_ctx,
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
                        ollama_num_ctx=case_num_ctx,
                        max_retries=max_retries,
                        retry_backoff_seconds=retry_backoff_seconds,
                    )
                except StructuredSmokeResponseError as exc:
                    _write_json(requests_dir / "context_request.json", exc.trace["request"])
                    _write_json(outputs_dir / "context_trace.json", exc.trace)
                    _write_optional_thinking(outputs_dir / "context_thinking.txt", exc.trace)
                    raise
                context_cache[context_cache_key] = _CachedContextResult(
                    prompt=context_prompt,
                    text=context_text,
                    trace=context_trace,
                )
            else:
                context_text = cached_context.text
                context_trace = cached_context.trace

            _materialize_context_artifacts(
                requests_dir=requests_dir,
                outputs_dir=outputs_dir,
                context_text=context_text,
                context_trace=context_trace,
            )

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
                ollama_num_ctx=resolve_case_num_ctx(
                    case.evidence_mode,
                    ollama_num_ctx,
                    cast(dict[str, int] | None, ollama_num_ctx_by_mode),
                ),
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
                case_summary_path(case_dir),
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
                    "case_summary_path": str(case_summary_path(case_dir)),
                    "context_prompt_path": str(inputs_dir / "context_prompt.txt"),
                    "context_prompt_text": context_prompt,
                    "trend_prompt_path": str(inputs_dir / "trend_prompt.txt"),
                    "trend_prompt_text": (inputs_dir / "trend_prompt.txt").read_text(encoding="utf-8"),
                    "image_path": str(image_path or ""),
                    "table_csv_path": str(
                        (inputs_dir / "trend_evidence_table.txt")
                        if (inputs_dir / "trend_evidence_table.txt").exists()
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
    report_json_path = sampled_smoke_report_path(run_root)
    report_markdown_path = run_root / "smoke_local_qwen_report.md"
    result = LocalQwenSampleSmokeResult(
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        output_root=run_root,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
        review_csv_path=review_csv_path,
        run_log_path=run_log_path,
        viewer_html_path=viewer_html_contract_path(run_root),
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
    _write_text(latest_report_pointer_path(output_root), str(report_json_path))
    return result


def _build_case_table(
    *,
    inputs_dir: Path,
    case: LocalQwenSampleCase,
    input_bundle: LocalQwenCaseInput,
) -> str:
    return _build_case_table_for_stride(inputs_dir=inputs_dir, case=case, input_bundle=input_bundle, stride=1)


def _context_cache_key(*, prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()


def _materialize_context_artifacts(
    *,
    requests_dir: Path,
    outputs_dir: Path,
    context_text: str,
    context_trace: dict[str, object],
) -> None:
    _write_json(requests_dir / "context_request.json", context_trace["request"])
    _write_text(outputs_dir / "context_output.txt", context_text)
    _write_json(outputs_dir / "context_trace.json", context_trace)
    _write_optional_thinking(outputs_dir / "context_thinking.txt", context_trace)


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
    evidence = build_statistical_evidence(
        frame=frame,
        reporter_pattern=input_bundle.reporter_pattern,
        compression_tier=max(stride - 1, 0),
    )
    summary_path, _payload_path, _series_path = render_evidence_artifacts(
        evidence=evidence,
        output_dir=inputs_dir,
        stem="trend_evidence_table",
    )
    return summary_path.read_text(encoding="utf-8")



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
    final_text = parse_structured_smoke_text(raw_text=raw_text, trace=trace, prompt=prompt_with_schema)
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
    thinking_text = extract_thinking_text(trace)
    if thinking_text:
        _write_text(path, thinking_text)


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
            if case.evidence_mode not in {"table", "plot+table"} or not looks_like_context_overflow(str(exc)):
                raise
            last_exc = exc
            continue
        except LLMProviderError as exc:
            if case.evidence_mode not in {"table", "plot+table"} or not looks_like_context_overflow(str(exc)):
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
