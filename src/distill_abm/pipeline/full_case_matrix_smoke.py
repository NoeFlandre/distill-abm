"""Run-separated full-case smoke matrix with resume, logging, and review artifacts."""

from __future__ import annotations

import json
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from distill_abm.llm.adapters.base import LLMAdapter, LLMProviderError
from distill_abm.llm.resilience import CIRCUIT_BREAKER_OPEN_SECONDS, is_transient_provider_error
from distill_abm.pipeline.doe_smoke_prompts import build_legacy_doe_context_prompt, build_legacy_doe_trend_prompt
from distill_abm.pipeline.full_case_matrix_run_review import build_run_review_rows, write_run_review_csv
from distill_abm.pipeline.full_case_review_csv import write_case_review_csv
from distill_abm.pipeline.full_case_smoke import (
    EvidenceMode,
    FullCasePlotInput,
    FullCaseSmokeInput,
    FullCaseValidationState,
    _backfill_validation_state_from_artifacts,
    _build_trend_table_csv,
    _enabled_features_from_variant,
    _is_context_accepted,
    _is_trend_accepted,
    _load_validation_state,
    _record_context_review_row,
    _record_trend_review_row,
    _validate_full_case_inputs,
    resolve_parallel_trend_workers,
)
from distill_abm.pipeline.helpers import encode_image
from distill_abm.pipeline.local_qwen_sample_response import (
    StructuredSmokeResponseError,
    looks_like_context_overflow,
)
from distill_abm.pipeline.local_qwen_sample_smoke import (
    _context_cache_key,
    _invoke_structured_smoke_text,
    _materialize_context_artifacts,
    _write_json,
    _write_optional_thinking,
    _write_text,
)
from distill_abm.pipeline.prompt_compression_artifacts import (
    PromptCompressionAttempt,
    write_prompt_compression_artifacts,
)
from distill_abm.pipeline.run_artifact_contracts import (
    FULL_CASE_MATRIX_REPORT_FILENAME,
    case_summary_path,
    latest_run_pointer_path,
    validation_state_path,
    viewer_html_path,
)
from distill_abm.pipeline.run_artifact_contracts import (
    run_log_path as run_log_contract_path,
)
from distill_abm.run_viewer import render_run_viewer
from distill_abm.structured_logging import attach_json_log_file


class FullCaseMatrixCaseSpec(BaseModel):
    """One ABM-wide full-case combination to run."""

    case_id: str
    abm: str
    evidence_mode: EvidenceMode
    prompt_variant: str
    repetition: int


class FullCaseMatrixCaseResult(BaseModel):
    """One executed full-case matrix case."""

    case_id: str
    abm: str
    evidence_mode: EvidenceMode
    prompt_variant: str
    repetition: int
    case_dir: Path
    success: bool
    resumed_from_existing: bool = False
    error: str | None = None


class FullCaseMatrixSmokeResult(BaseModel):
    """Top-level result for the ABM-wide full-case smoke matrix."""

    started_at_utc: str
    finished_at_utc: str
    output_root: Path
    run_id: str
    report_json_path: Path
    report_markdown_path: Path
    review_csv_path: Path
    run_log_path: Path
    viewer_html_path: Path
    success: bool
    failed_case_ids: list[str] = Field(default_factory=list)
    cases: list[FullCaseMatrixCaseResult] = Field(default_factory=list)


class _CachedContext(BaseModel):
    text: str
    trace: dict[str, object]


DEFAULT_MAX_CASE_ATTEMPTS = 3
DEFAULT_MATRIX_PASS_WAIT_SECONDS = CIRCUIT_BREAKER_OPEN_SECONDS
MAX_PARALLEL_TRENDS = 6
MAX_PARALLEL_CASES = 3


def resolve_parallel_case_workers(provider: str) -> int:
    """Return a provider-aware matrix case worker count."""

    if provider.strip().lower() == "mistral":
        return 1
    return MAX_PARALLEL_CASES


class _MatrixTrendExecutionResult(BaseModel):
    plot_input: FullCasePlotInput
    trend_dir: Path
    image_path: Path | None
    success: bool
    trend_text: str | None = None
    trend_trace: dict[str, object] | None = None
    error: str | None = None


def build_full_case_matrix_case_specs(
    *,
    abm: str,
    evidence_modes: tuple[EvidenceMode, ...],
    prompt_variants: tuple[str, ...],
    repetitions: tuple[int, ...],
) -> tuple[FullCaseMatrixCaseSpec, ...]:
    specs: list[FullCaseMatrixCaseSpec] = []
    for repetition in repetitions:
        for prompt_variant in prompt_variants:
            for evidence_mode in evidence_modes:
                specs.append(
                    FullCaseMatrixCaseSpec(
                        case_id=(
                            f"{len(specs) + 1:02d}_{abm}_{prompt_variant}_{evidence_mode.replace('+', '_plus_')}"
                            f"_rep{repetition}"
                        ),
                        abm=abm,
                        evidence_mode=evidence_mode,
                        prompt_variant=prompt_variant,
                        repetition=repetition,
                    )
                )
    return tuple(specs)


def run_full_case_matrix_smoke(
    *,
    case_input: FullCaseSmokeInput,
    adapter: LLMAdapter,
    model: str,
    output_root: Path,
    cases: tuple[FullCaseMatrixCaseSpec, ...],
    max_tokens: int = 32768,
    max_retries: int | None = None,
    retry_backoff_seconds: float | None = None,
    resume_existing: bool = True,
    max_case_attempts: int = DEFAULT_MAX_CASE_ATTEMPTS,
    on_case_completed: Callable[[FullCaseMatrixCaseResult], None] | None = None,
) -> FullCaseMatrixSmokeResult:
    """Run many full ABM cases in one separated run root with resume and review artifacts."""
    started_at = datetime.now(UTC)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = started_at.strftime("run_%Y%m%d_%H%M%S_%f")
    run_root = output_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    _write_text(latest_run_pointer_path(output_root), str(run_root))
    previous_run_root = _resolve_previous_run_root(output_root=output_root, current_run_id=run_id)
    run_log_path = attach_json_log_file(run_log_contract_path(run_root))
    _validate_full_case_inputs(case_input)

    context_cache: dict[str, _CachedContext] = {}
    context_futures: dict[str, Future[_CachedContext]] = {}
    context_lock = threading.Lock()
    case_results_by_id: dict[str, FullCaseMatrixCaseResult] = {}
    remaining_cases = list(cases)
    for _attempt in range(1, max(max_case_attempts, 1) + 1):
        next_remaining: list[FullCaseMatrixCaseSpec] = []
        max_case_workers = min(resolve_parallel_case_workers(adapter.provider), len(remaining_cases))
        with ThreadPoolExecutor(max_workers=max_case_workers) as executor:
            future_to_case: dict[Future[FullCaseMatrixCaseResult], FullCaseMatrixCaseSpec] = {}
            for case in remaining_cases:
                case_dir = run_root / "cases" / case.case_id
                previous_case_dir = previous_run_root / "cases" / case.case_id if previous_run_root else None
                if resume_existing and previous_case_dir and previous_case_dir.exists() and not case_dir.exists():
                    shutil.copytree(previous_case_dir, case_dir)
                else:
                    case_dir.mkdir(parents=True, exist_ok=True)
                future = executor.submit(
                    _run_full_case_matrix_case,
                    case_input=case_input,
                    adapter=adapter,
                    model=model,
                    case=case,
                    case_dir=case_dir,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    resume_existing=resume_existing,
                    context_cache=context_cache,
                    context_futures=context_futures,
                    context_lock=context_lock,
                    reused_from_previous=bool(previous_case_dir and previous_case_dir.exists()),
                )
                future_to_case[future] = case
            for future in as_completed(future_to_case):
                case = future_to_case[future]
                case_result = future.result()
                case_results_by_id[case.case_id] = case_result
                if on_case_completed is not None:
                    on_case_completed(case_result)
                if not case_result.success:
                    next_remaining.append(case)
        if not next_remaining:
            break
        wait_seconds = compute_matrix_retry_wait_seconds([case_results_by_id[case.case_id] for case in next_remaining])
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        remaining_cases = next_remaining

    case_results = [case_results_by_id[case.case_id] for case in cases]
    failed_case_ids = [case.case_id for case in case_results if not case.success]

    review_csv_path = run_root / "request_review.csv"
    write_run_review_csv(review_csv_path, build_run_review_rows(case_results))
    finished_at = datetime.now(UTC)
    smoke_result = FullCaseMatrixSmokeResult(
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        output_root=run_root,
        run_id=run_id,
        report_json_path=run_root / FULL_CASE_MATRIX_REPORT_FILENAME,
        report_markdown_path=run_root / "smoke_full_case_matrix_report.md",
        review_csv_path=review_csv_path,
        run_log_path=run_log_path,
        viewer_html_path=viewer_html_path(run_root),
        success=not failed_case_ids,
        failed_case_ids=failed_case_ids,
        cases=case_results,
    )
    _write_json(smoke_result.report_json_path, smoke_result.model_dump(mode="json"))
    _write_text(smoke_result.report_markdown_path, _render_report(smoke_result))
    smoke_result.viewer_html_path = render_run_viewer(run_root)
    _write_json(smoke_result.report_json_path, smoke_result.model_dump(mode="json"))
    return smoke_result


def _run_full_case_matrix_case(
    *,
    case_input: FullCaseSmokeInput,
    adapter: LLMAdapter,
    model: str,
    case: FullCaseMatrixCaseSpec,
    case_dir: Path,
    max_tokens: int,
    max_retries: int | None,
    retry_backoff_seconds: float | None,
    resume_existing: bool,
    context_cache: dict[str, _CachedContext],
    context_futures: dict[str, Future[_CachedContext]],
    context_lock: threading.Lock,
    reused_from_previous: bool,
) -> FullCaseMatrixCaseResult:
    inputs_dir = case_dir / "01_inputs"
    context_dir = case_dir / "02_context"
    trends_dir = case_dir / "03_trends"
    for directory in (inputs_dir, context_dir, trends_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shutil.copy2(case_input.parameters_path, inputs_dir / "parameters.txt")
    shutil.copy2(case_input.documentation_path, inputs_dir / "documentation.txt")
    context_prompt = build_legacy_doe_context_prompt(
        abm=case_input.abm,
        inputs_csv_path=case_input.parameters_path,
        inputs_doc_path=case_input.documentation_path,
        enabled=_enabled_features_from_variant(case.prompt_variant),
    )
    _write_text(inputs_dir / "context_prompt.txt", context_prompt)
    validation_state_file = validation_state_path(case_dir)
    validation_state = _load_validation_state(validation_state_file)
    _backfill_validation_state_from_artifacts(
        validation_state=validation_state,
        context_dir=context_dir,
        trends_dir=trends_dir,
        plots=case_input.plots,
    )
    review_rows: list[dict[str, str]] = []
    context_text = ""
    cached_context = context_cache.get(_context_cache_key(prompt=context_prompt, model=model))
    if resume_existing and _is_context_accepted(context_dir=context_dir, validation_state=validation_state):
        context_text = (context_dir / "context_output.txt").read_text(encoding="utf-8")
        if cached_context is None:
            context_cache[_context_cache_key(prompt=context_prompt, model=model)] = _CachedContext(
                text=context_text,
                trace=json.loads((context_dir / "context_trace.json").read_text(encoding="utf-8")),
            )
        _record_context_review_row(review_rows=review_rows, context_dir=context_dir, validation_status="accepted")
    elif cached_context is not None:
        _materialize_context_artifacts(
            requests_dir=context_dir,
            outputs_dir=context_dir,
            context_text=cached_context.text,
            context_trace=cached_context.trace,
        )
        validation_state.context = {"status": "accepted", "error": None}
        _record_context_review_row(review_rows=review_rows, context_dir=context_dir, validation_status="accepted")
        context_text = cached_context.text
    else:
        try:
            cached_context = _resolve_shared_matrix_context(
                adapter=adapter,
                model=model,
                prompt=context_prompt,
                max_tokens=max_tokens,
                ollama_num_ctx=0,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                context_cache=context_cache,
                context_futures=context_futures,
                context_lock=context_lock,
            )
        except StructuredSmokeResponseError as exc:
            _write_json(context_dir / "context_request.json", exc.trace["request"])
            _write_json(context_dir / "context_trace.json", exc.trace)
            _write_optional_thinking(context_dir / "context_thinking.txt", exc.trace)
            _write_text(context_dir / "error.txt", str(exc))
            validation_state.context = {"status": "retry", "error": str(exc)}
            _write_json(validation_state_file, validation_state.model_dump(mode="json"))
            _finalize_case(
                case_dir=case_dir,
                case=case,
                model=model,
                review_rows=review_rows,
                validation_state=validation_state,
            )
            return FullCaseMatrixCaseResult(
                case_id=case.case_id,
                abm=case.abm,
                evidence_mode=case.evidence_mode,
                prompt_variant=case.prompt_variant,
                repetition=case.repetition,
                case_dir=case_dir,
                success=False,
                resumed_from_existing=reused_from_previous,
                error=str(exc),
            )
        context_text = cached_context.text
        context_trace = cached_context.trace
        _materialize_context_artifacts(
            requests_dir=context_dir,
            outputs_dir=context_dir,
            context_text=context_text,
            context_trace=context_trace,
        )
        validation_state.context = {"status": "accepted", "error": None}
        if (context_dir / "error.txt").exists():
            (context_dir / "error.txt").unlink()
        _record_context_review_row(review_rows=review_rows, context_dir=context_dir, validation_status="accepted")

    frame = pd.read_csv(case_input.csv_path, sep=";")
    failed = False
    error_text = ""
    pending_plots: list[tuple[FullCasePlotInput, Path, Path | None]] = []
    for plot_input in case_input.plots:
        trend_dir = trends_dir / f"plot_{plot_input.plot_index:02d}"
        trend_dir.mkdir(parents=True, exist_ok=True)
        image_path = None
        if case.evidence_mode in {"plot", "plot+table"}:
            image_path = trend_dir / "trend_evidence_plot.png"
            shutil.copy2(plot_input.plot_path, image_path)
        if resume_existing and _is_trend_accepted(
            plot_index=plot_input.plot_index,
            trend_dir=trend_dir,
            validation_state=validation_state,
        ):
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
        pending_plots.append((plot_input, trend_dir, image_path))

    trend_results_by_index: dict[int, _MatrixTrendExecutionResult] = {}
    if pending_plots:
        max_trend_workers = min(resolve_parallel_trend_workers(adapter.provider), len(pending_plots))
        with ThreadPoolExecutor(max_workers=max_trend_workers) as executor:
            future_to_plot: dict[Future[_MatrixTrendExecutionResult], int] = {
                executor.submit(
                    _execute_matrix_trend,
                    adapter=adapter,
                    model=model,
                    case=case,
                    case_input=case_input,
                    plot_input=plot_input,
                    context_text=context_text,
                    trend_dir=trend_dir,
                    frame=frame,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    image_path=image_path,
                ): plot_input.plot_index
                for plot_input, trend_dir, image_path in pending_plots
            }
            for future in as_completed(future_to_plot):
                result_payload = future.result()
                trend_results_by_index[result_payload.plot_input.plot_index] = result_payload

    for plot_input in case_input.plots:
        trend_dir = trends_dir / f"plot_{plot_input.plot_index:02d}"
        image_path = trend_dir / "trend_evidence_plot.png" if (trend_dir / "trend_evidence_plot.png").exists() else None
        result_payload_opt = trend_results_by_index.get(plot_input.plot_index)
        if result_payload_opt is None:
            continue
        result_payload = result_payload_opt
        if result_payload.success:
            assert result_payload.trend_text is not None
            assert result_payload.trend_trace is not None
            trend_text = result_payload.trend_text
            trend_trace = result_payload.trend_trace
            _write_json(trend_dir / "trend_request.json", trend_trace["request"])
            _write_text(trend_dir / "trend_output.txt", trend_text)
            _write_json(trend_dir / "trend_trace.json", trend_trace)
            _write_optional_thinking(trend_dir / "trend_thinking.txt", trend_trace)
            if (trend_dir / "error.txt").exists():
                (trend_dir / "error.txt").unlink()
            validation_state.trends[str(plot_input.plot_index)] = {"status": "accepted", "error": None}
            _record_trend_review_row(
                review_rows=review_rows,
                plot_input=plot_input,
                trend_dir=trend_dir,
                image_path=image_path,
                success=True,
                error="",
                validation_status="accepted",
            )
        else:
            error = StructuredSmokeResponseError(
                result_payload.error or "trend execution failed",
                trace=result_payload.trend_trace or {"request": {}, "response": {"raw": None}},
                prompt=(trend_dir / "trend_prompt.txt").read_text(encoding="utf-8"),
            )
            error_text = str(error)
            _write_json(trend_dir / "trend_request.json", error.trace["request"])
            _write_json(trend_dir / "trend_trace.json", error.trace)
            _write_optional_thinking(trend_dir / "trend_thinking.txt", error.trace)
            _write_text(trend_dir / "error.txt", error_text)
            validation_state.trends[str(plot_input.plot_index)] = {"status": "retry", "error": error_text}
            _record_trend_review_row(
                review_rows=review_rows,
                plot_input=plot_input,
                trend_dir=trend_dir,
                image_path=image_path,
                success=False,
                error=error_text,
                validation_status="retry",
            )
            failed = True
            error_text = error_text

    _finalize_case(
        case_dir=case_dir,
        case=case,
        model=model,
        review_rows=review_rows,
        validation_state=validation_state,
    )
    return FullCaseMatrixCaseResult(
        case_id=case.case_id,
        abm=case.abm,
        evidence_mode=case.evidence_mode,
        prompt_variant=case.prompt_variant,
        repetition=case.repetition,
        case_dir=case_dir,
        success=not failed,
        resumed_from_existing=reused_from_previous,
        error=error_text or None,
    )


def _resolve_shared_matrix_context(
    *,
    adapter: LLMAdapter,
    model: str,
    prompt: str,
    max_tokens: int,
    ollama_num_ctx: int,
    max_retries: int | None,
    retry_backoff_seconds: float | None,
    context_cache: dict[str, _CachedContext],
    context_futures: dict[str, Future[_CachedContext]],
    context_lock: threading.Lock,
) -> _CachedContext:
    context_key = _context_cache_key(prompt=prompt, model=model)
    with context_lock:
        cached_context = context_cache.get(context_key)
        if cached_context is not None:
            return cached_context
        in_flight_future = context_futures.get(context_key)
        if in_flight_future is None:
            in_flight_future = Future()
            context_futures[context_key] = in_flight_future
            should_compute = True
        else:
            should_compute = False

    if should_compute:
        try:
            context_text, context_trace = _invoke_structured_smoke_text(
                adapter=adapter,
                model=model,
                prompt_with_schema=prompt,
                max_tokens=max_tokens,
                ollama_num_ctx=ollama_num_ctx,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            cached_context = _CachedContext(text=context_text, trace=context_trace)
        except Exception as exc:
            with context_lock:
                inflight = context_futures.pop(context_key, None)
                if inflight is not None and not inflight.done():
                    inflight.set_exception(exc)
            raise
        with context_lock:
            context_cache[context_key] = cached_context
            inflight = context_futures.pop(context_key, None)
            if inflight is not None and not inflight.done():
                inflight.set_result(cached_context)
        return cached_context

    return in_flight_future.result()


def _execute_matrix_trend(
    *,
    adapter: LLMAdapter,
    model: str,
    case: FullCaseMatrixCaseSpec,
    case_input: FullCaseSmokeInput,
    plot_input: FullCasePlotInput,
    context_text: str,
    trend_dir: Path,
    frame: pd.DataFrame,
    max_tokens: int,
    max_retries: int | None,
    retry_backoff_seconds: float | None,
    image_path: Path | None,
) -> _MatrixTrendExecutionResult:
    try:
        trend_text, trend_trace = _run_trend_with_fitting_table(
            adapter=adapter,
            model=model,
            case=case,
            case_input=case_input,
            plot_input=plot_input,
            context_text=context_text,
            trend_dir=trend_dir,
            frame=frame,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            image_path=image_path,
        )
    except StructuredSmokeResponseError as exc:
        return _MatrixTrendExecutionResult(
            plot_input=plot_input,
            trend_dir=trend_dir,
            image_path=image_path,
            success=False,
            trend_trace=dict(exc.trace),
            error=str(exc),
        )
    return _MatrixTrendExecutionResult(
        plot_input=plot_input,
        trend_dir=trend_dir,
        image_path=image_path,
        success=True,
        trend_text=trend_text,
        trend_trace=trend_trace,
    )


def _run_trend_with_fitting_table(
    *,
    adapter: LLMAdapter,
    model: str,
    case: FullCaseMatrixCaseSpec,
    case_input: FullCaseSmokeInput,
    plot_input: FullCasePlotInput,
    context_text: str,
    trend_dir: Path,
    frame: pd.DataFrame,
    max_tokens: int,
    max_retries: int | None,
    retry_backoff_seconds: float | None,
    image_path: Path | None,
) -> tuple[str, dict[str, object]]:
    last_exc: StructuredSmokeResponseError | None = None
    compression_attempts: list[PromptCompressionAttempt] = []
    compression_prompts: list[str] = []
    for stride in range(1, 65):
        table_csv = _build_trend_table_csv_for_stride(
            frame=frame,
            evidence_mode=case.evidence_mode,
            plot_input=plot_input,
            trend_dir=trend_dir,
            stride=stride,
        )
        trend_prompt = build_legacy_doe_trend_prompt(
            abm=case_input.abm,
            context_response=context_text,
            plot_description=plot_input.plot_description,
            evidence_mode=case.evidence_mode,
            table_csv=table_csv,
            enabled=_enabled_features_from_variant(case.prompt_variant),
        )
        _write_text(trend_dir / "trend_prompt.txt", trend_prompt)
        compression_attempts.append(
            PromptCompressionAttempt(
                attempt_index=len(compression_attempts) + 1,
                table_downsample_stride=stride,
                compression_tier=max(stride - 1, 0),
                prompt_length=len(trend_prompt),
            )
        )
        compression_prompts.append(trend_prompt)
        write_prompt_compression_artifacts(
            output_dir=trend_dir,
            attempts=compression_attempts,
            prompts=compression_prompts,
        )
        try:
            return _invoke_structured_smoke_text(
                adapter=adapter,
                model=model,
                prompt_with_schema=trend_prompt,
                image_b64=encode_image(image_path) if image_path is not None else None,
                max_tokens=max_tokens,
                ollama_num_ctx=0,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                request_metadata={"table_downsample_stride": stride},
            )
        except StructuredSmokeResponseError as exc:
            if case.evidence_mode not in {"table", "plot+table"} or not looks_like_context_overflow(str(exc)):
                raise
            last_exc = exc
        except LLMProviderError as exc:
            if case.evidence_mode not in {"table", "plot+table"} or not looks_like_context_overflow(str(exc)):
                raise
            last_exc = StructuredSmokeResponseError(
                str(exc),
                trace={"request": {"metadata": {"table_downsample_stride": stride}}, "response": {"raw": None}},
                prompt=trend_prompt,
            )
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("table fitting loop terminated without a result")


def _build_trend_table_csv_for_stride(
    *,
    frame: pd.DataFrame,
    evidence_mode: EvidenceMode,
    plot_input: FullCasePlotInput,
    trend_dir: Path,
    stride: int,
) -> str:
    if evidence_mode not in {"table", "plot+table"}:
        return ""
    return _build_trend_table_csv(
        frame=frame,
        evidence_mode=evidence_mode,
        plot_input=plot_input,
        trend_dir=trend_dir,
        compression_tier=max(stride - 1, 0),
    )


def _finalize_case(
    *,
    case_dir: Path,
    case: FullCaseMatrixCaseSpec,
    model: str,
    review_rows: list[dict[str, str]],
    validation_state: FullCaseValidationState,
) -> None:
    _write_json(
        case_summary_path(case_dir),
        {
            "case_id": case.case_id,
            "abm": case.abm,
            "evidence_mode": case.evidence_mode,
            "prompt_variant": case.prompt_variant,
            "repetition": case.repetition,
            "model": model,
        },
    )
    _write_json(validation_state_path(case_dir), validation_state.model_dump(mode="json"))
    _write_case_review_csv(case_dir / "review.csv", review_rows)


def _write_case_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    write_case_review_csv(path, rows)


def _resolve_previous_run_root(*, output_root: Path, current_run_id: str) -> Path | None:
    runs_root = output_root / "runs"
    if not runs_root.exists():
        return None
    candidates = sorted(
        (path for path in runs_root.iterdir() if path.is_dir() and path.name != current_run_id),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _render_report(result: FullCaseMatrixSmokeResult) -> str:
    lines = [
        "# Full Case Matrix Smoke Report",
        "",
        f"- success: `{str(result.success).lower()}`",
        f"- case_count: `{len(result.cases)}`",
        f"- failed_case_count: `{len(result.failed_case_ids)}`",
        f"- run_log_path: `{result.run_log_path}`",
        f"- viewer_html_path: `{result.viewer_html_path}`",
        "",
        "| case_id | evidence_mode | prompt_variant | repetition | success | reused |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in result.cases:
        lines.append(
            f"| {case.case_id} | {case.evidence_mode} | {case.prompt_variant} | {case.repetition} | "
            f"{str(case.success).lower()} | {str(case.resumed_from_existing).lower()} |"
        )
    return "\n".join(lines) + "\n"


def compute_matrix_retry_wait_seconds(case_results: list[FullCaseMatrixCaseResult]) -> float:
    """Return how long to wait before another failed-case pass."""
    for case_result in case_results:
        if case_result.error and (
            is_transient_provider_error(case_result.error) or "circuit open" in case_result.error.lower()
        ):
            return DEFAULT_MATRIX_PASS_WAIT_SECONDS
    return 0.0
