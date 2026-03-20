"""Reviewer-friendly smoke run for local summarizers on one validated full-case bundle."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field

from distill_abm.configs.models import SummarizerId
from distill_abm.pipeline.local_qwen_sample_response import validate_structured_smoke_text_content
from distill_abm.pipeline.report_writers import write_model_report_files
from distill_abm.pipeline.run_artifact_contracts import latest_run_pointer_path, read_active_run_lock, run_log_path
from distill_abm.structured_logging import attach_json_log_file, get_logger, log_event
from distill_abm.summarize.models import (
    summarize_with_bart,
    summarize_with_bert,
    summarize_with_longformer_ext,
    summarize_with_t5,
)
from distill_abm.summarize.postprocess import postprocess_summary
from distill_abm.utils import detect_placeholder_signals

SummarizerSmokeMode = Literal["none", "bart", "bert", "t5", "longformer_ext"]
ConfiguredSummarizerSmokeMode = Literal["bart", "bert", "t5", "longformer_ext"]


class ValidatedSmokeBundle(BaseModel):
    """One manually validated full-case bundle reused for summarizer smoke."""

    bundle_id: str
    case_id: str
    abm: str
    context_output_path: Path
    trend_output_paths: tuple[Path, ...]
    validation_note: str


class SummarizerModeResult(BaseModel):
    """One summarizer-mode outcome for a validated bundle."""

    mode: SummarizerSmokeMode
    success: bool
    output_path: Path
    raw_output_path: Path | None = None
    postprocess_changed: bool = False
    error: str | None = None
    duration_seconds: float
    input_length: int
    output_length: int = 0


class SummarizerBundleResult(BaseModel):
    """Smoke result for one validated bundle."""

    bundle_id: str
    case_id: str
    abm: str
    bundle_dir: Path
    success: bool
    source_validation_error: str | None = None
    modes: list[SummarizerModeResult] = Field(default_factory=list)


class SummarizerSmokeResult(BaseModel):
    """Top-level summarizer smoke result."""

    started_at_utc: str
    finished_at_utc: str
    output_root: Path
    run_id: str
    run_root: Path
    report_json_path: Path
    report_markdown_path: Path
    review_csv_path: Path
    validated_sources_path: Path
    run_log_path: Path
    success: bool
    failed_bundle_ids: list[str] = Field(default_factory=list)
    bundles: list[SummarizerBundleResult] = Field(default_factory=list)


def default_validated_smoke_bundles(
    source_root: Path,
    *,
    include_abms: tuple[str, ...] | None = None,
) -> tuple[ValidatedSmokeBundle, ...]:
    """Return validated smoke bundles discovered from full-case or matrix run artifacts."""
    suite_abms_root = source_root / "abms"
    if suite_abms_root.exists():
        return _discover_full_case_suite_bundles(source_root=source_root, include_abms=include_abms)
    matrix_report_path = source_root / "smoke_full_case_matrix_report.json"
    if matrix_report_path.exists():
        bundles = _discover_full_case_matrix_bundles(source_root, matrix_report_path)
        return _filter_bundles_by_abm(bundles, include_abms=include_abms)
    case_root = _discover_best_full_case_bundle(source_root / "cases")
    case_id = case_root.name
    bundles = (
        ValidatedSmokeBundle(
            bundle_id=case_id,
            case_id=case_id,
            abm="grazing",
            context_output_path=case_root / "02_context" / "context_output.txt",
            trend_output_paths=tuple(sorted((case_root / "03_trends").glob("plot_*/trend_output.txt"))),
            validation_note=(
                "Functionality smoke bundle selected from the latest completed Nemotron full-case artifacts. "
                "Used to exercise the summarizer pipeline over one context output plus all available trend outputs."
            ),
        ),
    )
    return _filter_bundles_by_abm(bundles, include_abms=include_abms)


def run_summarizer_smoke(
    *,
    source_root: Path,
    output_root: Path,
    resume: bool = False,
    watch: bool = False,
    poll_interval_seconds: float = 5.0,
    include_abms: tuple[str, ...] | None = None,
    validated_bundles: tuple[ValidatedSmokeBundle, ...] | None = None,
    summarizer_modes: tuple[SummarizerId, ...] | None = None,
    summarizer_fns: dict[ConfiguredSummarizerSmokeMode, Callable[[str], str]] | None = None,
) -> SummarizerSmokeResult:
    """Run all summarizers over validated full-case text bundles and persist review artifacts."""
    started_at = datetime.now(UTC)
    _prepare_output_root(output_root, resume=resume)
    run_id = started_at.strftime("run_%Y%m%d_%H%M%S_%f")
    run_root = output_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    latest_run_pointer_path(output_root).write_text(str(run_root), encoding="utf-8")
    previous_run_root = _resolve_previous_summarizer_run_root(output_root=output_root, current_run_id=run_id)
    logger = get_logger(__name__)
    attached_run_log_path = attach_json_log_file(run_log_path(run_root))
    available_modes: tuple[ConfiguredSummarizerSmokeMode, ...] = ("bart", "bert", "t5", "longformer_ext")
    resolved_summarizers = summarizer_fns or {
        "bart": summarize_with_bart,
        "bert": summarize_with_bert,
        "t5": summarize_with_t5,
        "longformer_ext": summarize_with_longformer_ext,
    }
    selected_summarizer_modes: tuple[ConfiguredSummarizerSmokeMode, ...] = summarizer_modes or tuple(
        mode for mode in available_modes if mode in resolved_summarizers
    )
    missing_modes = [mode for mode in selected_summarizer_modes if mode not in resolved_summarizers]
    if missing_modes:
        missing = ", ".join(missing_modes)
        raise ValueError(f"missing summarizer implementation(s) for requested mode(s): {missing}")
    bundle_results: list[SummarizerBundleResult] = []
    review_rows: list[dict[str, str]] = []
    failed_bundle_ids: list[str] = []
    validated_sources_path = run_root / "validated_bundles.json"
    processed_bundle_ids: set[str] = set()
    processed_bundles: list[ValidatedSmokeBundle] = []

    while True:
        selected_bundles = _discover_candidate_bundles(
            source_root=source_root,
            include_abms=include_abms,
            validated_bundles=validated_bundles,
            watch=watch,
        )
        pending_bundles = [bundle for bundle in selected_bundles if bundle.bundle_id not in processed_bundle_ids]
        for bundle in pending_bundles:
            processed_bundle_ids.add(bundle.bundle_id)
            processed_bundles.append(bundle)
            _process_bundle(
                bundle=bundle,
                run_root=run_root,
                previous_run_root=previous_run_root,
                resume=resume,
                logger=logger,
                resolved_summarizers=resolved_summarizers,
                selected_summarizer_modes=selected_summarizer_modes,
                bundle_results=bundle_results,
                review_rows=review_rows,
                failed_bundle_ids=failed_bundle_ids,
            )
        if not watch:
            break
        if not _source_run_is_active(source_root):
            break
        time.sleep(max(poll_interval_seconds, 0.0))

    validated_sources_path.write_text(
        json.dumps([bundle.model_dump(mode="json") for bundle in processed_bundles], indent=2),
        encoding="utf-8",
    )

    result = SummarizerSmokeResult(
        started_at_utc=started_at.isoformat(),
        finished_at_utc=datetime.now(UTC).isoformat(),
        output_root=output_root,
        run_id=run_id,
        run_root=run_root,
        report_json_path=run_root / "smoke_summarizers_report.json",
        report_markdown_path=run_root / "smoke_summarizers_report.md",
        review_csv_path=run_root / "review.csv",
        validated_sources_path=validated_sources_path,
        run_log_path=attached_run_log_path,
        success=not failed_bundle_ids,
        failed_bundle_ids=failed_bundle_ids,
        bundles=bundle_results,
    )
    _write_review_csv(result.review_csv_path, review_rows)
    write_model_report_files(
        result=result,
        report_json_path=result.report_json_path,
        report_markdown_path=result.report_markdown_path,
        markdown=_render_markdown_report(result),
    )
    return result


def _process_bundle(
    *,
    bundle: ValidatedSmokeBundle,
    run_root: Path,
    previous_run_root: Path | None,
    resume: bool,
    logger: logging.Logger,
    resolved_summarizers: dict[ConfiguredSummarizerSmokeMode, Callable[[str], str]],
    selected_summarizer_modes: tuple[ConfiguredSummarizerSmokeMode, ...],
    bundle_results: list[SummarizerBundleResult],
    review_rows: list[dict[str, str]],
    failed_bundle_ids: list[str],
) -> None:
    """Run summarizers for one validated bundle and append results to the accumulators."""

    bundle_dir = run_root / "bundles" / bundle.bundle_id
    if resume:
        _copy_previous_bundle_if_present(
            bundle_id=bundle.bundle_id,
            previous_run_root=previous_run_root,
            bundle_dir=bundle_dir,
        )
    input_dir = bundle_dir / "01_input"
    trend_dir = input_dir / "trend_outputs"
    summary_dir = bundle_dir / "02_summaries"
    metadata_dir = bundle_dir / "03_metadata"
    raw_summary_dir = metadata_dir / "raw_summaries"
    for directory in (input_dir, trend_dir, summary_dir, metadata_dir, raw_summary_dir):
        directory.mkdir(parents=True, exist_ok=True)
    log_event(logger, "summarizer_bundle_start", bundle_id=bundle.bundle_id, case_id=bundle.case_id, abm=bundle.abm)

    combined_input = ""
    source_validation_error: str | None = None
    try:
        context_text = _validate_source_text(bundle.context_output_path)
        trend_texts = [_validate_source_text(path) for path in bundle.trend_output_paths]
        if not trend_texts:
            raise ValueError("validated bundle must contain at least one trend output")
        combined_input = _combine_bundle_input(context_text=context_text, trend_texts=trend_texts)
        (input_dir / "context_output.txt").write_text(context_text, encoding="utf-8")
        for index, trend_text in enumerate(trend_texts, start=1):
            (trend_dir / f"{index:02d}.txt").write_text(trend_text, encoding="utf-8")
        (input_dir / "combined_input.txt").write_text(combined_input, encoding="utf-8")
        (bundle_dir / "00_bundle_summary.json").write_text(
            json.dumps(
                {
                    "bundle_id": bundle.bundle_id,
                    "case_id": bundle.case_id,
                    "abm": bundle.abm,
                    "context_output_path": str(bundle.context_output_path),
                    "trend_output_paths": [str(path) for path in bundle.trend_output_paths],
                    "validation_note": bundle.validation_note,
                    "context_length": len(context_text),
                    "trend_count": len(trend_texts),
                    "combined_input_length": len(combined_input),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        source_validation_error = str(exc)

    mode_results: list[SummarizerModeResult] = []
    if source_validation_error is None:
        all_modes = cast(tuple[SummarizerSmokeMode, ...], ("none",) + selected_summarizer_modes)
        for mode in all_modes:
            output_path = summary_dir / f"{mode}.txt"
            raw_output_path = None if mode == "none" else raw_summary_dir / f"{mode}.txt"
            existing_result = _load_resumable_mode_result(
                mode=mode,
                output_path=output_path,
                raw_output_path=raw_output_path,
                input_length=len(combined_input),
                resume=resume,
            )
            if existing_result is not None:
                mode_result = existing_result
                log_event(logger, "summarizer_mode_reused", bundle_id=bundle.bundle_id, mode=mode)
            else:
                started = time.perf_counter()
                try:
                    raw_summary_text = combined_input if mode == "none" else resolved_summarizers[mode](combined_input)
                    summary_text = raw_summary_text if mode == "none" else postprocess_summary(raw_summary_text).strip()
                    duration = time.perf_counter() - started
                    output_path.write_text(summary_text, encoding="utf-8")
                    if raw_output_path is not None:
                        raw_output_path.write_text(raw_summary_text, encoding="utf-8")
                    mode_result = SummarizerModeResult(
                        mode=mode,
                        success=bool(summary_text.strip()),
                        output_path=output_path,
                        raw_output_path=raw_output_path,
                        postprocess_changed=raw_summary_text.strip() != summary_text.strip(),
                        duration_seconds=duration,
                        input_length=len(combined_input),
                        output_length=len(summary_text),
                        error=None if summary_text.strip() else "summarizer produced empty text",
                    )
                    log_event(
                        logger,
                        "summarizer_mode_success",
                        bundle_id=bundle.bundle_id,
                        mode=mode,
                        output_length=len(summary_text),
                    )
                except Exception as exc:
                    duration = time.perf_counter() - started
                    output_path.write_text("", encoding="utf-8")
                    mode_result = SummarizerModeResult(
                        mode=mode,
                        success=False,
                        output_path=output_path,
                        raw_output_path=raw_output_path,
                        duration_seconds=duration,
                        input_length=len(combined_input),
                        output_length=0,
                        error=str(exc),
                    )
                    log_event(
                        logger,
                        "summarizer_mode_failure",
                        level=40,
                        bundle_id=bundle.bundle_id,
                        mode=mode,
                        error=str(exc),
                    )
            mode_results.append(mode_result)
            review_rows.append(
                {
                    "bundle_id": bundle.bundle_id,
                    "case_id": bundle.case_id,
                    "abm": bundle.abm,
                    "mode": mode,
                    "success": str(mode_result.success),
                    "context_output_path": str(bundle.context_output_path),
                    "trend_output_paths": "|".join(str(path) for path in bundle.trend_output_paths),
                    "combined_input_path": str(input_dir / "combined_input.txt"),
                    "summary_output_path": str(output_path),
                    "input_length": str(mode_result.input_length),
                    "output_length": str(mode_result.output_length),
                    "duration_seconds": f"{mode_result.duration_seconds:.6f}",
                    "validation_note": bundle.validation_note,
                    "error": mode_result.error or "",
                }
            )

    bundle_success = source_validation_error is None and all(result.success for result in mode_results)
    if not bundle_success:
        failed_bundle_ids.append(bundle.bundle_id)
    bundle_result = SummarizerBundleResult(
        bundle_id=bundle.bundle_id,
        case_id=bundle.case_id,
        abm=bundle.abm,
        bundle_dir=bundle_dir,
        success=bundle_success,
        source_validation_error=source_validation_error,
        modes=mode_results,
    )
    bundle_results.append(bundle_result)
    (metadata_dir / "mode_results.json").write_text(
        bundle_result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    log_event(logger, "summarizer_bundle_complete", bundle_id=bundle.bundle_id, success=bundle_success)


def _discover_candidate_bundles(
    *,
    source_root: Path,
    include_abms: tuple[str, ...] | None,
    validated_bundles: tuple[ValidatedSmokeBundle, ...] | None,
    watch: bool,
) -> tuple[ValidatedSmokeBundle, ...]:
    if validated_bundles is not None:
        return validated_bundles
    try:
        return default_validated_smoke_bundles(source_root, include_abms=include_abms)
    except FileNotFoundError:
        if watch and _source_run_is_active(source_root):
            return ()
        raise


def _source_run_is_active(source_root: Path) -> bool:
    """Return whether the source generation smoke still has a live active-run lock."""

    return read_active_run_lock(source_root) is not None


def _combine_bundle_input(*, context_text: str, trend_texts: list[str]) -> str:
    parts = [context_text.strip()]
    parts.extend(text.strip() for text in trend_texts if text.strip())
    return "\n\n".join(part for part in parts if part)


def _validate_source_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"missing validated source: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"validated source is empty: {path}")
    lowered = text.lower()
    if "analysis is currently unavailable" in lowered or "please try again later" in lowered:
        raise ValueError(f"validated source contains generic unavailable text: {path}")
    if detect_placeholder_signals(text):
        raise ValueError(f"validated source contains placeholder-like text: {path}")
    return text


def _write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "bundle_id",
        "case_id",
        "abm",
        "mode",
        "success",
        "context_output_path",
        "trend_output_paths",
        "combined_input_path",
        "summary_output_path",
        "input_length",
        "output_length",
        "duration_seconds",
        "validation_note",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown_report(result: SummarizerSmokeResult) -> str:
    lines = [
        "# Summarizer Smoke Report",
        "",
        f"- success: `{str(result.success).lower()}`",
        f"- bundle_count: `{len(result.bundles)}`",
        f"- failed_bundle_count: `{len(result.failed_bundle_ids)}`",
        f"- review_csv_path: `{result.review_csv_path}`",
        "",
        "| bundle_id | case_id | abm | success |",
        "| --- | --- | --- | --- |",
    ]
    for bundle in result.bundles:
        lines.append(f"| {bundle.bundle_id} | {bundle.case_id} | {bundle.abm} | {str(bundle.success).lower()} |")
    return "\n".join(lines) + "\n"


def _prepare_output_root(output_root: Path, *, resume: bool) -> None:
    if output_root.exists() and not resume:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _resolve_previous_summarizer_run_root(*, output_root: Path, current_run_id: str) -> Path | None:
    runs_root = output_root / "runs"
    if runs_root.exists():
        candidates = sorted(
            (path for path in runs_root.iterdir() if path.is_dir() and path.name != current_run_id),
            reverse=True,
        )
        if candidates:
            return candidates[0]
    legacy_bundles_root = output_root / "bundles"
    if legacy_bundles_root.exists():
        return output_root
    return None


def _copy_previous_bundle_if_present(*, bundle_id: str, previous_run_root: Path | None, bundle_dir: Path) -> None:
    if previous_run_root is None:
        return
    previous_bundle_dir = previous_run_root / "bundles" / bundle_id
    if not previous_bundle_dir.exists() or bundle_dir.exists():
        return
    shutil.copytree(previous_bundle_dir, bundle_dir)


def _load_resumable_mode_result(
    *,
    mode: SummarizerSmokeMode,
    output_path: Path,
    raw_output_path: Path | None,
    input_length: int,
    resume: bool,
) -> SummarizerModeResult | None:
    if not resume or not output_path.exists():
        return None
    try:
        summary_text = _validate_source_text(output_path)
    except Exception:
        return None
    existing_raw_output_path = raw_output_path if raw_output_path is not None and raw_output_path.exists() else None
    postprocess_changed = False
    if existing_raw_output_path is not None:
        raw_summary_text = existing_raw_output_path.read_text(encoding="utf-8").strip()
        postprocess_changed = raw_summary_text != summary_text.strip()
    return SummarizerModeResult(
        mode=mode,
        success=True,
        output_path=output_path,
        raw_output_path=existing_raw_output_path,
        postprocess_changed=postprocess_changed,
        duration_seconds=0.0,
        input_length=input_length,
        output_length=len(summary_text),
        error=None,
    )


def _discover_best_full_case_bundle(cases_root: Path) -> Path:
    candidates: list[tuple[int, str, Path]] = []
    for case_dir in sorted(cases_root.glob("*_full_case")):
        context_path = case_dir / "02_context" / "context_output.txt"
        trend_paths = sorted((case_dir / "03_trends").glob("plot_*/trend_output.txt"))
        if not context_path.exists() or not trend_paths:
            continue
        candidates.append((len(trend_paths), case_dir.name, case_dir))
    if not candidates:
        raise FileNotFoundError(f"no completed full-case smoke bundle found under {cases_root}")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _discover_full_case_matrix_bundles(source_root: Path, report_path: Path) -> tuple[ValidatedSmokeBundle, ...]:
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    bundles: list[ValidatedSmokeBundle] = []
    for case_payload in report_payload.get("cases", []):
        if not case_payload.get("success"):
            continue
        case_dir = Path(case_payload["case_dir"])
        context_output_path = case_dir / "02_context" / "context_output.txt"
        trend_output_paths = tuple(sorted((case_dir / "03_trends").glob("plot_*/trend_output.txt")))
        if not context_output_path.exists() or not trend_output_paths:
            continue
        case_id = str(case_payload["case_id"])
        abm = str(case_payload.get("abm", "unknown"))
        bundles.append(
            ValidatedSmokeBundle(
                bundle_id=case_id,
                case_id=case_id,
                abm=abm,
                context_output_path=context_output_path,
                trend_output_paths=trend_output_paths,
                validation_note=(
                    "Bundle discovered from a completed full-case matrix smoke run. "
                    "Used to exercise the summarizer pipeline over one context output plus all trend outputs."
                ),
            )
        )
    if not bundles:
        raise FileNotFoundError(f"no successful full-case matrix smoke bundles found under {source_root}")
    return tuple(sorted(bundles, key=lambda bundle: bundle.case_id))


def _discover_full_case_suite_bundles(
    *,
    source_root: Path,
    include_abms: tuple[str, ...] | None = None,
) -> tuple[ValidatedSmokeBundle, ...]:
    abms_root = source_root / "abms"
    target_abms = include_abms or tuple(sorted(path.name for path in abms_root.iterdir() if path.is_dir()))
    bundles: list[ValidatedSmokeBundle] = []
    missing_abms: list[str] = []
    for abm in target_abms:
        abm_root = abms_root / abm
        latest_run_path = abm_root / "latest_run.txt"
        if not latest_run_path.exists():
            missing_abms.append(abm)
            continue
        run_root = Path(latest_run_path.read_text(encoding="utf-8").strip())
        report_path = run_root / "smoke_full_case_matrix_report.json"
        abm_bundles: tuple[ValidatedSmokeBundle, ...] = ()
        if report_path.exists():
            abm_bundles = _discover_full_case_matrix_bundles(run_root, report_path)
        else:
            abm_bundles = _discover_live_full_case_matrix_bundles(run_root, fallback_abm=abm)
        if not abm_bundles:
            missing_abms.append(abm)
            continue
        bundles.extend(abm_bundles)
    if include_abms and missing_abms:
        raise FileNotFoundError(f"missing completed ABM runs for: {', '.join(sorted(missing_abms))}")
    if not bundles:
        raise FileNotFoundError(f"no successful full-case matrix smoke bundles found under suite root {source_root}")
    return tuple(sorted(bundles, key=lambda bundle: (bundle.abm, bundle.case_id)))


def _discover_live_full_case_matrix_bundles(run_root: Path, *, fallback_abm: str) -> tuple[ValidatedSmokeBundle, ...]:
    cases_root = run_root / "cases"
    if not cases_root.exists():
        return ()
    bundles: list[ValidatedSmokeBundle] = []
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        bundle = _build_live_validated_bundle(case_dir, fallback_abm=fallback_abm)
        if bundle is not None:
            bundles.append(bundle)
    return tuple(bundles)


def _build_live_validated_bundle(case_dir: Path, *, fallback_abm: str) -> ValidatedSmokeBundle | None:
    validation_path = case_dir / "validation_state.json"
    context_output_path = case_dir / "02_context" / "context_output.txt"
    if not validation_path.exists() or not context_output_path.exists():
        return None
    try:
        validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    context_payload = validation_payload.get("context", {})
    if not isinstance(context_payload, dict) or context_payload.get("status") != "accepted":
        return None
    try:
        validate_structured_smoke_text_content(context_output_path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    trend_payloads = validation_payload.get("trends", {})
    if not isinstance(trend_payloads, dict):
        return None
    trend_output_paths: list[Path] = []
    for key, payload in sorted(trend_payloads.items(), key=lambda item: int(item[0])):
        if not isinstance(payload, dict) or payload.get("status") != "accepted":
            return None
        plot_index = int(key)
        trend_output_path = case_dir / "03_trends" / f"plot_{plot_index:02d}" / "trend_output.txt"
        if not trend_output_path.exists():
            return None
        try:
            validate_structured_smoke_text_content(trend_output_path.read_text(encoding="utf-8"))
        except ValueError:
            return None
        trend_output_paths.append(trend_output_path)
    if not trend_output_paths:
        return None
    case_summary_path = case_dir / "00_case_summary.json"
    case_id = case_dir.name
    abm = fallback_abm
    if case_summary_path.exists():
        try:
            summary_payload = json.loads(case_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary_payload = {}
        if isinstance(summary_payload, dict):
            case_id = str(summary_payload.get("case_id", case_id))
            abm = str(summary_payload.get("abm", abm))
    return ValidatedSmokeBundle(
        bundle_id=case_id,
        case_id=case_id,
        abm=abm,
        context_output_path=context_output_path,
        trend_output_paths=tuple(trend_output_paths),
        validation_note=(
            "Bundle discovered from an accepted case in a live full-case suite run. "
            "Used to summarize outputs as soon as a case finishes."
        ),
    )


def _filter_bundles_by_abm(
    bundles: tuple[ValidatedSmokeBundle, ...],
    *,
    include_abms: tuple[str, ...] | None,
) -> tuple[ValidatedSmokeBundle, ...]:
    if not include_abms:
        return bundles
    selected = tuple(bundle for bundle in bundles if bundle.abm in include_abms)
    if not selected:
        raise FileNotFoundError(f"no validated bundles found for ABMs: {', '.join(include_abms)}")
    return selected
