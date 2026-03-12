"""Persistence and reproducibility helpers for benchmark pipeline runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from shutil import copy2
from typing import TYPE_CHECKING, cast

from distill_abm.configs.runtime_defaults import get_runtime_defaults
from distill_abm.eval.metrics import SummaryScores
from distill_abm.llm.request_defaults import resolve_request_temperature
from distill_abm.utils import detect_placeholder_signals

if TYPE_CHECKING:
    from distill_abm.configs.models import PromptsConfig
    from distill_abm.llm.adapters.base import LLMAdapter
    from distill_abm.pipeline.run import PipelineInputs, PipelineResult


def build_run_signature(inputs: PipelineInputs, prompts: PromptsConfig, adapter: LLMAdapter) -> str:
    """Compute a deterministic fingerprint for a pipeline run configuration."""
    runtime_defaults = get_runtime_defaults()
    request_temperature = resolve_request_temperature(adapter.provider)
    payload = {
        "inputs": {
            "csv_path": str(inputs.csv_path.resolve()),
            "parameters_path": str(inputs.parameters_path.resolve()),
            "documentation_path": str(inputs.documentation_path.resolve()),
            "metric_pattern": inputs.metric_pattern,
            "metric_description": inputs.metric_description,
            "plot_description": inputs.plot_description,
            "text_source_mode": inputs.text_source_mode,
            "evidence_mode": inputs.evidence_mode,
            "summarizers": list(inputs.summarizers),
            "enabled_style_features": list(inputs.enabled_style_features) if inputs.enabled_style_features else None,
            "scoring_reference_path": (
                str(inputs.scoring_reference_path.resolve()) if inputs.scoring_reference_path is not None else None
            ),
            "additional_scoring_reference_paths": {
                reference_name: str(reference_path.resolve())
                for reference_name, reference_path in sorted(inputs.additional_scoring_reference_paths.items())
            },
        },
        "input_file_hashes": {
            "csv": _hash_file(inputs.csv_path),
            "parameters": _hash_file(inputs.parameters_path),
            "documentation": _hash_file(inputs.documentation_path),
            "scoring_reference": _hash_file(inputs.scoring_reference_path),
            "additional_scoring_references": {
                reference_name: _hash_file(reference_path)
                for reference_name, reference_path in sorted(inputs.additional_scoring_reference_paths.items())
            },
        },
        "llm": {
            "provider": adapter.provider,
            "model": inputs.model,
            "temperature": request_temperature,
            "max_tokens": runtime_defaults.llm_request.max_tokens,
            "max_retries": runtime_defaults.llm_request.max_retries,
            "retry_backoff_seconds": runtime_defaults.llm_request.retry_backoff_seconds,
        },
        "prompts": prompts.model_dump(mode="json"),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_run_signature(inputs: PipelineInputs, prompts: PromptsConfig, adapter: LLMAdapter) -> str:
    """Backward-compatible private helper for existing call sites."""
    return build_run_signature(inputs=inputs, prompts=prompts, adapter=adapter)


def _hash_file(path: Path | None) -> str | None:
    """Return a stable digest for resumable-run inputs."""
    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_resumable_pipeline_result(output_dir: Path, run_signature: str) -> PipelineResult | None:
    """Load a cached pipeline result when signature and artifacts are still valid."""
    from distill_abm.pipeline.run import PipelineResult

    metadata_path = output_dir / "pipeline_run_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    reproducibility = _as_dict(payload.get("reproducibility"))
    if str(reproducibility.get("run_signature", "")) != run_signature:
        return None

    try:
        artifacts = _as_dict(payload.get("artifacts"))
        scores_payload = _as_dict(payload.get("scores"))
        responses = _as_dict(payload.get("responses"))

        plot_path, report_csv, stats_table_csv, stats_image_path = _extract_resumable_artifacts(artifacts)
        if plot_path is None or report_csv is None:
            return None

        selected_scores, full_scores, summary_scores = _extract_resumable_scores(scores_payload)
        if selected_scores is None:
            return None

        trend_full = str(responses.get("trend_full_response", ""))
        trend_summary_raw = responses.get("trend_summary_response")
        trend_summary = str(trend_summary_raw) if isinstance(trend_summary_raw, str) else None
        trend_response = trend_summary if trend_summary is not None else trend_full

        return PipelineResult(
            plot_path=plot_path,
            report_csv=report_csv,
            context_response=str(responses.get("context_response", "")),
            trend_response=trend_response,
            trend_full_response=trend_full,
            trend_summary_response=trend_summary,
            full_scores=full_scores,
            summary_scores=summary_scores,
            stats_table_csv=stats_table_csv,
            stats_image_path=stats_image_path,
            token_f1=selected_scores.token_f1,
            bleu=selected_scores.bleu,
            meteor=selected_scores.meteor,
            rouge1=selected_scores.rouge1,
            rouge2=selected_scores.rouge2,
            rouge_l=selected_scores.rouge_l,
            flesch_reading_ease=selected_scores.flesch_reading_ease,
            metadata_path=metadata_path,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _load_resumable_pipeline_result(output_dir: Path, run_signature: str) -> PipelineResult | None:
    """Backward-compatible private helper for legacy call sites and tests."""
    return load_resumable_pipeline_result(output_dir=output_dir, run_signature=run_signature)


def write_run_metadata(
    output_dir: Path,
    inputs: PipelineInputs,
    context_prompt: str,
    plot_path: Path,
    report_csv: Path,
    stats_table_csv_path: Path,
    stats_image_path: Path | None,
    trend_prompt: str,
    context_response: str,
    trend_full: str,
    trend_summary: str | None,
    selected_text_source: str,
    allow_summary_fallback: bool,
    full_scores: SummaryScores | None,
    summary_scores: SummaryScores | None,
    include_extended: bool,
    scoring_reference_text: str,
    scoring_reference_source: str,
    scoring_reference_path: Path | None,
    additional_scoring_references: Mapping[str, tuple[str, str, Path]] | None,
    additional_reference_scores: Mapping[str, Mapping[str, SummaryScores | None]] | None,
    include_pattern: str,
    evidence_mode: str,
    requested_evidence_mode: str,
    adapter: LLMAdapter,
    selected_scores: SummaryScores,
    trend_image_attached: bool,
    run_signature: str,
    context_trace: dict[str, object],
    trend_trace: dict[str, object],
    summarization_trace: dict[str, object],
    frame_summary: dict[str, object],
) -> Path:
    """Persist deterministic metadata for a completed pipeline run."""
    debug_trace = _write_debug_trace_bundle(
        output_dir=output_dir,
        inputs=inputs,
        plot_path=plot_path,
        report_csv=report_csv,
        stats_table_csv_path=stats_table_csv_path,
        stats_image_path=stats_image_path,
        scoring_reference_path=scoring_reference_path,
        additional_scoring_reference_paths=(
            {name: path for name, (_text, _source, path) in additional_scoring_references.items()}
            if additional_scoring_references is not None
            else None
        ),
        context_trace=context_trace,
        trend_trace=trend_trace,
        summarization_trace=summarization_trace,
        frame_summary=frame_summary,
    )
    metadata = _build_metadata_payload(
        output_dir=output_dir,
        inputs=inputs,
        context_prompt=context_prompt,
        plot_path=plot_path,
        report_csv=report_csv,
        stats_table_csv_path=stats_table_csv_path,
        stats_image_path=stats_image_path,
        trend_prompt=trend_prompt,
        context_response=context_response,
        trend_full=trend_full,
        trend_summary=trend_summary,
        selected_text_source=selected_text_source,
        allow_summary_fallback=allow_summary_fallback,
        full_scores=full_scores,
        summary_scores=summary_scores,
        include_extended=include_extended,
        scoring_reference_text=scoring_reference_text,
        scoring_reference_source=scoring_reference_source,
        scoring_reference_path=scoring_reference_path,
        additional_scoring_references=additional_scoring_references,
        additional_reference_scores=additional_reference_scores,
        include_pattern=include_pattern,
        evidence_mode=evidence_mode,
        requested_evidence_mode=requested_evidence_mode,
        adapter=adapter,
        selected_scores=selected_scores,
        trend_image_attached=trend_image_attached,
        run_signature=run_signature,
        debug_trace=debug_trace,
    )
    metadata_path = output_dir / "pipeline_run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def _write_run_metadata(
    output_dir: Path,
    inputs: PipelineInputs,
    context_prompt: str,
    plot_path: Path,
    report_csv: Path,
    stats_table_csv_path: Path,
    stats_image_path: Path | None,
    trend_prompt: str,
    context_response: str,
    trend_full: str,
    trend_summary: str | None,
    selected_text_source: str,
    allow_summary_fallback: bool,
    full_scores: SummaryScores | None,
    summary_scores: SummaryScores | None,
    include_extended: bool,
    scoring_reference_text: str,
    scoring_reference_source: str,
    scoring_reference_path: Path | None,
    additional_scoring_references: Mapping[str, tuple[str, str, Path]] | None,
    additional_reference_scores: Mapping[str, Mapping[str, SummaryScores | None]] | None,
    include_pattern: str,
    evidence_mode: str,
    requested_evidence_mode: str,
    adapter: LLMAdapter,
    selected_scores: SummaryScores,
    trend_image_attached: bool,
    run_signature: str,
    context_trace: dict[str, object],
    trend_trace: dict[str, object],
    summarization_trace: dict[str, object],
    frame_summary: dict[str, object],
) -> Path:
    """Backward-compatible private helper for older internal callers."""
    return write_run_metadata(
        output_dir=output_dir,
        inputs=inputs,
        context_prompt=context_prompt,
        plot_path=plot_path,
        report_csv=report_csv,
        stats_table_csv_path=stats_table_csv_path,
        stats_image_path=stats_image_path,
        trend_prompt=trend_prompt,
        context_response=context_response,
        trend_full=trend_full,
        trend_summary=trend_summary,
        selected_text_source=selected_text_source,
        allow_summary_fallback=allow_summary_fallback,
        full_scores=full_scores,
        summary_scores=summary_scores,
        include_extended=include_extended,
        scoring_reference_text=scoring_reference_text,
        scoring_reference_source=scoring_reference_source,
        scoring_reference_path=scoring_reference_path,
        additional_scoring_references=additional_scoring_references,
        additional_reference_scores=additional_reference_scores,
        include_pattern=include_pattern,
        evidence_mode=evidence_mode,
        requested_evidence_mode=requested_evidence_mode,
        adapter=adapter,
        selected_scores=selected_scores,
        trend_image_attached=trend_image_attached,
        run_signature=run_signature,
        context_trace=context_trace,
        trend_trace=trend_trace,
        summarization_trace=summarization_trace,
        frame_summary=frame_summary,
    )


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _extract_resumable_artifacts(artifacts_payload: dict[str, object]) -> tuple[Path, Path, str | None, Path | None]:
    plot_path = Path(str(artifacts_payload["plot_path"]))
    report_csv = Path(str(artifacts_payload["report_csv"]))
    if not plot_path.exists() or not report_csv.exists():
        raise ValueError("resumable artifacts are incomplete")

    stats_table_csv: str | None = None
    stats_table_csv_raw = artifacts_payload.get("stats_table_csv_path")
    if isinstance(stats_table_csv_raw, str) and stats_table_csv_raw:
        stats_table_csv_path = Path(stats_table_csv_raw)
        if stats_table_csv_path.exists():
            stats_table_csv = stats_table_csv_path.read_text(encoding="utf-8")

    stats_image_raw = artifacts_payload.get("stats_image_path")
    stats_image_path = Path(str(stats_image_raw)) if isinstance(stats_image_raw, str) and stats_image_raw else None
    if stats_image_path is not None and not stats_image_path.exists():
        stats_image_path = None

    return plot_path, report_csv, stats_table_csv, stats_image_path


def _extract_resumable_scores(
    scores_payload: dict[str, object],
) -> tuple[SummaryScores, SummaryScores | None, SummaryScores | None]:
    selected_scores = SummaryScores.model_validate(scores_payload["selected_scores"])
    full_scores_raw = scores_payload.get("full_scores")
    summary_scores_raw = scores_payload.get("summary_scores")
    full_scores = SummaryScores.model_validate(full_scores_raw) if full_scores_raw else None
    summary_scores = SummaryScores.model_validate(summary_scores_raw) if summary_scores_raw else None
    return selected_scores, full_scores, summary_scores


def _build_metadata_payload(
    output_dir: Path,
    inputs: PipelineInputs,
    context_prompt: str,
    plot_path: Path,
    report_csv: Path,
    stats_table_csv_path: Path,
    stats_image_path: Path | None,
    trend_prompt: str,
    context_response: str,
    trend_full: str,
    trend_summary: str | None,
    selected_text_source: str,
    allow_summary_fallback: bool,
    full_scores: SummaryScores | None,
    summary_scores: SummaryScores | None,
    include_extended: bool,
    scoring_reference_text: str,
    scoring_reference_source: str,
    scoring_reference_path: Path | None,
    additional_scoring_references: Mapping[str, tuple[str, str, Path]] | None,
    additional_reference_scores: Mapping[str, Mapping[str, SummaryScores | None]] | None,
    include_pattern: str,
    evidence_mode: str,
    requested_evidence_mode: str,
    adapter: LLMAdapter,
    selected_scores: SummaryScores,
    trend_image_attached: bool,
    run_signature: str,
    debug_trace: dict[str, object],
) -> dict[str, object]:
    runtime_defaults = get_runtime_defaults()
    default_temperature = resolve_request_temperature(adapter.provider)
    default_max_tokens = runtime_defaults.llm_request.max_tokens
    default_max_retries = runtime_defaults.llm_request.max_retries
    default_retry_backoff_seconds = runtime_defaults.llm_request.retry_backoff_seconds

    context_usage = _extract_trace_usage(debug_trace, "context_request_path")
    trend_usage = _extract_trace_usage(debug_trace, "trend_request_path")
    context_runtime = _extract_trace_runtime(debug_trace, "context_request_path")
    trend_runtime = _extract_trace_runtime(debug_trace, "trend_request_path")
    runtime_providers = _collect_runtime_values(context_runtime, trend_runtime, field="provider")
    runtime_precisions = _collect_runtime_values(context_runtime, trend_runtime, field="precision")
    observability = _build_observability_summary(context_usage=context_usage, trend_usage=trend_usage)
    additional_reference_metadata = _build_additional_reference_metadata(
        additional_scoring_references=additional_scoring_references,
        additional_reference_scores=additional_reference_scores,
    )
    return {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "csv_path": str(inputs.csv_path),
            "parameters_path": str(inputs.parameters_path),
            "documentation_path": str(inputs.documentation_path),
            "metric_pattern": inputs.metric_pattern,
            "metric_description": inputs.metric_description,
            "plot_description": inputs.plot_description,
            "evidence_mode": evidence_mode,
            "evidence_mode_requested": requested_evidence_mode,
            "text_source_mode": inputs.text_source_mode,
            "selected_text_source": selected_text_source,
            "summarizers": list(inputs.summarizers),
            "enabled_style_features": list(inputs.enabled_style_features) if inputs.enabled_style_features else None,
            "output_dir": str(inputs.output_dir),
            "scoring_reference_path": str(scoring_reference_path) if scoring_reference_path is not None else None,
            "additional_scoring_reference_paths": {
                reference_name: str(reference_path)
                for reference_name, (_text, _source, reference_path) in (additional_scoring_references or {}).items()
            },
            "resume_existing": inputs.resume_existing,
            "allow_summary_fallback": allow_summary_fallback,
        },
        "artifacts": {
            "plot_path": str(plot_path),
            "report_csv": str(report_csv),
            "stats_table_csv_path": str(stats_table_csv_path),
            "stats_image_path": str(stats_image_path) if stats_image_path is not None else None,
            "trend_evidence_image_path": (
                str(plot_path)
                if evidence_mode in {"plot", "plot+table"}
                else (str(stats_image_path) if stats_image_path else None)
            ),
        },
        "llm": {
            "provider": adapter.provider,
            "model": inputs.model,
            "precision": runtime_precisions[0] if len(runtime_precisions) == 1 else None,
            "request": {
                "temperature": default_temperature,
                "max_tokens": default_max_tokens,
                "max_retries": default_max_retries,
                "retry_backoff_seconds": default_retry_backoff_seconds,
            },
            "requests": {
                "context": {
                    "image_attached": False,
                    "temperature": default_temperature,
                    "max_tokens": default_max_tokens,
                    "max_retries": default_max_retries,
                    "retry_backoff_seconds": default_retry_backoff_seconds,
                },
                "trend": {
                    "image_attached": trend_image_attached,
                    "temperature": default_temperature,
                    "max_tokens": default_max_tokens,
                    "max_retries": default_max_retries,
                    "retry_backoff_seconds": default_retry_backoff_seconds,
                },
            },
            "usage": {
                "context": context_usage,
                "trend": trend_usage,
                "total": _combine_usage(context_usage, trend_usage),
            },
            "runtime": {
                "context": context_runtime,
                "trend": trend_runtime,
                "providers_used": runtime_providers,
                "precisions_used": runtime_precisions,
            },
            "observability": observability,
        },
        "prompts": {
            "context_prompt": context_prompt,
            "trend_prompt": trend_prompt,
        },
        "responses": {
            "context_response": context_response,
            "trend_full_response": trend_full,
            "trend_summary_response": trend_summary,
        },
        "reproducibility": {
            "include_pattern": include_pattern,
            "context_prompt_signature": hashlib.sha256(context_prompt.encode("utf-8")).hexdigest(),
            "trend_prompt_signature": hashlib.sha256(trend_prompt.encode("utf-8")).hexdigest(),
            "context_prompt_length": len(context_prompt),
            "trend_prompt_length": len(trend_prompt),
            "trend_full_length": len(trend_full),
            "trend_summary_present": trend_summary is not None,
            "trend_summary_length": len(trend_summary) if trend_summary is not None else None,
            "include_extended_columns": include_extended,
            "csv_encoding": "utf-8",
            "delimiter": ",",
            "run_signature": run_signature,
        },
        "summarizers": {
            "bart": {
                "model": "sshleifer/distilbart-cnn-12-6",
                "max_input_length": 1024,
                "min_summary_length": 50,
                "max_summary_length": 100,
                "enabled": "bart" in inputs.summarizers,
            },
            "bert": {
                "max_input_length": 512,
                "min_summary_length": 100,
                "max_summary_length": 150,
                "enabled": "bert" in inputs.summarizers,
            },
            "t5": {
                "model": "t5-small",
                "max_input_length": 1024,
                "min_summary_length": 40,
                "max_summary_length": 120,
                "enabled": "t5" in inputs.summarizers,
            },
            "longformer_ext": {
                "model": "allenai/led-base-16384",
                "max_input_length": 2048,
                "min_summary_length": 64,
                "max_summary_length": 180,
                "enabled": "longformer_ext" in inputs.summarizers,
            },
        },
        "scores": {
            "selected_scores": selected_scores.model_dump(),
            "full_scores": full_scores.model_dump() if full_scores is not None else None,
            "summary_scores": summary_scores.model_dump() if summary_scores is not None else None,
            "reference": {
                "source": scoring_reference_source,
                "path": str(scoring_reference_path) if scoring_reference_path is not None else None,
                "text": scoring_reference_text,
                "length": len(scoring_reference_text),
                "signature": hashlib.sha256(scoring_reference_text.encode("utf-8")).hexdigest(),
            },
            "additional_references": additional_reference_metadata,
        },
        "outputs": {
            "summary_csv_path": str(
                (output_dir / "combinations_report.csv") if (output_dir / "combinations_report.csv").exists() else ""
            ),
            "stats_table_csv_written": stats_table_csv_path.exists(),
        },
        "debug_trace": debug_trace,
    }


def _extract_trace_usage(debug_trace: dict[str, object], key: str) -> dict[str, int] | None:
    response = _read_trace_response(debug_trace, key)
    if not response:
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    if not all(isinstance(value, int) for value in (prompt, completion, total)):
        return None
    prompt_tokens = cast(int, prompt)
    completion_tokens = cast(int, completion)
    total_tokens = cast(int, total)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _extract_trace_runtime(debug_trace: dict[str, object], key: str) -> dict[str, str] | None:
    response = _read_trace_response(debug_trace, key)
    if not response:
        return None
    runtime = _as_dict(response.get("runtime"))
    provider = runtime.get("provider")
    precision = runtime.get("precision")
    payload: dict[str, str] = {}
    if isinstance(provider, str) and provider:
        payload["provider"] = provider
    if isinstance(precision, str) and precision:
        payload["precision"] = precision
    return payload or None


def _read_trace_response(debug_trace: dict[str, object], key: str) -> dict[str, object]:
    trace_path_value = debug_trace.get(key)
    if not isinstance(trace_path_value, str):
        return {}
    trace_path = Path(trace_path_value)
    if not trace_path.exists():
        return {}
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return _as_dict(payload.get("response"))


def _combine_usage(*usages: dict[str, int] | None) -> dict[str, int] | None:
    available = [usage for usage in usages if usage is not None]
    if not available:
        return None
    return {
        "prompt_tokens": sum(usage["prompt_tokens"] for usage in available),
        "completion_tokens": sum(usage["completion_tokens"] for usage in available),
        "total_tokens": sum(usage["total_tokens"] for usage in available),
    }


def _collect_runtime_values(
    *runtime_payloads: dict[str, str] | None,
    field: str,
) -> list[str]:
    values: list[str] = []
    for payload in runtime_payloads:
        if payload is None:
            continue
        value = payload.get(field)
        if value and value not in values:
            values.append(value)
    return values


def _build_observability_summary(
    *,
    context_usage: dict[str, int] | None,
    trend_usage: dict[str, int] | None,
) -> dict[str, object]:
    total_usage = _combine_usage(context_usage, trend_usage)
    return {
        "request_count": 2,
        "usage": {
            "context": context_usage,
            "trend": trend_usage,
            "total": total_usage,
        },
        "cost": {
            "status": "unpriced",
            "estimated_total_usd": None,
            "currency": "USD",
        },
    }


def _write_debug_trace_bundle(
    *,
    output_dir: Path,
    inputs: PipelineInputs,
    plot_path: Path,
    report_csv: Path,
    stats_table_csv_path: Path,
    stats_image_path: Path | None,
    scoring_reference_path: Path | None,
    additional_scoring_reference_paths: Mapping[str, Path] | None,
    context_trace: dict[str, object],
    trend_trace: dict[str, object],
    summarization_trace: dict[str, object],
    frame_summary: dict[str, object],
) -> dict[str, object]:
    debug_dir = output_dir / "debug_trace"
    inputs_dir = debug_dir / "inputs"
    llm_dir = debug_dir / "llm"
    summarization_dir = debug_dir / "summarization"
    manifests_dir = debug_dir / "manifests"
    for path in (inputs_dir, llm_dir, summarization_dir, manifests_dir):
        path.mkdir(parents=True, exist_ok=True)

    csv_snapshot = _snapshot_file(inputs.csv_path, inputs_dir / "simulation.csv")
    parameters_snapshot = _snapshot_file(inputs.parameters_path, inputs_dir / "parameters.txt")
    documentation_snapshot = _snapshot_file(inputs.documentation_path, inputs_dir / "documentation.txt")
    scoring_snapshot = (
        _snapshot_file(scoring_reference_path, inputs_dir / "scoring_reference.txt")
        if scoring_reference_path is not None
        else None
    )
    additional_scoring_snapshots = {
        reference_name: _snapshot_file(
            reference_path,
            inputs_dir / "additional_scoring_references" / f"{reference_name}.txt",
        )
        for reference_name, reference_path in (additional_scoring_reference_paths or {}).items()
    }

    context_request_path = llm_dir / "context_request.json"
    context_request_path.write_text(json.dumps(context_trace, indent=2, sort_keys=True), encoding="utf-8")
    trend_request_path = llm_dir / "trend_request.json"
    trend_request_path.write_text(json.dumps(trend_trace, indent=2, sort_keys=True), encoding="utf-8")
    summarization_trace_path = summarization_dir / "summarization_trace.json"
    summarization_trace_path.write_text(json.dumps(summarization_trace, indent=2, sort_keys=True), encoding="utf-8")

    input_validations = {
        "csv": _build_file_debug_record(inputs.csv_path),
        "parameters": _build_file_debug_record(inputs.parameters_path),
        "documentation": _build_file_debug_record(inputs.documentation_path),
        "scoring_reference": _build_file_debug_record(scoring_reference_path) if scoring_reference_path else None,
        "additional_scoring_references": {
            reference_name: _build_file_debug_record(reference_path)
            for reference_name, reference_path in (additional_scoring_reference_paths or {}).items()
        },
    }
    csv_validation = cast(dict[str, object] | None, input_validations["csv"])
    parameters_validation = cast(dict[str, object] | None, input_validations["parameters"])
    documentation_validation = cast(dict[str, object] | None, input_validations["documentation"])
    scoring_validation = cast(dict[str, object] | None, input_validations["scoring_reference"])
    additional_reference_validations = cast(
        dict[str, dict[str, object]],
        input_validations["additional_scoring_references"],
    )
    warning_input_validations: dict[str, dict[str, object] | None] = {
        "csv": csv_validation,
        "parameters": parameters_validation,
        "documentation": documentation_validation,
        "scoring_reference": scoring_validation,
    }
    for reference_name, record in additional_reference_validations.items():
        warning_input_validations[f"additional_scoring_reference:{reference_name}"] = record
    warnings = _collect_debug_warnings(input_validations=warning_input_validations, frame_summary=frame_summary)

    artifact_manifest = {
        "plot_path": _build_file_debug_record(plot_path),
        "report_csv": _build_file_debug_record(report_csv),
        "stats_table_csv": _build_file_debug_record(stats_table_csv_path),
        "stats_image_path": _build_file_debug_record(stats_image_path) if stats_image_path is not None else None,
    }
    artifact_manifest_path = manifests_dir / "artifact_manifest.json"
    artifact_manifest_path.write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True), encoding="utf-8")

    validations_path = manifests_dir / "input_validations.json"
    validations_path.write_text(
        json.dumps(
            {
                "input_validations": input_validations,
                "dataframe": frame_summary,
                "warnings": warnings,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "input_snapshots": {
            "csv_path": str(csv_snapshot),
            "parameters_path": str(parameters_snapshot),
            "documentation_path": str(documentation_snapshot),
            "scoring_reference_path": str(scoring_snapshot) if scoring_snapshot is not None else None,
            "additional_scoring_reference_paths": {
                reference_name: str(snapshot_path)
                for reference_name, snapshot_path in additional_scoring_snapshots.items()
            },
        },
        "context_request_path": str(context_request_path),
        "trend_request_path": str(trend_request_path),
        "summarization_trace_path": str(summarization_trace_path),
        "artifact_manifest_path": str(artifact_manifest_path),
        "input_validations_path": str(validations_path),
        "dataframe": frame_summary,
        "warnings": warnings,
    }


def _snapshot_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy2(source, destination)
    return destination


def _build_additional_reference_metadata(
    *,
    additional_scoring_references: Mapping[str, tuple[str, str, Path]] | None,
    additional_reference_scores: Mapping[str, Mapping[str, SummaryScores | None]] | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for reference_name, reference_scores in (additional_reference_scores or {}).items():
        if additional_scoring_references is None or reference_name not in additional_scoring_references:
            continue
        reference_text, reference_source, reference_path = additional_scoring_references[reference_name]
        selected_scores = reference_scores.get("selected_scores")
        full_scores = reference_scores.get("full_scores")
        summary_scores = reference_scores.get("summary_scores")
        metadata[reference_name] = {
            "reference": {
                "source": reference_source,
                "path": str(reference_path),
                "text": reference_text,
                "length": len(reference_text),
                "signature": hashlib.sha256(reference_text.encode("utf-8")).hexdigest(),
            },
            "selected_scores": selected_scores.model_dump() if selected_scores is not None else None,
            "full_scores": full_scores.model_dump() if full_scores is not None else None,
            "summary_scores": summary_scores.model_dump() if summary_scores is not None else None,
        }
    return metadata


def _build_file_debug_record(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"exists": False}
    exists = path.exists()
    record: dict[str, object] = {
        "path": str(path),
        "exists": exists,
    }
    if not exists:
        return record
    text = path.read_text(encoding="utf-8", errors="replace")
    preview = text[:200]
    placeholder_signals = detect_placeholder_signals(text)
    record.update(
        {
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "line_count": len(text.splitlines()),
            "preview": preview,
            "placeholder_signals": placeholder_signals,
        }
    )
    return record


def _collect_debug_warnings(
    *,
    input_validations: Mapping[str, dict[str, object] | None],
    frame_summary: Mapping[str, object],
) -> list[str]:
    warnings: list[str] = []
    for label, payload in input_validations.items():
        if not isinstance(payload, dict):
            continue
        if not payload.get("exists", False):
            warnings.append(f"{label} input is missing from the debug trace")
            continue
        size_bytes = payload.get("size_bytes", 0)
        if isinstance(size_bytes, int) and size_bytes == 0:
            warnings.append(f"{label} input is empty")
        placeholder_signals = payload.get("placeholder_signals", [])
        if isinstance(placeholder_signals, list):
            for signal in placeholder_signals:
                warnings.append(f"{label} input contains placeholder-like token '{signal}'")
    matched_columns = frame_summary.get("matched_metric_columns")
    if isinstance(matched_columns, list) and not matched_columns:
        warnings.append("simulation CSV did not contain any columns matching the configured metric pattern")
    return warnings
