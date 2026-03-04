"""Persistence and reproducibility helpers for benchmark pipeline runs."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from distill_abm.configs.runtime_defaults import get_runtime_defaults
from distill_abm.eval.metrics import SummaryScores

if TYPE_CHECKING:
    from distill_abm.configs.models import PromptsConfig
    from distill_abm.llm.adapters.base import LLMAdapter
    from distill_abm.pipeline.run import PipelineInputs, PipelineResult


def build_run_signature(inputs: PipelineInputs, prompts: PromptsConfig, adapter: LLMAdapter) -> str:
    """Compute a deterministic fingerprint for a pipeline run configuration."""
    runtime_defaults = get_runtime_defaults()
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
        },
        "input_file_hashes": {
            "csv": _hash_file(inputs.csv_path),
            "parameters": _hash_file(inputs.parameters_path),
            "documentation": _hash_file(inputs.documentation_path),
            "scoring_reference": _hash_file(inputs.scoring_reference_path),
        },
        "llm": {
            "provider": adapter.provider,
            "model": inputs.model,
            "temperature": runtime_defaults.llm_request.temperature,
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
    include_pattern: str,
    evidence_mode: str,
    requested_evidence_mode: str,
    adapter: LLMAdapter,
    selected_scores: SummaryScores,
    trend_image_attached: bool,
    run_signature: str,
) -> Path:
    """Persist deterministic metadata for a completed pipeline run."""
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
        include_pattern=include_pattern,
        evidence_mode=evidence_mode,
        requested_evidence_mode=requested_evidence_mode,
        adapter=adapter,
        selected_scores=selected_scores,
        trend_image_attached=trend_image_attached,
        run_signature=run_signature,
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
    include_pattern: str,
    evidence_mode: str,
    requested_evidence_mode: str,
    adapter: LLMAdapter,
    selected_scores: SummaryScores,
    trend_image_attached: bool,
    run_signature: str,
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
        include_pattern=include_pattern,
        evidence_mode=evidence_mode,
        requested_evidence_mode=requested_evidence_mode,
        adapter=adapter,
        selected_scores=selected_scores,
        trend_image_attached=trend_image_attached,
        run_signature=run_signature,
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
    include_pattern: str,
    evidence_mode: str,
    requested_evidence_mode: str,
    adapter: LLMAdapter,
    selected_scores: SummaryScores,
    trend_image_attached: bool,
    run_signature: str,
) -> dict[str, object]:
    runtime_defaults = get_runtime_defaults()
    default_temperature = runtime_defaults.llm_request.temperature
    default_max_tokens = runtime_defaults.llm_request.max_tokens
    default_max_retries = runtime_defaults.llm_request.max_retries
    default_retry_backoff_seconds = runtime_defaults.llm_request.retry_backoff_seconds

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
                "length": len(scoring_reference_text),
                "signature": hashlib.sha256(scoring_reference_text.encode("utf-8")).hexdigest(),
            },
        },
        "outputs": {
            "summary_csv_path": str(
                (output_dir / "combinations_report.csv")
                if (output_dir / "combinations_report.csv").exists()
                else ""
            ),
            "stats_table_csv_written": stats_table_csv_path.exists(),
        },
    }
