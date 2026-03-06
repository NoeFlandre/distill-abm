"""Pre-LLM DOE smoke reporting for full experiment-matrix inspection."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field

from distill_abm.configs.models import PromptsConfig, SummarizerId
from distill_abm.configs.runtime_defaults import get_runtime_defaults
from distill_abm.ingest.csv_ingest import load_simulation_csv
from distill_abm.pipeline import helpers
from distill_abm.utils import detect_placeholder_signals

DoESmokeStatus = Literal["ok", "failed"]
DoESmokeErrorCode = Literal[
    "missing_or_empty_artifact",
    "placeholder_detected",
    "unmatched_metric_pattern",
    "plot_count_mismatch",
    "model_preflight_failed",
]
DoETextSourceMode = Literal["summary_only", "full_text_only"]

CONTEXT_PLACEHOLDER = "<<context_response_from_context_llm>>"
CANONICAL_DOE_MODEL_IDS: tuple[str, ...] = ("qwen3_5_local", "kimi_k2_5", "gemini_3_1_pro_preview")
CANONICAL_EVIDENCE_MODES: tuple[Literal["plot", "table", "plot+table"], ...] = ("plot", "table", "plot+table")
CANONICAL_REPETITIONS: tuple[int, ...] = (1, 2, 3)


class DoESmokePlotInput(BaseModel):
    """One plot-level evidence item used by the real DOE."""

    plot_index: int
    reporter_pattern: str
    plot_description: str
    plot_path: Path


class DoESmokeAbmInput(BaseModel):
    """Resolved ABM-level pre-LLM inputs."""

    abm: str
    csv_path: Path
    parameters_path: Path
    documentation_path: Path
    metric_pattern: str
    metric_description: str
    plots: list[DoESmokePlotInput]
    source_viz_artifact_source: Literal["simulated", "fallback", "unknown"] = "unknown"


class DoESmokeModelSpec(BaseModel):
    """One candidate LLM participating in the DOE."""

    model_id: str
    provider: str
    model: str
    preflight_error: str | None = None


class DoESmokePromptVariant(BaseModel):
    """One prompt-style combination in the DOE."""

    variant_id: str
    enabled_style_features: tuple[str, ...] = ()


class DoESmokeSummarizationSpec(BaseModel):
    """One summarization condition in the DOE."""

    summarization_mode: str
    text_source_mode: DoETextSourceMode
    summarizers: tuple[SummarizerId, ...] = ()


class DoESmokeArtifact(BaseModel):
    """Stable artifact metadata for reports."""

    path: Path
    exists: bool
    size_bytes: int = 0
    sha256: str | None = None
    preview: str = ""


class DoESmokePlotPlan(BaseModel):
    """One planned trend request for one plot."""

    plot_index: int
    reporter_pattern: str
    plot_description: str
    prompt_path: Path
    plot_path: Path
    stats_table_csv_path: Path | None
    evidence_mode: str
    status: DoESmokeStatus
    error_code: DoESmokeErrorCode | None = None
    error: str | None = None


class DoESmokeSharedAbmResult(BaseModel):
    """Shared ABM bundle reused by many DOE cases."""

    abm: str
    shared_dir: Path
    csv_path: Path
    parameters_path: Path
    documentation_path: Path
    source_viz_artifact_source: str
    plot_count: int
    plot_request_count: int
    artifact_index_path: Path
    stage_errors: list[str] = Field(default_factory=list)


class DoESmokeCaseResult(BaseModel):
    """One exact DOE combination for one ABM."""

    case_id: str
    abm: str
    model_id: str
    provider: str
    model: str
    evidence_mode: str
    summarization_mode: str
    text_source_mode: str
    prompt_variant: str
    enabled_style_features: list[str] = Field(default_factory=list)
    summarizers: list[str] = Field(default_factory=list)
    repetition: int
    request_count: int
    failed_plot_indices: list[int] = Field(default_factory=list)
    status: DoESmokeStatus
    case_dir: Path
    case_manifest_path: Path
    context_prompt_path: Path
    context_request_plan_path: Path


class DoESmokeSuiteResult(BaseModel):
    """Top-level DOE smoke report."""

    started_at_utc: str
    finished_at_utc: str
    output_root: Path
    success: bool
    total_cases: int
    total_planned_requests: int
    total_context_requests: int
    total_trend_requests: int
    failed_case_ids: list[str] = Field(default_factory=list)
    design_matrix_csv_path: Path
    request_matrix_csv_path: Path
    report_markdown_path: Path
    report_json_path: Path
    abm_shared: dict[str, DoESmokeSharedAbmResult] = Field(default_factory=dict)
    cases: list[DoESmokeCaseResult] = Field(default_factory=list)


def canonical_prompt_variants() -> tuple[DoESmokePromptVariant, ...]:
    """Return the paper DOE prompt combinations."""
    return (
        DoESmokePromptVariant(variant_id="none", enabled_style_features=()),
        DoESmokePromptVariant(variant_id="role", enabled_style_features=("role",)),
        DoESmokePromptVariant(variant_id="insights", enabled_style_features=("insights",)),
        DoESmokePromptVariant(variant_id="example", enabled_style_features=("example",)),
        DoESmokePromptVariant(variant_id="role+example", enabled_style_features=("role", "example")),
        DoESmokePromptVariant(variant_id="role+insights", enabled_style_features=("role", "insights")),
        DoESmokePromptVariant(variant_id="insights+example", enabled_style_features=("insights", "example")),
        DoESmokePromptVariant(variant_id="all_three", enabled_style_features=("role", "insights", "example")),
    )


def canonical_summarization_specs() -> tuple[DoESmokeSummarizationSpec, ...]:
    """Return the paper DOE summarization conditions."""
    return (
        DoESmokeSummarizationSpec(
            summarization_mode="none",
            text_source_mode="full_text_only",
            summarizers=(),
        ),
        DoESmokeSummarizationSpec(
            summarization_mode="bart",
            text_source_mode="summary_only",
            summarizers=("bart",),
        ),
        DoESmokeSummarizationSpec(
            summarization_mode="bert",
            text_source_mode="summary_only",
            summarizers=("bert",),
        ),
        DoESmokeSummarizationSpec(
            summarization_mode="t5",
            text_source_mode="summary_only",
            summarizers=("t5",),
        ),
        DoESmokeSummarizationSpec(
            summarization_mode="longformer_ext",
            text_source_mode="summary_only",
            summarizers=("longformer_ext",),
        ),
    )


def run_doe_smoke_suite(
    *,
    abm_inputs: dict[str, DoESmokeAbmInput],
    prompts: PromptsConfig,
    model_specs: list[DoESmokeModelSpec],
    output_root: Path,
    evidence_modes: tuple[Literal["plot", "table", "plot+table"], ...] = CANONICAL_EVIDENCE_MODES,
    summarization_specs: tuple[DoESmokeSummarizationSpec, ...] | None = None,
    prompt_variants: tuple[DoESmokePromptVariant, ...] | None = None,
    repetitions: tuple[int, ...] = CANONICAL_REPETITIONS,
) -> DoESmokeSuiteResult:
    """Materialize the full pre-LLM DOE matrix without executing any model call."""
    started_at = datetime.now(UTC)
    output_root.mkdir(parents=True, exist_ok=True)
    summarization_specs = summarization_specs or canonical_summarization_specs()
    prompt_variants = prompt_variants or canonical_prompt_variants()
    defaults = get_runtime_defaults().llm_request

    shared_by_abm: dict[str, DoESmokeSharedAbmResult] = {}
    shared_context_paths: dict[tuple[str, str], Path] = {}
    shared_plot_prompt_paths: dict[tuple[str, str, str, int], Path] = {}
    failed_case_ids: list[str] = []
    case_results: list[DoESmokeCaseResult] = []
    design_rows: list[dict[str, str]] = []
    request_rows: list[dict[str, str]] = []

    for abm in sorted(abm_inputs):
        shared_by_abm[abm] = _materialize_shared_abm_bundle(
            abm_input=abm_inputs[abm],
            prompts=prompts,
            prompt_variants=prompt_variants,
            evidence_modes=evidence_modes,
            output_root=output_root,
            shared_context_paths=shared_context_paths,
            shared_plot_prompt_paths=shared_plot_prompt_paths,
        )

    for abm in sorted(abm_inputs):
        abm_input = abm_inputs[abm]
        shared = shared_by_abm[abm]
        for model_spec in model_specs:
            for evidence_mode in evidence_modes:
                for summarization in summarization_specs:
                    for prompt_variant in prompt_variants:
                        for repetition in repetitions:
                            case_id = (
                                f"{abm}::{model_spec.model_id}::{evidence_mode}::{summarization.summarization_mode}"
                                f"::{prompt_variant.variant_id}::rep{repetition}"
                            )
                            case_dir = output_root / "cases" / abm / model_spec.model_id / evidence_mode / (
                                f"{summarization.summarization_mode}__{prompt_variant.variant_id}__rep{repetition}"
                            )
                            case_dir.mkdir(parents=True, exist_ok=True)
                            model_preflight_error = model_spec.preflight_error
                            context_prompt_path = shared_context_paths[(abm, prompt_variant.variant_id)]
                            context_prompt_text = context_prompt_path.read_text(encoding="utf-8")
                            context_request_plan_path = case_dir / "context_request_plan.json"
                            context_request = _request_plan(
                                provider=model_spec.provider,
                                model=model_spec.model,
                                model_id=model_spec.model_id,
                                prompt_path=context_prompt_path,
                                prompt_text=context_prompt_text,
                                image_path=None,
                                stats_table_csv_path=None,
                                evidence_mode=evidence_mode,
                                summarization_mode=summarization.summarization_mode,
                                text_source_mode=summarization.text_source_mode,
                                prompt_variant=prompt_variant.variant_id,
                                enabled_style_features=prompt_variant.enabled_style_features,
                                summarizers=summarization.summarizers,
                                repetition=repetition,
                                request_kind="context",
                                plot_index=None,
                                temperature=defaults.temperature,
                                max_tokens=defaults.max_tokens,
                                max_retries=defaults.max_retries,
                                retry_backoff_seconds=defaults.retry_backoff_seconds,
                                unresolved_context=False,
                            )
                            if model_preflight_error:
                                context_request["status"] = "failed"
                                context_request["error_code"] = "model_preflight_failed"
                                context_request["error"] = model_preflight_error
                            context_request_plan_path.write_text(
                                json.dumps(context_request, indent=2, sort_keys=True),
                                encoding="utf-8",
                            )
                            request_rows.append(_request_row(case_id=case_id, abm=abm, request_plan=context_request))

                            trend_requests: list[dict[str, object]] = []
                            failed_plot_indices: list[int] = []
                            for plot_input in abm_input.plots:
                                prompt_path = shared_plot_prompt_paths[
                                    (abm, prompt_variant.variant_id, evidence_mode, plot_input.plot_index)
                                ]
                                prompt_text = prompt_path.read_text(encoding="utf-8")
                                stats_table_csv_path = _shared_stats_table_path(
                                    output_root=output_root,
                                    abm=abm,
                                    plot_index=plot_input.plot_index,
                                )
                                plot_status, plot_error_code, plot_error = _validate_plot_request(
                                    plot_input=plot_input,
                                    stats_table_csv_path=(
                                        stats_table_csv_path if evidence_mode in {"table", "plot+table"} else None
                                    ),
                                    evidence_mode=evidence_mode,
                                    shared_stage_errors=shared.stage_errors,
                                )
                                if model_preflight_error:
                                    plot_status = "failed"
                                    plot_error_code = "model_preflight_failed"
                                    plot_error = model_preflight_error
                                if plot_status == "failed":
                                    failed_plot_indices.append(plot_input.plot_index)
                                trend_request = _request_plan(
                                    provider=model_spec.provider,
                                    model=model_spec.model,
                                    model_id=model_spec.model_id,
                                    prompt_path=prompt_path,
                                    prompt_text=prompt_text,
                                    image_path=(
                                        plot_input.plot_path if evidence_mode in {"plot", "plot+table"} else None
                                    ),
                                    stats_table_csv_path=(
                                        stats_table_csv_path if evidence_mode in {"table", "plot+table"} else None
                                    ),
                                    evidence_mode=evidence_mode,
                                    summarization_mode=summarization.summarization_mode,
                                    text_source_mode=summarization.text_source_mode,
                                    prompt_variant=prompt_variant.variant_id,
                                    enabled_style_features=prompt_variant.enabled_style_features,
                                    summarizers=summarization.summarizers,
                                    repetition=repetition,
                                    request_kind="trend",
                                    plot_index=plot_input.plot_index,
                                    temperature=defaults.temperature,
                                    max_tokens=defaults.max_tokens,
                                    max_retries=defaults.max_retries,
                                    retry_backoff_seconds=defaults.retry_backoff_seconds,
                                    unresolved_context=True,
                                )
                                trend_request["reporter_pattern"] = plot_input.reporter_pattern
                                trend_request["plot_description"] = plot_input.plot_description
                                trend_request["status"] = plot_status
                                trend_request["error_code"] = plot_error_code
                                trend_request["error"] = plot_error
                                trend_requests.append(trend_request)
                                request_rows.append(_request_row(case_id=case_id, abm=abm, request_plan=trend_request))

                            case_manifest = {
                                "case_id": case_id,
                                "abm": abm,
                                "model_id": model_spec.model_id,
                                "provider": model_spec.provider,
                                "model": model_spec.model,
                                "evidence_mode": evidence_mode,
                                "summarization_mode": summarization.summarization_mode,
                                "text_source_mode": summarization.text_source_mode,
                                "prompt_variant": prompt_variant.variant_id,
                                "enabled_style_features": list(prompt_variant.enabled_style_features),
                                "summarizers": list(summarization.summarizers),
                                "repetition": repetition,
                                "source_viz_artifact_source": abm_input.source_viz_artifact_source,
                                "input_csv_path": str(abm_input.csv_path),
                                "parameters_path": str(abm_input.parameters_path),
                                "documentation_path": str(abm_input.documentation_path),
                                "context_request": context_request,
                                "trend_requests": trend_requests,
                                "shared_stage_errors": list(shared.stage_errors),
                                "model_preflight_error": model_preflight_error,
                            }
                            case_manifest_path = case_dir / "case_manifest.json"
                            case_manifest_path.write_text(
                                json.dumps(case_manifest, indent=2, sort_keys=True),
                                encoding="utf-8",
                            )
                            case_failed = bool(shared.stage_errors or failed_plot_indices or model_preflight_error)
                            case_status: DoESmokeStatus = "failed" if case_failed else "ok"
                            if case_status == "failed":
                                failed_case_ids.append(case_id)
                            case_results.append(
                                DoESmokeCaseResult(
                                    case_id=case_id,
                                    abm=abm,
                                    model_id=model_spec.model_id,
                                    provider=model_spec.provider,
                                    model=model_spec.model,
                                    evidence_mode=evidence_mode,
                                    summarization_mode=summarization.summarization_mode,
                                    text_source_mode=summarization.text_source_mode,
                                    prompt_variant=prompt_variant.variant_id,
                                    enabled_style_features=list(prompt_variant.enabled_style_features),
                                    summarizers=list(summarization.summarizers),
                                    repetition=repetition,
                                    request_count=1 + len(abm_input.plots),
                                    failed_plot_indices=failed_plot_indices,
                                    status=case_status,
                                    case_dir=case_dir,
                                    case_manifest_path=case_manifest_path,
                                    context_prompt_path=context_prompt_path,
                                    context_request_plan_path=context_request_plan_path,
                                )
                            )
                            design_rows.append(
                                {
                                    "case_id": case_id,
                                    "abm": abm,
                                    "model_id": model_spec.model_id,
                                    "provider": model_spec.provider,
                                    "model": model_spec.model,
                                    "evidence_mode": evidence_mode,
                                    "summarization_mode": summarization.summarization_mode,
                                    "text_source_mode": summarization.text_source_mode,
                                    "prompt_variant": prompt_variant.variant_id,
                                    "enabled_style_features": "|".join(prompt_variant.enabled_style_features),
                                    "summarizers": "|".join(summarization.summarizers),
                                    "repetition": str(repetition),
                                    "request_count": str(1 + len(abm_input.plots)),
                                    "status": case_status,
                                    "failed_plot_indices": "|".join(str(item) for item in failed_plot_indices),
                                    "case_manifest_path": str(case_manifest_path),
                                    "context_prompt_path": str(context_prompt_path),
                                    "context_request_plan_path": str(context_request_plan_path),
                                }
                            )

    design_matrix_csv_path = output_root / "design_matrix.csv"
    request_matrix_csv_path = output_root / "request_matrix.csv"
    _write_csv(
        path=design_matrix_csv_path,
        fieldnames=list(design_rows[0].keys()) if design_rows else [
            "case_id",
            "abm",
            "model_id",
            "provider",
            "model",
            "evidence_mode",
            "summarization_mode",
            "text_source_mode",
            "prompt_variant",
            "enabled_style_features",
            "summarizers",
            "repetition",
            "request_count",
            "status",
            "failed_plot_indices",
            "case_manifest_path",
            "context_prompt_path",
            "context_request_plan_path",
        ],
        rows=design_rows,
    )
    _write_csv(
        path=request_matrix_csv_path,
        fieldnames=list(request_rows[0].keys()) if request_rows else [
            "case_id",
            "abm",
            "request_kind",
            "plot_index",
            "model_id",
            "provider",
            "model",
            "evidence_mode",
            "summarization_mode",
            "text_source_mode",
            "prompt_variant",
            "enabled_style_features",
            "summarizers",
            "repetition",
            "prompt_path",
            "prompt_signature",
            "image_path",
            "stats_table_csv_path",
            "status",
            "error_code",
            "error",
        ],
        rows=request_rows,
    )

    total_context_requests = len(case_results)
    total_trend_requests = sum(len(abm_inputs[case.abm].plots) for case in case_results)
    total_planned_requests = total_context_requests + total_trend_requests
    finished_at = datetime.now(UTC)
    report_json_path = output_root / "doe_smoke_report.json"
    report_markdown_path = output_root / "doe_smoke_report.md"
    result = DoESmokeSuiteResult(
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        output_root=output_root,
        success=not failed_case_ids,
        total_cases=len(case_results),
        total_planned_requests=total_planned_requests,
        total_context_requests=total_context_requests,
        total_trend_requests=total_trend_requests,
        failed_case_ids=failed_case_ids,
        design_matrix_csv_path=design_matrix_csv_path,
        request_matrix_csv_path=request_matrix_csv_path,
        report_markdown_path=report_markdown_path,
        report_json_path=report_json_path,
        abm_shared=shared_by_abm,
        cases=case_results,
    )
    report_json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    report_markdown_path.write_text(_render_markdown_report(result), encoding="utf-8")
    return result


def _materialize_shared_abm_bundle(
    *,
    abm_input: DoESmokeAbmInput,
    prompts: PromptsConfig,
    prompt_variants: tuple[DoESmokePromptVariant, ...],
    evidence_modes: tuple[Literal["plot", "table", "plot+table"], ...],
    output_root: Path,
    shared_context_paths: dict[tuple[str, str], Path],
    shared_plot_prompt_paths: dict[tuple[str, str, str, int], Path],
) -> DoESmokeSharedAbmResult:
    shared_dir = output_root / "shared" / abm_input.abm
    prompts_dir = shared_dir / "prompts"
    tables_dir = shared_dir / "stats_tables"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    stage_errors: list[str] = []

    _validate_text_artifact(path=abm_input.parameters_path, label="parameters", stage_errors=stage_errors)
    _validate_text_artifact(path=abm_input.documentation_path, label="documentation", stage_errors=stage_errors)
    frame = load_simulation_csv(abm_input.csv_path)
    if len(abm_input.plots) <= 0:
        stage_errors.append("no plot definitions configured")
    for plot_input in abm_input.plots:
        if not plot_input.plot_path.exists() or plot_input.plot_path.stat().st_size <= 0:
            stage_errors.append(f"plot {plot_input.plot_index} missing or empty: {plot_input.plot_path}")
        stats_table_path = _shared_stats_table_path(
            output_root=output_root,
            abm=abm_input.abm,
            plot_index=plot_input.plot_index,
        )
        try:
            stats_table = helpers.build_stats_table(frame=frame, include_pattern=plot_input.reporter_pattern)
            stats_table_csv = helpers.build_stats_csv(stats_table)
            stats_table_path.write_text(stats_table_csv, encoding="utf-8")
        except Exception:
            stats_table_path.write_text("unmatched metric pattern\n", encoding="utf-8")
        if detect_placeholder_signals(plot_input.plot_description):
            stage_errors.append(f"plot {plot_input.plot_index} description contains placeholder-like text")

    for prompt_variant in prompt_variants:
        enabled = set(prompt_variant.enabled_style_features)
        context_prompt = helpers.build_context_prompt(
            inputs_csv_path=abm_input.parameters_path,
            inputs_doc_path=abm_input.documentation_path,
            prompts=prompts,
            enabled=enabled,
        )
        context_prompt_path = prompts_dir / prompt_variant.variant_id / "context_prompt.txt"
        context_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        context_prompt_path.write_text(context_prompt, encoding="utf-8")
        shared_context_paths[(abm_input.abm, prompt_variant.variant_id)] = context_prompt_path
        for evidence_mode in evidence_modes:
            for plot_input in abm_input.plots:
                stats_table_csv_path = _shared_stats_table_path(
                    output_root=output_root,
                    abm=abm_input.abm,
                    plot_index=plot_input.plot_index,
                )
                stats_table_csv = (
                    stats_table_csv_path.read_text(encoding="utf-8")
                    if evidence_mode in {"table", "plot+table"} and stats_table_csv_path.exists()
                    else ""
                )
                trend_prompt = helpers.build_trend_prompt(
                    prompts=prompts,
                    metric_description=abm_input.metric_description,
                    context=CONTEXT_PLACEHOLDER,
                    plot_description=plot_input.plot_description,
                    evidence_mode=evidence_mode,
                    stats_table_csv=stats_table_csv,
                    enabled=enabled,
                )
                trend_prompt_path = (
                    prompts_dir / prompt_variant.variant_id / evidence_mode / f"trend_plot_{plot_input.plot_index}.txt"
                )
                trend_prompt_path.parent.mkdir(parents=True, exist_ok=True)
                trend_prompt_path.write_text(trend_prompt, encoding="utf-8")
                shared_plot_prompt_paths[
                    (abm_input.abm, prompt_variant.variant_id, evidence_mode, plot_input.plot_index)
                ] = trend_prompt_path

    artifact_index_path = shared_dir / "artifact_index.json"
    artifact_index_path.write_text(
        json.dumps(
            {
                "csv_path": str(abm_input.csv_path),
                "parameters_path": str(abm_input.parameters_path),
                "documentation_path": str(abm_input.documentation_path),
                "plot_count": len(abm_input.plots),
                "source_viz_artifact_source": abm_input.source_viz_artifact_source,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return DoESmokeSharedAbmResult(
        abm=abm_input.abm,
        shared_dir=shared_dir,
        csv_path=abm_input.csv_path,
        parameters_path=abm_input.parameters_path,
        documentation_path=abm_input.documentation_path,
        source_viz_artifact_source=abm_input.source_viz_artifact_source,
        plot_count=len(abm_input.plots),
        plot_request_count=len(abm_input.plots),
        artifact_index_path=artifact_index_path,
        stage_errors=stage_errors,
    )


def _validate_text_artifact(*, path: Path, label: str, stage_errors: list[str]) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        stage_errors.append(f"{label} missing or empty: {path}")
        return
    preview = path.read_text(encoding="utf-8", errors="replace")
    hits = detect_placeholder_signals(preview)
    if hits:
        stage_errors.append(f"{label} contains placeholder-like text: {', '.join(hits)}")


def _shared_stats_table_path(*, output_root: Path, abm: str, plot_index: int) -> Path:
    return output_root / "shared" / abm / "stats_tables" / f"plot_{plot_index}.csv"


def _validate_plot_request(
    *,
    plot_input: DoESmokePlotInput,
    stats_table_csv_path: Path | None,
    evidence_mode: str,
    shared_stage_errors: list[str],
) -> tuple[DoESmokeStatus, DoESmokeErrorCode | None, str | None]:
    if shared_stage_errors:
        return "failed", "missing_or_empty_artifact", "; ".join(shared_stage_errors)
    if not plot_input.plot_path.exists() or plot_input.plot_path.stat().st_size <= 0:
        return "failed", "missing_or_empty_artifact", f"plot missing or empty: {plot_input.plot_path}"
    if evidence_mode in {"table", "plot+table"}:
        if stats_table_csv_path is None or not stats_table_csv_path.exists():
            return "failed", "missing_or_empty_artifact", "stats table missing"
        stats_text = stats_table_csv_path.read_text(encoding="utf-8", errors="replace")
        if "unmatched metric pattern" in stats_text:
            return "failed", "unmatched_metric_pattern", "metric pattern did not match any simulation CSV columns"
        if not stats_text.strip():
            return "failed", "missing_or_empty_artifact", f"stats table empty: {stats_table_csv_path}"
    if detect_placeholder_signals(plot_input.plot_description):
        return "failed", "placeholder_detected", "plot description contains placeholder-like text"
    return "ok", None, None


def _request_plan(
    *,
    provider: str,
    model: str,
    model_id: str,
    prompt_path: Path,
    prompt_text: str,
    image_path: Path | None,
    stats_table_csv_path: Path | None,
    evidence_mode: str,
    summarization_mode: str,
    text_source_mode: str,
    prompt_variant: str,
    enabled_style_features: tuple[str, ...],
    summarizers: tuple[SummarizerId, ...],
    repetition: int,
    request_kind: Literal["context", "trend"],
    plot_index: int | None,
    temperature: float | None,
    max_tokens: int | None,
    max_retries: int,
    retry_backoff_seconds: float,
    unresolved_context: bool,
) -> dict[str, object]:
    return {
        "request_kind": request_kind,
        "plot_index": plot_index,
        "provider": provider,
        "model": model,
        "model_id": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
        "retry_backoff_seconds": retry_backoff_seconds,
        "evidence_mode": evidence_mode,
        "summarization_mode": summarization_mode,
        "text_source_mode": text_source_mode,
        "prompt_variant": prompt_variant,
        "enabled_style_features": list(enabled_style_features),
        "summarizers": list(summarizers),
        "repetition": repetition,
        "prompt_path": str(prompt_path),
        "prompt_text": prompt_text,
        "prompt_length": len(prompt_text),
        "prompt_signature": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "image_attached": image_path is not None,
        "image_path": str(image_path) if image_path is not None else None,
        "stats_table_csv_path": str(stats_table_csv_path) if stats_table_csv_path is not None else None,
        "unresolved_context_placeholder": unresolved_context,
        "context_placeholder": CONTEXT_PLACEHOLDER if unresolved_context else None,
    }


def _request_row(*, case_id: str, abm: str, request_plan: dict[str, object]) -> dict[str, str]:
    enabled_style_features = cast(list[object], request_plan.get("enabled_style_features", []))
    summarizers = cast(list[object], request_plan.get("summarizers", []))
    return {
        "case_id": case_id,
        "abm": abm,
        "request_kind": str(request_plan["request_kind"]),
        "plot_index": str(request_plan["plot_index"] or ""),
        "model_id": str(request_plan["model_id"]),
        "provider": str(request_plan["provider"]),
        "model": str(request_plan["model"]),
        "evidence_mode": str(request_plan["evidence_mode"]),
        "summarization_mode": str(request_plan["summarization_mode"]),
        "text_source_mode": str(request_plan["text_source_mode"]),
        "prompt_variant": str(request_plan["prompt_variant"]),
        "enabled_style_features": "|".join(str(item) for item in enabled_style_features),
        "summarizers": "|".join(str(item) for item in summarizers),
        "repetition": str(request_plan["repetition"]),
        "prompt_path": str(request_plan["prompt_path"]),
        "prompt_signature": str(request_plan["prompt_signature"]),
        "image_path": str(request_plan["image_path"] or ""),
        "stats_table_csv_path": str(request_plan["stats_table_csv_path"] or ""),
        "status": str(request_plan.get("status", "ok")),
        "error_code": str(request_plan.get("error_code") or ""),
        "error": str(request_plan.get("error") or ""),
    }


def _write_csv(*, path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown_report(result: DoESmokeSuiteResult) -> str:
    case_count_by_abm: dict[str, int] = {}
    case_count_by_model: dict[str, int] = {}
    request_count_by_abm: dict[str, int] = {}
    failure_count_by_model: dict[str, int] = {}
    failure_count_by_abm: dict[str, int] = {}
    failure_reason_counts: dict[str, int] = {}
    for case in result.cases:
        case_count_by_abm[case.abm] = case_count_by_abm.get(case.abm, 0) + 1
        case_count_by_model[case.model_id] = case_count_by_model.get(case.model_id, 0) + 1
        request_count_by_abm[case.abm] = request_count_by_abm.get(case.abm, 0) + case.request_count
        if case.status == "failed":
            failure_count_by_abm[case.abm] = failure_count_by_abm.get(case.abm, 0) + 1
            failure_count_by_model[case.model_id] = failure_count_by_model.get(case.model_id, 0) + 1
            case_manifest = json.loads(case.case_manifest_path.read_text(encoding="utf-8"))
            context_error_code = str(case_manifest["context_request"].get("error_code") or "")
            if context_error_code:
                failure_reason_counts[context_error_code] = failure_reason_counts.get(context_error_code, 0) + 1
            for request in case_manifest["trend_requests"]:
                error_code = str(request.get("error_code") or "")
                if error_code:
                    failure_reason_counts[error_code] = failure_reason_counts.get(error_code, 0) + 1

    lines = [
        "# DOE Smoke Report",
        "",
        "## Overview",
        "",
        "This report materializes the exact pre-LLM design matrix for the current DOE setup.",
        "It groups shared ABM artifacts separately from case-specific request plans.",
        "",
        f"- total_cases: `{result.total_cases}`",
        f"- total_planned_requests: `{result.total_planned_requests}`",
        f"- total_context_requests: `{result.total_context_requests}`",
        f"- total_trend_requests: `{result.total_trend_requests}`",
        f"- success: `{str(result.success).lower()}`",
        f"- design_matrix_csv_path: `{result.design_matrix_csv_path}`",
        f"- request_matrix_csv_path: `{result.request_matrix_csv_path}`",
        f"- shared_root: `{result.output_root / 'shared'}`",
        f"- case_root: `{result.output_root / 'cases'}`",
        "",
        "## DOE Dimensions",
        "",
        f"- abm_count: `{len(result.abm_shared)}`",
        f"- model_count: `{len(case_count_by_model)}`",
        "- evidence_mode_count: `3`",
        "- summarization_count: `5`",
        "- prompt_variant_count: `8`",
        "- repetition_count: `3`",
        "",
        "## Case Distribution",
        "",
        "| group | id | case_count | request_count |",
        "| --- | --- | --- | --- |",
    ]
    for abm in sorted(case_count_by_abm):
        lines.append(f"| abm | {abm} | {case_count_by_abm[abm]} | {request_count_by_abm[abm]} |")
    for model_id in sorted(case_count_by_model):
        lines.append(f"| model | {model_id} | {case_count_by_model[model_id]} | - |")
    lines.extend(
        [
            "",
        "## Shared ABM Bundles",
        "",
        "| abm | plot_count | source | shared_dir | stage_errors |",
        "| --- | --- | --- | --- | --- |",
        ]
    )
    for abm, shared in sorted(result.abm_shared.items()):
        lines.append(
            f"| {abm} | {shared.plot_count} | {shared.source_viz_artifact_source} | "
            f"`{shared.shared_dir}` | {'; '.join(shared.stage_errors)} |"
        )
    lines.extend(
        [
            "",
            "## Failure Summary",
            "",
        ]
    )
    if not result.failed_case_ids:
        lines.append("No failed DOE smoke cases.")
    else:
        lines.extend(
            [
                "| group | id | failed_case_count |",
                "| --- | --- | --- |",
            ]
        )
        for abm in sorted(failure_count_by_abm):
            lines.append(f"| abm | {abm} | {failure_count_by_abm[abm]} |")
        for model_id in sorted(failure_count_by_model):
            lines.append(f"| model | {model_id} | {failure_count_by_model[model_id]} |")
        lines.extend(
            [
                "",
                "| error_code | occurrences |",
                "| --- | --- |",
            ]
        )
        for error_code in sorted(failure_reason_counts):
            lines.append(f"| {error_code} | {failure_reason_counts[error_code]} |")
        lines.extend(
            [
                "",
                "## Failed Case Examples",
                "",
            ]
        )
        for case_id in result.failed_case_ids[:20]:
            lines.append(f"- `{case_id}`")
        remaining_failed = len(result.failed_case_ids) - min(len(result.failed_case_ids), 20)
        if remaining_failed > 0:
            lines.append(
                f"- `{remaining_failed}` additional failed cases omitted here; "
                "inspect `design_matrix.csv` for the full list."
            )
    lines.extend(
        [
            "",
            "## Failed Cases",
            "",
        ]
    )
    if not result.failed_case_ids:
        lines.append("No failed DOE smoke cases.")
    else:
        lines.append("Use `design_matrix.csv` and per-case manifests under `cases/` for the complete failed-case set.")
    return "\n".join(lines) + "\n"
