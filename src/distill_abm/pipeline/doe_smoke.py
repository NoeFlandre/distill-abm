"""Pre-LLM DOE smoke reporting for full experiment-matrix inspection."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from distill_abm.configs.models import PromptsConfig
from distill_abm.configs.runtime_defaults import get_runtime_defaults
from distill_abm.ingest.csv_ingest import load_simulation_csv
from distill_abm.pipeline.doe_smoke_layout import (
    case_index_dir as _case_index_dir,
)
from distill_abm.pipeline.doe_smoke_layout import (
    layout_guide_path as _layout_guide_path,
)
from distill_abm.pipeline.doe_smoke_layout import (
    overview_dir as _overview_dir,
)
from distill_abm.pipeline.doe_smoke_layout import (
    shared_abm_dir as _shared_abm_dir,
)
from distill_abm.pipeline.doe_smoke_layout import (
    shared_context_prompt_path as _shared_context_prompt_path,
)
from distill_abm.pipeline.doe_smoke_layout import (
    shared_global_dir as _shared_global_dir,
)
from distill_abm.pipeline.doe_smoke_layout import (
    shared_inputs_dir as _shared_inputs_dir,
)
from distill_abm.pipeline.doe_smoke_layout import (
    shared_plot_copy_path as _shared_plot_copy_path,
)
from distill_abm.pipeline.doe_smoke_layout import (
    shared_plots_dir as _shared_plots_dir,
)
from distill_abm.pipeline.doe_smoke_layout import (
    shared_prompts_dir as _shared_prompts_dir,
)
from distill_abm.pipeline.doe_smoke_layout import (
    shared_root_dir as _shared_root_dir,
)
from distill_abm.pipeline.doe_smoke_layout import (
    shared_table_path as _shared_table_path,
)
from distill_abm.pipeline.doe_smoke_layout import (
    shared_table_payload_path as _shared_table_payload_path,
)
from distill_abm.pipeline.doe_smoke_layout import (
    shared_table_series_path as _shared_table_series_path,
)
from distill_abm.pipeline.doe_smoke_layout import (
    shared_tables_dir as _shared_tables_dir,
)
from distill_abm.pipeline.doe_smoke_layout import (
    shared_trend_prompt_path as _shared_trend_prompt_path,
)
from distill_abm.pipeline.doe_smoke_models import (
    CANONICAL_DOE_MODEL_IDS,
    CANONICAL_EVIDENCE_MODES,
    CANONICAL_REPETITIONS,
    DoESmokeAbmInput,
    DoESmokeCaseResult,
    DoESmokeErrorCode,
    DoESmokeModelSpec,
    DoESmokePlotInput,
    DoESmokePromptVariant,
    DoESmokeSharedAbmResult,
    DoESmokeStatus,
    DoESmokeSuiteResult,
    DoESmokeSummarizationSpec,
    canonical_prompt_variants,
    canonical_summarization_specs,
)
from distill_abm.pipeline.doe_smoke_prompts import (
    CONTEXT_PLACEHOLDER,
)
from distill_abm.pipeline.doe_smoke_prompts import (
    build_legacy_doe_context_prompt as _build_legacy_doe_context_prompt,
)
from distill_abm.pipeline.doe_smoke_prompts import (
    build_legacy_doe_trend_prompt as _build_legacy_doe_trend_prompt,
)
from distill_abm.pipeline.doe_smoke_prompts import (
    build_raw_table_csv as _build_raw_table_csv,
)
from distill_abm.pipeline.doe_smoke_prompts import (
    build_selected_series_csv as _build_selected_series_csv,
)
from distill_abm.pipeline.doe_smoke_reporting import (
    render_layout_guide as _render_layout_guide,
)
from distill_abm.pipeline.doe_smoke_reporting import (
    render_markdown_report as _render_markdown_report,
)
from distill_abm.pipeline.doe_smoke_reporting import (
    request_plan as _request_plan,
)
from distill_abm.pipeline.doe_smoke_reporting import (
    request_review_row as _request_review_row,
)
from distill_abm.pipeline.doe_smoke_reporting import (
    request_row as _request_row,
)
from distill_abm.pipeline.doe_smoke_reporting import (
    write_csv as _write_csv,
)
from distill_abm.pipeline.doe_smoke_reporting import (
    write_jsonl as _write_jsonl,
)
from distill_abm.pipeline.doe_smoke_reporting import (
    write_shared_global_indexes as _write_shared_global_indexes,
)
from distill_abm.utils import detect_placeholder_signals

__all__ = [
    "CANONICAL_DOE_MODEL_IDS",
    "CANONICAL_EVIDENCE_MODES",
    "CANONICAL_REPETITIONS",
    "DoESmokeAbmInput",
    "DoESmokeCaseResult",
    "DoESmokeErrorCode",
    "DoESmokeModelSpec",
    "DoESmokePlotInput",
    "DoESmokePromptVariant",
    "DoESmokeSharedAbmResult",
    "DoESmokeStatus",
    "DoESmokeSuiteResult",
    "DoESmokeSummarizationSpec",
    "canonical_prompt_variants",
    "canonical_summarization_specs",
    "run_doe_smoke_suite",
]


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
    _overview_dir(output_root).mkdir(parents=True, exist_ok=True)
    _shared_root_dir(output_root).mkdir(parents=True, exist_ok=True)
    _shared_global_dir(output_root).mkdir(parents=True, exist_ok=True)
    _case_index_dir(output_root).mkdir(parents=True, exist_ok=True)
    summarization_specs = summarization_specs or canonical_summarization_specs()
    prompt_variants = prompt_variants or canonical_prompt_variants()
    defaults = get_runtime_defaults().llm_request

    shared_by_abm: dict[str, DoESmokeSharedAbmResult] = {}
    shared_context_paths: dict[tuple[str, str], Path] = {}
    shared_plot_prompt_paths: dict[tuple[str, str, str, int], Path] = {}
    failed_case_ids: list[str] = []
    case_results: list[DoESmokeCaseResult] = []
    case_records: list[dict[str, object]] = []
    design_rows: list[dict[str, str]] = []
    request_rows: list[dict[str, str]] = []
    request_review_rows: list[dict[str, str]] = []

    _write_shared_global_indexes(
        output_root=output_root,
        model_specs=model_specs,
        summarization_specs=summarization_specs,
        prompt_variants=prompt_variants,
        evidence_modes=evidence_modes,
    )

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
                            context_prompt_path = shared_context_paths[(abm, prompt_variant.variant_id)]
                            context_prompt_text = context_prompt_path.read_text(encoding="utf-8")
                            context_request = _request_plan(
                                provider=model_spec.provider,
                                model=model_spec.model,
                                model_id=model_spec.model_id,
                                prompt_path=context_prompt_path,
                                prompt_text=context_prompt_text,
                                image_path=None,
                                table_csv_path=None,
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
                            request_rows.append(_request_row(case_id=case_id, abm=abm, request_plan=context_request))
                            request_review_rows.append(
                                _request_review_row(case_id=case_id, abm=abm, request_plan=context_request)
                            )

                            trend_requests: list[dict[str, object]] = []
                            failed_plot_indices: list[int] = []
                            for plot_input in abm_input.plots:
                                prompt_path = shared_plot_prompt_paths[
                                    (abm, prompt_variant.variant_id, evidence_mode, plot_input.plot_index)
                                ]
                                prompt_text = prompt_path.read_text(encoding="utf-8")
                                shared_plot_path = _shared_plot_copy_path(
                                    output_root=output_root,
                                    abm=abm,
                                    plot_index=plot_input.plot_index,
                                )
                                table_csv_path = _shared_table_path(
                                    output_root=output_root,
                                    abm=abm,
                                    plot_index=plot_input.plot_index,
                                )
                                plot_status, plot_error_code, plot_error = _validate_plot_request(
                                    plot_path=shared_plot_path,
                                    plot_description=plot_input.plot_description,
                                    table_csv_path=(
                                        table_csv_path if evidence_mode in {"table", "plot+table"} else None
                                    ),
                                    evidence_mode=evidence_mode,
                                    shared_stage_errors=shared.stage_errors,
                                )
                                if plot_status == "failed":
                                    failed_plot_indices.append(plot_input.plot_index)
                                trend_request = _request_plan(
                                    provider=model_spec.provider,
                                    model=model_spec.model,
                                    model_id=model_spec.model_id,
                                    prompt_path=prompt_path,
                                    prompt_text=prompt_text,
                                    image_path=(shared_plot_path if evidence_mode in {"plot", "plot+table"} else None),
                                    table_csv_path=(
                                        table_csv_path if evidence_mode in {"table", "plot+table"} else None
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
                                request_review_rows.append(
                                    _request_review_row(case_id=case_id, abm=abm, request_plan=trend_request)
                                )

                            case_record = {
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
                                "request_count": 1 + len(abm_input.plots),
                                "source_viz_artifact_source": abm_input.source_viz_artifact_source,
                                "input_csv_path": str(shared.csv_path),
                                "parameters_path": str(shared.parameters_path),
                                "documentation_path": str(shared.documentation_path),
                                "context_prompt_path": str(context_prompt_path),
                                "trend_prompt_paths": [str(request["prompt_path"]) for request in trend_requests],
                                "plot_paths": [
                                    str(request["image_path"]) for request in trend_requests if request["image_path"]
                                ],
                                "table_csv_paths": [
                                    str(request["table_csv_path"])
                                    for request in trend_requests
                                    if request["table_csv_path"]
                                ],
                                "failed_plot_indices": list(failed_plot_indices),
                                "shared_stage_errors": list(shared.stage_errors),
                            }
                            case_records.append(case_record)
                            case_failed = bool(shared.stage_errors or failed_plot_indices)
                            case_status: DoESmokeStatus = "failed" if case_failed else "ok"
                            if case_status == "failed":
                                failed_case_ids.append(case_id)
                            request_error_codes = [context_request.get("error_code")]
                            request_error_codes.extend(req.get("error_code") for req in trend_requests)
                            error_codes = sorted({str(code) for code in request_error_codes if code})
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
                                    error_codes=error_codes,
                                    status=case_status,
                                    context_prompt_path=context_prompt_path,
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
                                    "error_codes": "|".join(error_codes),
                                    "context_prompt_path": str(context_prompt_path),
                                }
                            )

    case_index_jsonl_path = _case_index_dir(output_root) / "cases.jsonl"
    request_index_jsonl_path = _case_index_dir(output_root) / "requests.jsonl"
    design_matrix_csv_path = _overview_dir(output_root) / "design_matrix.csv"
    request_matrix_csv_path = _overview_dir(output_root) / "request_matrix.csv"
    request_review_csv_path = _overview_dir(output_root) / "request_review.csv"
    _write_jsonl(path=case_index_jsonl_path, rows=case_records)
    _write_jsonl(path=request_index_jsonl_path, rows=[dict(row) for row in request_rows])
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
            "error_codes",
            "context_prompt_path",
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
            "table_csv_path",
            "status",
            "error_code",
            "error",
        ],
        rows=request_rows,
    )
    _write_csv(
        path=request_review_csv_path,
        fieldnames=list(request_review_rows[0].keys()) if request_review_rows else [
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
            "reporter_pattern",
            "plot_description",
            "prompt_path",
            "prompt_preview",
            "prompt_length",
            "prompt_signature",
            "image_path",
            "table_csv_path",
            "status",
            "error_code",
            "error",
        ],
        rows=request_review_rows,
    )

    total_context_requests = len(case_results)
    total_trend_requests = sum(len(abm_inputs[case.abm].plots) for case in case_results)
    total_planned_requests = total_context_requests + total_trend_requests
    finished_at = datetime.now(UTC)
    report_json_path = _overview_dir(output_root) / "doe_smoke_report.json"
    report_markdown_path = _overview_dir(output_root) / "doe_smoke_report.md"
    _layout_guide_path(output_root).write_text(_render_layout_guide(output_root), encoding="utf-8")
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
        case_index_jsonl_path=case_index_jsonl_path,
        request_index_jsonl_path=request_index_jsonl_path,
        design_matrix_csv_path=design_matrix_csv_path,
        request_matrix_csv_path=request_matrix_csv_path,
        request_review_csv_path=request_review_csv_path,
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
    shared_dir = _shared_abm_dir(output_root=output_root, abm=abm_input.abm)
    inputs_dir = _shared_inputs_dir(output_root=output_root, abm=abm_input.abm)
    plots_dir = _shared_plots_dir(output_root=output_root, abm=abm_input.abm)
    prompts_dir = _shared_prompts_dir(output_root=output_root, abm=abm_input.abm)
    tables_dir = _shared_tables_dir(output_root=output_root, abm=abm_input.abm)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    stage_errors: list[str] = []

    _validate_text_artifact(path=abm_input.parameters_path, label="parameters", stage_errors=stage_errors)
    _validate_text_artifact(path=abm_input.documentation_path, label="documentation", stage_errors=stage_errors)
    frame = load_simulation_csv(abm_input.csv_path)
    copied_csv_path = inputs_dir / "simulation.csv"
    copied_parameters_path = inputs_dir / "parameters.txt"
    copied_documentation_path = inputs_dir / "documentation.txt"
    shutil.copy2(abm_input.csv_path, copied_csv_path)
    shutil.copy2(abm_input.parameters_path, copied_parameters_path)
    shutil.copy2(abm_input.documentation_path, copied_documentation_path)
    if len(abm_input.plots) <= 0:
        stage_errors.append("no plot definitions configured")
    for plot_input in abm_input.plots:
        if not plot_input.plot_path.exists() or plot_input.plot_path.stat().st_size <= 0:
            stage_errors.append(f"plot {plot_input.plot_index} missing or empty: {plot_input.plot_path}")
        else:
            shutil.copy2(
                plot_input.plot_path,
                _shared_plot_copy_path(output_root=output_root, abm=abm_input.abm, plot_index=plot_input.plot_index),
            )
        table_path = _shared_table_path(
            output_root=output_root,
            abm=abm_input.abm,
            plot_index=plot_input.plot_index,
        )
        table_summary = _build_raw_table_csv(frame=frame, reporter_pattern=plot_input.reporter_pattern)
        table_path.write_text(table_summary, encoding="utf-8")
        _shared_table_series_path(
            output_root=output_root,
            abm=abm_input.abm,
            plot_index=plot_input.plot_index,
        ).write_text(
            _build_selected_series_csv(frame=frame, reporter_pattern=plot_input.reporter_pattern),
            encoding="utf-8",
        )
        _shared_table_payload_path(
            output_root=output_root,
            abm=abm_input.abm,
            plot_index=plot_input.plot_index,
        ).write_text(
            json.dumps(
                {
                    "reporter_pattern": plot_input.reporter_pattern,
                    "summary_path": str(table_path),
                    "series_path": str(
                        _shared_table_series_path(
                            output_root=output_root,
                            abm=abm_input.abm,
                            plot_index=plot_input.plot_index,
                        )
                    ),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if detect_placeholder_signals(plot_input.plot_description):
            stage_errors.append(f"plot {plot_input.plot_index} description contains placeholder-like text")

    for prompt_variant in prompt_variants:
        enabled = set(prompt_variant.enabled_style_features)
        context_prompt = _build_legacy_doe_context_prompt(
            abm=abm_input.abm,
            inputs_csv_path=copied_parameters_path,
            inputs_doc_path=copied_documentation_path,
            enabled=enabled,
        )
        context_prompt_path = _shared_context_prompt_path(
            output_root=output_root,
            abm=abm_input.abm,
            prompt_variant=prompt_variant.variant_id,
        )
        context_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        context_prompt_path.write_text(context_prompt, encoding="utf-8")
        shared_context_paths[(abm_input.abm, prompt_variant.variant_id)] = context_prompt_path
        for evidence_mode in evidence_modes:
            for plot_input in abm_input.plots:
                table_csv_path = _shared_table_path(
                    output_root=output_root,
                    abm=abm_input.abm,
                    plot_index=plot_input.plot_index,
                )
                table_csv = (
                    table_csv_path.read_text(encoding="utf-8")
                    if evidence_mode in {"table", "plot+table"} and table_csv_path.exists()
                    else ""
                )
                trend_prompt = _build_legacy_doe_trend_prompt(
                    abm=abm_input.abm,
                    context_response=CONTEXT_PLACEHOLDER,
                    plot_description=plot_input.plot_description,
                    evidence_mode=evidence_mode,
                    table_csv=table_csv,
                    enabled=enabled,
                )
                trend_prompt_path = _shared_trend_prompt_path(
                    output_root=output_root,
                    abm=abm_input.abm,
                    prompt_variant=prompt_variant.variant_id,
                    evidence_mode=evidence_mode,
                    plot_index=plot_input.plot_index,
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
                "source_csv_path": str(abm_input.csv_path),
                "source_parameters_path": str(abm_input.parameters_path),
                "source_documentation_path": str(abm_input.documentation_path),
                "copied_csv_path": str(copied_csv_path),
                "copied_parameters_path": str(copied_parameters_path),
                "copied_documentation_path": str(copied_documentation_path),
                "plot_count": len(abm_input.plots),
                "source_viz_artifact_source": abm_input.source_viz_artifact_source,
                "table_evidence_format": "statistical evidence derived from the plot-relevant simulation series only",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return DoESmokeSharedAbmResult(
        abm=abm_input.abm,
        shared_dir=shared_dir,
        csv_path=copied_csv_path,
        parameters_path=copied_parameters_path,
        documentation_path=copied_documentation_path,
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


def _validate_plot_request(
    *,
    plot_path: Path,
    plot_description: str,
    table_csv_path: Path | None,
    evidence_mode: str,
    shared_stage_errors: list[str],
) -> tuple[DoESmokeStatus, DoESmokeErrorCode | None, str | None]:
    if shared_stage_errors:
        return "failed", "missing_or_empty_artifact", "; ".join(shared_stage_errors)
    if not plot_path.exists() or plot_path.stat().st_size <= 0:
        return "failed", "missing_or_empty_artifact", f"plot missing or empty: {plot_path}"
    if evidence_mode in {"table", "plot+table"}:
        if table_csv_path is None or not table_csv_path.exists():
            return "failed", "missing_or_empty_artifact", "table evidence missing"
        table_text = table_csv_path.read_text(encoding="utf-8", errors="replace")
        if "No simulation series matched the requested metric pattern." in table_text:
            return "failed", "unmatched_metric_pattern", "metric pattern did not match any simulation CSV columns"
        if not table_text.strip():
            return "failed", "missing_or_empty_artifact", f"table evidence empty: {table_csv_path}"
    if detect_placeholder_signals(plot_description):
        return "failed", "placeholder_detected", "plot description contains placeholder-like text"
    return "ok", None, None
