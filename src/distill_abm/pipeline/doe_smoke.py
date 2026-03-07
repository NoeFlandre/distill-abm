"""Pre-LLM DOE smoke reporting for full experiment-matrix inspection."""

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

from distill_abm.configs.models import PromptsConfig, SummarizerId
from distill_abm.configs.runtime_defaults import get_runtime_defaults
from distill_abm.ingest.csv_ingest import load_simulation_csv, matching_columns
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
LEGACY_EXAMPLE_TEXT = (
    "Here is an example of the style of the report: “ The total number of earthquakes recorded in the region "
    "starts at 5,000 then it declines rapidly over the first 400 time steps. This initial decline could be "
    "attributed to the depletion of immediate stress points within the fault lines, as the most susceptible areas "
    "release their pent-up energy early in the simulation. It increases briefly at around 500 steps, likely due "
    "to the aftershocks and secondary stress points being triggered as the system seeks a new equilibrium. The "
    "simulation ends with the number of earthquakes near zero, indicating that the majority of the stress has been "
    "released and the system has stabilized.\n\nThe total seismic activity follows the same pattern, starting at "
    "10,000 events and declining steadily. This steady decline reflects a systematic reduction in seismic energy "
    "over time, as the energy distribution within the tectonic plates becomes more uniform. There is low variance "
    "across simulation runs in the first 100 steps, but deviations become noticeable afterward. This suggests that "
    "while the initial reactions to stress are highly predictable, as time progresses, the system's complexity "
    "introduces more variability. This variability could be due to differences in secondary stress points and the "
    "non-linear nature of seismic energy release over time.“"
)
LEGACY_INSIGHTS_TEXT = "When summarizing trends, provide brief insights about their implications for decision makers."
LEGACY_CONTEXT_PROMPT_TEMPLATE = (
    "Your goal is to explain an agent-based model. Your explanation must include the context of the model, its goals, "
    "and key parameters. For each parameter, include the range of values (if provided) and the value that we have set. "
    "Do not write any summary or conclusion. {parameters}\n\n"
    "The context and goals of the model are as follows. {documentation}"
)
LEGACY_TREND_PROMPT_TEMPLATE = (
    "We have a plot from repeated simulations of an agent based model. Your goal is to describe the trends in details "
    "from the plot, mentioning key time steps and values taken by the metric in the plot, and interpreting them based "
    "on the context of the model. The report must objectively describe the trends in the data without addressing the "
    "quality of the simulation. Do not refer to the plot or any visual in your description. If a plot has very simple "
    "dynamics, simply state them without expanding."
)
LEGACY_ABM_ROLE_TEXTS: dict[str, tuple[str, str]] = {
    "fauna": (
        "You are an expert in megaherbivore extinction without any statistics background.",
        "You are an expert in megaherbivore extinction with a statistic background.",
    ),
    "grazing": (
        "You are an expert in grazing systems without any statistics background.",
        "You are an expert in grazing systems with a statistic background.",
    ),
    "milk_consumption": (
        "You are an expert in Consumer Behavior without any statistics background.",
        "You are an expert in Consumer Behavior with a statistic background.",
    ),
}


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
    table_csv_path: Path | None
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
    error_codes: list[str] = Field(default_factory=list)
    status: DoESmokeStatus
    context_prompt_path: Path


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
    case_index_jsonl_path: Path
    request_index_jsonl_path: Path
    design_matrix_csv_path: Path
    request_matrix_csv_path: Path
    request_review_csv_path: Path
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


def _overview_dir(output_root: Path) -> Path:
    return output_root / "00_overview"


def _shared_root_dir(output_root: Path) -> Path:
    return output_root / "10_shared"


def _shared_global_dir(output_root: Path) -> Path:
    return _shared_root_dir(output_root) / "global"


def _case_index_dir(output_root: Path) -> Path:
    return output_root / "20_case_index"


def _shared_abm_dir(*, output_root: Path, abm: str) -> Path:
    return _shared_root_dir(output_root) / abm


def _shared_inputs_dir(*, output_root: Path, abm: str) -> Path:
    return _shared_abm_dir(output_root=output_root, abm=abm) / "01_inputs"


def _shared_evidence_dir(*, output_root: Path, abm: str) -> Path:
    return _shared_abm_dir(output_root=output_root, abm=abm) / "02_evidence"


def _shared_plots_dir(*, output_root: Path, abm: str) -> Path:
    return _shared_evidence_dir(output_root=output_root, abm=abm) / "plots"


def _shared_prompts_dir(*, output_root: Path, abm: str) -> Path:
    return _shared_abm_dir(output_root=output_root, abm=abm) / "03_prompts"


def _shared_tables_dir(*, output_root: Path, abm: str) -> Path:
    return _shared_evidence_dir(output_root=output_root, abm=abm) / "tables"


def _shared_context_prompt_path(*, output_root: Path, abm: str, prompt_variant: str) -> Path:
    return _shared_prompts_dir(output_root=output_root, abm=abm) / "context" / f"{prompt_variant}.txt"


def _shared_trend_prompt_path(
    *, output_root: Path, abm: str, prompt_variant: str, evidence_mode: str, plot_index: int
) -> Path:
    return (
        _shared_prompts_dir(output_root=output_root, abm=abm)
        / "trend"
        / evidence_mode
        / prompt_variant
        / f"plot_{plot_index}.txt"
    )


def _shared_plot_copy_path(*, output_root: Path, abm: str, plot_index: int) -> Path:
    return _shared_plots_dir(output_root=output_root, abm=abm) / f"plot_{plot_index}.png"


def _layout_guide_path(output_root: Path) -> Path:
    return _overview_dir(output_root) / "README.md"


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
        table_path.write_text(
            _build_raw_table_csv(frame=frame, reporter_pattern=plot_input.reporter_pattern),
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
                "table_evidence_format": "full raw simulation CSV subset for each plot reporter pattern",
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


def _shared_table_path(*, output_root: Path, abm: str, plot_index: int) -> Path:
    return _shared_tables_dir(output_root=output_root, abm=abm) / f"plot_{plot_index}.csv"


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
        if "unmatched metric pattern" in table_text:
            return "failed", "unmatched_metric_pattern", "metric pattern did not match any simulation CSV columns"
        if not table_text.strip():
            return "failed", "missing_or_empty_artifact", f"table evidence empty: {table_csv_path}"
    if detect_placeholder_signals(plot_description):
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
    table_csv_path: Path | None,
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
        "table_csv_path": str(table_csv_path) if table_csv_path is not None else None,
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
        "table_csv_path": str(request_plan["table_csv_path"] or ""),
        "status": str(request_plan.get("status", "ok")),
        "error_code": str(request_plan.get("error_code") or ""),
        "error": str(request_plan.get("error") or ""),
    }


def _request_review_row(*, case_id: str, abm: str, request_plan: dict[str, object]) -> dict[str, str]:
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
        "reporter_pattern": str(request_plan.get("reporter_pattern") or ""),
        "plot_description": str(request_plan.get("plot_description") or ""),
        "prompt_path": str(request_plan["prompt_path"]),
        "prompt_preview": _prompt_preview(str(request_plan["prompt_text"])),
        "prompt_length": str(request_plan["prompt_length"]),
        "prompt_signature": str(request_plan["prompt_signature"]),
        "image_path": str(request_plan["image_path"] or ""),
        "table_csv_path": str(request_plan["table_csv_path"] or ""),
        "status": str(request_plan.get("status", "ok")),
        "error_code": str(request_plan.get("error_code") or ""),
        "error": str(request_plan.get("error") or ""),
    }


def _prompt_preview(prompt_text: str, limit: int = 400) -> str:
    compact = " ".join(prompt_text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _legacy_role_texts(abm: str) -> tuple[str, str]:
    try:
        return LEGACY_ABM_ROLE_TEXTS[abm]
    except KeyError as exc:
        raise ValueError(f"missing legacy DOE role text for ABM '{abm}'") from exc


def _build_legacy_doe_context_prompt(
    *,
    abm: str,
    inputs_csv_path: Path,
    inputs_doc_path: Path,
    enabled: set[str] | None,
) -> str:
    parameters = inputs_csv_path.read_text(encoding="utf-8")
    documentation = inputs_doc_path.read_text(encoding="utf-8")
    base = LEGACY_CONTEXT_PROMPT_TEMPLATE.format(parameters=parameters, documentation=documentation)
    context_role, _trend_role = _legacy_role_texts(abm)
    if enabled is not None and "role" not in enabled:
        return base
    return f"{context_role}\n\n{base}"


def _build_legacy_doe_trend_prompt(
    *,
    abm: str,
    context_response: str,
    plot_description: str,
    evidence_mode: str,
    table_csv: str,
    enabled: set[str] | None,
) -> str:
    parts: list[str] = []
    active = enabled or set()
    _context_role, trend_role = _legacy_role_texts(abm)
    if enabled is None or "role" in active:
        parts.append(trend_role)
    parts.append(_legacy_trend_prompt_for_evidence_mode(evidence_mode))
    parts.append(f"The context and goals of the model are as follows. {context_response}")
    if enabled is None or "example" in active:
        parts.append(LEGACY_EXAMPLE_TEXT)
    if enabled is None or "insights" in active:
        parts.append(LEGACY_INSIGHTS_TEXT)
    if plot_description.strip():
        parts.append(
            _legacy_plot_description_for_evidence_mode(
                plot_description=plot_description,
                evidence_mode=evidence_mode,
            )
        )
    if evidence_mode in {"table", "plot+table"}:
        parts.append(f"Relevant simulation columns (CSV):\n{table_csv}")
    return "\n\n".join(parts)


def _build_raw_table_csv(*, frame: pd.DataFrame, reporter_pattern: str) -> str:
    columns = [str(column) for column in frame.columns]
    matched = matching_columns(columns, include_pattern=reporter_pattern)
    if not matched:
        return "unmatched metric pattern\n"
    leading_column = _preferred_time_column(columns)
    selected = [leading_column, *matched] if leading_column is not None else matched
    return frame[selected].to_csv(index=False)


def _preferred_time_column(columns: list[str]) -> str | None:
    for candidate in ("[step]", "time_step", "step"):
        if candidate in columns:
            return candidate
    return None


def _legacy_trend_prompt_for_evidence_mode(evidence_mode: str) -> str:
    if evidence_mode == "plot":
        return LEGACY_TREND_PROMPT_TEMPLATE
    if evidence_mode == "table":
        return (
            LEGACY_TREND_PROMPT_TEMPLATE
            .replace("a plot", "a data table", 1)
            .replace("from the plot", "from the data table", 1)
            .replace("the plot", "the data table")
            .replace("If a plot has very simple dynamics", "If a data table has very simple dynamics")
        )
    if evidence_mode == "plot+table":
        return LEGACY_TREND_PROMPT_TEMPLATE.replace(
            (
                "We have a plot from repeated simulations of an agent based model. "
                "Your goal is to describe the trends in details from the plot, "
                "mentioning key time steps and values taken by the metric in the plot, "
                "and interpreting them based on the context of the model. "
                "The report must objectively describe the trends in the data without "
                "addressing the quality of the simulation. "
                "Do not refer to the plot or any visual in your description. "
                "If a plot has very simple dynamics, simply state them without expanding."
            ),
            (
                "We have a plot and a data table from repeated simulations of an agent based model. "
                "Your goal is to describe the trends in details from the plot and the data table, "
                "mentioning key time steps and values taken by the metric in the plot and the data table, "
                "and interpreting them based on the context of the model. "
                "The report must objectively describe the trends in the data without "
                "addressing the quality of the simulation. "
                "Do not refer to the plot and the data table or any visual in your description. "
                "If the plot and data table have very simple dynamics, simply state them without expanding."
            ),
        )
    raise ValueError(f"unsupported DOE evidence mode: {evidence_mode}")


def _legacy_plot_description_for_evidence_mode(*, plot_description: str, evidence_mode: str) -> str:
    stripped = plot_description.strip()
    if evidence_mode == "plot":
        return stripped
    if evidence_mode == "table":
        return stripped.replace("The attachment is the plot representing", "The data table represents", 1)
    if evidence_mode == "plot+table":
        return stripped.replace(
            "The attachment is the plot representing",
            "The attachment includes the plot, and the data table represents",
            1,
        )
    raise ValueError(f"unsupported DOE evidence mode: {evidence_mode}")


def _write_csv(*, path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(*, path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_shared_global_indexes(
    *,
    output_root: Path,
    model_specs: list[DoESmokeModelSpec],
    summarization_specs: tuple[DoESmokeSummarizationSpec, ...],
    prompt_variants: tuple[DoESmokePromptVariant, ...],
    evidence_modes: tuple[Literal["plot", "table", "plot+table"], ...],
) -> None:
    shared_global_dir = _shared_global_dir(output_root)
    (shared_global_dir / "models.json").write_text(
        json.dumps([spec.model_dump(mode="json") for spec in model_specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (shared_global_dir / "summarization_modes.json").write_text(
        json.dumps([spec.model_dump(mode="json") for spec in summarization_specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (shared_global_dir / "prompt_variants.json").write_text(
        json.dumps([variant.model_dump(mode="json") for variant in prompt_variants], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (shared_global_dir / "evidence_modes.json").write_text(
        json.dumps(list(evidence_modes), indent=2),
        encoding="utf-8",
    )


def _render_layout_guide(output_root: Path) -> str:
    return (
        "# DOE smoke layout\n\n"
        "Use this directory in three passes:\n\n"
        "1. `00_overview/` for the global report and matrix CSVs.\n"
        "2. `10_shared/` for global DOE factors and ABM-level shared inputs, evidence, and prompts.\n"
        "3. `20_case_index/` for compact case and request indexes.\n\n"
        "## 00_overview\n\n"
        "- `doe_smoke_report.md`\n"
        "- `doe_smoke_report.json`\n"
        "- `design_matrix.csv`\n"
        "- `request_matrix.csv`\n\n"
        "- `request_review.csv`\n\n"
        "## 10_shared/global\n\n"
        "- `models.json`\n"
        "- `summarization_modes.json`\n"
        "- `prompt_variants.json`\n"
        "- `evidence_modes.json`\n\n"
        "## 10_shared/<abm>\n\n"
        "- `01_inputs/`: copied simulation CSV, parameters narrative, final documentation\n"
        "- `02_evidence/plots/`: copied plot images used by the DOE smoke\n"
        "- `02_evidence/tables/`: full raw simulation CSV subsets matched to each plot\n"
        "- `03_prompts/context/`: shared context prompts by prompt variant\n"
        "- `03_prompts/trend/<evidence_mode>/<prompt_variant>/`: per-plot trend prompts\n\n"
        "## 20_case_index\n\n"
        "- `cases.jsonl`: one compact JSON object per DOE case\n"
        "- `requests.jsonl`: one compact JSON object per planned request\n\n"
        "Use the matrix CSVs first. Use the JSONL indexes only when you need richer case detail without opening "
        "hundreds of files.\n"
        f"\nGenerated at: `{output_root}`\n"
    )


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
            for error_code in case.error_codes:
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
        f"- request_review_csv_path: `{result.request_review_csv_path}`",
        f"- layout_guide_path: `{_layout_guide_path(result.output_root)}`",
        f"- shared_root: `{_shared_root_dir(result.output_root)}`",
        f"- case_index_root: `{_case_index_dir(result.output_root)}`",
        f"- case_index_jsonl_path: `{result.case_index_jsonl_path}`",
        f"- request_index_jsonl_path: `{result.request_index_jsonl_path}`",
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
        lines.append(
            "Use `design_matrix.csv`, `request_matrix.csv`, and the compact indexes under `20_case_index/` "
            "for the complete failed-case set."
        )
    return "\n".join(lines) + "\n"
