"""Pre-LLM DOE smoke reporting for prompt, evidence, and artifact validation."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from distill_abm.configs.models import PromptsConfig
from distill_abm.configs.runtime_defaults import get_runtime_defaults
from distill_abm.ingest.csv_ingest import load_simulation_csv
from distill_abm.pipeline import helpers
from distill_abm.pipeline.smoke import SmokeCase, default_smoke_cases
from distill_abm.utils import detect_placeholder_signals
from distill_abm.viz.plots import MetricPlotBundle, plot_metric_bundles

DoESmokeStatus = Literal["ok", "failed"]
DoESmokeErrorCode = Literal[
    "missing_or_empty_artifact",
    "placeholder_detected",
    "unmatched_metric_pattern",
]

CONTEXT_PLACEHOLDER = "<<context_response_from_context_llm>>"


class DoESmokeAbmInput(BaseModel):
    """Resolved inputs used to inspect one ABM pre-LLM bundle."""

    abm: str
    csv_path: Path
    parameters_path: Path
    documentation_path: Path
    metric_pattern: str
    metric_description: str
    plot_description: str | None = None
    source_viz_plot_dir: Path | None = None
    source_viz_artifact_source: Literal["simulated", "fallback", "unknown"] = "unknown"


class DoESmokeStage(BaseModel):
    """One artifact-focused DOE smoke stage."""

    stage_id: str
    artifact_key: str
    description: str


class DoESmokeArtifact(BaseModel):
    """Artifact metadata for debug-friendly report inspection."""

    path: Path
    exists: bool
    size_bytes: int = 0
    sha256: str | None = None
    preview: str = ""


class DoESmokeStageResult(BaseModel):
    """One stage result for one ABM/case bundle."""

    stage: DoESmokeStage
    status: DoESmokeStatus
    artifact: DoESmokeArtifact
    error_code: DoESmokeErrorCode | None = None
    error: str | None = None


class DoESmokeCaseResult(BaseModel):
    """One pre-LLM case bundle for one ABM and one smoke setting."""

    abm: str
    case_id: str
    status: DoESmokeStatus
    output_dir: Path
    provider: str
    model: str
    evidence_mode: str
    text_source_mode: str
    enabled_style_features: list[str] = Field(default_factory=list)
    summarizers: list[str] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    max_retries: int
    retry_backoff_seconds: float
    input_csv_path: Path
    parameters_path: Path
    documentation_path: Path
    source_viz_plot_dir: Path | None = None
    source_viz_artifact_source: str = "unknown"
    matched_metric_columns: list[str] = Field(default_factory=list)
    pipeline_plot_path: Path | None = None
    stats_table_csv_path: Path | None = None
    context_prompt_path: Path
    trend_prompt_template_path: Path
    context_request_plan_path: Path
    trend_request_plan_path: Path
    evidence_image_path: Path | None = None
    stage_results: list[DoESmokeStageResult] = Field(default_factory=list)


class DoESmokeSuiteResult(BaseModel):
    """Top-level pre-LLM DOE smoke report."""

    started_at_utc: str
    finished_at_utc: str
    output_root: Path
    success: bool
    failed_case_ids: list[str] = Field(default_factory=list)
    selected_case_ids: list[str] = Field(default_factory=list)
    report_markdown_path: Path
    report_json_path: Path
    design_matrix_csv_path: Path
    cases: list[DoESmokeCaseResult] = Field(default_factory=list)


def default_doe_smoke_stages() -> list[DoESmokeStage]:
    """Return the canonical pre-LLM smoke stages for each DOE case."""
    return [
        DoESmokeStage(stage_id="simulation-csv", artifact_key="simulation_csv", description="Simulation CSV input."),
        DoESmokeStage(stage_id="parameters", artifact_key="parameters", description="Parameter narrative input."),
        DoESmokeStage(stage_id="documentation", artifact_key="documentation", description="Documentation input."),
        DoESmokeStage(stage_id="pipeline-plot", artifact_key="pipeline_plot", description="Generated pipeline plot."),
        DoESmokeStage(stage_id="stats-table", artifact_key="stats_table_csv", description="Generated stats table CSV."),
        DoESmokeStage(stage_id="context-prompt", artifact_key="context_prompt", description="Exact first-call prompt."),
        DoESmokeStage(
            stage_id="trend-prompt-template",
            artifact_key="trend_prompt_template",
            description="Second-call prompt template with unresolved context placeholder.",
        ),
        DoESmokeStage(
            stage_id="context-request-plan",
            artifact_key="context_request_plan",
            description="Exact first LLM request plan without execution.",
        ),
        DoESmokeStage(
            stage_id="trend-request-plan",
            artifact_key="trend_request_plan",
            description="Second LLM request plan with unresolved context placeholder.",
        ),
    ]


def run_doe_smoke_suite(
    *,
    abm_inputs: dict[str, DoESmokeAbmInput],
    prompts: PromptsConfig,
    provider: str,
    model: str,
    output_root: Path,
    cases: list[SmokeCase] | None = None,
    selected_case_ids: list[str] | None = None,
) -> DoESmokeSuiteResult:
    """Materialize and validate pre-LLM artifacts for every selected DOE case."""
    started_at = datetime.now(UTC)
    output_root.mkdir(parents=True, exist_ok=True)
    case_list = cases or default_smoke_cases()
    defaults = get_runtime_defaults().llm_request

    stage_defs = default_doe_smoke_stages()
    case_results: list[DoESmokeCaseResult] = []
    failed_case_ids: list[str] = []
    design_rows: list[dict[str, str]] = []

    for abm in sorted(abm_inputs):
        abm_input = abm_inputs[abm]
        shared_paths, matched_metric_columns = _materialize_shared_artifacts(
            abm_input=abm_input,
            output_root=output_root,
        )
        stats_table_csv_text = shared_paths["stats_table_csv"].read_text(encoding="utf-8")
        for case in case_list:
            case_dir = output_root / abm / "cases" / case.case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            enabled = None if case.enabled_style_features is None else set(case.enabled_style_features)
            context_prompt = helpers.build_context_prompt(
                inputs_csv_path=abm_input.parameters_path,
                inputs_doc_path=abm_input.documentation_path,
                prompts=prompts,
                enabled=enabled,
            )
            trend_prompt_template = helpers.build_trend_prompt(
                prompts=prompts,
                metric_description=abm_input.metric_description,
                context=CONTEXT_PLACEHOLDER,
                plot_description=abm_input.plot_description,
                evidence_mode=case.evidence_mode,
                stats_table_csv=stats_table_csv_text,
                enabled=enabled,
            )
            context_prompt_path = case_dir / "context_prompt.txt"
            context_prompt_path.write_text(context_prompt, encoding="utf-8")
            trend_prompt_template_path = case_dir / "trend_prompt_template.txt"
            trend_prompt_template_path.write_text(trend_prompt_template, encoding="utf-8")

            evidence_image_path = (
                shared_paths["pipeline_plot"] if case.evidence_mode in {"plot", "plot+table"} else None
            )
            context_request_plan_path = case_dir / "context_request_plan.json"
            trend_request_plan_path = case_dir / "trend_request_plan.json"
            _write_request_plan(
                output_path=context_request_plan_path,
                provider=provider,
                model=model,
                prompt_path=context_prompt_path,
                prompt_text=context_prompt,
                image_path=None,
                temperature=defaults.temperature,
                max_tokens=defaults.max_tokens,
                max_retries=defaults.max_retries,
                retry_backoff_seconds=defaults.retry_backoff_seconds,
                unresolved_context=False,
                evidence_mode=case.evidence_mode,
                text_source_mode=case.text_source_mode,
                summarizers=list(case.summarizers or ()),
            )
            _write_request_plan(
                output_path=trend_request_plan_path,
                provider=provider,
                model=model,
                prompt_path=trend_prompt_template_path,
                prompt_text=trend_prompt_template,
                image_path=evidence_image_path,
                temperature=defaults.temperature,
                max_tokens=defaults.max_tokens,
                max_retries=defaults.max_retries,
                retry_backoff_seconds=defaults.retry_backoff_seconds,
                unresolved_context=True,
                evidence_mode=case.evidence_mode,
                text_source_mode=case.text_source_mode,
                summarizers=list(case.summarizers or ()),
            )

            artifact_paths = {
                "simulation_csv": abm_input.csv_path,
                "parameters": abm_input.parameters_path,
                "documentation": abm_input.documentation_path,
                "pipeline_plot": shared_paths["pipeline_plot"],
                "stats_table_csv": shared_paths["stats_table_csv"],
                "context_prompt": context_prompt_path,
                "trend_prompt_template": trend_prompt_template_path,
                "context_request_plan": context_request_plan_path,
                "trend_request_plan": trend_request_plan_path,
            }
            stage_results = [
                _build_stage_result(
                    stage=stage,
                    artifact_path=artifact_paths[stage.artifact_key],
                    matched_metric_columns=matched_metric_columns,
                )
                for stage in stage_defs
            ]
            status: DoESmokeStatus = "failed" if any(item.status == "failed" for item in stage_results) else "ok"
            full_case_id = f"{abm}::{case.case_id}"
            if status == "failed":
                failed_case_ids.append(full_case_id)
            case_result = DoESmokeCaseResult(
                abm=abm,
                case_id=case.case_id,
                status=status,
                output_dir=case_dir,
                provider=provider,
                model=model,
                evidence_mode=case.evidence_mode,
                text_source_mode=case.text_source_mode,
                enabled_style_features=list(case.enabled_style_features or ()),
                summarizers=list(case.summarizers or ()),
                temperature=defaults.temperature,
                max_tokens=defaults.max_tokens,
                max_retries=defaults.max_retries,
                retry_backoff_seconds=defaults.retry_backoff_seconds,
                input_csv_path=abm_input.csv_path,
                parameters_path=abm_input.parameters_path,
                documentation_path=abm_input.documentation_path,
                source_viz_plot_dir=abm_input.source_viz_plot_dir,
                source_viz_artifact_source=abm_input.source_viz_artifact_source,
                matched_metric_columns=matched_metric_columns,
                pipeline_plot_path=shared_paths["pipeline_plot"],
                stats_table_csv_path=shared_paths["stats_table_csv"],
                context_prompt_path=context_prompt_path,
                trend_prompt_template_path=trend_prompt_template_path,
                context_request_plan_path=context_request_plan_path,
                trend_request_plan_path=trend_request_plan_path,
                evidence_image_path=evidence_image_path,
                stage_results=stage_results,
            )
            case_results.append(case_result)
            design_rows.append(
                {
                    "abm": abm,
                    "case_id": case.case_id,
                    "status": status,
                    "provider": provider,
                    "model": model,
                    "temperature": str(defaults.temperature),
                    "max_tokens": str(defaults.max_tokens),
                    "max_retries": str(defaults.max_retries),
                    "retry_backoff_seconds": str(defaults.retry_backoff_seconds),
                    "evidence_mode": case.evidence_mode,
                    "text_source_mode": case.text_source_mode,
                    "enabled_style_features": "|".join(case.enabled_style_features or ()),
                    "summarizers": "|".join(case.summarizers or ()),
                    "input_csv_path": str(abm_input.csv_path),
                    "parameters_path": str(abm_input.parameters_path),
                    "documentation_path": str(abm_input.documentation_path),
                    "pipeline_plot_path": str(shared_paths["pipeline_plot"]),
                    "stats_table_csv_path": str(shared_paths["stats_table_csv"]),
                    "evidence_image_path": str(evidence_image_path or ""),
                    "context_prompt_path": str(context_prompt_path),
                    "trend_prompt_template_path": str(trend_prompt_template_path),
                }
            )

    design_matrix_csv_path = output_root / "design_matrix.csv"
    _write_design_matrix(design_matrix_csv_path, design_rows)
    finished_at = datetime.now(UTC)
    report_json_path = output_root / "doe_smoke_report.json"
    report_markdown_path = output_root / "doe_smoke_report.md"
    result = DoESmokeSuiteResult(
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        output_root=output_root,
        success=not failed_case_ids,
        failed_case_ids=failed_case_ids,
        selected_case_ids=selected_case_ids or [case.case_id for case in case_list],
        report_markdown_path=report_markdown_path,
        report_json_path=report_json_path,
        design_matrix_csv_path=design_matrix_csv_path,
        cases=case_results,
    )
    report_json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    report_markdown_path.write_text(_render_markdown_report(result), encoding="utf-8")
    return result


def _materialize_shared_artifacts(
    *,
    abm_input: DoESmokeAbmInput,
    output_root: Path,
) -> tuple[dict[str, Path], list[str]]:
    shared_dir = output_root / abm_input.abm / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    frame = load_simulation_csv(abm_input.csv_path)
    matched_metric_columns = [str(column) for column in frame.columns if abm_input.metric_pattern in str(column)]
    if not matched_metric_columns:
        plot_path = shared_dir / "pipeline_plot.png"
        stats_table_csv_path = shared_dir / "stats_table.csv"
        stats_table_csv_path.write_text(
            "metric pattern did not match any simulation CSV columns\n",
            encoding="utf-8",
        )
        return {"pipeline_plot": plot_path, "stats_table_csv": stats_table_csv_path}, matched_metric_columns

    plot_path = plot_metric_bundles(
        frame=frame,
        bundles=[
            MetricPlotBundle(
                include_pattern=abm_input.metric_pattern,
                title="Simulation trend",
                y_label="value",
            )
        ],
        output_dir=shared_dir,
    )[0]
    if plot_path.name != "pipeline_plot.png":
        normalized_plot_path = shared_dir / "pipeline_plot.png"
        normalized_plot_path.write_bytes(plot_path.read_bytes())
        plot_path = normalized_plot_path
    stats_table = helpers.build_stats_table(frame=frame, include_pattern=abm_input.metric_pattern)
    stats_table_csv = helpers.build_stats_csv(stats_table)
    stats_table_csv_path = shared_dir / "stats_table.csv"
    stats_table_csv_path.write_text(stats_table_csv, encoding="utf-8")
    return {"pipeline_plot": plot_path, "stats_table_csv": stats_table_csv_path}, matched_metric_columns


def _write_request_plan(
    *,
    output_path: Path,
    provider: str,
    model: str,
    prompt_path: Path,
    prompt_text: str,
    image_path: Path | None,
    temperature: float | None,
    max_tokens: int | None,
    max_retries: int,
    retry_backoff_seconds: float,
    unresolved_context: bool,
    evidence_mode: str,
    text_source_mode: str,
    summarizers: list[str],
) -> None:
    payload = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
        "retry_backoff_seconds": retry_backoff_seconds,
        "prompt_path": str(prompt_path),
        "prompt_text": prompt_text,
        "prompt_length": len(prompt_text),
        "prompt_signature": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "image_attached": image_path is not None,
        "image_path": str(image_path) if image_path is not None else None,
        "unresolved_context_placeholder": unresolved_context,
        "context_placeholder": CONTEXT_PLACEHOLDER if unresolved_context else None,
        "evidence_mode": evidence_mode,
        "text_source_mode": text_source_mode,
        "summarizers": summarizers,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_stage_result(
    *,
    stage: DoESmokeStage,
    artifact_path: Path,
    matched_metric_columns: list[str],
) -> DoESmokeStageResult:
    artifact = _inspect_artifact(artifact_path)
    if stage.stage_id in {"pipeline-plot", "stats-table"} and not matched_metric_columns:
        return DoESmokeStageResult(
            stage=stage,
            status="failed",
            artifact=artifact,
            error_code="unmatched_metric_pattern",
            error="metric pattern did not match any simulation CSV columns",
        )
    if not artifact.exists or artifact.size_bytes <= 0:
        return DoESmokeStageResult(
            stage=stage,
            status="failed",
            artifact=artifact,
            error_code="missing_or_empty_artifact",
            error=f"artifact missing or empty: {artifact_path}",
        )
    if (
        stage.artifact_key in {"parameters", "documentation", "context_prompt", "trend_prompt_template"}
        and artifact.preview
    ):
        placeholder_hits = detect_placeholder_signals(artifact.preview)
        if placeholder_hits:
            return DoESmokeStageResult(
                stage=stage,
                status="failed",
                artifact=artifact,
                error_code="placeholder_detected",
                error=f"placeholder-like content detected: {', '.join(placeholder_hits)}",
            )
    return DoESmokeStageResult(stage=stage, status="ok", artifact=artifact)


def _inspect_artifact(path: Path) -> DoESmokeArtifact:
    if not path.exists():
        return DoESmokeArtifact(path=path, exists=False)
    size_bytes = path.stat().st_size
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    preview = ""
    if path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
        preview = path.read_text(encoding="utf-8", errors="replace")[:500]
    return DoESmokeArtifact(path=path, exists=True, size_bytes=size_bytes, sha256=sha256, preview=preview)


def _write_design_matrix(path: Path, rows: list[dict[str, str]]) -> None:
    headers = [
        "abm",
        "case_id",
        "status",
        "provider",
        "model",
        "temperature",
        "max_tokens",
        "max_retries",
        "retry_backoff_seconds",
        "evidence_mode",
        "text_source_mode",
        "enabled_style_features",
        "summarizers",
        "input_csv_path",
        "parameters_path",
        "documentation_path",
        "pipeline_plot_path",
        "stats_table_csv_path",
        "evidence_image_path",
        "context_prompt_path",
        "trend_prompt_template_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown_report(result: DoESmokeSuiteResult) -> str:
    lines = [
        "# DOE Smoke Report",
        "",
        "## Overview",
        "",
        "This report validates the pre-LLM payload construction for the current smoke design.",
        "It does not execute any LLM call.",
        "",
        "## Pre-LLM Structure",
        "",
        "1. Load simulation CSV, parameter narrative, and final documentation.",
        "2. Generate the single pipeline evidence plot from the configured metric pattern.",
        "3. Generate the stats table CSV injected for `table` and `plot+table` evidence modes.",
        "4. Build the exact context prompt.",
        "5. Build the trend prompt template with an unresolved placeholder for the context LLM response.",
        "6. Write the exact request plans, including model and hyperparameter settings, without calling the model.",
        "",
        f"- started_at_utc: `{result.started_at_utc}`",
        f"- finished_at_utc: `{result.finished_at_utc}`",
        f"- output_root: `{result.output_root}`",
        f"- success: `{str(result.success).lower()}`",
        f"- design_matrix_csv_path: `{result.design_matrix_csv_path}`",
        "",
        "## Cases",
        "",
        "| abm | case_id | status | evidence_mode | text_source_mode | model |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in result.cases:
        lines.append(
            f"| {case.abm} | {case.case_id} | {case.status} | {case.evidence_mode} | "
            f"{case.text_source_mode} | {case.provider}:{case.model} |"
        )
    for case in result.cases:
        lines.extend(
            [
                "",
                f"## {case.abm} / {case.case_id}",
                "",
                f"- source_viz_artifact_source: `{case.source_viz_artifact_source}`",
                f"- input_csv_path: `{case.input_csv_path}`",
                f"- parameters_path: `{case.parameters_path}`",
                f"- documentation_path: `{case.documentation_path}`",
                f"- pipeline_plot_path: `{case.pipeline_plot_path}`",
                f"- stats_table_csv_path: `{case.stats_table_csv_path}`",
                f"- evidence_image_path: `{case.evidence_image_path}`",
                f"- context_prompt_path: `{case.context_prompt_path}`",
                f"- trend_prompt_template_path: `{case.trend_prompt_template_path}`",
                f"- matched_metric_columns: `{', '.join(case.matched_metric_columns)}`",
                "",
                "| stage | status | artifact | error |",
                "| --- | --- | --- | --- |",
            ]
        )
        for stage_result in case.stage_results:
            lines.append(
                f"| {stage_result.stage.stage_id} | {stage_result.status} | "
                f"`{stage_result.artifact.path}` | {stage_result.error or ''} |"
            )
    return "\n".join(lines) + "\n"
