"""End-to-end smoke-suite orchestration for local Ollama Qwen runs."""

from __future__ import annotations

import csv
import json
import shutil
import traceback
from datetime import UTC, datetime
from itertools import cycle
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from distill_abm.configs.models import PromptsConfig
from distill_abm.configs.runtime_defaults import get_runtime_defaults
from distill_abm.eval.doe_full import analyze_factorial_anova
from distill_abm.eval.qualitative_runner import evaluate_qualitative_score
from distill_abm.llm.adapters.base import LLMAdapter
from distill_abm.pipeline.run import (
    EvidenceMode,
    PipelineInputs,
    ScoreMode,
    SummarizationMode,
    run_pipeline,
    run_pipeline_sweep,
)

SmokeStatus = Literal["ok", "failed", "skipped"]
SmokeResponseKind = Literal["context", "trend"]

RESPONSE_BUNDLE_COLUMNS: tuple[str, ...] = (
    "run_output_dir",
    "case_id",
    "response_kind",
    "case_status",
    "resumed_from_existing",
    "provider",
    "model",
    "temperature",
    "max_tokens",
    "max_retries",
    "retry_backoff_seconds",
    "evidence_mode",
    "summarization_mode",
    "score_on",
    "enabled_style_features",
    "additional_summarizers",
    "input_csv_path",
    "parameters_path",
    "documentation_path",
    "scoring_reference_path",
    "scoring_reference_source",
    "scoring_reference_text",
    "prompt_path",
    "prompt_text",
    "prompt_signature",
    "prompt_length",
    "response_path",
    "response_text",
    "response_length",
    "evidence_image_path",
    "plot_path",
    "stats_table_csv_path",
    "report_csv_path",
    "metadata_path",
    "case_manifest_path",
    "selected_token_f1",
    "selected_bleu",
    "selected_meteor",
    "selected_rouge1",
    "selected_rouge2",
    "selected_rouge_l",
    "selected_flesch_reading_ease",
    "full_token_f1",
    "full_bleu",
    "full_meteor",
    "full_rouge1",
    "full_rouge2",
    "full_rouge_l",
    "full_flesch_reading_ease",
    "summary_token_f1",
    "summary_bleu",
    "summary_meteor",
    "summary_rouge1",
    "summary_rouge2",
    "summary_rouge_l",
    "summary_flesch_reading_ease",
    "error",
    "inputs_json",
    "llm_json",
    "scores_json",
    "reproducibility_json",
    "summarizers_json",
)


class SmokeCase(BaseModel):
    """Represents one pipeline case in the smoke matrix."""

    case_id: str
    evidence_mode: EvidenceMode
    summarization_mode: SummarizationMode
    score_on: ScoreMode
    enabled_style_features: tuple[str, ...] | None = None
    additional_summarizers: tuple[Literal["t5", "longformer_ext"], ...] | None = None


class QualitativeOutcome(BaseModel):
    """Stores one qualitative metric outcome for a smoke case."""

    metric: Literal["coverage", "faithfulness"]
    status: SmokeStatus
    score: int | None = None
    reasoning: str | None = None
    model: str | None = None
    prompt_path: Path | None = None
    output_path: Path | None = None
    error: str | None = None


class SmokeCaseResult(BaseModel):
    """Stores artifacts and status for one smoke case."""

    case: SmokeCase
    status: SmokeStatus
    output_dir: Path
    report_csv: Path | None = None
    plot_path: Path | None = None
    metadata_path: Path | None = None
    context_prompt_path: Path | None = None
    trend_prompt_path: Path | None = None
    stats_table_csv_path: Path | None = None
    context_response_path: Path | None = None
    trend_full_response_path: Path | None = None
    trend_summary_response_path: Path | None = None
    case_rows_csv_path: Path | None = None
    case_manifest_path: Path | None = None
    resumed_from_existing: bool = False
    qualitative: list[QualitativeOutcome] = Field(default_factory=list)
    error: str | None = None


class SmokeSuiteInputs(BaseModel):
    """Defines required input paths and runtime settings for smoke execution."""

    csv_path: Path
    parameters_path: Path
    documentation_path: Path
    output_dir: Path
    model: str
    metric_pattern: str
    metric_description: str
    plot_description: str | None = None
    sweep_plot_descriptions: list[str] | None = None
    additional_summarizers: tuple[Literal["t5", "longformer_ext"], ...] = ()
    scoring_reference_path: Path | None = None


class SmokeSuiteResult(BaseModel):
    """Top-level smoke-suite result and report pointers."""

    provider: str
    model: str
    started_at_utc: str
    finished_at_utc: str
    inputs: SmokeSuiteInputs
    qualitative_policy: Literal["debug_same_model"] = "debug_same_model"
    success: bool
    failed_cases: list[str] = Field(default_factory=list)
    cases: list[SmokeCaseResult] = Field(default_factory=list)
    doe_status: SmokeStatus = "skipped"
    doe_output_csv: Path | None = None
    doe_error: str | None = None
    sweep_status: SmokeStatus = "skipped"
    sweep_output_csv: Path | None = None
    sweep_error: str | None = None
    run_master_csv_path: Path | None = None
    global_master_csv_path: Path | None = None
    report_markdown_path: Path
    report_json_path: Path


def default_smoke_cases() -> list[SmokeCase]:
    """Return the canonical smoke matrix for evidence and text-path ablations."""
    cases: list[SmokeCase] = []
    evidence_modes: tuple[EvidenceMode, ...] = ("plot", "table-csv", "plot+table")
    summary_pairs: tuple[tuple[SummarizationMode, ScoreMode], ...] = (
        ("full", "full"),
        ("summary", "summary"),
        ("both", "both"),
    )
    for evidence_mode in evidence_modes:
        for summarization_mode, score_on in summary_pairs:
            case_id = f"{evidence_mode.replace('+', '_plus_')}-{summarization_mode}-{score_on}"
            cases.append(
                SmokeCase(
                    case_id=case_id,
                    evidence_mode=evidence_mode,
                    summarization_mode=summarization_mode,
                    score_on=score_on,
                )
            )
    return cases


def default_branch_smoke_cases() -> list[SmokeCase]:
    """Return a compact three-branch smoke profile for debugging prompt/summarizer variants."""
    return [
        SmokeCase(
            case_id="branch-role-full",
            evidence_mode="plot",
            summarization_mode="full",
            score_on="full",
            enabled_style_features=("role",),
            additional_summarizers=(),
        ),
        SmokeCase(
            case_id="branch-insights-summary-t5",
            evidence_mode="table-csv",
            summarization_mode="summary",
            score_on="summary",
            enabled_style_features=("insights",),
            additional_summarizers=("t5",),
        ),
        SmokeCase(
            case_id="branch-role-insights-summary-longformer",
            evidence_mode="plot+table",
            summarization_mode="summary",
            score_on="summary",
            enabled_style_features=("role", "insights"),
            additional_summarizers=("longformer_ext",),
        ),
    ]


def run_qwen_smoke_suite(
    inputs: SmokeSuiteInputs,
    prompts: PromptsConfig,
    adapter: LLMAdapter,
    run_qualitative: bool,
    doe_input_csv: Path | None,
    run_sweep: bool,
    cases: list[SmokeCase] | None = None,
    resume_existing: bool = True,
) -> SmokeSuiteResult:
    """Execute a full Qwen smoke suite and emit human-readable plus JSON reports."""
    started_at = datetime.now(UTC)
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    case_list = cases or default_smoke_cases()

    case_results: list[SmokeCaseResult] = []
    for case in case_list:
        case_results.append(
            _run_smoke_case(
                case=case,
                inputs=inputs,
                prompts=prompts,
                adapter=adapter,
                run_qualitative=run_qualitative,
                resume_existing=resume_existing,
            )
        )

    for case_result in case_results:
        _ensure_case_response_bundles(case_result=case_result, inputs=inputs)

    run_master_csv = _write_run_master_csv(output_root=inputs.output_dir, case_results=case_results)
    global_master_csv = _write_global_master_csv(run_master_csv=run_master_csv)

    failed_cases = [result.case.case_id for result in case_results if result.status == "failed"]
    doe_status, doe_output_csv, doe_error = _run_doe_if_requested(
        output_root=inputs.output_dir,
        doe_input_csv=doe_input_csv,
        resume_existing=resume_existing,
    )
    sweep_status, sweep_output_csv, sweep_error = _run_sweep_if_requested(
        output_root=inputs.output_dir,
        inputs=inputs,
        prompts=prompts,
        adapter=adapter,
        case_results=case_results,
        run_sweep=run_sweep,
        resume_existing=resume_existing,
    )

    success = not failed_cases and doe_status != "failed" and sweep_status != "failed"
    finished_at = datetime.now(UTC)
    report_json_path = inputs.output_dir / "smoke_report.json"
    report_markdown_path = inputs.output_dir / "smoke_report.md"
    suite = SmokeSuiteResult(
        provider=adapter.provider,
        model=inputs.model,
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        inputs=inputs,
        qualitative_policy="debug_same_model",
        success=success,
        failed_cases=failed_cases,
        cases=case_results,
        doe_status=doe_status,
        doe_output_csv=doe_output_csv,
        doe_error=doe_error,
        sweep_status=sweep_status,
        sweep_output_csv=sweep_output_csv,
        sweep_error=sweep_error,
        run_master_csv_path=run_master_csv,
        global_master_csv_path=global_master_csv,
        report_markdown_path=report_markdown_path,
        report_json_path=report_json_path,
    )
    report_json_path.write_text(suite.model_dump_json(indent=2), encoding="utf-8")
    report_markdown_path.write_text(_render_markdown_report(suite), encoding="utf-8")
    return suite


def _run_smoke_case(
    case: SmokeCase,
    inputs: SmokeSuiteInputs,
    prompts: PromptsConfig,
    adapter: LLMAdapter,
    run_qualitative: bool,
    resume_existing: bool,
) -> SmokeCaseResult:
    case_dir = inputs.output_dir / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    case_manifest = case_dir / "case_manifest.json"
    if resume_existing:
        resumed = _load_resumable_case(case_manifest)
        if resumed is not None:
            resumed.resumed_from_existing = True
            return resumed
    try:
        result = run_pipeline(
            inputs=PipelineInputs(
                csv_path=inputs.csv_path,
                parameters_path=inputs.parameters_path,
                documentation_path=inputs.documentation_path,
                output_dir=case_dir,
                model=inputs.model,
                metric_pattern=inputs.metric_pattern,
                metric_description=inputs.metric_description,
                plot_description=inputs.plot_description,
                evidence_mode=case.evidence_mode,
                summarization_mode=case.summarization_mode,
                score_on=case.score_on,
                additional_summarizers=case.additional_summarizers or inputs.additional_summarizers,
                enabled_style_features=case.enabled_style_features,
                scoring_reference_path=inputs.scoring_reference_path,
            ),
            prompts=prompts,
            adapter=adapter,
        )
    except Exception:
        case_result = SmokeCaseResult(
            case=case,
            status="failed",
            output_dir=case_dir,
            error=traceback.format_exc(),
        )
        return _write_case_manifest(case_result)

    context_prompt_path, trend_prompt_path = _write_prompt_artifacts(case_dir, result.metadata_path)
    context_response_path = case_dir / "context_response.txt"
    context_response_path.write_text(result.context_response, encoding="utf-8")
    trend_full_response_path = case_dir / "trend_full_response.txt"
    trend_full_response_path.write_text(result.trend_full_response, encoding="utf-8")

    trend_summary_response_path: Path | None = None
    if result.trend_summary_response is not None:
        trend_summary_response_path = case_dir / "trend_summary_response.txt"
        trend_summary_response_path.write_text(result.trend_summary_response, encoding="utf-8")

    stats_table_csv_path: Path | None = None
    if result.stats_table_csv is not None:
        stats_table_csv_path = case_dir / "stats_table.csv"
        stats_table_csv_path.write_text(result.stats_table_csv, encoding="utf-8")

    qualitative = _run_case_qualitative(
        case=case,
        case_dir=case_dir,
        prompts=prompts,
        adapter=adapter,
        model=inputs.model,
        source_text=result.context_response,
        summary_text=result.trend_response,
        source_image_path=result.plot_path if case.evidence_mode in {"plot", "plot+table", "plot+stats"} else None,
        run_qualitative=run_qualitative,
    )
    qualitative_failed = any(outcome.status == "failed" for outcome in qualitative)
    case_status: SmokeStatus = "failed" if qualitative_failed else "ok"
    case_error = "qualitative evaluation failed" if qualitative_failed else None
    case_result = SmokeCaseResult(
        case=case,
        status=case_status,
        output_dir=case_dir,
        report_csv=result.report_csv,
        plot_path=result.plot_path,
        metadata_path=result.metadata_path,
        context_prompt_path=context_prompt_path,
        trend_prompt_path=trend_prompt_path,
        stats_table_csv_path=stats_table_csv_path,
        context_response_path=context_response_path,
        trend_full_response_path=trend_full_response_path,
        trend_summary_response_path=trend_summary_response_path,
        qualitative=qualitative,
        error=case_error,
    )
    return _write_case_manifest(case_result)


def _write_case_manifest(case_result: SmokeCaseResult) -> SmokeCaseResult:
    manifest_path = case_result.output_dir / "case_manifest.json"
    case_result.case_manifest_path = manifest_path
    manifest_path.write_text(case_result.model_dump_json(indent=2), encoding="utf-8")
    return case_result


def _write_prompt_artifacts(case_dir: Path, metadata_path: Path | None) -> tuple[Path | None, Path | None]:
    if metadata_path is None or not metadata_path.exists():
        return None, None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    prompts_block = payload.get("prompts", {})
    context_prompt = str(prompts_block.get("context_prompt", ""))
    trend_prompt = str(prompts_block.get("trend_prompt", ""))
    context_prompt_path = case_dir / "context_prompt.txt"
    trend_prompt_path = case_dir / "trend_prompt.txt"
    context_prompt_path.write_text(context_prompt, encoding="utf-8")
    trend_prompt_path.write_text(trend_prompt, encoding="utf-8")
    return context_prompt_path, trend_prompt_path


def _run_case_qualitative(
    case: SmokeCase,
    case_dir: Path,
    prompts: PromptsConfig,
    adapter: LLMAdapter,
    model: str,
    source_text: str,
    summary_text: str,
    source_image_path: Path | None,
    run_qualitative: bool,
) -> list[QualitativeOutcome]:
    if not run_qualitative:
        return [
            QualitativeOutcome(metric="coverage", status="skipped"),
            QualitativeOutcome(metric="faithfulness", status="skipped"),
        ]

    outcomes: list[QualitativeOutcome] = []
    for metric in ("coverage", "faithfulness"):
        prompt_template = prompts.coverage_eval_prompt if metric == "coverage" else prompts.faithfulness_eval_prompt
        prompt_text = prompt_template.format(summary=summary_text, source=source_text)
        prompt_path = case_dir / f"qualitative_{metric}_prompt.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        output_path = case_dir / f"qualitative_{metric}_response.txt"
        try:
            score = evaluate_qualitative_score(
                summary=summary_text,
                source=source_text,
                metric=metric,
                model=model,
                prompts=prompts,
                adapter=adapter,
                source_image_path=source_image_path,
            )
            output_path.write_text(score.reasoning, encoding="utf-8")
            outcomes.append(
                QualitativeOutcome(
                    metric=metric,
                    status="ok",
                    score=score.score,
                    reasoning=score.reasoning,
                    model=score.model,
                    prompt_path=prompt_path,
                    output_path=output_path,
                )
            )
        except Exception:
            outcomes.append(
                QualitativeOutcome(
                    metric=metric,
                    status="failed",
                    prompt_path=prompt_path,
                    output_path=output_path,
                    error=traceback.format_exc(),
                )
            )
    return outcomes


def _run_doe_if_requested(
    output_root: Path, doe_input_csv: Path | None, resume_existing: bool
) -> tuple[SmokeStatus, Path | None, str | None]:
    if doe_input_csv is None:
        return "skipped", None, None
    output_csv = output_root / "doe" / "anova_factorial_contributions.csv"
    if resume_existing and output_csv.exists():
        return "ok", output_csv, None
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame = analyze_factorial_anova(doe_input_csv, output_csv, max_interaction_order=2)
    except Exception:
        return "failed", output_csv, traceback.format_exc()
    if frame is None:
        return "failed", output_csv, "analyze_factorial_anova returned no result"
    return "ok", output_csv, None


def _run_sweep_if_requested(
    output_root: Path,
    inputs: SmokeSuiteInputs,
    prompts: PromptsConfig,
    adapter: LLMAdapter,
    case_results: list[SmokeCaseResult],
    run_sweep: bool,
    resume_existing: bool,
) -> tuple[SmokeStatus, Path | None, str | None]:
    if not run_sweep:
        return "skipped", None, None
    sweep_output = output_root / "sweep" / "combinations_report.csv"
    sweep_descriptions = inputs.sweep_plot_descriptions or [inputs.plot_description or inputs.metric_description]
    available_plots = [case.plot_path for case in case_results if case.plot_path is not None]
    if not available_plots:
        if resume_existing and sweep_output.exists():
            return "ok", sweep_output, None
        return "failed", None, "no successful case produced a plot image for sweep execution"
    plot_count = len(sweep_descriptions)
    if plot_count <= 0:
        return "failed", None, "sweep plot descriptions cannot be empty"
    if len(available_plots) >= plot_count:
        sweep_image_paths = available_plots[:plot_count]
    else:
        sweep_image_paths = [plot for _, plot in zip(range(plot_count), cycle(available_plots), strict=False)]
    try:
        run_pipeline_sweep(
            inputs=PipelineInputs(
                csv_path=inputs.csv_path,
                parameters_path=inputs.parameters_path,
                documentation_path=inputs.documentation_path,
                output_dir=output_root / "sweep",
                model=inputs.model,
                metric_pattern=inputs.metric_pattern,
                metric_description=inputs.metric_description,
                plot_description=inputs.plot_description,
                additional_summarizers=inputs.additional_summarizers,
            ),
            prompts=prompts,
            adapter=adapter,
            image_paths=sweep_image_paths,
            plot_descriptions=sweep_descriptions,
            output_csv=sweep_output,
            resume_existing=resume_existing,
        )
    except Exception:
        return "failed", sweep_output, traceback.format_exc()
    return "ok", sweep_output, None


def _ensure_case_response_bundles(case_result: SmokeCaseResult, inputs: SmokeSuiteInputs) -> None:
    case_dir = case_result.output_dir
    inputs_dir = case_dir / "inputs"
    prompts_dir = case_dir / "prompts"
    evidence_dir = case_dir / "evidence"
    outputs_dir = case_dir / "outputs"
    responses_root = case_dir / "responses"
    for path in (inputs_dir, prompts_dir, evidence_dir, outputs_dir, responses_root):
        path.mkdir(parents=True, exist_ok=True)

    _copy_if_exists(inputs.csv_path, inputs_dir / "simulation.csv")
    _copy_if_exists(inputs.parameters_path, inputs_dir / "parameters.txt")
    _copy_if_exists(inputs.documentation_path, inputs_dir / "documentation.txt")
    if inputs.scoring_reference_path is not None:
        _copy_if_exists(inputs.scoring_reference_path, inputs_dir / "ground_truth.txt")

    _copy_if_exists(case_result.context_prompt_path, prompts_dir / "context_prompt.txt")
    _copy_if_exists(case_result.trend_prompt_path, prompts_dir / "trend_prompt.txt")
    _copy_if_exists(case_result.plot_path, evidence_dir / "plot.png")
    _copy_if_exists(case_result.stats_table_csv_path, evidence_dir / "stats_table.csv")
    _copy_if_exists(case_result.report_csv, outputs_dir / "report.csv")
    _copy_if_exists(case_result.metadata_path, outputs_dir / "pipeline_run_metadata.json")
    _copy_if_exists(case_result.case_manifest_path, outputs_dir / "case_manifest.json")

    rows = _build_case_response_rows(case_result=case_result, smoke_inputs=inputs)
    if not rows:
        return
    for row in rows:
        response_kind = str(row["response_kind"])
        response_dir = responses_root / response_kind
        response_dir.mkdir(parents=True, exist_ok=True)
        response_path = Path(str(row["response_path"])) if row["response_path"] else None
        if response_path is not None and response_path.exists():
            _copy_if_exists(response_path, response_dir / "response.txt")
        _write_csv_rows(response_dir / "response_bundle.csv", [row], RESPONSE_BUNDLE_COLUMNS)
    case_rows_csv = case_dir / "case_responses.csv"
    _write_csv_rows(case_rows_csv, rows, RESPONSE_BUNDLE_COLUMNS)
    case_result.case_rows_csv_path = case_rows_csv


def _build_case_response_rows(case_result: SmokeCaseResult, smoke_inputs: SmokeSuiteInputs) -> list[dict[str, str]]:
    metadata_payload: dict[str, object] | None = None
    if case_result.metadata_path is not None and case_result.metadata_path.exists():
        try:
            metadata_payload = json.loads(case_result.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata_payload = None

    if metadata_payload is None:
        return [_build_fallback_error_row(case_result=case_result, smoke_inputs=smoke_inputs)]

    (
        inputs_block,
        llm_block,
        request_block,
        prompts_block,
        responses_block,
        artifacts_block,
        scores_block,
        reference_block,
        reproducibility_block,
        summarizers_block,
    ) = _extract_metadata_blocks(metadata_payload)

    selected_scores = _dict(scores_block.get("selected_scores"))
    full_scores = _dict(scores_block.get("full_scores"))
    summary_scores = _dict(scores_block.get("summary_scores"))

    base = {
        "run_output_dir": str(smoke_inputs.output_dir),
        "case_id": case_result.case.case_id,
        "case_status": case_result.status,
        "resumed_from_existing": str(case_result.resumed_from_existing),
        "provider": str(llm_block.get("provider", "")),
        "model": str(llm_block.get("model", "")),
        "temperature": _stringify(request_block.get("temperature")),
        "max_tokens": _stringify(request_block.get("max_tokens")),
        "max_retries": _stringify(request_block.get("max_retries")),
        "retry_backoff_seconds": _stringify(request_block.get("retry_backoff_seconds")),
        "evidence_mode": str(inputs_block.get("evidence_mode", case_result.case.evidence_mode)),
        "summarization_mode": str(inputs_block.get("summarization_mode", case_result.case.summarization_mode)),
        "score_on": str(inputs_block.get("score_on", case_result.case.score_on)),
        "enabled_style_features": _stringify(inputs_block.get("enabled_style_features")),
        "additional_summarizers": _stringify(inputs_block.get("additional_summarizers")),
        "input_csv_path": str(inputs_block.get("csv_path", smoke_inputs.csv_path)),
        "parameters_path": str(inputs_block.get("parameters_path", smoke_inputs.parameters_path)),
        "documentation_path": str(inputs_block.get("documentation_path", smoke_inputs.documentation_path)),
        "scoring_reference_path": _stringify(reference_block.get("path")),
        "scoring_reference_source": _stringify(reference_block.get("source")),
        "scoring_reference_text": _stringify(reference_block.get("text")),
        "evidence_image_path": _stringify(artifacts_block.get("trend_evidence_image_path")),
        "plot_path": _stringify(artifacts_block.get("plot_path")),
        "stats_table_csv_path": _stringify(artifacts_block.get("stats_table_csv_path")),
        "report_csv_path": _stringify(artifacts_block.get("report_csv")),
        "metadata_path": str(case_result.metadata_path) if case_result.metadata_path else "",
        "case_manifest_path": str(case_result.case_manifest_path) if case_result.case_manifest_path else "",
        **_flatten_score_fields(selected_scores, "selected"),
        **_flatten_score_fields(full_scores, "full"),
        **_flatten_score_fields(summary_scores, "summary"),
        "error": _stringify(case_result.error),
        "inputs_json": json.dumps(inputs_block, sort_keys=True),
        "llm_json": json.dumps(llm_block, sort_keys=True),
        "scores_json": json.dumps(scores_block, sort_keys=True),
        "reproducibility_json": json.dumps(reproducibility_block, sort_keys=True),
        "summarizers_json": json.dumps(summarizers_block, sort_keys=True),
    }

    metadata_fields = (prompts_block, reproducibility_block, responses_block)
    return [
        _build_case_response_row(
            base=base,
            case_result=case_result,
            metadata_fields=metadata_fields,
            response_kind="context",
            prompt_path=case_result.context_prompt_path,
            response_path=case_result.context_response_path,
        ),
        _build_case_response_row(
            base=base,
            case_result=case_result,
            metadata_fields=metadata_fields,
            response_kind="trend",
            prompt_path=case_result.trend_prompt_path,
            response_path=case_result.trend_full_response_path,
        ),
    ]


def _extract_metadata_blocks(metadata_payload: dict[str, object] | None) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    payload = metadata_payload or {}
    inputs_block = _dict(payload.get("inputs"))
    llm_block = _dict(payload.get("llm"))
    request_block = _dict(llm_block.get("request"))
    prompts_block = _dict(payload.get("prompts"))
    responses_block = _dict(payload.get("responses"))
    artifacts_block = _dict(payload.get("artifacts"))
    scores_block = _dict(payload.get("scores"))
    reference_block = _dict(scores_block.get("reference"))
    reproducibility_block = _dict(payload.get("reproducibility"))
    summarizers_block = _dict(payload.get("summarizers"))
    return (
        inputs_block,
        llm_block,
        request_block,
        prompts_block,
        responses_block,
        artifacts_block,
        scores_block,
        reference_block,
        reproducibility_block,
        summarizers_block,
    )


def _flatten_score_fields(scores: dict[str, object], prefix: str) -> dict[str, str]:
    return {
        f"{prefix}_token_f1": _stringify(scores.get("token_f1")),
        f"{prefix}_bleu": _stringify(scores.get("bleu")),
        f"{prefix}_meteor": _stringify(scores.get("meteor")),
        f"{prefix}_rouge1": _stringify(scores.get("rouge1")),
        f"{prefix}_rouge2": _stringify(scores.get("rouge2")),
        f"{prefix}_rouge_l": _stringify(scores.get("rouge_l")),
        f"{prefix}_flesch_reading_ease": _stringify(scores.get("flesch_reading_ease")),
    }


def _build_case_response_row(
    *,
    base: dict[str, str],
    case_result: SmokeCaseResult,
    metadata_fields: tuple[dict[str, object], dict[str, object], dict[str, object]],
    response_kind: str,
    prompt_path: Path | None,
    response_path: Path | None,
) -> dict[str, str]:
    prompts_block, reproducibility_block, responses_block = metadata_fields
    is_context = response_kind == "context"
    if is_context:
        prompt_text = _stringify(prompts_block.get("context_prompt"))
        response_text = _stringify(responses_block.get("context_response"))
        response_path_value = response_path if response_path is not None else case_result.context_response_path
        prompt_signature = _stringify(reproducibility_block.get("context_prompt_signature"))
        prompt_length = _stringify(reproducibility_block.get("context_prompt_length"))
    else:
        prompt_text = _stringify(prompts_block.get("trend_prompt"))
        response_text = _stringify(responses_block.get("trend_full_response"))
        response_path_value = response_path if response_path is not None else case_result.trend_full_response_path
        prompt_signature = _stringify(reproducibility_block.get("trend_prompt_signature"))
        prompt_length = _stringify(reproducibility_block.get("trend_prompt_length"))

    row = dict(base)
    row.update(
        {
            "response_kind": response_kind,
            "prompt_path": str(prompt_path) if prompt_path else "",
            "prompt_text": prompt_text,
            "prompt_signature": prompt_signature,
            "prompt_length": prompt_length,
            "response_path": str(response_path_value) if response_path_value else "",
            "response_text": response_text,
            "response_length": str(len(response_text)),
        }
    )
    return row


def _build_fallback_error_row(case_result: SmokeCaseResult, smoke_inputs: SmokeSuiteInputs) -> dict[str, str]:
    return {
        "run_output_dir": str(smoke_inputs.output_dir),
        "case_id": case_result.case.case_id,
        "response_kind": "context",
        "case_status": case_result.status,
        "resumed_from_existing": str(case_result.resumed_from_existing),
        "provider": "",
        "model": smoke_inputs.model,
        "temperature": "",
        "max_tokens": "",
        "max_retries": "",
        "retry_backoff_seconds": "",
        "evidence_mode": case_result.case.evidence_mode,
        "summarization_mode": case_result.case.summarization_mode,
        "score_on": case_result.case.score_on,
        "enabled_style_features": _stringify(case_result.case.enabled_style_features),
        "additional_summarizers": _stringify(
            case_result.case.additional_summarizers or smoke_inputs.additional_summarizers
        ),
        "input_csv_path": str(smoke_inputs.csv_path),
        "parameters_path": str(smoke_inputs.parameters_path),
        "documentation_path": str(smoke_inputs.documentation_path),
        "scoring_reference_path": str(smoke_inputs.scoring_reference_path or ""),
        "scoring_reference_source": "",
        "scoring_reference_text": "",
        "prompt_path": "",
        "prompt_text": "",
        "prompt_signature": "",
        "prompt_length": "",
        "response_path": "",
        "response_text": "",
        "response_length": "",
        "evidence_image_path": "",
        "plot_path": str(case_result.plot_path or ""),
        "stats_table_csv_path": str(case_result.stats_table_csv_path or ""),
        "report_csv_path": str(case_result.report_csv or ""),
        "metadata_path": str(case_result.metadata_path or ""),
        "case_manifest_path": str(case_result.case_manifest_path or ""),
        "selected_token_f1": "",
        "selected_bleu": "",
        "selected_meteor": "",
        "selected_rouge1": "",
        "selected_rouge2": "",
        "selected_rouge_l": "",
        "selected_flesch_reading_ease": "",
        "full_token_f1": "",
        "full_bleu": "",
        "full_meteor": "",
        "full_rouge1": "",
        "full_rouge2": "",
        "full_rouge_l": "",
        "full_flesch_reading_ease": "",
        "summary_token_f1": "",
        "summary_bleu": "",
        "summary_meteor": "",
        "summary_rouge1": "",
        "summary_rouge2": "",
        "summary_rouge_l": "",
        "summary_flesch_reading_ease": "",
        "error": _stringify(case_result.error),
        "inputs_json": "",
        "llm_json": "",
        "scores_json": "",
        "reproducibility_json": "",
        "summarizers_json": "",
    }


def _write_run_master_csv(output_root: Path, case_results: list[SmokeCaseResult]) -> Path:
    rows: list[dict[str, str]] = []
    for case_result in case_results:
        case_rows_path = case_result.case_rows_csv_path or (case_result.output_dir / "case_responses.csv")
        if not case_rows_path.exists():
            continue
        rows.extend(_read_csv_rows(case_rows_path))
    run_master_csv = output_root / "master_responses.csv"
    _write_csv_rows(run_master_csv, _dedupe_rows(rows), RESPONSE_BUNDLE_COLUMNS)
    return run_master_csv


def _write_global_master_csv(run_master_csv: Path) -> Path:
    cwd = Path.cwd().resolve()
    try:
        relative = run_master_csv.resolve().relative_to(cwd)
    except ValueError:
        return run_master_csv
    if not relative.parts or relative.parts[0].lower() != "results":
        return run_master_csv

    global_master = Path("results") / "master_responses.csv"
    global_master.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_csv_rows(global_master) if global_master.exists() else []
    incoming = _read_csv_rows(run_master_csv)
    merged = _dedupe_rows([*existing, *incoming])
    _write_csv_rows(global_master, merged, RESPONSE_BUNDLE_COLUMNS)
    return global_master


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _write_csv_rows(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str, str, str]] = set()
    deduped: list[dict[str, str]] = []
    for row in rows:
        key = (
            row.get("run_output_dir", ""),
            row.get("case_id", ""),
            row.get("response_kind", ""),
            row.get("prompt_signature", ""),
            row.get("response_path", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _copy_if_exists(source: Path | None, destination: Path) -> None:
    if source is None or not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _render_markdown_report(result: SmokeSuiteResult) -> str:
    runtime_defaults = get_runtime_defaults()
    lines: list[str] = []
    lines.append("# Qwen Smoke Suite Report")
    lines.append("")
    lines.append(f"- Provider: `{result.provider}`")
    lines.append(f"- Model: `{result.model}`")
    lines.append(f"- Started (UTC): `{result.started_at_utc}`")
    lines.append(f"- Finished (UTC): `{result.finished_at_utc}`")
    lines.append(f"- Success: `{result.success}`")
    lines.append("- Qualitative policy: `debug_same_model` (debug-only: same generation model is reused for scoring)")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- CSV path: `{result.inputs.csv_path}`")
    lines.append(f"- Parameters path: `{result.inputs.parameters_path}`")
    lines.append(f"- Documentation path: `{result.inputs.documentation_path}`")
    lines.append(f"- Output dir: `{result.inputs.output_dir}`")
    lines.append(f"- Metric pattern: `{result.inputs.metric_pattern}`")
    lines.append(f"- Metric description: `{result.inputs.metric_description}`")
    lines.append(f"- Plot description: `{result.inputs.plot_description}`")
    lines.append(f"- Scoring reference path: `{result.inputs.scoring_reference_path}`")
    lines.append(
        "- Request defaults: "
        f"`temperature={runtime_defaults.llm_request.temperature}`, "
        f"`max_tokens={runtime_defaults.llm_request.max_tokens}`, "
        f"`max_retries={runtime_defaults.llm_request.max_retries}`, "
        f"`retry_backoff_seconds={runtime_defaults.llm_request.retry_backoff_seconds}`"
    )
    lines.append("")
    lines.append("## Case Matrix")
    lines.append("")
    lines.append(
        "| Case | Evidence | Summarization | Score On | Style Features | Summarizers | "
        "Status | Resumed | Report CSV | Plot | Metadata | Manifest |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for case in result.cases:
        lines.append(
            f"| `{case.case.case_id}` | `{case.case.evidence_mode}` | `{case.case.summarization_mode}` | "
            f"`{case.case.score_on}` | `{case.case.enabled_style_features}` | `{case.case.additional_summarizers}` | "
            f"`{case.status}` | `{case.resumed_from_existing}` | `{case.report_csv}` | "
            f"`{case.plot_path}` | `{case.metadata_path}` | `{case.case_manifest_path}` |"
        )

    lines.append("")
    lines.append("## DOE")
    lines.append("")
    lines.append(f"- Status: `{result.doe_status}`")
    lines.append(f"- Output: `{result.doe_output_csv}`")
    if result.doe_error:
        lines.append(f"- Error: `{result.doe_error}`")

    lines.append("")
    lines.append("## Prompt Sweep")
    lines.append("")
    lines.append(f"- Status: `{result.sweep_status}`")
    lines.append(f"- Output: `{result.sweep_output_csv}`")
    if result.sweep_error:
        lines.append(f"- Error: `{result.sweep_error}`")

    lines.append("")
    lines.append("## Failures")
    lines.append("")
    if not result.failed_cases and result.doe_status != "failed" and result.sweep_status != "failed":
        lines.append("- None")
    else:
        if result.failed_cases:
            lines.append(f"- Failed cases: `{', '.join(result.failed_cases)}`")
        for case in result.cases:
            if case.error:
                lines.append(f"- `{case.case.case_id}`: `{case.error}`")
            for outcome in case.qualitative:
                if outcome.error:
                    lines.append(f"- `{case.case.case_id}` `{outcome.metric}`: `{outcome.error}`")
        if result.doe_error:
            lines.append(f"- DOE: `{result.doe_error}`")
        if result.sweep_error:
            lines.append(f"- Sweep: `{result.sweep_error}`")

    lines.append("")
    lines.append("## Debug Artifacts")
    lines.append("")
    lines.append("- Each case folder contains prompt, response, stats-table, and metadata artifacts.")
    lines.append(
        "- Use `pipeline_run_metadata.json` in each case folder for full prompts, signatures, "
        "hyperparameters, and scores."
    )
    lines.append(f"- Run master CSV: `{result.run_master_csv_path}`")
    lines.append(f"- Global master CSV: `{result.global_master_csv_path}`")
    lines.append("")
    return "\n".join(lines)


def _load_resumable_case(case_manifest: Path) -> SmokeCaseResult | None:
    if not case_manifest.exists():
        return None
    try:
        loaded = SmokeCaseResult.model_validate_json(case_manifest.read_text(encoding="utf-8"))
    except Exception:
        return None
    if loaded.status != "ok":
        return None
    loaded.case_manifest_path = case_manifest
    return loaded
