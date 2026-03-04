"""End-to-end smoke-suite orchestration for local Ollama Qwen runs."""

from __future__ import annotations

import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from distill_abm.configs.models import PromptsConfig
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


class SmokeCase(BaseModel):
    """Represents one pipeline case in the smoke matrix."""

    case_id: str
    evidence_mode: EvidenceMode
    summarization_mode: SummarizationMode
    score_on: ScoreMode


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
    case_manifest_path: Path | None = None
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
    additional_summarizers: tuple[Literal["t5", "longformer_ext"], ...] = ()


class SmokeSuiteResult(BaseModel):
    """Top-level smoke-suite result and report pointers."""

    provider: str
    model: str
    started_at_utc: str
    finished_at_utc: str
    inputs: SmokeSuiteInputs
    success: bool
    failed_cases: list[str] = Field(default_factory=list)
    cases: list[SmokeCaseResult] = Field(default_factory=list)
    doe_status: SmokeStatus = "skipped"
    doe_output_csv: Path | None = None
    doe_error: str | None = None
    sweep_status: SmokeStatus = "skipped"
    sweep_output_csv: Path | None = None
    sweep_error: str | None = None
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


def run_qwen_smoke_suite(
    inputs: SmokeSuiteInputs,
    prompts: PromptsConfig,
    adapter: LLMAdapter,
    run_qualitative: bool,
    doe_input_csv: Path | None,
    run_sweep: bool,
    cases: list[SmokeCase] | None = None,
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
            )
        )

    failed_cases = [result.case.case_id for result in case_results if result.status == "failed"]
    doe_status, doe_output_csv, doe_error = _run_doe_if_requested(
        output_root=inputs.output_dir,
        doe_input_csv=doe_input_csv,
    )
    sweep_status, sweep_output_csv, sweep_error = _run_sweep_if_requested(
        output_root=inputs.output_dir,
        inputs=inputs,
        prompts=prompts,
        adapter=adapter,
        case_results=case_results,
        run_sweep=run_sweep,
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
        success=success,
        failed_cases=failed_cases,
        cases=case_results,
        doe_status=doe_status,
        doe_output_csv=doe_output_csv,
        doe_error=doe_error,
        sweep_status=sweep_status,
        sweep_output_csv=sweep_output_csv,
        sweep_error=sweep_error,
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
) -> SmokeCaseResult:
    case_dir = inputs.output_dir / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
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
                additional_summarizers=inputs.additional_summarizers,
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
    manifest_path.write_text(case_result.model_dump_json(indent=2), encoding="utf-8")
    case_result.case_manifest_path = manifest_path
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


def _run_doe_if_requested(output_root: Path, doe_input_csv: Path | None) -> tuple[SmokeStatus, Path | None, str | None]:
    if doe_input_csv is None:
        return "skipped", None, None
    output_csv = output_root / "doe" / "anova_factorial_contributions.csv"
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
) -> tuple[SmokeStatus, Path | None, str | None]:
    if not run_sweep:
        return "skipped", None, None
    first_plot = next((case.plot_path for case in case_results if case.plot_path is not None), None)
    if first_plot is None:
        return "failed", None, "no successful case produced a plot image for sweep execution"
    sweep_output = output_root / "sweep" / "combinations_report.csv"
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
            image_paths=[first_plot],
            plot_descriptions=[inputs.plot_description or inputs.metric_description],
            output_csv=sweep_output,
        )
    except Exception:
        return "failed", sweep_output, traceback.format_exc()
    return "ok", sweep_output, None


def _render_markdown_report(result: SmokeSuiteResult) -> str:
    lines: list[str] = []
    lines.append("# Qwen Smoke Suite Report")
    lines.append("")
    lines.append(f"- Provider: `{result.provider}`")
    lines.append(f"- Model: `{result.model}`")
    lines.append(f"- Started (UTC): `{result.started_at_utc}`")
    lines.append(f"- Finished (UTC): `{result.finished_at_utc}`")
    lines.append(f"- Success: `{result.success}`")
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
    lines.append("- Request defaults: `temperature=0.5`, `max_tokens=1000`")
    lines.append("")
    lines.append("## Case Matrix")
    lines.append("")
    lines.append("| Case | Evidence | Summarization | Score On | Status | Report CSV | Plot | Metadata | Manifest |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for case in result.cases:
        lines.append(
            f"| `{case.case.case_id}` | `{case.case.evidence_mode}` | `{case.case.summarization_mode}` | "
            f"`{case.case.score_on}` | `{case.status}` | `{case.report_csv}` | "
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
    lines.append("")
    return "\n".join(lines)
