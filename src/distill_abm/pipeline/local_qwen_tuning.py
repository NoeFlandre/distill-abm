"""Local-Qwen tuning workflow for context-window and token-budget ablations."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from distill_abm.llm.adapters.base import LLMAdapter
from distill_abm.pipeline.doe_smoke_prompts import (
    build_legacy_doe_context_prompt,
    build_legacy_doe_trend_prompt,
    build_raw_table_csv,
)
from distill_abm.pipeline.local_qwen_sample_response import StructuredSmokeText
from distill_abm.pipeline.local_qwen_sample_smoke import (
    LocalQwenCaseInput,
    LocalQwenSampleCase,
    LocalQwenSampleSmokeResult,
    default_local_qwen_sample_cases,
    run_local_qwen_sample_smoke,
)

EvidenceMode = Literal["plot", "table", "plot+table"]
EVIDENCE_MODES: tuple[EvidenceMode, ...] = ("plot", "table", "plot+table")


class LocalQwenTuningTrialResult(BaseModel):
    """One tuning trial for one evidence mode and context window."""

    evidence_mode: EvidenceMode
    num_ctx: int
    max_tokens: int
    success: bool
    trial_output_root: Path
    failed_case_ids: list[str] = Field(default_factory=list)
    average_context_total_tokens: float | None = None
    average_trend_total_tokens: float | None = None
    max_context_total_tokens: int | None = None
    max_trend_total_tokens: int | None = None
    error_messages: list[str] = Field(default_factory=list)


class LocalQwenTuningRecommendation(BaseModel):
    """Recommended limits for one evidence mode."""

    evidence_mode: EvidenceMode
    recommended_num_ctx: int | None = None
    recommended_max_tokens: int | None = None
    average_context_total_tokens: float | None = None
    average_trend_total_tokens: float | None = None
    max_context_total_tokens: int | None = None
    max_trend_total_tokens: int | None = None
    based_on_successful_trial: bool = False


class LocalQwenTuningResult(BaseModel):
    """Top-level result for the local-Qwen tuning workflow."""

    started_at_utc: str
    finished_at_utc: str
    output_root: Path
    report_json_path: Path
    report_markdown_path: Path
    trials_csv_path: Path
    success: bool
    recommendations: list[LocalQwenTuningRecommendation] = Field(default_factory=list)
    trials: list[LocalQwenTuningTrialResult] = Field(default_factory=list)


class LocalQwenTuningMonitorTrial(BaseModel):
    """One tuning-trial monitor row."""

    evidence_mode: EvidenceMode
    num_ctx: int
    status: str
    current_case_id: str | None = None
    failed_case_count: int = 0


class LocalQwenTuningMonitorSnapshot(BaseModel):
    """Compact monitor view over tuning progress."""

    output_root: Path
    exists: bool
    trials: list[LocalQwenTuningMonitorTrial] = Field(default_factory=list)


def run_local_qwen_tuning(
    *,
    case_inputs: dict[str, LocalQwenCaseInput],
    adapter: LLMAdapter,
    model: str,
    output_root: Path,
    num_ctx_candidates: tuple[int, ...],
    max_tokens_candidates: tuple[int, ...],
    cases: tuple[LocalQwenSampleCase, ...] | None = None,
    resume_existing: bool = False,
) -> LocalQwenTuningResult:
    """Run context-window ablations and summarize practical runtime limits by evidence mode."""
    started_at = datetime.now(UTC)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_cases = cases or default_local_qwen_sample_cases()
    grouped_cases = _group_cases_by_evidence_mode(selected_cases)
    trial_results: list[LocalQwenTuningTrialResult] = []
    recommendations: list[LocalQwenTuningRecommendation] = []

    for evidence_mode in EVIDENCE_MODES:
        mode_cases = grouped_cases[evidence_mode]
        successful_trial: LocalQwenTuningTrialResult | None = None
        for num_ctx in num_ctx_candidates:
            for max_tokens in max_tokens_candidates:
                minimum_required_num_ctx = _estimate_minimum_num_ctx_for_mode(
                    mode_cases=mode_cases,
                    case_inputs=case_inputs,
                    max_tokens=max_tokens,
                )
                trial_output_root = (
                    output_root / "trials" / evidence_mode / f"num_ctx_{num_ctx}" / f"max_tokens_{max_tokens}"
                )
                if num_ctx < minimum_required_num_ctx:
                    trial_results.append(
                        LocalQwenTuningTrialResult(
                            evidence_mode=evidence_mode,
                            num_ctx=num_ctx,
                            max_tokens=max_tokens,
                            success=False,
                            trial_output_root=trial_output_root,
                            error_messages=[
                                (
                                    f"skipped: estimated minimum required num_ctx is {minimum_required_num_ctx} "
                                    f"for max_tokens={max_tokens}"
                                )
                            ],
                        )
                    )
                    continue
                smoke_result = run_local_qwen_sample_smoke(
                    case_inputs=case_inputs,
                    adapter=adapter,
                    model=model,
                    output_root=trial_output_root,
                    cases=mode_cases,
                    max_tokens=max_tokens,
                    ollama_num_ctx=num_ctx,
                    resume_existing=resume_existing,
                    stop_on_failure=True,
                    max_retries=0,
                    retry_backoff_seconds=0.0,
                )
                trial = _build_trial_result(
                    evidence_mode=evidence_mode,
                    num_ctx=num_ctx,
                    max_tokens=max_tokens,
                    smoke_result=smoke_result,
                )
                trial_results.append(trial)
                if trial.success:
                    successful_trial = trial
                    break
            if successful_trial is not None:
                break
        recommendations.append(_build_recommendation(evidence_mode=evidence_mode, trial=successful_trial))

    finished_at = datetime.now(UTC)
    trials_csv_path = output_root / "local_qwen_tuning_trials.csv"
    report_json_path = output_root / "local_qwen_tuning_report.json"
    report_markdown_path = output_root / "local_qwen_tuning_report.md"
    _write_trials_csv(trials_csv_path, trial_results)
    result = LocalQwenTuningResult(
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        output_root=output_root,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
        trials_csv_path=trials_csv_path,
        success=all(item.based_on_successful_trial for item in recommendations),
        recommendations=recommendations,
        trials=trial_results,
    )
    report_json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    report_markdown_path.write_text(_render_report(result), encoding="utf-8")
    return result


def _group_cases_by_evidence_mode(
    cases: tuple[LocalQwenSampleCase, ...],
) -> dict[EvidenceMode, tuple[LocalQwenSampleCase, ...]]:
    grouped: dict[EvidenceMode, list[LocalQwenSampleCase]] = defaultdict(list)
    for case in cases:
        grouped[case.evidence_mode].append(case)
    return {mode: tuple(grouped[mode]) for mode in EVIDENCE_MODES}


def _build_trial_result(
    *,
    evidence_mode: EvidenceMode,
    num_ctx: int,
    max_tokens: int,
    smoke_result: LocalQwenSampleSmokeResult,
) -> LocalQwenTuningTrialResult:
    context_totals: list[int] = []
    trend_totals: list[int] = []
    errors: list[str] = []
    for case in smoke_result.cases:
        context_trace_path = case.case_dir / "03_outputs" / "context_trace.json"
        trend_trace_path = case.case_dir / "03_outputs" / "trend_trace.json"
        error_path = case.case_dir / "03_outputs" / "error.txt"
        context_total = _extract_total_tokens(context_trace_path)
        trend_total = _extract_total_tokens(trend_trace_path)
        if context_total is not None:
            context_totals.append(context_total)
        if trend_total is not None:
            trend_totals.append(trend_total)
        if error_path.exists():
            errors.append(error_path.read_text(encoding="utf-8").strip())

    return LocalQwenTuningTrialResult(
        evidence_mode=evidence_mode,
        num_ctx=num_ctx,
        max_tokens=max_tokens,
        success=smoke_result.success,
        trial_output_root=smoke_result.output_root,
        failed_case_ids=list(smoke_result.failed_case_ids),
        average_context_total_tokens=_average(context_totals),
        average_trend_total_tokens=_average(trend_totals),
        max_context_total_tokens=max(context_totals) if context_totals else None,
        max_trend_total_tokens=max(trend_totals) if trend_totals else None,
        error_messages=errors,
    )


def _estimate_minimum_num_ctx_for_mode(
    *,
    mode_cases: tuple[LocalQwenSampleCase, ...],
    case_inputs: dict[str, LocalQwenCaseInput],
    max_tokens: int,
) -> int:
    estimated_requirements = [
        _estimate_case_num_ctx_requirement(case=case, input_bundle=case_inputs[case.abm], max_tokens=max_tokens)
        for case in mode_cases
    ]
    return max(estimated_requirements, default=max_tokens)


def _estimate_case_num_ctx_requirement(
    *,
    case: LocalQwenSampleCase,
    input_bundle: LocalQwenCaseInput,
    max_tokens: int,
) -> int:
    enabled = set(case.prompt_variant.replace("all_three", "role+insights+example").split("+"))
    if case.prompt_variant == "none":
        enabled = set()
    context_prompt = build_legacy_doe_context_prompt(
        abm=input_bundle.abm,
        inputs_csv_path=input_bundle.parameters_path,
        inputs_doc_path=input_bundle.documentation_path,
        enabled=enabled,
    )
    context_input_tokens = _estimate_total_request_tokens(
        prompt=context_prompt,
        image_attached=False,
        max_tokens=max_tokens,
    )
    table_csv = ""
    if case.evidence_mode in {"table", "plot+table"}:
        frame = pd.read_csv(input_bundle.csv_path, sep=";")
        table_csv = build_raw_table_csv(frame=frame, reporter_pattern=input_bundle.reporter_pattern)
    trend_prompt = build_legacy_doe_trend_prompt(
        abm=input_bundle.abm,
        context_response="CONTEXT_RESPONSE_PLACEHOLDER",
        plot_description=input_bundle.plot_description,
        evidence_mode=case.evidence_mode,
        table_csv=table_csv,
        enabled=enabled,
    )
    trend_input_tokens = _estimate_total_request_tokens(
        prompt=trend_prompt,
        image_attached=case.evidence_mode in {"plot", "plot+table"},
        max_tokens=max_tokens,
    )
    return max(context_input_tokens, trend_input_tokens)


def _estimate_total_request_tokens(*, prompt: str, image_attached: bool, max_tokens: int) -> int:
    schema = StructuredSmokeText.model_json_schema()
    prompt_with_schema = (
        f"{prompt}\n\n"
        "Return your final answer as a JSON object that matches this schema exactly:\n"
        f"{json.dumps(schema, indent=2, sort_keys=True)}"
    )
    estimated_prompt_tokens = math.ceil(len(prompt_with_schema) / 4)
    image_budget = 1024 if image_attached else 0
    return estimated_prompt_tokens + image_budget + max_tokens


def _build_recommendation(
    *,
    evidence_mode: EvidenceMode,
    trial: LocalQwenTuningTrialResult | None,
) -> LocalQwenTuningRecommendation:
    if trial is None:
        return LocalQwenTuningRecommendation(evidence_mode=evidence_mode)
    return LocalQwenTuningRecommendation(
        evidence_mode=evidence_mode,
        recommended_num_ctx=trial.num_ctx,
        recommended_max_tokens=trial.max_tokens,
        average_context_total_tokens=trial.average_context_total_tokens,
        average_trend_total_tokens=trial.average_trend_total_tokens,
        max_context_total_tokens=trial.max_context_total_tokens,
        max_trend_total_tokens=trial.max_trend_total_tokens,
        based_on_successful_trial=True,
    )


def _extract_total_tokens(trace_path: Path) -> int | None:
    if not trace_path.exists():
        return None
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    response = payload.get("response")
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    total = usage.get("total_tokens")
    return total if isinstance(total, int) else None


def _average(values: list[int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _write_trials_csv(path: Path, trials: list[LocalQwenTuningTrialResult]) -> None:
    fieldnames = [
        "evidence_mode",
        "num_ctx",
        "max_tokens",
        "success",
        "trial_output_root",
        "failed_case_ids",
        "average_context_total_tokens",
        "average_trend_total_tokens",
        "max_context_total_tokens",
        "max_trend_total_tokens",
        "error_messages",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            writer.writerow(
                {
                    "evidence_mode": trial.evidence_mode,
                    "num_ctx": trial.num_ctx,
                    "max_tokens": trial.max_tokens,
                    "success": str(trial.success).lower(),
                    "trial_output_root": trial.trial_output_root,
                    "failed_case_ids": "|".join(trial.failed_case_ids),
                    "average_context_total_tokens": trial.average_context_total_tokens,
                    "average_trend_total_tokens": trial.average_trend_total_tokens,
                    "max_context_total_tokens": trial.max_context_total_tokens,
                    "max_trend_total_tokens": trial.max_trend_total_tokens,
                    "error_messages": " || ".join(trial.error_messages),
                }
            )


def _render_report(result: LocalQwenTuningResult) -> str:
    lines = [
        "# Local Qwen Tuning Report",
        "",
        f"- success: `{str(result.success).lower()}`",
        f"- trial_count: `{len(result.trials)}`",
        f"- trials_csv_path: `{result.trials_csv_path}`",
        "",
        "## Recommendations",
        "",
        (
            "| evidence_mode | recommended_num_ctx | recommended_max_tokens | "
            "avg_context_total_tokens | avg_trend_total_tokens | "
            "max_context_total_tokens | max_trend_total_tokens |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for recommendation in result.recommendations:
        lines.append(
            f"| {recommendation.evidence_mode} | {recommendation.recommended_num_ctx or ''} | "
            f"{recommendation.recommended_max_tokens or ''} | "
            f"{recommendation.average_context_total_tokens or ''} | "
            f"{recommendation.average_trend_total_tokens or ''} | "
            f"{recommendation.max_context_total_tokens or ''} | "
            f"{recommendation.max_trend_total_tokens or ''} |"
        )
    lines.extend(
        [
            "",
            "## Trials",
            "",
            "| evidence_mode | num_ctx | max_tokens | success | failed_case_ids |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for trial in result.trials:
        lines.append(
            f"| {trial.evidence_mode} | {trial.num_ctx} | {trial.max_tokens} | "
            f"{str(trial.success).lower()} | {', '.join(trial.failed_case_ids)} |"
        )
    return "\n".join(lines) + "\n"
