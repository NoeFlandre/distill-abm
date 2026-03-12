"""Smoke-suite helper functions split out from :mod:`pipeline.smoke`."""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from pathlib import Path

from distill_abm.configs.models import PromptsConfig
from distill_abm.eval.qualitative_runner import evaluate_qualitative_score
from distill_abm.llm.adapters.base import LLMAdapter
from distill_abm.pipeline import run as run_module
from distill_abm.pipeline.run import PipelineInputs, PipelineResult
from distill_abm.pipeline.smoke_io import copy_if_exists, dedupe_rows, read_csv_rows, write_csv_rows
from distill_abm.pipeline.smoke_manifests import load_resumable_case, write_case_manifest
from distill_abm.pipeline.smoke_optional_steps import run_doe_if_requested, run_sweep_if_requested
from distill_abm.pipeline.smoke_reporting import (
    render_markdown_report,
    write_global_master_csv,
    write_run_master_csv,
)
from distill_abm.pipeline.smoke_response_bundle import (
    build_case_response_row,
    build_fallback_error_row,
    dict_block,
    extract_metadata_blocks,
    flatten_score_fields,
    stringify,
)
from distill_abm.pipeline.smoke_types import (
    RESPONSE_BUNDLE_COLUMNS,
    QualitativeOutcome,
    SmokeCase,
    SmokeCaseResult,
    SmokeStatus,
    SmokeSuiteInputs,
    SmokeSuiteResult,
)


def _run_smoke_case(
    case: SmokeCase,
    inputs: SmokeSuiteInputs,
    prompts: PromptsConfig,
    adapter: LLMAdapter,
    run_qualitative: bool,
    resume_existing: bool,
    run_pipeline_fn: Callable[[PipelineInputs, PromptsConfig, LLMAdapter], PipelineResult] = run_module.run_pipeline,
) -> SmokeCaseResult:
    """Run one smoke case and materialize case-level outputs."""
    case_dir = inputs.output_dir / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    case_manifest = case_dir / "case_manifest.json"
    if resume_existing:
        resumed = load_resumable_case(case_manifest)
        if resumed is not None:
            resumed.resumed_from_existing = True
            return resumed
    try:
        result = run_pipeline_fn(
            PipelineInputs(
                csv_path=inputs.csv_path,
                parameters_path=inputs.parameters_path,
                documentation_path=inputs.documentation_path,
                output_dir=case_dir,
                model=inputs.model,
                metric_pattern=inputs.metric_pattern,
                metric_description=inputs.metric_description,
                plot_description=inputs.plot_description,
                evidence_mode=case.evidence_mode,
                text_source_mode=case.text_source_mode,
                allow_summary_fallback=inputs.allow_summary_fallback,
                summarizers=case.summarizers or inputs.summarizers,
                enabled_style_features=case.enabled_style_features,
                scoring_reference_path=inputs.scoring_reference_path,
            ),
            prompts,
            adapter,
        )
    except Exception:
        return write_case_manifest(
            SmokeCaseResult(
                case=case,
                status="failed",
                output_dir=case_dir,
                error=traceback.format_exc(),
            )
        )

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
        source_image_path=result.plot_path if case.evidence_mode in {"plot", "plot+table"} else None,
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
    return write_case_manifest(case_result)


def _write_case_manifest(case_result: SmokeCaseResult) -> SmokeCaseResult:
    return write_case_manifest(case_result)


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
    return run_doe_if_requested(output_root, doe_input_csv, resume_existing)


def _run_sweep_if_requested(
    output_root: Path,
    inputs: SmokeSuiteInputs,
    prompts: PromptsConfig,
    adapter: LLMAdapter,
    case_results: list[SmokeCaseResult],
    run_sweep: bool,
    resume_existing: bool,
) -> tuple[SmokeStatus, Path | None, str | None]:
    return run_sweep_if_requested(
        output_root=output_root,
        inputs=inputs,
        prompts=prompts,
        adapter=adapter,
        case_results=case_results,
        run_sweep=run_sweep,
        resume_existing=resume_existing,
    )


def _ensure_case_response_bundles(case_result: SmokeCaseResult, smoke_inputs: SmokeSuiteInputs) -> None:
    case_dir = case_result.output_dir
    inputs_dir = case_dir / "inputs"
    prompts_dir = case_dir / "prompts"
    evidence_dir = case_dir / "evidence"
    outputs_dir = case_dir / "outputs"
    responses_root = case_dir / "responses"
    for path in (inputs_dir, prompts_dir, evidence_dir, outputs_dir, responses_root):
        path.mkdir(parents=True, exist_ok=True)

    copy_if_exists(smoke_inputs.csv_path, inputs_dir / "simulation.csv")
    copy_if_exists(smoke_inputs.parameters_path, inputs_dir / "parameters.txt")
    copy_if_exists(smoke_inputs.documentation_path, inputs_dir / "documentation.txt")
    if smoke_inputs.scoring_reference_path is not None:
        copy_if_exists(smoke_inputs.scoring_reference_path, inputs_dir / "ground_truth.txt")
    for reference_name, reference_path in sorted(smoke_inputs.additional_scoring_reference_paths.items()):
        copy_if_exists(reference_path, inputs_dir / f"{reference_name}.txt")

    copy_if_exists(case_result.context_prompt_path, prompts_dir / "context_prompt.txt")
    copy_if_exists(case_result.trend_prompt_path, prompts_dir / "trend_prompt.txt")
    copy_if_exists(case_result.plot_path, evidence_dir / "plot.png")
    copy_if_exists(case_result.stats_table_csv_path, evidence_dir / "stats_table.csv")
    copy_if_exists(case_result.report_csv, outputs_dir / "report.csv")
    copy_if_exists(case_result.metadata_path, outputs_dir / "pipeline_run_metadata.json")
    copy_if_exists(case_result.case_manifest_path, outputs_dir / "case_manifest.json")

    rows = _build_case_response_rows(case_result=case_result, smoke_inputs=smoke_inputs)
    if not rows:
        return
    for row in rows:
        response_kind = str(row["response_kind"])
        response_dir = responses_root / response_kind
        response_dir.mkdir(parents=True, exist_ok=True)
        response_path = Path(str(row["response_path"])) if row["response_path"] else None
        if response_path is not None and response_path.exists():
            copy_if_exists(response_path, response_dir / "response.txt")
        write_csv_rows(response_dir / "response_bundle.csv", [row], RESPONSE_BUNDLE_COLUMNS)
    case_rows_csv = case_dir / "case_responses.csv"
    write_csv_rows(case_rows_csv, rows, RESPONSE_BUNDLE_COLUMNS)
    case_result.case_rows_csv_path = case_rows_csv


def _build_case_response_rows(case_result: SmokeCaseResult, smoke_inputs: SmokeSuiteInputs) -> list[dict[str, str]]:
    metadata_payload: dict[str, object] | None = None
    if case_result.metadata_path is not None and case_result.metadata_path.exists():
        try:
            metadata_payload = json.loads(case_result.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata_payload = None

    if metadata_payload is None:
        return build_fallback_error_row(case_result=case_result, smoke_inputs=smoke_inputs)

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
    ) = extract_metadata_blocks(metadata_payload)

    selected_scores = dict_block(scores_block.get("selected_scores"))
    full_scores = dict_block(scores_block.get("full_scores"))
    summary_scores = dict_block(scores_block.get("summary_scores"))
    runtime_block = dict_block(llm_block.get("runtime"))

    base = {
        "run_output_dir": str(smoke_inputs.output_dir),
        "case_id": case_result.case.case_id,
        "case_status": case_result.status,
        "resumed_from_existing": str(case_result.resumed_from_existing),
        "provider": str(llm_block.get("provider", "")),
        "model": str(llm_block.get("model", "")),
        "precision": stringify(llm_block.get("precision")),
        "runtime_providers_used": stringify(runtime_block.get("providers_used")),
        "runtime_precisions_used": stringify(runtime_block.get("precisions_used")),
        "temperature": stringify(request_block.get("temperature")),
        "max_tokens": stringify(request_block.get("max_tokens")),
        "max_retries": stringify(request_block.get("max_retries")),
        "retry_backoff_seconds": stringify(request_block.get("retry_backoff_seconds")),
        "evidence_mode": str(inputs_block.get("evidence_mode", case_result.case.evidence_mode)),
        "text_source_mode": str(inputs_block.get("text_source_mode", case_result.case.text_source_mode)),
        "enabled_style_features": stringify(inputs_block.get("enabled_style_features")),
        "summarizers": stringify(inputs_block.get("summarizers")),
        "input_csv_path": str(inputs_block.get("csv_path", smoke_inputs.csv_path)),
        "parameters_path": str(inputs_block.get("parameters_path", smoke_inputs.parameters_path)),
        "documentation_path": str(inputs_block.get("documentation_path", smoke_inputs.documentation_path)),
        "scoring_reference_path": stringify(reference_block.get("path")),
        "scoring_reference_source": stringify(reference_block.get("source")),
        "scoring_reference_text": stringify(reference_block.get("text")),
        "evidence_image_path": stringify(artifacts_block.get("trend_evidence_image_path")),
        "plot_path": stringify(artifacts_block.get("plot_path")),
        "stats_table_csv_path": stringify(artifacts_block.get("stats_table_csv_path")),
        "report_csv_path": stringify(artifacts_block.get("report_csv")),
        "metadata_path": str(case_result.metadata_path or ""),
        "case_manifest_path": str(case_result.case_manifest_path or ""),
        "error": stringify(case_result.error),
        **flatten_score_fields(selected_scores, "selected"),
        **flatten_score_fields(full_scores, "full"),
        **flatten_score_fields(summary_scores, "summary"),
        "inputs_json": json.dumps(inputs_block, sort_keys=True),
        "llm_json": json.dumps(llm_block, sort_keys=True),
        "scores_json": json.dumps(scores_block, sort_keys=True),
        "reproducibility_json": json.dumps(reproducibility_block, sort_keys=True),
        "summarizers_json": json.dumps(summarizers_block, sort_keys=True),
    }

    metadata_fields = (prompts_block, reproducibility_block, responses_block)
    return [
        build_case_response_row(
            base=base,
            case_result=case_result,
            metadata_fields=metadata_fields,
            response_kind="context",
            prompt_path=case_result.context_prompt_path,
            response_path=case_result.context_response_path,
        ),
        build_case_response_row(
            base=base,
            case_result=case_result,
            metadata_fields=metadata_fields,
            response_kind="trend",
            prompt_path=case_result.trend_prompt_path,
            response_path=case_result.trend_full_response_path,
        ),
    ]


def _write_run_master_csv(output_root: Path, case_results: list[SmokeCaseResult]) -> Path:
    return write_run_master_csv(output_root, case_results)


def _write_global_master_csv(run_master_csv: Path) -> Path:
    return write_global_master_csv(run_master_csv)


_extract_metadata_blocks = extract_metadata_blocks
_flatten_score_fields = flatten_score_fields
_build_case_response_row = build_case_response_row
_build_fallback_error_row = build_fallback_error_row
_read_csv_rows = read_csv_rows
_write_csv_rows = write_csv_rows
_dedupe_rows = dedupe_rows
_copy_if_exists = copy_if_exists
_dict = dict_block
_stringify = stringify


def _load_resumable_case(case_manifest: Path) -> SmokeCaseResult | None:
    return load_resumable_case(case_manifest)


def _render_markdown_report(result: SmokeSuiteResult) -> str:
    return render_markdown_report(result)
