"""End-to-end smoke-suite orchestration for local Ollama Qwen runs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from distill_abm.configs.models import PromptsConfig
from distill_abm.llm.adapters.base import LLMAdapter
from distill_abm.pipeline import smoke_helpers, smoke_types
from distill_abm.pipeline.run import EvidenceMode, TextSourceMode, run_pipeline

SmokeCase = smoke_types.SmokeCase
QualitativeOutcome = smoke_types.QualitativeOutcome
SmokeCaseResult = smoke_types.SmokeCaseResult
SmokeSuiteInputs = smoke_types.SmokeSuiteInputs
SmokeSuiteResult = smoke_types.SmokeSuiteResult
RESPONSE_BUNDLE_COLUMNS = smoke_types.RESPONSE_BUNDLE_COLUMNS

SmokeStatus = Literal["ok", "failed", "skipped"]


def default_smoke_cases() -> list[SmokeCase]:
    """Return the canonical smoke matrix for evidence and text-source ablations."""
    cases: list[SmokeCase] = []
    evidence_modes: tuple[EvidenceMode, ...] = ("plot", "table", "plot+table")
    text_modes: tuple[TextSourceMode, ...] = ("full_text_only", "summary_only")
    for evidence_mode in evidence_modes:
        for text_source_mode in text_modes:
            case_id = f"{evidence_mode.replace('+', '_plus_')}-{text_source_mode}"
            cases.append(SmokeCase(case_id=case_id, evidence_mode=evidence_mode, text_source_mode=text_source_mode))
    return cases


def default_branch_smoke_cases() -> list[SmokeCase]:
    """Return a compact three-branch smoke profile for prompt/summarizer variants."""
    return [
        SmokeCase(
            case_id="branch-role-full-text",
            evidence_mode="plot",
            text_source_mode="full_text_only",
            enabled_style_features=("role",),
            summarizers=("bart", "bert", "t5", "longformer_ext"),
        ),
        SmokeCase(
            case_id="branch-insights-summary-t5",
            evidence_mode="table",
            text_source_mode="summary_only",
            enabled_style_features=("insights",),
            summarizers=("t5",),
        ),
        SmokeCase(
            case_id="branch-role-insights-summary-longformer",
            evidence_mode="plot+table",
            text_source_mode="summary_only",
            enabled_style_features=("role", "insights"),
            summarizers=("longformer_ext",),
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
    """Run one smoke case through the shared helper layer."""
    return smoke_helpers._run_smoke_case(
        case=case,
        inputs=inputs,
        prompts=prompts,
        adapter=adapter,
        run_qualitative=run_qualitative,
        resume_existing=resume_existing,
        run_pipeline_fn=run_pipeline,
    )


# Re-export helper functions for compatibility with existing tests and callers.
_write_case_manifest = smoke_helpers._write_case_manifest
_write_prompt_artifacts = smoke_helpers._write_prompt_artifacts
_run_case_qualitative = smoke_helpers._run_case_qualitative
_run_doe_if_requested = smoke_helpers._run_doe_if_requested
_run_sweep_if_requested = smoke_helpers._run_sweep_if_requested


def _ensure_case_response_bundles(case_result: SmokeCaseResult, inputs: SmokeSuiteInputs) -> None:
    return smoke_helpers._ensure_case_response_bundles(case_result=case_result, smoke_inputs=inputs)


_build_case_response_rows = smoke_helpers._build_case_response_rows
_extract_metadata_blocks = smoke_helpers._extract_metadata_blocks
_flatten_score_fields = smoke_helpers._flatten_score_fields
_build_case_response_row = smoke_helpers._build_case_response_row
_build_fallback_error_row = smoke_helpers._build_fallback_error_row
_write_run_master_csv = smoke_helpers._write_run_master_csv
_write_global_master_csv = smoke_helpers._write_global_master_csv
_read_csv_rows = smoke_helpers._read_csv_rows
_write_csv_rows = smoke_helpers._write_csv_rows
_dedupe_rows = smoke_helpers._dedupe_rows
_copy_if_exists = smoke_helpers._copy_if_exists
_dict = smoke_helpers._dict
_stringify = smoke_helpers._stringify
_render_markdown_report = smoke_helpers._render_markdown_report
_load_resumable_case = smoke_helpers._load_resumable_case
