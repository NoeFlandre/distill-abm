"""Reporting helpers for smoke-suite workflows."""

from __future__ import annotations

from pathlib import Path

from distill_abm.configs.runtime_defaults import get_runtime_defaults
from distill_abm.pipeline.smoke_io import dedupe_rows, read_csv_rows, write_csv_rows
from distill_abm.pipeline.smoke_types import RESPONSE_BUNDLE_COLUMNS, SmokeCaseResult, SmokeSuiteResult


def write_run_master_csv(output_root: Path, case_results: list[SmokeCaseResult]) -> Path:
    """Merge per-case response bundles into the current run-level CSV."""
    rows: list[dict[str, str]] = []
    for case_result in case_results:
        case_rows_path = case_result.case_rows_csv_path or (case_result.output_dir / "case_responses.csv")
        if not case_rows_path.exists():
            continue
        rows.extend(read_csv_rows(case_rows_path))
    run_master_csv = output_root / "master_responses.csv"
    write_csv_rows(run_master_csv, dedupe_rows(rows), RESPONSE_BUNDLE_COLUMNS)
    return run_master_csv


def write_global_master_csv(run_master_csv: Path) -> Path:
    """Merge one run-level master CSV into the repository-global master CSV when applicable."""
    cwd = Path.cwd().resolve()
    try:
        relative = run_master_csv.resolve().relative_to(cwd)
    except ValueError:
        return run_master_csv
    if not relative.parts or relative.parts[0].lower() != "results":
        return run_master_csv

    global_master = Path("results") / "master_responses.csv"
    global_master.parent.mkdir(parents=True, exist_ok=True)
    existing = read_csv_rows(global_master) if global_master.exists() else []
    incoming = read_csv_rows(run_master_csv)
    merged = dedupe_rows([*existing, *incoming])
    write_csv_rows(global_master, merged, RESPONSE_BUNDLE_COLUMNS)
    return global_master


def render_markdown_report(result: SmokeSuiteResult) -> str:
    """Render the human-readable smoke-suite summary markdown."""
    runtime_defaults = get_runtime_defaults()
    lines: list[str] = []
    lines.append("# Qwen Smoke Suite Report")
    lines.append("")
    lines.append(f"- Provider: `{result.provider}`")
    lines.append(f"- Model: `{result.model}`")
    lines.append(f"- Started (UTC): `{result.started_at_utc}`")
    lines.append(f"- Finished (UTC): `{result.finished_at_utc}`")
    lines.append(f"- Success: `{result.success}`")
    lines.append("- Qualitative policy: `same_generation_model_for_scoring`")
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
        "| Case | Evidence | Text Source | Style Features | Summarizers | "
        "Status | Resumed | Report CSV | Plot | Metadata | Manifest |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for case in result.cases:
        lines.append(
            f"| `{case.case.case_id}` | `{case.case.evidence_mode}` | `{case.case.text_source_mode}` | "
            f"`{case.case.enabled_style_features}` | `{case.case.summarizers}` | "
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
    lines.append("Each case folder contains prompt, response, stats-table, and metadata artifacts.")
    lines.append(
        "- Use `pipeline_run_metadata.json` in each case folder for full prompts, signatures, "
        "hyperparameters, and scores."
    )
    lines.append(f"- Run master CSV: `{result.run_master_csv_path}`")
    lines.append(f"- Global master CSV: `{result.global_master_csv_path}`")
    lines.append("")
    return "\n".join(lines)
