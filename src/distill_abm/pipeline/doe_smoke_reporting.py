"""Reporting helpers for DOE smoke outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from distill_abm.configs.models import SummarizerId
from distill_abm.pipeline.doe_smoke_layout import case_index_dir, layout_guide_path, shared_global_dir, shared_root_dir
from distill_abm.pipeline.doe_smoke_prompts import CONTEXT_PLACEHOLDER

if TYPE_CHECKING:
    from distill_abm.pipeline.doe_smoke import (
        DoESmokeModelSpec,
        DoESmokePromptVariant,
        DoESmokeSuiteResult,
        DoESmokeSummarizationSpec,
    )


def request_plan(
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


def request_row(*, case_id: str, abm: str, request_plan: dict[str, object]) -> dict[str, str]:
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


def request_review_row(*, case_id: str, abm: str, request_plan: dict[str, object]) -> dict[str, str]:
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
        "prompt_preview": prompt_preview(str(request_plan["prompt_text"])),
        "prompt_length": str(request_plan["prompt_length"]),
        "prompt_signature": str(request_plan["prompt_signature"]),
        "image_path": str(request_plan["image_path"] or ""),
        "table_csv_path": str(request_plan["table_csv_path"] or ""),
        "status": str(request_plan.get("status", "ok")),
        "error_code": str(request_plan.get("error_code") or ""),
        "error": str(request_plan.get("error") or ""),
    }


def prompt_preview(prompt_text: str, limit: int = 400) -> str:
    compact = " ".join(prompt_text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def write_csv(*, path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(*, path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_shared_global_indexes(
    *,
    output_root: Path,
    model_specs: list[DoESmokeModelSpec],
    summarization_specs: tuple[DoESmokeSummarizationSpec, ...],
    prompt_variants: tuple[DoESmokePromptVariant, ...],
    evidence_modes: tuple[Literal["plot", "table", "plot+table"], ...],
) -> None:
    target_dir = shared_global_dir(output_root)
    (target_dir / "models.json").write_text(
        json.dumps([spec.model_dump(mode="json") for spec in model_specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target_dir / "summarization_modes.json").write_text(
        json.dumps([spec.model_dump(mode="json") for spec in summarization_specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target_dir / "prompt_variants.json").write_text(
        json.dumps([variant.model_dump(mode="json") for variant in prompt_variants], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target_dir / "evidence_modes.json").write_text(json.dumps(list(evidence_modes), indent=2), encoding="utf-8")


def render_layout_guide(output_root: Path) -> str:
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


def render_markdown_report(result: DoESmokeSuiteResult) -> str:
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
        f"- layout_guide_path: `{layout_guide_path(result.output_root)}`",
        f"- shared_root: `{shared_root_dir(result.output_root)}`",
        f"- case_index_root: `{case_index_dir(result.output_root)}`",
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
