"""Minimal sampled local-Qwen smoke run for end-to-end prompt/evidence inspection."""

from __future__ import annotations

import csv
import json
import shutil
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
from distill_abm.pipeline.helpers import encode_image, invoke_adapter_with_trace
from distill_abm.utils import detect_placeholder_signals

EvidenceMode = Literal["plot", "table", "plot+table"]
SMOKE_MAX_TOKENS = 4096
SMOKE_OLLAMA_NUM_CTX = 16384


class StructuredSmokeText(BaseModel):
    """Structured smoke output schema for one text response."""

    response_text: str


class LocalQwenCaseInput(BaseModel):
    """Resolved per-ABM inputs for sampled local Qwen smoke cases."""

    abm: str
    csv_path: Path
    parameters_path: Path
    documentation_path: Path
    reporter_pattern: str
    plot_description: str
    plot_path: Path


class LocalQwenSampleCase(BaseModel):
    """One sampled case in the local-Qwen smoke matrix."""

    case_id: str
    abm: str
    evidence_mode: EvidenceMode
    prompt_variant: str


class LocalQwenSampleCaseResult(BaseModel):
    """One executed sampled smoke case."""

    case_id: str
    abm: str
    evidence_mode: EvidenceMode
    prompt_variant: str
    case_dir: Path
    success: bool
    error: str | None = None


class LocalQwenSampleSmokeResult(BaseModel):
    """Top-level result for the sampled local-Qwen smoke."""

    started_at_utc: str
    finished_at_utc: str
    output_root: Path
    report_json_path: Path
    report_markdown_path: Path
    review_csv_path: Path
    success: bool
    failed_case_ids: list[str] = Field(default_factory=list)
    cases: list[LocalQwenSampleCaseResult] = Field(default_factory=list)


def default_local_qwen_sample_cases() -> tuple[LocalQwenSampleCase, ...]:
    """Return a small stratified sample across ABMs, evidence modes, and prompt variants."""
    return (
        LocalQwenSampleCase(case_id="01_fauna_none_plot", abm="fauna", evidence_mode="plot", prompt_variant="none"),
        LocalQwenSampleCase(
            case_id="02_grazing_role_table",
            abm="grazing",
            evidence_mode="table",
            prompt_variant="role",
        ),
        LocalQwenSampleCase(
            case_id="03_milk_insights_plot_plus_table",
            abm="milk_consumption",
            evidence_mode="plot+table",
            prompt_variant="insights",
        ),
        LocalQwenSampleCase(
            case_id="04_fauna_example_table",
            abm="fauna",
            evidence_mode="table",
            prompt_variant="example",
        ),
        LocalQwenSampleCase(
            case_id="05_grazing_role_plus_example_plot_plus_table",
            abm="grazing",
            evidence_mode="plot+table",
            prompt_variant="role+example",
        ),
        LocalQwenSampleCase(
            case_id="06_milk_role_plus_insights_plot",
            abm="milk_consumption",
            evidence_mode="plot",
            prompt_variant="role+insights",
        ),
        LocalQwenSampleCase(
            case_id="07_fauna_insights_plus_example_plot",
            abm="fauna",
            evidence_mode="plot",
            prompt_variant="insights+example",
        ),
        LocalQwenSampleCase(
            case_id="08_grazing_all_three_table",
            abm="grazing",
            evidence_mode="table",
            prompt_variant="all_three",
        ),
    )


def run_local_qwen_sample_smoke(
    *,
    case_inputs: dict[str, LocalQwenCaseInput],
    adapter: LLMAdapter,
    model: str,
    output_root: Path,
    cases: tuple[LocalQwenSampleCase, ...] | None = None,
) -> LocalQwenSampleSmokeResult:
    """Run a minimal real local-Qwen smoke across sampled DOE-style cases."""
    started_at = datetime.now(UTC)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_cases = cases or default_local_qwen_sample_cases()
    review_rows: list[dict[str, str]] = []
    results: list[LocalQwenSampleCaseResult] = []
    failed_case_ids: list[str] = []

    for case in selected_cases:
        case_dir = output_root / "cases" / case.case_id
        inputs_dir = case_dir / "01_inputs"
        requests_dir = case_dir / "02_requests"
        outputs_dir = case_dir / "03_outputs"
        for directory in (inputs_dir, requests_dir, outputs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        input_bundle = case_inputs[case.abm]
        try:
            _validate_case_inputs(input_bundle)
            enabled = set(case.prompt_variant.replace("all_three", "role+insights+example").split("+"))
            if case.prompt_variant == "none":
                enabled = set()
            context_prompt_base = build_legacy_doe_context_prompt(
                abm=input_bundle.abm,
                inputs_csv_path=input_bundle.parameters_path,
                inputs_doc_path=input_bundle.documentation_path,
                enabled=enabled,
            )
            shutil.copy2(input_bundle.parameters_path, inputs_dir / "parameters.txt")
            shutil.copy2(input_bundle.documentation_path, inputs_dir / "documentation.txt")

            context_text, context_trace, context_prompt = _invoke_structured_smoke_text(
                adapter=adapter,
                model=model,
                prompt=context_prompt_base,
            )
            _write_text(inputs_dir / "context_prompt.txt", context_prompt)
            _write_json(requests_dir / "context_request.json", context_trace["request"])
            _write_text(outputs_dir / "context_output.txt", context_text)
            _write_json(outputs_dir / "context_trace.json", context_trace)
            _write_optional_thinking(outputs_dir / "context_thinking.txt", context_trace)

            table_csv = _build_case_table(inputs_dir=inputs_dir, case=case, input_bundle=input_bundle)
            trend_prompt_base = build_legacy_doe_trend_prompt(
                abm=input_bundle.abm,
                context_response=context_text,
                plot_description=input_bundle.plot_description,
                evidence_mode=case.evidence_mode,
                table_csv=table_csv,
                enabled=enabled,
            )

            image_b64: str | None = None
            image_path: Path | None = None
            if case.evidence_mode in {"plot", "plot+table"}:
                image_path = inputs_dir / "trend_evidence_plot.png"
                shutil.copy2(input_bundle.plot_path, image_path)
                image_b64 = encode_image(image_path)

            trend_text, trend_trace, trend_prompt = _invoke_structured_smoke_text(
                adapter=adapter,
                model=model,
                prompt=trend_prompt_base,
                image_b64=image_b64,
            )
            _write_text(inputs_dir / "trend_prompt.txt", trend_prompt)
            _write_json(requests_dir / "trend_request.json", trend_trace["request"])
            _write_text(outputs_dir / "trend_output.txt", trend_text)
            _write_json(outputs_dir / "trend_trace.json", trend_trace)
            _write_optional_thinking(outputs_dir / "trend_thinking.txt", trend_trace)
            _write_json(
                requests_dir / "hyperparameters.json",
                {
                    "context": context_trace["request"],
                    "trend": trend_trace["request"],
                },
            )
            _write_json(
                case_dir / "00_case_summary.json",
                {
                    "case_id": case.case_id,
                    "abm": case.abm,
                    "evidence_mode": case.evidence_mode,
                    "prompt_variant": case.prompt_variant,
                    "model": model,
                    "inputs_dir": str(inputs_dir),
                    "requests_dir": str(requests_dir),
                    "outputs_dir": str(outputs_dir),
                },
            )

            review_rows.append(
                {
                    "case_id": case.case_id,
                    "abm": case.abm,
                    "evidence_mode": case.evidence_mode,
                    "prompt_variant": case.prompt_variant,
                    "model": model,
                    "case_summary_path": str(case_dir / "00_case_summary.json"),
                    "context_prompt_path": str(inputs_dir / "context_prompt.txt"),
                    "context_prompt_text": context_prompt,
                    "trend_prompt_path": str(inputs_dir / "trend_prompt.txt"),
                    "trend_prompt_text": trend_prompt,
                    "image_path": str(image_path or ""),
                    "table_csv_path": str(inputs_dir / "trend_evidence_table.csv" if table_csv else ""),
                    "parameters_path": str(inputs_dir / "parameters.txt"),
                    "documentation_path": str(inputs_dir / "documentation.txt"),
                    "hyperparameters_path": str(requests_dir / "hyperparameters.json"),
                    "context_output_path": str(outputs_dir / "context_output.txt"),
                    "context_output_text": context_text,
                    "trend_output_path": str(outputs_dir / "trend_output.txt"),
                    "trend_output_text": trend_text,
                }
            )
            results.append(
                LocalQwenSampleCaseResult(
                    case_id=case.case_id,
                    abm=case.abm,
                    evidence_mode=case.evidence_mode,
                    prompt_variant=case.prompt_variant,
                    case_dir=case_dir,
                    success=True,
                )
            )
        except Exception as exc:
            failed_case_ids.append(case.case_id)
            _write_text(outputs_dir / "error.txt", str(exc))
            results.append(
                LocalQwenSampleCaseResult(
                    case_id=case.case_id,
                    abm=case.abm,
                    evidence_mode=case.evidence_mode,
                    prompt_variant=case.prompt_variant,
                    case_dir=case_dir,
                    success=False,
                    error=str(exc),
                )
            )

    review_csv_path = output_root / "request_review.csv"
    _write_review_csv(review_csv_path, review_rows)
    finished_at = datetime.now(UTC)
    report_json_path = output_root / "smoke_local_qwen_report.json"
    report_markdown_path = output_root / "smoke_local_qwen_report.md"
    result = LocalQwenSampleSmokeResult(
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        output_root=output_root,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
        review_csv_path=review_csv_path,
        success=not failed_case_ids,
        failed_case_ids=failed_case_ids,
        cases=results,
    )
    _write_json(report_json_path, result.model_dump(mode="json"))
    _write_text(report_markdown_path, _render_report(result))
    return result


def _build_case_table(
    *,
    inputs_dir: Path,
    case: LocalQwenSampleCase,
    input_bundle: LocalQwenCaseInput,
) -> str:
    if case.evidence_mode not in {"table", "plot+table"}:
        return ""
    frame = pd.read_csv(input_bundle.csv_path, sep=";")
    table_csv = build_raw_table_csv(frame=frame, reporter_pattern=input_bundle.reporter_pattern)
    table_path = inputs_dir / "trend_evidence_table.csv"
    _write_text(table_path, table_csv)
    return table_csv


def _invoke_structured_smoke_text(
    *,
    adapter: LLMAdapter,
    model: str,
    prompt: str,
    image_b64: str | None = None,
) -> tuple[str, dict[str, object], str]:
    schema = StructuredSmokeText.model_json_schema()
    prompt_with_schema = (
        f"{prompt}\n\n"
        "Return your final answer as a JSON object that matches this schema exactly:\n"
        f"{json.dumps(schema, indent=2, sort_keys=True)}"
    )
    raw_text, trace = invoke_adapter_with_trace(
        adapter=adapter,
        model=model,
        prompt=prompt_with_schema,
        image_b64=image_b64,
        max_tokens=SMOKE_MAX_TOKENS,
        request_metadata={
            "ollama_num_ctx": SMOKE_OLLAMA_NUM_CTX,
            "ollama_format": schema,
            "preserve_raw_text": True,
        },
    )
    try:
        parsed = StructuredSmokeText.model_validate_json(raw_text)
    except Exception as exc:
        thinking_text = _extract_thinking_text(trace)
        if thinking_text:
            raise ValueError("model returned only thinking without a final structured response") from exc
        raise ValueError("model did not return valid structured JSON for the smoke output") from exc
    final_text = parsed.response_text.strip()
    if not final_text:
        raise ValueError("structured smoke output contained an empty response_text")
    response_block = trace.get("response")
    if isinstance(response_block, dict):
        response_block["parsed_response"] = parsed.model_dump(mode="json")
    return final_text, trace, prompt_with_schema


def _validate_case_inputs(input_bundle: LocalQwenCaseInput) -> None:
    required_paths = (
        input_bundle.csv_path,
        input_bundle.parameters_path,
        input_bundle.documentation_path,
        input_bundle.plot_path,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"missing required smoke input: {path}")
        if path.is_file() and path.stat().st_size <= 0:
            raise ValueError(f"empty required smoke input: {path}")
    if detect_placeholder_signals(input_bundle.documentation_path.read_text(encoding="utf-8", errors="replace")):
        raise ValueError(f"placeholder-like text detected in documentation: {input_bundle.documentation_path}")
    if detect_placeholder_signals(input_bundle.parameters_path.read_text(encoding="utf-8", errors="replace")):
        raise ValueError(f"placeholder-like text detected in parameters: {input_bundle.parameters_path}")
    if detect_placeholder_signals(input_bundle.plot_description):
        raise ValueError("placeholder-like text detected in plot description")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_optional_thinking(path: Path, trace: dict[str, object]) -> None:
    thinking_text = _extract_thinking_text(trace)
    if thinking_text:
        _write_text(path, thinking_text)


def _extract_thinking_text(trace: dict[str, object]) -> str:
    response_block = trace.get("response")
    if not isinstance(response_block, dict):
        return ""
    raw_block = response_block.get("raw")
    if not isinstance(raw_block, dict):
        return ""
    message_block = raw_block.get("message")
    if not isinstance(message_block, dict):
        return ""
    thinking = message_block.get("thinking")
    return str(thinking).strip() if isinstance(thinking, str) else ""


def _write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "case_id",
        "abm",
        "evidence_mode",
        "prompt_variant",
        "model",
        "case_summary_path",
        "context_prompt_path",
        "context_prompt_text",
        "trend_prompt_path",
        "trend_prompt_text",
        "image_path",
        "table_csv_path",
        "parameters_path",
        "documentation_path",
        "hyperparameters_path",
        "context_output_path",
        "context_output_text",
        "trend_output_path",
        "trend_output_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_report(result: LocalQwenSampleSmokeResult) -> str:
    lines = [
        "# Local Qwen Sample Smoke Report",
        "",
        f"- success: `{str(result.success).lower()}`",
        f"- case_count: `{len(result.cases)}`",
        f"- failed_case_count: `{len(result.failed_case_ids)}`",
        f"- review_csv_path: `{result.review_csv_path}`",
        "",
        "| case_id | abm | evidence_mode | prompt_variant | success |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in result.cases:
        lines.append(
            f"| {case.case_id} | {case.abm} | {case.evidence_mode} | {case.prompt_variant} | "
            f"{str(case.success).lower()} |"
        )
    return "\n".join(lines) + "\n"
