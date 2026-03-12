"""Response-bundle row builders shared by smoke reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path

from distill_abm.pipeline.smoke_types import SmokeCaseResult, SmokeSuiteInputs


def dict_block(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}


def stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def extract_metadata_blocks(metadata_payload: dict[str, object]) -> tuple[
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
    inputs_block = dict_block(payload.get("inputs"))
    llm_block = dict_block(payload.get("llm"))
    request_block = dict_block(llm_block.get("request"))
    prompts_block = dict_block(payload.get("prompts"))
    responses_block = dict_block(payload.get("responses"))
    artifacts_block = dict_block(payload.get("artifacts"))
    scores_block = dict_block(payload.get("scores"))
    reference_block = dict_block(scores_block.get("reference"))
    reproducibility_block = dict_block(payload.get("reproducibility"))
    summarizers_block = dict_block(payload.get("summarizers"))
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


def flatten_score_fields(scores: dict[str, object], prefix: str) -> dict[str, str]:
    return {
        f"{prefix}_token_f1": stringify(scores.get("token_f1")),
        f"{prefix}_bleu": stringify(scores.get("bleu")),
        f"{prefix}_meteor": stringify(scores.get("meteor")),
        f"{prefix}_rouge1": stringify(scores.get("rouge1")),
        f"{prefix}_rouge2": stringify(scores.get("rouge2")),
        f"{prefix}_rouge_l": stringify(scores.get("rouge_l")),
        f"{prefix}_flesch_reading_ease": stringify(scores.get("flesch_reading_ease")),
    }


def build_case_response_row(
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
        prompt_text = stringify(prompts_block.get("context_prompt"))
        response_text = stringify(responses_block.get("context_response"))
        response_path_value = response_path if response_path is not None else case_result.context_response_path
        prompt_signature = stringify(reproducibility_block.get("context_prompt_signature"))
        prompt_length = stringify(reproducibility_block.get("context_prompt_length"))
    else:
        prompt_text = stringify(prompts_block.get("trend_prompt"))
        response_text = stringify(responses_block.get("trend_full_response"))
        response_path_value = response_path if response_path is not None else case_result.trend_full_response_path
        prompt_signature = stringify(reproducibility_block.get("trend_prompt_signature"))
        prompt_length = stringify(reproducibility_block.get("trend_prompt_length"))

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


def build_fallback_error_row(case_result: SmokeCaseResult, smoke_inputs: SmokeSuiteInputs) -> list[dict[str, str]]:
    return [
        {
            "run_output_dir": str(smoke_inputs.output_dir),
            "case_id": case_result.case.case_id,
            "response_kind": "context",
            "case_status": case_result.status,
            "resumed_from_existing": str(case_result.resumed_from_existing),
            "provider": "",
            "model": smoke_inputs.model,
            "precision": "",
            "runtime_providers_used": "",
            "runtime_precisions_used": "",
            "temperature": "",
            "max_tokens": "",
            "max_retries": "",
            "retry_backoff_seconds": "",
            "evidence_mode": case_result.case.evidence_mode,
            "text_source_mode": case_result.case.text_source_mode,
            "enabled_style_features": stringify(case_result.case.enabled_style_features),
            "summarizers": stringify(case_result.case.summarizers or smoke_inputs.summarizers),
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
            "error": stringify(case_result.error),
            "inputs_json": "",
            "llm_json": "",
            "scores_json": "",
            "reproducibility_json": "",
            "summarizers_json": "",
        }
    ]
