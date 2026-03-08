from __future__ import annotations

from pathlib import Path

from distill_abm.pipeline.smoke_response_bundle import (
    build_case_response_row,
    build_fallback_error_row,
    extract_metadata_blocks,
    flatten_score_fields,
)
from distill_abm.pipeline.smoke_types import SmokeCase, SmokeCaseResult, SmokeSuiteInputs


def test_extract_metadata_blocks_returns_dicts_for_missing_sections() -> None:
    blocks = extract_metadata_blocks({"inputs": {"csv_path": "sim.csv"}})

    assert blocks[0]["csv_path"] == "sim.csv"
    assert blocks[1] == {}
    assert blocks[9] == {}


def test_flatten_score_fields_stringifies_known_metrics() -> None:
    flattened = flatten_score_fields({"token_f1": 0.5, "bleu": 0.6}, "selected")

    assert flattened["selected_token_f1"] == "0.5"
    assert flattened["selected_bleu"] == "0.6"
    assert flattened["selected_rouge_l"] == ""


def test_build_case_response_row_uses_context_fields() -> None:
    case = SmokeCase(case_id="case-1", evidence_mode="plot", text_source_mode="summary_only")
    case_result = SmokeCaseResult(case=case, status="ok", output_dir=Path("out"))
    row = build_case_response_row(
        base={"case_id": "case-1"},
        case_result=case_result,
        metadata_fields=(
            {"context_prompt": "ctx prompt", "trend_prompt": "trend prompt"},
            {"context_prompt_signature": "sig", "context_prompt_length": 10},
            {"context_response": "ctx resp", "trend_full_response": "trend resp"},
        ),
        response_kind="context",
        prompt_path=Path("context_prompt.txt"),
        response_path=Path("context_output.txt"),
    )

    assert row["prompt_text"] == "ctx prompt"
    assert row["response_text"] == "ctx resp"
    assert row["prompt_signature"] == "sig"
    assert row["response_path"] == "context_output.txt"


def test_build_fallback_error_row_preserves_case_error() -> None:
    case = SmokeCase(case_id="case-1", evidence_mode="plot", text_source_mode="summary_only")
    case_result = SmokeCaseResult(case=case, status="failed", output_dir=Path("out"), error="boom")
    rows = build_fallback_error_row(
        case_result=case_result,
        smoke_inputs=SmokeSuiteInputs(
            csv_path=Path("sim.csv"),
            parameters_path=Path("params.txt"),
            documentation_path=Path("docs.txt"),
            output_dir=Path("out"),
            model="model",
            metric_pattern="metric",
            metric_description="desc",
        ),
    )

    assert len(rows) == 1
    assert rows[0]["case_status"] == "failed"
    assert rows[0]["error"] == "boom"
