from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from distill_abm.configs.models import PromptsConfig
from distill_abm.llm.adapters.base import LLMAdapter, LLMRequest, LLMResponse
from distill_abm.pipeline import smoke as smoke_module
from distill_abm.pipeline.smoke import (
    SmokeCase,
    SmokeCaseResult,
    SmokeSuiteInputs,
    default_branch_smoke_cases,
    default_smoke_cases,
    run_qwen_smoke_suite,
)


class SmokeFakeAdapter(LLMAdapter):
    provider = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(provider="fake", model=request.model, text=f"resp-{self.calls}", raw={})


def test_default_smoke_cases_cover_full_matrix() -> None:
    cases = default_smoke_cases()
    assert len(cases) == 6
    assert {case.evidence_mode for case in cases} == {"plot", "table", "plot+table"}
    assert {case.text_source_mode for case in cases} == {"summary_only", "full_text_only"}


def test_default_branch_smoke_cases_cover_three_variants() -> None:
    cases = default_branch_smoke_cases()
    assert len(cases) == 3
    assert {case.case_id for case in cases} == {
        "branch-role-full-text",
        "branch-insights-summary-t5",
        "branch-role-insights-summary-longformer",
    }


def test_run_qwen_smoke_suite_writes_matrix_and_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1;mean-incum-2\n0;1;2\n1;2;3\n", encoding="utf-8")
    params = tmp_path / "params.txt"
    params.write_text("param=1\n", encoding="utf-8")
    docs = tmp_path / "docs.txt"
    docs.write_text("documentation block\n", encoding="utf-8")

    doe_input_csv = tmp_path / "doe.csv"
    pd.DataFrame(
        {
            "Model": ["Qwen", "Qwen", "Qwen", "Qwen"],
            "WithExamples": ["Yes", "No", "Yes", "No"],
            "BLEU": [0.4, 0.2, 0.45, 0.25],
        }
    ).to_csv(doe_input_csv, index=False)

    prompts = PromptsConfig(
        context_prompt="Context {parameters} {documentation}",
        trend_prompt="Trend {description} {context}",
        coverage_eval_prompt="Coverage score: 4. {summary}. {source}",
        faithfulness_eval_prompt="Faithfulness score: 4. {summary}. {source}",
        style_features={"role": "ROLE", "insights": "INSIGHTS"},
    )
    adapter = SmokeFakeAdapter()
    result = run_qwen_smoke_suite(
        inputs=SmokeSuiteInputs(
            csv_path=csv_path,
            parameters_path=params,
            documentation_path=docs,
            output_dir=tmp_path / "smoke",
            model="qwen3.5:0.8b",
            metric_pattern="mean-incum",
            metric_description="weekly milk trend",
            plot_description="plot description text",
            allow_summary_fallback=True,
        ),
        prompts=prompts,
        adapter=adapter,
        run_qualitative=False,
        doe_input_csv=doe_input_csv,
        run_sweep=True,
    )

    assert result.success is True
    assert result.report_markdown_path.exists()
    assert result.report_json_path.exists()
    assert result.run_master_csv_path is not None and result.run_master_csv_path.exists()
    assert result.global_master_csv_path is not None and result.global_master_csv_path.exists()
    assert len(result.cases) == 6
    assert result.doe_status == "ok"
    assert result.sweep_status == "ok"


def test_run_qwen_smoke_suite_resumes_existing_successful_case_without_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1\n0;1\n1;2\n", encoding="utf-8")
    params = tmp_path / "params.txt"
    params.write_text("param=1\n", encoding="utf-8")
    docs = tmp_path / "docs.txt"
    docs.write_text("documentation block\n", encoding="utf-8")
    prompts = PromptsConfig(
        context_prompt="Context {parameters} {documentation}",
        trend_prompt="Trend {description} {context}",
    )
    case = SmokeCase(case_id="plot-full-text", evidence_mode="plot", text_source_mode="full_text_only")

    case_dir = tmp_path / "smoke" / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = case_dir / "case_manifest.json"
    manifest_payload = {
        "case": case.model_dump(),
        "status": "ok",
        "output_dir": str(case_dir),
        "resumed_from_existing": False,
        "report_csv": None,
        "plot_path": None,
        "metadata_path": None,
        "context_prompt_path": None,
        "trend_prompt_path": None,
        "stats_table_csv_path": None,
        "context_response_path": None,
        "trend_full_response_path": None,
        "trend_summary_response_path": None,
        "case_rows_csv_path": None,
        "case_manifest_path": str(manifest_path),
        "qualitative": [],
        "error": None,
    }
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    def fail_if_pipeline_called(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("run_pipeline should not execute when resume is enabled and manifest exists")

    monkeypatch.setattr(smoke_module, "run_pipeline", fail_if_pipeline_called)

    result = run_qwen_smoke_suite(
        inputs=SmokeSuiteInputs(
            csv_path=csv_path,
            parameters_path=params,
            documentation_path=docs,
            output_dir=tmp_path / "smoke",
            model="qwen3.5:0.8b",
            metric_pattern="mean-incum",
            metric_description="weekly milk trend",
            plot_description="plot description",
        ),
        prompts=prompts,
        adapter=SmokeFakeAdapter(),
        run_qualitative=False,
        doe_input_csv=None,
        run_sweep=False,
        cases=[case],
        resume_existing=True,
    )

    assert result.success is True
    assert len(result.cases) == 1
    assert result.cases[0].resumed_from_existing is True


def test_build_case_response_rows_serializes_complete_metadata_path(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = {
        "inputs": {
            "evidence_mode": "plot",
            "text_source_mode": "summary_only",
            "enabled_style_features": ["role"],
            "summarizers": ["t5"],
            "csv_path": "sim.csv",
            "parameters_path": "params.txt",
            "documentation_path": "docs.txt",
        },
        "llm": {
            "provider": "fake",
            "model": "fake-model",
            "request": {
                "temperature": 0.5,
                "max_tokens": 1000,
                "max_retries": 2,
                "retry_backoff_seconds": 0.1,
            },
        },
        "prompts": {
            "context_prompt": "context prompt text",
            "trend_prompt": "trend prompt text",
        },
        "responses": {
            "context_response": "context response text",
            "trend_full_response": "trend response text",
        },
        "artifacts": {
            "trend_evidence_image_path": "plot.png",
            "plot_path": "plot.png",
            "stats_table_csv_path": "stats.csv",
            "report_csv": "report.csv",
        },
        "scores": {
            "selected_scores": {
                "token_f1": 0.5,
                "bleu": 0.6,
                "meteor": 0.7,
                "rouge1": 0.8,
                "rouge2": 0.9,
                "rouge_l": 1.0,
                "flesch_reading_ease": 70.0,
            },
            "full_scores": {
                "token_f1": 0.4,
                "bleu": 0.3,
                "meteor": 0.2,
                "rouge1": 0.1,
                "rouge2": 0.2,
                "rouge_l": 0.3,
                "flesch_reading_ease": 60.0,
            },
            "summary_scores": {
                "token_f1": 0.9,
                "bleu": 0.8,
                "meteor": 0.7,
                "rouge1": 0.6,
                "rouge2": 0.5,
                "rouge_l": 0.4,
                "flesch_reading_ease": 65.0,
            },
            "reference": {
                "path": "ground_truth.txt",
                "source": "human_ground_truth_file",
                "text": "reference text",
            },
        },
        "reproducibility": {
            "context_prompt_signature": "ctx_sig",
            "trend_prompt_signature": "trend_sig",
            "context_prompt_length": 5,
            "trend_prompt_length": 6,
            "trend_summary_present": False,
        },
        "summarizers": {},
    }

    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    case = SmokeCase(case_id="plot-summary", evidence_mode="plot", text_source_mode="summary_only")
    case_result = SmokeCaseResult(
        case=case,
        status="ok",
        output_dir=tmp_path,
        metadata_path=metadata_path,
    )
    rows = smoke_module._build_case_response_rows(
        case_result=case_result,
        smoke_inputs=SmokeSuiteInputs(
            csv_path=Path("sim.csv"),
            parameters_path=Path("params.txt"),
            documentation_path=Path("docs.txt"),
            output_dir=Path("out"),
            model="model",
            metric_pattern="mean",
            metric_description="desc",
        ),
    )

    assert len(rows) == 2
    assert rows[0]["text_source_mode"] == "summary_only"
    assert rows[0]["summarizers"] == '[\"t5\"]'


def test_build_case_response_rows_returns_fallback_row_for_invalid_metadata(tmp_path: Path) -> None:
    metadata_path = tmp_path / "pipeline_run_metadata.json"
    metadata_path.write_text("{", encoding="utf-8")
    case = SmokeCase(case_id="plot-summary", evidence_mode="plot", text_source_mode="summary_only")
    case_result = SmokeCaseResult(
        case=case,
        status="failed",
        output_dir=tmp_path,
        metadata_path=metadata_path,
        error="boom",
    )

    rows = smoke_module._build_case_response_rows(
        case_result=case_result,
        smoke_inputs=SmokeSuiteInputs(
            csv_path=Path("sim.csv"),
            parameters_path=Path("params.txt"),
            documentation_path=Path("docs.txt"),
            output_dir=Path("out"),
            model="model",
            metric_pattern="mean",
            metric_description="desc",
        ),
    )
    assert len(rows) == 1
    assert rows[0]["case_status"] == "failed"
    assert rows[0]["error"] == "boom"


def test_build_case_response_rows_handles_missing_metadata_file(tmp_path: Path) -> None:
    """Test that _build_case_response_rows handles missing metadata file gracefully."""
    case = SmokeCase(case_id="plot-summary", evidence_mode="plot", text_source_mode="summary_only")
    # No metadata file at all
    case_result = SmokeCaseResult(
        case=case,
        status="ok",
        output_dir=tmp_path,
        metadata_path=None,  # No metadata file
    )

    rows = smoke_module._build_case_response_rows(
        case_result=case_result,
        smoke_inputs=SmokeSuiteInputs(
            csv_path=Path("sim.csv"),
            parameters_path=Path("params.txt"),
            documentation_path=Path("docs.txt"),
            output_dir=Path("out"),
            model="model",
            metric_pattern="mean",
            metric_description="desc",
        ),
    )

    # Should still produce rows with default values
    assert len(rows) >= 1
    assert rows[0]["case_id"] == "plot-summary"
