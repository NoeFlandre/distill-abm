import json
from pathlib import Path

import pandas as pd
import pytest

from distill_abm.configs.models import PromptsConfig
from distill_abm.llm.adapters.base import LLMAdapter, LLMRequest, LLMResponse
from distill_abm.pipeline.smoke import (
    SmokeCase,
    SmokeCaseResult,
    SmokeSuiteInputs,
    _build_case_response_rows,
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
        return LLMResponse(
            provider="fake",
            model=request.model,
            text=f"fake-response-{self.calls}",
            raw={},
        )


def test_default_smoke_cases_cover_full_matrix() -> None:
    cases = default_smoke_cases()
    assert len(cases) == 9

    observed = {(case.evidence_mode, case.summarization_mode, case.score_on) for case in cases}
    expected = {
        ("plot", "full", "full"),
        ("plot", "summary", "summary"),
        ("plot", "both", "both"),
        ("table-csv", "full", "full"),
        ("table-csv", "summary", "summary"),
        ("table-csv", "both", "both"),
        ("plot+table", "full", "full"),
        ("plot+table", "summary", "summary"),
        ("plot+table", "both", "both"),
    }
    assert observed == expected


def test_default_branch_smoke_cases_cover_three_variants() -> None:
    cases = default_branch_smoke_cases()
    assert len(cases) == 3
    assert {case.case_id for case in cases} == {
        "branch-role-full",
        "branch-insights-summary-t5",
        "branch-role-insights-summary-longformer",
    }
    assert cases[0].enabled_style_features == ("role",)
    assert cases[1].enabled_style_features == ("insights",)
    assert cases[1].additional_summarizers == ("t5",)
    assert cases[2].additional_summarizers == ("longformer_ext",)


def test_run_qwen_smoke_suite_writes_matrix_and_reports(tmp_path: Path) -> None:
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
    assert len(result.cases) == 9
    assert all(case.status == "ok" for case in result.cases)
    assert all(case.case_manifest_path is not None for case in result.cases)
    assert all(case.case_rows_csv_path is not None for case in result.cases)
    assert all((case.output_dir / "responses" / "context" / "response_bundle.csv").exists() for case in result.cases)
    assert all((case.output_dir / "responses" / "trend" / "response_bundle.csv").exists() for case in result.cases)

    payload = json.loads(result.report_json_path.read_text(encoding="utf-8"))
    assert payload["model"] == "qwen3.5:0.8b"
    assert len(payload["cases"]) == 9
    assert payload["sweep_status"] == "ok"
    assert payload["doe_status"] == "ok"
    assert payload["qualitative_policy"] == "debug_same_model"


def test_run_qwen_smoke_suite_records_pipeline_failure(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
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

    def failing_run_pipeline(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr("distill_abm.pipeline.smoke.run_pipeline", failing_run_pipeline)

    result = run_qwen_smoke_suite(
        inputs=SmokeSuiteInputs(
            csv_path=csv_path,
            parameters_path=params,
            documentation_path=docs,
            output_dir=tmp_path / "smoke",
            model="qwen3.5:0.8b",
            metric_pattern="mean-incum",
            metric_description="weekly milk trend",
        ),
        prompts=prompts,
        adapter=SmokeFakeAdapter(),
        run_qualitative=False,
        doe_input_csv=None,
        run_sweep=False,
    )

    assert result.success is False
    assert result.failed_cases
    assert result.report_markdown_path.exists()
    assert "boom" in result.report_json_path.read_text(encoding="utf-8")


def test_run_qwen_smoke_suite_uses_multiple_images_for_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1\n0;1\n1;2\n", encoding="utf-8")
    params = tmp_path / "params.txt"
    params.write_text("param=1\n", encoding="utf-8")
    docs = tmp_path / "docs.txt"
    docs.write_text("documentation block\n", encoding="utf-8")
    doe_input_csv = tmp_path / "doe.csv"
    pd.DataFrame({"Model": ["Qwen", "Qwen"], "WithExamples": ["Yes", "No"], "BLEU": [0.4, 0.2]}).to_csv(
        doe_input_csv, index=False
    )

    captured: dict[str, object] = {}

    def fake_run_pipeline_sweep(*, image_paths, plot_descriptions, **kwargs):  # type: ignore[no-untyped-def]
        captured["count"] = len(image_paths)
        captured["descriptions"] = list(plot_descriptions)
        output_csv = kwargs["output_csv"]
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_csv.write_text("Combination Description,Context Prompt,Context Response\n", encoding="utf-8")
        return output_csv

    monkeypatch.setattr("distill_abm.pipeline.smoke.run_pipeline_sweep", fake_run_pipeline_sweep)
    prompts = PromptsConfig(
        context_prompt="Context {parameters} {documentation}",
        trend_prompt="Trend {description} {context}",
        style_features={"role": "ROLE", "example": "EXAMPLE", "insights": "INSIGHTS"},
    )
    result = run_qwen_smoke_suite(
        inputs=SmokeSuiteInputs(
            csv_path=csv_path,
            parameters_path=params,
            documentation_path=docs,
            output_dir=tmp_path / "smoke",
            model="qwen3.5:0.8b",
            metric_pattern="mean-incum",
            metric_description="weekly milk trend",
            sweep_plot_descriptions=["plot-1", "plot-2", "plot-3", "plot-4", "plot-5"],
        ),
        prompts=prompts,
        adapter=SmokeFakeAdapter(),
        run_qualitative=False,
        doe_input_csv=doe_input_csv,
        run_sweep=True,
    )

    assert result.success is True
    assert captured["count"] == 5
    assert captured["descriptions"] == ["plot-1", "plot-2", "plot-3", "plot-4", "plot-5"]


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
    case = SmokeCase(case_id="plot-full-full", evidence_mode="plot", summarization_mode="full", score_on="full")
    case_dir = tmp_path / "smoke" / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    case_manifest = case_dir / "case_manifest.json"
    case_manifest.write_text(
        json.dumps(
            {
                "case": case.model_dump(),
                "status": "ok",
                "output_dir": str(case_dir),
                "report_csv": str(case_dir / "report.csv"),
                "plot_path": str(case_dir / "plot.png"),
                "metadata_path": str(case_dir / "pipeline_run_metadata.json"),
                "context_prompt_path": str(case_dir / "context_prompt.txt"),
                "trend_prompt_path": str(case_dir / "trend_prompt.txt"),
                "stats_table_csv_path": None,
                "context_response_path": str(case_dir / "context_response.txt"),
                "trend_full_response_path": str(case_dir / "trend_full_response.txt"),
                "trend_summary_response_path": None,
                "case_manifest_path": str(case_manifest),
                "qualitative": [],
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    def should_not_run_pipeline(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("run_pipeline should not execute when resuming a successful case")

    monkeypatch.setattr("distill_abm.pipeline.smoke.run_pipeline", should_not_run_pipeline)

    result = run_qwen_smoke_suite(
        inputs=SmokeSuiteInputs(
            csv_path=csv_path,
            parameters_path=params,
            documentation_path=docs,
            output_dir=tmp_path / "smoke",
            model="qwen3.5:0.8b",
            metric_pattern="mean-incum",
            metric_description="weekly milk trend",
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
    assert result.cases[0].status == "ok"
    assert result.cases[0].resumed_from_existing is True


def test_build_case_response_rows_serializes_complete_metadata_path() -> None:
    metadata_path = Path("metadata.json")
    metadata = {
        "inputs": {
            "evidence_mode": "plot",
            "summarization_mode": "both",
            "score_on": "both",
            "enabled_style_features": ["role"],
            "additional_summarizers": ["t5"],
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
    case = SmokeCase(case_id="plot-full-full", evidence_mode="plot", summarization_mode="full", score_on="full")
    case_result = SmokeCaseResult(
        case=case,
        status="ok",
        output_dir=Path("outputs/case"),
        report_csv=Path("outputs/report.csv"),
        plot_path=Path("outputs/plot.png"),
        metadata_path=metadata_path,
        context_prompt_path=Path("outputs/context_prompt.txt"),
        trend_prompt_path=Path("outputs/trend_prompt.txt"),
        context_response_path=Path("outputs/context_response.txt"),
        trend_full_response_path=Path("outputs/trend_full_response.txt"),
        case_manifest_path=Path("outputs/case_manifest.json"),
    )
    inputs = SmokeSuiteInputs(
        csv_path=Path("sim.csv"),
        parameters_path=Path("params.txt"),
        documentation_path=Path("docs.txt"),
        output_dir=Path("outputs"),
        model="fake-model",
        metric_pattern="mean-incum",
        metric_description="test",
    )

    rows = _build_case_response_rows(case_result=case_result, smoke_inputs=inputs)

    assert len(rows) == 2
    context_row = rows[0]
    trend_row = rows[1]
    assert context_row["response_kind"] == "context"
    assert trend_row["response_kind"] == "trend"
    assert context_row["provider"] == "fake"
    assert context_row["temperature"] == "0.5"
    assert context_row["response_text"] == "context response text"
    assert trend_row["response_text"] == "trend response text"
    assert context_row["selected_token_f1"] == "0.5"
    assert trend_row["summary_rouge1"] == "0.6"


def test_build_case_response_rows_returns_fallback_row_for_invalid_metadata(tmp_path: Path) -> None:
    metadata_path = tmp_path / "pipeline_run_metadata.json"
    metadata_path.write_text("{", encoding="utf-8")
    case = SmokeCase(case_id="plot-full-full", evidence_mode="plot", summarization_mode="full", score_on="full")
    case_result = SmokeCaseResult(
        case=case,
        status="ok",
        output_dir=tmp_path,
        metadata_path=metadata_path,
    )
    inputs = SmokeSuiteInputs(
        csv_path=tmp_path / "sim.csv",
        parameters_path=tmp_path / "params.txt",
        documentation_path=tmp_path / "docs.txt",
        output_dir=tmp_path,
        model="fake-model",
        metric_pattern="mean-incum",
        metric_description="test",
    )

    rows = _build_case_response_rows(case_result=case_result, smoke_inputs=inputs)

    assert len(rows) == 1
    assert rows[0]["response_kind"] == "context"
    assert rows[0]["provider"] == ""
    assert rows[0]["case_status"] == "ok"


def test_build_case_response_rows_stringifies_missing_prompt_and_response_fields(tmp_path: Path) -> None:
    metadata_path = tmp_path / "pipeline_run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "inputs": {},
                "llm": {"provider": "fake", "model": "fake-model", "request": {}},
                "artifacts": {},
                "scores": {"selected_scores": {}},
                "reproducibility": {},
            }
        ),
        encoding="utf-8",
    )

    case = SmokeCase(case_id="plot-full-full", evidence_mode="plot", summarization_mode="full", score_on="full")
    case_result = SmokeCaseResult(
        case=case,
        status="ok",
        output_dir=tmp_path,
        metadata_path=metadata_path,
    )
    inputs = SmokeSuiteInputs(
        csv_path=tmp_path / "sim.csv",
        parameters_path=tmp_path / "params.txt",
        documentation_path=tmp_path / "docs.txt",
        output_dir=tmp_path,
        model="fake-model",
        metric_pattern="mean-incum",
        metric_description="test",
    )

    context_row, trend_row = _build_case_response_rows(case_result=case_result, smoke_inputs=inputs)

    assert context_row["prompt_text"] == ""
    assert context_row["prompt_path"] == ""
    assert context_row["response_text"] == ""
    assert context_row["response_length"] == "0"
    assert trend_row["prompt_text"] == ""
    assert trend_row["response_text"] == ""
