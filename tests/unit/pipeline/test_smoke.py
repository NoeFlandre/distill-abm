import json
from pathlib import Path

import pandas as pd

from distill_abm.configs.models import PromptsConfig
from distill_abm.llm.adapters.base import LLMAdapter, LLMRequest, LLMResponse
from distill_abm.pipeline.smoke import SmokeSuiteInputs, default_smoke_cases, run_qwen_smoke_suite


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
            model="qwen2.5:latest",
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
    assert len(result.cases) == 9
    assert all(case.status == "ok" for case in result.cases)
    assert all(case.case_manifest_path is not None for case in result.cases)

    payload = json.loads(result.report_json_path.read_text(encoding="utf-8"))
    assert payload["model"] == "qwen2.5:latest"
    assert len(payload["cases"]) == 9
    assert payload["sweep_status"] == "ok"
    assert payload["doe_status"] == "ok"


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
            model="qwen2.5:latest",
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
