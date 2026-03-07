from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from distill_abm.llm.adapters.base import LLMAdapter, LLMRequest, LLMResponse
from distill_abm.pipeline.local_qwen_sample_smoke import (
    LocalQwenCaseInput,
    LocalQwenSampleCase,
    default_local_qwen_sample_cases,
    run_local_qwen_sample_smoke,
)


class FakeAdapter(LLMAdapter):
    provider = "ollama"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        text = f"response-{len(self.requests)}"
        return LLMResponse(
            provider=self.provider,
            model=request.model,
            text=json.dumps({"response_text": text}),
            raw={"ok": True, "message": {"thinking": f"thinking-{len(self.requests)}"}},
        )


class ThinkingOnlyAdapter(LLMAdapter):
    provider = "ollama"

    def complete(self, request: LLMRequest) -> LLMResponse:
        _ = request
        return LLMResponse(
            provider=self.provider,
            model="qwen3.5:0.8b",
            text="",
            raw={
                "message": {"content": "", "thinking": "reasoning only"},
                "done_reason": "length",
                "eval_count": 1024,
                "prompt_eval_count": 300,
            },
        )


def _write_case_input(tmp_path: Path) -> LocalQwenCaseInput:
    csv_path = tmp_path / "simulation.csv"
    pd.DataFrame(
        {
            "[step]": [0, 1, 2],
            "metric-a": [10.0, 11.0, 12.0],
        }
    ).to_csv(csv_path, index=False, sep=";")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("number-households: 60\n", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("Milk model documentation.\n", encoding="utf-8")
    plot_path = tmp_path / "plot_1.png"
    plot_path.write_bytes(b"png")
    return LocalQwenCaseInput(
        abm="milk_consumption",
        csv_path=csv_path,
        parameters_path=parameters_path,
        documentation_path=documentation_path,
        reporter_pattern="metric-a",
        plot_description="The attachment is the plot representing whole-milk consumption per agent.",
        plot_path=plot_path,
    )


def test_default_local_qwen_sample_cases_cover_prompt_variants_and_evidence_modes() -> None:
    cases = default_local_qwen_sample_cases()

    assert len(cases) == 8
    assert {case.prompt_variant for case in cases} == {
        "none",
        "role",
        "insights",
        "example",
        "role+example",
        "role+insights",
        "insights+example",
        "all_three",
    }
    assert {case.evidence_mode for case in cases} == {"plot", "table", "plot+table"}
    assert {case.abm for case in cases} == {"fauna", "grazing", "milk_consumption"}


def test_run_local_qwen_sample_smoke_writes_review_friendly_case_artifacts(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    result = run_local_qwen_sample_smoke(
        case_inputs={"milk_consumption": _write_case_input(tmp_path)},
        adapter=adapter,
        model="qwen3.5:0.8b",
        output_root=tmp_path / "smoke",
        cases=(
            LocalQwenSampleCase(
                case_id="milk_plot_case",
                abm="milk_consumption",
                evidence_mode="plot",
                prompt_variant="none",
            ),
        ),
    )

    assert result.success is True
    assert len(result.cases) == 1
    case_dir = result.cases[0].case_dir
    assert (case_dir / "00_case_summary.json").exists()
    assert (case_dir / "01_inputs" / "context_prompt.txt").exists()
    assert (case_dir / "03_outputs" / "context_output.txt").read_text(encoding="utf-8") == "response-1"
    assert (case_dir / "03_outputs" / "context_thinking.txt").read_text(encoding="utf-8") == "thinking-1"
    assert (case_dir / "01_inputs" / "trend_prompt.txt").exists()
    assert (case_dir / "03_outputs" / "trend_output.txt").read_text(encoding="utf-8") == "response-2"
    assert (case_dir / "03_outputs" / "trend_thinking.txt").read_text(encoding="utf-8") == "thinking-2"
    assert (case_dir / "01_inputs" / "trend_evidence_plot.png").exists()
    assert not (case_dir / "01_inputs" / "trend_evidence_table.csv").exists()
    assert (case_dir / "02_requests" / "hyperparameters.json").exists()
    assert (case_dir / "02_requests" / "context_request.json").exists()
    assert (case_dir / "02_requests" / "trend_request.json").exists()
    context_trace = json.loads((case_dir / "03_outputs" / "context_trace.json").read_text(encoding="utf-8"))
    trend_trace = json.loads((case_dir / "03_outputs" / "trend_trace.json").read_text(encoding="utf-8"))
    assert context_trace["request"]["model"] == "qwen3.5:0.8b"
    assert context_trace["request"]["max_tokens"] == 32768
    assert context_trace["request"]["metadata"]["ollama_num_ctx"] == 131072
    assert context_trace["request"]["metadata"]["ollama_format"]["type"] == "object"
    assert trend_trace["request"]["image_attached"] is True
    assert "response-1" in (case_dir / "01_inputs" / "trend_prompt.txt").read_text(encoding="utf-8")
    assert result.review_csv_path.exists()
    review_csv = result.review_csv_path.read_text(encoding="utf-8")
    assert "case_summary_path" in review_csv
    assert "context_prompt_text" in review_csv
    assert "trend_output_text" in review_csv


def test_run_local_qwen_sample_smoke_resume_reuses_successful_case(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    case = LocalQwenSampleCase(
        case_id="milk_plot_case",
        abm="milk_consumption",
        evidence_mode="plot",
        prompt_variant="none",
    )
    output_root = tmp_path / "smoke"
    run_local_qwen_sample_smoke(
        case_inputs={"milk_consumption": _write_case_input(tmp_path)},
        adapter=adapter,
        model="qwen3.5:0.8b",
        output_root=output_root,
        cases=(case,),
    )

    resumed_adapter = FakeAdapter()
    resumed = run_local_qwen_sample_smoke(
        case_inputs={"milk_consumption": _write_case_input(tmp_path)},
        adapter=resumed_adapter,
        model="qwen3.5:0.8b",
        output_root=output_root,
        cases=(case,),
        resume_existing=True,
    )

    assert resumed.success is True
    assert resumed.cases[0].resumed_from_existing is True
    assert resumed_adapter.requests == []


def test_run_local_qwen_sample_smoke_preserves_partial_artifacts_on_thinking_only_failure(tmp_path: Path) -> None:
    result = run_local_qwen_sample_smoke(
        case_inputs={"milk_consumption": _write_case_input(tmp_path)},
        adapter=ThinkingOnlyAdapter(),
        model="qwen3.5:0.8b",
        output_root=tmp_path / "smoke",
        cases=(
            LocalQwenSampleCase(
                case_id="milk_plot_case",
                abm="milk_consumption",
                evidence_mode="plot",
                prompt_variant="none",
            ),
        ),
    )

    assert result.success is False
    case_dir = result.cases[0].case_dir
    assert (case_dir / "01_inputs" / "context_prompt.txt").exists()
    assert (case_dir / "02_requests" / "context_request.json").exists()
    assert (case_dir / "03_outputs" / "context_trace.json").exists()
    assert (case_dir / "03_outputs" / "context_thinking.txt").read_text(encoding="utf-8") == "reasoning only"
    error_text = (case_dir / "03_outputs" / "error.txt").read_text(encoding="utf-8")
    assert "exhausted max_tokens on thinking" in error_text
