from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd

from distill_abm.llm.adapters.base import LLMAdapter, LLMProviderError, LLMRequest, LLMResponse
from distill_abm.pipeline.local_qwen_sample_smoke import (
    LocalQwenCaseInput,
    LocalQwenSampleCase,
    _invoke_structured_smoke_text,
    default_local_qwen_sample_cases,
    run_local_qwen_sample_smoke,
)


class FakeAdapter(LLMAdapter):
    provider = "openrouter"

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

    @property
    def context_request_count(self) -> int:
        return sum(
            1 for request in self.requests if "Your goal is to explain an agent-based model." in request.user_prompt()
        )

    @property
    def trend_request_count(self) -> int:
        return sum(1 for request in self.requests if "We have a " in request.user_prompt())


class ThinkingOnlyAdapter(LLMAdapter):
    provider = "openrouter"

    def complete(self, request: LLMRequest) -> LLMResponse:
        _ = request
        return LLMResponse(
            provider=self.provider,
            model="qwen/qwen3.5-27b",
            text="",
            raw={
                "message": {"content": "", "thinking": "reasoning only"},
                "done_reason": "length",
                "eval_count": 1024,
                "prompt_eval_count": 300,
            },
        )


class TruncatedStructuredAdapter(LLMAdapter):
    provider = "openrouter"

    def complete(self, request: LLMRequest) -> LLMResponse:
        _ = request
        return LLMResponse(
            provider=self.provider,
            model="qwen/qwen3.5-27b",
            text=json.dumps({"response_text": "partial but not trustworthy"}),
            raw={
                "message": {"content": '{"response_text":"partial but not trustworthy"}', "thinking": "reasoning"},
                "done_reason": "length",
                "eval_count": 2048,
                "prompt_eval_count": 300,
            },
        )


class FailFirstThenCountAdapter(LLMAdapter):
    provider = "openrouter"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMResponse(
                provider=self.provider,
                model="qwen/qwen3.5-27b",
                text="",
                raw={
                    "message": {"content": "", "thinking": "reasoning only"},
                    "done_reason": "length",
                    "eval_count": 1024,
                    "prompt_eval_count": 300,
                },
            )
        return LLMResponse(
            provider=self.provider,
            model="qwen/qwen3.5-27b",
            text=json.dumps({"response_text": f"response-{len(self.requests)}"}),
            raw={"message": {"thinking": f"thinking-{len(self.requests)}"}},
        )


class ContextOverflowThenSuccessAdapter(LLMAdapter):
    provider = "openrouter"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMResponse(
                provider=self.provider,
                model=request.model,
                text=json.dumps({"response_text": "context-ok"}),
                raw={"choices": [{"message": {"content": '{"response_text":"context-ok"}'}, "finish_reason": "stop"}]},
            )
        if request.metadata.get("table_downsample_stride") == 1:
            raise ValueError(
                "This model's maximum context length is 131072 tokens. " "Your request has too many input tokens."
            )
        return LLMResponse(
            provider=self.provider,
            model=request.model,
            text=json.dumps({"response_text": "trend-ok"}),
            raw={"choices": [{"message": {"content": '{"response_text":"trend-ok"}'}, "finish_reason": "stop"}]},
        )


class GenericUnavailableAdapter(LLMAdapter):
    provider = "openrouter"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMResponse(
                provider=self.provider,
                model=request.model,
                text=json.dumps({"response_text": "context-ok"}),
                raw={"choices": [{"message": {"content": '{"response_text":"context-ok"}'}, "finish_reason": "stop"}]},
            )
        return LLMResponse(
            provider=self.provider,
            model=request.model,
            text=json.dumps(
                {
                    "response_text": (
                        "The analysis is currently unavailable. "
                        "The requested information cannot be retrieved or interpreted at this time. "
                        "Please try again later or consult additional resources for further details."
                    )
                }
            ),
            raw={"choices": [{"message": {"content": "unavailable"}, "finish_reason": "stop"}]},
        )


class StructuredFailureThenPromptedJsonAdapter(LLMAdapter):
    provider = "openrouter"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMResponse(
                provider=self.provider,
                model=request.model,
                text="",
                raw={
                    "choices": None,
                    "error": {"code": 500, "message": "Internal Server Error"},
                },
            )
        return LLMResponse(
            provider=self.provider,
            model=request.model,
            text='```json\n{"response_text":"fallback-ok"}\n```',
            raw={
                "choices": [{"message": {"content": '```json\n{"response_text":"fallback-ok"}\n```'}}],
            },
        )


class RaisedStructuredFailureThenPromptedJsonAdapter(LLMAdapter):
    provider = "openrouter"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise LLMProviderError(
                "openrouter completion failed: Error code: 500 - {'error': {'message': 'Internal Server Error'}}"
            )
        return LLMResponse(
            provider=self.provider,
            model=request.model,
            text='{"response_text":"fallback-ok-after-error"}',
            raw={"choices": [{"message": {"content": '{"response_text":"fallback-ok-after-error"}'}}]},
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
        model="qwen/qwen3.5-27b",
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
    assert not (case_dir / "01_inputs" / "trend_evidence_table.txt").exists()
    assert (case_dir / "02_requests" / "hyperparameters.json").exists()
    assert (case_dir / "02_requests" / "context_request.json").exists()
    assert (case_dir / "02_requests" / "trend_request.json").exists()
    context_trace = json.loads((case_dir / "03_outputs" / "context_trace.json").read_text(encoding="utf-8"))
    trend_trace = json.loads((case_dir / "03_outputs" / "trend_trace.json").read_text(encoding="utf-8"))
    assert context_trace["request"]["model"] == "qwen/qwen3.5-27b"
    assert context_trace["request"]["max_tokens"] == 32768
    assert context_trace["request"]["metadata"]["ollama_num_ctx"] == 131072
    assert context_trace["request"]["metadata"]["ollama_format"]["type"] == "object"
    assert context_trace["request"]["metadata"]["structured_output_schema"]["type"] == "object"
    assert context_trace["request"]["metadata"]["structured_output_name"] == "structured_smoke_text"
    assert trend_trace["request"]["image_attached"] is True
    trend_prompt_text = (case_dir / "01_inputs" / "trend_prompt.txt").read_text(encoding="utf-8")
    context_prompt_text = (case_dir / "01_inputs" / "context_prompt.txt").read_text(encoding="utf-8")
    assert "response-1" in trend_prompt_text
    assert "Return your final answer as a JSON object" not in context_prompt_text
    assert "Return your final answer as a JSON object" not in trend_prompt_text
    compression_payload = json.loads(
        (case_dir / "01_inputs" / "trend_prompt_compression.json").read_text(encoding="utf-8")
    )
    assert compression_payload["triggered"] is False
    assert compression_payload["compression_count"] == 0
    assert compression_payload["attempt_count"] == 1
    assert compression_payload["attempts"] == [
        {
            "attempt_index": 1,
            "table_downsample_stride": 1,
            "compression_tier": 0,
            "prompt_length": len(trend_prompt_text),
        }
    ]
    run_summary = json.loads(result.prompt_compression_summary_path.read_text(encoding="utf-8"))
    assert run_summary["total_entries"] == 1
    assert run_summary["triggered_entries"] == 0
    assert run_summary["total_compressions"] == 0
    assert run_summary["entries"][0]["case_id"] == "milk_plot_case"
    assert run_summary["entries"][0]["scope"] == "sample_case"
    assert run_summary["entries"][0]["triggered"] is False
    assert result.review_csv_path.exists()
    review_csv = result.review_csv_path.read_text(encoding="utf-8")
    assert "case_summary_path" in review_csv
    assert "context_prompt_text" in review_csv
    assert "trend_output_text" in review_csv


def test_invoke_structured_smoke_text_retries_without_response_format_on_openrouter_empty_error() -> None:
    adapter = StructuredFailureThenPromptedJsonAdapter()

    final_text, trace = _invoke_structured_smoke_text(
        adapter=adapter,
        model="qwen/qwen3.5-27b",
        prompt_with_schema="Explain the ABM.",
        max_tokens=512,
        ollama_num_ctx=131072,
    )

    assert final_text == "fallback-ok"
    assert len(adapter.requests) == 2
    assert "structured_output_schema" in adapter.requests[0].metadata
    assert "structured_output_schema" not in adapter.requests[1].metadata
    assert 'Return only a valid JSON object with exactly one key' in adapter.requests[1].user_prompt()
    fallback = cast(dict[str, object], trace["structured_output_fallback"])
    assert fallback["triggered"] is True
    assert fallback["fallback_mode"] == "prompted_json_without_response_format"


def test_invoke_structured_smoke_text_retries_without_response_format_on_openrouter_provider_error() -> None:
    adapter = RaisedStructuredFailureThenPromptedJsonAdapter()

    final_text, trace = _invoke_structured_smoke_text(
        adapter=adapter,
        model="qwen/qwen3.5-27b",
        prompt_with_schema="Explain the ABM.",
        max_tokens=512,
        ollama_num_ctx=131072,
        max_retries=0,
    )

    assert final_text == "fallback-ok-after-error"
    assert len(adapter.requests) == 2
    assert "structured_output_schema" in adapter.requests[0].metadata
    assert "structured_output_schema" not in adapter.requests[1].metadata
    fallback = cast(dict[str, object], trace["structured_output_fallback"])
    assert fallback["triggered"] is True
    assert "Internal Server Error" in str(fallback["reason"])


def test_run_local_qwen_sample_smoke_resume_reuses_successful_case(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    case = LocalQwenSampleCase(
        case_id="milk_plot_case",
        abm="milk_consumption",
        evidence_mode="plot",
        prompt_variant="none",
    )
    output_root = tmp_path / "smoke"
    first_result = run_local_qwen_sample_smoke(
        case_inputs={"milk_consumption": _write_case_input(tmp_path)},
        adapter=adapter,
        model="qwen/qwen3.5-27b",
        output_root=output_root,
        cases=(case,),
    )

    resumed_adapter = FakeAdapter()
    resumed = run_local_qwen_sample_smoke(
        case_inputs={"milk_consumption": _write_case_input(tmp_path)},
        adapter=resumed_adapter,
        model="qwen/qwen3.5-27b",
        output_root=output_root,
        cases=(case,),
        resume_existing=True,
    )

    assert resumed.success is True
    assert resumed.cases[0].resumed_from_existing is True
    assert resumed.output_root != first_result.output_root
    assert resumed.cases[0].case_dir != first_result.cases[0].case_dir
    assert (output_root / "latest_run.txt").read_text(encoding="utf-8").strip() == str(resumed.output_root)
    assert resumed_adapter.requests == []


def test_run_local_qwen_sample_smoke_preserves_partial_artifacts_on_thinking_only_failure(tmp_path: Path) -> None:
    result = run_local_qwen_sample_smoke(
        case_inputs={"milk_consumption": _write_case_input(tmp_path)},
        adapter=ThinkingOnlyAdapter(),
        model="qwen/qwen3.5-27b",
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


def test_run_local_qwen_sample_smoke_rejects_truncated_structured_output(tmp_path: Path) -> None:
    result = run_local_qwen_sample_smoke(
        case_inputs={"milk_consumption": _write_case_input(tmp_path)},
        adapter=TruncatedStructuredAdapter(),
        model="qwen/qwen3.5-27b",
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
    assert (case_dir / "03_outputs" / "context_trace.json").exists()
    error_text = (case_dir / "03_outputs" / "error.txt").read_text(encoding="utf-8")
    assert "reached max_tokens before completing the structured response" in error_text


def test_run_local_qwen_sample_smoke_can_stop_after_first_failure(tmp_path: Path) -> None:
    adapter = FailFirstThenCountAdapter()
    result = run_local_qwen_sample_smoke(
        case_inputs={"milk_consumption": _write_case_input(tmp_path)},
        adapter=adapter,
        model="qwen/qwen3.5-27b",
        output_root=tmp_path / "smoke",
        cases=(
            LocalQwenSampleCase(
                case_id="milk_plot_case_1",
                abm="milk_consumption",
                evidence_mode="plot",
                prompt_variant="none",
            ),
            LocalQwenSampleCase(
                case_id="milk_plot_case_2",
                abm="milk_consumption",
                evidence_mode="plot",
                prompt_variant="none",
            ),
        ),
        stop_on_failure=True,
    )

    assert result.success is False
    assert len(result.cases) == 1
    assert result.failed_case_ids == ["milk_plot_case_1"]
    assert len(adapter.requests) == 1


def test_run_local_qwen_sample_smoke_compresses_statistical_table_when_prompt_too_large(tmp_path: Path) -> None:
    adapter = ContextOverflowThenSuccessAdapter()
    result = run_local_qwen_sample_smoke(
        case_inputs={"milk_consumption": _write_case_input(tmp_path)},
        adapter=adapter,
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        output_root=tmp_path / "smoke",
        cases=(
            LocalQwenSampleCase(
                case_id="milk_table_case",
                abm="milk_consumption",
                evidence_mode="table",
                prompt_variant="none",
            ),
        ),
        max_retries=0,
    )

    assert result.success is True
    assert len(adapter.requests) == 3
    assert adapter.requests[2].metadata["table_downsample_stride"] == 2
    case_dir = result.cases[0].case_dir
    table_text = (case_dir / "01_inputs" / "trend_evidence_table.txt").read_text(encoding="utf-8")
    assert "Statistical evidence for simulation series matching `metric-a`." in table_text
    compression_payload = json.loads(
        (case_dir / "01_inputs" / "trend_prompt_compression.json").read_text(encoding="utf-8")
    )
    assert compression_payload["triggered"] is True
    assert compression_payload["compression_count"] == 1
    assert compression_payload["attempt_count"] == 2
    assert compression_payload["attempts"][0]["table_downsample_stride"] == 1
    assert compression_payload["attempts"][0]["compression_tier"] == 0
    assert compression_payload["attempts"][1]["table_downsample_stride"] == 2
    assert compression_payload["attempts"][1]["compression_tier"] == 1
    original_prompt = (case_dir / "01_inputs" / "trend_prompt_pre_compression.txt").read_text(encoding="utf-8")
    compressed_prompt = (case_dir / "01_inputs" / "trend_prompt_compressed.txt").read_text(encoding="utf-8")
    final_prompt = (case_dir / "01_inputs" / "trend_prompt.txt").read_text(encoding="utf-8")
    assert original_prompt != compressed_prompt
    assert compressed_prompt == final_prompt
    run_summary = json.loads(result.prompt_compression_summary_path.read_text(encoding="utf-8"))
    assert run_summary["total_entries"] == 1
    assert run_summary["triggered_entries"] == 1
    assert run_summary["total_compressions"] == 1
    assert run_summary["entries"][0]["attempt_count"] == 2


def test_run_local_qwen_sample_smoke_flags_generic_unavailable_output_as_failure(tmp_path: Path) -> None:
    result = run_local_qwen_sample_smoke(
        case_inputs={"milk_consumption": _write_case_input(tmp_path)},
        adapter=GenericUnavailableAdapter(),
        model="nvidia/nemotron-nano-12b-v2-vl:free",
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
    assert result.failed_case_ids == ["milk_plot_case"]
    error_path = result.cases[0].case_dir / "03_outputs" / "error.txt"
    assert error_path.exists()
    assert "generic unavailable" in error_path.read_text(encoding="utf-8").lower()


def test_run_local_qwen_sample_smoke_reuses_context_outputs_for_same_abm_and_role_state(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    case_input = _write_case_input(tmp_path)
    result = run_local_qwen_sample_smoke(
        case_inputs={"milk_consumption": case_input},
        adapter=adapter,
        model="qwen/qwen3.5-27b",
        output_root=tmp_path / "smoke",
        cases=(
            LocalQwenSampleCase(
                case_id="milk_role_plot_case",
                abm="milk_consumption",
                evidence_mode="plot",
                prompt_variant="role",
            ),
            LocalQwenSampleCase(
                case_id="milk_role_table_case",
                abm="milk_consumption",
                evidence_mode="table",
                prompt_variant="role",
            ),
        ),
    )

    assert result.success is True
    assert adapter.context_request_count == 1
    assert adapter.trend_request_count == 2
    first_context = (result.cases[0].case_dir / "03_outputs" / "context_output.txt").read_text(encoding="utf-8")
    second_context = (result.cases[1].case_dir / "03_outputs" / "context_output.txt").read_text(encoding="utf-8")
    assert first_context == "response-1"
    assert second_context == "response-1"
