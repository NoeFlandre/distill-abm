from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from distill_abm.llm.adapters.base import LLMAdapter, LLMRequest, LLMResponse
from distill_abm.pipeline.local_qwen_sample_smoke import LocalQwenCaseInput, LocalQwenSampleCase
from distill_abm.pipeline.local_qwen_tuning import run_local_qwen_tuning


class FakeUsageAdapter(LLMAdapter):
    provider = "ollama"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        num_ctx = request.metadata["ollama_num_ctx"]
        prompt_length = len(request.user_prompt())
        if request.image_b64 and isinstance(num_ctx, int) and num_ctx < 32000 and prompt_length > 1000:
            raise RuntimeError("context window too small")
        response_index = len(self.requests)
        return LLMResponse(
            provider=self.provider,
            model=request.model,
            text=json.dumps({"response_text": f"response-{response_index}"}),
            raw={
                "message": {"thinking": f"thinking-{response_index}"},
                "usage": {
                    "prompt_tokens": 100 + response_index,
                    "completion_tokens": 30 + response_index,
                    "total_tokens": 130 + response_index,
                },
            },
        )


def _write_case_input(tmp_path: Path, *, abm: str) -> LocalQwenCaseInput:
    csv_path = tmp_path / f"{abm}.csv"
    pd.DataFrame({"[step]": [0, 1, 2], "metric-a": [1.0, 2.0, 3.0]}).to_csv(csv_path, index=False, sep=";")
    parameters_path = tmp_path / f"{abm}_parameters.txt"
    documentation_path = tmp_path / f"{abm}_documentation.txt"
    plot_path = tmp_path / f"{abm}_plot.png"
    parameters_path.write_text("number-households: 60\n", encoding="utf-8")
    documentation_path.write_text("Model documentation.\n", encoding="utf-8")
    plot_path.write_bytes(b"png")
    return LocalQwenCaseInput(
        abm=abm,
        csv_path=csv_path,
        parameters_path=parameters_path,
        documentation_path=documentation_path,
        reporter_pattern="metric-a",
        plot_description="The attachment is the plot representing the metric.",
        plot_path=plot_path,
    )


def test_run_local_qwen_tuning_selects_smallest_successful_num_ctx(tmp_path: Path) -> None:
    result = run_local_qwen_tuning(
        case_inputs={
            "fauna": _write_case_input(tmp_path, abm="fauna"),
            "grazing": _write_case_input(tmp_path, abm="grazing"),
            "milk_consumption": _write_case_input(tmp_path, abm="milk_consumption"),
        },
        adapter=FakeUsageAdapter(),
        model="qwen3.5:0.8b",
        output_root=tmp_path / "tuning",
        num_ctx_candidates=(8192, 16384, 32768),
        max_tokens_candidates=(1024, 2048),
        cases=(
            LocalQwenSampleCase(case_id="plot_case", abm="fauna", evidence_mode="plot", prompt_variant="none"),
            LocalQwenSampleCase(case_id="table_case", abm="grazing", evidence_mode="table", prompt_variant="role"),
            LocalQwenSampleCase(
                case_id="plot_table_case",
                abm="milk_consumption",
                evidence_mode="plot+table",
                prompt_variant="insights",
            ),
        ),
    )

    assert result.success is True
    recommendations = {item.evidence_mode: item for item in result.recommendations}
    assert recommendations["plot"].recommended_num_ctx == 8192
    assert recommendations["table"].recommended_num_ctx == 8192
    assert recommendations["plot"].recommended_max_tokens == 1024
    assert recommendations["table"].recommended_max_tokens == 1024
    assert recommendations["plot+table"].recommended_num_ctx == 8192
    assert recommendations["plot+table"].recommended_max_tokens == 1024
    assert result.trials_csv_path.exists()
