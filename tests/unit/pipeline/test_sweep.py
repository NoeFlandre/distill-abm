from __future__ import annotations

import csv
from pathlib import Path

from distill_abm.configs.models import PromptsConfig
from distill_abm.llm.adapters.base import LLMAdapter, LLMRequest, LLMResponse
from distill_abm.pipeline.run import (
    PipelineInputs,
    SweepRunResult,
    build_style_feature_combinations,
    run_pipeline_sweep,
    write_combinations_csv,
)


class CapturingAdapter(LLMAdapter):
    provider = "capture"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(provider="capture", model=request.model, text=f"resp-{len(self.requests)}", raw={})


def test_build_style_feature_combinations_generates_all_subsets() -> None:
    prompts = PromptsConfig(
        context_prompt="Context {parameters} {documentation}",
        trend_prompt="Trend {description}",
        style_features={"role": "ROLE", "example": "EXAMPLE", "insights": "INSIGHTS"},
    )
    combos = build_style_feature_combinations(prompts)
    descriptions = [description for description, _ in combos]
    assert descriptions == [
        "None",
        "role",
        "example",
        "insights",
        "role + example",
        "role + insights",
        "example + insights",
        "role + example + insights",
    ]


def test_run_pipeline_sweep_runs_context_and_one_call_per_image_per_combination(tmp_path: Path) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1;mean-incum-2\n0;1;2\n1;2;3\n", encoding="utf-8")
    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("d=1", encoding="utf-8")
    image_1 = tmp_path / "1.png"
    image_2 = tmp_path / "2.png"
    image_1.write_bytes(b"one")
    image_2.write_bytes(b"two")
    adapter = CapturingAdapter()

    output_csv = run_pipeline_sweep(
        inputs=PipelineInputs(
            csv_path=csv_path,
            parameters_path=params,
            documentation_path=docs,
            output_dir=tmp_path / "out",
            model="fake-model",
            metric_pattern="mean-incum",
            metric_description="weekly milk",
            skip_summarization=True,
        ),
        prompts=PromptsConfig(
            context_prompt="Context {parameters} {documentation}",
            trend_prompt="Trend {description} {context}",
            style_features={"role": "ROLE", "example": "EXAMPLE", "insights": "INSIGHTS"},
        ),
        adapter=adapter,
        image_paths=[image_1, image_2],
        plot_descriptions=["PLOT-1", "PLOT-2"],
    )

    assert output_csv.exists()
    # 8 feature combinations. Each performs 1 context call + 2 trend/image calls.
    assert len(adapter.requests) == 24
    assert adapter.requests[0].user_prompt().startswith("Context")
    assert adapter.requests[1].user_prompt().endswith("PLOT-1")
    assert adapter.requests[2].user_prompt().endswith("PLOT-2")
    assert adapter.requests[1].image_b64 is not None
    assert adapter.requests[2].image_b64 is not None


def test_write_combinations_csv_uses_notebook_wide_schema(tmp_path: Path) -> None:
    path = tmp_path / "llm_responses.csv"
    write_combinations_csv(
        path,
        [
            SweepRunResult(
                combination_description="role + example",
                context_prompt="cp",
                context_response="cr",
                trend_analysis_prompts=["p1", "p2"],
                trend_analysis_responses=["r1", "r2"],
            )
        ],
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == [
        "Combination Description",
        "Context Prompt",
        "Context Response",
        "Trend Analysis Prompt 1",
        "Trend Analysis Response 1",
        "Trend Analysis Prompt 2",
        "Trend Analysis Response 2",
    ]
    assert rows[1] == ["role + example", "cp", "cr", "p1", "r1", "p2", "r2"]
