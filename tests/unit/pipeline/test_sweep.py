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


class NamedAdapter(LLMAdapter):
    provider = "named"

    def __init__(self, label: str) -> None:
        self.label = label
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(provider=self.label, model=request.model, text=f"{self.label}-resp", raw={})


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


def test_write_combinations_csv_uses_wide_schema(tmp_path: Path) -> None:
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


def test_run_pipeline_sweep_supports_separate_context_and_trend_adapters(tmp_path: Path) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1;mean-incum-2\n0;1;2\n1;2;3\n", encoding="utf-8")
    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("d=1", encoding="utf-8")
    image = tmp_path / "1.png"
    image.write_bytes(b"one")

    default_adapter = NamedAdapter("default")
    context_adapter = NamedAdapter("context")
    trend_adapter = NamedAdapter("trend")

    run_pipeline_sweep(
        inputs=PipelineInputs(
            csv_path=csv_path,
            parameters_path=params,
            documentation_path=docs,
            output_dir=tmp_path / "out",
            model="unused-default-model",
            metric_pattern="mean-incum",
            metric_description="weekly milk",
        ),
        prompts=PromptsConfig(
            context_prompt="Context {parameters} {documentation}",
            trend_prompt="Trend {description} {context}",
            style_features={"role": "ROLE", "example": "EXAMPLE", "insights": "INSIGHTS"},
        ),
        adapter=default_adapter,
        context_adapter=context_adapter,
        trend_adapter=trend_adapter,
        context_model="context-model",
        trend_model="trend-model",
        image_paths=[image],
        plot_descriptions=["PLOT-1"],
    )

    assert len(default_adapter.requests) == 0
    # 8 combinations -> 8 context calls and 8 trend/image calls
    assert len(context_adapter.requests) == 8
    assert len(trend_adapter.requests) == 8
    assert context_adapter.requests[0].model == "context-model"
    assert trend_adapter.requests[0].model == "trend-model"
    assert trend_adapter.requests[0].image_b64 is not None


def test_write_combinations_csv_plot_headers_and_resume_mode(tmp_path: Path) -> None:
    path = tmp_path / "llm_responses.csv"
    rows_first = [
        SweepRunResult(
            combination_description="None",
            context_prompt="cp1",
            context_response="cr1",
            trend_analysis_prompts=["p1", "p2"],
            trend_analysis_responses=["r1", "r2"],
        )
    ]
    write_combinations_csv(path, rows_first, csv_column_style="plot", resume_existing=True)
    rows_second = [
        SweepRunResult(
            combination_description="None",
            context_prompt="cp1-updated",
            context_response="cr1-updated",
            trend_analysis_prompts=["new-p1", "new-p2"],
            trend_analysis_responses=["new-r1", "new-r2"],
        ),
        SweepRunResult(
            combination_description="role",
            context_prompt="cp2",
            context_response="cr2",
            trend_analysis_prompts=["rp1", "rp2"],
            trend_analysis_responses=["rr1", "rr2"],
        ),
    ]
    write_combinations_csv(path, rows_second, csv_column_style="plot", resume_existing=True)

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == [
        "Combination Description",
        "Context Prompt",
        "Context Response",
        "Plot 1 Prompt",
        "Plot 1 Analysis",
        "Plot 2 Prompt",
        "Plot 2 Analysis",
    ]
    # Existing row remains unchanged in filled prompt/analysis slots when resuming.
    assert rows[1] == ["None", "cp1", "cr1", "p1", "r1", "p2", "r2"]
    assert rows[2] == ["role", "cp2", "cr2", "rp1", "rr1", "rp2", "rr2"]


def test_run_pipeline_sweep_resume_skips_existing_combinations_without_llm_calls(tmp_path: Path) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1\n0;1\n1;2\n", encoding="utf-8")
    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("d=1", encoding="utf-8")
    image = tmp_path / "1.png"
    image.write_bytes(b"one")
    output_csv = tmp_path / "out" / "combinations_report.csv"
    prompts = PromptsConfig(
        context_prompt="Context {parameters} {documentation}",
        trend_prompt="Trend {description} {context}",
        style_features={"role": "ROLE", "example": "EXAMPLE", "insights": "INSIGHTS"},
    )

    all_rows: list[SweepRunResult] = []
    for description, _ in build_style_feature_combinations(prompts):
        all_rows.append(
            SweepRunResult(
                combination_description=description,
                context_prompt=f"context {description}",
                context_response=f"context response {description}",
                trend_analysis_prompts=["p1"],
                trend_analysis_responses=["r1"],
            )
        )
    write_combinations_csv(output_csv, all_rows, resume_existing=False)

    adapter = CapturingAdapter()
    out = run_pipeline_sweep(
        inputs=PipelineInputs(
            csv_path=csv_path,
            parameters_path=params,
            documentation_path=docs,
            output_dir=tmp_path / "out",
            model="fake-model",
            metric_pattern="mean-incum",
            metric_description="weekly milk",
        ),
        prompts=prompts,
        adapter=adapter,
        image_paths=[image],
        plot_descriptions=["PLOT-1"],
        output_csv=output_csv,
        resume_existing=True,
    )

    assert out == output_csv
    assert len(adapter.requests) == 0
