from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from distill_abm.configs.models import PromptsConfig
from distill_abm.pipeline.doe_smoke import DoESmokeAbmInput, run_doe_smoke_suite
from distill_abm.pipeline.smoke import SmokeCase


def _write_case_inputs(tmp_path: Path) -> DoESmokeAbmInput:
    csv_path = tmp_path / "simulation.csv"
    pd.DataFrame(
        {
            "[run number]": [1, 1, 1],
            "mean-incum": [10.0, 12.0, 11.5],
            "[step]": [0, 1, 2],
        }
    ).to_csv(csv_path, index=False)
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("number-of-agents: 1000\nhabit-on?: true\n", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("Milk model context.\n", encoding="utf-8")

    return DoESmokeAbmInput(
        abm="milk_consumption",
        csv_path=csv_path,
        parameters_path=parameters_path,
        documentation_path=documentation_path,
        metric_pattern="mean-incum",
        metric_description="average weekly whole milk consumption per agent",
        plot_description="The attachment is the whole milk consumption plot.",
        source_viz_plot_dir=None,
        source_viz_artifact_source="fallback",
    )


def test_run_doe_smoke_suite_writes_pre_llm_case_bundles(tmp_path: Path) -> None:
    prompts = PromptsConfig(
        context_prompt="Context prompt\n{parameters}\n{documentation}",
        trend_prompt="Trend prompt\nMetric description: {description}\nContext: {context}",
        style_features={"role": "You are a scientist.", "insights": "Keep it concise."},
    )
    inputs = {"milk_consumption": _write_case_inputs(tmp_path)}
    cases = [SmokeCase(case_id="plot-summary", evidence_mode="plot", text_source_mode="summary_only")]

    result = run_doe_smoke_suite(
        abm_inputs=inputs,
        prompts=prompts,
        provider="openrouter",
        model="moonshotai/kimi-k2.5",
        output_root=tmp_path / "doe-smoke",
        cases=cases,
    )

    assert result.success is True
    assert result.design_matrix_csv_path.exists()
    assert result.report_json_path.exists()
    assert result.report_markdown_path.exists()
    assert len(result.cases) == 1
    case = result.cases[0]
    assert case.status == "ok"
    assert case.context_prompt_path is not None and case.context_prompt_path.exists()
    assert case.trend_prompt_template_path is not None and case.trend_prompt_template_path.exists()
    assert case.pipeline_plot_path is not None and case.pipeline_plot_path.exists()
    assert case.stats_table_csv_path is not None and case.stats_table_csv_path.exists()
    assert case.evidence_image_path == case.pipeline_plot_path
    request_payload = json.loads(case.context_request_plan_path.read_text(encoding="utf-8"))
    assert request_payload["image_attached"] is False


def test_run_doe_smoke_suite_flags_placeholder_inputs(tmp_path: Path) -> None:
    prompts = PromptsConfig(
        context_prompt="Context prompt\n{parameters}\n{documentation}",
        trend_prompt="Trend prompt\nMetric description: {description}\nContext: {context}",
    )
    input_bundle = _write_case_inputs(tmp_path)
    input_bundle.documentation_path.write_text("TODO placeholder documentation.\n", encoding="utf-8")

    result = run_doe_smoke_suite(
        abm_inputs={"milk_consumption": input_bundle},
        prompts=prompts,
        provider="openrouter",
        model="moonshotai/kimi-k2.5",
        output_root=tmp_path / "doe-smoke",
        cases=[SmokeCase(case_id="plot-summary", evidence_mode="plot", text_source_mode="summary_only")],
    )

    assert result.success is False
    assert result.failed_case_ids == ["milk_consumption::plot-summary"]
    doc_stage = next(stage for stage in result.cases[0].stage_results if stage.stage.stage_id == "documentation")
    assert doc_stage.status == "failed"
    assert doc_stage.error_code == "placeholder_detected"


def test_run_doe_smoke_suite_records_unmatched_metric_pattern_without_crashing(tmp_path: Path) -> None:
    prompts = PromptsConfig(
        context_prompt="Context prompt\n{parameters}\n{documentation}",
        trend_prompt="Trend prompt\nMetric description: {description}\nContext: {context}",
    )
    input_bundle = _write_case_inputs(tmp_path)
    input_bundle.metric_pattern = "missing-pattern"

    result = run_doe_smoke_suite(
        abm_inputs={"milk_consumption": input_bundle},
        prompts=prompts,
        provider="openrouter",
        model="moonshotai/kimi-k2.5",
        output_root=tmp_path / "doe-smoke",
        cases=[SmokeCase(case_id="plot-summary", evidence_mode="plot", text_source_mode="summary_only")],
    )

    assert result.success is False
    plot_stage = next(stage for stage in result.cases[0].stage_results if stage.stage.stage_id == "pipeline-plot")
    assert plot_stage.status == "failed"
    assert plot_stage.error_code == "unmatched_metric_pattern"
