from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from distill_abm.configs.models import PromptsConfig
from distill_abm.pipeline.doe_smoke import (
    DoESmokeAbmInput,
    DoESmokeModelSpec,
    DoESmokePlotInput,
    DoESmokePromptVariant,
    DoESmokeSummarizationSpec,
    canonical_prompt_variants,
    canonical_summarization_specs,
    run_doe_smoke_suite,
)
from distill_abm.pipeline.doe_smoke_prompts import legacy_plot_description_for_evidence_mode


def _write_case_inputs(tmp_path: Path) -> DoESmokeAbmInput:
    csv_path = tmp_path / "simulation.csv"
    pd.DataFrame(
        {
            "[step]": [0, 1, 2],
            "metric-a": [10.0, 12.0, 11.5],
            "metric-b": [7.0, 7.5, 8.0],
        }
    ).to_csv(csv_path, index=False, sep=";")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("number-of-agents: 1000\nhabit-on?: true\n", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("Milk model context.\n", encoding="utf-8")
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    for index in range(1, 3):
        (plots_dir / f"{index}.png").write_bytes(b"png")

    return DoESmokeAbmInput(
        abm="milk_consumption",
        csv_path=csv_path,
        parameters_path=parameters_path,
        documentation_path=documentation_path,
        metric_pattern="metric-a",
        metric_description="average weekly whole milk consumption per agent",
        plots=[
            DoESmokePlotInput(
                plot_index=1,
                reporter_pattern="metric-a",
                plot_description=(
                    "The attachment is the plot representing the average weekly consumption "
                    "of whole milk per agent."
                ),
                plot_path=plots_dir / "1.png",
            ),
            DoESmokePlotInput(
                plot_index=2,
                reporter_pattern="metric-b",
                plot_description=(
                    "The attachment is the plot representing the average weekly consumption "
                    "of skimmed and semi-skimmed milk per agent."
                ),
                plot_path=plots_dir / "2.png",
            ),
        ],
        source_viz_artifact_source="fallback",
    )


def test_run_doe_smoke_suite_writes_grouped_shared_and_case_artifacts(tmp_path: Path) -> None:
    prompts = PromptsConfig(
        context_prompt="Context prompt\n{parameters}\n{documentation}",
        trend_prompt="Trend prompt\nMetric description: {description}\nContext: {context}",
        style_features={"role": "You are a scientist.", "insights": "Keep it concise.", "example": "Give examples."},
    )
    result = run_doe_smoke_suite(
        abm_inputs={"milk_consumption": _write_case_inputs(tmp_path)},
        prompts=prompts,
        model_specs=[DoESmokeModelSpec(model_id="kimi_k2_5", provider="openrouter", model="moonshotai/kimi-k2.5")],
        output_root=tmp_path / "doe-smoke",
        evidence_modes=("plot",),
        summarization_specs=(
            DoESmokeSummarizationSpec(
                summarization_mode="none",
                text_source_mode="full_text_only",
                summarizers=(),
            ),
        ),
        prompt_variants=(DoESmokePromptVariant(variant_id="none", enabled_style_features=()),),
        repetitions=(1,),
    )

    assert result.success is True
    assert result.total_cases == 1
    assert result.total_planned_requests == 3
    assert result.design_matrix_csv_path.exists()
    assert result.request_matrix_csv_path.exists()
    assert result.request_review_csv_path.exists()
    assert result.case_index_jsonl_path.exists()
    assert result.request_index_jsonl_path.exists()
    assert result.design_matrix_csv_path.parent.name == "00_overview"
    assert result.abm_shared["milk_consumption"].shared_dir.exists()
    assert result.abm_shared["milk_consumption"].shared_dir == tmp_path / "doe-smoke" / "10_shared" / "milk_consumption"
    assert (result.abm_shared["milk_consumption"].shared_dir / "01_inputs" / "simulation.csv").exists()
    assert (result.abm_shared["milk_consumption"].shared_dir / "02_evidence" / "plots" / "plot_1.png").exists()
    assert (result.abm_shared["milk_consumption"].shared_dir / "02_evidence" / "tables" / "plot_1.txt").exists()
    assert (result.abm_shared["milk_consumption"].shared_dir / "03_prompts" / "context" / "none.txt").exists()
    case = result.cases[0]
    assert case.error_codes == []
    case_record = json.loads(result.case_index_jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert Path(case_record["context_prompt_path"]).exists()
    assert len(case_record["trend_prompt_paths"]) == 2
    assert Path(case_record["trend_prompt_paths"][0]).exists()
    assert "10_shared" in str(case_record["plot_paths"][0])
    review_header, first_review = result.request_review_csv_path.read_text(encoding="utf-8").splitlines()[:2]
    assert "prompt_preview" in review_header
    assert "prompt_length" in review_header
    assert "prompt_signature" in review_header
    assert "image_path" in review_header
    assert "table_csv_path" in review_header
    assert "Your goal is to explain an agent-based model." in first_review


def test_run_doe_smoke_suite_uses_legacy_style_prompt_composition_and_statistical_table_evidence(
    tmp_path: Path,
) -> None:
    prompts = PromptsConfig(
        context_prompt="Context prompt\n{parameters}\n{documentation}",
        trend_prompt="Trend prompt\nMetric description: {description}\nContext: {context}",
    )
    result = run_doe_smoke_suite(
        abm_inputs={"milk_consumption": _write_case_inputs(tmp_path)},
        prompts=prompts,
        model_specs=[DoESmokeModelSpec(model_id="kimi_k2_5", provider="openrouter", model="moonshotai/kimi-k2.5")],
        output_root=tmp_path / "doe-smoke",
        evidence_modes=("table",),
        summarization_specs=(
            DoESmokeSummarizationSpec(
                summarization_mode="none",
                text_source_mode="full_text_only",
                summarizers=(),
            ),
        ),
        prompt_variants=(
            DoESmokePromptVariant(
                variant_id="role+insights",
                enabled_style_features=("role", "insights"),
            ),
        ),
        repetitions=(1,),
    )

    context_prompt = (
        result.abm_shared["milk_consumption"].shared_dir / "03_prompts" / "context" / "role+insights.txt"
    ).read_text(encoding="utf-8")
    assert context_prompt.startswith("You are an expert in Consumer Behavior without any statistics background.")

    trend_prompt = (
        result.abm_shared["milk_consumption"].shared_dir
        / "03_prompts"
        / "trend"
        / "table"
        / "role+insights"
        / "plot_1.txt"
    ).read_text(encoding="utf-8")
    assert "We have a data table from repeated simulations of an agent based model." in trend_prompt
    assert "If a data table has very simple dynamics" in trend_prompt
    assert "Metric description:" not in trend_prompt
    assert "Context:" not in trend_prompt
    assert "<<context_response_from_context_llm>>" in trend_prompt
    assert "You are an expert in Consumer Behavior with a statistic background." in trend_prompt
    assert (
        "When summarizing trends, provide brief insights about their implications for decision makers."
        in trend_prompt
    )
    assert "The data table represents the average weekly consumption of whole milk per agent." in trend_prompt
    assert "Statistical summary of the relevant simulation output:" in trend_prompt
    assert "Series: metric-a" in trend_prompt
    assert "rolling Mann-Kendall:" in trend_prompt

    table_summary = (
        result.abm_shared["milk_consumption"].shared_dir / "02_evidence" / "tables" / "plot_1.txt"
    ).read_text(encoding="utf-8")
    assert "Statistical evidence for simulation series matching `metric-a`." in table_summary
    series_csv = (
        result.abm_shared["milk_consumption"].shared_dir / "02_evidence" / "tables" / "plot_1_series.csv"
    ).read_text(encoding="utf-8")
    assert series_csv == "[step],metric-a\n0,10.0\n1,12.0\n2,11.5\n"


def test_run_doe_smoke_suite_adapts_plot_and_table_prompt_wording_to_combined_evidence(tmp_path: Path) -> None:
    prompts = PromptsConfig(
        context_prompt="Context prompt\n{parameters}\n{documentation}",
        trend_prompt="Trend prompt\nMetric description: {description}\nContext: {context}",
    )
    result = run_doe_smoke_suite(
        abm_inputs={"milk_consumption": _write_case_inputs(tmp_path)},
        prompts=prompts,
        model_specs=[DoESmokeModelSpec(model_id="kimi_k2_5", provider="openrouter", model="moonshotai/kimi-k2.5")],
        output_root=tmp_path / "doe-smoke",
        evidence_modes=("plot+table",),
        summarization_specs=(
            DoESmokeSummarizationSpec(
                summarization_mode="none",
                text_source_mode="full_text_only",
                summarizers=(),
            ),
        ),
        prompt_variants=(DoESmokePromptVariant(variant_id="none", enabled_style_features=()),),
        repetitions=(1,),
    )

    trend_prompt = (
        result.abm_shared["milk_consumption"].shared_dir
        / "03_prompts"
        / "trend"
        / "plot+table"
        / "none"
        / "plot_1.txt"
    ).read_text(encoding="utf-8")
    assert "We have a plot and a data table from repeated simulations of an agent based model." in trend_prompt
    assert "If the plot and data table have very simple dynamics" in trend_prompt
    assert (
        "The attachment includes the plot, and the data table represents the average weekly consumption "
        "of whole milk per agent."
        in trend_prompt
    )


def test_legacy_plot_description_for_evidence_mode_rewrites_plot_only_language_for_table() -> None:
    rewritten = legacy_plot_description_for_evidence_mode(
        plot_description=(
            "This plot represents the average initial risk attitude of all social learning agents."
        ),
        evidence_mode="table",
    )

    assert rewritten == "This data table represents the average initial risk attitude of all social learning agents."


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
        model_specs=[DoESmokeModelSpec(model_id="kimi_k2_5", provider="openrouter", model="moonshotai/kimi-k2.5")],
        output_root=tmp_path / "doe-smoke",
        evidence_modes=("plot",),
        summarization_specs=(
            DoESmokeSummarizationSpec(
                summarization_mode="none",
                text_source_mode="full_text_only",
                summarizers=(),
            ),
        ),
        prompt_variants=(DoESmokePromptVariant(variant_id="none", enabled_style_features=()),),
        repetitions=(1,),
    )

    assert result.success is False
    assert result.failed_case_ids == ["milk_consumption::kimi_k2_5::plot::none::none::rep1"]


def test_run_doe_smoke_suite_records_unmatched_plot_reporter_without_crashing(tmp_path: Path) -> None:
    prompts = PromptsConfig(
        context_prompt="Context prompt\n{parameters}\n{documentation}",
        trend_prompt="Trend prompt\nMetric description: {description}\nContext: {context}",
    )
    input_bundle = _write_case_inputs(tmp_path)
    input_bundle.plots[1].reporter_pattern = "missing-pattern"

    result = run_doe_smoke_suite(
        abm_inputs={"milk_consumption": input_bundle},
        prompts=prompts,
        model_specs=[DoESmokeModelSpec(model_id="kimi_k2_5", provider="openrouter", model="moonshotai/kimi-k2.5")],
        output_root=tmp_path / "doe-smoke",
        evidence_modes=("plot+table",),
        summarization_specs=(
            DoESmokeSummarizationSpec(
                summarization_mode="bart",
                text_source_mode="summary_only",
                summarizers=("bart",),
            ),
        ),
        prompt_variants=(DoESmokePromptVariant(variant_id="role", enabled_style_features=("role",)),),
        repetitions=(1,),
    )

    assert result.success is False
    request_records = [
        json.loads(line)
        for line in result.request_index_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed_request = next(
        record
        for record in request_records
        if record["request_kind"] == "trend" and record["plot_index"] == "2"
    )
    assert failed_request["status"] == "failed"
    assert failed_request["error_code"] == "unmatched_metric_pattern"
    assert failed_request["table_csv_path"].endswith("plot_2.txt")
    assert result.cases[0].error_codes == ["unmatched_metric_pattern"]


def test_canonical_doe_dimensions_match_benchmark_contract() -> None:
    prompt_variants = canonical_prompt_variants()
    summarization_specs = canonical_summarization_specs()

    assert [variant.variant_id for variant in prompt_variants] == [
        "none",
        "role",
        "insights",
        "example",
        "role+example",
        "role+insights",
        "insights+example",
        "all_three",
    ]
    assert [spec.summarization_mode for spec in summarization_specs] == [
        "none",
        "bart",
        "bert",
        "t5",
        "longformer_ext",
    ]


def test_run_doe_smoke_suite_ignores_model_runtime_preflight_for_pre_llm_design_review(tmp_path: Path) -> None:
    prompts = PromptsConfig(
        context_prompt="Context prompt\n{parameters}\n{documentation}",
        trend_prompt="Trend prompt\nMetric description: {description}\nContext: {context}",
    )
    result = run_doe_smoke_suite(
        abm_inputs={"milk_consumption": _write_case_inputs(tmp_path)},
        prompts=prompts,
        model_specs=[
            DoESmokeModelSpec(
                model_id="qwen3_5_27b",
                provider="openrouter",
                model="qwen/qwen3.5-27b",
                preflight_error="debug preflight warning",
            )
        ],
        output_root=tmp_path / "doe-smoke",
        evidence_modes=("plot",),
        summarization_specs=(
            DoESmokeSummarizationSpec(
                summarization_mode="none",
                text_source_mode="full_text_only",
                summarizers=(),
            ),
        ),
        prompt_variants=(DoESmokePromptVariant(variant_id="none", enabled_style_features=()),),
        repetitions=(1,),
    )

    assert result.success is True
    case_record = json.loads(result.case_index_jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert case_record["model_id"] == "qwen3_5_27b"
    assert result.cases[0].error_codes == []
