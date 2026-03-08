from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from distill_abm.configs.models import PromptsConfig
from distill_abm.pipeline.smoke_optional_steps import run_doe_if_requested, run_sweep_if_requested
from distill_abm.pipeline.smoke_types import SmokeCase, SmokeCaseResult, SmokeSuiteInputs


def test_run_doe_if_requested_skips_when_no_input_csv(tmp_path: Path) -> None:
    status, output_csv, error = run_doe_if_requested(
        output_root=tmp_path,
        doe_input_csv=None,
        resume_existing=False,
    )

    assert status == "skipped"
    assert output_csv is None
    assert error is None


def test_run_doe_if_requested_writes_output_csv(tmp_path: Path) -> None:
    input_csv = tmp_path / "doe.csv"
    pd.DataFrame(
        {
            "Model": ["Qwen", "Qwen", "Qwen", "Qwen"],
            "WithExamples": ["Yes", "No", "Yes", "No"],
            "BLEU": [0.4, 0.2, 0.45, 0.25],
        }
    ).to_csv(input_csv, index=False)

    status, output_csv, error = run_doe_if_requested(
        output_root=tmp_path,
        doe_input_csv=input_csv,
        resume_existing=False,
    )

    assert status == "ok"
    assert output_csv is not None and output_csv.exists()
    assert error is None


def test_run_sweep_if_requested_fails_without_available_plots(tmp_path: Path) -> None:
    status, output_csv, error = run_sweep_if_requested(
        output_root=tmp_path,
        inputs=SmokeSuiteInputs(
            csv_path=tmp_path / "simulation.csv",
            parameters_path=tmp_path / "parameters.txt",
            documentation_path=tmp_path / "documentation.txt",
            output_dir=tmp_path / "smoke",
            model="qwen3.5:0.8b",
            metric_pattern="metric",
            metric_description="description",
        ),
        prompts=PromptsConfig(context_prompt="ctx", trend_prompt="trend"),
        adapter=SimpleNamespace(),
        case_results=[],
        run_sweep=True,
        resume_existing=False,
        run_pipeline_sweep_fn=lambda **_: None,
    )

    assert status == "failed"
    assert output_csv is None
    assert error == "no successful case produced a plot image for sweep execution"


def test_run_sweep_if_requested_reports_success_when_stubbed(tmp_path: Path) -> None:
    csv_path = tmp_path / "simulation.csv"
    csv_path.write_text("tick;metric\n0;1\n", encoding="utf-8")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("param=1\n", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("docs\n", encoding="utf-8")
    plot_path = tmp_path / "plot.png"
    plot_path.write_bytes(b"png")

    status, output_csv, error = run_sweep_if_requested(
        output_root=tmp_path,
        inputs=SmokeSuiteInputs(
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            output_dir=tmp_path / "smoke",
            model="qwen3.5:0.8b",
            metric_pattern="metric",
            metric_description="description",
            plot_description="plot",
        ),
        prompts=PromptsConfig(context_prompt="ctx", trend_prompt="trend"),
        adapter=SimpleNamespace(),
        case_results=[
            SmokeCaseResult(
                case=SmokeCase(case_id="case-1", evidence_mode="plot", text_source_mode="summary_only"),
                status="ok",
                output_dir=tmp_path / "case-1",
                plot_path=plot_path,
            )
        ],
        run_sweep=True,
        resume_existing=False,
        run_pipeline_sweep_fn=lambda **_: None,
    )

    assert status == "ok"
    assert output_csv == tmp_path / "sweep" / "combinations_report.csv"
    assert error is None
