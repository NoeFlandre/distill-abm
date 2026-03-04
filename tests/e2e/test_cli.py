import json
from pathlib import Path
from typing import Any, cast

import pytest
import typer
from typer.testing import CliRunner

import distill_abm.cli as cli_module
from distill_abm.cli import app
from distill_abm.configs.models import ABMConfig
from distill_abm.llm.adapters.base import LLMAdapter, LLMRequest, LLMResponse

runner = CliRunner()


def test_cli_run_pipeline(tmp_path: Path) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1\n0;1\n1;2\n", encoding="utf-8")

    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    prompts = tmp_path / "prompts.yaml"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("d=1", encoding="utf-8")
    prompts.write_text(
        'context_prompt: "Context {parameters} {documentation}"\ntrend_prompt: "Trend {description}"\n',
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "run",
            "--csv-path",
            str(csv_path),
            "--parameters-path",
            str(params),
            "--documentation-path",
            str(docs),
            "--prompts-path",
            str(prompts),
            "--output-dir",
            str(output_dir),
            "--provider",
            "echo",
            "--model",
            "echo-model",
            "--metric-pattern",
            "mean-incum",
            "--metric-description",
            "weekly milk",
            "--skip-summarization",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "report.csv").exists()


def test_cli_run_pipeline_stats_markdown_mode(tmp_path: Path) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1\n0;1\n1;2\n", encoding="utf-8")

    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    prompts = tmp_path / "prompts.yaml"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("d=1", encoding="utf-8")
    prompts.write_text(
        'context_prompt: "Context {parameters} {documentation}"\ntrend_prompt: "Trend {description}"\n',
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "run",
            "--csv-path",
            str(csv_path),
            "--parameters-path",
            str(params),
            "--documentation-path",
            str(docs),
            "--prompts-path",
            str(prompts),
            "--output-dir",
            str(output_dir),
            "--provider",
            "echo",
            "--model",
            "echo-model",
            "--metric-pattern",
            "mean-incum",
            "--metric-description",
            "weekly milk",
            "--skip-summarization",
            "--evidence-mode",
            "stats-markdown",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "report.csv").exists()


class QualFakeAdapter(LLMAdapter):
    provider = "fake"

    def complete(self, request: LLMRequest) -> LLMResponse:
        _ = request
        return LLMResponse(
            provider="fake",
            model="fake-model",
            text="Coverage score: 3. Reasoning: Most core dynamics are represented.",
            raw={},
        )


def test_cli_evaluate_qualitative_outputs_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = tmp_path / "prompts.yaml"
    prompts.write_text(
        "\n".join(
            [
                'context_prompt: "Context {parameters} {documentation}"',
                'trend_prompt: "Trend {description}"',
                'coverage_eval_prompt: "Evaluate coverage. Source: {source} Summary: {summary}"',
                'faithfulness_eval_prompt: "Evaluate faithfulness. Source: {source} Summary: {summary}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("distill_abm.cli.create_adapter", lambda provider, model: QualFakeAdapter())
    result = runner.invoke(
        app,
        [
            "evaluate-qualitative",
            "--summary-text",
            "This is a generated summary.",
            "--source-text",
            "This is source context.",
            "--metric",
            "coverage",
            "--provider",
            "fake",
            "--model",
            "fake-model",
            "--prompts-path",
            str(prompts),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["score"] == 3
    assert payload["reasoning"].startswith("Coverage score")
    assert payload["model"] == "fake-model"


def test_cli_run_with_abm_passes_first_plot_description(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1\n0;1\n1;2\n", encoding="utf-8")
    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    prompts = tmp_path / "prompts.yaml"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("d=1", encoding="utf-8")
    prompts.write_text(
        'context_prompt: "Context {parameters} {documentation}"\ntrend_prompt: "Trend {description}"\n',
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_load_abm_config(_path: Path) -> ABMConfig:
        return ABMConfig(
            name="milk_consumption",
            metric_pattern="mean-incum",
            metric_description="weekly milk",
            plot_descriptions=["first plot description", "second"],
        )

    class _Result:
        def __init__(self, output_dir: Path) -> None:
            self.plot_path = output_dir / "plot.png"
            self.report_csv = output_dir / "report.csv"

    def fake_run_pipeline(*, inputs, prompts, adapter):  # type: ignore[no-untyped-def]
        captured["inputs"] = inputs
        captured["prompts"] = prompts
        captured["adapter"] = adapter
        output_dir = inputs.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "plot.png"
        report_csv = output_dir / "report.csv"
        plot_path.write_text("plot", encoding="utf-8")
        report_csv.write_text("report", encoding="utf-8")
        return _Result(output_dir)

    monkeypatch.setattr("distill_abm.cli.load_abm_config", fake_load_abm_config)
    monkeypatch.setattr("distill_abm.cli.run_pipeline", fake_run_pipeline)

    result = runner.invoke(
        app,
        [
            "run",
            "--csv-path",
            str(csv_path),
            "--parameters-path",
            str(params),
            "--documentation-path",
            str(docs),
            "--prompts-path",
            str(prompts),
            "--output-dir",
            str(tmp_path / "out"),
            "--provider",
            "echo",
            "--model",
            "echo-model",
            "--abm",
            "milk_consumption",
        ],
    )

    assert result.exit_code == 0
    assert "inputs" in captured
    inputs = captured["inputs"]
    assert cast(Any, inputs).plot_description == "first plot description"


def test_cli_run_direct_call_with_abm_defaults_plot_description(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1\n0;1\n1;2\n", encoding="utf-8")

    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    prompts = tmp_path / "prompts.yaml"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("d=1", encoding="utf-8")
    prompts.write_text(
        'context_prompt: "Context {parameters} {documentation}"\ntrend_prompt: "Trend {description}"\n',
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_load_abm_config(_path: Path) -> ABMConfig:
        return ABMConfig(
            name="grazing",
            metric_pattern="mean-incum",
            metric_description="weekly grazing",
            plot_descriptions=["auto plot", "fallback"],
        )

    class _Result:
        def __init__(self, output_dir: Path) -> None:
            self.plot_path = output_dir / "plot.png"
            self.report_csv = output_dir / "report.csv"

    def fake_run_pipeline(*, inputs, prompts, adapter):  # type: ignore[no-untyped-def]
        captured["inputs"] = inputs
        captured["prompts"] = prompts
        captured["adapter"] = adapter
        return _Result(inputs.output_dir)

    monkeypatch.setattr("distill_abm.cli.load_abm_config", fake_load_abm_config)
    monkeypatch.setattr("distill_abm.cli.run_pipeline", fake_run_pipeline)

    cli_module.run(
        csv_path=csv_path,
        parameters_path=params,
        documentation_path=docs,
        prompts_path=prompts,
        output_dir=tmp_path / "out",
        provider="echo",
        model="echo-model",
        metric_pattern="mean-incum",
        metric_description="manual",
        plot_description=None,
        evidence_mode="plot",
        skip_summarization=False,
        abm="grazing",
    )

    assert "inputs" in captured
    assert cast(Any, captured["inputs"]).plot_description == "auto plot"


def test_cli_analyze_doe_exit_without_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    def fake_analyze_factorial_anova(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr("distill_abm.cli.analyze_factorial_anova", fake_analyze_factorial_anova)
    with pytest.raises(typer.Exit) as exc_info:
        cli_module.analyze_doe(input_csv=input_csv, output_csv=tmp_path / "results" / "out.csv")
    assert exc_info.value.exit_code == 1


def test_cli_main_invokes_app(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"value": False}

    class _DummyApp:
        def __call__(self) -> None:
            called["value"] = True

    monkeypatch.setattr("distill_abm.cli.app", _DummyApp())
    cli_module.main()
    assert called["value"]
