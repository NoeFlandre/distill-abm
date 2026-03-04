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


def test_cli_run_pipeline_table_csv_mode(tmp_path: Path) -> None:
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
            "table-csv",
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


def test_cli_run_pipeline_forwards_summarization_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    class _Result:
        def __init__(self, output_dir: Path) -> None:
            self.plot_path = output_dir / "plot.png"
            self.report_csv = output_dir / "report.csv"
            self.context_response = "context"
            self.trend_response = "trend"
            self.trend_full_response = "trend"
            self.trend_summary_response = None
            self.full_scores = None
            self.summary_scores = None
            self.stats_table_csv = None
            self.stats_image_path = None
            self.token_f1 = 0.0
            self.bleu = 0.0
            self.meteor = 0.0
            self.rouge1 = 0.0
            self.rouge2 = 0.0
            self.rouge_l = 0.0
            self.flesch_reading_ease = 0.0

    def fake_run_pipeline(*, inputs, prompts, adapter):  # type: ignore[no-untyped-def]
        captured["inputs"] = inputs
        return _Result(inputs.output_dir)

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
            "--summarization-mode",
            "both",
            "--score-on",
            "full",
            "--evidence-mode",
            "plot",
        ],
    )

    assert result.exit_code == 0
    assert "inputs" in captured
    captured_inputs = cast(Any, captured["inputs"])
    assert captured_inputs.summarization_mode == "both"
    assert captured_inputs.score_on == "full"
    assert captured_inputs.skip_summarization is False


def test_cli_run_pipeline_forwards_summary_only_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    class _Result:
        def __init__(self, output_dir: Path) -> None:
            self.plot_path = output_dir / "plot.png"
            self.report_csv = output_dir / "report.csv"
            self.context_response = "context"
            self.trend_response = "trend"
            self.trend_full_response = "trend"
            self.trend_summary_response = "summary"
            self.full_scores = None
            self.summary_scores = None
            self.stats_table_csv = None
            self.stats_image_path = None
            self.token_f1 = 0.0
            self.bleu = 0.0
            self.meteor = 0.0
            self.rouge1 = 0.0
            self.rouge2 = 0.0
            self.rouge_l = 0.0
            self.flesch_reading_ease = 0.0

    def fake_run_pipeline(*, inputs, prompts, adapter):  # type: ignore[no-untyped-def]
        captured["inputs"] = inputs
        return _Result(inputs.output_dir)

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
            "--summarization-mode",
            "summary",
            "--score-on",
            "summary",
        ],
    )

    assert result.exit_code == 0
    assert "inputs" in captured
    captured_inputs = cast(Any, captured["inputs"])
    assert captured_inputs.summarization_mode == "summary"
    assert captured_inputs.score_on == "summary"


def test_cli_run_pipeline_forwards_additional_summarizers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    class _Result:
        def __init__(self, output_dir: Path) -> None:
            self.plot_path = output_dir / "plot.png"
            self.report_csv = output_dir / "report.csv"
            self.context_response = "context"
            self.trend_response = "trend"
            self.trend_full_response = "trend"
            self.trend_summary_response = None
            self.full_scores = None
            self.summary_scores = None
            self.stats_table_csv = None
            self.stats_image_path = None
            self.token_f1 = 0.0
            self.bleu = 0.0
            self.meteor = 0.0
            self.rouge1 = 0.0
            self.rouge2 = 0.0
            self.rouge_l = 0.0
            self.flesch_reading_ease = 0.0

    def fake_run_pipeline(*, inputs, prompts, adapter):  # type: ignore[no-untyped-def]
        captured["inputs"] = inputs
        return _Result(inputs.output_dir)

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
            "--additional-summarizer",
            "t5",
            "--additional-summarizer",
            "longformer_ext",
        ],
    )

    assert result.exit_code == 0
    assert "inputs" in captured
    captured_inputs = cast(Any, captured["inputs"])
    assert captured_inputs.additional_summarizers == ("t5", "longformer_ext")


def test_cli_run_pipeline_rejects_invalid_additional_summarizer(tmp_path: Path) -> None:
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
            "--additional-summarizer",
            "bad-option",
        ],
    )

    assert result.exit_code != 0
    assert "unsupported additional summarizer" in result.output


def test_cli_run_pipeline_defaults_summarization_and_score_modes_to_both(
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
        output_dir.joinpath("plot.png").write_text("plot", encoding="utf-8")
        output_dir.joinpath("report.csv").write_text("report", encoding="utf-8")
        return _Result(output_dir)

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
            "--metric-pattern",
            "mean-incum",
            "--metric-description",
            "weekly milk",
        ],
    )

    assert result.exit_code == 0
    captured_inputs = cast(Any, captured["inputs"])
    assert captured_inputs.summarization_mode == "both"
    assert captured_inputs.score_on == "both"


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


def test_cli_smoke_qwen_forwards_inputs_and_reports_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1\n0;1\n1;2\n", encoding="utf-8")
    params = tmp_path / "params.txt"
    params.write_text("param=1\n", encoding="utf-8")
    docs = tmp_path / "docs.txt"
    docs.write_text("doc\n", encoding="utf-8")
    prompts = tmp_path / "prompts.yaml"
    prompts.write_text(
        "\n".join(
            [
                'context_prompt: "Context {parameters} {documentation}"',
                'trend_prompt: "Trend {description}"',
                'coverage_eval_prompt: "Coverage score: 4"',
                'faithfulness_eval_prompt: "Faithfulness score: 4"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    doe_csv = tmp_path / "doe.csv"
    doe_csv.write_text("Model,WithExamples,BLEU\nQwen,Yes,0.4\nQwen,No,0.3\n", encoding="utf-8")

    captured: dict[str, object] = {}

    class _Result:
        def __init__(self, output_dir: Path) -> None:
            self.success = True
            self.failed_cases: list[str] = []
            self.report_markdown_path = output_dir / "smoke_report.md"
            self.report_json_path = output_dir / "smoke_report.json"
            self.doe_output_csv: Path | None = output_dir / "anova.csv"
            self.sweep_output_csv: Path | None = output_dir / "sweep.csv"

    def fake_run_qwen_smoke_suite(*, inputs, prompts, adapter, run_qualitative, doe_input_csv, run_sweep):  # type: ignore[no-untyped-def]
        captured["inputs"] = inputs
        captured["prompts"] = prompts
        captured["adapter"] = adapter
        captured["run_qualitative"] = run_qualitative
        captured["doe_input_csv"] = doe_input_csv
        captured["run_sweep"] = run_sweep
        inputs.output_dir.mkdir(parents=True, exist_ok=True)
        result = _Result(inputs.output_dir)
        result.report_markdown_path.write_text("# smoke\n", encoding="utf-8")
        result.report_json_path.write_text("{}", encoding="utf-8")
        return result

    monkeypatch.setattr("distill_abm.cli.run_qwen_smoke_suite", fake_run_qwen_smoke_suite)
    result = runner.invoke(
        app,
        [
            "smoke-qwen",
            "--csv-path",
            str(csv_path),
            "--parameters-path",
            str(params),
            "--documentation-path",
            str(docs),
            "--doe-input-csv",
            str(doe_csv),
            "--prompts-path",
            str(prompts),
            "--output-dir",
            str(tmp_path / "smoke"),
            "--model",
            "qwen2.5:latest",
            "--metric-pattern",
            "mean-incum",
            "--metric-description",
            "weekly milk",
            "--plot-description",
            "plot desc",
            "--skip-qualitative",
            "--skip-sweep",
        ],
    )

    assert result.exit_code == 0
    assert "smoke report (markdown):" in result.output
    assert "smoke report (json):" in result.output
    assert "inputs" in captured
    smoke_inputs = cast(Any, captured["inputs"])
    assert smoke_inputs.model == "qwen2.5:latest"
    assert smoke_inputs.metric_pattern == "mean-incum"
    assert smoke_inputs.metric_description == "weekly milk"
    assert smoke_inputs.plot_description == "plot desc"
    assert captured["run_qualitative"] is False
    assert captured["run_sweep"] is False
