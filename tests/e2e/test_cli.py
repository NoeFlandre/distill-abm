import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from distill_abm.cli import app
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
