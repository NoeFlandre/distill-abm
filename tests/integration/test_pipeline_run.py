from pathlib import Path

import pytest

from distill_abm.configs.models import PromptsConfig
from distill_abm.llm.adapters.base import LLMAdapter, LLMRequest, LLMResponse
from distill_abm.pipeline.run import PipelineInputs, run_pipeline


class FakeAdapter(LLMAdapter):
    provider = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(provider="fake", model=request.model, text=f"resp-{self.calls}", raw={})


class CapturingAdapter(LLMAdapter):
    provider = "capture"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(provider="capture", model=request.model, text="captured-response", raw={})


def test_run_pipeline_creates_artifacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1;mean-incum-2\n0;1;2\n1;2;3\n", encoding="utf-8")

    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("doc", encoding="utf-8")

    prompts = PromptsConfig(
        context_prompt="Context {parameters} {documentation}",
        trend_prompt="Trend {description}",
    )
    adapter = FakeAdapter()
    result = run_pipeline(
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
    )

    assert result.plot_path.exists()
    assert result.report_csv.exists()
    assert adapter.calls == 2


def test_run_pipeline_skip_summarization_bypasses_model_summarizers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1;mean-incum-2\n0;1;2\n1;2;3\n", encoding="utf-8")

    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("doc", encoding="utf-8")

    monkeypatch.setattr("distill_abm.pipeline.run.summarize_with_bart", lambda _text: "should-not-run")
    monkeypatch.setattr("distill_abm.pipeline.run.summarize_with_bert", lambda _text: "should-not-run")

    adapter = FakeAdapter()
    result = run_pipeline(
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
            trend_prompt="Trend {description}",
        ),
        adapter=adapter,
    )

    assert result.trend_response == "resp-2"
    assert adapter.calls == 2


def test_run_pipeline_uses_bart_bert_summaries_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1;mean-incum-2\n0;1;2\n1;2;3\n", encoding="utf-8")

    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("doc", encoding="utf-8")

    monkeypatch.setattr("distill_abm.pipeline.run.summarize_with_bart", lambda text: f"bart::{text}")
    monkeypatch.setattr("distill_abm.pipeline.run.summarize_with_bert", lambda text: f"bert::{text}")

    adapter = FakeAdapter()
    result = run_pipeline(
        inputs=PipelineInputs(
            csv_path=csv_path,
            parameters_path=params,
            documentation_path=docs,
            output_dir=tmp_path / "out",
            model="fake-model",
            metric_pattern="mean-incum",
            metric_description="weekly milk",
            skip_summarization=False,
        ),
        prompts=PromptsConfig(
            context_prompt="Context {parameters} {documentation}",
            trend_prompt="Trend {description}",
        ),
        adapter=adapter,
    )

    assert result.trend_response == "bart::resp-2\nbert::resp-2"
    assert adapter.calls == 2


def test_run_pipeline_stats_markdown_mode_uses_text_only_evidence(tmp_path: Path) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1;mean-incum-2\n0;1;2\n1;2;3\n", encoding="utf-8")
    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("doc", encoding="utf-8")
    adapter = CapturingAdapter()

    run_pipeline(
        inputs=PipelineInputs(
            csv_path=csv_path,
            parameters_path=params,
            documentation_path=docs,
            output_dir=tmp_path / "out",
            model="fake-model",
            metric_pattern="mean-incum",
            metric_description="weekly milk",
            skip_summarization=True,
            evidence_mode="stats-markdown",
        ),
        prompts=PromptsConfig(
            context_prompt="Context {parameters} {documentation}",
            trend_prompt="Trend {description}",
        ),
        adapter=adapter,
    )

    assert len(adapter.requests) == 2
    trend_request = adapter.requests[-1]
    assert trend_request.image_b64 is None
    assert "| time_step | mean | std | min | max | median |" in trend_request.user_prompt()


def test_run_pipeline_stats_image_mode_uses_table_image(tmp_path: Path) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1;mean-incum-2\n0;1;2\n1;2;3\n", encoding="utf-8")
    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("doc", encoding="utf-8")
    adapter = CapturingAdapter()

    result = run_pipeline(
        inputs=PipelineInputs(
            csv_path=csv_path,
            parameters_path=params,
            documentation_path=docs,
            output_dir=tmp_path / "out",
            model="fake-model",
            metric_pattern="mean-incum",
            metric_description="weekly milk",
            skip_summarization=True,
            evidence_mode="stats-image",
        ),
        prompts=PromptsConfig(
            context_prompt="Context {parameters} {documentation}",
            trend_prompt="Trend {description}",
        ),
        adapter=adapter,
    )

    assert len(adapter.requests) == 2
    trend_request = adapter.requests[-1]
    assert trend_request.image_b64 is not None
    assert result.stats_image_path is not None
    assert result.stats_image_path.exists()


def test_run_pipeline_plot_plus_stats_uses_plot_image_and_markdown(tmp_path: Path) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1;mean-incum-2\n0;1;2\n1;2;3\n", encoding="utf-8")
    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("doc", encoding="utf-8")
    adapter = CapturingAdapter()

    run_pipeline(
        inputs=PipelineInputs(
            csv_path=csv_path,
            parameters_path=params,
            documentation_path=docs,
            output_dir=tmp_path / "out",
            model="fake-model",
            metric_pattern="mean-incum",
            metric_description="weekly milk",
            skip_summarization=True,
            evidence_mode="plot+stats",
        ),
        prompts=PromptsConfig(
            context_prompt="Context {parameters} {documentation}",
            trend_prompt="Trend {description}",
        ),
        adapter=adapter,
    )

    trend_request = adapter.requests[-1]
    assert trend_request.image_b64 is not None
    assert "| time_step | mean | std | min | max | median |" in trend_request.user_prompt()
