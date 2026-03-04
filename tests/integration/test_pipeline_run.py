import csv
import json
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


def test_run_pipeline_defaults_to_both_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1;mean-incum-2\n0;1;2\n1;2;3\n", encoding="utf-8")

    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("doc", encoding="utf-8")

    monkeypatch.setattr("distill_abm.pipeline.run.summarize_with_bart", lambda text: f"bart::{text}")
    monkeypatch.setattr("distill_abm.pipeline.run.summarize_with_bert", lambda text: f"bert::{text}")

    inputs = PipelineInputs(
        csv_path=csv_path,
        parameters_path=params,
        documentation_path=docs,
        output_dir=tmp_path / "out",
        model="fake-model",
        metric_pattern="mean-incum",
        metric_description="weekly milk",
    )
    assert inputs.summarization_mode == "both"
    assert inputs.score_on == "both"

    adapter = FakeAdapter()
    result = run_pipeline(
        inputs=inputs,
        prompts=PromptsConfig(
            context_prompt="Context {parameters} {documentation}",
            trend_prompt="Trend {description}",
        ),
        adapter=adapter,
    )

    assert result.full_scores is not None
    assert result.summary_scores is not None
    with result.report_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    assert "full_token_f1" in header
    assert "summary_token_f1" in header


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


def test_run_pipeline_can_emit_both_full_and_summary_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
            summarization_mode="both",
            score_on="both",
            skip_summarization=False,
        ),
        prompts=PromptsConfig(
            context_prompt="Context {parameters} {documentation}",
            trend_prompt="Trend {description}",
        ),
        adapter=adapter,
    )

    assert result.trend_full_response == "resp-2"
    assert result.trend_summary_response == "bart::resp-2\nbert::resp-2"
    assert result.trend_response == "bart::resp-2\nbert::resp-2"
    assert result.summary_scores is not None
    assert result.token_f1 == result.summary_scores.token_f1
    with result.report_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    assert "full_token_f1" in header
    assert "summary_token_f1" in header
    assert "trend_full_response" in header
    assert "trend_summary_response" in header


def test_run_pipeline_scores_full_when_selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
            summarization_mode="both",
            score_on="full",
        ),
        prompts=PromptsConfig(
            context_prompt="Context {parameters} {documentation}",
            trend_prompt="Trend {description}",
        ),
        adapter=adapter,
    )

    assert result.trend_response == "resp-2"
    assert result.token_f1 >= 0.0


def test_run_pipeline_summary_only_mode_outputs_summary_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
            summarization_mode="summary",
            score_on="summary",
        ),
        prompts=PromptsConfig(
            context_prompt="Context {parameters} {documentation}",
            trend_prompt="Trend {description}",
        ),
        adapter=adapter,
    )

    assert result.trend_full_response == "resp-2"
    assert result.trend_summary_response == "bart::resp-2\nbert::resp-2"
    assert result.trend_response == "bart::resp-2\nbert::resp-2"
    assert result.summary_scores is None
    assert result.full_scores is None
    with result.report_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    assert "full_token_f1" not in header
    assert "summary_token_f1" not in header
    assert "trend_full_response" not in header
    assert "trend_summary_response" not in header


def test_run_pipeline_full_only_mode_outputs_full_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
            summarization_mode="full",
            score_on="full",
        ),
        prompts=PromptsConfig(
            context_prompt="Context {parameters} {documentation}",
            trend_prompt="Trend {description}",
        ),
        adapter=adapter,
    )

    assert result.trend_full_response == "resp-2"
    assert result.trend_summary_response is None
    assert result.trend_response == "resp-2"


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
    assert all(request.temperature == 0.5 for request in adapter.requests)
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


def test_run_pipeline_writes_reproducibility_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
            summarization_mode="both",
            score_on="both",
        ),
        prompts=PromptsConfig(
            context_prompt="Context {parameters} {documentation}",
            trend_prompt="Trend {description}",
        ),
        adapter=adapter,
    )

    assert result.metadata_path is not None
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert metadata["inputs"]["metric_pattern"] == "mean-incum"
    assert metadata["inputs"]["metric_description"] == "weekly milk"
    assert metadata["artifacts"]["plot_path"] == str(result.plot_path)
    assert metadata["artifacts"]["report_csv"] == str(result.report_csv)
    assert metadata["llm"]["provider"] == "fake"
    assert metadata["llm"]["request"]["temperature"] == 0.5
    assert metadata["llm"]["request"]["max_tokens"] == 1000
    assert len(metadata["reproducibility"]["context_prompt_signature"]) == 64
    assert len(metadata["reproducibility"]["trend_prompt_signature"]) == 64
    assert metadata["reproducibility"]["delimiter"] == ","
    assert metadata["summarizers"]["longformer_like"]["model"] == "allenai/led-base-16384"
    assert metadata["scores"]["selected_scores"]["token_f1"] == result.token_f1
    assert metadata["scores"]["summary_scores"] is not None


def test_run_pipeline_applies_notebook_style_prompt_parts_and_plot_description(tmp_path: Path) -> None:
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
            evidence_mode="plot",
            plot_description="PLOT DESCRIPTION",
        ),
        prompts=PromptsConfig(
            context_prompt="Context block:\n{parameters}\n{documentation}",
            trend_prompt="Trend block:\n{description}\n{context}",
            style_features={
                "role": "ROLE",
                "example": "EXAMPLE",
                "insights": "INSIGHTS",
            },
        ),
        adapter=adapter,
    )

    assert len(adapter.requests) == 2
    context_request = adapter.requests[0]
    trend_request = adapter.requests[1]

    assert context_request.user_prompt().startswith("ROLE\n\nContext block:")
    assert "p=1" in context_request.user_prompt()
    assert "doc" in context_request.user_prompt()

    user_prompt = trend_request.user_prompt()
    assert user_prompt.startswith("ROLE\n\nTrend block:")
    assert "\n\nEXAMPLE" in user_prompt
    assert "\n\nINSIGHTS" in user_prompt
    assert user_prompt.endswith("PLOT DESCRIPTION")


def test_run_pipeline_uses_requested_additional_summarizers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1;mean-incum-2\n0;1;2\n1;2;3\n", encoding="utf-8")

    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("doc", encoding="utf-8")

    monkeypatch.setattr("distill_abm.pipeline.run.summarize_with_bart", lambda text: f"bart::{text}")
    monkeypatch.setattr("distill_abm.pipeline.run.summarize_with_bert", lambda text: f"bert::{text}")
    monkeypatch.setattr("distill_abm.pipeline.run.summarize_with_t5", lambda text: f"t5::{text}")
    monkeypatch.setattr("distill_abm.pipeline.run.summarize_with_longformer_ext", lambda text: f"led::{text}")

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
            summarization_mode="both",
            score_on="summary",
            additional_summarizers=("t5", "longformer_ext"),
        ),
        prompts=PromptsConfig(
            context_prompt="Context {parameters} {documentation}",
            trend_prompt="Trend {description}",
        ),
        adapter=adapter,
    )

    assert result.trend_summary_response is not None
    assert "bart::resp-2" in result.trend_summary_response
    assert "bert::resp-2" in result.trend_summary_response
    assert "t5::resp-2" in result.trend_summary_response
    assert "led::resp-2" in result.trend_summary_response
    assert result.metadata_path is not None
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["inputs"]["additional_summarizers"] == ["t5", "longformer_ext"]
    assert metadata["summarizers"]["t5"]["enabled"] is True
    assert metadata["summarizers"]["longformer_like"]["enabled"] is True
