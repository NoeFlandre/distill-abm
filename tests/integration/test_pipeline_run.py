from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill_abm.configs.models import PromptsConfig
from distill_abm.eval.metrics import SummaryScores
from distill_abm.llm.adapters.base import LLMAdapter, LLMRequest, LLMResponse
from distill_abm.pipeline import run as run_module
from distill_abm.pipeline.run import PipelineInputs, run_pipeline


class FakeAdapter(LLMAdapter):
    provider = "fake"

    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(provider="fake", model=request.model, text=f"resp-{len(self.calls)}", raw={})


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1;mean-incum-2\n0;1;2\n1;2;3\n", encoding="utf-8")
    parameters_path = tmp_path / "params.txt"
    parameters_path.write_text("param=1\n", encoding="utf-8")
    documentation_path = tmp_path / "docs.txt"
    documentation_path.write_text("documentation block\n", encoding="utf-8")
    return csv_path, parameters_path, documentation_path


def _prompts() -> PromptsConfig:
    return PromptsConfig(
        context_prompt="Context {parameters} {documentation}",
        trend_prompt="Trend {description} {context}",
        style_features={"role": "ROLE", "example": "EXAMPLE", "insights": "INSIGHTS"},
    )


def test_run_pipeline_summary_only_uses_all_requested_summarizers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path, parameters_path, documentation_path = _write_inputs(tmp_path)

    monkeypatch.setattr(run_module, "summarize_with_bart", lambda text: f"bart::{text}")
    monkeypatch.setattr(run_module, "summarize_with_bert", lambda text: f"bert::{text}")
    monkeypatch.setattr(run_module, "summarize_with_t5", lambda text: f"t5::{text}")
    monkeypatch.setattr(run_module, "summarize_with_longformer_ext", lambda text: f"led::{text}")

    adapter = FakeAdapter()
    result = run_pipeline(
        inputs=PipelineInputs(
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            output_dir=tmp_path / "out",
            model="fake-model",
            metric_pattern="mean-incum",
            metric_description="weekly milk",
            text_source_mode="summary_only",
            evidence_mode="plot+table",
            summarizers=("bart", "bert", "t5", "longformer_ext"),
        ),
        prompts=_prompts(),
        adapter=adapter,
    )

    assert result.trend_summary_response is not None
    assert "bart::resp-2" in result.trend_summary_response
    assert "bert::resp-2" in result.trend_summary_response
    assert "t5::resp-2" in result.trend_summary_response
    assert "led::resp-2" in result.trend_summary_response

    assert result.metadata_path is not None
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["inputs"]["text_source_mode"] == "summary_only"
    assert metadata["inputs"]["selected_text_source"] == "summary_only"
    assert metadata["inputs"]["summarizers"] == ["bart", "bert", "t5", "longformer_ext"]


def test_run_pipeline_full_text_only_bypasses_summarizers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path, parameters_path, documentation_path = _write_inputs(tmp_path)

    def _fail(_: str) -> str:
        raise AssertionError("summarizer should not be called for full_text_only")

    monkeypatch.setattr(run_module, "summarize_with_bart", _fail)
    monkeypatch.setattr(run_module, "summarize_with_bert", _fail)
    monkeypatch.setattr(run_module, "summarize_with_t5", _fail)
    monkeypatch.setattr(run_module, "summarize_with_longformer_ext", _fail)

    adapter = FakeAdapter()
    result = run_pipeline(
        inputs=PipelineInputs(
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            output_dir=tmp_path / "out",
            model="fake-model",
            metric_pattern="mean-incum",
            metric_description="weekly milk",
            text_source_mode="full_text_only",
            evidence_mode="plot",
        ),
        prompts=_prompts(),
        adapter=adapter,
    )

    assert result.trend_summary_response is None
    assert result.trend_response == result.trend_full_response

    assert result.metadata_path is not None
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["inputs"]["text_source_mode"] == "full_text_only"
    assert metadata["inputs"]["selected_text_source"] == "full_text_only"


def test_run_pipeline_table_mode_uses_text_only_evidence(tmp_path: Path) -> None:
    csv_path, parameters_path, documentation_path = _write_inputs(tmp_path)
    adapter = FakeAdapter()

    run_pipeline(
        inputs=PipelineInputs(
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            output_dir=tmp_path / "out",
            model="fake-model",
            metric_pattern="mean-incum",
            metric_description="weekly milk",
            text_source_mode="full_text_only",
            evidence_mode="table",
        ),
        prompts=_prompts(),
        adapter=adapter,
    )

    assert len(adapter.calls) == 2
    assert adapter.calls[1].image_b64 is None


def test_run_pipeline_uses_scoring_reference_file_when_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path, parameters_path, documentation_path = _write_inputs(tmp_path)
    reference_path = tmp_path / "ground_truth.txt"
    reference_path.write_text("HUMAN-GROUND-TRUTH", encoding="utf-8")

    references: list[str] = []

    def fake_score_summary(reference: str, candidate: str) -> SummaryScores:
        references.append(reference)
        return SummaryScores(
            token_f1=0.5,
            precision=0.5,
            recall=0.5,
            bleu=0.5,
            meteor=0.5,
            rouge1=0.5,
            rouge2=0.5,
            rouge_l=0.5,
            flesch_reading_ease=50.0,
            reference_length=1,
            candidate_length=1,
        )

    monkeypatch.setattr(run_module, "score_summary", fake_score_summary)

    run_pipeline(
        inputs=PipelineInputs(
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            output_dir=tmp_path / "out",
            model="fake-model",
            metric_pattern="mean-incum",
            metric_description="weekly milk",
            text_source_mode="summary_only",
            evidence_mode="plot",
            scoring_reference_path=reference_path,
        ),
        prompts=_prompts(),
        adapter=FakeAdapter(),
    )

    assert references
    assert all(reference == "HUMAN-GROUND-TRUTH" for reference in references)


def test_run_pipeline_resume_existing_reuses_artifacts(tmp_path: Path) -> None:
    csv_path, parameters_path, documentation_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "out"
    prompts = _prompts()
    adapter = FakeAdapter()

    first = run_pipeline(
        inputs=PipelineInputs(
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            output_dir=output_dir,
            model="fake-model",
            metric_pattern="mean-incum",
            metric_description="weekly milk",
            text_source_mode="full_text_only",
            evidence_mode="plot",
            resume_existing=False,
        ),
        prompts=prompts,
        adapter=adapter,
    )

    calls_after_first = len(adapter.calls)
    second = run_pipeline(
        inputs=PipelineInputs(
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            output_dir=output_dir,
            model="fake-model",
            metric_pattern="mean-incum",
            metric_description="weekly milk",
            text_source_mode="full_text_only",
            evidence_mode="plot",
            resume_existing=True,
        ),
        prompts=prompts,
        adapter=adapter,
    )

    assert len(adapter.calls) == calls_after_first
    assert first.report_csv == second.report_csv
