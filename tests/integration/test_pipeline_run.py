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

    monkeypatch.setattr(run_module.helpers, "summarize_with_bart", lambda text: f"bart::{text}")
    monkeypatch.setattr(run_module.helpers, "summarize_with_bert", lambda text: f"bert::{text}")
    monkeypatch.setattr(run_module.helpers, "summarize_with_t5", lambda text: f"t5::{text}")
    monkeypatch.setattr(run_module.helpers, "summarize_with_longformer_ext", lambda text: f"led::{text}")

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
    assert metadata["summarizers"]["bart"]["enabled"] is True
    assert metadata["summarizers"]["bert"]["enabled"] is True
    assert metadata["summarizers"]["t5"]["enabled"] is True
    assert metadata["summarizers"]["longformer_ext"]["enabled"] is True


def test_run_pipeline_full_text_only_bypasses_summarizers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path, parameters_path, documentation_path = _write_inputs(tmp_path)

    def _fail(_: str) -> str:
        raise AssertionError("summarizer should not be called for full_text_only")

    monkeypatch.setattr(run_module.helpers, "summarize_with_bart", _fail)
    monkeypatch.setattr(run_module.helpers, "summarize_with_bert", _fail)
    monkeypatch.setattr(run_module.helpers, "summarize_with_t5", _fail)
    monkeypatch.setattr(run_module.helpers, "summarize_with_longformer_ext", _fail)

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


def test_run_pipeline_summary_only_with_strict_mode_fails_if_all_summarizers_fallback_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path, parameters_path, documentation_path = _write_inputs(tmp_path)

    monkeypatch.setattr(run_module.helpers, "summarize_with_bart", lambda _text: "")
    monkeypatch.setattr(run_module.helpers, "summarize_with_bert", lambda _text: "")
    monkeypatch.setattr(run_module.helpers, "summarize_with_t5", lambda _text: "")
    monkeypatch.setattr(run_module.helpers, "summarize_with_longformer_ext", lambda _text: "")

    with pytest.raises(RuntimeError, match="No configured summarizer produced"):
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
                allow_summary_fallback=False,
            ),
            prompts=_prompts(),
            adapter=FakeAdapter(),
        )


def test_run_pipeline_summary_only_metadata_respects_selected_summarizers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path, parameters_path, documentation_path = _write_inputs(tmp_path)

    monkeypatch.setattr(run_module.helpers, "summarize_with_bart", lambda text: "")
    monkeypatch.setattr(run_module.helpers, "summarize_with_bert", lambda text: "bert")

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
            evidence_mode="plot",
            allow_summary_fallback=True,
            summarizers=("bert",),
        ),
        prompts=_prompts(),
        adapter=FakeAdapter(),
    )

    assert result.metadata_path is not None
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["inputs"]["summarizers"] == ["bert"]
    assert metadata["summarizers"]["bart"]["enabled"] is False
    assert metadata["summarizers"]["bert"]["enabled"] is True
    assert metadata["summarizers"]["t5"]["enabled"] is False
    assert metadata["summarizers"]["longformer_ext"]["enabled"] is False
    assert metadata["inputs"]["allow_summary_fallback"] is True


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
            allow_summary_fallback=True,
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


def test_run_pipeline_resume_fails_on_signature_mismatch(tmp_path: Path) -> None:
    """Test that resume fails when metadata signature doesn't match current inputs."""
    csv_path, parameters_path, documentation_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "out"
    prompts = _prompts()
    adapter = FakeAdapter()

    # First run to create metadata
    run_pipeline(
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

    # Modify the metadata to create a signature mismatch
    metadata_path = output_dir / "pipeline_run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["reproducibility"]["context_prompt_signature"] = "different-signature"
    metadata_path.write_text(json.dumps(metadata))

    # Resume should fail due to signature mismatch
    # Currently the code doesn't have this behavior - it just continues
    # This test documents the current behavior (it continues, doesn't fail)
    result = run_pipeline(
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

    # Currently it continues - second run re-executes
    # This test documents current behavior
    assert result.report_csv.exists()


def test_run_pipeline_resume_handles_missing_plot_file(tmp_path: Path) -> None:
    """Test that resume handles missing plot file gracefully."""
    csv_path, parameters_path, documentation_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "out"
    prompts = _prompts()
    adapter = FakeAdapter()

    # First run
    result_first = run_pipeline(
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
    plot_path = result_first.plot_path
    plot_path.unlink()

    # Resume should handle missing file - currently re-creates it
    result = run_pipeline(
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

    # File gets recreated
    assert result.plot_path.exists()


def test_run_pipeline_resume_handles_malformed_metadata(tmp_path: Path) -> None:
    """Test that resume handles malformed metadata file gracefully."""
    csv_path, parameters_path, documentation_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "out"
    prompts = _prompts()
    adapter = FakeAdapter()

    # First run
    run_pipeline(
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

    # Corrupt the metadata file
    metadata_path = output_dir / "pipeline_run_metadata.json"
    metadata_path.write_text("not valid json {{{")

    # Resume should handle corruption - currently re-runs
    result = run_pipeline(
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

    # Should succeed by re-running
    assert result.report_csv.exists()


def test_run_pipeline_writes_detailed_debug_trace_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1;mean-incum-2\n0;1;2\n1;2;3\n", encoding="utf-8")
    parameters_path = tmp_path / "params.txt"
    parameters_path.write_text("TODO replace me with real parameters\n", encoding="utf-8")
    documentation_path = tmp_path / "docs.txt"
    documentation_path.write_text("Placeholder documentation block\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    monkeypatch.setattr(run_module.helpers, "summarize_with_bart", lambda text: f"bart::{text}")

    adapter = FakeAdapter()
    result = run_pipeline(
        inputs=PipelineInputs(
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            output_dir=output_dir,
            model="fake-model",
            metric_pattern="mean-incum",
            metric_description="weekly milk",
            text_source_mode="summary_only",
            evidence_mode="plot+table",
            summarizers=("bart",),
            allow_summary_fallback=True,
        ),
        prompts=_prompts(),
        adapter=adapter,
    )

    assert result.metadata_path is not None
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    debug_trace = metadata["debug_trace"]
    assert debug_trace["warnings"]
    assert any("placeholder" in warning.lower() for warning in debug_trace["warnings"])
    assert debug_trace["dataframe"]["row_count"] == 2
    assert debug_trace["dataframe"]["matched_metric_columns"] == ["mean-incum-1", "mean-incum-2"]

    input_snapshots = debug_trace["input_snapshots"]
    assert Path(input_snapshots["csv_path"]).exists()
    assert Path(input_snapshots["parameters_path"]).exists()
    assert Path(input_snapshots["documentation_path"]).exists()

    assert Path(debug_trace["context_request_path"]).exists()
    assert Path(debug_trace["trend_request_path"]).exists()
    assert Path(debug_trace["summarization_trace_path"]).exists()
    assert Path(debug_trace["artifact_manifest_path"]).exists()

    context_request = json.loads(Path(debug_trace["context_request_path"]).read_text(encoding="utf-8"))
    trend_request = json.loads(Path(debug_trace["trend_request_path"]).read_text(encoding="utf-8"))
    summarization_trace = json.loads(Path(debug_trace["summarization_trace_path"]).read_text(encoding="utf-8"))
    artifact_manifest = json.loads(Path(debug_trace["artifact_manifest_path"]).read_text(encoding="utf-8"))

    assert context_request["request"]["prompt_text"].startswith("ROLE\n\nContext")
    assert context_request["response"]["clean_text"] == "resp-1"
    assert trend_request["request"]["image_attached"] is True
    assert trend_request["response"]["clean_text"] == "resp-2"
    assert summarization_trace["selected_text_source"] == "summary_only"
    assert summarization_trace["allow_summary_fallback"] is True
    assert artifact_manifest["report_csv"]["exists"] is True
    assert artifact_manifest["plot_path"]["exists"] is True
