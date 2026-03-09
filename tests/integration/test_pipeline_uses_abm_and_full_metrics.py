from __future__ import annotations

from pathlib import Path

import pytest

from distill_abm.configs.loader import load_abm_config
from distill_abm.configs.models import PromptsConfig
from distill_abm.eval.metrics import SummaryScores
from distill_abm.llm.adapters.base import LLMAdapter, LLMRequest, LLMResponse
from distill_abm.pipeline import run as run_module
from distill_abm.pipeline.run import PipelineInputs, run_pipeline


class FakeAdapter(LLMAdapter):
    provider = "fake"

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(provider="fake", model=request.model, text=request.user_prompt(), raw={})


def test_pipeline_outputs_full_metrics_with_abm_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = tmp_path / "sim.csv"
    csv.write_text("tick;mean-incum-1\n0;1\n1;2\n", encoding="utf-8")

    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("d=1", encoding="utf-8")
    monkeypatch.setattr(
        run_module,
        "score_summary",
        lambda reference, candidate: SummaryScores(
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
        ),
    )

    abm = load_abm_config(Path("configs/abms/milk_consumption.yaml"))
    result = run_pipeline(
        inputs=PipelineInputs(
            csv_path=csv,
            parameters_path=params,
            documentation_path=docs,
            output_dir=tmp_path / "out",
            model="fake-model",
            metric_pattern=abm.metric_pattern,
            metric_description=abm.metric_description,
            allow_summary_fallback=True,
        ),
        prompts=PromptsConfig(
            context_prompt="Context {parameters} {documentation}",
            trend_prompt="Trend {description}",
        ),
        adapter=FakeAdapter(),
    )

    assert result.bleu >= 0.0
    assert result.rouge_l >= 0.0
    assert isinstance(result.flesch_reading_ease, float)
