from pathlib import Path

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
