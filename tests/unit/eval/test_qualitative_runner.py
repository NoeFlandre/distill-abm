from __future__ import annotations

from pathlib import Path

import pytest

from distill_abm.configs.models import PromptsConfig
from distill_abm.eval.qualitative_runner import QualitativeScoreResult, evaluate_qualitative_score
from distill_abm.llm.adapters.base import LLMAdapter, LLMRequest, LLMResponse


class FakeAdapter(LLMAdapter):
    provider = "fake"

    def __init__(self, text: str) -> None:
        self.text = text
        self.last_request: LLMRequest | None = None

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.last_request = request
        return LLMResponse(provider=self.provider, model=request.model, text=self.text, raw={})


def _prompts() -> PromptsConfig:
    return PromptsConfig(
        context_prompt="context {parameters} {documentation}",
        trend_prompt="trend {description}",
        coverage_eval_prompt="Coverage eval source={source} summary={summary}",
        faithfulness_eval_prompt="Faithfulness eval source={source} summary={summary}",
    )


def test_evaluate_coverage_returns_structured_result() -> None:
    adapter = FakeAdapter("Coverage score: 4. Reasoning: strong topical overlap.")
    result = evaluate_qualitative_score(
        summary="summary text",
        source="source context",
        metric="coverage",
        model="fake-model",
        prompts=_prompts(),
        adapter=adapter,
    )

    assert isinstance(result, QualitativeScoreResult)
    assert result.score == 4
    assert result.reasoning == "Coverage score: 4. Reasoning: strong topical overlap."
    assert result.model == "fake-model"
    assert adapter.last_request is not None
    assert adapter.last_request.temperature == 0.5


def test_evaluate_faithfulness_passes_image_to_adapter(tmp_path: Path) -> None:
    image = tmp_path / "plot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    adapter = FakeAdapter("Faithfulness score: 5. Reasoning: claims align with source.")
    _ = evaluate_qualitative_score(
        summary="summary text",
        source="source context",
        metric="faithfulness",
        model="fake-model",
        prompts=_prompts(),
        adapter=adapter,
        source_image_path=image,
    )
    assert adapter.last_request is not None
    assert adapter.last_request.image_b64 is not None


def test_evaluate_raises_when_score_not_extractable() -> None:
    adapter = FakeAdapter("No numeric score found.")
    with pytest.raises(ValueError):
        evaluate_qualitative_score(
            summary="summary text",
            source="source context",
            metric="coverage",
            model="fake-model",
            prompts=_prompts(),
            adapter=adapter,
        )
