"""LLM-backed qualitative coverage/faithfulness evaluation helpers."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from distill_abm.configs.models import PromptsConfig
from distill_abm.eval.qualitative import extract_coverage_score, extract_faithfulness_score
from distill_abm.llm.adapters.base import LLMAdapter, LLMMessage, LLMRequest
from distill_abm.llm.request_defaults import resolve_request_temperature

QualitativeMetric = Literal["coverage", "faithfulness"]


class QualitativeScoreResult(BaseModel):
    """Structured qualitative evaluation output for CLI/API consumers."""

    score: int
    reasoning: str
    model: str


def evaluate_qualitative_score(
    summary: str,
    source: str,
    metric: QualitativeMetric,
    model: str,
    prompts: PromptsConfig,
    adapter: LLMAdapter,
    source_image_path: Path | None = None,
) -> QualitativeScoreResult:
    """Calls an LLM for qualitative evaluation and extracts a 1-5 score."""
    prompt = _prompt_for_metric(prompts=prompts, metric=metric, summary=summary, source=source)
    request = LLMRequest(
        model=model,
        messages=[LLMMessage(role="user", content=prompt)],
        temperature=resolve_request_temperature(adapter.provider),
        image_b64=_encode_image_if_present(source_image_path),
    )
    response = adapter.complete(request)
    reasoning = response.text.strip()
    score = _extract_score(metric=metric, response_text=reasoning)
    return QualitativeScoreResult(score=score, reasoning=reasoning, model=response.model)


def _prompt_for_metric(prompts: PromptsConfig, metric: QualitativeMetric, summary: str, source: str) -> str:
    template = prompts.coverage_eval_prompt if metric == "coverage" else prompts.faithfulness_eval_prompt
    return template.format(summary=summary, source=source)


def _extract_score(metric: QualitativeMetric, response_text: str) -> int:
    raw = extract_coverage_score(response_text) if metric == "coverage" else extract_faithfulness_score(response_text)
    if not raw:
        raise ValueError(f"could not extract {metric} score from model response")
    score = int(raw)
    if score < 1 or score > 5:
        raise ValueError(f"extracted {metric} score out of range: {score}")
    return score


def _encode_image_if_present(image_path: Path | None) -> str | None:
    if image_path is None:
        return None
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")
