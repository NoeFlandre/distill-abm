"""Deterministic adapter for offline testing and dry runs."""

from __future__ import annotations

from distill_abm.llm.adapters.base import LLMAdapter, LLMRequest, LLMResponse


class EchoAdapter(LLMAdapter):
    """Echoes prompt payloads so pipelines can run without external APIs."""

    provider = "echo"

    def __init__(self, model: str) -> None:
        self.model = model

    def complete(self, request: LLMRequest) -> LLMResponse:
        text = request.user_prompt()
        return LLMResponse(provider=self.provider, model=self.model or request.model, text=text, raw={})
