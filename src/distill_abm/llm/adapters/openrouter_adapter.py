"""OpenRouter adapter using the OpenAI-compatible chat completions API."""

from __future__ import annotations

import os
from typing import Any

from distill_abm.llm.adapters.base import LLMAdapter, LLMProviderError, LLMRequest, LLMResponse
from distill_abm.llm.adapters.openai_adapter import _build_payload

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterAdapter(LLMAdapter):
    """Calls OpenRouter through the OpenAI Python SDK interface."""

    provider = "openrouter"

    def __init__(
        self,
        model: str,
        client: Any | None = None,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        api_key: str | None = None,
        site_url: str | None = None,
        app_name: str = "distill-abm",
    ) -> None:
        self.model = model
        self._client = client
        self.base_url = base_url
        self.api_key = api_key
        self.site_url = site_url
        self.app_name = app_name

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload = _build_payload(request)
        payload["model"] = self.model or request.model
        try:
            completion = self._client_for_request().chat.completions.create(**payload)
        except Exception as exc:
            raise LLMProviderError(f"openrouter completion failed: {exc}") from exc
        text = completion.choices[0].message.content or ""
        model = getattr(completion, "model", request.model)
        return LLMResponse(provider=self.provider, model=model, text=text, raw={"provider": "openrouter"})

    def _client_for_request(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except Exception as exc:
            raise LLMProviderError(f"openrouter OpenAI SDK unavailable: {exc}") from exc
        api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMProviderError("openrouter api key missing: set OPENROUTER_API_KEY")
        default_headers: dict[str, str] = {}
        if self.site_url:
            default_headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            default_headers["X-Title"] = self.app_name
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            default_headers=default_headers or None,
        )
        return self._client
