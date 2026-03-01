"""Janus-Pro adapter using an HTTP inference endpoint."""

from __future__ import annotations

from typing import Protocol

from distill_abm.llm.adapters.base import LLMAdapter, LLMProviderError, LLMRequest, LLMResponse


class JanusClient(Protocol):
    """Minimal protocol to keep JanusAdapter testable without heavy dependencies."""

    def generate(
        self,
        prompt: str,
        image_b64: str | None,
        model: str,
        max_tokens: int | None,
        temperature: float | None,
    ) -> str: ...


class HttpJanusClient:
    """Default Janus client that posts inference payloads to a local endpoint."""

    def __init__(self, endpoint: str = "http://localhost:8000/generate") -> None:
        self.endpoint = endpoint

    def generate(
        self,
        prompt: str,
        image_b64: str | None,
        model: str,
        max_tokens: int | None,
        temperature: float | None,
    ) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "image_b64": image_b64,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            import requests

            response = requests.post(self.endpoint, json=payload, timeout=120)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise LLMProviderError(f"janus endpoint error: {exc}") from exc
        text = body.get("text") or body.get("response") or ""
        return str(text)


class JanusAdapter(LLMAdapter):
    """Runs Janus multimodal generation behind the shared adapter contract."""

    provider = "janus"

    def __init__(self, model: str, client: JanusClient | None = None) -> None:
        self.model = model
        self.client = client or HttpJanusClient()

    def complete(self, request: LLMRequest) -> LLMResponse:
        prompt = request.user_prompt()
        try:
            text = self.client.generate(
                prompt=prompt,
                image_b64=request.image_b64,
                model=self.model or request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"janus completion failed: {exc}") from exc
        return LLMResponse(provider=self.provider, model=self.model or request.model, text=text, raw={})
