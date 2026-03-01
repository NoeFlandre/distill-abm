"""Ollama chat adapter for local models such as DeepSeek-R1."""

from __future__ import annotations

from typing import Any

from distill_abm.llm.adapters.base import LLMAdapter, LLMProviderError, LLMRequest, LLMResponse


class OllamaAdapter(LLMAdapter):
    """Calls Ollama's chat endpoint and normalizes responses."""

    provider = "ollama"

    def __init__(self, model: str, client: Any | None = None, host: str | None = None) -> None:
        self.model = model
        self.host = host
        self._client = client

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "model": self.model or request.model,
            "messages": _build_messages(request),
            "options": {"temperature": request.temperature},
        }
        try:
            result = self._client_for_request().chat(**payload)
        except Exception as exc:
            raise LLMProviderError(f"ollama completion failed: {exc}") from exc
        content = result.get("message", {}).get("content", "")
        model = str(result.get("model", request.model))
        return LLMResponse(provider=self.provider, model=model, text=content, raw=result)

    def _client_for_request(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import ollama
        except Exception as exc:
            raise LLMProviderError(f"ollama package unavailable: {exc}") from exc
        self._client = ollama.Client(host=self.host) if self.host else ollama.Client()
        return self._client


def _build_messages(request: LLMRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": message.role, "content": message.content} for message in request.messages
    ]
    if not request.image_b64:
        return messages
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "user":
            messages[index]["images"] = [request.image_b64]
            break
    return messages
