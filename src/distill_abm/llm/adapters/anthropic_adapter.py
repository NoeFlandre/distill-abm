"""Anthropic message API adapter."""

from __future__ import annotations

from typing import Any

from distill_abm.llm.adapters.base import LLMAdapter, LLMProviderError, LLMRequest, LLMResponse


class AnthropicAdapter(LLMAdapter):
    """Calls Claude models while keeping the rest of the pipeline provider-neutral."""

    provider = "anthropic"

    def __init__(self, model: str, client: Any | None = None) -> None:
        self.model = model
        self._client = client

    def complete(self, request: LLMRequest) -> LLMResponse:
        system, messages = _to_anthropic_messages(request)
        payload: dict[str, Any] = {
            "model": self.model or request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if system:
            payload["system"] = system
        try:
            result = self._client_for_request().messages.create(**payload)
        except Exception as exc:
            raise LLMProviderError(f"anthropic completion failed: {exc}") from exc
        text = _extract_text_blocks(result.content)
        model = getattr(result, "model", request.model)
        return LLMResponse(provider=self.provider, model=model, text=text, raw={"provider": "anthropic"})

    def _client_for_request(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except Exception as exc:
            raise LLMProviderError(f"anthropic package unavailable: {exc}") from exc
        self._client = anthropic.Anthropic()
        return self._client


def _to_anthropic_messages(request: LLMRequest) -> tuple[str | None, list[dict[str, Any]]]:
    system = None
    messages: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role == "system":
            system = message.content
            continue
        messages.append({"role": message.role, "content": message.content})
    if request.image_b64:
        messages = _attach_image(messages, request.image_b64)
    return system, messages


def _attach_image(messages: list[dict[str, Any]], image_b64: str) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = [*messages]
    for index in range(len(updated) - 1, -1, -1):
        if updated[index]["role"] != "user":
            continue
        text = str(updated[index]["content"])
        updated[index]["content"] = [
            {"type": "text", "text": text},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
            },
        ]
        break
    return updated


def _extract_text_blocks(blocks: Any) -> str:
    parts = [getattr(block, "text", "") for block in blocks if getattr(block, "type", "") == "text"]
    return "\n".join(part for part in parts if part)
