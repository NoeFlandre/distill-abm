"""OpenAI chat completion adapter with optional image support."""

from __future__ import annotations

from typing import Any

from distill_abm.llm.adapters.base import LLMAdapter, LLMProviderError, LLMRequest, LLMResponse
from distill_abm.llm.adapters.timeout_utils import run_with_timeout


class OpenAIAdapter(LLMAdapter):
    """Calls the OpenAI chat API through a stable adapter contract."""

    provider = "openai"

    def __init__(self, model: str, client: Any | None = None, timeout_seconds: float = 120.0) -> None:
        self.model = model
        self._client = client
        self.timeout_seconds = timeout_seconds

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload = _build_payload(request)
        payload["model"] = self.model or request.model
        completion = run_with_timeout(
            timeout_seconds=self.timeout_seconds,
            label="openai completion",
            fn=lambda: self._client_for_request().chat.completions.create(**payload),
        )
        text = completion.choices[0].message.content or ""
        model = getattr(completion, "model", request.model)
        return LLMResponse(provider=self.provider, model=model, text=text, raw={"provider": "openai"})

    def _client_for_request(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except Exception as exc:
            raise LLMProviderError(f"openai package unavailable: {exc}") from exc
        self._client = OpenAI()
        return self._client


def _build_payload(request: LLMRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    if request.image_b64:
        payload["messages"] = _with_image_message(payload["messages"], request.image_b64)
    return payload


def _with_image_message(messages: list[dict[str, str]], image_b64: str) -> list[dict[str, Any]]:
    if not messages:
        return [{"role": "user", "content": [{"type": "text", "text": ""}, _image_block(image_b64)]}]
    updated: list[dict[str, Any]] = [*messages]
    for index in range(len(updated) - 1, -1, -1):
        if updated[index]["role"] != "user":
            continue
        text = str(updated[index]["content"])
        updated[index] = {"role": "user", "content": [{"type": "text", "text": text}, _image_block(image_b64)]}
        break
    return updated


def _image_block(image_b64: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
