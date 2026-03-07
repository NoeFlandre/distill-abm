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
        normalized = _normalize_completion(completion)
        text = _extract_completion_text(normalized)
        model = str(normalized.get("model", request.model))
        normalized.setdefault("provider", "openai")
        return LLMResponse(provider=self.provider, model=model, text=text, raw=normalized)

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
    schema = request.metadata.get("structured_output_schema")
    if isinstance(schema, dict):
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": str(request.metadata.get("structured_output_name", "structured_output")),
                "schema": schema,
            },
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


def _normalize_completion(completion: Any) -> dict[str, Any]:
    if isinstance(completion, dict):
        return dict(completion)
    if hasattr(completion, "model_dump"):
        return dict(completion.model_dump())
    if hasattr(completion, "dict"):
        return dict(completion.dict())
    payload: dict[str, Any] = {}
    model = getattr(completion, "model", None)
    choices = getattr(completion, "choices", None)
    usage = getattr(completion, "usage", None)
    if model is not None:
        payload["model"] = model
    if choices is not None:
        normalized_choices: list[Any] = []
        if isinstance(choices, list):
            for choice in choices:
                if hasattr(choice, "model_dump"):
                    normalized_choices.append(choice.model_dump())
                elif hasattr(choice, "dict"):
                    normalized_choices.append(choice.dict())
                else:
                    message = getattr(choice, "message", None)
                    finish_reason = getattr(choice, "finish_reason", None)
                    normalized_choice: dict[str, Any] = {}
                    if message is not None:
                        if hasattr(message, "model_dump"):
                            normalized_choice["message"] = message.model_dump()
                        elif hasattr(message, "dict"):
                            normalized_choice["message"] = message.dict()
                        else:
                            content = getattr(message, "content", None)
                            if content is not None:
                                normalized_choice["message"] = {"content": content}
                    if finish_reason is not None:
                        normalized_choice["finish_reason"] = finish_reason
                    normalized_choices.append(normalized_choice if normalized_choice else choice)
            payload["choices"] = normalized_choices
        else:
            payload["choices"] = choices
    if usage is not None:
        payload["usage"] = usage
    return payload


def _extract_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if hasattr(first, "model_dump"):
        first = first.model_dump()
    elif hasattr(first, "dict"):
        first = first.dict()
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if message is not None and hasattr(message, "model_dump"):
        message = message.model_dump()
    elif message is not None and hasattr(message, "dict"):
        message = message.dict()
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
            return "".join(text_parts)
    return ""
