"""Ollama chat adapter for local models such as DeepSeek-R1."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from distill_abm.llm.adapters.base import LLMAdapter, LLMProviderError, LLMRequest, LLMResponse


class OllamaAdapter(LLMAdapter):
    """Calls Ollama's chat endpoint and normalizes responses."""

    provider = "ollama"

    def __init__(
        self,
        model: str,
        client: Any | None = None,
        host: str | None = None,
        timeout_seconds: float = 90.0,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout_seconds = timeout_seconds
        self._client = client

    def complete(self, request: LLMRequest) -> LLMResponse:
        options: dict[str, float | int | None] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        ollama_num_ctx = request.metadata.get("ollama_num_ctx")
        if isinstance(ollama_num_ctx, int) and ollama_num_ctx > 0:
            options["num_ctx"] = ollama_num_ctx
        payload = {
            "model": self.model or request.model,
            "messages": _build_messages(request),
            "options": options,
        }
        if "ollama_think" in request.metadata:
            payload["think"] = request.metadata["ollama_think"]
        if "ollama_format" in request.metadata:
            payload["format"] = request.metadata["ollama_format"]
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._client_for_request().chat, **payload)
                result = future.result(timeout=self.timeout_seconds)
        except FuturesTimeoutError as exc:
            raise LLMProviderError(f"ollama completion timed out after {self.timeout_seconds:.1f}s") from exc
        except Exception as exc:
            raise LLMProviderError(f"ollama completion failed: {exc}") from exc
        normalized = _normalize_chat_result(result)
        message = normalized.get("message", {})
        content = str(message.get("content", "")) if isinstance(message, dict) else ""
        model = str(normalized.get("model", request.model))
        return LLMResponse(provider=self.provider, model=model, text=content, raw=normalized)

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


def _normalize_chat_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        payload: dict[str, Any] = result
    elif hasattr(result, "model_dump"):
        payload = dict(result.model_dump())
    elif hasattr(result, "dict"):
        payload = dict(result.dict())
    else:
        payload = {}

    if not payload:
        model = getattr(result, "model", None)
        message_obj = getattr(result, "message", None)
        content = getattr(message_obj, "content", None)
        if model is not None:
            payload["model"] = model
        if content is not None:
            payload["message"] = {"content": content}

    message = payload.get("message")
    if message is not None and not isinstance(message, dict):
        content = getattr(message, "content", None)
        thinking = getattr(message, "thinking", None)
        normalized_message: dict[str, Any] = {}
        if content is not None:
            normalized_message["content"] = content
        if thinking is not None:
            normalized_message["thinking"] = thinking
        if normalized_message:
            payload["message"] = normalized_message
    return payload
