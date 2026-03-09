"""Helpers for OpenAI-compatible chat completion payloads and responses."""

from __future__ import annotations

from typing import Any

from distill_abm.llm.adapters.base import LLMRequest


def build_openai_compatible_payload(request: LLMRequest) -> dict[str, Any]:
    """Build an OpenAI-compatible chat-completions payload from a generic request."""
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


def normalize_openai_compatible_completion(completion: Any) -> dict[str, Any]:
    """Normalize OpenAI-compatible SDK responses into plain dictionaries."""
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


def extract_openai_compatible_completion_text(payload: dict[str, Any]) -> str:
    """Extract the assistant text from a normalized OpenAI-compatible completion payload."""
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
