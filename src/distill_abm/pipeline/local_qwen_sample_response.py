"""Response parsing and failure classification for sampled smoke runs."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

GENERIC_UNAVAILABLE_PATTERNS = (
    "analysis is currently unavailable",
    "requested information cannot be retrieved",
    "please try again later",
    "consult additional resources",
)
MAX_STRUCTURED_SMOKE_RESPONSE_CHARS = 50_000


class StructuredSmokeResponseError(ValueError):
    """Structured smoke response failed validation but still produced a trace worth keeping."""

    def __init__(self, message: str, *, trace: Mapping[str, object], prompt: str) -> None:
        super().__init__(message)
        self.trace = dict(trace)
        self.prompt = prompt


class StructuredSmokeText(BaseModel):
    """Structured smoke output schema for one text response."""

    response_text: str


def extract_thinking_text(trace: Mapping[str, object]) -> str:
    """Extract provider reasoning text from a normalized trace payload."""
    response_block = trace.get("response")
    if not isinstance(response_block, dict):
        return ""
    raw_block = response_block.get("raw")
    if not isinstance(raw_block, dict):
        return ""
    message_block = raw_block.get("message")
    if not isinstance(message_block, dict):
        return ""
    thinking = message_block.get("thinking")
    return str(thinking).strip() if isinstance(thinking, str) else ""


def looks_like_context_overflow(error: str) -> bool:
    """Return whether an error message looks like provider-side context overflow."""
    lowered = error.lower()
    return (
        "maximum context length" in lowered
        or "too many input tokens" in lowered
        or "context length" in lowered
        or "input tokens" in lowered
    )


def is_generic_unavailable_response(text: str) -> bool:
    """Return whether a response matches the generic unavailable fallback pattern."""
    lowered = text.lower()
    return all(pattern in lowered for pattern in GENERIC_UNAVAILABLE_PATTERNS[:3]) or any(
        pattern in lowered for pattern in GENERIC_UNAVAILABLE_PATTERNS
    )


def parse_structured_smoke_text(*, raw_text: str, trace: Mapping[str, object], prompt: str) -> str:
    """Parse and validate the final structured smoke text from a provider trace."""
    response_block = trace.get("response")
    raw_response = response_block if isinstance(response_block, dict) else {}
    raw_payload = raw_response.get("raw")
    raw_payload_dict = raw_payload if isinstance(raw_payload, dict) else {}
    try:
        parsed = StructuredSmokeText.model_validate_json(raw_text)
    except Exception as exc:
        thinking_text = extract_thinking_text(trace)
        done_reason = raw_payload_dict.get("done_reason")
        eval_count = raw_payload_dict.get("eval_count")
        if thinking_text:
            if done_reason == "length":
                raise StructuredSmokeResponseError(
                    (
                        "model exhausted max_tokens on thinking before emitting a final structured response"
                        + (f" (eval_count={eval_count})" if isinstance(eval_count, int) else "")
                    ),
                    trace=trace,
                    prompt=prompt,
                ) from exc
            raise StructuredSmokeResponseError(
                "model returned only thinking without a final structured response",
                trace=trace,
                prompt=prompt,
            ) from exc
        raise StructuredSmokeResponseError(
            "model did not return valid structured JSON for the smoke output",
            trace=trace,
            prompt=prompt,
        ) from exc
    done_reason = raw_payload_dict.get("done_reason")
    if done_reason == "length":
        raise StructuredSmokeResponseError(
            "model reached max_tokens before completing the structured response",
            trace=trace,
            prompt=prompt,
        )
    final_text = parsed.response_text.strip()
    if not final_text:
        raise StructuredSmokeResponseError(
            "structured smoke output contained an empty response_text",
            trace=trace,
            prompt=prompt,
        )
    if len(final_text) > MAX_STRUCTURED_SMOKE_RESPONSE_CHARS:
        raise StructuredSmokeResponseError(
            (
                "structured smoke output is pathologically large"
                f" ({len(final_text)} chars > {MAX_STRUCTURED_SMOKE_RESPONSE_CHARS})"
            ),
            trace=trace,
            prompt=prompt,
        )
    if is_generic_unavailable_response(final_text):
        raise StructuredSmokeResponseError(
            "generic unavailable response detected in structured smoke output",
            trace=trace,
            prompt=prompt,
        )
    if isinstance(response_block, dict):
        response_block["parsed_response"] = parsed.model_dump(mode="json")
    return final_text
