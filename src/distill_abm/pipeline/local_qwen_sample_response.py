"""Response parsing and failure classification for sampled smoke runs."""

from __future__ import annotations

import json
import re
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


def validate_structured_smoke_text_content(text: str) -> str:
    """Validate persisted structured smoke output text before reuse."""
    final_text = text.strip()
    if not final_text:
        raise ValueError("structured smoke output contained an empty response_text")
    if len(final_text) > MAX_STRUCTURED_SMOKE_RESPONSE_CHARS:
        raise ValueError(
            "structured smoke output is pathologically large"
            f" ({len(final_text)} chars > {MAX_STRUCTURED_SMOKE_RESPONSE_CHARS})"
        )
    if is_generic_unavailable_response(final_text):
        raise ValueError("generic unavailable response detected in structured smoke output")
    return final_text


def should_retry_without_structured_output(trace: Mapping[str, object]) -> bool:
    """Return whether a provider trace matches the empty structured-output failure mode."""
    response_block = trace.get("response")
    if not isinstance(response_block, dict):
        return False
    if str(response_block.get("provider", "")).strip().lower() != "openrouter":
        return False
    if response_block.get("clean_text_length") not in {0, "0"}:
        return False
    raw_payload = response_block.get("raw")
    if not isinstance(raw_payload, dict):
        return False
    error_block = raw_payload.get("error")
    if not isinstance(error_block, dict):
        return False
    code = error_block.get("code")
    choices = raw_payload.get("choices")
    return code == 500 and choices is None


def should_retry_without_structured_output_error(*, provider: str, error: str) -> bool:
    """Return whether a raised provider error matches the structured-output failure mode."""
    if provider.strip().lower() != "openrouter":
        return False
    lowered = error.lower()
    return (
        "error code: 500" in lowered
        or "internal server error" in lowered
        or "structured response was empty" in lowered
    )


def _candidate_json_payloads(raw_text: str) -> list[str]:
    stripped = raw_text.strip()
    if not stripped:
        return []
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(stripped[first : last + 1].strip())
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def parse_structured_smoke_text(*, raw_text: str, trace: Mapping[str, object], prompt: str) -> str:
    """Parse and validate the final structured smoke text from a provider trace."""
    response_block = trace.get("response")
    raw_response = response_block if isinstance(response_block, dict) else {}
    raw_payload = raw_response.get("raw")
    raw_payload_dict = raw_payload if isinstance(raw_payload, dict) else {}
    parse_error: Exception | None = None
    parsed: StructuredSmokeText | None = None
    for candidate in _candidate_json_payloads(raw_text):
        try:
            parsed = StructuredSmokeText.model_validate(json.loads(candidate))
            break
        except Exception as exc:
            parse_error = exc
    if parsed is None:
        parse_failure = parse_error if parse_error is not None else ValueError("empty structured smoke response")
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
                ) from parse_failure
            raise StructuredSmokeResponseError(
                "model returned only thinking without a final structured response",
                trace=trace,
                prompt=prompt,
            ) from parse_failure
        raise StructuredSmokeResponseError(
            "model did not return valid structured JSON for the smoke output",
            trace=trace,
            prompt=prompt,
        ) from parse_failure
    done_reason = raw_payload_dict.get("done_reason")
    if done_reason == "length":
        raise StructuredSmokeResponseError(
            "model reached max_tokens before completing the structured response",
            trace=trace,
            prompt=prompt,
        )
    try:
        final_text = validate_structured_smoke_text_content(parsed.response_text)
    except ValueError as exc:
        raise StructuredSmokeResponseError(
            str(exc),
            trace=trace,
            prompt=prompt,
        ) from exc
    if isinstance(response_block, dict):
        response_block["parsed_response"] = parsed.model_dump(mode="json")
    return final_text
