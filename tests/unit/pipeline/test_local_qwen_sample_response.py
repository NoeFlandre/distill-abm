from __future__ import annotations

import pytest

from distill_abm.pipeline.local_qwen_sample_response import (
    StructuredSmokeResponseError,
    extract_thinking_text,
    is_generic_unavailable_response,
    looks_like_context_overflow,
    parse_structured_smoke_text,
)


def test_extract_thinking_text_reads_nested_message_field() -> None:
    trace = {"response": {"raw": {"message": {"thinking": "reasoning"}}}}

    assert extract_thinking_text(trace) == "reasoning"


def test_looks_like_context_overflow_matches_provider_messages() -> None:
    assert looks_like_context_overflow("This model's maximum context length is 131072 tokens.") is True
    assert looks_like_context_overflow("other failure") is False


def test_is_generic_unavailable_response_flags_known_fallback_text() -> None:
    text = (
        "The analysis is currently unavailable. "
        "The requested information cannot be retrieved or interpreted at this time. "
        "Please try again later or consult additional resources for further details."
    )

    assert is_generic_unavailable_response(text) is True


def test_parse_structured_smoke_text_returns_final_text_for_valid_payload() -> None:
    trace = {"response": {"raw": {"message": {"content": '{\"response_text\":\"ok\"}'}, "done_reason": "stop"}}}

    parsed_text = parse_structured_smoke_text(raw_text='{"response_text":"ok"}', trace=trace, prompt="prompt")

    assert parsed_text == "ok"


def test_parse_structured_smoke_text_rejects_thinking_only_length_stop() -> None:
    trace = {
        "response": {
            "raw": {
                "message": {"content": "", "thinking": "reasoning only"},
                "done_reason": "length",
                "eval_count": 1024,
            }
        }
    }

    with pytest.raises(StructuredSmokeResponseError, match="thinking"):
        parse_structured_smoke_text(raw_text="", trace=trace, prompt="prompt")


def test_parse_structured_smoke_text_rejects_pathologically_large_response() -> None:
    trace = {"response": {"raw": {"message": {"content": '{"response_text":"x"}'}, "done_reason": "stop"}}}
    oversized = '{"response_text":"' + ("x" * 50001) + '"}'

    with pytest.raises(StructuredSmokeResponseError, match="pathologically large"):
        parse_structured_smoke_text(raw_text=oversized, trace=trace, prompt="prompt")
