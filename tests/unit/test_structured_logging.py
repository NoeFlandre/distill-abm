from __future__ import annotations

import json
import logging

from distill_abm.structured_logging import JsonLogFormatter


def test_json_log_formatter_includes_event_data() -> None:
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="distill_abm.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.event_data = {"provider": "ollama", "model": "qwen3.5:0.8b"}  # type: ignore[attr-defined]

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "distill_abm.test"
    assert payload["message"] == "hello"
    assert payload["provider"] == "ollama"
    assert payload["model"] == "qwen3.5:0.8b"
