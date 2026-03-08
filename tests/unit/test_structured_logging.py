from __future__ import annotations

import json
import logging
from pathlib import Path

from distill_abm.structured_logging import JsonLogFormatter, attach_json_log_file, get_logger


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
    record.event_data = {"provider": "ollama", "model": "qwen3.5:0.8b"}

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "distill_abm.test"
    assert payload["message"] == "hello"
    assert payload["provider"] == "ollama"
    assert payload["model"] == "qwen3.5:0.8b"


def test_attach_json_log_file_writes_json_lines(tmp_path: Path) -> None:
    log_path = attach_json_log_file(tmp_path / "run.log.jsonl")
    logger = get_logger("distill_abm.test.file")
    logger.info("smoke_event", extra={"event_data": {"case_id": "case-1"}})

    contents = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert contents
    payload = json.loads(contents[-1])
    assert payload["message"] == "smoke_event"
    assert payload["case_id"] == "case-1"
