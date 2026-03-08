"""Centralized structured logging helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from distill_abm.configs.loader import ConfigError, load_logging_config
from distill_abm.configs.models import LoggingConfig

_CONFIGURED = False
_FILE_HANDLERS: set[Path] = set()


class JsonLogFormatter(logging.Formatter):
    """Render log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        event_name = record.getMessage()
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "event": event_name,
            "message": event_name,
        }
        event_data = getattr(record, "event_data", None)
        if isinstance(event_data, dict):
            payload.update(event_data)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def configure_logging(config: LoggingConfig | None = None) -> None:
    """Configure process-wide structured logging once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    resolved = config or _load_default_logging_config()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    logging.basicConfig(level=resolved.level.upper(), handlers=[handler], force=True)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured structured logger."""
    configure_logging()
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, /, level: int = logging.INFO, **event_data: Any) -> None:
    """Write one structured event with a stable `event` field."""
    logger.log(level, event, extra={"event_data": event_data})


def attach_json_log_file(path: Path) -> Path:
    """Attach one persistent JSON-lines log file to the root logger."""
    configure_logging()
    resolved = path.resolve()
    if resolved in _FILE_HANDLERS:
        return resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(resolved, encoding="utf-8")
    handler.setFormatter(JsonLogFormatter())
    logging.getLogger().addHandler(handler)
    _FILE_HANDLERS.add(resolved)
    return resolved


def _load_default_logging_config() -> LoggingConfig:
    try:
        return load_logging_config(Path("configs/logging.yaml"))
    except ConfigError:
        return LoggingConfig()
