"""Centralized structured logging helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from distill_abm.configs.loader import ConfigError, load_logging_config
from distill_abm.configs.models import LoggingConfig

_CONFIGURED = False


class JsonLogFormatter(logging.Formatter):
    """Render log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
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


def _load_default_logging_config() -> LoggingConfig:
    try:
        return load_logging_config(Path("configs/logging.yaml"))
    except ConfigError:
        return LoggingConfig()
