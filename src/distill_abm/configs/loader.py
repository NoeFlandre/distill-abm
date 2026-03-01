"""YAML-backed config loading with strict validation and clear errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from distill_abm.configs.models import ABMConfig, EvaluationConfig, LoggingConfig, ModelsConfig, PromptsConfig


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or validated."""


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to read config at {path}: {exc}") from exc
    if not isinstance(content, dict):
        raise ConfigError(f"config at {path} must be a mapping")
    return content


def load_models_config(path: Path) -> ModelsConfig:
    try:
        return ModelsConfig.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"invalid models config at {path}: {exc}") from exc


def load_prompts_config(path: Path) -> PromptsConfig:
    try:
        return PromptsConfig.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"invalid prompts config at {path}: {exc}") from exc


def load_abm_config(path: Path) -> ABMConfig:
    try:
        return ABMConfig.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"invalid ABM config at {path}: {exc}") from exc


def load_evaluation_config(path: Path) -> EvaluationConfig:
    try:
        return EvaluationConfig.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"invalid evaluation config at {path}: {exc}") from exc


def load_logging_config(path: Path) -> LoggingConfig:
    try:
        return LoggingConfig.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"invalid logging config at {path}: {exc}") from exc
