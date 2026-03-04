"""YAML-backed config loading with strict validation and clear errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

import yaml
from pydantic import ValidationError

from distill_abm.configs.models import (
    ABMConfig,
    EvaluationConfig,
    ExperimentSettings,
    LoggingConfig,
    ModelsConfig,
    PromptsConfig,
    RuntimeDefaultsConfig,
)

ModelType = TypeVar(
    "ModelType",
    ABMConfig,
    EvaluationConfig,
    ExperimentSettings,
    LoggingConfig,
    ModelsConfig,
    PromptsConfig,
    RuntimeDefaultsConfig,
)


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or validated."""


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to read config at {path}: {exc}") from exc
    if not isinstance(content, dict):
        raise ConfigError(f"config at {path} must be a mapping")
    return cast(dict[str, Any], content)


def _load_yaml_model(path: Path, model_class: type[ModelType]) -> ModelType:
    try:
        return model_class.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"invalid config at {path}: {exc}") from exc


def load_models_config(path: Path) -> ModelsConfig:
    """Load and validate the provider model mapping."""
    return _load_yaml_model(path, ModelsConfig)


def load_prompts_config(path: Path) -> PromptsConfig:
    """Load and validate prompt templates."""
    return _load_yaml_model(path, PromptsConfig)


def load_abm_config(path: Path) -> ABMConfig:
    """Load and validate ABM defaults."""
    return _load_yaml_model(path, ABMConfig)


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load and validate evaluation settings."""
    return _load_yaml_model(path, EvaluationConfig)


def load_logging_config(path: Path) -> LoggingConfig:
    """Load and validate logging behavior."""
    return _load_yaml_model(path, LoggingConfig)


def load_experiment_settings(path: Path) -> ExperimentSettings:
    """Load and validate paper-aligned experiment settings."""
    return _load_yaml_model(path, ExperimentSettings)


def load_runtime_defaults_config(path: Path) -> RuntimeDefaultsConfig:
    """Load and validate centralized runtime defaults."""
    return _load_yaml_model(path, RuntimeDefaultsConfig)
