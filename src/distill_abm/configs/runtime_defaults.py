"""Runtime-defaults resolver for centralized CLI and request configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from distill_abm.configs.loader import ConfigError, load_runtime_defaults_config
from distill_abm.configs.models import RuntimeDefaultsConfig

RUNTIME_DEFAULTS_ENV = "DISTILL_ABM_RUNTIME_DEFAULTS_PATH"
DEFAULT_RUNTIME_DEFAULTS_PATH = Path("configs/runtime_defaults.yaml")


def resolve_runtime_defaults_path() -> Path:
    """Resolve runtime-defaults config path from env override or repository default."""
    override = os.getenv(RUNTIME_DEFAULTS_ENV)
    if override and override.strip():
        return Path(override).expanduser()
    return DEFAULT_RUNTIME_DEFAULTS_PATH


@lru_cache(maxsize=1)
def get_runtime_defaults() -> RuntimeDefaultsConfig:
    """Return centralized runtime defaults with safe fallback to model defaults."""
    path = resolve_runtime_defaults_path()
    try:
        return load_runtime_defaults_config(path)
    except ConfigError:
        return RuntimeDefaultsConfig()


def clear_runtime_defaults_cache() -> None:
    """Clear cached runtime defaults (used by tests and dynamic reload flows)."""
    get_runtime_defaults.cache_clear()
