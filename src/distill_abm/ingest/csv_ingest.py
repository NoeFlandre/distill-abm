"""CSV ingestion utilities extracted from notebook preprocessing steps."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class IngestError(RuntimeError):
    """Raised when simulation CSV data cannot be parsed safely."""


def load_simulation_csv(path: Path, delimiter: str = ";") -> pd.DataFrame:
    """Loads simulation outputs with explicit delimiter control for NetLogo exports."""
    try:
        return pd.read_csv(path, delimiter=delimiter)
    except Exception as exc:
        raise IngestError(f"failed to load CSV {path}: {exc}") from exc


def matching_columns(
    columns: list[str],
    include_pattern: str,
    exclude_pattern: str | None = None,
) -> list[str]:
    """Filters candidate metric columns for plotting and trend prompts."""
    filtered = [column for column in columns if include_pattern in column]
    if exclude_pattern is None:
        return filtered
    return [column for column in filtered if exclude_pattern not in column]
