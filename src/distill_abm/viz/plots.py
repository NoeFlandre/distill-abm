"""Plot generation from repeated ABM simulation runs."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from distill_abm.ingest.csv_ingest import matching_columns


class PlotError(RuntimeError):
    """Raised when plotting inputs do not contain requested metrics."""


def plot_metric_bundle(
    frame: pd.DataFrame,
    include_pattern: str,
    output_dir: Path,
    title: str,
    y_label: str,
    exclude_pattern: str | None = None,
) -> Path:
    """Creates one metric plot so LLM prompts can consume image evidence."""
    columns = matching_columns(list(frame.columns), include_pattern, exclude_pattern)
    if not columns:
        raise PlotError(f"no columns found for pattern '{include_pattern}'")
    numeric = _numeric_frame(frame, columns)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_slug(include_pattern)}.png"
    _draw_plot(numeric, output_path, title=title, y_label=y_label)
    return output_path


def _numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    converted = frame[columns].apply(pd.to_numeric, errors="coerce")
    return cast(pd.DataFrame, converted.ffill().fillna(0.0))


def _draw_plot(data: pd.DataFrame, output_path: Path, title: str, y_label: str) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    for column in data.columns:
        axis.plot(data.index, data[column], alpha=0.25, linewidth=1.0)
    axis.plot(data.index, data.mean(axis=1), color="black", linewidth=2.0, label="mean")
    axis.set_title(title)
    axis.set_xlabel("time step")
    axis.set_ylabel(y_label)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _slug(pattern: str) -> str:
    return pattern.strip().replace(" ", "_").replace("/", "_").lower()
