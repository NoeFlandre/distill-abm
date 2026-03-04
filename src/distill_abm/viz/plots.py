"""Plot generation from repeated ABM simulation runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from distill_abm.ingest.csv_ingest import matching_columns


class PlotError(RuntimeError):
    """Raised when plotting inputs do not contain requested metrics."""


@dataclass(frozen=True)
class MetricPlotBundle:
    """Describes one notebook-style metric plot request."""

    include_pattern: str
    title: str
    y_label: str
    exclude_pattern: str | None = None
    show_mean_line: bool = True


def plot_metric_bundle(
    frame: pd.DataFrame,
    include_pattern: str,
    output_dir: Path,
    title: str,
    y_label: str,
    exclude_pattern: str | None = None,
    show_mean_line: bool = True,
) -> Path:
    """Creates one metric plot so LLM prompts can consume image evidence."""
    columns = matching_columns(list(frame.columns), include_pattern, exclude_pattern)
    if not columns:
        raise PlotError(f"no columns found for pattern '{include_pattern}'")
    numeric = _numeric_frame(frame, columns)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_slug(include_pattern)}.png"
    _draw_plot(numeric, output_path, title=title, y_label=y_label, show_mean_line=show_mean_line)
    return output_path


def plot_metric_bundles(
    frame: pd.DataFrame,
    bundles: list[MetricPlotBundle],
    output_dir: Path,
) -> list[Path]:
    """Creates many notebook-style metric plots from one CSV frame in a single call."""
    if not bundles:
        raise PlotError("no metric bundles provided")
    output_paths: list[Path] = []
    for bundle in bundles:
        output_paths.append(
            plot_metric_bundle(
                frame=frame,
                include_pattern=bundle.include_pattern,
                output_dir=output_dir,
                title=bundle.title,
                y_label=bundle.y_label,
                exclude_pattern=bundle.exclude_pattern,
                show_mean_line=bundle.show_mean_line,
            )
        )
    return output_paths


def generate_stats_table(
    frame: pd.DataFrame,
    include_pattern: str,
    exclude_pattern: str | None = None,
) -> pd.DataFrame:
    """Builds per-time-step descriptive statistics across matching simulation runs."""
    columns = matching_columns(list(frame.columns), include_pattern, exclude_pattern)
    if not columns:
        raise PlotError(f"no columns found for pattern '{include_pattern}'")
    numeric = _numeric_frame(frame, columns)
    stats = pd.DataFrame(
        {
            "time_step": numeric.index.astype(int),
            "mean": numeric.mean(axis=1),
            "std": numeric.std(axis=1, ddof=0),
            "min": numeric.min(axis=1),
            "max": numeric.max(axis=1),
            "median": numeric.median(axis=1),
        }
    )
    return stats


def render_stats_table_markdown(stats_table: pd.DataFrame, precision: int = 4) -> str:
    """Renders stats table in clean markdown format for text-only LLM prompts."""
    headers = ["time_step", "mean", "std", "min", "max", "median"]
    table = stats_table[headers]
    lines = [
        "| time_step | mean | std | min | max | median |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    time_steps, means, stds, mins, maxs, medians = _stats_columns(table)
    for time_step, mean, std, min_value, max_value, median in zip(
        time_steps, means, stds, mins, maxs, medians, strict=True
    ):
        lines.append(
            f"| {time_step} | "
            f"{mean:.{precision}f} | "
            f"{std:.{precision}f} | "
            f"{min_value:.{precision}f} | "
            f"{max_value:.{precision}f} | "
            f"{median:.{precision}f} |"
        )
    return "\n".join(lines)


def render_stats_table_image(
    stats_table: pd.DataFrame,
    output_path: Path,
    title: str = "Simulation Statistics Table",
) -> Path:
    """Renders stats table as a PNG image for multimodal LLM inputs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(stats_table)
    figure_height = max(2.0, min(20.0, 1.2 + rows * 0.28))
    figure, axis = plt.subplots(figsize=(11.5, figure_height))
    axis.axis("off")
    axis.set_title(title, fontsize=12, pad=10)

    headers = ["time_step", "mean", "std", "min", "max", "median"]
    table = stats_table[headers]
    time_steps, means, stds, mins, maxs, medians = _stats_columns(table)
    cell_text = [
        [
            str(time_step),
            f"{mean:.4f}",
            f"{std:.4f}",
            f"{min_value:.4f}",
            f"{max_value:.4f}",
            f"{median:.4f}",
        ]
        for time_step, mean, std, min_value, max_value, median in zip(
            time_steps, means, stds, mins, maxs, medians, strict=True
        )
    ]
    mpl_table = axis.table(
        cellText=cell_text,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(8)
    mpl_table.scale(1.0, 1.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _stats_columns(
    table: pd.DataFrame,
) -> tuple[list[int], list[float], list[float], list[float], list[float], list[float]]:
    time_steps = [int(float(value)) for value in pd.to_numeric(table["time_step"], errors="coerce").fillna(0.0)]
    means = [float(value) for value in pd.to_numeric(table["mean"], errors="coerce").fillna(0.0)]
    stds = [float(value) for value in pd.to_numeric(table["std"], errors="coerce").fillna(0.0)]
    mins = [float(value) for value in pd.to_numeric(table["min"], errors="coerce").fillna(0.0)]
    maxs = [float(value) for value in pd.to_numeric(table["max"], errors="coerce").fillna(0.0)]
    medians = [float(value) for value in pd.to_numeric(table["median"], errors="coerce").fillna(0.0)]
    return time_steps, means, stds, mins, maxs, medians


def _numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    converted = frame[columns].apply(pd.to_numeric, errors="coerce")
    return cast(pd.DataFrame, converted.ffill().fillna(0.0))


def _draw_plot(data: pd.DataFrame, output_path: Path, title: str, y_label: str, show_mean_line: bool) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    for column in data.columns:
        axis.plot(data.index, data[column], alpha=0.25, linewidth=1.0, label=str(column))
    if show_mean_line:
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
