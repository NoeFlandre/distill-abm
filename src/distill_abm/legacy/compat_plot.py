"""Plot helper compatibility wrappers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from distill_abm.viz.plots import plot_metric_bundle


def plot_columns(
    data: pd.DataFrame,
    column_pattern: str,
    y_label: str,
    title: str,
    exclude_pattern: str | None = None,
) -> Path:
    return plot_metric_bundle(
        frame=data,
        include_pattern=column_pattern,
        exclude_pattern=exclude_pattern,
        output_dir=Path("results/plots"),
        title=title,
        y_label=y_label,
        show_mean_line=False,
    )
