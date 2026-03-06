from pathlib import Path

import pandas as pd
import pytest

from distill_abm.viz import plots
from distill_abm.viz.plots import MetricPlotBundle, plot_metric_bundle, plot_metric_bundles


def test_plot_metric_bundle_writes_image(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "tick": [0, 1, 2],
            "mean-incum-run-1": [1.0, 2.0, 3.0],
            "mean-incum-run-2": [2.0, 3.0, 4.0],
        }
    )

    output = plot_metric_bundle(
        frame=frame,
        include_pattern="mean-incum",
        output_dir=tmp_path,
        title="Test",
        y_label="Value",
    )
    assert output.exists()
    assert output.suffix == ".png"


def test_plot_metric_bundle_respects_show_mean_line_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {
            "tick": [0, 1],
            "mean-incum-run-1": [1.0, 2.0],
            "mean-incum-run-2": [2.0, 3.0],
        }
    )
    seen: list[bool] = []

    def fake_draw(
        data: pd.DataFrame,
        output_path: Path,
        title: str,
        y_label: str,
        x_label: str,
        show_mean_line: bool,
    ) -> None:
        _ = data, title, y_label, x_label
        seen.append(show_mean_line)
        output_path.write_bytes(b"png")

    monkeypatch.setattr(plots, "_draw_plot", fake_draw)
    output = plot_metric_bundle(
        frame=frame,
        include_pattern="mean-incum",
        output_dir=tmp_path,
        title="Test",
        y_label="Value",
        show_mean_line=False,
    )
    assert output.exists()
    assert seen == [False]


def test_plot_metric_bundles_writes_multiple_images(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "tick": [0, 1],
            "mean-incum-run-1": [1.0, 2.0],
            "mean-alt-run-1": [3.0, 4.0],
        }
    )
    bundles = [
        MetricPlotBundle(include_pattern="mean-incum", title="Incum", y_label="Value"),
        MetricPlotBundle(include_pattern="mean-alt", title="Alt", y_label="Value"),
    ]
    outputs = plot_metric_bundles(frame=frame, bundles=bundles, output_dir=tmp_path)
    assert len(outputs) == 2
    assert all(path.exists() for path in outputs)
    assert outputs[0].name == "mean-incum.png"
    assert outputs[1].name == "mean-alt.png"
