from pathlib import Path

import pandas as pd

from distill_abm.viz.plots import plot_metric_bundle


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
