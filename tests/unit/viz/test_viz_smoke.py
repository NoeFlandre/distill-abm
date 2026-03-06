from __future__ import annotations

from pathlib import Path

from distill_abm.viz.viz_smoke import default_viz_smoke_stages, run_viz_smoke_suite


def test_default_viz_smoke_stages_are_granular() -> None:
    assert [stage.stage_id for stage in default_viz_smoke_stages()] == [
        "csv-load",
        "plot",
        "stats-csv",
        "stats-markdown",
        "stats-image",
    ]


def test_run_viz_smoke_suite_writes_stage_level_report(tmp_path: Path) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text(
        "tick;mean-incum-run-1;mean-incum-run-2\n0;1;2\n1;2;3\n",
        encoding="utf-8",
    )

    result = run_viz_smoke_suite(
        abm_inputs={"milk_consumption": (csv_path, "mean-incum")},
        output_root=tmp_path / "viz-smoke",
    )

    assert result.success is True
    assert result.report_json_path.exists()
    assert result.report_markdown_path.exists()
    assert len(result.abms) == 1
    assert [stage.stage.stage_id for stage in result.abms[0].stage_results] == [
        "csv-load",
        "plot",
        "stats-csv",
        "stats-markdown",
        "stats-image",
    ]
