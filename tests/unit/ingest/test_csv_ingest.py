from pathlib import Path

from distill_abm.ingest.csv_ingest import load_simulation_csv, matching_columns


def test_load_simulation_csv_with_semicolon(tmp_path: Path) -> None:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1;ignore\n1;2.5;x\n2;3.5;y\n", encoding="utf-8")

    frame = load_simulation_csv(csv_path)
    assert list(frame.columns) == ["tick", "mean-incum-1", "ignore"]
    assert float(frame.iloc[0]["mean-incum-1"]) == 2.5


def test_matching_columns_with_exclusion() -> None:
    columns = ["mean-incum-1", "mean-incum-2", "mean-incum-stdev"]
    selected = matching_columns(columns, include_pattern="mean-incum", exclude_pattern="stdev")
    assert selected == ["mean-incum-1", "mean-incum-2"]
