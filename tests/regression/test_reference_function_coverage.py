from __future__ import annotations

import json
from pathlib import Path

import distill_abm.compat as compat


def test_snapshot_notebook_functions_are_accounted_for() -> None:
    coverage_rows = json.loads(Path("docs/notebook_coverage_matrix.json").read_text(encoding="utf-8"))
    snapshot_functions = {str(row["name"]) for row in coverage_rows}
    implemented = set(compat.__all__)
    allowed_unmigrated = {"main"}
    missing = sorted(snapshot_functions - implemented - allowed_unmigrated)
    assert missing == []
