from __future__ import annotations

import json
from pathlib import Path

from distill_abm.legacy.notebook_loader import required_notebook_dependencies_by_path


def test_runtime_notebook_dependency_report_matches_loader_and_manifest() -> None:
    report_path = Path("docs/runtime_notebook_dependencies.json")
    rows = json.loads(report_path.read_text(encoding="utf-8"))

    report_map = {row["notebook_path"]: sorted(row["required_functions"]) for row in rows}
    loader_map = {
        path.as_posix(): sorted(functions) for path, functions in required_notebook_dependencies_by_path().items()
    }
    assert report_map == loader_map

    manifest_rows = json.loads(Path("docs/archive_full_manifest.json").read_text(encoding="utf-8"))
    runtime_paths = sorted(row["path"] for row in manifest_rows if row["classification"] == "runtime_required")
    assert sorted(report_map.keys()) == runtime_paths
