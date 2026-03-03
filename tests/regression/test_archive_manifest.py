from __future__ import annotations

import json
from pathlib import Path


def test_archive_manifest_covers_every_archive_file() -> None:
    manifest_path = Path("docs/archive_full_manifest.json")
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive_files = sorted(path for path in Path("archive").rglob("*") if path.is_file())
    assert len(rows) == len(archive_files)


def test_archive_manifest_has_no_unresolved_required_mappings() -> None:
    rows = json.loads(Path("docs/archive_full_manifest.json").read_text(encoding="utf-8"))
    required = {"runtime_required", "prompt_reference", "human_ground_truth", "experiment_setting"}
    for row in rows:
        classification = row["classification"]
        action = row["action"]
        target_path = row["target_path"]
        rationale = row["rationale"]
        assert classification in {
            "runtime_required",
            "prompt_reference",
            "human_ground_truth",
            "experiment_setting",
            "historical_nonruntime",
            "legacy_visualization",
        }
        assert action in {"migrate", "retain_record_only", "archive_separately", "discard_with_rationale"}
        assert isinstance(rationale, str) and rationale.strip()
        if classification in required:
            assert target_path is not None and str(target_path).strip()
        if action == "migrate":
            assert target_path is not None and str(target_path).strip()
            assert Path(str(target_path)).exists()
