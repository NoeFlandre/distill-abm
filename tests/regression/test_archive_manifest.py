from __future__ import annotations

import json
import subprocess
import unicodedata
from pathlib import Path


def _tracked_archive_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "archive"],
        check=True,
        capture_output=True,
        text=False,
    )
    raw_items = [item for item in result.stdout.split(b"\x00") if item]
    files = [Path(item.decode("utf-8", errors="surrogateescape")) for item in raw_items]
    return sorted(path for path in files if path.is_file() and path.name != ".DS_Store")


def _path_exists_with_unicode_normalization(path_str: str) -> bool:
    path = Path(path_str)
    if path.exists():
        return True
    if path.is_absolute():
        return False
    candidates: list[Path] = [Path(".")]
    for part in path.parts:
        normalized_part = unicodedata.normalize("NFC", part)
        next_candidates: list[Path] = []
        for base in candidates:
            if not base.exists() or not base.is_dir():
                continue
            for child in base.iterdir():
                if unicodedata.normalize("NFC", child.name) == normalized_part:
                    next_candidates.append(child)
        candidates = next_candidates
        if not candidates:
            return False
    return any(candidate.exists() for candidate in candidates)


def test_archive_manifest_covers_every_archive_file() -> None:
    manifest_path = Path("docs/archive_full_manifest.json")
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive_files = _tracked_archive_files()
    assert len(rows) == len(archive_files)
    assert {row["path"] for row in rows} == {path.as_posix() for path in archive_files}


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
        assert action in {"migrate", "retain_record_only"}
        assert isinstance(rationale, str) and rationale.strip()
        if classification in required:
            assert target_path is not None and str(target_path).strip()
        if action == "migrate":
            assert target_path is not None and str(target_path).strip()
            assert _path_exists_with_unicode_normalization(str(target_path))


def test_archive_manifest_migrate_targets_exclude_junk_temp_files() -> None:
    rows = json.loads(Path("docs/archive_full_manifest.json").read_text(encoding="utf-8"))
    for row in rows:
        if row["action"] != "migrate":
            continue
        filename = Path(row["path"]).name
        assert filename != ".DS_Store"
        assert not filename.startswith("~$")


def test_archive_manifest_excludes_ds_store_noise() -> None:
    rows = json.loads(Path("docs/archive_full_manifest.json").read_text(encoding="utf-8"))
    assert all(Path(row["path"]).name != ".DS_Store" for row in rows)


def test_archive_manifest_marks_runtime_required_notebooks_explicitly() -> None:
    rows = json.loads(Path("docs/archive_full_manifest.json").read_text(encoding="utf-8"))
    runtime_rows = [row for row in rows if row["classification"] == "runtime_required"]
    for row in runtime_rows:
        assert row["path"].endswith(".ipynb")
        assert row["action"] in {"migrate", "retain_record_only"}
        assert row["action"] != "discard_with_rationale"
        assert row["target_path"] is not None and str(row["target_path"]).strip()


def test_archive_manifest_keeps_csv_and_plot_artifacts() -> None:
    rows = json.loads(Path("docs/archive_full_manifest.json").read_text(encoding="utf-8"))
    keep_exts = {".csv", ".png", ".jpg", ".jpeg", ".svg"}
    for row in rows:
        if row["extension"] not in keep_exts:
            continue
        assert row["action"] in {"retain_record_only", "migrate"}
        assert row["action"] != "discard_with_rationale"


def test_archive_manifest_has_no_discard_actions() -> None:
    rows = json.loads(Path("docs/archive_full_manifest.json").read_text(encoding="utf-8"))
    assert all(row["action"] != "discard_with_rationale" for row in rows)


def test_archive_manifest_has_zero_runtime_required_rows_at_current_stage() -> None:
    rows = json.loads(Path("docs/archive_full_manifest.json").read_text(encoding="utf-8"))
    runtime_rows = [row for row in rows if row["classification"] == "runtime_required"]
    assert runtime_rows == []


def test_archive_manifest_keeps_all_notebooks() -> None:
    rows = json.loads(Path("docs/archive_full_manifest.json").read_text(encoding="utf-8"))
    notebook_rows = [row for row in rows if row["extension"] == ".ipynb"]
    for row in notebook_rows:
        assert row["action"] == "retain_record_only"
