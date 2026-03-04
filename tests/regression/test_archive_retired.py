from __future__ import annotations

from pathlib import Path


def test_archive_directory_is_removed() -> None:
    assert not Path("archive").exists()


def test_archive_results_are_mirrored_under_fixtures() -> None:
    mirror_root = Path("tests/fixtures/notebook_parity/archive_assets/legacy_repo")
    mirrored_files = [path for path in mirror_root.rglob("*") if path.is_file()]
    assert mirrored_files != []
