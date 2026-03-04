from __future__ import annotations

from pathlib import Path


def test_reference_repo_is_archived() -> None:
    root = Path(".")
    assert not (root / "archive" / "reference_repo").exists()
    for archived_root in ["Code", "Paper", "Presentations", "Readings", "References", "Results"]:
        assert not (root / "archive" / archived_root).exists()
        assert not (root / archived_root).exists()
    assert (root / "tests" / "fixtures" / "notebook_parity" / "archive_assets" / "reference_repo").exists()
