from __future__ import annotations

from pathlib import Path


def test_legacy_repo_is_archived() -> None:
    root = Path(".")
    assert not (root / "archive" / "legacy_repo").exists()
    for legacy in ["Code", "Paper", "Presentations", "Readings", "References", "Results"]:
        assert not (root / "archive" / legacy).exists()
        assert not (root / legacy).exists()
    assert (root / "tests" / "fixtures" / "notebook_parity" / "archive_assets" / "legacy_repo").exists()
