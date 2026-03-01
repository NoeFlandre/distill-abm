from __future__ import annotations

from pathlib import Path


def test_legacy_repo_is_archived() -> None:
    root = Path(".")
    assert (root / "archive" / "legacy_repo").exists()
    assert (root / "archive" / "legacy_repo" / "Code").exists()
    for legacy in ["Code", "Paper", "Presentations", "Readings", "References", "Results"]:
        assert not (root / legacy).exists()
