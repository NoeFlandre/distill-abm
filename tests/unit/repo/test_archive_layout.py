from __future__ import annotations

from pathlib import Path


def test_legacy_archive_directories_are_absent() -> None:
    root = Path(".")
    assert not (root / "archive" / "reference_repo").exists()
    for retired_root in ["Code", "Paper", "Presentations", "Readings", "References", "Results"]:
        assert not (root / "archive" / retired_root).exists()


def test_active_runtime_has_no_legacy_compat_modules() -> None:
    root = Path("src/distill_abm")
    assert not (root / "compat").exists()
    assert not (root / "reference").exists()
