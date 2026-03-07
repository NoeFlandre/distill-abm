from __future__ import annotations

from pathlib import Path

from distill_abm.utils import detect_placeholder_signals, validate_file_for_placeholder_signals


def test_detect_placeholder_signals_is_case_insensitive() -> None:
    assert detect_placeholder_signals("TODO placeholder Dummy text") == ["placeholder", "todo", "dummy"]


def test_validate_file_for_placeholder_signals_handles_missing_path(tmp_path: Path) -> None:
    assert validate_file_for_placeholder_signals(tmp_path / "missing.txt") == []
    assert validate_file_for_placeholder_signals(None) == []


def test_validate_file_for_placeholder_signals_reads_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"
    target.write_text("This is a TBD draft with lorem ipsum.", encoding="utf-8")

    assert validate_file_for_placeholder_signals(target) == ["tbd", "lorem ipsum"]
