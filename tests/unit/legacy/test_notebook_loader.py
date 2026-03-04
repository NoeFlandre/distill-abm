from __future__ import annotations

import json
from pathlib import Path

import pytest

import distill_abm.legacy.notebook_loader as notebook_loader
from distill_abm.legacy.notebook_loader import (
    REQUIRED_NOTEBOOK_FUNCTIONS,
    _is_better_source,
    _path_priority,
    available_function_names,
    get_notebook_function,
    get_notebook_source_path,
    missing_required_notebook_functions,
    required_notebook_dependencies_by_path,
    required_notebook_function_sources,
    should_dispatch_notebook,
)


def _write_notebook(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "synthetic-cell",
                "metadata": {},
                "outputs": [],
                "source": source,
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _set_note_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(notebook_loader, "NOTEBOOK_ROOT", root)
    notebook_loader._build_registry.cache_clear()


def test_priority_flags_are_case_insensitive() -> None:
    lower_archive = Path("archive/legacy_repo/Code/evaluation/doe/archives/sample.ipynb")
    lower_copy = Path("archive/legacy_repo/Code/models/fauna/example-copy1.ipynb")
    checkpoint = Path("archive/legacy_repo/Code/models/fauna/.ipynb_checkpoints/sample.ipynb")

    assert _path_priority(lower_archive)[0] == 0
    assert _path_priority(lower_copy)[2] == 0
    assert _path_priority(checkpoint)[1] == 0


def test_better_source_prefers_primary_over_archive_copy_checkpoint() -> None:
    primary = Path("archive/legacy_repo/Code/Evaluation/DOE/DoE.ipynb")
    archive = Path("archive/legacy_repo/Code/Evaluation/DOE/archives/DoE.ipynb")
    copied = Path("archive/legacy_repo/Code/Models/Fauna/example-copy1.ipynb")
    checkpoint = Path("archive/legacy_repo/Code/Evaluation/DOE/.ipynb_checkpoints/DoE-checkpoint.ipynb")

    assert _is_better_source(primary, archive)
    assert _is_better_source(primary, copied)
    assert _is_better_source(primary, checkpoint)
    assert not _is_better_source(archive, primary)


def test_loader_uses_priority_order_with_synthetic_notebooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_note_root(monkeypatch, tmp_path)
    _write_notebook(tmp_path / "Model/archives/f.ipynb", "def pick():\n    return 'archive'\n")
    _write_notebook(tmp_path / "Model/.ipynb_checkpoints/f.ipynb", "def pick():\n    return 'checkpoint'\n")
    _write_notebook(tmp_path / "Model/f-copy.ipynb", "def pick():\n    return 'copy'\n")
    _write_notebook(tmp_path / "Model/f.ipynb", "def pick():\n    return 'primary'\n")

    assert "pick" in available_function_names()
    assert get_notebook_function("pick")() == "primary"
    assert get_notebook_source_path("pick") == tmp_path / "Model/f.ipynb"


def test_missing_notebook_function_raises_key_error_with_empty_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_note_root(monkeypatch, tmp_path)
    with pytest.raises(KeyError):
        get_notebook_function("_does_not_exist_")


def test_loader_exposes_all_required_notebook_functions_for_safe_decommission() -> None:
    assert missing_required_notebook_functions() == []


def test_required_function_sources_resolve_for_notebook_deletion_planning() -> None:
    sources = required_notebook_function_sources()
    assert sorted(sources.keys()) == sorted(REQUIRED_NOTEBOOK_FUNCTIONS)
    for source in sources.values():
        assert source.exists()


def test_required_dependencies_group_by_notebook_path() -> None:
    grouped = required_notebook_dependencies_by_path()
    flattened = [name for names in grouped.values() for name in names]
    assert sorted(flattened) == sorted(REQUIRED_NOTEBOOK_FUNCTIONS)
    for source, names in grouped.items():
        assert source.exists()
        assert names == sorted(names)


def test_should_dispatch_notebook_tracks_required_function_set() -> None:
    assert should_dispatch_notebook("clean_context_response") is False
