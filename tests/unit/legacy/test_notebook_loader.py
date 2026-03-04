from __future__ import annotations

from pathlib import Path

import pytest

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


def test_loader_finds_expected_core_functions() -> None:
    names = available_function_names()
    assert len(names) >= 60
    assert "remove_urls" in names
    assert "compute_scores" in names
    assert "analyze_factorial_anova" in names


def test_loader_prefers_primary_not_archive_or_copy() -> None:
    for name in ["analyze_factorial_anova", "extract_faithfulness_score", "append_to_csv"]:
        source = str(get_notebook_source_path(name))
        assert "/Archives/" not in source
        assert "Copy" not in Path(source).name


def test_provenance_avoids_archive_copy_and_checkpoints_case_insensitive() -> None:
    for name in ["remove_urls", "extract_faithfulness_score", "append_to_csv"]:
        source = get_notebook_source_path(name)
        source_text = str(source).lower()
        assert source.exists()
        assert source.is_relative_to(Path("archive/legacy_repo/Code"))
        assert "/archives/" not in source_text
        assert ".ipynb_checkpoints" not in source_text
        assert "copy" not in source.name.lower()


def test_missing_notebook_function_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_notebook_function("_does_not_exist_")


def test_loader_exposes_all_required_notebook_functions_for_safe_decommission() -> None:
    assert missing_required_notebook_functions() == []


def test_required_function_sources_resolve_for_notebook_deletion_planning() -> None:
    sources = required_notebook_function_sources()
    assert sorted(sources.keys()) == sorted(REQUIRED_NOTEBOOK_FUNCTIONS)
    for source in sources.values():
        assert source.exists()
        assert source.is_relative_to(Path("archive/legacy_repo/Code"))


def test_required_dependencies_group_by_notebook_path() -> None:
    grouped = required_notebook_dependencies_by_path()
    flattened = [name for names in grouped.values() for name in names]
    assert sorted(flattened) == sorted(REQUIRED_NOTEBOOK_FUNCTIONS)
    for source, names in grouped.items():
        assert source.exists()
        assert names == sorted(names)


def test_should_dispatch_notebook_tracks_required_function_set() -> None:
    assert should_dispatch_notebook("clean_context_response") is False
