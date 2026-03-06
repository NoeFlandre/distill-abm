from __future__ import annotations

from pathlib import Path

import pytest
import typer

from distill_abm.cli_support import (
    discover_configured_abms,
    parse_summarizers,
    resolve_abm_experiment_parameters_path,
    resolve_model_filenames,
    select_smoke_cases,
)


def test_parse_summarizers_rejects_unknown_values() -> None:
    with pytest.raises(typer.BadParameter):
        parse_summarizers(["bert", "unknown"], fallback=("bart",))


def test_parse_summarizers_deduplicates_and_uses_fallback() -> None:
    assert parse_summarizers(["bert", "bert", "t5"], fallback=("bart",)) == ("bert", "t5")
    assert parse_summarizers(None, fallback=("bart", "t5")) == ("bart", "t5")


def test_select_smoke_cases_validates_unknown_case_ids() -> None:
    with pytest.raises(typer.BadParameter):
        select_smoke_cases(case_ids=["missing-case"], max_cases=None, profile="matrix")


def test_select_smoke_cases_respects_profile_and_limit() -> None:
    selected = select_smoke_cases(case_ids=None, max_cases=2, profile="three-branches")

    assert selected is not None
    assert len(selected) == 2
    assert selected[0].case_id == "branch-role-full-text"


def test_resolve_model_filenames_handles_milk_special_case() -> None:
    assert resolve_model_filenames("fauna") == ("fauna.nlogo", "fauna_model.nlogo")
    assert resolve_model_filenames("milk_consumption") == (
        "milk_consumption.nlogo",
        "milk_consumption_model.nlogo",
        "model.nlogo",
    )


def test_resolve_abm_experiment_parameters_path_prefers_existing_file(tmp_path: Path) -> None:
    model_dir = tmp_path / "fauna_abm"
    model_dir.mkdir()
    target = model_dir / "experiment_parameters.json"
    target.write_text("{}", encoding="utf-8")

    resolved = resolve_abm_experiment_parameters_path(model_dir=model_dir, abm="fauna", explicit=None)

    assert resolved == target


def test_discover_configured_abms_reads_repo_configs() -> None:
    assert discover_configured_abms() == ("fauna", "grazing", "milk_consumption")
