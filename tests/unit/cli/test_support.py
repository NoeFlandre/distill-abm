from __future__ import annotations

from pathlib import Path

import pytest
import typer

from distill_abm.cli_support import (
    discover_configured_abms,
    load_experiment_parameters,
    parse_summarizers,
    resolve_abm_experiment_parameters_path,
    resolve_abm_model_path,
    resolve_model_filenames,
    resolve_viz_smoke_specs,
    select_smoke_cases,
    validate_model_policy,
)


def test_parse_summarizers_rejects_unknown_values() -> None:
    with pytest.raises(typer.BadParameter):
        parse_summarizers(["bert", "unknown"], fallback=("bart",))


def test_parse_summarizers_deduplicates_and_uses_fallback() -> None:
    assert parse_summarizers(["bert", "bert", "t5"], fallback=("bart",)) == ("bert", "t5")
    assert parse_summarizers(None, fallback=("bart", "t5")) == ("bart", "t5")


def test_parse_summarizers_normalizes_whitespace() -> None:
    assert parse_summarizers([" bert ", "t5 "], fallback=("bart",)) == ("bert", "t5")


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


def test_load_experiment_parameters_validates_json_file(tmp_path: Path) -> None:
    target = tmp_path / "params.json"
    target.write_text('{"runs": 3, "flag": true}', encoding="utf-8")

    assert load_experiment_parameters(target) == {"runs": 3, "flag": True}


def test_load_experiment_parameters_rejects_non_object_json(tmp_path: Path) -> None:
    target = tmp_path / "params.json"
    target.write_text('[1, 2]', encoding="utf-8")

    with pytest.raises(typer.BadParameter, match="top level"):
        load_experiment_parameters(target)


def test_resolve_abm_model_path_rejects_ambiguous_matches(tmp_path: Path) -> None:
    root = tmp_path
    (root / "fauna.nlogo").write_text("", encoding="utf-8")
    abm_dir = root / "fauna_abm"
    abm_dir.mkdir()
    (abm_dir / "fauna_model.nlogo").write_text("", encoding="utf-8")

    with pytest.raises(typer.BadParameter, match=r"multiple \.nlogo files"):
        resolve_abm_model_path(abm="fauna", models_root=root)


def test_resolve_abm_model_path_does_not_mutate_model_layout(tmp_path: Path) -> None:
    root = tmp_path
    root_model = root / "fauna.nlogo"
    root_model.write_text("", encoding="utf-8")

    resolved = resolve_abm_model_path(abm="fauna", models_root=root)

    assert resolved == root_model
    assert not (root / "fauna_abm").exists()


def test_resolve_viz_smoke_specs_rejects_missing_netlogo_viz_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "fauna_abm" / "fauna.nlogo"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("", encoding="utf-8")

    class DummyConfig:
        netlogo_viz = None

    monkeypatch.setattr("distill_abm.cli_support.load_abm_config", lambda _path: DummyConfig())

    with pytest.raises(ValueError, match="missing netlogo_viz config"):
        resolve_viz_smoke_specs(requested_abms=["fauna"], models_root=tmp_path)


def test_validate_model_policy_allows_supported_benchmark_model_with_flag() -> None:
    validate_model_policy(
        provider="openrouter",
        model="google/gemini-3.1-pro-preview",
        allow_debug_model=True,
    )


def test_validate_model_policy_allows_debug_only_api_model_with_flag() -> None:
    validate_model_policy(
        provider="openrouter",
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        allow_debug_model=True,
    )
