from __future__ import annotations

from pathlib import Path

from distill_abm.configs.loader import load_abm_config


def test_integrated_benchmark_assets_exist_for_all_abms() -> None:
    expected_paths = [
        Path("data/abms/fauna/legacy/reduced_version.csv"),
        Path("data/abms/fauna/legacy/plots"),
        Path("data/abms/grazing/legacy/reduced3.csv"),
        Path("data/abms/grazing/legacy/plots"),
        Path("data/abms/grazing/RAGE_PlottingOutput.nls"),
        Path("data/abms/grazing/RAGE_VegetationLivestockModel.nls"),
        Path("data/abms/milk_consumption/legacy/reducedmilk.csv"),
        Path("data/abms/milk_consumption/legacy/plots"),
        Path("data/abms/milk_consumption/netlogodata_downsampled.csv"),
        Path("data/abms/milk_consumption/netlogovaluedata.csv"),
    ]

    missing = [str(path) for path in expected_paths if not path.exists()]
    assert not missing, "\n".join(missing)


def test_data_is_subdivided_by_abm() -> None:
    for abm in ["fauna", "grazing", "milk_consumption"]:
        folder = Path("data") / "abms" / abm
        assert folder.exists(), f"missing: {folder}"
        assert (folder / "reference_narrative.txt").exists(), f"missing narrative: {folder}"


def test_data_root_has_only_expected_abm_and_summary_roots() -> None:
    required_paths = {"abms", "summaries"}
    allowed_extra = {"paper"}
    present = {path.name for path in Path("data").iterdir() if path.is_dir() and path.name != ".DS_Store"}
    assert required_paths.issubset(present), "required data directories are missing"
    assert present.issubset(required_paths | allowed_extra), (
        f"unexpected top-level data entries: {sorted(present - (required_paths | allowed_extra))}"
    )
    assert "fauna_abm" not in present
    assert "grazing_abm" not in present
    assert "milk_consumption_abm" not in present
    assert "human_reference" not in present


def test_repo_plot_assets_match_configured_plot_counts() -> None:
    for abm in ["fauna", "grazing", "milk_consumption"]:
        config = load_abm_config(Path("configs/abms") / f"{abm}.yaml")
        assert config.netlogo_viz is not None

        expected_count = len(config.netlogo_viz.plots)
        runtime_plots = sorted((Path("data") / "abms" / abm / "plots").glob("*.png"))
        fallback_plots = sorted((Path("data") / "abms" / abm / "legacy" / "plots").glob("*.png"))

        assert len(runtime_plots) == expected_count
        assert len(fallback_plots) == expected_count
