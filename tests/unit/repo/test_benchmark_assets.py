from __future__ import annotations

from pathlib import Path

from distill_abm.configs.loader import load_abm_config


def test_integrated_benchmark_assets_exist_for_all_abms() -> None:
    expected_paths = [
        Path("data/fauna_abm/legacy/reduced_version.csv"),
        Path("data/fauna_abm/legacy/plots"),
        Path("data/grazing_abm/legacy/reduced3.csv"),
        Path("data/grazing_abm/legacy/plots"),
        Path("data/grazing_abm/RAGE_PlottingOutput.nls"),
        Path("data/grazing_abm/RAGE_VegetationLivestockModel.nls"),
        Path("data/milk_consumption_abm/legacy/reducedmilk.csv"),
        Path("data/milk_consumption_abm/legacy/plots"),
        Path("data/milk_consumption_abm/netlogodata_downsampled.csv"),
        Path("data/milk_consumption_abm/netlogovaluedata.csv"),
    ]

    missing = [str(path) for path in expected_paths if not path.exists()]
    assert not missing, "\n".join(missing)


def test_repo_plot_assets_match_configured_plot_counts() -> None:
    for abm in ["fauna", "grazing", "milk_consumption"]:
        config = load_abm_config(Path("configs/abms") / f"{abm}.yaml")
        assert config.netlogo_viz is not None

        expected_count = len(config.netlogo_viz.plots)
        runtime_plots = sorted((Path("data") / f"{abm}_abm" / "plots").glob("*.png"))
        fallback_plots = sorted((Path("data") / f"{abm}_abm" / "legacy" / "plots").glob("*.png"))

        assert len(runtime_plots) == expected_count
        assert len(fallback_plots) == expected_count
