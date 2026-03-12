from __future__ import annotations

from pathlib import Path

from distill_abm.configs.loader import load_abm_config


def test_load_all_abm_configs() -> None:
    for name in ["fauna", "grazing", "milk_consumption"]:
        cfg = load_abm_config(Path("configs/abms") / f"{name}.yaml")
        assert cfg.name
        assert cfg.metric_pattern
        assert cfg.metric_description
        assert len(cfg.plot_descriptions) > 0


def test_load_viz_config_fallbacks_and_explicit_parameters() -> None:
    fauna_cfg = load_abm_config(Path("configs/abms/fauna.yaml"))
    milk_cfg = load_abm_config(Path("configs/abms/milk_consumption.yaml"))
    grazing_cfg = load_abm_config(Path("configs/abms/grazing.yaml"))

    assert fauna_cfg.netlogo_viz is not None
    assert fauna_cfg.netlogo_viz.fallback_mode == "always"
    assert fauna_cfg.netlogo_viz.fallback_csv == "data/abms/fauna/legacy/reduced_version.csv"
    assert fauna_cfg.netlogo_viz.fallback_plot_dir == "data/abms/fauna/legacy/plots"

    assert milk_cfg.netlogo_viz is not None
    assert milk_cfg.netlogo_viz.fallback_mode == "always"
    assert milk_cfg.netlogo_viz.experiment_parameters["network-type"] == '"watts-strogatz"'
    assert milk_cfg.netlogo_viz.fallback_csv == "data/abms/milk_consumption/legacy/reducedmilk.csv"
    assert milk_cfg.netlogo_viz.fallback_plot_dir == "data/abms/milk_consumption/legacy/plots"

    assert grazing_cfg.netlogo_viz is not None
    assert grazing_cfg.netlogo_viz.fallback_mode == "always"
    assert grazing_cfg.netlogo_viz.experiment_parameters["behavioral-type"] == "E-RO"
    assert grazing_cfg.netlogo_viz.fallback_csv == "data/abms/grazing/legacy/reduced3.csv"
    assert grazing_cfg.netlogo_viz.fallback_plot_dir == "data/abms/grazing/legacy/plots"
