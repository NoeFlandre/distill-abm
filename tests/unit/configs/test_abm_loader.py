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
