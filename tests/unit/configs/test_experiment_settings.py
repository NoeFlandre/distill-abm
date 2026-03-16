from pathlib import Path

from distill_abm.configs.loader import load_experiment_settings


def test_load_experiment_settings_config() -> None:
    settings = load_experiment_settings(Path("configs/experiment_settings.yaml"))

    assert settings.ground_truth.fauna.endswith("fauna_scoring_ground_truth.txt")
    assert settings.ground_truth.grazing.endswith("grazing_scoring_ground_truth.txt")
    assert settings.ground_truth.milk_consumption.endswith("milk_scoring_ground_truth.txt")

    assert Path(settings.ground_truth.fauna).exists()
    assert Path(settings.ground_truth.grazing).exists()
    assert Path(settings.ground_truth.milk_consumption).exists()
    assert "data/summaries/authors/" in settings.ground_truth.fauna
    assert "data/summaries/authors/" in settings.ground_truth.grazing
    assert "data/summaries/authors/" in settings.ground_truth.milk_consumption

    assert settings.modeler_ground_truth is not None
    assert settings.modeler_ground_truth.milk_consumption is not None
    assert settings.modeler_ground_truth.fauna is not None
    assert settings.modeler_ground_truth.grazing is None
    assert settings.modeler_ground_truth.fauna.endswith("fauna_modeler_ground_truth.txt")
    assert settings.modeler_ground_truth.milk_consumption.endswith("milk_modeler_ground_truth.txt")
    assert Path(settings.modeler_ground_truth.fauna).exists()
    assert Path(settings.modeler_ground_truth.milk_consumption).exists()
    assert "data/summaries/modelers/" in settings.modeler_ground_truth.fauna
    assert "data/summaries/modelers/" in settings.modeler_ground_truth.milk_consumption

    assert settings.gpt5_2_short_ground_truth is not None
    assert settings.gpt5_2_short_ground_truth.fauna.endswith("fauna_gpt5.2_short_ground_truth.txt")
    assert settings.gpt5_2_short_ground_truth.grazing.endswith("grazing_gpt5.2_short_ground_truth.txt")
    assert settings.gpt5_2_short_ground_truth.milk_consumption.endswith("milk_gpt5.2_short_ground_truth.txt")
    assert Path(settings.gpt5_2_short_ground_truth.fauna).exists()
    assert Path(settings.gpt5_2_short_ground_truth.grazing).exists()
    assert Path(settings.gpt5_2_short_ground_truth.milk_consumption).exists()
    assert "data/summaries/gpt5.2/" in settings.gpt5_2_short_ground_truth.fauna
    assert "data/summaries/gpt5.2/" in settings.gpt5_2_short_ground_truth.grazing
    assert "data/summaries/gpt5.2/" in settings.gpt5_2_short_ground_truth.milk_consumption

    assert settings.gpt5_2_long_ground_truth is not None
    assert settings.gpt5_2_long_ground_truth.fauna.endswith("fauna_gpt5.2_long_ground_truth.txt")
    assert settings.gpt5_2_long_ground_truth.grazing.endswith("grazing_gpt5.2_long_ground_truth.txt")
    assert settings.gpt5_2_long_ground_truth.milk_consumption.endswith("milk_gpt5.2_long_ground_truth.txt")
    assert Path(settings.gpt5_2_long_ground_truth.fauna).exists()
    assert Path(settings.gpt5_2_long_ground_truth.grazing).exists()
    assert Path(settings.gpt5_2_long_ground_truth.milk_consumption).exists()
    assert "data/summaries/gpt5.2/" in settings.gpt5_2_long_ground_truth.fauna
    assert "data/summaries/gpt5.2/" in settings.gpt5_2_long_ground_truth.grazing
    assert "data/summaries/gpt5.2/" in settings.gpt5_2_long_ground_truth.milk_consumption

    assert settings.human_reference_dir == "data/summaries"
    assert Path(settings.human_reference_dir).exists()
