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

    assert settings.modeler_ground_truth is not None
    assert settings.modeler_ground_truth.milk_consumption is not None
    assert settings.modeler_ground_truth.fauna is None
    assert settings.modeler_ground_truth.grazing is None
    assert settings.modeler_ground_truth.milk_consumption.endswith("milk_modeler_ground_truth.txt")
    assert Path(settings.modeler_ground_truth.milk_consumption).exists()

    assert settings.qualitative_example_text_dir is not None
    assert "prompt_assets" in settings.qualitative_example_text_dir
