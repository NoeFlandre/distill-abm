from pathlib import Path

from distill_abm.configs.loader import load_notebook_experiment_settings


def test_load_notebook_experiment_settings_config() -> None:
    settings = load_notebook_experiment_settings(Path("configs/notebook_experiment_settings.yaml"))
    assert settings.llm_defaults.temperature == 0.5
    assert settings.llm_defaults.max_tokens == 1000
    assert settings.doe_defaults.repetitions == 3
    assert len(settings.grazint_netlogo_to_csv.saved_experiment_parameters) == 31
    assert settings.fauna_from_netlogo_to_csv.num_runs == 40
    assert settings.fauna_from_netlogo_to_csv.interval == 50
    assert settings.milk_netlogo_to_csv.num_runs == 40
    assert settings.milk_netlogo_to_csv.interval == 50
    assert settings.grazint_netlogo_to_csv.num_runs == 40
    assert settings.grazint_netlogo_to_csv.interval == 50
    assert len(settings.fauna_from_netlogo_to_csv.reporters) == 12
    assert len(settings.grazint_netlogo_to_csv.reporters) == 12
    assert len(settings.milk_netlogo_to_csv.reporters) == 12
    assert settings.fauna_from_netlogo_to_csv.runtime_experiment_parameters["number-of-agents"] == 1000
    assert settings.grazint_netlogo_to_csv.runtime_experiment_parameters["number-E-LBD"] == 20
    assert settings.milk_netlogo_to_csv.runtime_experiment_parameters["social-conformity"] == 0.28
    assert settings.fauna_from_netlogo_to_csv.saved_experiment_parameters["init-foragers"] == 20
    assert settings.grazint_netlogo_to_csv.saved_experiment_parameters["number-households"] == 60
    assert settings.milk_netlogo_to_csv.saved_experiment_parameters["network-parameter"] == 8
    assert settings.summary_generation.fauna.num_plots == 14
    assert settings.summary_generation.grazing.num_plots == 10
    assert settings.summary_generation.milk.num_plots == 12
    assert settings.scoring.fauna_ground_truth_path.endswith("fauna_scoring_ground_truth.txt")
    assert settings.scoring.grazing_ground_truth_path.endswith("grazing_scoring_ground_truth.txt")
    assert settings.scoring.milk_ground_truth_path.endswith("milk_scoring_ground_truth.txt")
    assert Path(settings.scoring.fauna_ground_truth_path).exists()
    assert Path(settings.scoring.grazing_ground_truth_path).exists()
    assert Path(settings.scoring.milk_ground_truth_path).exists()
    assert "notebook_prompt_assets" in settings.qualitative_example_text_dir
