from pathlib import Path

from distill_abm.configs.loader import load_notebook_experiment_settings


def test_load_notebook_experiment_settings_config() -> None:
    settings = load_notebook_experiment_settings(Path("configs/notebook_experiment_settings.yaml"))
    assert settings.llm_defaults.temperature == 0.5
    assert settings.llm_defaults.max_tokens == 1000
    assert settings.doe_defaults.repetitions == 3
    assert "notebook_prompt_assets" in settings.qualitative_example_text_dir
