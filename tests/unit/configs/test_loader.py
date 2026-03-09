from collections.abc import Callable
from pathlib import Path

import pytest

from distill_abm.configs.loader import (
    ConfigError,
    load_abm_config,
    load_evaluation_config,
    load_experiment_settings,
    load_logging_config,
    load_models_config,
    load_prompts_config,
    load_runtime_defaults_config,
)


def test_load_models_config(tmp_path: Path) -> None:
    content = """
models:
  kimi:
    provider: openrouter
    model: moonshotai/kimi-k2.5
"""
    path = tmp_path / "models.yaml"
    path.write_text(content, encoding="utf-8")

    config = load_models_config(path)
    assert config.models["kimi"].provider == "openrouter"
    assert config.models["kimi"].model == "moonshotai/kimi-k2.5"


def test_load_prompts_config(tmp_path: Path) -> None:
    content = """
context_prompt: "context {parameters} {documentation}"
trend_prompt: "trend {description}"
"""
    path = tmp_path / "prompts.yaml"
    path.write_text(content, encoding="utf-8")

    config = load_prompts_config(path)
    assert "{parameters}" in config.context_prompt


def test_load_prompts_config_includes_qualitative_templates(tmp_path: Path) -> None:
    content = """
context_prompt: "context {parameters} {documentation}"
trend_prompt: "trend {description}"
coverage_eval_prompt: "Evaluate coverage from source: {source} summary: {summary}"
faithfulness_eval_prompt: "Evaluate faithfulness from source: {source} summary: {summary}"
"""
    path = tmp_path / "prompts.yaml"
    path.write_text(content, encoding="utf-8")

    config = load_prompts_config(path)
    assert "{source}" in config.coverage_eval_prompt
    assert "{summary}" in config.faithfulness_eval_prompt


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("models:\n  a: [", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_models_config(path)


def test_non_mapping_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "scalar.yaml"
    path.write_text("7", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_models_config(path)


@pytest.mark.parametrize(
    "loader",
    [
        load_models_config,
        load_prompts_config,
        load_abm_config,
        load_evaluation_config,
        load_logging_config,
        load_runtime_defaults_config,
        load_experiment_settings,
    ],
)
def test_missing_config_file_raises(tmp_path: Path, loader: Callable[[Path], object]) -> None:
    path = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError):
        loader(path)


@pytest.mark.parametrize(
    "loader, body",
    [
        (load_models_config, "models:\n  kimi: 1"),
        (load_prompts_config, "context_prompt: ['bad']"),
        (load_abm_config, "name: 1"),
        (load_evaluation_config, "use_reference_metrics: not-bool"),
        (load_experiment_settings, "ground_truth: not-a-dict"),
        (load_runtime_defaults_config, "llm_request: 7"),
    ],
)
def test_invalid_config_payload_raises(
    tmp_path: Path,
    loader: Callable[[Path], object],
    body: str,
) -> None:
    path = tmp_path / "payload.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError):
        loader(path)


def test_load_experiment_settings_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "experiment_settings.yaml"
    path.write_text(
        """
ground_truth:
  fauna: /tmp/fauna.txt
  grazing: /tmp/grazing.txt
  milk_consumption: /tmp/milk.txt
modeler_ground_truth:
  milk_consumption: /tmp/milk-modeler.txt
qualitative_example_text_dir: /tmp/examples
human_reference_dir: /tmp/human-reference
""",
        encoding="utf-8",
    )

    settings = load_experiment_settings(path)
    assert settings.ground_truth.fauna == "/tmp/fauna.txt"
    assert settings.ground_truth.grazing == "/tmp/grazing.txt"
    assert settings.ground_truth.milk_consumption == "/tmp/milk.txt"
    assert settings.modeler_ground_truth is not None
    assert settings.modeler_ground_truth.milk_consumption == "/tmp/milk-modeler.txt"
    assert settings.qualitative_example_text_dir == "/tmp/examples"
    assert settings.human_reference_dir == "/tmp/human-reference"


def test_repository_prompts_include_style_factors() -> None:
    config = load_prompts_config(Path("configs/prompts.yaml"))
    assert "Do not write any summary or conclusion." in config.context_prompt
    assert "Do not refer to the plot or any visual in your description." in config.trend_prompt
    assert "rate a report based on its coverage" in config.coverage_eval_prompt
    assert "rate a report based on its faithfulness" in config.faithfulness_eval_prompt
    assert set(("role", "example", "insights")) <= set(config.style_features)
