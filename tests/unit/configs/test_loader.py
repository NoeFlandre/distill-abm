from collections.abc import Callable
from pathlib import Path

import pytest

from distill_abm.configs.loader import (
    ConfigError,
    load_abm_config,
    load_evaluation_config,
    load_logging_config,
    load_models_config,
    load_notebook_experiment_settings,
    load_prompts_config,
    load_runtime_defaults_config,
)


def test_load_models_config(tmp_path: Path) -> None:
    content = """
models:
  openai:
    provider: openai
    model: gpt-4o
"""
    path = tmp_path / "models.yaml"
    path.write_text(content, encoding="utf-8")

    config = load_models_config(path)
    assert config.models["openai"].provider == "openai"
    assert config.models["openai"].model == "gpt-4o"


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
    ],
)
def test_missing_config_file_raises(tmp_path: Path, loader: Callable[[Path], object]) -> None:
    path = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError):
        loader(path)


@pytest.mark.parametrize(
    "loader, body",
    [
        (load_models_config, "models:\n  openai: 1"),
        (load_prompts_config, "context_prompt: ['bad']"),
        (load_abm_config, "name: 1"),
        (load_evaluation_config, "use_reference_metrics: not-bool"),
        (load_notebook_experiment_settings, "llm_defaults: not-a-dict"),
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


def test_load_notebook_experiment_settings_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "experiment_settings.yaml"
    path.write_text(
        """
llm_defaults:
  openai_model: gpt-4o
  anthropic_model: claude-3-5-sonnet-20241022
  max_tokens: 1000
  temperature: 0.5
doe_defaults:
  repetitions: 3
  max_interaction_order: 2
fauna_from_netlogo_to_csv:
  netlogo_home: /tmp/netlogo
  model_path: /tmp/model.nlogo
  output_csv_path: /tmp/output.csv
  reporters:
    - m1
  runtime_experiment_parameters: {"a": 1}
  saved_experiment_parameters: {"a": 1}
  num_runs: 1
  max_ticks: 1
  interval: 1
grazint_netlogo_to_csv:
  netlogo_home: /tmp/netlogo
  model_path: /tmp/model.nlogo
  output_csv_path: /tmp/output.csv
  reporters:
    - m1
  runtime_experiment_parameters: {"a": 1}
  saved_experiment_parameters: {"a": 1}
  num_runs: 1
  max_ticks: 1
  interval: 1
milk_netlogo_to_csv:
  netlogo_home: /tmp/netlogo
  model_path: /tmp/model.nlogo
  output_csv_path: /tmp/output.csv
  reporters:
    - m1
  runtime_experiment_parameters: {"a": 1}
  saved_experiment_parameters: {"a": 1}
  num_runs: 1
  max_ticks: 1
  interval: 1
qualitative_example_text_dir: /tmp/examples
human_reference_dir: /tmp/human-reference
summary_generation:
  fauna:
    num_plots: 14
  grazing:
    num_plots: 10
  milk:
    num_plots: 12
scoring:
  fauna_ground_truth_path: /tmp/fauna.txt
  grazing_ground_truth_path: /tmp/grazing.txt
  milk_ground_truth_path: /tmp/milk.txt
""",
        encoding="utf-8",
    )

    settings = load_notebook_experiment_settings(path)
    assert settings.fauna_from_netlogo_to_csv.netlogo_home == "/tmp/netlogo"
    assert settings.grazint_netlogo_to_csv.output_csv_path == "/tmp/output.csv"
    assert settings.milk_netlogo_to_csv.saved_experiment_parameters["a"] == 1
    assert settings.summary_generation.fauna.num_plots == 14
    assert settings.scoring.milk_ground_truth_path == "/tmp/milk.txt"


def test_repository_prompts_follow_notebook_wording() -> None:
    config = load_prompts_config(Path("configs/prompts.yaml"))
    assert "Do not write any summary or conclusion." in config.context_prompt
    assert "Do not refer to the plot or any visual in your description." in config.trend_prompt
    assert "rate a report based on its coverage with respect to an input context and input plots" in (
        config.coverage_eval_prompt
    )
    assert "rate a report based on its faithfulness with respect to an input context and input plots" in (
        config.faithfulness_eval_prompt
    )
    assert "Image used for the example" in config.coverage_eval_prompt
    assert "Image used for the example" in config.faithfulness_eval_prompt
    assert "The main car brand used by agents is Toyota." in config.coverage_eval_prompt
    assert "The main car brand used by agents is Toyota." in config.faithfulness_eval_prompt
