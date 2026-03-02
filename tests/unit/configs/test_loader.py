from pathlib import Path

import pytest

from distill_abm.configs.loader import ConfigError, load_models_config, load_prompts_config


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
