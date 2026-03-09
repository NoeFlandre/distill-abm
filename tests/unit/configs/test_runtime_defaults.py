from collections.abc import Iterator
from pathlib import Path

import pytest

from distill_abm.configs.loader import load_runtime_defaults_config
from distill_abm.configs.runtime_defaults import clear_runtime_defaults_cache, get_runtime_defaults
from distill_abm.llm.adapters.base import LLMMessage, LLMRequest


@pytest.fixture(autouse=True)
def _reset_runtime_defaults_cache() -> Iterator[None]:
    clear_runtime_defaults_cache()
    yield
    clear_runtime_defaults_cache()


def test_load_runtime_defaults_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "runtime_defaults.yaml"
    path.write_text(
        """
llm_request:
  temperature: 0.25
  max_tokens: 777
run:
  provider: openrouter
  model: moonshotai/kimi-k2.5
  output_dir: results/pipeline
  metric_pattern: mean-incum
  metric_description: weekly milk
  evidence_mode: plot+table
  text_source_mode: summary_only
  summarizers:
    - bart
    - t5
qualitative:
  provider: openrouter
  model: google/gemini-3.1-pro-preview
smoke:
  output_dir: results/smoke_debug
  model: google/gemini-3.1-pro-preview
  metric_pattern: mean-incum
  metric_description: weekly milk
  evidence_mode: table
  text_source_mode: full_text_only
doe:
  output_csv: results/doe/anova_factorial_contributions.csv
  max_interaction_order: 2
""",
        encoding="utf-8",
    )
    config = load_runtime_defaults_config(path)
    assert config.llm_request.temperature == 0.25
    assert config.llm_request.max_tokens == 777
    assert config.run.model == "moonshotai/kimi-k2.5"
    assert config.run.evidence_mode == "plot+table"
    assert config.run.text_source_mode == "summary_only"
    assert config.run.summarizers == ("bart", "t5")


def test_get_runtime_defaults_uses_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "runtime_defaults.yaml"
    path.write_text(
        """
llm_request:
  temperature: 0.12
  max_tokens: 123
run:
  provider: openrouter
  model: moonshotai/kimi-k2.5
  output_dir: results/pipeline
  metric_pattern: mean
  metric_description: simulation trend
  evidence_mode: plot
  text_source_mode: full_text_only
qualitative:
  provider: openrouter
  model: google/gemini-3.1-pro-preview
smoke:
  output_dir: results/smoke_debug
  model: google/gemini-3.1-pro-preview
  metric_pattern: mean
  metric_description: simulation trend
  evidence_mode: plot+table
  text_source_mode: summary_only
doe:
  output_csv: results/doe/anova_factorial_contributions.csv
  max_interaction_order: 2
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("DISTILL_ABM_RUNTIME_DEFAULTS_PATH", str(path))
    clear_runtime_defaults_cache()
    defaults = get_runtime_defaults()
    assert defaults.llm_request.temperature == 0.12
    assert defaults.llm_request.max_tokens == 123


def test_llm_request_defaults_come_from_runtime_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "runtime_defaults.yaml"
    path.write_text(
        """
llm_request:
  temperature: 0.33
  max_tokens: 345
run:
  provider: openrouter
  model: moonshotai/kimi-k2.5
  output_dir: results/pipeline
  metric_pattern: mean
  metric_description: simulation trend
  evidence_mode: plot+table
  text_source_mode: summary_only
qualitative:
  provider: openrouter
  model: google/gemini-3.1-pro-preview
smoke:
  output_dir: results/smoke_debug
  model: google/gemini-3.1-pro-preview
  metric_pattern: mean
  metric_description: simulation trend
  evidence_mode: plot+table
  text_source_mode: summary_only
doe:
  output_csv: results/doe/anova_factorial_contributions.csv
  max_interaction_order: 2
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("DISTILL_ABM_RUNTIME_DEFAULTS_PATH", str(path))
    clear_runtime_defaults_cache()
    request = LLMRequest(model="qwen/qwen3.5-27b", messages=[LLMMessage(role="user", content="hello")])
    assert request.temperature == 0.33
    assert request.max_tokens == 345
