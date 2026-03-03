"""Typed configuration models used across the pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProviderName = Literal["openai", "anthropic", "ollama", "janus", "echo"]


class ModelEntry(BaseModel):
    """Defines provider-specific runtime options loaded from YAML."""

    provider: ProviderName
    model: str
    base_url: str | None = None
    api_key_env: str | None = None


class ModelsConfig(BaseModel):
    """Maps model aliases to concrete provider and model definitions."""

    models: dict[str, ModelEntry]


class PromptsConfig(BaseModel):
    """Stores editable prompt templates to avoid notebook hardcoding."""

    context_prompt: str
    trend_prompt: str
    coverage_eval_prompt: str = (
        "Your task is to rate a report based on its coverage with respect to an input context and input plots.\n"
        "Coverage is on a scale from 1 (worst) to 5 (perfect).\n"
        "Your answer must state the number you give for coverage and your reasoning.\n"
        "Input context:\n{source}\n\n"
        "Report to rate:\n{summary}\n\n"
        "Return 'Coverage score: <1-5>' and brief reasoning."
    )
    faithfulness_eval_prompt: str = (
        "Your task is to rate a report based on its faithfulness with respect to an input context and input plots.\n"
        "Faithfulness is on a scale from 1 (worst) to 5 (perfect).\n"
        "Your answer must state the number you give for faithfulness and your reasoning.\n"
        "Input context:\n{source}\n\n"
        "Report to rate:\n{summary}\n\n"
        "Return 'Faithfulness score: <1-5>' and brief reasoning."
    )
    style_features: dict[str, str] = Field(default_factory=dict)


class ABMConfig(BaseModel):
    """Defines case-study-specific defaults extracted from notebook workflows."""

    name: str
    metric_pattern: str
    metric_description: str
    plot_descriptions: list[str]
    default_input_csv: str | None = None
    default_parameters_txt: str | None = None
    default_documentation_txt: str | None = None


class EvaluationConfig(BaseModel):
    """Defines default evaluation behavior for summary scoring."""

    use_legacy_metrics: bool = True
    use_token_f1: bool = True


class LoggingConfig(BaseModel):
    """Provides centralized runtime logging settings."""

    level: str = "INFO"
    format: str = "%(asctime)s %(levelname)s %(name)s %(message)s"


class NotebookLLMDefaults(BaseModel):
    """Captures notebook-era model invocation defaults."""

    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 1000
    temperature: float = 0.5


class NotebookDoEDefaults(BaseModel):
    """Captures notebook DoE helper defaults."""

    repetitions: int = 3
    max_interaction_order: int = 2


class NotebookExperimentSettings(BaseModel):
    """Stores canonical notebook experiment references and defaults."""

    llm_defaults: NotebookLLMDefaults
    doe_defaults: NotebookDoEDefaults
    qualitative_example_text_dir: str
    human_reference_dir: str
