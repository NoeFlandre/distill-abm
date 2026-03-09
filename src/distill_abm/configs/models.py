"""Typed configuration models used across the paper-aligned pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

ProviderName = Literal["openai", "openrouter", "anthropic", "ollama", "mistral", "echo"]
SummarizerId = Literal["bart", "bert", "t5", "longformer_ext"]
RuntimeEvidenceMode = Literal["plot", "table", "plot+table"]
RuntimeTextSourceMode = Literal["summary_only", "full_text_only"]


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
    """Stores editable prompt templates and optional style-factor instructions."""

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


class ABMNetLogoVizPlotConfig(BaseModel):
    """Defines one ordered plot emitted by the NetLogo visualization smoke workflow."""

    reporter_pattern: str
    title: str
    y_label: str
    x_label: str = "Steps"
    exclude_pattern: str | None = None
    show_mean_line: bool = False


class ABMNetLogoVizConfig(BaseModel):
    """Defines how one ABM is simulated and plotted before any LLM inference."""

    experiment_name: str
    num_runs: int
    max_ticks: int
    interval: int
    experiment_parameters: dict[str, bool | int | float | str] = Field(default_factory=dict)
    smoke_num_runs: int | None = None
    smoke_max_ticks: int | None = None
    smoke_interval: int | None = None
    fallback_mode: Literal["on_error", "always"] = "on_error"
    fallback_csv: str | None = None
    fallback_plot_dir: str | None = None
    reporters: list[str]
    plots: list[ABMNetLogoVizPlotConfig]


class ABMConfig(BaseModel):
    """Defines case-study-specific defaults."""

    name: str
    metric_pattern: str
    metric_description: str
    plot_descriptions: list[str]
    default_input_csv: str | None = None
    default_parameters_txt: str | None = None
    default_documentation_txt: str | None = None
    netlogo_viz: ABMNetLogoVizConfig | None = None


class EvaluationConfig(BaseModel):
    """Defines default evaluation behavior for summary scoring."""

    use_reference_metrics: bool = Field(
        default=True,
        validation_alias=AliasChoices("use_reference_metrics"),
    )
    use_token_f1: bool = True


class LoggingConfig(BaseModel):
    """Provides centralized runtime logging settings."""

    level: str = "INFO"
    format: str = "%(asctime)s %(levelname)s %(name)s %(message)s"


class ExperimentGroundTruthConfig(BaseModel):
    """Human reference files used for ABM-specific lexical scoring."""

    fauna: str
    grazing: str
    milk_consumption: str


class OptionalExperimentGroundTruthConfig(BaseModel):
    """Optional secondary human reference files used for additional lexical scoring."""

    fauna: str | None = None
    grazing: str | None = None
    milk_consumption: str | None = None


class ExperimentSettings(BaseModel):
    """Paper-aligned experiment settings required by CLI and reproducibility tooling."""

    ground_truth: ExperimentGroundTruthConfig
    modeler_ground_truth: OptionalExperimentGroundTruthConfig | None = None
    qualitative_example_text_dir: str | None = None
    human_reference_dir: str | None = None


class RuntimeLLMRequestDefaults(BaseModel):
    """Defines default request sampling parameters across providers."""

    temperature: float = 1.0
    max_tokens: int = 1000
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0


class RuntimeRunDefaults(BaseModel):
    """Defines CLI defaults for the main `run` command."""

    provider: ProviderName = "openrouter"
    model: str = "moonshotai/kimi-k2.5"
    output_dir: str = "results/pipeline"
    metric_pattern: str = "mean"
    metric_description: str = "simulation trend"
    evidence_mode: RuntimeEvidenceMode = "plot+table"
    text_source_mode: RuntimeTextSourceMode = "summary_only"
    summarizers: tuple[SummarizerId, ...] = ("bart", "bert", "t5", "longformer_ext")


class RuntimeQualitativeDefaults(BaseModel):
    """Defines CLI defaults for qualitative evaluation command."""

    provider: ProviderName = "openrouter"
    model: str = "google/gemini-3.1-pro-preview"


class RuntimeSmokeDefaults(BaseModel):
    """Defines CLI defaults for the debug smoke command."""

    provider: ProviderName = "openrouter"
    output_dir: str = "results/smoke_debug"
    model: str = "google/gemini-3.1-pro-preview"
    metric_pattern: str = "mean"
    metric_description: str = "simulation trend"
    evidence_mode: RuntimeEvidenceMode = "plot+table"
    text_source_mode: RuntimeTextSourceMode = "summary_only"
    summarizers: tuple[SummarizerId, ...] = ("bart", "bert", "t5", "longformer_ext")
    run_qualitative: bool = True
    run_sweep: bool = True


class RuntimeDoEDefaults(BaseModel):
    """Defines CLI defaults for DOE analysis command."""

    output_csv: str = "results/doe/anova_factorial_contributions.csv"
    max_interaction_order: int = 2


class RuntimeDefaultsConfig(BaseModel):
    """Centralized runtime defaults consumed by CLI, adapters, and pipeline metadata."""

    llm_request: RuntimeLLMRequestDefaults = RuntimeLLMRequestDefaults()
    run: RuntimeRunDefaults = RuntimeRunDefaults()
    qualitative: RuntimeQualitativeDefaults = RuntimeQualitativeDefaults()
    smoke: RuntimeSmokeDefaults = RuntimeSmokeDefaults()
    doe: RuntimeDoEDefaults = RuntimeDoEDefaults()
