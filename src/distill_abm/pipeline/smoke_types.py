"""Structured types for smoke-suite orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from distill_abm.configs.models import SummarizerId
from distill_abm.pipeline.run import EvidenceMode, TextSourceMode

SmokeStatus = Literal["ok", "failed", "skipped"]
SmokeResponseKind = Literal["context", "trend"]

RESPONSE_BUNDLE_COLUMNS: tuple[str, ...] = (
    "run_output_dir",
    "case_id",
    "response_kind",
    "case_status",
    "resumed_from_existing",
    "provider",
    "model",
    "temperature",
    "max_tokens",
    "max_retries",
    "retry_backoff_seconds",
    "evidence_mode",
    "text_source_mode",
    "enabled_style_features",
    "summarizers",
    "input_csv_path",
    "parameters_path",
    "documentation_path",
    "scoring_reference_path",
    "scoring_reference_source",
    "scoring_reference_text",
    "prompt_path",
    "prompt_text",
    "prompt_signature",
    "prompt_length",
    "response_path",
    "response_text",
    "response_length",
    "evidence_image_path",
    "plot_path",
    "stats_table_csv_path",
    "report_csv_path",
    "metadata_path",
    "case_manifest_path",
    "selected_token_f1",
    "selected_bleu",
    "selected_meteor",
    "selected_rouge1",
    "selected_rouge2",
    "selected_rouge_l",
    "selected_flesch_reading_ease",
    "full_token_f1",
    "full_bleu",
    "full_meteor",
    "full_rouge1",
    "full_rouge2",
    "full_rouge_l",
    "full_flesch_reading_ease",
    "summary_token_f1",
    "summary_bleu",
    "summary_meteor",
    "summary_rouge1",
    "summary_rouge2",
    "summary_rouge_l",
    "summary_flesch_reading_ease",
    "error",
    "inputs_json",
    "llm_json",
    "scores_json",
    "reproducibility_json",
    "summarizers_json",
)


class SmokeCase(BaseModel):
    """Represents one smoke case in the smoke matrix."""

    case_id: str
    evidence_mode: EvidenceMode
    text_source_mode: TextSourceMode
    enabled_style_features: tuple[str, ...] | None = None
    summarizers: tuple[SummarizerId, ...] | None = None


class QualitativeOutcome(BaseModel):
    """Stores one qualitative metric outcome for a smoke case."""

    metric: str
    status: str
    score: int | None = None
    reasoning: str | None = None
    model: str | None = None
    prompt_path: Path | None = None
    output_path: Path | None = None
    error: str | None = None


class SmokeCaseResult(BaseModel):
    """Stores artifacts and status for one smoke case."""

    case: SmokeCase
    status: str
    output_dir: Path
    report_csv: Path | None = None
    plot_path: Path | None = None
    metadata_path: Path | None = None
    context_prompt_path: Path | None = None
    trend_prompt_path: Path | None = None
    stats_table_csv_path: Path | None = None
    context_response_path: Path | None = None
    trend_full_response_path: Path | None = None
    trend_summary_response_path: Path | None = None
    case_rows_csv_path: Path | None = None
    case_manifest_path: Path | None = None
    resumed_from_existing: bool = False
    qualitative: list[QualitativeOutcome] = Field(default_factory=list)
    error: str | None = None


class SmokeSuiteInputs(BaseModel):
    """Defines required input paths and runtime settings for smoke execution."""

    csv_path: Path
    parameters_path: Path
    documentation_path: Path
    output_dir: Path
    model: str
    metric_pattern: str
    metric_description: str
    plot_description: str | None = None
    sweep_plot_descriptions: list[str] | None = None
    summarizers: tuple[SummarizerId, ...] = ("bart", "bert", "t5", "longformer_ext")
    allow_summary_fallback: bool = False
    text_source_mode: TextSourceMode = "summary_only"
    evidence_mode: EvidenceMode = "plot+table"
    scoring_reference_path: Path | None = None
    additional_scoring_reference_paths: dict[str, Path] = Field(default_factory=dict)


class SmokeSuiteResult(BaseModel):
    """Top-level smoke-suite result and report pointers."""

    provider: str
    model: str
    started_at_utc: str
    finished_at_utc: str
    inputs: SmokeSuiteInputs
    qualitative_policy: str = "debug_same_model"
    success: bool
    failed_cases: list[str] = Field(default_factory=list)
    cases: list[SmokeCaseResult] = Field(default_factory=list)
    doe_status: str = "skipped"
    doe_output_csv: Path | None = None
    doe_error: str | None = None
    sweep_status: str = "skipped"
    sweep_output_csv: Path | None = None
    sweep_error: str | None = None
    run_master_csv_path: Path | None = None
    global_master_csv_path: Path | None = None
    report_markdown_path: Path
    report_json_path: Path
