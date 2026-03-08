"""Typed DOE smoke models and canonical matrix definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from distill_abm.configs.models import SummarizerId

DoESmokeStatus = Literal["ok", "failed"]
DoESmokeErrorCode = Literal[
    "missing_or_empty_artifact",
    "placeholder_detected",
    "unmatched_metric_pattern",
    "plot_count_mismatch",
    "model_preflight_failed",
]
DoETextSourceMode = Literal["summary_only", "full_text_only"]

CANONICAL_DOE_MODEL_IDS: tuple[str, ...] = ("qwen3_5_local", "kimi_k2_5", "gemini_3_1_pro_preview")
CANONICAL_EVIDENCE_MODES: tuple[Literal["plot", "table", "plot+table"], ...] = ("plot", "table", "plot+table")
CANONICAL_REPETITIONS: tuple[int, ...] = (1, 2, 3)


class DoESmokePlotInput(BaseModel):
    """One plot-level evidence item used by the real DOE."""

    plot_index: int
    reporter_pattern: str
    plot_description: str
    plot_path: Path


class DoESmokeAbmInput(BaseModel):
    """Resolved ABM-level pre-LLM inputs."""

    abm: str
    csv_path: Path
    parameters_path: Path
    documentation_path: Path
    metric_pattern: str
    metric_description: str
    plots: list[DoESmokePlotInput]
    source_viz_artifact_source: Literal["simulated", "fallback", "unknown"] = "unknown"


class DoESmokeModelSpec(BaseModel):
    """One candidate LLM participating in the DOE."""

    model_id: str
    provider: str
    model: str
    preflight_error: str | None = None


class DoESmokePromptVariant(BaseModel):
    """One prompt-style combination in the DOE."""

    variant_id: str
    enabled_style_features: tuple[str, ...] = ()


class DoESmokeSummarizationSpec(BaseModel):
    """One summarization condition in the DOE."""

    summarization_mode: str
    text_source_mode: DoETextSourceMode
    summarizers: tuple[SummarizerId, ...] = ()


class DoESmokeArtifact(BaseModel):
    """Stable artifact metadata for reports."""

    path: Path
    exists: bool
    size_bytes: int = 0
    sha256: str | None = None
    preview: str = ""


class DoESmokePlotPlan(BaseModel):
    """One planned trend request for one plot."""

    plot_index: int
    reporter_pattern: str
    plot_description: str
    prompt_path: Path
    plot_path: Path
    table_csv_path: Path | None
    evidence_mode: str
    status: DoESmokeStatus
    error_code: DoESmokeErrorCode | None = None
    error: str | None = None


class DoESmokeSharedAbmResult(BaseModel):
    """Shared ABM bundle reused by many DOE cases."""

    abm: str
    shared_dir: Path
    csv_path: Path
    parameters_path: Path
    documentation_path: Path
    source_viz_artifact_source: str
    plot_count: int
    plot_request_count: int
    artifact_index_path: Path
    stage_errors: list[str] = Field(default_factory=list)


class DoESmokeCaseResult(BaseModel):
    """One exact DOE combination for one ABM."""

    case_id: str
    abm: str
    model_id: str
    provider: str
    model: str
    evidence_mode: str
    summarization_mode: str
    text_source_mode: str
    prompt_variant: str
    enabled_style_features: list[str] = Field(default_factory=list)
    summarizers: list[str] = Field(default_factory=list)
    repetition: int
    request_count: int
    failed_plot_indices: list[int] = Field(default_factory=list)
    error_codes: list[str] = Field(default_factory=list)
    status: DoESmokeStatus
    context_prompt_path: Path


class DoESmokeSuiteResult(BaseModel):
    """Top-level DOE smoke report."""

    started_at_utc: str
    finished_at_utc: str
    output_root: Path
    success: bool
    total_cases: int
    total_planned_requests: int
    total_context_requests: int
    total_trend_requests: int
    failed_case_ids: list[str] = Field(default_factory=list)
    case_index_jsonl_path: Path
    request_index_jsonl_path: Path
    design_matrix_csv_path: Path
    request_matrix_csv_path: Path
    request_review_csv_path: Path
    report_markdown_path: Path
    report_json_path: Path
    abm_shared: dict[str, DoESmokeSharedAbmResult] = Field(default_factory=dict)
    cases: list[DoESmokeCaseResult] = Field(default_factory=list)


def canonical_prompt_variants() -> tuple[DoESmokePromptVariant, ...]:
    """Return the paper DOE prompt combinations."""
    return (
        DoESmokePromptVariant(variant_id="none", enabled_style_features=()),
        DoESmokePromptVariant(variant_id="role", enabled_style_features=("role",)),
        DoESmokePromptVariant(variant_id="insights", enabled_style_features=("insights",)),
        DoESmokePromptVariant(variant_id="example", enabled_style_features=("example",)),
        DoESmokePromptVariant(variant_id="role+example", enabled_style_features=("role", "example")),
        DoESmokePromptVariant(variant_id="role+insights", enabled_style_features=("role", "insights")),
        DoESmokePromptVariant(variant_id="insights+example", enabled_style_features=("insights", "example")),
        DoESmokePromptVariant(variant_id="all_three", enabled_style_features=("role", "insights", "example")),
    )


def canonical_summarization_specs() -> tuple[DoESmokeSummarizationSpec, ...]:
    """Return the paper DOE summarization conditions."""
    return (
        DoESmokeSummarizationSpec(
            summarization_mode="none",
            text_source_mode="full_text_only",
            summarizers=(),
        ),
        DoESmokeSummarizationSpec(
            summarization_mode="bart",
            text_source_mode="summary_only",
            summarizers=("bart",),
        ),
        DoESmokeSummarizationSpec(
            summarization_mode="bert",
            text_source_mode="summary_only",
            summarizers=("bert",),
        ),
        DoESmokeSummarizationSpec(
            summarization_mode="t5",
            text_source_mode="summary_only",
            summarizers=("t5",),
        ),
        DoESmokeSummarizationSpec(
            summarization_mode="longformer_ext",
            text_source_mode="summary_only",
            summarizers=("longformer_ext",),
        ),
    )
