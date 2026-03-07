"""Stable CLI result models and artifact descriptors."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, Field


class ArtifactDescriptor(BaseModel):
    """Stable artifact description for agent-readable manifests."""

    path: Path
    exists: bool
    size_bytes: int = 0
    sha256: str | None = None


class IngestCommandResult(BaseModel):
    """Structured result for single-model ingest CLI runs."""

    command: str = "ingest-netlogo"
    output_dir: Path
    artifact_manifest_path: Path
    artifacts: dict[str, ArtifactDescriptor]


class IngestSuiteCommandResult(BaseModel):
    """Structured result for multi-model ingest CLI runs."""

    command: str = "ingest-netlogo-suite"
    output_root: Path
    artifact_manifest_path: Path
    abms: dict[str, dict[str, ArtifactDescriptor]]
    skipped_abms: list[str] = Field(default_factory=list)


class RunCommandResult(BaseModel):
    """Structured result for one pipeline run."""

    command: str = "run"
    output_dir: Path
    plot_path: Path
    report_csv_path: Path
    metadata_path: Path | None = None
    artifact_manifest_path: Path
    artifacts: dict[str, ArtifactDescriptor]


class SmokeCommandResult(BaseModel):
    """Structured result for smoke-style CLI runs."""

    command: str
    success: bool
    report_json_path: Path
    report_markdown_path: Path
    failed_items: list[str] = Field(default_factory=list)
    nested_artifacts: dict[str, Path] = Field(default_factory=dict)


class DoeCommandResult(BaseModel):
    """Structured result for DOE analysis."""

    command: str = "analyze-doe"
    success: bool
    output_csv: Path


class DescribeAbmResult(BaseModel):
    """Read-only summary of one configured ABM."""

    abm: str
    config_path: Path
    model_path: Path
    experiment_parameters_path: Path | None = None
    scoring_reference_path: Path | None = None
    metric_pattern: str
    metric_description: str
    plot_descriptions: list[str] = Field(default_factory=list)


class DescribeRunResult(BaseModel):
    """Read-only summary of one existing run output directory."""

    output_dir: Path
    metadata_path: Path
    available_artifacts: dict[str, Path] = Field(default_factory=dict)
    run_signature: str | None = None
    selected_text_source: str | None = None
    evidence_mode: str | None = None
    requested_evidence_mode: str | None = None
    matched_metric_columns: list[str] = Field(default_factory=list)


class DescribeArtifactsResult(BaseModel):
    """Read-only summary of an existing ingest artifact directory."""

    root: Path
    manifest_path: Path | None = None
    artifact_index_path: Path | None = None
    artifacts: dict[str, ArtifactDescriptor] = Field(default_factory=dict)


class HealthCheckItem(BaseModel):
    """One health-check item with status and details."""

    ok: bool
    detail: str


class HealthCheckResult(BaseModel):
    """Structured result for repository health checks."""

    command: str = "health-check"
    success: bool
    checks: dict[str, HealthCheckItem]


def build_artifact_descriptors(paths: Mapping[str, Path | None]) -> dict[str, ArtifactDescriptor]:
    """Describe a set of artifact paths with stable metadata."""
    return {
        key: describe_artifact(path)
        for key, path in paths.items()
        if path is not None
    }


def describe_artifact(path: Path) -> ArtifactDescriptor:
    """Return a stable artifact descriptor with existence, size, and digest."""
    if not path.exists():
        return ArtifactDescriptor(path=path, exists=False)
    return ArtifactDescriptor(
        path=path,
        exists=True,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
