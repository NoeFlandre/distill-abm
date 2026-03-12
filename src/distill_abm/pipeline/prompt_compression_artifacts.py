"""Shared prompt-compression artifact writers for smoke workflows."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

PROMPT_COMPRESSION_FILENAME = "trend_prompt_compression.json"
PROMPT_COMPRESSION_SUMMARY_FILENAME = "prompt_compression_summary.json"
PRE_COMPRESSION_PROMPT_FILENAME = "trend_prompt_pre_compression.txt"
COMPRESSED_PROMPT_FILENAME = "trend_prompt_compressed.txt"


class PromptCompressionAttempt(BaseModel):
    """One prompt-generation attempt for a trend request."""

    attempt_index: int
    table_downsample_stride: int
    compression_tier: int
    prompt_length: int


class PromptCompressionArtifacts(BaseModel):
    """Persisted summary of prompt compression across attempts."""

    triggered: bool
    compression_count: int
    attempt_count: int
    attempts: list[PromptCompressionAttempt] = Field(default_factory=list)


class PromptCompressionRunEntry(BaseModel):
    """One prompt-compression record inside a run-level summary."""

    source_run_root: Path
    scope: str
    case_id: str
    abm: str
    evidence_mode: str
    prompt_variant: str
    repetition: int | None = None
    plot_index: int | None = None
    artifacts_dir: Path
    compression_artifact_path: Path
    pre_compression_prompt_path: Path | None = None
    compressed_prompt_path: Path | None = None
    triggered: bool
    compression_count: int
    attempt_count: int
    attempts: list[PromptCompressionAttempt] = Field(default_factory=list)


class PromptCompressionRunSummary(BaseModel):
    """Aggregated prompt-compression summary for one run root."""

    run_root: Path
    total_entries: int
    triggered_entries: int
    total_compressions: int
    entries: list[PromptCompressionRunEntry] = Field(default_factory=list)


def write_prompt_compression_artifacts(
    *,
    output_dir: Path,
    attempts: Sequence[PromptCompressionAttempt],
    prompts: Sequence[str],
) -> None:
    """Write the current prompt-compression state for a trend directory."""
    if len(attempts) != len(prompts):
        raise ValueError("prompt compression attempts and prompts must have the same length")
    if not attempts:
        raise ValueError("prompt compression artifacts require at least one attempt")

    payload = PromptCompressionArtifacts(
        triggered=len(attempts) > 1,
        compression_count=max(len(attempts) - 1, 0),
        attempt_count=len(attempts),
        attempts=list(attempts),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / PROMPT_COMPRESSION_FILENAME).write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    pre_compression_path = output_dir / PRE_COMPRESSION_PROMPT_FILENAME
    compressed_path = output_dir / COMPRESSED_PROMPT_FILENAME
    if payload.triggered:
        pre_compression_path.write_text(prompts[0], encoding="utf-8")
        compressed_path.write_text(prompts[-1], encoding="utf-8")
        return

    for path in (pre_compression_path, compressed_path):
        if path.exists():
            path.unlink()


def read_prompt_compression_artifacts(output_dir: Path) -> PromptCompressionArtifacts | None:
    """Read one trend prompt-compression artifact payload when present."""
    path = output_dir / PROMPT_COMPRESSION_FILENAME
    if not path.exists():
        return None
    return PromptCompressionArtifacts.model_validate_json(path.read_text(encoding="utf-8"))


def build_prompt_compression_run_entry(
    *,
    source_run_root: Path,
    scope: str,
    case_id: str,
    abm: str,
    evidence_mode: str,
    prompt_variant: str,
    artifacts_dir: Path,
    repetition: int | None = None,
    plot_index: int | None = None,
) -> PromptCompressionRunEntry | None:
    """Build one run-summary entry from a case/trend artifact directory."""
    payload = read_prompt_compression_artifacts(artifacts_dir)
    if payload is None:
        return None
    pre_compression_path = artifacts_dir / PRE_COMPRESSION_PROMPT_FILENAME
    compressed_path = artifacts_dir / COMPRESSED_PROMPT_FILENAME
    return PromptCompressionRunEntry(
        source_run_root=source_run_root,
        scope=scope,
        case_id=case_id,
        abm=abm,
        evidence_mode=evidence_mode,
        prompt_variant=prompt_variant,
        repetition=repetition,
        plot_index=plot_index,
        artifacts_dir=artifacts_dir,
        compression_artifact_path=artifacts_dir / PROMPT_COMPRESSION_FILENAME,
        pre_compression_prompt_path=pre_compression_path if pre_compression_path.exists() else None,
        compressed_prompt_path=compressed_path if compressed_path.exists() else None,
        triggered=payload.triggered,
        compression_count=payload.compression_count,
        attempt_count=payload.attempt_count,
        attempts=payload.attempts,
    )


def build_prompt_compression_run_summary(
    *,
    run_root: Path,
    entries: Sequence[PromptCompressionRunEntry],
) -> PromptCompressionRunSummary:
    """Build the aggregated prompt-compression summary payload for one run."""
    normalized_entries = list(entries)
    return PromptCompressionRunSummary(
        run_root=run_root,
        total_entries=len(normalized_entries),
        triggered_entries=sum(1 for entry in normalized_entries if entry.triggered),
        total_compressions=sum(entry.compression_count for entry in normalized_entries),
        entries=normalized_entries,
    )


def write_prompt_compression_run_summary(
    *,
    run_root: Path,
    entries: Sequence[PromptCompressionRunEntry],
) -> Path:
    """Write the run-level prompt-compression summary JSON."""
    summary = build_prompt_compression_run_summary(run_root=run_root, entries=entries)
    path = run_root / PROMPT_COMPRESSION_SUMMARY_FILENAME
    path.write_text(json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def read_prompt_compression_run_summary(path: Path) -> PromptCompressionRunSummary | None:
    """Read a run-level prompt-compression summary when present."""
    if not path.exists():
        return None
    return PromptCompressionRunSummary.model_validate_json(path.read_text(encoding="utf-8"))
