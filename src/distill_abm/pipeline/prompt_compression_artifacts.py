"""Shared prompt-compression artifact writers for smoke workflows."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

PROMPT_COMPRESSION_FILENAME = "trend_prompt_compression.json"
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
