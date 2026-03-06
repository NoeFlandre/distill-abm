"""Granular smoke reporting for NetLogo ingestion artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from distill_abm.ingest.netlogo_workflow import run_ingest_workflow
from distill_abm.utils import detect_placeholder_signals

IngestSmokeStatus = Literal["ok", "failed"]


class IngestSmokeStage(BaseModel):
    """One artifact-focused ingest smoke stage."""

    stage_id: str
    artifact_key: str
    description: str
    inspect_placeholder_signals: bool = True


class IngestSmokeArtifact(BaseModel):
    """Debug-friendly artifact metadata for one stage."""

    path: Path
    exists: bool
    size_bytes: int = 0
    sha256: str | None = None
    preview: str = ""
    placeholder_signals: list[str] = Field(default_factory=list)


class IngestSmokeStageResult(BaseModel):
    """Result for one stage on one ABM model."""

    stage: IngestSmokeStage
    status: IngestSmokeStatus
    artifact: IngestSmokeArtifact
    error: str | None = None


class IngestSmokeAbmResult(BaseModel):
    """Aggregated smoke result for one ABM."""

    abm: str
    model_path: Path
    output_dir: Path
    status: IngestSmokeStatus
    artifact_index_path: Path | None = None
    stage_results: list[IngestSmokeStageResult] = Field(default_factory=list)
    error: str | None = None


class IngestSmokeSuiteResult(BaseModel):
    """Suite-level ingest smoke report."""

    started_at_utc: str
    finished_at_utc: str
    output_root: Path
    success: bool
    failed_abms: list[str] = Field(default_factory=list)
    selected_stage_ids: list[str] = Field(default_factory=list)
    abms: list[IngestSmokeAbmResult] = Field(default_factory=list)
    report_markdown_path: Path
    report_json_path: Path


def default_ingest_smoke_stages() -> list[IngestSmokeStage]:
    """Return the canonical artifact checks for ingestion debugging."""
    return [
        IngestSmokeStage(
            stage_id="experiment-parameters",
            artifact_key="experiment_parameters_json",
            description="Resolved experiment parameter JSON.",
        ),
        IngestSmokeStage(
            stage_id="gui-parameters",
            artifact_key="gui_parameters_json",
            description="Extracted GUI controls JSON.",
        ),
        IngestSmokeStage(
            stage_id="updated-parameters",
            artifact_key="updated_gui_parameters_json",
            description="GUI controls updated with experiment values.",
        ),
        IngestSmokeStage(
            stage_id="narrative",
            artifact_key="narrative_txt",
            description="Narrative text synthesized from parameters.",
        ),
        IngestSmokeStage(
            stage_id="documentation",
            artifact_key="documentation_json",
            description="Raw documentation extraction.",
        ),
        IngestSmokeStage(
            stage_id="cleaned-documentation",
            artifact_key="cleaned_documentation_json",
            description="Documentation after URL cleanup.",
        ),
        IngestSmokeStage(
            stage_id="final-documentation",
            artifact_key="final_documentation_txt",
            description="Final documentation text handed downstream.",
        ),
        IngestSmokeStage(
            stage_id="code",
            artifact_key="extracted_code_txt",
            description="Extracted NetLogo source code text.",
            inspect_placeholder_signals=False,
        ),
    ]


def run_ingest_smoke_suite(
    *,
    abm_models: dict[str, Path],
    output_root: Path,
    stage_ids: list[str] | None = None,
) -> IngestSmokeSuiteResult:
    """Run ingestion workflow for each ABM and report artifact-level smoke outcomes."""
    started_at = datetime.now(UTC)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_stages = _select_stages(stage_ids)

    abm_results: list[IngestSmokeAbmResult] = []
    failed_abms: list[str] = []
    for abm, model_path in sorted(abm_models.items()):
        output_dir = output_root / abm
        try:
            artifact_paths = run_ingest_workflow(model_path=model_path, experiment_parameters={}, output_dir=output_dir)
            artifact_index_path = output_dir / "ingest_artifact_index.json"
            artifact_index_path.write_text(
                json.dumps({key: str(path) for key, path in artifact_paths.items()}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            stage_results = [
                _build_stage_result(stage=stage, artifact_path=artifact_paths[stage.artifact_key])
                for stage in selected_stages
            ]
            status: IngestSmokeStatus = "failed" if any(item.status == "failed" for item in stage_results) else "ok"
            if status == "failed":
                failed_abms.append(abm)
            abm_results.append(
                IngestSmokeAbmResult(
                    abm=abm,
                    model_path=model_path,
                    output_dir=output_dir,
                    status=status,
                    artifact_index_path=artifact_index_path,
                    stage_results=stage_results,
                )
            )
        except Exception as exc:
            failed_abms.append(abm)
            abm_results.append(
                IngestSmokeAbmResult(
                    abm=abm,
                    model_path=model_path,
                    output_dir=output_dir,
                    status="failed",
                    error=str(exc),
                )
            )

    finished_at = datetime.now(UTC)
    report_json_path = output_root / "ingest_smoke_report.json"
    report_markdown_path = output_root / "ingest_smoke_report.md"
    result = IngestSmokeSuiteResult(
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        output_root=output_root,
        success=not failed_abms,
        failed_abms=failed_abms,
        selected_stage_ids=[stage.stage_id for stage in selected_stages],
        abms=abm_results,
        report_markdown_path=report_markdown_path,
        report_json_path=report_json_path,
    )
    report_json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    report_markdown_path.write_text(_render_markdown_report(result), encoding="utf-8")
    return result


def _select_stages(stage_ids: list[str] | None) -> list[IngestSmokeStage]:
    stages = default_ingest_smoke_stages()
    if not stage_ids:
        return stages
    by_id = {stage.stage_id: stage for stage in stages}
    unknown = [stage_id for stage_id in stage_ids if stage_id not in by_id]
    if unknown:
        known = ", ".join(sorted(by_id))
        raise ValueError(f"unknown ingest smoke stage(s): {', '.join(unknown)}. Known stages: {known}")
    return [by_id[stage_id] for stage_id in stage_ids]


def _build_stage_result(*, stage: IngestSmokeStage, artifact_path: Path) -> IngestSmokeStageResult:
    artifact = _inspect_artifact(artifact_path, inspect_placeholder_signals=stage.inspect_placeholder_signals)
    status: IngestSmokeStatus = (
        "ok" if artifact.exists and artifact.size_bytes > 0 and not artifact.placeholder_signals else "failed"
    )
    error = None
    if not artifact.exists or artifact.size_bytes <= 0:
        error = f"artifact missing or empty: {artifact_path}"
    elif artifact.placeholder_signals:
        signals = ", ".join(artifact.placeholder_signals)
        error = f"artifact contains placeholder-like content: {signals}"
    return IngestSmokeStageResult(stage=stage, status=status, artifact=artifact, error=error)


def _inspect_artifact(path: Path, *, inspect_placeholder_signals: bool) -> IngestSmokeArtifact:
    if not path.exists():
        return IngestSmokeArtifact(path=path, exists=False)
    text = path.read_text(encoding="utf-8", errors="replace")
    return IngestSmokeArtifact(
        path=path,
        exists=True,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        preview=text[:200],
        placeholder_signals=detect_placeholder_signals(text) if inspect_placeholder_signals else [],
    )


def _render_markdown_report(result: IngestSmokeSuiteResult) -> str:
    lines = [
        "# Ingest Smoke Report",
        "",
        f"- success: `{result.success}`",
        f"- stages: `{', '.join(result.selected_stage_ids)}`",
        f"- failed_abms: `{', '.join(result.failed_abms) if result.failed_abms else 'none'}`",
        "",
    ]
    for abm in result.abms:
        lines.append(f"## {abm.abm}")
        lines.append(f"- status: `{abm.status}`")
        if abm.error:
            lines.append(f"- error: `{abm.error}`")
        for stage_result in abm.stage_results:
            lines.append(
                f"- {stage_result.stage.stage_id}: `{stage_result.status}` -> `{stage_result.artifact.path}`"
            )
        lines.append("")
    return "\n".join(lines)
