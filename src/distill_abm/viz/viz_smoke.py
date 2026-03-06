"""Granular smoke reporting for visualization artifact generation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from distill_abm.ingest.csv_ingest import load_simulation_csv
from distill_abm.viz.plots import (
    PlotError,
    generate_stats_table,
    plot_metric_bundle,
    render_stats_table_image,
    render_stats_table_markdown,
)

VizSmokeStatus = Literal["ok", "failed"]
VizSmokeErrorCode = Literal[
    "missing_or_empty_artifact",
    "csv_load_failed",
    "plot_generation_failed",
    "stats_generation_failed",
]


class VizSmokeStage(BaseModel):
    """One artifact-focused visualization smoke stage."""

    stage_id: str
    artifact_key: str
    description: str


class VizSmokeArtifact(BaseModel):
    """Debug-friendly artifact metadata for one visualization stage."""

    path: Path
    exists: bool
    size_bytes: int = 0
    sha256: str | None = None
    preview: str = ""


class VizSmokeStageResult(BaseModel):
    """Result for one stage on one ABM visualization input."""

    stage: VizSmokeStage
    status: VizSmokeStatus
    artifact: VizSmokeArtifact
    error_code: VizSmokeErrorCode | None = None
    error: str | None = None


class VizSmokeAbmResult(BaseModel):
    """Aggregated viz smoke result for one ABM."""

    abm: str
    csv_path: Path
    metric_pattern: str
    output_dir: Path
    status: VizSmokeStatus
    artifact_index_path: Path | None = None
    stage_results: list[VizSmokeStageResult] = Field(default_factory=list)
    error_code: VizSmokeErrorCode | None = None
    error: str | None = None


class VizSmokeSuiteResult(BaseModel):
    """Suite-level visualization smoke report."""

    started_at_utc: str
    finished_at_utc: str
    output_root: Path
    success: bool
    failed_abms: list[str] = Field(default_factory=list)
    selected_stage_ids: list[str] = Field(default_factory=list)
    abms: list[VizSmokeAbmResult] = Field(default_factory=list)
    report_markdown_path: Path
    report_json_path: Path


def default_viz_smoke_stages() -> list[VizSmokeStage]:
    """Return the canonical visualization checks for agent debugging."""
    return [
        VizSmokeStage(stage_id="csv-load", artifact_key="csv_summary_json", description="Loaded CSV summary."),
        VizSmokeStage(stage_id="plot", artifact_key="plot_png", description="Generated metric plot PNG."),
        VizSmokeStage(stage_id="stats-csv", artifact_key="stats_table_csv", description="Generated stats table CSV."),
        VizSmokeStage(
            stage_id="stats-markdown",
            artifact_key="stats_table_markdown",
            description="Generated stats table markdown.",
        ),
        VizSmokeStage(
            stage_id="stats-image",
            artifact_key="stats_table_png",
            description="Generated stats table PNG image.",
        ),
    ]


def run_viz_smoke_suite(
    *,
    abm_inputs: dict[str, tuple[Path, str]],
    output_root: Path,
    stage_ids: list[str] | None = None,
) -> VizSmokeSuiteResult:
    """Run visualization generation checks for each ABM CSV input and report artifact outcomes."""
    started_at = datetime.now(UTC)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_stages = _select_stages(stage_ids)

    abm_results: list[VizSmokeAbmResult] = []
    failed_abms: list[str] = []
    for abm, (csv_path, metric_pattern) in sorted(abm_inputs.items()):
        output_dir = output_root / abm
        try:
            artifact_paths = _generate_viz_artifacts(
                csv_path=csv_path,
                metric_pattern=metric_pattern,
                output_dir=output_dir,
            )
            artifact_index_path = output_dir / "viz_artifact_index.json"
            artifact_index_path.write_text(
                json.dumps({key: str(path) for key, path in artifact_paths.items()}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            stage_results = [
                _build_stage_result(stage=stage, artifact_path=artifact_paths[stage.artifact_key])
                for stage in selected_stages
            ]
            status: VizSmokeStatus = "failed" if any(item.status == "failed" for item in stage_results) else "ok"
            if status == "failed":
                failed_abms.append(abm)
            abm_results.append(
                VizSmokeAbmResult(
                    abm=abm,
                    csv_path=csv_path,
                    metric_pattern=metric_pattern,
                    output_dir=output_dir,
                    status=status,
                    artifact_index_path=artifact_index_path,
                    stage_results=stage_results,
                )
            )
        except (PlotError, ValueError) as exc:
            failed_abms.append(abm)
            abm_results.append(
                VizSmokeAbmResult(
                    abm=abm,
                    csv_path=csv_path,
                    metric_pattern=metric_pattern,
                    output_dir=output_dir,
                    status="failed",
                    error_code="plot_generation_failed",
                    error=str(exc),
                )
            )
        except Exception as exc:
            failed_abms.append(abm)
            abm_results.append(
                VizSmokeAbmResult(
                    abm=abm,
                    csv_path=csv_path,
                    metric_pattern=metric_pattern,
                    output_dir=output_dir,
                    status="failed",
                    error_code="csv_load_failed",
                    error=str(exc),
                )
            )

    finished_at = datetime.now(UTC)
    report_json_path = output_root / "viz_smoke_report.json"
    report_markdown_path = output_root / "viz_smoke_report.md"
    result = VizSmokeSuiteResult(
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


def _select_stages(stage_ids: list[str] | None) -> list[VizSmokeStage]:
    stages = default_viz_smoke_stages()
    if not stage_ids:
        return stages
    by_id = {stage.stage_id: stage for stage in stages}
    unknown = [stage_id for stage_id in stage_ids if stage_id not in by_id]
    if unknown:
        known = ", ".join(sorted(by_id))
        raise ValueError(f"unknown viz smoke stage(s): {', '.join(unknown)}. Known stages: {known}")
    return [by_id[stage_id] for stage_id in stage_ids]


def _generate_viz_artifacts(*, csv_path: Path, metric_pattern: str, output_dir: Path) -> dict[str, Path]:
    frame = load_simulation_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_summary_json = output_dir / "csv_summary.json"
    csv_summary_json.write_text(
        json.dumps(
            {
                "row_count": len(frame),
                "column_count": len(frame.columns),
                "columns": [str(column) for column in frame.columns],
                "metric_pattern": metric_pattern,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_png = plot_metric_bundle(
        frame=frame,
        include_pattern=metric_pattern,
        output_dir=output_dir,
        title=f"{metric_pattern} smoke plot",
        y_label="value",
    )
    stats_table = generate_stats_table(frame, include_pattern=metric_pattern)
    stats_table_csv = output_dir / "stats_table.csv"
    stats_table.to_csv(stats_table_csv, index=False)
    stats_table_markdown = output_dir / "stats_table.md"
    stats_table_markdown.write_text(render_stats_table_markdown(stats_table), encoding="utf-8")
    stats_table_png = render_stats_table_image(stats_table, output_dir / "stats_table.png")

    return {
        "csv_summary_json": csv_summary_json,
        "plot_png": plot_png,
        "stats_table_csv": stats_table_csv,
        "stats_table_markdown": stats_table_markdown,
        "stats_table_png": stats_table_png,
    }


def _build_stage_result(*, stage: VizSmokeStage, artifact_path: Path) -> VizSmokeStageResult:
    artifact = _inspect_artifact(artifact_path)
    status: VizSmokeStatus = "ok" if artifact.exists and artifact.size_bytes > 0 else "failed"
    error = None
    error_code: VizSmokeErrorCode | None = None
    if not artifact.exists or artifact.size_bytes <= 0:
        error = f"artifact missing or empty: {artifact_path}"
        error_code = "missing_or_empty_artifact"
    return VizSmokeStageResult(stage=stage, status=status, artifact=artifact, error_code=error_code, error=error)


def _inspect_artifact(path: Path) -> VizSmokeArtifact:
    if not path.exists():
        return VizSmokeArtifact(path=path, exists=False)
    preview = ""
    if path.suffix in {".json", ".csv", ".md", ".txt"}:
        preview = path.read_text(encoding="utf-8", errors="replace")[:200]
    return VizSmokeArtifact(
        path=path,
        exists=True,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        preview=preview,
    )


def _render_markdown_report(result: VizSmokeSuiteResult) -> str:
    lines = [
        "# Viz Smoke Report",
        "",
        f"- success: `{result.success}`",
        f"- stages: `{', '.join(result.selected_stage_ids)}`",
        f"- failed_abms: `{', '.join(result.failed_abms) if result.failed_abms else 'none'}`",
        "",
    ]
    for abm in result.abms:
        lines.append(f"## {abm.abm}")
        lines.append(f"- status: `{abm.status}`")
        lines.append(f"- csv_path: `{abm.csv_path}`")
        lines.append(f"- metric_pattern: `{abm.metric_pattern}`")
        if abm.error_code:
            lines.append(f"- error_code: `{abm.error_code}`")
        if abm.error:
            lines.append(f"- error: `{abm.error}`")
        for stage_result in abm.stage_results:
            lines.append(f"- {stage_result.stage.stage_id}: `{stage_result.status}` -> `{stage_result.artifact.path}`")
            if stage_result.error_code:
                lines.append(f"  error_code: `{stage_result.error_code}`")
            if stage_result.error:
                lines.append(f"  error: `{stage_result.error}`")
        lines.append("")
    return "\n".join(lines)
