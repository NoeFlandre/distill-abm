"""NetLogo-to-plot smoke reporting for pre-LLM visualization validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from distill_abm.ingest.csv_ingest import load_simulation_csv
from distill_abm.ingest.netlogo_steps import NetLogoLinkProtocol
from distill_abm.ingest.netlogo_workflow import run_netlogo_experiment
from distill_abm.viz.plots import MetricPlotBundle, plot_metric_bundle_to_path

VizSmokeStatus = Literal["ok", "failed"]
VizSmokeErrorCode = Literal[
    "missing_or_empty_artifact",
    "simulation_failed",
    "plot_generation_failed",
    "csv_load_failed",
    "missing_viz_config",
]


class VizSmokeSpec(BaseModel):
    """Resolved ABM simulation-and-plot spec used by the smoke workflow."""

    abm: str
    model_path: Path
    experiment_name: str
    experiment_parameters: dict[str, bool | int | float | str]
    num_runs: int
    max_ticks: int
    interval: int
    fallback_mode: Literal["on_error", "always"] = "on_error"
    fallback_csv: Path | None = None
    fallback_plot_dir: Path | None = None
    reporters: list[str]
    plots: list[MetricPlotBundle]


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
    model_path: Path
    experiment_name: str
    parameters_path: Path
    generated_csv_path: Path
    output_dir: Path
    plot_dir: Path
    artifact_source: Literal["simulated", "fallback"]
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


def default_viz_smoke_stages(specs: dict[str, VizSmokeSpec]) -> list[VizSmokeStage]:
    """Return the canonical visualization checks for agent debugging."""
    stages: list[VizSmokeStage] = [
        VizSmokeStage(
            stage_id="simulation-csv",
            artifact_key="generated_csv",
            description="Generated simulation CSV directly from the NetLogo model.",
        )
    ]
    max_plots = max((len(spec.plots) for spec in specs.values()), default=0)
    for index in range(max_plots):
        stages.append(
            VizSmokeStage(
                stage_id=f"plot-{index + 1}",
                artifact_key=f"plot_{index + 1}",
                description=f"Generated plot {index + 1} destined for trend-description prompts.",
            )
        )
    return stages


def run_viz_smoke_suite(
    *,
    specs: dict[str, VizSmokeSpec],
    netlogo_home: str,
    output_root: Path,
    stage_ids: list[str] | None = None,
    netlogo_link_factory: Callable[..., NetLogoLinkProtocol] | None = None,
) -> VizSmokeSuiteResult:
    """Run NetLogo simulations and generate the ordered plot PNGs used before LLM inference."""
    started_at = datetime.now(UTC)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_stages = _select_stages(default_viz_smoke_stages(specs), stage_ids)

    abm_results: list[VizSmokeAbmResult] = []
    failed_abms: list[str] = []
    for abm, spec in sorted(specs.items()):
        output_dir = output_root / abm
        plot_dir = output_dir / "plots"
        parameters_path = output_dir / "resolved_parameters.json"
        generated_csv_path = output_dir / "simulation.csv"
        try:
            artifact_paths = _generate_viz_artifacts(
                spec=spec,
                netlogo_home=netlogo_home,
                output_dir=output_dir,
                plot_dir=plot_dir,
                parameters_path=parameters_path,
                generated_csv_path=generated_csv_path,
                netlogo_link_factory=netlogo_link_factory,
            )
            artifact_index_path = output_dir / "viz_artifact_index.json"
            artifact_index_path.write_text(
                json.dumps({key: str(path) for key, path in artifact_paths.items()}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            stage_results = [
                _build_stage_result(
                    stage=stage,
                    artifact_path=artifact_paths.get(stage.artifact_key),
                    missing_artifact_ok=stage.artifact_key.startswith("plot_")
                    and stage.artifact_key not in artifact_paths,
                )
                for stage in selected_stages
            ]
            status: VizSmokeStatus = "failed" if any(item.status == "failed" for item in stage_results) else "ok"
            if status == "failed":
                failed_abms.append(abm)
            abm_results.append(
                VizSmokeAbmResult(
                    abm=abm,
                    model_path=spec.model_path,
                    experiment_name=spec.experiment_name,
                    parameters_path=parameters_path,
                    generated_csv_path=generated_csv_path,
                    output_dir=output_dir,
                    plot_dir=plot_dir,
                    artifact_source=_read_artifact_source(artifact_paths),
                    status=status,
                    artifact_index_path=artifact_index_path,
                    stage_results=stage_results,
                )
            )
        except Exception as exc:
            failed_abms.append(abm)
            abm_results.append(
                VizSmokeAbmResult(
                    abm=abm,
                    model_path=spec.model_path,
                    experiment_name=spec.experiment_name,
                    parameters_path=parameters_path,
                    generated_csv_path=generated_csv_path,
                    output_dir=output_dir,
                    plot_dir=plot_dir,
                    artifact_source="simulated",
                    status="failed",
                    error_code="simulation_failed",
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


def _select_stages(stages: list[VizSmokeStage], stage_ids: list[str] | None) -> list[VizSmokeStage]:
    if not stage_ids:
        return stages
    by_id = {stage.stage_id: stage for stage in stages}
    unknown = [stage_id for stage_id in stage_ids if stage_id not in by_id]
    if unknown:
        known = ", ".join(sorted(by_id))
        raise ValueError(f"unknown viz smoke stage(s): {', '.join(unknown)}. Known stages: {known}")
    return [by_id[stage_id] for stage_id in stage_ids]


def _generate_viz_artifacts(
    *,
    spec: VizSmokeSpec,
    netlogo_home: str,
    output_dir: Path,
    plot_dir: Path,
    parameters_path: Path,
    generated_csv_path: Path,
    netlogo_link_factory: Callable[..., NetLogoLinkProtocol] | None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    parameters_path.write_text(json.dumps(spec.experiment_parameters, indent=2, sort_keys=True), encoding="utf-8")
    if spec.fallback_mode == "always":
        return _materialize_fallback_artifacts(
            spec=spec,
            output_dir=output_dir,
            plot_dir=plot_dir,
            generated_csv_path=generated_csv_path,
            parameters_path=parameters_path,
        )
    try:
        run_netlogo_experiment(
            netlogo_home=netlogo_home,
            model_path=spec.model_path,
            experiment_parameters=spec.experiment_parameters,
            reporters=spec.reporters,
            num_runs=spec.num_runs,
            max_ticks=spec.max_ticks,
            interval=spec.interval,
            output_csv_path=generated_csv_path,
            netlogo_link_factory=netlogo_link_factory,
        )
        frame = load_simulation_csv(generated_csv_path)

        artifact_paths: dict[str, Path] = {"resolved_parameters": parameters_path, "generated_csv": generated_csv_path}
        for index, bundle in enumerate(spec.plots, start=1):
            output_path = plot_dir / f"{index}.png"
            plot_metric_bundle_to_path(
                frame=frame,
                include_pattern=bundle.include_pattern,
                output_path=output_path,
                title=bundle.title,
                y_label=bundle.y_label,
                x_label=bundle.x_label,
                exclude_pattern=bundle.exclude_pattern,
                show_mean_line=bundle.show_mean_line,
            )
            artifact_paths[f"plot_{index}"] = output_path
        source_path = output_dir / "artifact_source.txt"
        source_path.write_text("simulated\n", encoding="utf-8")
        artifact_paths["artifact_source"] = source_path
        return artifact_paths
    except Exception:
        if spec.fallback_csv is None or spec.fallback_plot_dir is None:
            raise
        return _materialize_fallback_artifacts(
            spec=spec,
            output_dir=output_dir,
            plot_dir=plot_dir,
            generated_csv_path=generated_csv_path,
            parameters_path=parameters_path,
        )


def _materialize_fallback_artifacts(
    *,
    spec: VizSmokeSpec,
    output_dir: Path,
    plot_dir: Path,
    generated_csv_path: Path,
    parameters_path: Path,
) -> dict[str, Path]:
    if spec.fallback_csv is None or spec.fallback_plot_dir is None:
        raise ValueError("fallback artifacts are not configured")
    if not spec.fallback_csv.exists():
        raise FileNotFoundError(f"fallback CSV not found: {spec.fallback_csv}")
    if not spec.fallback_plot_dir.exists():
        raise FileNotFoundError(f"fallback plot directory not found: {spec.fallback_plot_dir}")

    generated_csv_path.write_bytes(spec.fallback_csv.read_bytes())
    artifact_paths: dict[str, Path] = {
        "resolved_parameters": parameters_path,
        "generated_csv": generated_csv_path,
    }
    for index, _bundle in enumerate(spec.plots, start=1):
        source_plot = spec.fallback_plot_dir / f"{index}.png"
        if not source_plot.exists():
            continue
        output_path = plot_dir / f"{index}.png"
        output_path.write_bytes(source_plot.read_bytes())
        artifact_paths[f"plot_{index}"] = output_path
    source_path = output_dir / "artifact_source.txt"
    source_path.write_text("fallback\n", encoding="utf-8")
    artifact_paths["artifact_source"] = source_path
    return artifact_paths


def _build_stage_result(
    *,
    stage: VizSmokeStage,
    artifact_path: Path | None,
    missing_artifact_ok: bool,
) -> VizSmokeStageResult:
    if artifact_path is None and missing_artifact_ok:
        artifact = VizSmokeArtifact(path=Path("<not-generated>"), exists=False)
        return VizSmokeStageResult(
            stage=stage,
            status="ok",
            artifact=artifact,
        )
    if artifact_path is None:
        artifact = VizSmokeArtifact(path=Path("<missing>"), exists=False)
        return VizSmokeStageResult(
            stage=stage,
            status="failed",
            artifact=artifact,
            error_code="missing_or_empty_artifact",
            error=f"artifact missing: {stage.artifact_key}",
        )
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
    size_bytes = path.stat().st_size
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    preview = ""
    if path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
        preview = path.read_text(encoding="utf-8")[:300]
    return VizSmokeArtifact(path=path, exists=True, size_bytes=size_bytes, sha256=sha256, preview=preview)


def _read_artifact_source(artifact_paths: dict[str, Path]) -> Literal["simulated", "fallback"]:
    source_path = artifact_paths.get("artifact_source")
    if source_path is None or not source_path.exists():
        return "simulated"
    source = source_path.read_text(encoding="utf-8").strip()
    return "fallback" if source == "fallback" else "simulated"


def _render_markdown_report(result: VizSmokeSuiteResult) -> str:
    lines = [
        "# Visualization Smoke Report",
        "",
        f"- started_at_utc: `{result.started_at_utc}`",
        f"- finished_at_utc: `{result.finished_at_utc}`",
        f"- output_root: `{result.output_root}`",
        f"- success: `{str(result.success).lower()}`",
        "",
        "## Summary",
        "",
        "| abm | experiment | status | csv | plots |",
        "| --- | --- | --- | --- | --- |",
    ]
    for abm_result in result.abms:
        plot_count = sum(1 for stage in abm_result.stage_results if stage.stage.stage_id.startswith("plot-"))
        lines.append(
            f"| {abm_result.abm} | {abm_result.experiment_name} | {abm_result.status} | "
            f"`{abm_result.generated_csv_path.name}` | {plot_count} |"
        )
    for abm_result in result.abms:
        lines.extend(
            [
                "",
                f"## {abm_result.abm}",
                "",
                f"- model_path: `{abm_result.model_path}`",
                f"- experiment_name: `{abm_result.experiment_name}`",
                f"- parameters_path: `{abm_result.parameters_path}`",
                f"- generated_csv_path: `{abm_result.generated_csv_path}`",
                f"- plot_dir: `{abm_result.plot_dir}`",
                f"- artifact_source: `{abm_result.artifact_source}`",
                f"- status: `{abm_result.status}`",
            ]
        )
        if abm_result.error is not None:
            lines.append(f"- error: `{abm_result.error}`")
        lines.extend(
            [
                "",
                "| stage | status | artifact | size_bytes | error |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for stage_result in abm_result.stage_results:
            lines.append(
                f"| {stage_result.stage.stage_id} | {stage_result.status} | "
                f"`{stage_result.artifact.path}` | {stage_result.artifact.size_bytes} | "
                f"{stage_result.error or ''} |"
            )
    return "\n".join(lines) + "\n"
