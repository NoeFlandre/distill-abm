"""Typer CLI entrypoint for distill-abm paper-aligned workflows."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from shutil import copy2
from typing import Annotated, Literal, cast

import typer

from distill_abm.agent_validation import ValidationProfile, run_validation_suite
from distill_abm.cli_models import (
    ArtifactDescriptor,
    DescribeAbmResult,
    DescribeArtifactsResult,
    DescribeRunResult,
    DoeCommandResult,
    IngestCommandResult,
    IngestSuiteCommandResult,
    RunCommandResult,
    SmokeCommandResult,
    build_artifact_descriptors,
    describe_artifact,
)
from distill_abm.configs.loader import (
    load_abm_config,
    load_experiment_settings,
    load_models_config,
    load_prompts_config,
)
from distill_abm.configs.models import ModelEntry, SummarizerId
from distill_abm.configs.runtime_defaults import get_runtime_defaults
from distill_abm.eval.doe_full import analyze_factorial_anova
from distill_abm.eval.qualitative_runner import QualitativeMetric, evaluate_qualitative_score
from distill_abm.ingest.ingest_smoke import run_ingest_smoke_suite
from distill_abm.ingest.netlogo_workflow import resolve_experiment_parameters, run_ingest_workflow
from distill_abm.llm.factory import create_adapter
from distill_abm.pipeline.run import EvidenceMode, PipelineInputs, TextSourceMode, run_pipeline
from distill_abm.pipeline.smoke import (
    SmokeCase,
    SmokeSuiteInputs,
    default_branch_smoke_cases,
    default_smoke_cases,
    run_qwen_smoke_suite,
)
from distill_abm.viz.plots import MetricPlotBundle
from distill_abm.viz.viz_smoke import VizSmokeSpec, run_viz_smoke_suite

app = typer.Typer(help="Run ABM distillation workflows.")
RUNTIME_DEFAULTS = get_runtime_defaults()
DEFAULT_EVIDENCE_MODE: EvidenceMode = RUNTIME_DEFAULTS.run.evidence_mode
DEFAULT_TEXT_SOURCE_MODE: TextSourceMode = RUNTIME_DEFAULTS.run.text_source_mode
DEFAULT_SUMMARIZERS: tuple[SummarizerId, ...] = RUNTIME_DEFAULTS.run.summarizers

BENCHMARK_MODELS: set[tuple[str, str]] = {
    ("openrouter", "moonshotai/kimi-k2.5"),
    ("openrouter", "google/gemini-3.1-pro-preview"),
    ("ollama", "qwen3.5:0.8b"),
}
DEBUG_MODEL: tuple[str, str] = ("openrouter", "qwen/qwen3-vl-235b-a22b-thinking")

__all__ = [
    "analyze_doe",
    "app",
    "evaluate_qualitative",
    "ingest_netlogo",
    "ingest_netlogo_suite",
    "main",
    "run",
    "smoke_ingest_netlogo",
    "smoke_qwen",
    "smoke_viz",
    "subprocess",
    "validate_workspace",
]


@app.callback()
def cli() -> None:
    """Keeps Typer in command-group mode so `run` is explicit."""


@app.command()
def run(
    csv_path: Annotated[
        Path,
        typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    ],
    parameters_path: Annotated[
        Path,
        typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    ],
    documentation_path: Annotated[
        Path,
        typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    ],
    prompts_path: Annotated[
        Path,
        typer.Option(exists=True),
    ] = Path("configs/prompts.yaml"),
    models_path: Annotated[
        Path,
        typer.Option(exists=True, help="Model registry YAML path."),
    ] = Path("configs/models.yaml"),
    output_dir: Annotated[Path, typer.Option()] = Path(RUNTIME_DEFAULTS.run.output_dir),
    provider: Annotated[str, typer.Option()] = RUNTIME_DEFAULTS.run.provider,
    model: Annotated[str, typer.Option()] = RUNTIME_DEFAULTS.run.model,
    model_id: Annotated[
        str | None,
        typer.Option(help="Optional model alias from configs/models.yaml (recommended)."),
    ] = None,
    allow_debug_model: Annotated[
        bool,
        typer.Option(help="Allow debug model in this run (disabled by default for benchmark integrity)."),
    ] = False,
    metric_pattern: Annotated[str, typer.Option()] = RUNTIME_DEFAULTS.run.metric_pattern,
    metric_description: Annotated[str, typer.Option()] = RUNTIME_DEFAULTS.run.metric_description,
    plot_description: Annotated[
        str | None,
        typer.Option(help="Optional evidence description for the plotted metric."),
    ] = None,
    evidence_mode: Annotated[
        EvidenceMode,
        typer.Option(help="Evidence ablation mode: plot, table, or plot+table."),
    ] = DEFAULT_EVIDENCE_MODE,
    text_source_mode: Annotated[
        TextSourceMode,
        typer.Option(help="Text source mode: summary_only or full_text_only."),
    ] = DEFAULT_TEXT_SOURCE_MODE,
    summarizer: Annotated[
        list[str] | None,
        typer.Option(
            "--summarizer",
            help="Summary backend roster. Repeatable: bart, bert, t5, longformer_ext.",
        ),
    ] = None,
    allow_summary_fallback: Annotated[
        bool,
        typer.Option(help="Allow summary-only mode to fall back to full-text when summarizers fail."),
    ] = False,
    abm: Annotated[str | None, typer.Option(help="ABM config name in configs/abms/<name>.yaml")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Run one end-to-end pipeline execution from CSV to scored report."""
    prompts = load_prompts_config(prompts_path)
    if model_id is not None:
        provider, model = _resolve_model_from_registry(models_path=models_path, model_id=model_id)
    _validate_model_policy(provider=provider, model=model, allow_debug_model=allow_debug_model)

    scoring_reference_path: Path | None = None
    if abm:
        abm_config = load_abm_config(Path("configs/abms") / f"{abm}.yaml")
        metric_pattern = abm_config.metric_pattern
        metric_description = abm_config.metric_description
        if plot_description is None and abm_config.plot_descriptions:
            plot_description = abm_config.plot_descriptions[0]
        scoring_reference_path = _resolve_scoring_reference_path(abm)

    adapter = create_adapter(provider=provider, model=model)
    result = run_pipeline(
        inputs=PipelineInputs(
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            output_dir=output_dir,
            model=model,
            metric_pattern=metric_pattern,
            metric_description=metric_description,
            plot_description=plot_description,
            evidence_mode=evidence_mode,
            text_source_mode=text_source_mode,
            allow_summary_fallback=allow_summary_fallback,
            summarizers=_parse_summarizers(summarizer, fallback=DEFAULT_SUMMARIZERS),
            scoring_reference_path=scoring_reference_path,
        ),
        prompts=prompts,
        adapter=adapter,
    )
    artifact_manifest_path = output_dir / "run_artifact_manifest.json"
    command_result = RunCommandResult(
        output_dir=output_dir,
        plot_path=result.plot_path,
        report_csv_path=result.report_csv,
        metadata_path=getattr(result, "metadata_path", None),
        artifact_manifest_path=artifact_manifest_path,
        artifacts=build_artifact_descriptors(
            {
                "plot": result.plot_path,
                "report_csv": result.report_csv,
                "metadata": getattr(result, "metadata_path", None),
                "stats_image": getattr(result, "stats_image_path", None),
            }
        ),
    )
    artifact_manifest_path.write_text(command_result.model_dump_json(indent=2), encoding="utf-8")
    if json_output:
        typer.echo(command_result.model_dump_json(indent=2))
        return
    typer.echo(f"plot: {result.plot_path}")
    typer.echo(f"report: {result.report_csv}")
    typer.echo(f"artifact manifest: {artifact_manifest_path}")


@app.command("ingest-netlogo")
def ingest_netlogo(
    model_path: Annotated[
        Path,
        typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    ],
    experiment_parameters_path: Annotated[
        Path | None,
        typer.Option(help="Optional JSON file with experiment parameters."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Directory for artifacts. Defaults to results/ingest/<model-stem>."),
    ] = None,
    suffix: Annotated[str, typer.Option(help="Suffix used in workflow artifact names.")] = "",
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Run NetLogo preprocessing workflow and persist extracted artifacts."""
    resolved_output_dir = output_dir if output_dir is not None else Path("results") / "ingest" / model_path.stem
    experiment_parameters = _load_experiment_parameters(experiment_parameters_path)
    artifacts = run_ingest_workflow(
        model_path=model_path,
        experiment_parameters=experiment_parameters,
        output_dir=resolved_output_dir,
        suffix=suffix,
    )
    artifact_manifest_path = resolved_output_dir / "ingest_manifest.json"
    command_result = IngestCommandResult(
        output_dir=resolved_output_dir,
        artifact_manifest_path=artifact_manifest_path,
        artifacts=build_artifact_descriptors(artifacts),
    )
    artifact_manifest_path.write_text(command_result.model_dump_json(indent=2), encoding="utf-8")
    if json_output:
        typer.echo(command_result.model_dump_json(indent=2))
        return
    typer.echo(f"NetLogo ingestion artifacts written to: {resolved_output_dir.resolve()}")
    typer.echo(f"artifact manifest: {artifact_manifest_path}")
    for key, path in sorted(artifacts.items()):
        typer.echo(f"{key}: {path}")


@app.command("ingest-netlogo-suite")
def ingest_netlogo_suite(
    abms: Annotated[
        list[str] | None,
        typer.Option(
            "--abm",
            help="ABM names to process. Repeat for multiple. Defaults to all configured ABMs.",
        ),
    ] = None,
    models_root: Annotated[
        Path,
        typer.Option(
            help="Root directory containing ABM models. Supports per-ABM folders and root-level model files.",
        ),
    ] = Path("data"),
    output_root: Annotated[
        Path,
        typer.Option(help="Root directory for generated artifacts."),
    ] = Path("results/ingest"),
    suffix: Annotated[str, typer.Option(help="Suffix used in workflow artifact names.")] = "",
    continue_on_missing: Annotated[
        bool,
        typer.Option(help="Continue processing remaining ABMs if one ABM cannot be ingested."),
    ] = False,
    default_experiment_parameters_path: Annotated[
        Path | None,
        typer.Option(help="Optional shared experiment-parameters JSON path for all ABMs."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Run NetLogo ingestion for multiple ABMs into dedicated output folders."""
    requested = sorted(set(abms)) if abms else _discover_configured_abms()
    missing: list[str] = []
    shared_params = _load_experiment_parameters(default_experiment_parameters_path)
    suite_artifacts: dict[str, dict[str, ArtifactDescriptor]] = {}

    for abm in requested:
        try:
            model_path = _resolve_abm_model_path(abm=abm, models_root=models_root)
            parameter_path = _resolve_abm_experiment_parameters_path(
                model_dir=model_path.parent,
                abm=abm,
                explicit=default_experiment_parameters_path,
            )
            if default_experiment_parameters_path is not None:
                experiment_parameters = shared_params
            else:
                experiment_parameters = _load_experiment_parameters(parameter_path)
            output_dir = output_root / abm
            artifacts = run_ingest_workflow(
                model_path=model_path,
                experiment_parameters=experiment_parameters,
                output_dir=output_dir,
                suffix=suffix,
            )
            manifest_path = output_dir / "ingest_manifest.json"
            command_result = IngestCommandResult(
                output_dir=output_dir,
                artifact_manifest_path=manifest_path,
                artifacts=build_artifact_descriptors(artifacts),
            )
            manifest_path.write_text(command_result.model_dump_json(indent=2), encoding="utf-8")
            suite_artifacts[abm] = command_result.artifacts
            typer.echo(f"[{abm}] NetLogo ingestion artifacts written to: {output_dir.resolve()}")
            for key, path in sorted(artifacts.items()):
                typer.echo(f"{abm}::{key}: {path}")
        except typer.BadParameter as exc:
            message = f"failed for {abm}: {exc}"
            if continue_on_missing:
                missing.append(message)
                continue
            raise typer.BadParameter(message) from exc

    suite_manifest_path = output_root / "ingest_suite_manifest.json"
    suite_result = IngestSuiteCommandResult(
        output_root=output_root,
        artifact_manifest_path=suite_manifest_path,
        abms=suite_artifacts,
        skipped_abms=missing,
    )
    suite_manifest_path.write_text(suite_result.model_dump_json(indent=2), encoding="utf-8")
    if json_output:
        typer.echo(suite_result.model_dump_json(indent=2))
        return
    typer.echo(f"suite artifact manifest: {suite_manifest_path}")
    if missing:
        typer.echo("ingest completed with skipped ABMs:")
        for issue in missing:
            typer.echo(f" - {issue}")


@app.command("analyze-doe")
def analyze_doe(
    input_csv: Annotated[Path, typer.Option(..., exists=True, file_okay=True, dir_okay=False)],
    output_csv: Annotated[Path, typer.Option()] = Path(RUNTIME_DEFAULTS.doe.output_csv),
    max_interaction_order: Annotated[int, typer.Option()] = RUNTIME_DEFAULTS.doe.max_interaction_order,
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Run full factorial ANOVA contribution analysis."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result = analyze_factorial_anova(input_csv, output_csv, max_interaction_order=max_interaction_order)
    if result is None:
        raise typer.Exit(code=1)
    command_result = DoeCommandResult(success=True, output_csv=output_csv)
    if json_output:
        typer.echo(command_result.model_dump_json(indent=2))
        return
    typer.echo(f"wrote: {output_csv}")


@app.command("evaluate-qualitative")
def evaluate_qualitative(
    summary_text: Annotated[str, typer.Option(help="Generated summary text to evaluate.")],
    source_text: Annotated[str, typer.Option(help="Source context text used as qualitative reference.")],
    metric: Annotated[QualitativeMetric, typer.Option(help="Which qualitative metric to score.")],
    source_image_path: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Optional plot or evidence image for vision-capable models.",
        ),
    ] = None,
    prompts_path: Annotated[
        Path,
        typer.Option(exists=True),
    ] = Path("configs/prompts.yaml"),
    provider: Annotated[str, typer.Option()] = RUNTIME_DEFAULTS.qualitative.provider,
    model: Annotated[str, typer.Option()] = RUNTIME_DEFAULTS.qualitative.model,
    allow_debug_model: Annotated[
        bool,
        typer.Option(help="Allow debug model in qualitative scoring command."),
    ] = True,
) -> None:
    """Evaluate coverage or faithfulness with an LLM and return JSON output."""
    _validate_model_policy(provider=provider, model=model, allow_debug_model=allow_debug_model)
    prompts = load_prompts_config(prompts_path)
    adapter = create_adapter(provider=provider, model=model)
    result = evaluate_qualitative_score(
        summary=summary_text,
        source=source_text,
        metric=metric,
        model=model,
        prompts=prompts,
        adapter=adapter,
        source_image_path=source_image_path,
    )
    typer.echo(result.model_dump_json())


@app.command("smoke-qwen")
def smoke_qwen(
    csv_path: Annotated[
        Path,
        typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    ],
    parameters_path: Annotated[
        Path,
        typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    ],
    documentation_path: Annotated[
        Path,
        typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    ],
    doe_input_csv: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=True, dir_okay=False),
    ] = None,
    prompts_path: Annotated[
        Path,
        typer.Option(exists=True),
    ] = Path("configs/prompts.yaml"),
    output_dir: Annotated[Path, typer.Option()] = Path(RUNTIME_DEFAULTS.smoke.output_dir),
    provider: Annotated[str, typer.Option()] = RUNTIME_DEFAULTS.smoke.provider,
    model: Annotated[str, typer.Option()] = RUNTIME_DEFAULTS.smoke.model,
    allow_debug_model: Annotated[
        bool,
        typer.Option(help="Allow debug model in smoke runs."),
    ] = True,
    metric_pattern: Annotated[str, typer.Option()] = RUNTIME_DEFAULTS.smoke.metric_pattern,
    metric_description: Annotated[str, typer.Option()] = RUNTIME_DEFAULTS.smoke.metric_description,
    plot_description: Annotated[str | None, typer.Option()] = None,
    evidence_mode: Annotated[EvidenceMode, typer.Option()] = RUNTIME_DEFAULTS.smoke.evidence_mode,
    text_source_mode: Annotated[TextSourceMode, typer.Option()] = RUNTIME_DEFAULTS.smoke.text_source_mode,
    allow_summary_fallback: Annotated[
        bool,
        typer.Option(help="Allow smoke summary-only mode to fall back to full-text when summarizers fail."),
    ] = False,
    abm: Annotated[str | None, typer.Option(help="ABM config name in configs/abms/<name>.yaml")] = None,
    summarizer: Annotated[
        list[str] | None,
        typer.Option(
            "--summarizer",
            help="Summary backend roster. Repeatable: bart, bert, t5, longformer_ext.",
        ),
    ] = None,
    skip_qualitative: Annotated[
        bool,
        typer.Option(help="Skip qualitative coverage/faithfulness checks in the smoke suite."),
    ] = not RUNTIME_DEFAULTS.smoke.run_qualitative,
    skip_sweep: Annotated[
        bool,
        typer.Option(help="Skip prompt-combination sweep execution in the smoke suite."),
    ] = not RUNTIME_DEFAULTS.smoke.run_sweep,
    profile: Annotated[
        Literal["matrix", "three-branches"],
        typer.Option(help="Smoke profile: full matrix or compact three-branch debug profile."),
    ] = "matrix",
    case_id: Annotated[
        list[str] | None,
        typer.Option(
            "--case-id",
            help="Optional smoke case id filter. Repeat this option to run a subset of the matrix.",
        ),
    ] = None,
    max_cases: Annotated[
        int | None,
        typer.Option(min=1, help="Optional cap on number of smoke cases after filtering."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Reuse existing smoke artifacts and skip already completed work."),
    ] = True,
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Run full debug smoke validation across evidence/text modes plus DoE and sweep artifacts."""
    _validate_model_policy(provider=provider, model=model, allow_debug_model=allow_debug_model)

    prompts = load_prompts_config(prompts_path)
    sweep_plot_descriptions: list[str] | None = None
    scoring_reference_path: Path | None = None
    if abm:
        abm_config = load_abm_config(Path("configs/abms") / f"{abm}.yaml")
        metric_pattern = abm_config.metric_pattern
        metric_description = abm_config.metric_description
        if plot_description is None and abm_config.plot_descriptions:
            plot_description = abm_config.plot_descriptions[0]
        sweep_plot_descriptions = list(abm_config.plot_descriptions)
        scoring_reference_path = _resolve_scoring_reference_path(abm)
    selected_cases = _select_smoke_cases(case_ids=case_id, max_cases=max_cases, profile=profile)
    adapter = create_adapter(provider=provider, model=model)
    result = run_qwen_smoke_suite(
        inputs=SmokeSuiteInputs(
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            output_dir=output_dir,
            model=model,
            metric_pattern=metric_pattern,
            metric_description=metric_description,
            plot_description=plot_description,
            sweep_plot_descriptions=sweep_plot_descriptions,
            allow_summary_fallback=allow_summary_fallback,
            summarizers=_parse_summarizers(summarizer, fallback=RUNTIME_DEFAULTS.smoke.summarizers),
            text_source_mode=text_source_mode,
            evidence_mode=evidence_mode,
            scoring_reference_path=scoring_reference_path,
        ),
        prompts=prompts,
        adapter=adapter,
        run_qualitative=not skip_qualitative,
        doe_input_csv=doe_input_csv,
        run_sweep=not skip_sweep,
        cases=selected_cases,
        resume_existing=resume,
    )
    command_result = SmokeCommandResult(
        command="smoke-qwen",
        success=result.success,
        report_json_path=result.report_json_path,
        report_markdown_path=result.report_markdown_path,
        failed_items=result.failed_cases,
        nested_artifacts={
            key: value
            for key, value in {
                "doe_output_csv": result.doe_output_csv,
                "sweep_output_csv": result.sweep_output_csv,
                "run_master_csv": getattr(result, "run_master_csv_path", None),
                "global_master_csv": getattr(result, "global_master_csv_path", None),
            }.items()
            if value is not None
        },
    )
    if json_output:
        typer.echo(command_result.model_dump_json(indent=2))
        if not result.success:
            raise typer.Exit(code=1)
        return
    typer.echo(f"smoke report (markdown): {result.report_markdown_path}")
    typer.echo(f"smoke report (json): {result.report_json_path}")
    if result.doe_output_csv is not None:
        typer.echo(f"doe output: {result.doe_output_csv}")
    if result.sweep_output_csv is not None:
        typer.echo(f"sweep output: {result.sweep_output_csv}")
    if not result.success:
        failed = ", ".join(result.failed_cases) if result.failed_cases else "doe/sweep"
        typer.echo(f"smoke suite failed: {failed}")
        raise typer.Exit(code=1)


@app.command("smoke-ingest-netlogo")
def smoke_ingest_netlogo(
    abms: Annotated[
        list[str] | None,
        typer.Option(
            "--abm",
            help="ABM names to process. Repeat for multiple. Defaults to all configured ABMs.",
        ),
    ] = None,
    models_root: Annotated[
        Path,
        typer.Option(
            help="Root directory containing ABM models. Supports per-ABM folders and root-level model files.",
        ),
    ] = Path("data"),
    output_root: Annotated[
        Path,
        typer.Option(help="Root directory for ingestion smoke artifacts."),
    ] = Path("results/ingest_smoke"),
    stage: Annotated[
        list[str] | None,
        typer.Option("--stage", help="Optional ingest stage filter. Repeat to focus the smoke report."),
    ] = None,
    require_stage: Annotated[
        list[str] | None,
        typer.Option("--require-stage", help="Assert that these stages were selected in the report."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Run artifact-focused smoke checks for NetLogo ingestion across configured ABMs."""
    requested = sorted(set(abms)) if abms else list(_discover_configured_abms())
    abm_models = {abm: _resolve_abm_model_path(abm=abm, models_root=models_root) for abm in requested}
    try:
        result = run_ingest_smoke_suite(abm_models=abm_models, output_root=output_root, stage_ids=stage)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if require_stage:
        missing_required = [item for item in require_stage if item not in result.selected_stage_ids]
        if missing_required:
            raise typer.BadParameter(
                f"required stage(s) missing from ingest smoke selection: {', '.join(missing_required)}"
            )
    command_result = SmokeCommandResult(
        command="smoke-ingest-netlogo",
        success=result.success,
        report_json_path=result.report_json_path,
        report_markdown_path=result.report_markdown_path,
        failed_items=result.failed_abms,
    )
    if json_output:
        typer.echo(command_result.model_dump_json(indent=2))
        if not result.success:
            raise typer.Exit(code=1)
        return
    typer.echo(f"ingest smoke report (markdown): {result.report_markdown_path}")
    typer.echo(f"ingest smoke report (json): {result.report_json_path}")
    if not result.success:
        typer.echo(f"ingest smoke failed: {', '.join(result.failed_abms)}")
        raise typer.Exit(code=1)


@app.command("smoke-viz")
def smoke_viz(
    abms: Annotated[
        list[str] | None,
        typer.Option(
            "--abm",
            help="ABM names to process. Repeat for multiple. Defaults to all configured ABMs.",
        ),
    ] = None,
    models_root: Annotated[
        Path,
        typer.Option(
            help="Root directory containing ABM models. Supports per-ABM folders and root-level model files.",
        ),
    ] = Path("data"),
    netlogo_home: Annotated[
        str,
        typer.Option(
            envvar="DISTILL_ABM_NETLOGO_HOME",
            help="NetLogo installation directory used by pynetlogo. Can also be provided via DISTILL_ABM_NETLOGO_HOME.",
        ),
    ] = "",
    stage: Annotated[
        list[str] | None,
        typer.Option("--stage", help="Optional viz smoke stage filter. Repeat to focus the smoke report."),
    ] = None,
    require_stage: Annotated[
        list[str] | None,
        typer.Option("--require-stage", help="Assert that these stages were selected in the report."),
    ] = None,
    output_root: Annotated[
        Path,
        typer.Option(help="Root directory for visualization smoke artifacts."),
    ] = Path("results/viz_smoke_latest"),
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Run NetLogo simulations and generate the ordered plot PNGs used before LLM inference."""
    requested = sorted(set(abms)) if abms else list(_discover_configured_abms())
    try:
        if not netlogo_home.strip():
            raise ValueError(
                "missing NetLogo installation directory. Provide --netlogo-home or set DISTILL_ABM_NETLOGO_HOME."
            )
        specs = _resolve_viz_smoke_specs(requested_abms=requested, models_root=models_root)
        result = run_viz_smoke_suite(
            specs=specs,
            netlogo_home=netlogo_home,
            output_root=output_root,
            stage_ids=stage,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if require_stage:
        missing_required = [item for item in require_stage if item not in result.selected_stage_ids]
        if missing_required:
            raise typer.BadParameter(
                f"required stage(s) missing from viz smoke selection: {', '.join(missing_required)}"
            )
    command_result = SmokeCommandResult(
        command="smoke-viz",
        success=result.success,
        report_json_path=result.report_json_path,
        report_markdown_path=result.report_markdown_path,
        failed_items=result.failed_abms,
    )
    if json_output:
        typer.echo(command_result.model_dump_json(indent=2))
        if not result.success:
            raise typer.Exit(code=1)
        return
    typer.echo(f"viz smoke report (markdown): {result.report_markdown_path}")
    typer.echo(f"viz smoke report (json): {result.report_json_path}")
    if not result.success:
        typer.echo(f"viz smoke failed: {', '.join(result.failed_abms)}")
        raise typer.Exit(code=1)


@app.command("validate-workspace")
def validate_workspace(
    checks: Annotated[
        list[str] | None,
        typer.Option(
            "--check",
            help="Optional validation check filter. Repeatable: pytest, ruff, mypy, build, smoke-ingest-netlogo.",
        ),
    ] = None,
    abms: Annotated[
        list[str] | None,
        typer.Option(
            "--abm",
            help="ABM names to process for the smoke-ingest-netlogo validation step. Defaults to all configured ABMs.",
        ),
    ] = None,
    models_root: Annotated[
        Path,
        typer.Option(
            help="Root directory containing ABM models for the smoke-ingest-netlogo validation step.",
        ),
    ] = Path("data"),
    ingest_stage: Annotated[
        list[str] | None,
        typer.Option("--ingest-stage", help="Optional ingest stage filter for the smoke-ingest-netlogo check."),
    ] = None,
    profile: Annotated[
        ValidationProfile,
        typer.Option(help="Validation profile: quick, default, or full."),
    ] = "default",
    output_root: Annotated[
        Path,
        typer.Option(help="Directory for structured validation reports and nested artifacts."),
    ] = Path("results/agent_validation/latest"),
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the full validation report JSON to stdout."),
    ] = False,
) -> None:
    """Run the canonical non-LLM validation suite for coding-agent verification."""
    requested = sorted(set(abms)) if abms else list(_discover_configured_abms())
    abm_models = {abm: _resolve_abm_model_path(abm=abm, models_root=models_root) for abm in requested}
    try:
        result = run_validation_suite(
            output_root=output_root,
            abm_models=abm_models,
            checks=checks,
            ingest_stage_ids=ingest_stage,
            profile=profile,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(f"validation report (markdown): {result.report_markdown_path}")
        typer.echo(f"validation report (json): {result.report_json_path}")
        if result.ingest_smoke_report_json_path is not None:
            typer.echo(f"ingest smoke report (json): {result.ingest_smoke_report_json_path}")
        if result.ingest_smoke_report_markdown_path is not None:
            typer.echo(f"ingest smoke report (markdown): {result.ingest_smoke_report_markdown_path}")
    if not result.success:
        typer.echo(f"validation failed: {', '.join(result.failed_checks)}")
        raise typer.Exit(code=1)


@app.command("describe-abm")
def describe_abm(
    abm: Annotated[str, typer.Option(help="ABM config name in configs/abms/<name>.yaml")],
    models_root: Annotated[
        Path,
        typer.Option(help="Root directory containing ABM models."),
    ] = Path("data"),
    json_output: Annotated[bool, typer.Option("--json", help="Print structured JSON to stdout.")] = False,
) -> None:
    """Describe the resolved configuration and local assets for one ABM without running the pipeline."""
    config_path = Path("configs/abms") / f"{abm}.yaml"
    abm_config = load_abm_config(config_path)
    model_path = _resolve_abm_model_path(abm=abm, models_root=models_root)
    experiment_parameters_path = _resolve_abm_experiment_parameters_path(
        model_dir=model_path.parent,
        abm=abm,
        explicit=None,
    )
    result = DescribeAbmResult(
        abm=abm,
        config_path=config_path,
        model_path=model_path,
        experiment_parameters_path=experiment_parameters_path,
        scoring_reference_path=(
            _resolve_scoring_reference_path(abm) if abm in {"fauna", "grazing", "milk_consumption"} else None
        ),
        metric_pattern=abm_config.metric_pattern,
        metric_description=abm_config.metric_description,
        plot_descriptions=list(abm_config.plot_descriptions),
    )
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"abm: {result.abm}")
    typer.echo(f"config: {result.config_path}")
    typer.echo(f"model: {result.model_path}")
    if result.experiment_parameters_path is not None:
        typer.echo(f"experiment parameters: {result.experiment_parameters_path}")
    if result.scoring_reference_path is not None:
        typer.echo(f"scoring reference: {result.scoring_reference_path}")


@app.command("describe-ingest-artifacts")
def describe_ingest_artifacts(
    root: Annotated[Path, typer.Option(..., exists=True, file_okay=False, dir_okay=True)],
    json_output: Annotated[bool, typer.Option("--json", help="Print structured JSON to stdout.")] = False,
) -> None:
    """Describe an existing ingest artifact directory without rerunning ingestion."""
    manifest_path = root / "ingest_manifest.json"
    artifact_index_path = root / "ingest_artifact_index.json"
    artifacts: dict[str, ArtifactDescriptor] = {}
    source_path: Path | None = None
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in payload.get("artifacts", {}).items():
            artifacts[key] = ArtifactDescriptor.model_validate(value)
        source_path = manifest_path
    elif artifact_index_path.exists():
        payload = json.loads(artifact_index_path.read_text(encoding="utf-8"))
        artifacts = build_artifact_descriptors({key: Path(value) for key, value in payload.items()})
        source_path = artifact_index_path
    else:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                artifacts[str(path.relative_to(root))] = describe_artifact(path)
    result = DescribeArtifactsResult(
        root=root,
        manifest_path=manifest_path if manifest_path.exists() else None,
        artifact_index_path=artifact_index_path if artifact_index_path.exists() else None,
        artifacts=artifacts,
    )
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"root: {root}")
    if source_path is not None:
        typer.echo(f"source: {source_path}")
    for key, artifact in sorted(result.artifacts.items()):
        typer.echo(f"{key}: {artifact.path}")


@app.command("describe-run")
def describe_run(
    output_dir: Annotated[Path, typer.Option(..., exists=True, file_okay=False, dir_okay=True)],
    json_output: Annotated[bool, typer.Option("--json", help="Print structured JSON to stdout.")] = False,
) -> None:
    """Describe an existing run output directory from its metadata without rerunning the pipeline."""
    metadata_path = output_dir / "pipeline_run_metadata.json"
    if not metadata_path.exists():
        raise typer.BadParameter(f"missing pipeline metadata file: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifacts_payload = _as_dict(payload.get("artifacts"))
    reproducibility = _as_dict(payload.get("reproducibility"))
    execution = _as_dict(payload.get("execution"))
    debug_trace = _as_dict(payload.get("debug_trace"))
    frame_summary = _as_dict(debug_trace.get("frame_summary"))
    matched_metric_columns_raw = frame_summary.get("matched_metric_columns", [])
    available_artifacts = {
        key: Path(value)
        for key, value in artifacts_payload.items()
        if isinstance(value, str) and value
    }
    result = DescribeRunResult(
        output_dir=output_dir,
        metadata_path=metadata_path,
        available_artifacts=available_artifacts,
        run_signature=str(reproducibility.get("run_signature")) if reproducibility.get("run_signature") else None,
        selected_text_source=(
            str(execution.get("selected_text_source")) if execution.get("selected_text_source") else None
        ),
        evidence_mode=str(execution.get("evidence_mode")) if execution.get("evidence_mode") else None,
        requested_evidence_mode=(
            str(execution.get("requested_evidence_mode")) if execution.get("requested_evidence_mode") else None
        ),
        matched_metric_columns=(
            [str(item) for item in matched_metric_columns_raw if isinstance(item, str)]
            if isinstance(matched_metric_columns_raw, list)
            else []
        ),
    )
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"output_dir: {result.output_dir}")
    typer.echo(f"metadata: {result.metadata_path}")
    if result.run_signature:
        typer.echo(f"run_signature: {result.run_signature}")
    if result.selected_text_source:
        typer.echo(f"selected_text_source: {result.selected_text_source}")
    if result.evidence_mode:
        typer.echo(f"evidence_mode: {result.evidence_mode}")


def main() -> None:
    """Entrypoint callable used by setuptools/uv script wiring."""
    app()


def _load_experiment_parameters(path: Path | None) -> dict[str, bool | int | float | str]:
    """Load experiment parameter overrides from an optional JSON file."""
    if path is None:
        return {}

    if not path.exists():
        raise typer.BadParameter(f"experiment parameters file not found: {path}")
    payload = path.read_text(encoding="utf-8")
    if not payload.strip():
        return {}

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"experiment parameters file must contain JSON object: {path}") from exc

    if not isinstance(parsed, dict):
        raise typer.BadParameter("experiment parameters file must contain a JSON object at top level.")

    sanitized: dict[str, bool | int | float | str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            raise typer.BadParameter("experiment parameters keys must be strings.")
        if not isinstance(value, (bool, int, float, str)):
            raise typer.BadParameter(
                f"unsupported value type for experiment parameter '{key}': {type(value).__name__}. "
                "Allowed types are bool, int, float, and str."
            )
        sanitized[key] = value

    return sanitized


def _as_dict(value: object) -> dict[str, object]:
    """Normalize optional JSON payload fragments to dictionaries."""
    return value if isinstance(value, dict) else {}


def _discover_configured_abms() -> tuple[str, ...]:
    """Return ABM identifiers configured in configs/abms."""
    return tuple(sorted(p.stem for p in Path("configs/abms").glob("*.yaml")))


def _resolve_model_filenames(abm: str) -> tuple[str, ...]:
    """Return preferred NetLogo model filenames for an ABM."""
    candidates = [f"{abm}.nlogo", f"{abm}_model.nlogo"]
    if abm == "milk_consumption":
        candidates.append("model.nlogo")
    return tuple(dict.fromkeys(candidates))


def _resolve_abm_model_path(*, abm: str, models_root: Path) -> Path:
    """Find a single NetLogo model for an ABM and fail with a clear message if absent/ambiguous."""
    candidate_roots = [models_root, models_root / f"{abm}_abm", models_root / abm]
    model_filenames = _resolve_model_filenames(abm)
    matches: list[Path] = []
    for directory in candidate_roots:
        if not directory.exists():
            continue
        if directory.is_file():
            if directory.name in model_filenames:
                matches.append(directory)
            continue
        for model_name in model_filenames:
            candidate = directory / model_name
            if candidate.exists():
                matches.append(candidate)

    if not matches:
        candidates_desc = ", ".join(str(directory / name) for directory in candidate_roots for name in model_filenames)
        raise typer.BadParameter(
            f"no .nlogo file found for ABM '{abm}' in {models_root}. "
            f"Searched: {candidates_desc}"
        )
    matches = list(
        dict.fromkeys(
            _promote_root_model(models_root=models_root, abm=abm, model_path=match)
            for match in matches
        )
    )

    if len(matches) > 1:
        names = ", ".join(str(match.relative_to(models_root)) for match in matches)
        raise typer.BadParameter(f"multiple .nlogo files found for ABM '{abm}': {names}.")
    return matches[0]


def _promote_root_model(*, models_root: Path, abm: str, model_path: Path) -> Path:
    """Copy root-level models into the ABM folder to keep canonical artifacts in one location."""
    if model_path.parent != models_root:
        return model_path

    target_dir = models_root / f"{abm}_abm"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / model_path.name
    if target_path.exists():
        return target_path
    try:
        copy2(model_path, target_path)
    except OSError:
        return model_path
    return target_path


def _resolve_abm_experiment_parameters_path(*, model_dir: Path, abm: str, explicit: Path | None) -> Path | None:
    """Resolve per-ABM experiment-parameters file if available."""
    if explicit is not None:
        if explicit.exists():
            return explicit
        return None

    candidates = [
        model_dir / "experiment_parameters.json",
        model_dir / "experiment-parameters.json",
        model_dir / f"{abm}_experiment_parameters.json",
        model_dir / f"{abm}-experiment_parameters.json",
        model_dir / f"{abm}-experiment-parameters.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _resolve_viz_smoke_specs(*, requested_abms: list[str], models_root: Path) -> dict[str, VizSmokeSpec]:
    """Resolve ABM NetLogo simulation-and-plot specs for visualization smoke runs."""
    specs: dict[str, VizSmokeSpec] = {}
    missing_viz_config: list[str] = []
    for abm in requested_abms:
        abm_config = load_abm_config(Path("configs/abms") / f"{abm}.yaml")
        model_path = _resolve_abm_model_path(abm=abm, models_root=models_root)
        if abm_config.netlogo_viz is None:
            missing_viz_config.append(abm)
            continue
        viz_config = abm_config.netlogo_viz
        experiment_parameters = resolve_experiment_parameters(
            model_path=model_path,
            experiment_parameters=viz_config.experiment_parameters,
            preferred_experiment=viz_config.experiment_name,
        )
        specs[abm] = VizSmokeSpec(
            abm=abm,
            model_path=model_path,
            experiment_name=viz_config.experiment_name,
            experiment_parameters=experiment_parameters,
            num_runs=viz_config.smoke_num_runs or viz_config.num_runs,
            max_ticks=viz_config.smoke_max_ticks or viz_config.max_ticks,
            interval=viz_config.smoke_interval or viz_config.interval,
            fallback_mode=viz_config.fallback_mode,
            fallback_csv=Path(viz_config.fallback_csv) if viz_config.fallback_csv else None,
            fallback_plot_dir=Path(viz_config.fallback_plot_dir) if viz_config.fallback_plot_dir else None,
            reporters=list(viz_config.reporters),
            plots=[
                MetricPlotBundle(
                    include_pattern=plot.reporter_pattern,
                    title=plot.title,
                    y_label=plot.y_label,
                    x_label=plot.x_label,
                    exclude_pattern=plot.exclude_pattern,
                    show_mean_line=plot.show_mean_line,
                )
                for plot in viz_config.plots
            ],
        )
    if missing_viz_config:
        joined = ", ".join(missing_viz_config)
        raise ValueError(
            f"missing netlogo_viz config for ABM(s): {joined}. "
            "Add the simulation-and-plot spec under netlogo_viz in the ABM config."
        )
    return specs


def _parse_summarizers(values: list[str] | None, fallback: tuple[SummarizerId, ...]) -> tuple[SummarizerId, ...]:
    allowed = {"bart", "bert", "t5", "longformer_ext"}
    normalized = tuple(dict.fromkeys(values or list(fallback)))
    invalid = [value for value in normalized if value not in allowed]
    if invalid:
        raise typer.BadParameter(
            "unsupported summarizer(s): "
            f"{', '.join(invalid)}. Allowed: bart, bert, t5, longformer_ext."
        )
    return cast(tuple[SummarizerId, ...], normalized)


def _resolve_model_from_registry(models_path: Path, model_id: str) -> tuple[str, str]:
    config = load_models_config(models_path)
    try:
        entry: ModelEntry = config.models[model_id]
    except KeyError as exc:
        available = ", ".join(sorted(config.models))
        raise typer.BadParameter(f"unknown model_id '{model_id}'. Available: {available}") from exc
    return entry.provider, entry.model


def _resolve_scoring_reference_path(abm: str) -> Path:
    settings = load_experiment_settings(Path("configs/experiment_settings.yaml"))
    mapping = {
        "fauna": settings.ground_truth.fauna,
        "grazing": settings.ground_truth.grazing,
        "milk_consumption": settings.ground_truth.milk_consumption,
    }
    if abm not in mapping:
        raise typer.BadParameter(
            f"unsupported ABM for scoring reference: {abm}. Allowed: fauna, grazing, milk_consumption."
        )
    return Path(mapping[abm])


def _assert_ollama_model_available(model: str) -> None:
    try:
        completed = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise typer.BadParameter(f"failed to run 'ollama list' to verify local model '{model}': {exc}") from exc

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not any(line.split()[0] == model for line in lines[1:] if line.split()):
        raise typer.BadParameter(
            f"required local model '{model}' not found in 'ollama list'. Pull it before benchmark runs."
        )


def _validate_model_policy(provider: str, model: str, allow_debug_model: bool) -> None:
    key = (provider.strip().lower(), model.strip())
    if key == DEBUG_MODEL:
        if not allow_debug_model:
            raise typer.BadParameter(
                "debug model is blocked for benchmark runs. Use --allow-debug-model to run debug workflows."
            )
        return

    if key not in BENCHMARK_MODELS:
        allowed = ", ".join(f"{p}:{m}" for p, m in sorted(BENCHMARK_MODELS))
        raise typer.BadParameter(
            f"unsupported benchmark model '{provider}:{model}'. Allowed benchmark models: {allowed}."
        )

    if key == ("ollama", "qwen3.5:0.8b"):
        _assert_ollama_model_available(model)


def _select_smoke_cases(
    case_ids: list[str] | None, max_cases: int | None, profile: Literal["matrix", "three-branches"]
) -> list[SmokeCase] | None:
    all_cases = default_smoke_cases() if profile == "matrix" else default_branch_smoke_cases()
    if not case_ids:
        if max_cases is None and profile == "matrix":
            return None
        if max_cases is None:
            return all_cases
        return all_cases[:max_cases]

    by_id = {case.case_id: case for case in all_cases}
    unknown = [value for value in case_ids if value not in by_id]
    if unknown:
        known = ", ".join(sorted(by_id))
        raise typer.BadParameter(f"unknown --case-id value(s): {', '.join(unknown)}. Known cases: {known}")
    seen: set[str] = set()
    selected: list[SmokeCase] = []
    for value in case_ids:
        if value in seen:
            continue
        seen.add(value)
        selected.append(by_id[value])
    if max_cases is not None:
        selected = selected[:max_cases]
    if not selected:
        raise typer.BadParameter("at least one smoke case must be selected")
    return selected


if __name__ == "__main__":
    main()
