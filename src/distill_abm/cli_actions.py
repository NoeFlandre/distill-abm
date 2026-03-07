"""Behavior-preserving command implementations extracted from the Typer entrypoint."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import typer

from distill_abm.agent_validation import ValidationProfile
from distill_abm.cli_models import (
    ArtifactDescriptor,
    DescribeAbmResult,
    DescribeArtifactsResult,
    DescribeRunResult,
    DoeCommandResult,
    HealthCheckItem,
    HealthCheckResult,
    IngestCommandResult,
    IngestSuiteCommandResult,
    RunCommandResult,
    SmokeCommandResult,
    build_artifact_descriptors,
    describe_artifact,
)
from distill_abm.cli_output import emit_smoke_command_result, ensure_required_stage_ids
from distill_abm.cli_support import (
    as_dict,
    discover_configured_abms,
    load_experiment_parameters,
    resolve_abm_experiment_parameters_path,
    resolve_abm_model_path,
)
from distill_abm.configs.models import SummarizerId
from distill_abm.eval.qualitative_runner import QualitativeMetric
from distill_abm.ingest.netlogo_workflow import run_ingest_workflow
from distill_abm.pipeline.doe_smoke import (
    CANONICAL_DOE_MODEL_IDS,
    CANONICAL_EVIDENCE_MODES,
    CANONICAL_REPETITIONS,
    DoESmokeAbmInput,
    DoESmokeModelSpec,
    DoESmokePlotInput,
    canonical_prompt_variants,
    canonical_summarization_specs,
)
from distill_abm.pipeline.local_qwen_monitor import (
    collect_local_qwen_monitor_snapshot,
    render_local_qwen_monitor,
    stream_local_qwen_monitor,
)
from distill_abm.pipeline.local_qwen_sample_smoke import LocalQwenCaseInput
from distill_abm.pipeline.local_qwen_tuning import LocalQwenTuningResult
from distill_abm.pipeline.run import EvidenceMode, PipelineInputs, TextSourceMode
from distill_abm.pipeline.smoke import SmokeSuiteInputs

LOCAL_QWEN_TIMEOUT_SECONDS = 900.0


def execute_run_command(
    *,
    csv_path: Path,
    parameters_path: Path,
    documentation_path: Path,
    prompts_path: Path,
    models_path: Path,
    output_dir: Path,
    provider: str,
    model: str,
    model_id: str | None,
    allow_debug_model: bool,
    metric_pattern: str,
    metric_description: str,
    plot_description: str | None,
    evidence_mode: EvidenceMode,
    text_source_mode: TextSourceMode,
    summarizer: list[str] | None,
    allow_summary_fallback: bool,
    abm: str | None,
    json_output: bool,
    default_summarizers: tuple[SummarizerId, ...],
    validate_model_policy: Callable[..., None],
    resolve_model_from_registry: Callable[[Path, str], tuple[str, str]],
    parse_summarizers: Callable[..., tuple[SummarizerId, ...]],
    resolve_scoring_reference_path: Callable[[str], Path],
    create_adapter_fn: Callable[[str, str], Any],
    run_pipeline_fn: Callable[..., Any],
    load_abm_config_fn: Callable[[Path], Any],
    load_prompts_config_fn: Callable[[Path], Any],
) -> None:
    prompts: Any = load_prompts_config_fn(prompts_path)
    if model_id is not None:
        provider, model = resolve_model_from_registry(models_path, model_id)
    validate_model_policy(provider=provider, model=model, allow_debug_model=allow_debug_model)

    scoring_reference_path: Path | None = None
    if abm:
        abm_config: Any = load_abm_config_fn(Path("configs/abms") / f"{abm}.yaml")
        metric_pattern = abm_config.metric_pattern
        metric_description = abm_config.metric_description
        if plot_description is None and abm_config.plot_descriptions:
            plot_description = abm_config.plot_descriptions[0]
        scoring_reference_path = resolve_scoring_reference_path(abm)

    adapter: Any = _create_local_ollama_adapter(
        create_adapter_fn=create_adapter_fn,
        provider=provider,
        model=model,
        timeout_seconds=LOCAL_QWEN_TIMEOUT_SECONDS,
    )
    result: Any = run_pipeline_fn(
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
            summarizers=parse_summarizers(summarizer, fallback=default_summarizers),
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


def execute_ingest_netlogo_command(
    *,
    model_path: Path,
    experiment_parameters_path: Path | None,
    output_dir: Path | None,
    suffix: str,
    json_output: bool,
) -> None:
    resolved_output_dir = output_dir if output_dir is not None else Path("results") / "ingest" / model_path.stem
    experiment_parameters = load_experiment_parameters(experiment_parameters_path)
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


def execute_ingest_netlogo_suite_command(
    *,
    abms: list[str] | None,
    models_root: Path,
    output_root: Path,
    suffix: str,
    continue_on_missing: bool,
    default_experiment_parameters_path: Path | None,
    json_output: bool,
) -> None:
    requested = sorted(set(abms)) if abms else discover_configured_abms()
    missing: list[str] = []
    shared_params = load_experiment_parameters(default_experiment_parameters_path)
    suite_artifacts: dict[str, dict[str, ArtifactDescriptor]] = {}

    for abm in requested:
        try:
            model_path = resolve_abm_model_path(abm=abm, models_root=models_root)
            parameter_path = resolve_abm_experiment_parameters_path(
                model_dir=model_path.parent,
                abm=abm,
                explicit=default_experiment_parameters_path,
            )
            if default_experiment_parameters_path is not None:
                experiment_parameters = shared_params
            else:
                experiment_parameters = load_experiment_parameters(parameter_path)
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


def execute_analyze_doe_command(
    *,
    input_csv: Path,
    output_csv: Path,
    max_interaction_order: int,
    json_output: bool,
    analyze_factorial_anova_fn: Callable[..., Any],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result = analyze_factorial_anova_fn(input_csv, output_csv, max_interaction_order=max_interaction_order)
    if result is None:
        raise typer.Exit(code=1)
    command_result = DoeCommandResult(success=True, output_csv=output_csv)
    if json_output:
        typer.echo(command_result.model_dump_json(indent=2))
        return
    typer.echo(f"wrote: {output_csv}")


def execute_evaluate_qualitative_command(
    *,
    summary_text: str,
    source_text: str,
    metric: QualitativeMetric,
    source_image_path: Path | None,
    prompts_path: Path,
    provider: str,
    model: str,
    allow_debug_model: bool,
    validate_model_policy: Callable[..., None],
    create_adapter_fn: Callable[[str, str], Any],
    load_prompts_config_fn: Callable[[Path], Any],
    evaluate_qualitative_score_fn: Callable[..., Any],
) -> None:
    validate_model_policy(provider=provider, model=model, allow_debug_model=allow_debug_model)
    prompts: Any = load_prompts_config_fn(prompts_path)
    adapter: Any = _create_local_ollama_adapter(
        create_adapter_fn=create_adapter_fn,
        provider=provider,
        model=model,
        timeout_seconds=LOCAL_QWEN_TIMEOUT_SECONDS,
    )
    result: Any = evaluate_qualitative_score_fn(
        summary=summary_text,
        source=source_text,
        metric=metric,
        model=model,
        prompts=prompts,
        adapter=adapter,
        source_image_path=source_image_path,
    )
    typer.echo(result.model_dump_json())


def execute_smoke_qwen_command(
    *,
    csv_path: Path,
    parameters_path: Path,
    documentation_path: Path,
    doe_input_csv: Path | None,
    prompts_path: Path,
    output_dir: Path,
    provider: str,
    model: str,
    allow_debug_model: bool,
    metric_pattern: str,
    metric_description: str,
    plot_description: str | None,
    evidence_mode: EvidenceMode,
    text_source_mode: TextSourceMode,
    allow_summary_fallback: bool,
    abm: str | None,
    summarizer: list[str] | None,
    skip_qualitative: bool,
    skip_sweep: bool,
    profile: Literal["matrix", "three-branches"],
    case_id: list[str] | None,
    max_cases: int | None,
    resume: bool,
    json_output: bool,
    smoke_default_summarizers: tuple[SummarizerId, ...],
    validate_model_policy: Callable[..., None],
    select_smoke_cases: Callable[..., Any],
    parse_summarizers: Callable[..., tuple[SummarizerId, ...]],
    resolve_scoring_reference_path: Callable[[str], Path],
    create_adapter_fn: Callable[[str, str], Any],
    run_qwen_smoke_suite_fn: Callable[..., Any],
    load_abm_config_fn: Callable[[Path], Any],
    load_prompts_config_fn: Callable[[Path], Any],
) -> None:
    validate_model_policy(provider=provider, model=model, allow_debug_model=allow_debug_model)
    prompts: Any = load_prompts_config_fn(prompts_path)
    sweep_plot_descriptions: list[str] | None = None
    scoring_reference_path: Path | None = None
    if abm:
        abm_config: Any = load_abm_config_fn(Path("configs/abms") / f"{abm}.yaml")
        metric_pattern = abm_config.metric_pattern
        metric_description = abm_config.metric_description
        if plot_description is None and abm_config.plot_descriptions:
            plot_description = abm_config.plot_descriptions[0]
        sweep_plot_descriptions = list(abm_config.plot_descriptions)
        scoring_reference_path = resolve_scoring_reference_path(abm)
    selected_cases: Any = select_smoke_cases(case_ids=case_id, max_cases=max_cases, profile=profile)
    adapter: Any = _create_local_ollama_adapter(
        create_adapter_fn=create_adapter_fn,
        provider=provider,
        model=model,
        timeout_seconds=LOCAL_QWEN_TIMEOUT_SECONDS,
    )
    result: Any = run_qwen_smoke_suite_fn(
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
            summarizers=parse_summarizers(summarizer, fallback=smoke_default_summarizers),
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


def execute_smoke_doe_command(
    *,
    abms: list[str] | None,
    models_root: Path,
    ingest_root: Path,
    viz_root: Path,
    prompts_path: Path,
    models_path: Path,
    model_ids: list[str] | None,
    output_root: Path,
    json_output: bool,
    discover_abms: Callable[[], tuple[str, ...]],
    resolve_model_from_registry: Callable[[Path, str], tuple[str, str]],
    resolve_model_path: Callable[..., Path],
    run_doe_smoke_suite_fn: Callable[..., Any],
    load_abm_config_fn: Callable[[Path], Any],
    load_prompts_config_fn: Callable[[Path], Any],
) -> None:
    prompts: Any = load_prompts_config_fn(prompts_path)
    requested = sorted(set(abms)) if abms else list(discover_abms())
    selected_model_ids = tuple(dict.fromkeys(model_ids or list(CANONICAL_DOE_MODEL_IDS)))
    model_specs: list[DoESmokeModelSpec] = []
    for candidate_model_id in selected_model_ids:
        provider, model = resolve_model_from_registry(models_path, candidate_model_id)
        model_specs.append(DoESmokeModelSpec(model_id=candidate_model_id, provider=provider, model=model))

    abm_inputs: dict[str, DoESmokeAbmInput] = {}
    for abm in requested:
        abm_config: Any = load_abm_config_fn(Path("configs/abms") / f"{abm}.yaml")
        _ = resolve_model_path(abm=abm, models_root=models_root)
        input_csv_path = viz_root / abm / "simulation.csv"
        parameters_path = ingest_root / abm / "TXT" / "narrative_combined.txt"
        documentation_path = ingest_root / abm / "TXT" / "final_documentation.txt"
        artifact_source_path = viz_root / abm / "artifact_source.txt"
        artifact_source: Literal["simulated", "fallback", "unknown"] = "unknown"
        if artifact_source_path.exists():
            loaded_source = artifact_source_path.read_text(encoding="utf-8").strip()
            if loaded_source in {"simulated", "fallback"}:
                artifact_source = cast(Literal["simulated", "fallback"], loaded_source)
        if abm_config.netlogo_viz is None:
            raise typer.BadParameter(f"missing netlogo_viz config for ABM '{abm}'")
        plot_specs: list[DoESmokePlotInput] = []
        plot_config_count = len(abm_config.netlogo_viz.plots)
        plot_description_count = len(abm_config.plot_descriptions)
        max_plot_count = max(plot_config_count, plot_description_count)
        for plot_index in range(1, max_plot_count + 1):
            reporter_pattern = (
                abm_config.netlogo_viz.plots[plot_index - 1].reporter_pattern
                if plot_index <= plot_config_count
                else "missing-reporter-pattern"
            )
            plot_description = (
                abm_config.plot_descriptions[plot_index - 1]
                if plot_index <= plot_description_count
                else "TODO missing plot description"
            )
            plot_specs.append(
                DoESmokePlotInput(
                    plot_index=plot_index,
                    reporter_pattern=reporter_pattern,
                    plot_description=plot_description,
                    plot_path=viz_root / abm / "plots" / f"{plot_index}.png",
                )
            )
        abm_inputs[abm] = DoESmokeAbmInput(
            abm=abm,
            csv_path=input_csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            metric_pattern=abm_config.metric_pattern,
            metric_description=abm_config.metric_description,
            plots=plot_specs,
            source_viz_artifact_source=artifact_source,
        )

    result: Any = run_doe_smoke_suite_fn(
        abm_inputs=abm_inputs,
        prompts=prompts,
        model_specs=model_specs,
        output_root=output_root,
        evidence_modes=CANONICAL_EVIDENCE_MODES,
        summarization_specs=canonical_summarization_specs(),
        prompt_variants=canonical_prompt_variants(),
        repetitions=CANONICAL_REPETITIONS,
    )
    command_result = SmokeCommandResult(
        command="smoke-doe",
        success=result.success,
        report_json_path=result.report_json_path,
        report_markdown_path=result.report_markdown_path,
        failed_items=result.failed_case_ids,
        nested_artifacts={
            "design_matrix_csv": result.design_matrix_csv_path,
            "request_matrix_csv": result.request_matrix_csv_path,
        },
    )
    if json_output:
        typer.echo(command_result.model_dump_json(indent=2))
        if not result.success:
            raise typer.Exit(code=1)
        return
    typer.echo(f"doe smoke report (markdown): {result.report_markdown_path}")
    typer.echo(f"doe smoke report (json): {result.report_json_path}")
    typer.echo(f"design matrix (csv): {result.design_matrix_csv_path}")
    if not result.success:
        typer.echo(f"doe smoke failed: {', '.join(result.failed_case_ids)}")
        raise typer.Exit(code=1)


def execute_smoke_local_qwen_command(
    *,
    abms: list[str] | None,
    models_root: Path,
    ingest_root: Path,
    viz_root: Path,
    models_path: Path,
    model_id: str,
    output_root: Path,
    max_tokens: int,
    num_ctx: int,
    plot_num_ctx: int | None,
    table_num_ctx: int | None,
    plot_table_num_ctx: int | None,
    resume: bool,
    json_output: bool,
    discover_abms: Callable[[], tuple[str, ...]],
    resolve_model_from_registry: Callable[[Path, str], tuple[str, str]],
    resolve_model_path: Callable[..., Path],
    assert_ollama_model_available: Callable[[str], None],
    create_adapter_fn: Callable[[str, str], Any],
    run_local_qwen_sample_smoke_fn: Callable[..., Any],
    load_abm_config_fn: Callable[[Path], Any],
) -> None:
    requested = sorted(set(abms)) if abms else list(discover_abms())
    provider, model = resolve_model_from_registry(models_path, model_id)
    if provider not in {"ollama", "openrouter"}:
        raise typer.BadParameter(
            f"model id '{model_id}' must resolve to an ollama or openrouter model for smoke-local-qwen."
        )
    if provider == "ollama":
        assert_ollama_model_available(model)
    if max_tokens <= 0:
        raise typer.BadParameter("--max-tokens must be positive")
    if num_ctx <= 0:
        raise typer.BadParameter("--num-ctx must be positive")
    for label, value in (
        ("--plot-num-ctx", plot_num_ctx),
        ("--table-num-ctx", table_num_ctx),
        ("--plot-table-num-ctx", plot_table_num_ctx),
    ):
        if value is not None and value <= 0:
            raise typer.BadParameter(f"{label} must be positive")

    case_inputs: dict[str, LocalQwenCaseInput] = {}
    for abm in requested:
        abm_config: Any = load_abm_config_fn(Path("configs/abms") / f"{abm}.yaml")
        _ = resolve_model_path(abm=abm, models_root=models_root)
        if abm_config.netlogo_viz is None or not abm_config.netlogo_viz.plots:
            raise typer.BadParameter(f"missing netlogo_viz plot config for ABM '{abm}'")
        if not abm_config.plot_descriptions:
            raise typer.BadParameter(f"missing plot descriptions for ABM '{abm}'")
        first_plot = abm_config.netlogo_viz.plots[0]
        case_inputs[abm] = LocalQwenCaseInput(
            abm=abm,
            csv_path=viz_root / abm / "simulation.csv",
            parameters_path=ingest_root / abm / "TXT" / "narrative_combined.txt",
            documentation_path=ingest_root / abm / "TXT" / "final_documentation.txt",
            reporter_pattern=first_plot.reporter_pattern,
            plot_description=abm_config.plot_descriptions[0],
            plot_path=viz_root / abm / "plots" / "1.png",
        )

    adapter: Any = _create_local_ollama_adapter(
        create_adapter_fn=create_adapter_fn,
        provider=provider,
        model=model,
        timeout_seconds=LOCAL_QWEN_TIMEOUT_SECONDS,
    )
    result: Any = run_local_qwen_sample_smoke_fn(
        case_inputs=case_inputs,
        adapter=adapter,
        model=model,
        output_root=output_root,
        max_tokens=max_tokens,
        ollama_num_ctx=num_ctx,
        ollama_num_ctx_by_mode={
            "plot": plot_num_ctx or num_ctx,
            "table": table_num_ctx or num_ctx,
            "plot+table": plot_table_num_ctx or num_ctx,
        },
        resume_existing=resume,
    )
    command_result = SmokeCommandResult(
        command="smoke-local-qwen",
        success=result.success,
        report_json_path=result.report_json_path,
        report_markdown_path=result.report_markdown_path,
        failed_items=result.failed_case_ids,
        nested_artifacts={"request_review_csv": result.review_csv_path},
    )
    emit_smoke_command_result(
        command_result=command_result,
        json_output=json_output,
        markdown_label="sampled llm smoke report (markdown)",
        json_label="sampled llm smoke report (json)",
        failure_label="sampled llm smoke failed",
    )


def execute_monitor_local_qwen_command(
    *,
    output_root: Path,
    watch: bool,
    interval_seconds: float,
    json_output: bool,
) -> None:
    if watch:
        if json_output:
            raise typer.BadParameter("--json cannot be used together with --watch")
        stream_local_qwen_monitor(output_root=output_root, interval_seconds=interval_seconds)
        return

    snapshot = collect_local_qwen_monitor_snapshot(output_root)
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "output_root": str(snapshot.output_root),
                    "exists": snapshot.exists,
                    "mode": snapshot.mode,
                    "total_cases": snapshot.total_cases,
                    "completed_cases": snapshot.completed_cases,
                    "failed_cases": snapshot.failed_cases,
                    "running_case_id": snapshot.running_case_id,
                    "terminal": snapshot.terminal,
                    "cases": [case.__dict__ for case in snapshot.cases],
                },
                indent=2,
            )
        )
        return
    typer.echo(render_local_qwen_monitor(snapshot))


def execute_tune_local_qwen_command(
    *,
    abms: list[str] | None,
    models_root: Path,
    ingest_root: Path,
    viz_root: Path,
    models_path: Path,
    model_id: str,
    output_root: Path,
    num_ctx_candidates: list[int] | None,
    max_tokens_candidates: list[int] | None,
    resume: bool,
    json_output: bool,
    discover_abms: Callable[[], tuple[str, ...]],
    resolve_model_from_registry: Callable[[Path, str], tuple[str, str]],
    resolve_model_path: Callable[..., Path],
    assert_ollama_model_available: Callable[[str], None],
    create_adapter_fn: Callable[[str, str], Any],
    run_local_qwen_tuning_fn: Callable[..., LocalQwenTuningResult],
    load_abm_config_fn: Callable[[Path], Any],
) -> None:
    requested = sorted(set(abms)) if abms else list(discover_abms())
    provider, model = resolve_model_from_registry(models_path, model_id)
    if provider != "ollama":
        raise typer.BadParameter(f"model id '{model_id}' must resolve to an ollama model for tune-local-qwen.")
    assert_ollama_model_available(model)
    resolved_num_ctx_candidates = tuple(sorted(set(num_ctx_candidates or [8192, 16384, 32768, 65536, 131072])))
    resolved_max_tokens_candidates = tuple(sorted(set(max_tokens_candidates or [1024, 2048, 4096, 8192, 16384])))
    if any(item <= 0 for item in resolved_max_tokens_candidates):
        raise typer.BadParameter("--max-tokens values must be positive")
    case_inputs: dict[str, LocalQwenCaseInput] = {}
    for abm in requested:
        abm_config: Any = load_abm_config_fn(Path("configs/abms") / f"{abm}.yaml")
        _ = resolve_model_path(abm=abm, models_root=models_root)
        if abm_config.netlogo_viz is None or not abm_config.netlogo_viz.plots:
            raise typer.BadParameter(f"missing netlogo_viz plot config for ABM '{abm}'")
        if not abm_config.plot_descriptions:
            raise typer.BadParameter(f"missing plot descriptions for ABM '{abm}'")
        first_plot = abm_config.netlogo_viz.plots[0]
        case_inputs[abm] = LocalQwenCaseInput(
            abm=abm,
            csv_path=viz_root / abm / "simulation.csv",
            parameters_path=ingest_root / abm / "TXT" / "narrative_combined.txt",
            documentation_path=ingest_root / abm / "TXT" / "final_documentation.txt",
            reporter_pattern=first_plot.reporter_pattern,
            plot_description=abm_config.plot_descriptions[0],
            plot_path=viz_root / abm / "plots" / "1.png",
        )

    adapter: Any = _create_local_ollama_adapter(
        create_adapter_fn=create_adapter_fn,
        provider=provider,
        model=model,
        timeout_seconds=LOCAL_QWEN_TIMEOUT_SECONDS,
    )
    result = run_local_qwen_tuning_fn(
        case_inputs=case_inputs,
        adapter=adapter,
        model=model,
        output_root=output_root,
        num_ctx_candidates=resolved_num_ctx_candidates,
        max_tokens_candidates=resolved_max_tokens_candidates,
        resume_existing=resume,
    )
    command_result = SmokeCommandResult(
        command="tune-local-qwen",
        success=result.success,
        report_json_path=result.report_json_path,
        report_markdown_path=result.report_markdown_path,
        failed_items=[
            recommendation.evidence_mode
            for recommendation in result.recommendations
            if not recommendation.based_on_successful_trial
        ],
        nested_artifacts={"trials_csv": result.trials_csv_path},
    )
    emit_smoke_command_result(
        command_result=command_result,
        json_output=json_output,
        markdown_label="local qwen tuning report (markdown)",
        json_label="local qwen tuning report (json)",
        failure_label="local qwen tuning failed",
    )


def execute_smoke_ingest_command(
    *,
    abms: list[str] | None,
    models_root: Path,
    output_root: Path,
    stage: list[str] | None,
    require_stage: list[str] | None,
    json_output: bool,
    run_ingest_smoke_suite_fn: Callable[..., Any],
) -> None:
    requested = sorted(set(abms)) if abms else list(discover_configured_abms())
    abm_models = {abm: resolve_abm_model_path(abm=abm, models_root=models_root) for abm in requested}
    try:
        result: Any = run_ingest_smoke_suite_fn(abm_models=abm_models, output_root=output_root, stage_ids=stage)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if require_stage:
        ensure_required_stage_ids(
            selected_stage_ids=result.selected_stage_ids,
            required_stage_ids=require_stage,
            label="ingest smoke",
        )
    command_result = SmokeCommandResult(
        command="smoke-ingest-netlogo",
        success=result.success,
        report_json_path=result.report_json_path,
        report_markdown_path=result.report_markdown_path,
        failed_items=result.failed_abms,
    )
    emit_smoke_command_result(
        command_result=command_result,
        json_output=json_output,
        markdown_label="ingest smoke report (markdown)",
        json_label="ingest smoke report (json)",
        failure_label="ingest smoke failed",
    )


def _create_local_ollama_adapter(
    *,
    create_adapter_fn: Callable[..., Any],
    provider: str,
    model: str,
    timeout_seconds: float,
) -> Any:
    try:
        return create_adapter_fn(provider, model, timeout_seconds=timeout_seconds)
    except TypeError:
        return create_adapter_fn(provider, model)


def execute_smoke_viz_command(
    *,
    abms: list[str] | None,
    models_root: Path,
    netlogo_home: str,
    stage: list[str] | None,
    require_stage: list[str] | None,
    output_root: Path,
    json_output: bool,
    resolve_viz_smoke_specs: Callable[..., Any],
    run_viz_smoke_suite_fn: Callable[..., Any],
) -> None:
    requested = sorted(set(abms)) if abms else list(discover_configured_abms())
    try:
        if not netlogo_home.strip():
            raise ValueError(
                "missing NetLogo installation directory. Provide --netlogo-home or set DISTILL_ABM_NETLOGO_HOME."
            )
        specs: Any = resolve_viz_smoke_specs(requested_abms=requested, models_root=models_root)
        result: Any = run_viz_smoke_suite_fn(
            specs=specs,
            netlogo_home=netlogo_home,
            output_root=output_root,
            stage_ids=stage,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if require_stage:
        ensure_required_stage_ids(
            selected_stage_ids=result.selected_stage_ids,
            required_stage_ids=require_stage,
            label="viz smoke",
        )
    command_result = SmokeCommandResult(
        command="smoke-viz",
        success=result.success,
        report_json_path=result.report_json_path,
        report_markdown_path=result.report_markdown_path,
        failed_items=result.failed_abms,
    )
    emit_smoke_command_result(
        command_result=command_result,
        json_output=json_output,
        markdown_label="viz smoke report (markdown)",
        json_label="viz smoke report (json)",
        failure_label="viz smoke failed",
    )


def execute_validate_workspace_command(
    *,
    checks: list[str] | None,
    abms: list[str] | None,
    models_root: Path,
    ingest_stage: list[str] | None,
    profile: ValidationProfile,
    output_root: Path,
    json_output: bool,
    run_validation_suite_fn: Callable[..., Any],
) -> None:
    requested = sorted(set(abms)) if abms else list(discover_configured_abms())
    abm_models = {abm: resolve_abm_model_path(abm=abm, models_root=models_root) for abm in requested}
    try:
        result: Any = run_validation_suite_fn(
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


def execute_health_check_command(
    *,
    models_root: Path,
    ingest_root: Path,
    viz_root: Path,
    include_ollama: bool,
    json_output: bool,
    discover_abms: Callable[[], tuple[str, ...]],
    resolve_model_path: Callable[..., Path],
    resolve_model_from_registry: Callable[[Path, str], tuple[str, str]],
    models_path: Path,
    assert_ollama_model_available: Callable[[str], None],
    load_abm_config_fn: Callable[[Path], Any],
) -> None:
    checks: dict[str, HealthCheckItem] = {}
    requested = list(discover_abms())
    checks["configured_abms"] = HealthCheckItem(ok=bool(requested), detail=", ".join(requested))

    try:
        provider, model = resolve_model_from_registry(models_path, "qwen3_5_local")
        checks["model_registry"] = HealthCheckItem(ok=True, detail=f"qwen3_5_local -> {provider}:{model}")
    except Exception as exc:
        checks["model_registry"] = HealthCheckItem(ok=False, detail=str(exc))

    abm_ok = True
    abm_details: list[str] = []
    for abm in requested:
        try:
            config = load_abm_config_fn(Path("configs/abms") / f"{abm}.yaml")
            resolve_model_path(abm=abm, models_root=models_root)
            plot_count = len(getattr(config, "plot_descriptions", []))
            abm_details.append(f"{abm}({plot_count} plots)")
        except Exception as exc:
            abm_ok = False
            abm_details.append(f"{abm}(error: {exc})")
    checks["abm_configs"] = HealthCheckItem(ok=abm_ok, detail=", ".join(abm_details))

    checks["ingest_root"] = HealthCheckItem(
        ok=ingest_root.exists(),
        detail=str(ingest_root),
    )
    checks["viz_root"] = HealthCheckItem(
        ok=viz_root.exists(),
        detail=str(viz_root),
    )

    if include_ollama:
        try:
            assert_ollama_model_available("qwen3.5:0.8b")
            checks["ollama_local_qwen"] = HealthCheckItem(ok=True, detail="qwen3.5:0.8b available")
        except Exception as exc:
            checks["ollama_local_qwen"] = HealthCheckItem(ok=False, detail=str(exc))

    success = all(item.ok for item in checks.values())
    result = HealthCheckResult(success=success, checks=checks)
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        if not success:
            raise typer.Exit(code=1)
        return
    for name, item in result.checks.items():
        status = "ok" if item.ok else "failed"
        typer.echo(f"{name}: {status} - {item.detail}")
    if not success:
        raise typer.Exit(code=1)


def execute_describe_abm_command(
    *,
    abm: str,
    models_root: Path,
    json_output: bool,
    resolve_scoring_reference_path: Callable[[str], Path],
    load_abm_config_fn: Callable[[Path], Any],
) -> None:
    config_path = Path("configs/abms") / f"{abm}.yaml"
    abm_config: Any = load_abm_config_fn(config_path)
    model_path = resolve_abm_model_path(abm=abm, models_root=models_root)
    experiment_parameters_path = resolve_abm_experiment_parameters_path(
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
            resolve_scoring_reference_path(abm) if abm in {"fauna", "grazing", "milk_consumption"} else None
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


def execute_describe_ingest_artifacts_command(*, root: Path, json_output: bool) -> None:
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


def execute_describe_run_command(*, output_dir: Path, json_output: bool) -> None:
    metadata_path = output_dir / "pipeline_run_metadata.json"
    if not metadata_path.exists():
        raise typer.BadParameter(f"missing pipeline metadata file: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifacts_payload = as_dict(payload.get("artifacts"))
    reproducibility = as_dict(payload.get("reproducibility"))
    execution = as_dict(payload.get("execution"))
    debug_trace = as_dict(payload.get("debug_trace"))
    frame_summary = as_dict(debug_trace.get("frame_summary"))
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
