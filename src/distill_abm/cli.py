"""Typer CLI entrypoint for distill-abm paper-aligned workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from distill_abm.agent_validation import ValidationProfile, run_validation_suite
from distill_abm.cli_actions import (
    execute_analyze_doe_command,
    execute_describe_abm_command,
    execute_describe_ingest_artifacts_command,
    execute_describe_run_command,
    execute_evaluate_qualitative_command,
    execute_health_check_command,
    execute_ingest_netlogo_command,
    execute_ingest_netlogo_suite_command,
    execute_monitor_local_qwen_command,
    execute_render_run_viewer_command,
    execute_run_command,
    execute_smoke_doe_command,
    execute_smoke_full_case_command,
    execute_smoke_full_case_matrix_command,
    execute_smoke_full_case_suite_command,
    execute_smoke_ingest_command,
    execute_smoke_local_qwen_command,
    execute_smoke_optimization_gemini_chain_command,
    execute_smoke_quantitative_command,
    execute_smoke_quantitative_multi_llm_command,
    execute_smoke_qwen_command,
    execute_smoke_summarizers_command,
    execute_smoke_viz_command,
    execute_sync_results_bucket_command,
    execute_validate_workspace_command,
)
from distill_abm.cli_defaults import (
    DEFAULT_EVIDENCE_MODE,
    DEFAULT_FULL_CASE_MATRIX_EVIDENCE_MODES,
    DEFAULT_FULL_CASE_MATRIX_PROMPT_VARIANTS,
    DEFAULT_FULL_CASE_MATRIX_REPETITIONS,
    DEFAULT_SUMMARIZERS,
    DEFAULT_TEXT_SOURCE_MODE,
    RUNTIME_DEFAULTS,
    resolve_full_case_matrix_evidence_modes,
    resolve_full_case_matrix_prompt_variants,
    resolve_full_case_matrix_repetitions,
)
from distill_abm.cli_policy import validate_benchmark_model_policy
from distill_abm.cli_quality_gate import QualityGateScope, resolve_quality_gate_selection
from distill_abm.cli_support import (
    BENCHMARK_MODELS,
    discover_configured_abms,
    parse_summarizers,
    resolve_abm_model_path,
    resolve_additional_scoring_reference_paths,
    resolve_doe_summarization_specs,
    resolve_model_from_registry,
    resolve_scoring_reference_path,
    resolve_viz_smoke_specs,
    select_smoke_cases,
)
from distill_abm.configs.loader import (
    load_abm_config,
    load_prompts_config,
)
from distill_abm.eval.doe_full import analyze_factorial_anova
from distill_abm.eval.qualitative_runner import QualitativeMetric, evaluate_qualitative_score
from distill_abm.ingest.ingest_smoke import run_ingest_smoke_suite
from distill_abm.llm.factory import create_adapter
from distill_abm.pipeline.doe_smoke import (
    run_doe_smoke_suite,
)
from distill_abm.pipeline.exploitation_factor_study import run_exploitation_factor_study
from distill_abm.pipeline.full_case_matrix_smoke import run_full_case_matrix_smoke
from distill_abm.pipeline.full_case_smoke import run_full_case_smoke
from distill_abm.pipeline.full_case_suite_smoke import run_full_case_suite_smoke
from distill_abm.pipeline.llm_same_settings_study import run_llm_same_settings_study
from distill_abm.pipeline.local_qwen_sample_smoke import run_local_qwen_sample_smoke
from distill_abm.pipeline.quantitative_smoke import (
    run_quantitative_smoke,
    run_quantitative_smoke_multi_llm,
)
from distill_abm.pipeline.run import EvidenceMode, TextSourceMode, run_pipeline
from distill_abm.pipeline.smoke import (
    run_qwen_smoke_suite,
)
from distill_abm.pipeline.summarizer_smoke import run_summarizer_smoke
from distill_abm.viz.viz_smoke import run_viz_smoke_suite

app = typer.Typer(help="Run ABM distillation workflows.")

ARCHIVE_RESULTS_ROOT = Path("results/archive")

# Backward-compatible helper aliases kept for tests and local call sites.
_resolve_scoring_reference_path = resolve_scoring_reference_path
_resolve_additional_scoring_reference_paths = resolve_additional_scoring_reference_paths
_resolve_viz_smoke_specs = resolve_viz_smoke_specs
_select_smoke_cases = select_smoke_cases
_parse_summarizers = parse_summarizers

__all__ = [
    "analyze_doe",
    "app",
    "evaluate_qualitative",
    "health_check",
    "ingest_netlogo",
    "ingest_netlogo_suite",
    "main",
    "monitor_local_qwen",
    "quality_gate",
    "render_run_viewer",
    "run",
    "smoke_doe",
    "smoke_full_case",
    "smoke_ingest_netlogo",
    "smoke_local_qwen",
    "smoke_optimization_gemini_chain",
    "smoke_quantitative",
    "smoke_qwen",
    "smoke_summarizers",
    "smoke_viz",
    "study_exploitation_factors",
    "study_llm_same_settings",
    "sync_results_bucket",
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
    execute_run_command(
        csv_path=csv_path,
        parameters_path=parameters_path,
        documentation_path=documentation_path,
        prompts_path=prompts_path,
        models_path=models_path,
        output_dir=output_dir,
        provider=provider,
        model=model,
        model_id=model_id,
        allow_debug_model=allow_debug_model,
        metric_pattern=metric_pattern,
        metric_description=metric_description,
        plot_description=plot_description,
        evidence_mode=evidence_mode,
        text_source_mode=text_source_mode,
        summarizer=summarizer,
        allow_summary_fallback=allow_summary_fallback,
        abm=abm,
        json_output=json_output,
        default_summarizers=DEFAULT_SUMMARIZERS,
        validate_model_policy=_validate_model_policy,
        resolve_model_from_registry=resolve_model_from_registry,
        parse_summarizers=_parse_summarizers,
        resolve_scoring_reference_path=_resolve_scoring_reference_path,
        resolve_additional_scoring_reference_paths=_resolve_additional_scoring_reference_paths,
        create_adapter_fn=create_adapter,
        run_pipeline_fn=run_pipeline,
        load_abm_config_fn=load_abm_config,
        load_prompts_config_fn=load_prompts_config,
    )


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
    execute_ingest_netlogo_command(
        model_path=model_path,
        experiment_parameters_path=experiment_parameters_path,
        output_dir=output_dir,
        suffix=suffix,
        json_output=json_output,
    )


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
    execute_ingest_netlogo_suite_command(
        abms=abms,
        models_root=models_root,
        output_root=output_root,
        suffix=suffix,
        continue_on_missing=continue_on_missing,
        default_experiment_parameters_path=default_experiment_parameters_path,
        json_output=json_output,
    )


@app.command("analyze-doe")
def analyze_doe(
    input_csv: Annotated[Path, typer.Option(..., exists=True, file_okay=True, dir_okay=False)],
    output_csv: Annotated[Path, typer.Option()] = Path(RUNTIME_DEFAULTS.doe.output_csv),
    max_interaction_order: Annotated[int, typer.Option()] = RUNTIME_DEFAULTS.doe.max_interaction_order,
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Run full factorial ANOVA contribution analysis."""
    execute_analyze_doe_command(
        input_csv=input_csv,
        output_csv=output_csv,
        max_interaction_order=max_interaction_order,
        json_output=json_output,
        analyze_factorial_anova_fn=analyze_factorial_anova,
    )


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
    execute_evaluate_qualitative_command(
        summary_text=summary_text,
        source_text=source_text,
        metric=metric,
        source_image_path=source_image_path,
        prompts_path=prompts_path,
        provider=provider,
        model=model,
        allow_debug_model=allow_debug_model,
        validate_model_policy=_validate_model_policy,
        create_adapter_fn=create_adapter,
        load_prompts_config_fn=load_prompts_config,
        evaluate_qualitative_score_fn=evaluate_qualitative_score,
    )


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
    execute_smoke_qwen_command(
        csv_path=csv_path,
        parameters_path=parameters_path,
        documentation_path=documentation_path,
        doe_input_csv=doe_input_csv,
        prompts_path=prompts_path,
        output_dir=output_dir,
        provider=provider,
        model=model,
        allow_debug_model=allow_debug_model,
        metric_pattern=metric_pattern,
        metric_description=metric_description,
        plot_description=plot_description,
        evidence_mode=evidence_mode,
        text_source_mode=text_source_mode,
        allow_summary_fallback=allow_summary_fallback,
        abm=abm,
        summarizer=summarizer,
        skip_qualitative=skip_qualitative,
        skip_sweep=skip_sweep,
        profile=profile,
        case_id=case_id,
        max_cases=max_cases,
        resume=resume,
        json_output=json_output,
        smoke_default_summarizers=RUNTIME_DEFAULTS.smoke.summarizers,
        validate_model_policy=_validate_model_policy,
        select_smoke_cases=_select_smoke_cases,
        parse_summarizers=_parse_summarizers,
        resolve_scoring_reference_path=_resolve_scoring_reference_path,
        resolve_additional_scoring_reference_paths=_resolve_additional_scoring_reference_paths,
        create_adapter_fn=create_adapter,
        run_qwen_smoke_suite_fn=run_qwen_smoke_suite,
        load_abm_config_fn=load_abm_config,
        load_prompts_config_fn=load_prompts_config,
    )


@app.command("smoke-doe")
def smoke_doe(
    abms: Annotated[
        list[str] | None,
        typer.Option(
            "--abm",
            help="ABM names to inspect. Repeat for multiple. Defaults to all configured ABMs.",
        ),
    ] = None,
    models_root: Annotated[
        Path,
        typer.Option(help="Root directory containing ABM model files for asset discovery."),
    ] = Path("data"),
    ingest_root: Annotated[
        Path,
        typer.Option(help="Root directory containing ingest smoke outputs."),
    ] = ARCHIVE_RESULTS_ROOT / "ingest_smoke_latest",
    viz_root: Annotated[
        Path,
        typer.Option(help="Root directory containing visualization smoke outputs."),
    ] = ARCHIVE_RESULTS_ROOT / "viz_smoke_latest",
    prompts_path: Annotated[
        Path,
        typer.Option(exists=True),
    ] = Path("configs/prompts.yaml"),
    models_path: Annotated[
        Path,
        typer.Option(exists=True, help="Model registry YAML path."),
    ] = Path("configs/models.yaml"),
    model_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--model-id",
            help=(
                "Candidate model aliases from configs/models.yaml. "
                "Repeat to filter. Defaults to the canonical DOE trio."
            ),
        ),
    ] = None,
    evidence_mode: Annotated[
        list[str] | None,
        typer.Option(
            "--evidence-mode",
            help=(
                "Evidence modes to include. Repeat for multiple. "
                f"Defaults to {', '.join(DEFAULT_FULL_CASE_MATRIX_EVIDENCE_MODES)}."
            ),
        ),
    ] = None,
    prompt_variant: Annotated[
        list[str] | None,
        typer.Option(
            "--prompt-variant",
            help=(
                "Prompt variants to include. Repeat for multiple. "
                f"Defaults to {', '.join(DEFAULT_FULL_CASE_MATRIX_PROMPT_VARIANTS)}."
            ),
        ),
    ] = None,
    repetition: Annotated[
        list[int] | None,
        typer.Option(
            "--repetition",
            help=(
                "Repetitions to include. Repeat for multiple. "
                f"Defaults to {', '.join(str(item) for item in DEFAULT_FULL_CASE_MATRIX_REPETITIONS)}."
            ),
        ),
    ] = None,
    summarization_mode: Annotated[
        list[str] | None,
        typer.Option(
            "--summarization-mode",
            help="Summarization conditions to include. Repeatable: none, bart, bert, t5, longformer_ext.",
        ),
    ] = None,
    output_root: Annotated[
        Path,
        typer.Option(help="Directory for DOE smoke reports, shared artifacts, and compact case indexes."),
    ] = ARCHIVE_RESULTS_ROOT / "doe_smoke_latest",
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Inspect the full pre-LLM DOE design and materialize shared artifacts plus compact case indexes."""
    execute_smoke_doe_command(
        abms=abms,
        models_root=models_root,
        ingest_root=ingest_root,
        viz_root=viz_root,
        prompts_path=prompts_path,
        models_path=models_path,
        model_ids=model_ids,
        output_root=output_root,
        evidence_modes=resolve_full_case_matrix_evidence_modes(evidence_mode),
        summarization_specs=resolve_doe_summarization_specs(summarization_mode),
        prompt_variants=resolve_full_case_matrix_prompt_variants(prompt_variant),
        repetitions=resolve_full_case_matrix_repetitions(repetition),
        json_output=json_output,
        discover_abms=discover_configured_abms,
        resolve_model_from_registry=resolve_model_from_registry,
        resolve_model_path=resolve_abm_model_path,
        run_doe_smoke_suite_fn=run_doe_smoke_suite,
        load_abm_config_fn=load_abm_config,
        load_prompts_config_fn=load_prompts_config,
    )


@app.command("smoke-local-qwen")
def smoke_local_qwen(
    abms: Annotated[
        list[str] | None,
        typer.Option(
            "--abm",
            help="ABM names to inspect. Repeat for multiple. Defaults to all configured ABMs.",
        ),
    ] = None,
    models_root: Annotated[
        Path,
        typer.Option(help="Root directory containing ABM model files for asset discovery."),
    ] = Path("data"),
    ingest_root: Annotated[
        Path,
        typer.Option(help="Root directory containing ingest smoke outputs."),
    ] = ARCHIVE_RESULTS_ROOT / "ingest_smoke_latest",
    viz_root: Annotated[
        Path,
        typer.Option(help="Root directory containing visualization smoke outputs."),
    ] = ARCHIVE_RESULTS_ROOT / "viz_smoke_latest",
    models_path: Annotated[
        Path,
        typer.Option(exists=True, help="Model registry YAML path."),
    ] = Path("configs/models.yaml"),
    model_id: Annotated[
        str,
        typer.Option(help="Model alias from configs/models.yaml."),
    ] = "nemotron_nano_12b_v2_vl_free",
    output_root: Annotated[
        Path,
        typer.Option(help="Directory for the sampled local-Qwen smoke artifacts."),
    ] = ARCHIVE_RESULTS_ROOT / "local_qwen_smoke_latest",
    max_tokens: Annotated[
        int,
        typer.Option(help="Override max output tokens for the sampled LLM smoke."),
    ] = 10000,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Reuse successful local-Qwen smoke cases and rerun failed ones."),
    ] = True,
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Run a small real sampled smoke to inspect exact prompts, evidence, hyperparameters, and outputs."""
    execute_smoke_local_qwen_command(
        abms=abms,
        models_root=models_root,
        ingest_root=ingest_root,
        viz_root=viz_root,
        models_path=models_path,
        model_id=model_id,
        output_root=output_root,
        max_tokens=max_tokens,
        resume=resume,
        json_output=json_output,
        discover_abms=discover_configured_abms,
        resolve_model_from_registry=resolve_model_from_registry,
        resolve_model_path=resolve_abm_model_path,
        create_adapter_fn=create_adapter,
        run_local_qwen_sample_smoke_fn=run_local_qwen_sample_smoke,
        load_abm_config_fn=load_abm_config,
    )


@app.command("render-run-viewer")
def render_run_viewer(
    run_root: Annotated[
        Path,
        typer.Option(help="Concrete run directory, or a root containing latest_run.txt."),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(help="Optional explicit output HTML path. Defaults to <run-root>/review.html."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Render a minimalist static HTML reviewer for a case-based run directory."""
    execute_render_run_viewer_command(
        run_root=run_root,
        output_path=output_path,
        json_output=json_output,
    )


@app.command("smoke-summarizers")
def smoke_summarizers(
    source_root: Annotated[
        Path,
        typer.Option(help="Root directory containing the vetted full-case LLM smoke outputs."),
    ] = ARCHIVE_RESULTS_ROOT / "full_case_smoke_latest",
    abm: Annotated[
        list[str] | None,
        typer.Option(
            "--abm",
            help="Optional ABM filter when the source root is a multi-ABM suite run. Repeat for multiple.",
        ),
    ] = None,
    summarizer: Annotated[
        list[str] | None,
        typer.Option(
            "--summarizer",
            help="Summarizer backends to include in addition to the implicit none/full-text path.",
        ),
    ] = None,
    output_root: Annotated[
        Path,
        typer.Option(help="Directory for summarizer smoke artifacts."),
    ] = ARCHIVE_RESULTS_ROOT / "summarizer_smoke_latest",
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Reuse successful summarizer outputs and rerun only failed or missing modes.",
        ),
    ] = True,
    watch: Annotated[
        bool,
        typer.Option(
            "--watch/--no-watch",
            help="Keep polling the source root and summarize accepted bundles as soon as they appear.",
        ),
    ] = False,
    poll_interval_seconds: Annotated[
        float,
        typer.Option(help="Polling interval used with --watch while waiting for new accepted bundles."),
    ] = 5.0,
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Run the summarization stack over one hand-vetted full-case LLM bundle."""
    execute_smoke_summarizers_command(
        source_root=source_root,
        abms=tuple(abm) if abm else None,
        output_root=output_root,
        resume=resume,
        watch=watch,
        poll_interval_seconds=poll_interval_seconds,
        summarizer_modes=_parse_summarizers(summarizer, fallback=DEFAULT_SUMMARIZERS) if summarizer else None,
        json_output=json_output,
        run_summarizer_smoke_fn=run_summarizer_smoke,
    )


@app.command("smoke-quantitative")
def smoke_quantitative(
    source_root: Annotated[
        Path,
        typer.Option(help="Root directory containing a completed summarizer smoke run or single-LLM quantitative run."),
    ] = ARCHIVE_RESULTS_ROOT / "summarizer_smoke_latest",
    output_root: Annotated[
        Path,
        typer.Option(help="Directory for quantitative smoke artifacts."),
    ] = ARCHIVE_RESULTS_ROOT / "quantitative_smoke_latest",
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Reuse valid scored records and rerun only failed or missing quantitative rows.",
        ),
    ] = True,
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Score completed summarizer outputs or reuse quantitative rows and build DOE analysis tables."""
    execute_smoke_quantitative_command(
        source_root=source_root,
        output_root=output_root,
        resume=resume,
        json_output=json_output,
        run_quantitative_smoke_fn=run_quantitative_smoke,
    )


@app.command("smoke-quantitative-multi-llm")
def smoke_quantitative_multi_llm(
    source_root: Annotated[
        list[Path],
        typer.Option(
            "--source-root",
            exists=True,
            file_okay=False,
            dir_okay=True,
            help=(
                "Completed single-LLM quantitative smoke roots or summarizer smoke roots to merge. "
                "Repeat for one root per LLM."
            ),
        ),
    ],
    output_root: Annotated[
        Path,
        typer.Option(help="Directory for multi-LLM quantitative smoke artifacts."),
    ] = ARCHIVE_RESULTS_ROOT / "quantitative_smoke_multi_llm_latest",
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Reuse valid quantitative rows from the latest run."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Score completed single-LLM quantitative or summarizer smokes jointly and add LLM as a factor."""
    execute_smoke_quantitative_multi_llm_command(
        source_roots=tuple(source_root),
        output_root=output_root,
        resume=resume,
        json_output=json_output,
        run_quantitative_smoke_multi_llm_fn=run_quantitative_smoke_multi_llm,
    )


@app.command("smoke-optimization-gemini-chain")
def smoke_optimization_gemini_chain(
    models_root: Annotated[
        Path,
        typer.Option(help="Root directory containing ABM model files for asset discovery."),
    ] = Path("data"),
    netlogo_home: Annotated[
        str,
        typer.Option(
            envvar="DISTILL_ABM_NETLOGO_HOME",
            help="NetLogo installation directory used by pynetlogo. Can also be provided via DISTILL_ABM_NETLOGO_HOME.",
        ),
    ] = "",
    prompts_path: Annotated[
        Path,
        typer.Option(exists=True),
    ] = Path("configs/prompts.yaml"),
    models_path: Annotated[
        Path,
        typer.Option(exists=True, help="Model registry YAML path."),
    ] = Path("configs/models.yaml"),
    output_root: Annotated[
        Path,
        typer.Option(help="Root directory for the Gemini optimization chain."),
    ] = Path("results/optimisation/gemini-3.1-pro-preview_optimization_all_abms_chain"),
    max_tokens: Annotated[
        int,
        typer.Option(help="Maximum output token budget for each call in the full-case suite smoke."),
    ] = 32768,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Reuse successful stage outputs and rerun only failed or missing work.",
        ),
    ] = True,
) -> None:
    """Run the fixed-factor Gemini optimization chain across the standard six smoke stages."""
    execute_smoke_optimization_gemini_chain_command(
        models_root=models_root,
        netlogo_home=netlogo_home,
        prompts_path=prompts_path,
        models_path=models_path,
        output_root=output_root,
        evidence_modes=("plot",),
        prompt_variants=("example",),
        repetitions=(1, 2, 3),
        summarization_modes=("bart", "bert", "t5"),
        model_id="gemini_3_1_pro_preview",
        max_tokens=max_tokens,
        resume=resume,
        execute_smoke_ingest_command_fn=execute_smoke_ingest_command,
        execute_smoke_viz_command_fn=execute_smoke_viz_command,
        execute_smoke_doe_command_fn=execute_smoke_doe_command,
        execute_smoke_full_case_suite_command_fn=execute_smoke_full_case_suite_command,
        execute_smoke_summarizers_command_fn=execute_smoke_summarizers_command,
        execute_smoke_quantitative_command_fn=execute_smoke_quantitative_command,
        run_ingest_smoke_suite_fn=run_ingest_smoke_suite,
        resolve_viz_smoke_specs_fn=_resolve_viz_smoke_specs,
        run_viz_smoke_suite_fn=run_viz_smoke_suite,
        discover_abms_fn=discover_configured_abms,
        resolve_model_from_registry_fn=resolve_model_from_registry,
        resolve_model_path_fn=lambda abm, models_root: resolve_abm_model_path(abm=abm, models_root=models_root),
        run_doe_smoke_suite_fn=run_doe_smoke_suite,
        load_abm_config_fn=load_abm_config,
        load_prompts_config_fn=load_prompts_config,
        create_adapter_fn=create_adapter,
        run_full_case_suite_smoke_fn=run_full_case_suite_smoke,
        validate_model_policy_fn=_validate_model_policy,
        run_summarizer_smoke_fn=run_summarizer_smoke,
        run_quantitative_smoke_fn=run_quantitative_smoke,
    )


@app.command("sync-results-bucket")
def sync_results_bucket(
    source_root: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="Local results directory to mirror to the bucket.",
        ),
    ] = Path("results"),
    bucket_uri: Annotated[
        str,
        typer.Option(help="Destination Hugging Face bucket URI."),
    ] = "hf://buckets/NoeFlandre/distill-abms-results",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--apply", help="Preview the sync instead of executing it."),
    ] = False,
    delete: Annotated[
        bool,
        typer.Option("--delete/--no-delete", help="Delete remote files missing from the local results tree."),
    ] = True,
    allow_empty_source: Annotated[
        bool,
        typer.Option(
            "--allow-empty-source/--no-allow-empty-source",
            help="Allow an apply sync even when the local results tree has no syncable result files after exclusions.",
        ),
    ] = False,
    plan_path: Annotated[
        Path | None,
        typer.Option(help="Optional JSONL path for saving a dry-run sync plan."),
    ] = None,
    token_env_var: Annotated[
        str,
        typer.Option(help="Environment variable to read a Hugging Face token from, if needed."),
    ] = "HF_TOKEN",
    json_output: Annotated[bool, typer.Option("--json", help="Print structured JSON to stdout.")] = False,
) -> None:
    """Mirror the local results tree to the Hugging Face results bucket with one command."""
    execute_sync_results_bucket_command(
        source_root=source_root,
        bucket_uri=bucket_uri,
        dry_run=dry_run,
        delete=delete,
        allow_empty_source=allow_empty_source,
        plan_path=plan_path,
        token_env_var=token_env_var,
        json_output=json_output,
    )


@app.command("study-exploitation-factors")
def study_exploitation_factors(
    source_root: Annotated[
        Path,
        typer.Option(
            "--source-root",
            exists=True,
            file_okay=False,
            dir_okay=True,
            help=(
                "One completed quantitative smoke root or concrete run directory to analyze."
            ),
        ),
    ],
    output_root: Annotated[
        Path,
        typer.Option(help="Directory for the side-study artifacts."),
    ] = Path("results/archive/side_studies/exploitation_factor_followup"),
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Run a side-study over existing quantitative artifacts to clarify factor behavior for exploitation."""
    result = run_exploitation_factor_study(
        source_root=source_root,
        output_root=output_root,
    )
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"run_root: {result.run_root}")
    typer.echo(f"report_json: {result.report_json_path}")
    typer.echo(f"report_markdown: {result.report_markdown_path}")


@app.command("study-llm-same-settings")
def study_llm_same_settings(
    anchor_source_root: Annotated[
        Path,
        typer.Option(
            "--anchor-source-root",
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="Completed anchor quantitative smoke root or concrete run directory to use as the settings anchor.",
        ),
    ],
    comparison_source_roots: Annotated[
        list[Path],
        typer.Option(
            "--comparison-source-root",
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="Completed comparison quantitative smoke roots or concrete run directories.",
        ),
    ],
    output_root: Annotated[
        Path,
        typer.Option(help="Directory for the same-settings LLM comparison artifacts."),
    ] = Path("results/side_studies/optimization_same_settings_llm_comparison"),
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Compare optimization-phase LLM runs on one shared same-settings slice."""
    result = run_llm_same_settings_study(
        anchor_source_root=anchor_source_root,
        comparison_source_roots=comparison_source_roots,
        output_root=output_root,
    )
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"run_root: {result.run_root}")
    typer.echo(f"report_json: {result.report_json_path}")
    typer.echo(f"report_markdown: {result.report_markdown_path}")


@app.command("smoke-full-case")
def smoke_full_case(
    abm: Annotated[str, typer.Option(help="ABM config name in configs/abms/<name>.yaml")] = "grazing",
    models_root: Annotated[
        Path,
        typer.Option(help="Root directory containing ABM model files for asset discovery."),
    ] = Path("data"),
    ingest_root: Annotated[
        Path,
        typer.Option(help="Root directory containing ingest smoke outputs."),
    ] = ARCHIVE_RESULTS_ROOT / "ingest_smoke_latest",
    viz_root: Annotated[
        Path,
        typer.Option(help="Root directory containing visualization smoke outputs."),
    ] = ARCHIVE_RESULTS_ROOT / "viz_smoke_latest",
    models_path: Annotated[
        Path,
        typer.Option(exists=True, help="Model registry YAML path."),
    ] = Path("configs/models.yaml"),
    model_id: Annotated[
        str,
        typer.Option(help="OpenRouter model alias from configs/models.yaml."),
    ] = "nemotron_nano_12b_v2_vl_free",
    output_root: Annotated[
        Path,
        typer.Option(help="Directory for full-case smoke artifacts."),
    ] = ARCHIVE_RESULTS_ROOT / "full_case_smoke_latest",
    evidence_mode: Annotated[EvidenceMode, typer.Option()] = "table",
    prompt_variant: Annotated[
        str,
        typer.Option(
            help=(
                "Prompt variant: none, role, insights, example, role+example, role+insights, "
                "insights+example, all_three."
            )
        ),
    ] = "role",
    max_tokens: Annotated[
        int,
        typer.Option(help="Maximum output token budget for each call in the full-case smoke."),
    ] = 32768,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Reuse accepted context/trend artifacts and rerun only failed ones."),
    ] = True,
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Run one real Nemotron case through one context prompt and all trend prompts for that ABM."""
    execute_smoke_full_case_command(
        abm=abm,
        models_root=models_root,
        ingest_root=ingest_root,
        viz_root=viz_root,
        models_path=models_path,
        model_id=model_id,
        output_root=output_root,
        evidence_mode=evidence_mode,
        prompt_variant=prompt_variant,
        max_tokens=max_tokens,
        resume=resume,
        json_output=json_output,
        resolve_model_from_registry=resolve_model_from_registry,
        resolve_model_path=resolve_abm_model_path,
        create_adapter_fn=create_adapter,
        run_full_case_smoke_fn=run_full_case_smoke,
        load_abm_config_fn=load_abm_config,
    )


@app.command("smoke-full-case-matrix")
def smoke_full_case_matrix(
    abm: Annotated[str, typer.Option(help="ABM config name in configs/abms/<name>.yaml")] = "grazing",
    models_root: Annotated[
        Path,
        typer.Option(help="Root directory containing ABM model files for asset discovery."),
    ] = Path("data"),
    ingest_root: Annotated[
        Path,
        typer.Option(help="Root directory containing ingest smoke outputs."),
    ] = ARCHIVE_RESULTS_ROOT / "ingest_smoke_latest",
    viz_root: Annotated[
        Path,
        typer.Option(help="Root directory containing visualization smoke outputs."),
    ] = ARCHIVE_RESULTS_ROOT / "viz_smoke_latest",
    models_path: Annotated[
        Path,
        typer.Option(exists=True, help="Model registry YAML path."),
    ] = Path("configs/models.yaml"),
    model_id: Annotated[
        str,
        typer.Option(help="OpenRouter model alias from configs/models.yaml."),
    ] = "nemotron_nano_12b_v2_vl_free",
    output_root: Annotated[
        Path,
        typer.Option(help="Directory for full-case matrix smoke artifacts."),
    ] = ARCHIVE_RESULTS_ROOT / "nemotron_abm_smoke_latest",
    evidence_mode: Annotated[
        list[str] | None,
        typer.Option(
            "--evidence-mode",
            help=(
                "Evidence modes to include. Repeat for multiple. "
                f"Defaults to {', '.join(DEFAULT_FULL_CASE_MATRIX_EVIDENCE_MODES)}."
            ),
        ),
    ] = None,
    prompt_variant: Annotated[
        list[str] | None,
        typer.Option(
            "--prompt-variant",
            help=(
                "Prompt variants to include. Repeat for multiple. "
                f"Defaults to {', '.join(DEFAULT_FULL_CASE_MATRIX_PROMPT_VARIANTS)}."
            ),
        ),
    ] = None,
    repetition: Annotated[
        list[int] | None,
        typer.Option(
            "--repetition",
            help=(
                "Repetitions to include. Repeat for multiple. "
                f"Defaults to {', '.join(str(item) for item in DEFAULT_FULL_CASE_MATRIX_REPETITIONS)}."
            ),
        ),
    ] = None,
    max_tokens: Annotated[
        int,
        typer.Option(help="Maximum output token budget for each call in the full-case matrix smoke."),
    ] = 32768,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Reuse accepted cases and rerun only failed or missing ones."),
    ] = True,
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Run one ABM across evidence, prompt, and repetition combinations with one context plus all trends per case."""
    execute_smoke_full_case_matrix_command(
        abm=abm,
        models_root=models_root,
        ingest_root=ingest_root,
        viz_root=viz_root,
        models_path=models_path,
        model_id=model_id,
        output_root=output_root,
        evidence_modes=resolve_full_case_matrix_evidence_modes(evidence_mode),
        prompt_variants=resolve_full_case_matrix_prompt_variants(prompt_variant),
        repetitions=resolve_full_case_matrix_repetitions(repetition),
        max_tokens=max_tokens,
        resume=resume,
        json_output=json_output,
        resolve_model_from_registry=resolve_model_from_registry,
        resolve_model_path=resolve_abm_model_path,
        create_adapter_fn=create_adapter,
        load_abm_config_fn=load_abm_config,
        run_full_case_matrix_smoke_fn=run_full_case_matrix_smoke,
    )


@app.command("smoke-full-case-suite")
def smoke_full_case_suite(
    models_root: Annotated[
        Path,
        typer.Option(help="Root directory containing ABM model files for asset discovery."),
    ] = Path("data"),
    ingest_root: Annotated[
        Path,
        typer.Option(help="Root directory containing ingest smoke outputs."),
    ] = ARCHIVE_RESULTS_ROOT / "ingest_smoke_latest",
    viz_root: Annotated[
        Path,
        typer.Option(help="Root directory containing visualization smoke outputs."),
    ] = ARCHIVE_RESULTS_ROOT / "viz_smoke_latest",
    models_path: Annotated[
        Path,
        typer.Option(exists=True, help="Model registry YAML path."),
    ] = Path("configs/models.yaml"),
    model_id: Annotated[
        str,
        typer.Option(help="Model registry alias to use for the real-inference suite."),
    ] = "mistral_medium_debug",
    output_root: Annotated[
        Path,
        typer.Option(help="Directory for full-case suite smoke artifacts."),
    ] = ARCHIVE_RESULTS_ROOT / "full_case_suite_smoke_latest",
    evidence_mode: Annotated[
        list[str] | None,
        typer.Option(
            "--evidence-mode",
            help=(
                "Evidence modes to include. Repeat for multiple. "
                f"Defaults to {', '.join(DEFAULT_FULL_CASE_MATRIX_EVIDENCE_MODES)}."
            ),
        ),
    ] = None,
    prompt_variant: Annotated[
        list[str] | None,
        typer.Option(
            "--prompt-variant",
            help=(
                "Prompt variants to include. Repeat for multiple. "
                f"Defaults to {', '.join(DEFAULT_FULL_CASE_MATRIX_PROMPT_VARIANTS)}."
            ),
        ),
    ] = None,
    repetition: Annotated[
        list[int] | None,
        typer.Option(
            "--repetition",
            help=(
                "Repetitions to include. Repeat for multiple. "
                f"Defaults to {', '.join(str(item) for item in DEFAULT_FULL_CASE_MATRIX_REPETITIONS)}."
            ),
        ),
    ] = None,
    max_tokens: Annotated[
        int,
        typer.Option(help="Maximum output token budget for each call in the full-case suite smoke."),
    ] = 32768,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Reuse accepted cases and rerun only failed or missing ones."),
    ] = True,
    allow_debug_model: Annotated[
        bool,
        typer.Option("--allow-debug-model/--no-allow-debug-model", help="Allow a debug-only model id for this smoke."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Run the full inference smoke across all ABMs with one context plus all trends per case."""
    execute_smoke_full_case_suite_command(
        models_root=models_root,
        ingest_root=ingest_root,
        viz_root=viz_root,
        models_path=models_path,
        model_id=model_id,
        output_root=output_root,
        evidence_modes=resolve_full_case_matrix_evidence_modes(evidence_mode),
        prompt_variants=resolve_full_case_matrix_prompt_variants(prompt_variant),
        repetitions=resolve_full_case_matrix_repetitions(repetition),
        max_tokens=max_tokens,
        resume=resume,
        json_output=json_output,
        allow_debug_model=allow_debug_model,
        resolve_model_from_registry=resolve_model_from_registry,
        resolve_model_path=lambda abm, models_root: resolve_abm_model_path(abm=abm, models_root=models_root),
        create_adapter_fn=create_adapter,
        load_abm_config_fn=load_abm_config,
        validate_model_policy=_validate_model_policy,
        run_full_case_suite_smoke_fn=run_full_case_suite_smoke,
    )


@app.command("monitor-local-qwen")
def monitor_local_qwen(
    output_root: Annotated[
        Path,
        typer.Option(help="Directory containing local-Qwen smoke artifacts."),
    ] = ARCHIVE_RESULTS_ROOT / "local_qwen_smoke_latest",
    watch: Annotated[
        bool,
        typer.Option(help="Continuously refresh the dashboard until the run reaches a terminal state."),
    ] = False,
    interval_seconds: Annotated[
        float,
        typer.Option("--interval-seconds", help="Polling interval for --watch mode."),
    ] = 2.0,
    exit_when_terminal: Annotated[
        bool,
        typer.Option(
            "--exit-when-terminal/--stay-open",
            help="In --watch mode, exit automatically on terminal state or keep the dashboard open.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print structured JSON instead of the dashboard."),
    ] = False,
) -> None:
    """Show a compact live dashboard for the local-Qwen smoke run."""
    execute_monitor_local_qwen_command(
        output_root=output_root,
        watch=watch,
        interval_seconds=interval_seconds,
        exit_when_terminal=exit_when_terminal,
        json_output=json_output,
    )


@app.command("monitor-run")
def monitor_run(
    output_root: Annotated[
        Path,
        typer.Option(help="Directory containing a case-based smoke run or a latest_run.txt pointer."),
    ] = ARCHIVE_RESULTS_ROOT / "nemotron_abm_smoke_latest",
    watch: Annotated[
        bool,
        typer.Option(help="Continuously refresh the dashboard until the run reaches a terminal state."),
    ] = False,
    interval_seconds: Annotated[
        float,
        typer.Option("--interval-seconds", help="Polling interval for --watch mode."),
    ] = 2.0,
    exit_when_terminal: Annotated[
        bool,
        typer.Option(
            "--exit-when-terminal/--stay-open",
            help="In --watch mode, exit automatically on terminal state or keep the dashboard open.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print structured JSON instead of the dashboard."),
    ] = False,
) -> None:
    """Show a compact live dashboard for any case-based smoke run."""
    execute_monitor_local_qwen_command(
        output_root=output_root,
        watch=watch,
        interval_seconds=interval_seconds,
        exit_when_terminal=exit_when_terminal,
        json_output=json_output,
    )


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
    ] = ARCHIVE_RESULTS_ROOT / "ingest_smoke",
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
    execute_smoke_ingest_command(
        abms=abms,
        models_root=models_root,
        output_root=output_root,
        stage=stage,
        require_stage=require_stage,
        json_output=json_output,
        run_ingest_smoke_suite_fn=run_ingest_smoke_suite,
    )


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
    ] = ARCHIVE_RESULTS_ROOT / "viz_smoke_latest",
    json_output: Annotated[bool, typer.Option("--json", help="Print a structured JSON result to stdout.")] = False,
) -> None:
    """Run NetLogo simulations and generate the ordered plot PNGs used before LLM inference."""
    execute_smoke_viz_command(
        abms=abms,
        models_root=models_root,
        netlogo_home=netlogo_home,
        stage=stage,
        require_stage=require_stage,
        output_root=output_root,
        json_output=json_output,
        resolve_viz_smoke_specs=_resolve_viz_smoke_specs,
        run_viz_smoke_suite_fn=run_viz_smoke_suite,
    )


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
    ] = ARCHIVE_RESULTS_ROOT / "agent_validation/latest",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the full validation report JSON to stdout."),
    ] = False,
) -> None:
    """Run the canonical non-LLM validation suite for coding-agent verification."""
    execute_validate_workspace_command(
        checks=checks,
        abms=abms,
        models_root=models_root,
        ingest_stage=ingest_stage,
        profile=profile,
        output_root=output_root,
        json_output=json_output,
        run_validation_suite_fn=run_validation_suite,
    )


@app.command("quality-gate")
def quality_gate(
    scope: Annotated[
        QualityGateScope,
        typer.Option(help="Convenience validation scope: static, pre-llm, or full."),
    ] = "full",
    checks: Annotated[
        list[str] | None,
        typer.Option(
            "--check",
            help="Optional validation check filter. Overrides the scope default when provided.",
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
        ValidationProfile | None,
        typer.Option(help="Optional explicit validation profile override."),
    ] = None,
    output_root: Annotated[
        Path,
        typer.Option(help="Directory for structured validation reports and nested artifacts."),
    ] = ARCHIVE_RESULTS_ROOT / "agent_validation/latest",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the full validation report JSON to stdout."),
    ] = False,
) -> None:
    """Run the canonical validation command with a scope-oriented convenience wrapper."""
    selection = resolve_quality_gate_selection(
        scope=scope,
        explicit_checks=checks,
        explicit_profile=profile,
    )
    execute_validate_workspace_command(
        checks=selection.checks,
        abms=abms,
        models_root=models_root,
        ingest_stage=ingest_stage,
        profile=selection.profile,
        output_root=output_root,
        json_output=json_output,
        run_validation_suite_fn=run_validation_suite,
    )


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
    execute_describe_abm_command(
        abm=abm,
        models_root=models_root,
        json_output=json_output,
        resolve_scoring_reference_path=_resolve_scoring_reference_path,
        resolve_additional_scoring_reference_paths=_resolve_additional_scoring_reference_paths,
        load_abm_config_fn=load_abm_config,
    )


@app.command("describe-ingest-artifacts")
def describe_ingest_artifacts(
    root: Annotated[Path, typer.Option(..., exists=True, file_okay=False, dir_okay=True)],
    json_output: Annotated[bool, typer.Option("--json", help="Print structured JSON to stdout.")] = False,
) -> None:
    """Describe an existing ingest artifact directory without rerunning ingestion."""
    execute_describe_ingest_artifacts_command(root=root, json_output=json_output)


@app.command("describe-run")
def describe_run(
    output_dir: Annotated[Path, typer.Option(..., exists=True, file_okay=False, dir_okay=True)],
    json_output: Annotated[bool, typer.Option("--json", help="Print structured JSON to stdout.")] = False,
) -> None:
    """Describe an existing run output directory from its metadata without rerunning the pipeline."""
    execute_describe_run_command(output_dir=output_dir, json_output=json_output)


@app.command("health-check")
def health_check(
    models_root: Annotated[
        Path,
        typer.Option(help="Root directory containing ABM model files for asset discovery."),
    ] = Path("data"),
    ingest_root: Annotated[
        Path,
        typer.Option(help="Expected ingest smoke output root."),
    ] = ARCHIVE_RESULTS_ROOT / "ingest_smoke_latest",
    viz_root: Annotated[
        Path,
        typer.Option(help="Expected visualization smoke output root."),
    ] = ARCHIVE_RESULTS_ROOT / "viz_smoke_latest",
    models_path: Annotated[
        Path,
        typer.Option(exists=True, help="Model registry YAML path."),
    ] = Path("configs/models.yaml"),
    json_output: Annotated[bool, typer.Option("--json", help="Print structured JSON to stdout.")] = False,
) -> None:
    """Run lightweight operator health checks without executing the pipeline."""
    execute_health_check_command(
        models_root=models_root,
        ingest_root=ingest_root,
        viz_root=viz_root,
        json_output=json_output,
        discover_abms=discover_configured_abms,
        resolve_model_path=resolve_abm_model_path,
        resolve_model_from_registry=resolve_model_from_registry,
        models_path=models_path,
        load_abm_config_fn=load_abm_config,
    )


def main() -> None:
    """Entrypoint callable used by setuptools/uv script wiring."""
    app()


def _validate_model_policy(provider: str, model: str, allow_debug_model: bool) -> None:
    validate_benchmark_model_policy(
        provider=provider,
        model=model,
        allow_debug_model=allow_debug_model,
        benchmark_models=BENCHMARK_MODELS,
    )


if __name__ == "__main__":
    main()
