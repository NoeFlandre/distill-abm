"""Typer CLI entrypoint for distill-abm production workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from distill_abm.configs.loader import load_abm_config, load_prompts_config
from distill_abm.eval.doe_full import analyze_factorial_anova
from distill_abm.eval.qualitative_runner import QualitativeMetric, evaluate_qualitative_score
from distill_abm.llm.factory import create_adapter
from distill_abm.pipeline.run import (
    AdditionalSummarizer,
    EvidenceMode,
    PipelineInputs,
    ScoreMode,
    SummarizationMode,
    run_pipeline,
)

app = typer.Typer(help="Run ABM distillation workflows.")


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
    output_dir: Annotated[Path, typer.Option()] = Path("results/pipeline"),
    provider: Annotated[str, typer.Option()] = "echo",
    model: Annotated[str, typer.Option()] = "echo-model",
    metric_pattern: Annotated[str, typer.Option()] = "mean",
    metric_description: Annotated[str, typer.Option()] = "simulation trend",
    plot_description: Annotated[
        str | None,
        typer.Option(help="Optional evidence description for the plotted metric."),
    ] = None,
    evidence_mode: Annotated[
        EvidenceMode,
        typer.Option(help="Evidence provided to trend analysis: plot, stats-markdown, stats-image, or plot+stats."),
    ] = "plot",
    skip_summarization: Annotated[
        bool,
        typer.Option(help="Skip BART/BERT summarization and keep the full LLM report text."),
    ] = False,
    summarization_mode: Annotated[
        SummarizationMode,
        typer.Option(help="full: keep raw trend text, summary: use summarized trend text, both: store/report both."),
    ] = "both",
    additional_summarizer: Annotated[
        list[str] | None,
        typer.Option(
            "--additional-summarizer",
            help="Optional extra summary backends in addition to BART/BERT. Repeatable: t5, longformer_ext.",
        ),
    ] = None,
    score_on: Annotated[
        ScoreMode,
        typer.Option(
            help="Which text should be used for scoring: full, summary, or both. "
            "Both adds both score sets to report output."
        ),
    ] = "both",
    abm: Annotated[str | None, typer.Option(help="ABM config name in configs/abms/<name>.yaml")] = None,
) -> None:
    """Runs one end-to-end pipeline execution from CSV to scored report."""
    prompts = load_prompts_config(prompts_path)
    if abm:
        abm_config = load_abm_config(Path("configs/abms") / f"{abm}.yaml")
        metric_pattern = abm_config.metric_pattern
        metric_description = abm_config.metric_description
        if plot_description is None and abm_config.plot_descriptions:
            plot_description = abm_config.plot_descriptions[0]
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
            skip_summarization=skip_summarization,
            summarization_mode=summarization_mode,
            additional_summarizers=_parse_additional_summarizers(additional_summarizer),
            score_on=score_on,
        ),
        prompts=prompts,
        adapter=adapter,
    )
    typer.echo(f"plot: {result.plot_path}")
    typer.echo(f"report: {result.report_csv}")


@app.command("analyze-doe")
def analyze_doe(
    input_csv: Annotated[Path, typer.Option(..., exists=True, file_okay=True, dir_okay=False)],
    output_csv: Annotated[Path, typer.Option()] = Path("results/doe/anova_factorial_contributions.csv"),
    max_interaction_order: Annotated[int, typer.Option()] = 2,
) -> None:
    """Runs full factorial ANOVA contribution analysis."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result = analyze_factorial_anova(input_csv, output_csv, max_interaction_order=max_interaction_order)
    if result is None:
        raise typer.Exit(code=1)
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
    provider: Annotated[str, typer.Option()] = "echo",
    model: Annotated[str, typer.Option()] = "echo-model",
) -> None:
    """Evaluates coverage or faithfulness with an LLM and returns JSON output."""
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


def main() -> None:
    """Preserves setuptools/uv script compatibility with explicit callable."""
    app()


def _parse_additional_summarizers(values: list[str] | None) -> tuple[AdditionalSummarizer, ...]:
    allowed = {"t5", "longformer_ext"}
    normalized = tuple(dict.fromkeys(values or []))
    invalid = [value for value in normalized if value not in allowed]
    if invalid:
        raise typer.BadParameter(
            f"unsupported additional summarizer(s): {', '.join(invalid)}. Allowed: t5, longformer_ext."
        )
    return cast(tuple[AdditionalSummarizer, ...], normalized)


if __name__ == "__main__":
    main()
