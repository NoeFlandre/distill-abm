"""Typer CLI entrypoint for distill-abm production workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from distill_abm.configs.loader import load_abm_config, load_prompts_config
from distill_abm.eval.doe_full import analyze_factorial_anova
from distill_abm.llm.factory import create_adapter
from distill_abm.pipeline.run import PipelineInputs, run_pipeline

app = typer.Typer(help="Run ABM distillation workflows without notebooks.")


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
    skip_summarization: Annotated[
        bool,
        typer.Option(help="Skip BART/BERT summarization and keep the full LLM report text."),
    ] = False,
    abm: Annotated[str | None, typer.Option(help="ABM config name in configs/abms/<name>.yaml")] = None,
) -> None:
    """Runs one end-to-end pipeline execution from CSV to scored report."""
    prompts = load_prompts_config(prompts_path)
    if abm:
        abm_config = load_abm_config(Path("configs/abms") / f"{abm}.yaml")
        metric_pattern = abm_config.metric_pattern
        metric_description = abm_config.metric_description
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
            skip_summarization=skip_summarization,
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


def main() -> None:
    """Preserves setuptools/uv script compatibility with explicit callable."""
    app()


if __name__ == "__main__":
    main()
