# distill-abm

Production Python package for ABM-to-LLM distillation with reference-parity validation.

## Repository layout

- `src/distill_abm/` - production code
- `tests/` - unit, integration, CLI, and parity tests
- `configs/` - runtime defaults for prompts, ABMs, evaluation, and logging
- `configs/notebook_prompt_reference.yaml` - reference templates extracted from scripts
- `configs/notebook_experiment_settings.yaml` - captured experiment defaults for ingestion and summarization
- `archive/legacy_repo/` - preserved reference implementation snapshot (scripts + artifacts)
- `tests/fixtures/notebook_parity/` - canonical mirrored fixtures used for migration coverage
- `docs/` - architecture, parity, and audit evidence

## Codebase organization

```text
distill-abm/
├── configs/
│   ├── models.yaml
│   ├── prompts.yaml
│   ├── evaluation.yaml
│   ├── logging.yaml
│   ├── notebook_prompt_reference.yaml
│   ├── notebook_experiment_settings.yaml
│   └── notebook_prompt_assets/
├── src/distill_abm/
│   ├── compat/             # compatibility surface for historical interfaces
│   ├── configs/            # typed config loading and models
│   ├── eval/               # metric scoring and qualitative parsing
│   ├── ingest/             # NetLogo and CSV ingestion
│   ├── llm/                # provider adapters and request factory
│   ├── pipeline/           # end-to-end orchestration
│   ├── summarize/          # text cleanup and summarization
│   └── viz/                # plot and stats-table rendering
├── docs/                  # architecture + parity evidence
├── tests/
└── archive/legacy_repo/   # retained reference scripts
```

Compatibility imports are preserved as `distill_abm.compat` shims for backward compatibility.

The top-level runtime entrypoint is `src/distill_abm/cli.py`. The canonical compatibility surface is `src/distill_abm/compat`, with fallback dispatch implemented by `src/distill_abm/compat/compat_callables.py` and reference loader `src/distill_abm/compat/reference_loader.py`.

## Quickstart

```bash
uv sync --extra dev
uv run pre-commit install
uv run pre-commit run --all-files
uv run ruff check .
uv run black --check .
uv run mypy src tests
uv run pytest --cov=distill_abm --cov-report=term-missing --cov-fail-under=85
```

## Deployment

1. Install dependencies:

```bash
uv sync --extra dev
```

2. Configure provider credentials in your environment (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) for remote adapters.

3. Run either local commands or containerized workflow:

```bash
docker build -t distill-abm .
docker run --rm -v \"$(pwd):/app\" distill-abm distill-abm run ...
```

4. Keep each output directory plus `pipeline_run_metadata.json` in immutable storage for full replay.

## CLI

Run the core pipeline:

```bash
uv run distill-abm run \
  --csv-path path/to/reduced.csv \
  --parameters-path path/to/params.txt \
  --documentation-path path/to/docs.txt \
  --abm milk_consumption \
  --provider openai \
  --model gpt-4o
```

Override trend description text:

```bash
uv run distill-abm run ... --plot-description "The attachment shows the livestock count over time."
```

Run evidence ablations:

```bash
uv run distill-abm run ... --evidence-mode plot
uv run distill-abm run ... --evidence-mode stats-markdown
uv run distill-abm run ... --evidence-mode stats-image
uv run distill-abm run ... --evidence-mode plot+stats
```

Run full-text baseline or summary-based runs:

```bash
uv run distill-abm run \
  --csv-path path/to/reduced.csv \
  --parameters-path path/to/params.txt \
  --documentation-path path/to/docs.txt \
  --evidence-mode plot+stats \
  --skip-summarization
```

Generate both full-text and summary artifacts and score on both for side-by-side comparison:

```bash
uv run distill-abm run \
  --csv-path path/to/reduced.csv \
  --parameters-path path/to/params.txt \
  --documentation-path path/to/docs.txt \
  --summarization-mode both \
  --score-on both
```

Run reference-style combination sweeps in code (`role`, `example`, `insights` matrix):

```python
from pathlib import Path

from distill_abm.configs.loader import load_prompts_config
from distill_abm.llm.factory import create_adapter
from distill_abm.pipeline.run import PipelineInputs, run_pipeline_sweep

prompts = load_prompts_config(Path("configs/prompts.yaml"))
adapter = create_adapter(provider="openai", model="gpt-4o")

output_csv = run_pipeline_sweep(
    inputs=PipelineInputs(
        csv_path=Path("results/reduced.csv"),
        parameters_path=Path("params.txt"),
        documentation_path=Path("documentation.txt"),
        output_dir=Path("results/sweep"),
        metric_pattern="mean-incum",
        metric_description="weekly milk trend",
        model="gpt-4o",
    ),
    prompts=prompts,
    adapter=adapter,
    image_paths=[Path("results/plots/p1.png"), Path("results/plots/p2.png")],
    plot_descriptions=["Plot 1 description", "Plot 2 description"],
)

print(output_csv)
```

For multi-provider workflows:

- `context_adapter` / `trend_adapter`
- `context_model` / `trend_model`
- `csv_column_style="plot"` for wide output matching historic schemas
- `resume_existing=True` to continue an existing combinations CSV

## Qualitative evaluation

```bash
uv run distill-abm evaluate-qualitative \
  --summary-text "Generated ABM summary..." \
  --source-text "ABM source context..." \
  --metric coverage \
  --provider openai \
  --model gpt-4o

uv run distill-abm evaluate-qualitative \
  --summary-text "Generated ABM summary..." \
  --source-text "ABM source context..." \
  --metric faithfulness \
  --source-image-path results/pipeline/mean-incum.png \
  --provider openai \
  --model gpt-4o
```

Expected output schema:

```json
{
  "score": 4,
  "reasoning": "Coverage is strong. Core dynamics and regime changes are represented.",
  "model": "gpt-4o"
}
```

## DoE analysis

```bash
uv run distill-abm analyze-doe \
  --input-csv path/to/FinalResultsYesNo.csv \
  --output-csv results/doe/anova_factorial_contributions.csv
```

## Docker

```bash
docker build -t distill-abm .
docker run --rm distill-abm
```

## CI

GitHub Actions configuration is at `.github/workflows/ci.yml` and runs:

- `pre-commit run --all-files`
- `ruff check .`
- `black --check .`
- `mypy src tests`
- `pytest --cov=distill_abm --cov-report=term-missing --cov-fail-under=85`

## Evidence and parity

- `docs/WALKTHROUGH.md`: reviewer-oriented end-to-end behavior.
- `docs/ARCHITECTURE.md`: module boundaries and data flow.
- `docs/PARITY.md`: parity policy and proof by test.
- `docs/archive_full_manifest.json`: migration and retention policy for every archived file.
- `docs/runtime_notebook_dependencies.json`: reference dispatch dependencies for audit.
- `configs/notebook_prompt_reference.yaml`: reference prompt texts for behavior lock.

### Reference artifacts

- `archive/legacy_repo/` retains source reference scripts and artifacts.
- `tests/fixtures/notebook_parity/` stores byte-equivalent mirrors used by migration audits.
- `archive/` is retained for auditability and does not drive runtime execution.

Compatibility details used by parity:

- `tests/regression/test_reference_equivalence.py` covers core runtime parity.
- `tests/regression/test_reference_function_coverage.py` validates function accounting.
- `tests/regression/test_runtime_reference_dependencies.py` validates dispatch source map integrity.

## Known limits

- Reference loader execution is AST-restricted and intentionally does not execute full side-effectful reference cells.
- `return_csv` and `return_csv_2` are protected by fallback behavior if reference execution is unavailable.
- Default request temperature is `0.5`.

### Reproducibility manifest

- `pipeline_run_metadata.json` is emitted for each `run` execution.
- It captures input artifact paths, prompt signatures, provider settings (`temperature`, `max_tokens`), and score outputs.
- The metadata file is intended to be versioned together with `report.csv` and generated plot files.

## License

See repository license file.
