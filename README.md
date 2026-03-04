# distill-abm

Production Python package for ABM-to-LLM distillation, report generation, and evaluation.

## Repository layout

- `src/distill_abm/` - production code
- `tests/` - unit, integration, and end-to-end tests
- `configs/` - runtime defaults for prompts, ABMs, evaluation, and logging
- `configs/runtime_defaults.yaml` - centralized defaults for model/provider/modes and generation parameters
- `configs/notebook_prompt_reference.yaml` - frozen reference prompt templates (filename retained for compatibility)
- `configs/notebook_experiment_settings.yaml` - frozen reference experiment settings (filename retained for compatibility)
- `archive/reference_repo/` - preserved reference implementation snapshot (scripts + artifacts)
- `tests/fixtures/notebook_parity/` - mirrored reference fixtures used for reproducibility checks
- `docs/` - architecture, walkthrough, hyperparameters, and testing evidence

## Codebase organization

```text
distill-abm/
├── configs/
│   ├── models.yaml
│   ├── runtime_defaults.yaml
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
└── archive/reference_repo/   # retained reference scripts
```

Compatibility imports are preserved as `distill_abm.compat` shims for stable interfaces.

The top-level runtime entrypoint is `src/distill_abm/cli.py`. The compatibility surface is `src/distill_abm/compat`, with fallback dispatch implemented by `src/distill_abm/compat/compat_callables.py` and source loader `src/distill_abm/compat/reference_loader.py`.

## Documentation map

- `docs/WALKTHROUGH.md` - end-to-end flow, step by step
- `docs/ARCHITECTURE.md` - module-by-module organization and runtime data flow
- `docs/HYPERPARAMETERS.md` - complete list of runtime hyperparameter values
- `docs/TESTING_REPORT.md` - complete testing methodology and gate criteria

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

2. Configure provider credentials in your environment (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`) for remote adapters.

3. Run either local commands or containerized workflow:

```bash
docker build -t distill-abm .
docker run --rm -v \"$(pwd):/app\" distill-abm distill-abm run ...
```

4. Keep each output directory plus `pipeline_run_metadata.json` in immutable storage for full replay.

## Hyperparameters

All hyperparameter values used by the runtime are documented in:

- `docs/HYPERPARAMETERS.md`

This includes:

- LLM request defaults (`temperature`, `max_tokens`, retry/backoff)
- summarizer backend parameters (BART, BERT, T5, Longformer-like)
- evidence/summarization/scoring mode defaults
- DoE and ingestion defaults
- quality gate thresholds

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

When `--abm` is provided, lexical metrics (BLEU/METEOR/ROUGE/Flesch) are computed against the ABM human-reference text from `configs/notebook_ground_truth/*.txt` instead of the generated context text.

Override trend description text:

```bash
uv run distill-abm run ... --plot-description "The attachment shows the livestock count over time."
```

Run evidence ablations:

```bash
uv run distill-abm run ... --evidence-mode plot
uv run distill-abm run ... --evidence-mode table-csv
uv run distill-abm run ... --evidence-mode plot+table
```

Run a full smoke suite (provider/model selectable):

```bash
uv run distill-abm smoke-qwen \
  --csv-path path/to/reduced.csv \
  --parameters-path path/to/params.txt \
  --documentation-path path/to/docs.txt \
  --provider ollama \
  --model qwen3.5:0.8b \
  --doe-input-csv path/to/FinalResultsYesNo.csv \
  --metric-pattern mean-incum \
  --metric-description "average weekly whole milk consumption per agent" \
  --plot-description "weekly whole milk consumption trend"
```

Quick smoke (single case, 2 inference calls) for prompt/response debugging:

```bash
uv run distill-abm smoke-qwen \
  --csv-path path/to/reduced.csv \
  --parameters-path path/to/params.txt \
  --documentation-path path/to/docs.txt \
  --provider openrouter \
  --model qwen/qwen3-vl-235b-a22b-thinking \
  --case-id plot-full-full \
  --max-cases 1 \
  --skip-qualitative \
  --skip-sweep
```

`smoke-qwen` is resumable by default (`--resume`): completed successful cases/DoE outputs are reused, and existing sweep combinations are skipped to avoid duplicate experiments.

Smoke outputs are written under `results/smoke_qwen/` by default:

- `smoke_report.md` and `smoke_report.json` with full matrix status and errors
- `cases/<case-id>/` per-mode artifacts (prompts, responses, `report.csv`, `pipeline_run_metadata.json`, `case_manifest.json`)
- `doe/anova_factorial_contributions.csv`
- `sweep/combinations_report.csv`

For debugging, each `pipeline_run_metadata.json` includes full prompts, responses, attached evidence image path, and metric reference provenance (`scores.reference.source/path/signature`).

The default smoke matrix runs these combinations:

- evidence mode: `plot`, `table-csv`, `plot+table`
- summarization/scoring: `full/full`, `summary/summary`, `both/both`

Run full-text baseline or summary-based runs:

```bash
uv run distill-abm run \
  --csv-path path/to/reduced.csv \
  --parameters-path path/to/params.txt \
  --documentation-path path/to/docs.txt \
  --evidence-mode plot+table \
  --skip-summarization
```

Run summary-only output/scoring (no full-text scoring path selected):

```bash
uv run distill-abm run \
  --csv-path path/to/reduced.csv \
  --parameters-path path/to/params.txt \
  --documentation-path path/to/docs.txt \
  --summarization-mode summary \
  --score-on summary
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

Enable additional local summarizers (T5 and/or Longformer-like LED) on top of the default BART+BERT stack:

```bash
uv run distill-abm run \
  --csv-path path/to/reduced.csv \
  --parameters-path path/to/params.txt \
  --documentation-path path/to/docs.txt \
  --additional-summarizer t5 \
  --additional-summarizer longformer_ext
```

Notes:

- default summary stack is BART + BERT
- `--additional-summarizer` is repeatable (`t5`, `longformer_ext`)
- if one backend is unavailable at runtime, pipeline falls back to remaining summarizers without failing the run

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

## Documentation and audit artifacts

- `docs/WALKTHROUGH.md`: reviewer-oriented end-to-end behavior.
- `docs/ARCHITECTURE.md`: module boundaries and data flow.
- `docs/PARITY.md`: compatibility and reproducibility policy.
- `docs/HYPERPARAMETERS.md`: complete hyperparameter values and defaults.
- `docs/TESTING_REPORT.md`: single authoritative testing reference (scope, methods, gates, and outcomes).
- `docs/archive_full_manifest.json`: retention policy for every archived file.
- `docs/runtime_notebook_dependencies.json`: reference dispatch dependency map for audit.
- `configs/notebook_prompt_reference.yaml`: frozen reference prompt texts used for behavior lock.

### Archived artifacts (non-runtime)

- `archive/reference_repo/` retains source reference scripts and artifacts.
- `tests/fixtures/notebook_parity/` stores byte-equivalent mirrors used by replay audits.
- `archive/` is retained for auditability and does not drive runtime execution.

Compatibility checks:

- `tests/regression/test_reference_equivalence.py` covers core runtime equivalence.
- `tests/regression/test_reference_function_coverage.py` validates function accounting.
- `tests/regression/test_runtime_reference_dependencies.py` validates dispatch source map integrity.

## Known limits

- Reference loader execution is AST-restricted and intentionally does not execute full side-effectful reference cells.
- `return_csv` and `return_csv_2` are protected by fallback behavior if reference execution is unavailable.
- Default request temperature is `0.5` (see `docs/HYPERPARAMETERS.md`).

### Reproducibility manifest

- `pipeline_run_metadata.json` is emitted for each `run` execution.
- It captures input artifact paths, full prompt texts and signatures, provider settings (`temperature`, `max_tokens`, `max_retries`, `retry_backoff_seconds`), and score outputs.
- The metadata file is intended to be versioned together with `report.csv` and generated plot files.

## License

See repository license file.
