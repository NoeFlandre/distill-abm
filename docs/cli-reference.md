# CLI reference

The public operational interface is the `distill-abm` Typer CLI. This page groups the commands by purpose; the CLI remains the source for the complete option list and current defaults.

```bash
uv run distill-abm --help
uv run distill-abm <command> --help
```

Many commands support `--json` for structured output. Paths default to repository-relative locations, and most commands expose an output-root option for isolation.

## Core pipeline

### `run`

Run one end-to-end CSV-to-report pipeline. Required inputs are `--csv-path`, `--parameters-path`, and `--documentation-path`.

```bash
uv run distill-abm run \
  --csv-path /path/to/simulation.csv \
  --parameters-path /path/to/parameters.txt \
  --documentation-path /path/to/documentation.txt \
  --model-id kimi_k2_5 \
  --abm fauna
```

Important factors include `--evidence-mode` (`plot`, `table`, or `plot+table`), `--text-source-mode` (`summary_only` or `full_text_only`), repeatable `--summarizer`, and `--output-dir`. Debug models require explicit `--allow-debug-model` consent.

### `ingest-netlogo` and `ingest-netlogo-suite`

Extract model artifacts for one model or for multiple configured ABMs. These commands are useful before visualization or inference and write to `results/ingest` by default.

## Evidence and evaluation

| Command | Use |
| --- | --- |
| `smoke-ingest-netlogo` | Audit configured ingestion stages before LLM calls. |
| `smoke-viz` | Run visualization smoke checks and produce ordered plot artifacts. |
| `smoke-doe` | Materialize the pre-LLM design and request-review artifacts. |
| `analyze-doe` | Compute factorial ANOVA contribution tables from an input CSV. |
| `evaluate-qualitative` | Ask a configured model to score coverage or faithfulness. |
| `smoke-summarizers` | Exercise summarizers over a validated full-case bundle. |
| `smoke-quantitative` | Score completed quantitative or summarizer outputs and build tables. |
| `smoke-quantitative-multi-llm` | Compare completed quantitative outputs with LLM as a factor. |

Use these stages to isolate input, evidence, summarization, and scoring failures instead of diagnosing everything through a full inference run.

## Inference smoke workflows

| Command | Use |
| --- | --- |
| `smoke-qwen` | Run a broad debug smoke over evidence/text modes and optional DOE/sweep steps. |
| `smoke-local-qwen` | Run a small sampled real-inference smoke with detailed prompts and traces. |
| `smoke-full-case` | Run one full case with one context and all trends for an ABM. |
| `smoke-full-case-matrix` | Run combinations of evidence, prompt variants, and repetitions for one ABM. |
| `smoke-full-case-suite` | Run the full-case suite across configured ABMs. |
| `smoke-optimization-gemini-chain` | Run the fixed-factor optimization chain across its standard stages. |

These workflows can call external providers, create many artifacts, and take substantially longer than unit tests. Inspect `--help` before selecting models, output roots, resume behavior, or case filters.

## Studies and operations

| Command | Use |
| --- | --- |
| `study-exploitation-factors` | Analyze existing quantitative artifacts for factor behavior. |
| `study-llm-same-settings` | Compare model runs on a shared same-settings slice. |
| `sync-results-bucket` | Mirror local `results/` to the Hugging Face bucket. Use dry-run first. |
| `validate-workspace` | Run the canonical non-LLM validation suite. |
| `quality-gate` | Select static, pre-LLM, or full validation scopes. |
| `health-check` | Run lightweight operator checks without executing the pipeline. |
| `describe-abm` | Inspect one resolved ABM configuration and its local assets. |
| `describe-ingest-artifacts` | Inspect an existing ingestion artifact directory. |
| `describe-run` | Inspect an existing run from its metadata. |
| `render-run-viewer` | Render a static HTML reviewer for a case-based run. |
| `monitor-run` | Follow a case-based run from filesystem snapshots. |
| `monitor-local-qwen` | Follow the local-Qwen smoke dashboard. |

For result synchronization and deletion safeguards, follow [Results and synchronization](RESULTS_BUCKET.md). For local checks, start with:

```bash
uv run distill-abm quality-gate --scope static --json
```
