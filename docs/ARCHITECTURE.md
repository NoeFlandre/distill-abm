# Architecture

## End-To-End Flow

1. `distill_abm.cli` parses the command and resolves model and ABM presets.
2. `distill_abm.pipeline.run.run_pipeline` orchestrates evidence generation, context generation, trend generation, optional summarization, scoring, and metadata.
3. `distill_abm.eval` computes lexical metrics and downstream DOE or ANOVA summaries.
4. `results/` stores generated artifacts locally, while the Hugging Face bucket is the durable publication-facing results store.

The manuscript-level workflow uses this runtime in two experimental stages:

1. a screening stage over prompt factors, evidence modes, summarizers, ABMs, and low-cost benchmark LLMs
2. an optimization stage that reuses the retained settings with stronger deployment-oriented LLMs

## Data Flow

The main `run` workflow follows this order:

1. load the simulation CSV and ABM-specific defaults
2. generate the plot artifact
3. derive statistical evidence for `table` or `plot+table` modes
4. build the context prompt from parameters and documentation
5. call the selected LLM for context generation
6. build the trend prompt from context plus plot and optional table evidence
7. call the selected LLM for trend generation
8. optionally summarize the generated trend text
9. score the selected output against configured references
10. write reports, metadata, and debug traces

`table` evidence means statistical evidence derived from the plot-relevant simulation series, not a raw CSV dump.

At the paper level, the evaluated reference families are author summaries, modeler summaries, GPT-5.2 short summaries, and GPT-5.2 long reports.

## Major Modules

### `src/distill_abm/cli.py`

- CLI entrypoint and public workflow routing
- benchmark-model policy enforcement
- read-only inspection and results-bucket sync commands

### `src/distill_abm/cli_actions.py`

- command-level argument resolution
- model alias resolution from `configs/models.yaml`
- ABM preset application from `configs/abms/*.yaml`

### `src/distill_abm/pipeline/`

- end-to-end run orchestration
- smoke workflows and multi-stage audit runs
- report writing, metadata, and run-separated artifact layout
- DOE-style screening and optimization support for paper-facing benchmark studies

### `src/distill_abm/ingest/`

- CSV ingestion and NetLogo preprocessing
- extraction of ABM documentation, parameters, and code artifacts

### `src/distill_abm/viz/`

- repeated-simulation plotting
- statistical evidence generation used by table-style prompts

### `src/distill_abm/summarize/`

- summarizer backends for `bart`, `bert`, `t5`, and `longformer_ext`
- text normalization and postprocessing helpers

### `src/distill_abm/eval/`

- lexical metrics
- reference scoring
- DOE and factorial analysis
- quantitative outputs used for ANOVA and variance-contribution reporting

### `src/distill_abm/llm/`

- provider adapters
- provider-specific request defaults

## Stable Artifact Surfaces

The main run writes:

- `plot_*.png`
- `stats_table.csv`
- `report.csv`
- `pipeline_run_metadata.json`
- `debug_trace/`

Case-based smoke workflows use run-separated output roots under `runs/run_<timestamp>/` together with `latest_run.txt` and `run.log.jsonl`.

## Configuration Sources

- `configs/models.yaml` - model aliases and provider routing
- `configs/runtime_defaults.yaml` - request defaults and pipeline defaults
- `configs/experiment_settings.yaml` - scoring-reference mappings
- `configs/prompts.yaml` - prompt templates and style features
- `configs/abms/*.yaml` - ABM presets and plotting configuration
