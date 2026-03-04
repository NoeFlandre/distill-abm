# End-to-End Walkthrough

This document follows one full run from raw simulation inputs to scored report output.

## 1) Overview

The system turns NetLogo simulation outputs into text-and-metric summaries using a deterministic pipeline:

1. ingest simulation inputs
2. generate trend artifacts (plots + statistics)
3. build context and trend prompts
4. query one or more LLM providers
5. optionally summarize with BART/BERT (or keep full text)
6. post-process text into final report fields
7. compute score metrics

The pipeline supports both:

- **reference mode** (`--plot` / `--stats-markdown` / `--stats-image` / `plot+stats`)
- **full-text mode** (`--summarization-mode full` or `--skip-summarization`)
- **summary mode** (`--summarization-mode summary`)
- **dual mode** (`--summarization-mode both`, stores both trend text forms)

## 2) Data Ingestion

### Inputs required by the pipeline

For a standard ABM run, provide:

- simulation CSV output (reduced values)
- NetLogo parameters text file
- NetLogo documentation text file

### Ingestion call path

`run_pipeline`/`run_pipeline_sweep` in `distill_abm.pipeline.run` reads:

- `csv_path`: metric trajectories over tick time
- `parameters_path`: default or chosen experiment values
- `documentation_path`: model documentation narrative

### Reference ingestion flow

The module `distill_abm.ingest.netlogo_notebook_workflow` re-implements the reference workflow logic:

- parses ABM metadata from CSV headers and configured reporters
- builds a consistent ordering of simulation files and metric columns
- supports multiple source layouts so older and newer simulation naming can both be loaded
- prepares outputs used downstream by prompts and plotting

## 3) Plotting

### Plot generation

`distill_abm.viz` generates trend figures from the requested metric.

For each metric, `run_pipeline` creates:

- a PNG line chart in `output_dir` (default `results/pipeline`)
- an optional stats table (mean/std/min/max/median per time step)
- an optional stats table image (for multimodal evidence)

### Plot output controls

- `run_pipeline` accepts `metric_pattern` and `metric_description`.
- the plot filename derives from the metric pattern and timestamp context.
- each image is saved deterministically so downstream caches can reuse the same path.

## 4) Context generation

### Prompt construction

Context prompts are built in `distill_abm.pipeline.run` from:

- the configured `context_prompt` template
- extracted model metadata and documentation
- optional `style_features.role`

### Provider call

`llm.factory.create_adapter` resolves the configured provider/model pair. The context phase request goes through:

- adapter request schema from `distill_abm.llm.models`
- structured request with deterministic prompt formatting
- optional truncation and token controls from provider config

## 5) Trend analysis

### Prompt construction

Trend prompts include:

- trend prompt template
- optional style features (`role`, `example`, `insights`)
- plot/asset context text if available
- selected evidence mode (`plot`, `stats-markdown`, `stats-image`, `plot+stats`)

### Evidence modes

- `plot` : sends chart image bytes + prompt instructions.
- `stats-markdown` : sends markdown stats table inline in prompt text.
- `stats-image` : sends generated stats table image.
- `plot+stats` : sends both plot and markdown stats in one combined evidence request.

### Multi-image trend phase

`run_pipeline_sweep` sends trend prompts over multiple plot images by constructing a `combination` row per prompt/image pair, matching compatibility sweep behavior.

## 6) Summarization

### Default behavior

By default (`summarization_mode=both`, `score_on=both`), pipeline summarization and scoring computes both paths:

- BART summarization pass
- BERT summarization pass on grouped plot analyses
- Additional optional helpers are exposed via `distill_abm.summarize.models`:
  - `summarize_with_t5`
  - `summarize_with_longformer_ext`

You can force full-text usage with `--summarization-mode full` (or legacy `--skip-summarization`) or capture both text versions with `--summarization-mode both`.

### Dual-path and scoring modes

`run_pipeline` accepts `score_on` to choose scoring source:

- `score_on=full`: score the raw trend text.
- `score_on=summary`: score summarized trend text.
- `score_on=both`: compute both and write both sets into `report.csv` as extended columns.

Use `score_on=both` with `summarization-mode both` for a full comparison run where trend response, score columns, and report schema include:

- `trend_full_response`
- `trend_summary_response`
- `full_*` and `summary_*` score prefixes in `report.csv`

### Summary batch behavior

`distill_abm.summarize.summarize_csv_batch` handles:

- per-row summary columns
- min/max length constraints for each stage
- batched grouping for long plot responses

## 7) Post-processing

Post-processing always runs before writing final outputs:

- remove markdown symbols
- clean citation-like and URL-like fragments
- normalize whitespace and punctuation
- strip artifacts from model side effects

Functions are in `distill_abm.summarize.postprocess` and `distill_abm.summarize.legacy`.

## 8) Scoring

Scoring modules are in `distill_abm.eval`:

- token overlap and `SummaryScores` in `distill_abm.eval.metrics`
- lexical/quality style metrics in `distill_abm.eval.legacy_scores`
- qualitative parser/extractor helpers in `distill_abm.eval.qualitative`

## 9) Prompt sweeps

`run_pipeline_sweep` is the explicit parity API for combination runs:

- builds all combinations through `build_style_feature_combinations`
- writes wide sweep output with `write_combinations_csv`
- supports context/trend adapter split and separate model selection
- supports append/update behavior for resumed runs (`resume_existing=True`)

Column naming can match historical reference output using `csv_column_style="plot"`.

## 10) CLI usage

Primary commands exposed by `distill_abm.cli`:

- `run` : executes one run from CSV/params/docs
- `analyze-doe` : generates ANOVA/DoE contributions
- `evaluate-qualitative` : evaluates coverage or faithfulness (1-5)

### Qualitative output contract

`evaluate-qualitative` returns JSON:

- `score` (integer 1-5)
- `reasoning` (short explanation)
- `model` (provider model name)

## 11) Configuration and runtime behavior

Configuration files:

- `configs/prompts.yaml` : editable prompt templates and style features
- `configs/models.yaml` : provider aliases and model defaults
- `configs/notebook_prompt_reference.yaml` : source templates for reference parity
- `configs/notebook_experiment_settings.yaml` : captured reporter/run settings per ABM
- `configs/abms/*.yaml` : model-specific defaults used by `--abm`

Compatibility behavior:

- `distill_abm.compat` exposes historical names and optional reference-first call paths
- if reference call fails or is not required, deterministic production implementations are used
- archive evidence and audit files validate this behavior (`docs/archive_full_manifest.md`, `docs/runtime_notebook_dependencies.json`)

## 12) Reproducibility Artifacts

Every `run` invocation writes `pipeline_run_metadata.json` in the same output directory as `report.csv` and the generated plot.

The metadata contains:

- absolute and relative input paths used for that run
- resolved LLM provider/model and request hyperparameters
- context and trend prompt signatures (SHA-256) plus prompt lengths
- evidence/summarization mode settings (`evidence_mode`, `summarization_mode`, `score_on`)
- score summaries and output artifact locations

To replay a result:

1. Keep `pipeline_run_metadata.json` with the corresponding outputs.
2. Reuse the same:
   - `csv_path`, `parameters_path`, `documentation_path`
   - `metric_pattern`, `metric_description`
   - `--evidence-mode`, `--summarization-mode`, `--score-on`
   - `plot_description` and prompt templates in `configs/prompts.yaml`
3. Run the same command with the same provider and model.

## Summary

You can trace every production output from:

1. inputs (`csv_path`, `parameters_path`, `documentation_path`)
2. preprocessing (`ingest`)
3. trends (`viz`)
4. context and trend LLM calls (`llm` + `pipeline`)
5. optional summarization (`summarize`)
6. text normalization (`summarize`/`postprocess`)
7. report writing (`pipeline`)
8. scoring (`eval`)
9. parity validation (`docs/` + `tests/`)
