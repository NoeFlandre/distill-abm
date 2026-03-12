# Run Execution Order

This walkthrough answers the user-facing question: when you run `distill-abm run` with a chosen model, evidence mode, and summarizer roster, what actually happens?

It focuses on the main `run` command, not the smoke/DOE/full-case variants.

## User Entry Point

Example:

```bash
uv run distill-abm run \
  --csv-path data.csv \
  --parameters-path params.txt \
  --documentation-path docs.txt \
  --model-id qwen3_5_27b \
  --evidence-mode plot+table \
  --text-source-mode summary_only \
  --summarizer bart \
  --summarizer t5 \
  --abm grazing
```

The CLI entrypoint is [`src/distill_abm/cli.py`](../src/distill_abm/cli.py). The `run` command collects the user knobs at [`src/distill_abm/cli.py:143`](../src/distill_abm/cli.py#L143):

- `provider` / `model` or `model_id`
- `metric_pattern`, `metric_description`, `plot_description`
- `evidence_mode`
- `text_source_mode`
- `summarizer`
- `allow_summary_fallback`
- optional `abm`

The command immediately delegates to [`execute_run_command`](../src/distill_abm/cli_actions.py#L80).

## Top-Level Call Chain

From the user's command to the inner pipeline, the order is:

1. `cli.py::run(...)`
2. [`cli_actions.py::execute_run_command(...)`](../src/distill_abm/cli_actions.py#L80)
3. prompt config load from `configs/prompts.yaml`
4. optional model alias resolution from `configs/models.yaml`
5. benchmark/debug model policy validation
6. optional ABM config load from `configs/abms/<abm>.yaml`
7. optional ABM defaulting for metric pattern, metric description, plot description, and scoring references
8. adapter creation for the selected provider/model
9. `PipelineInputs(...)` construction
10. [`run.py::run_pipeline(...)`](../src/distill_abm/pipeline/run.py#L108)

The most important argument plumbing happens in [`src/distill_abm/cli_actions.py:104`](../src/distill_abm/cli_actions.py#L104) through [`src/distill_abm/cli_actions.py:147`](../src/distill_abm/cli_actions.py#L147):

- `model_id` can override manual `provider` + `model`
- `abm` can override metric defaults and attach scoring references
- `summarizer` values are normalized into the configured summarizer tuple
- the final resolved values are packed into `PipelineInputs`

## What `run_pipeline` Does

Once inside [`run_pipeline`](../src/distill_abm/pipeline/run.py#L108), the execution order is:

1. Create `output_dir`.
2. Build a run signature and optionally resume from an existing matching run.
3. Load the simulation CSV.
4. Find metric columns that match `metric_pattern`.
5. Generate the primary trend plot PNG.
6. Resolve the requested evidence mode.
7. Compute statistical evidence for table-style prompts.
8. Write `stats_table.csv`.
9. Build the context prompt from parameters, documentation, and optional style features.
10. Make the first LLM call to generate context.
11. Build the trend prompt using:
    - `metric_description`
    - generated context
    - optional `plot_description`
    - optional style features
    - optional table evidence text
12. Attach the plot image when evidence mode includes plot evidence.
13. Make the second LLM call to generate the trend narrative.
14. Resolve the scoring reference:
    - ABM ground truth if `--abm` was supplied
    - otherwise the generated context response
15. Run the configured summarizers over the trend text unless `text_source_mode=full_text_only`.
16. Decide which text becomes the report/scoring candidate:
    - summary text if one exists
    - otherwise full trend text
17. Score the selected text, full text, and optional summary text.
18. Write `report.csv`.
19. Write `pipeline_run_metadata.json` and debug-trace artifacts.
20. Return `PipelineResult`.

The core implementation for that order is visible directly in [`src/distill_abm/pipeline/run.py:110`](../src/distill_abm/pipeline/run.py#L110) through [`src/distill_abm/pipeline/run.py:259`](../src/distill_abm/pipeline/run.py#L259).

## How The Main Knobs Change Behavior

### `--model` / `--model-id`

- Resolved in the CLI layer before the pipeline starts.
- Used to create the provider adapter.
- Passed into `PipelineInputs.model` and then into both LLM requests.

### `--evidence-mode`

Handled in the pipeline at [`src/distill_abm/pipeline/run.py:132`](../src/distill_abm/pipeline/run.py#L132).

- `plot`: attach the generated plot image, do not rely on table evidence text alone
- `table`: include statistical evidence text, do not attach an image
- `plot+table`: include both

### `--text-source-mode`

Handled when summarization resolves at [`src/distill_abm/pipeline/run.py:171`](../src/distill_abm/pipeline/run.py#L171).

- `full_text_only`: skip summarization and use the full trend response
- `summary_only`: try the summarizers and prefer the summary output

If `--allow-summary-fallback` is enabled, summary-only mode can still fall back to full text when all summarizers fail or return empty content.

### `--summarizer`

Normalized in the CLI layer and stored in `PipelineInputs.summarizers`.

The summarizer roster is consumed inside `_summarize_report_text(...)`, which delegates into the summarizer helpers in [`src/distill_abm/pipeline/helpers.py`](../src/distill_abm/pipeline/helpers.py) and the actual runners in [`src/distill_abm/summarize/models.py`](../src/distill_abm/summarize/models.py).

The configured summarizers are run in order, and non-empty outputs are combined into the final summary text.

### `--abm`

This is the main shortcut for repository presets.

When present, the CLI resolves the ABM config and uses it to:

- default the metric pattern
- default the metric description
- default the plot description
- attach the correct primary scoring reference
- attach optional secondary scoring references

That behavior lives in [`src/distill_abm/cli_actions.py:111`](../src/distill_abm/cli_actions.py#L111) through [`src/distill_abm/cli_actions.py:120`](../src/distill_abm/cli_actions.py#L120).

## Artifacts The User Gets

The main run produces these top-level artifacts in `output_dir`:

- `plot_*.png`
- `stats_table.csv`
- `report.csv`
- `pipeline_run_metadata.json`
- `debug_trace/`

The metadata file is the best place to inspect "what happened" after a run. It records:

- resolved inputs
- selected evidence mode
- selected text source
- prompt text
- raw and summarized responses
- scores
- summarizer configuration
- per-request usage/runtime metadata
- reproducibility signatures

The metadata assembly lives in [`src/distill_abm/pipeline/run_state.py`](../src/distill_abm/pipeline/run_state.py), especially [`src/distill_abm/pipeline/run_state.py:391`](../src/distill_abm/pipeline/run_state.py#L391) through [`src/distill_abm/pipeline/run_state.py:462`](../src/distill_abm/pipeline/run_state.py#L462).

## Condensed Mental Model

For a user, the simplest mental model is:

`CLI args -> resolve presets/defaults -> create adapter -> load CSV -> build plot/table evidence -> context LLM call -> trend LLM call -> optional summarizers -> scoring -> report + metadata`

If someone wants to understand a surprising result, the fastest inspection order is:

1. `pipeline_run_metadata.json`
2. `debug_trace/llm/context_request.json`
3. `debug_trace/llm/trend_request.json`
4. `report.csv`
5. `stats_table.csv`

The metadata file also records the exact trace paths under the debug-trace section, so if the trace layout changes again, `pipeline_run_metadata.json` remains the safest starting point.
