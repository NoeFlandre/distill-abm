# distill-abm

Paper-aligned ABM distillation pipeline with reproducible experiment orchestration.

## What This Repository Does
`distill-abm` implements the workflow described in `data/paper/Main.txt`:

1. Build ABM context from parameters + documentation.
2. Generate trend narratives from simulation evidence.
3. Optionally summarize trend narratives.
4. Score outputs with BLEU, METEOR, ROUGE-1/2/L, and Flesch.
5. Run DOE/ANOVA analysis over experiment outputs.
6. Produce publication-oriented summary tables from completed smoke runs.

## Benchmark Model Policy
Benchmark runs are restricted to:

1. `moonshotai/kimi-k2.5` via OpenRouter
2. `google/gemini-3.1-pro-preview` via OpenRouter
3. `qwen/qwen3.5-27b` via OpenRouter

The CLI enforces this policy.

Debug-only model note:
- `nemotron_nano_12b_v2_vl_free` is available for smoke/debug work only.
- It is not part of the benchmark model policy.
- Use it only with `--allow-debug-model`.
- `mistral_medium_debug` is available for smoke/debug work only.
- It is not part of the benchmark model policy.
- Use it only with `--allow-debug-model` and `MISTRAL_API_KEY` set.
- Mistral debug requests use `temperature=0.2`; other providers remain at `1.0` unless explicitly overridden.
- The all-ABMs generation audit run is exposed through `smoke-full-case-suite`.
- `smoke-full-case-suite` writes a stable suite root with:
  - `suite_progress.json`
  - one nested matrix run per ABM under `abms/<abm>/runs/`
  - one stable `abms/<abm>/current/` view with the latest report, CSV, and log for that ABM
- Mistral suite execution is paced and worker-limited specifically for its API budget, so the scheduler stays below the provider request ceiling instead of overscheduling pointlessly.

## Summarizers
First-class summarizers:

1. `bart`
2. `bert`
3. `t5`
4. `longformer_ext`

## Core Ablation Axes
1. Evidence mode: `plot`, `table`, `plot+table`
   - `table` means a statistical evidence dump computed from the plot-relevant simulation series only, not a raw CSV dump
   - when a reporter pattern matches many repeated-simulation series, the detailed signal analysis is computed on their tick-wise mean so the audit path stays robust and reviewable
2. Text source mode: `summary_only`, `full_text_only`

## Repository Layout

```text
configs/
  abms/                      ABM presets
  models.yaml                canonical model registry
  runtime_defaults.yaml      runtime defaults (modes, summarizers, requests)
  experiment_settings.yaml   ABM ground-truth mapping
  prompts.yaml               prompt templates + style factors
  ground_truth/              author + modeler references for lexical scoring
  prompt_assets/             qualitative prompt examples

src/distill_abm/
  cli.py                     Typer entrypoint
  pipeline/                  run + sweep + smoke orchestration
  llm/                       provider adapters and factory
  summarize/                 summarizer runners + text cleanup
  eval/                      lexical metrics + DOE analysis
  ingest/                    CSV and NetLogo preprocessing
  viz/                       plotting and stats table generation

docs/
  ARCHITECTURE.md
  WALKTHROUGH.md
  HYPERPARAMETERS.md
  TESTING_REPORT.md
  TRACEABILITY_MATRIX.md
  DECISION_LOG.md

tests/
  e2e/
  integration/
  unit/
```

Recent internal refactors preserved the public smoke/monitor contracts while splitting a few dense modules into narrower helpers:
- `pipeline/local_qwen_monitor.py` now keeps the TUI/rendering surface, while `pipeline/local_qwen_monitor_snapshots.py` owns sampled/full-case/tuning/suite snapshot collection.
- `pipeline/full_case_suite_progress.py` owns the stable suite progress/current-view contract used by `smoke-full-case-suite`.
- `pipeline/full_case_review_csv.py` owns the shared per-case full-case review CSV writer used by both single-case and matrix smoke runs.
- `run_viewer_payloads.py` owns typed payload construction for the static `review.html` viewer.

## Quick Start

```bash
uv sync --frozen --extra dev
```

Run one pipeline execution:

```bash
uv run distill-abm run \
  --csv-path data/samples/sim.csv \
  --parameters-path data/samples/params.txt \
  --documentation-path data/samples/docs.txt \
  --model-id kimi_k2_5 \
  --evidence-mode plot+table \
  --text-source-mode summary_only \
  --summarizer bart --summarizer bert --summarizer t5 --summarizer longformer_ext
```

Run debug smoke suite:

```bash
uv run distill-abm smoke-qwen \
  --csv-path data/samples/sim.csv \
  --parameters-path data/samples/params.txt \
  --documentation-path data/samples/docs.txt \
  --allow-debug-model
```

Run granular NetLogo ingestion smoke checks:

```bash
uv run distill-abm smoke-ingest-netlogo \
  --models-root data \
  --stage documentation \
  --stage final-documentation
```

Run granular visualization smoke checks:

```bash
uv run distill-abm smoke-viz \
  --models-root data \
  --netlogo-home /path/to/NetLogo \
  --stage simulation-csv \
  --stage plot-1 \
  --json
```

Run pre-LLM DOE smoke checks:

```bash
uv run distill-abm smoke-doe \
  --ingest-root results/ingest_smoke_latest \
  --viz-root results/viz_smoke_latest \
  --json
```

Run a small real local-Qwen smoke for prompt/evidence/output inspection:

```bash
uv run distill-abm smoke-local-qwen \
  --ingest-root results/ingest_smoke_latest \
  --viz-root results/viz_smoke_latest \
  --json
```

Run a lightweight repository health check:

```bash
uv run distill-abm health-check \
  --models-root data \
  --ingest-root results/ingest_smoke_latest \
  --viz-root results/viz_smoke_latest \
  --json
```

Monitor a case-based smoke run live:

```bash
uv run distill-abm monitor-local-qwen --watch --interval-seconds 2

uv run distill-abm monitor-run \
  --output-root results/nemotron_abm_smoke_latest \
  --watch --interval-seconds 2
```

DOE smoke notes:

- `smoke-doe` does not call any LLM.
- It materializes the full pre-LLM DOE matrix across ABMs, benchmark models, evidence modes, summarization conditions, prompt variants, and repetitions.
- It treats candidate models as design factors only, so local model availability does not affect DOE smoke success or failure.
- It groups shared DOE factors under `results/doe_smoke_latest/10_shared/global/`, shared ABM artifacts under `results/doe_smoke_latest/10_shared/<abm>/`, and compact case/request indexes under `results/doe_smoke_latest/20_case_index/`.
- It writes the exact context prompt, the exact trend prompt for each plot, the per-request model and hyperparameter settings, the exact image/table evidence paths, and the unresolved context placeholder that would still exist before the first LLM call.
- Table evidence is generated from the matched plot series only and includes descriptive statistics, extrema, inflection points, rolling Mann-Kendall, change points, and oscillation summaries.
- It uses the latest ingest and visualization smoke outputs as the default pre-LLM inputs, so the report makes it easy to catch wrong documentation, wrong parameter narrative, wrong simulation CSV, wrong evidence image, wrong prompt composition, wrong model choice, or placeholder leakage before any model execution.
- It writes `design_matrix.csv`, `request_matrix.csv`, `cases.jsonl`, `requests.jsonl`, and a grouped markdown/json report under `results/doe_smoke_latest/`.

Visualization smoke caveat:

- Unlike the older CSV-driven debug path, `smoke-viz` now resolves ABM-specific NetLogo settings from `configs/abms/<name>.yaml`.
- Each ABM must define a `netlogo_viz` section in `configs/abms/<name>.yaml` with the experiment name, reporter list, and ordered plot list.
- The command writes one simulation CSV plus ordered plot PNGs under `results/viz_smoke_latest/<abm>/`.
- The repository now preserves validated reference CSVs and plot images under `data/<abm>_abm/legacy/` so production smoke runs can emit deterministic debug artifacts without depending on the temporary notebook folder.
- The report records the artifact source per ABM as either `simulated` or `fallback`.
- The workflow depends on `pynetlogo` and a working NetLogo installation directory passed via `--netlogo-home` or `DISTILL_ABM_NETLOGO_HOME`.
- The NetLogo execution path remains available and repo-local for all benchmark models; milk-specific input CSVs and grazing include files are now stored in the project data directories instead of external notebook paths.

NetLogo ingestion caveat:

- These artifacts are extracted dynamically from the provided `.nlogo` files; they are not static repository fixtures.
- The extraction logic is designed for standard NetLogo model structure and for the model patterns used in this repository.
- It is expected to work for these benchmark models and similar NetLogo files, but it is not guaranteed to work unchanged for every arbitrary `.nlogo` file.
- In particular, documentation extraction assumes a recognizable NetLogo info section or usable top-of-file comments, and parameter extraction assumes interface/experiment sections in formats the parser already supports.

Run the preferred non-LLM validation suite for coding-agent verification:

```bash
uv run distill-abm validate-workspace --json
```

Scope-oriented convenience wrapper:

```bash
uv run distill-abm quality-gate --scope pre-llm --json
```

This command is the current repository entrypoint for non-LLM local verification. It runs the standard local checks, emits a structured JSON report, and nests the NetLogo ingest-smoke report under `results/agent_validation/latest/` so agents can inspect outcomes without ad hoc artifact hunting. As with the testing report, treat the command output from the present workspace as authoritative rather than relying on a stale claimed status in documentation.

Repo workflow notes:

- See `docs/AGENT_WORKFLOW.md` for the repository-specific agent workflow, verification order, and evidence expectations.
- See `docs/AGENT_BACKLOG.md` for the current prioritized backlog of agent-friendly quality improvements that remain after the present refactors.
- See `docs/MANUAL_VALIDATION.md` for the current evidence-style manual verification record covering the pre-LLM workflow surfaces.

Agent-oriented CLI additions:

- Most verification-oriented commands now support `--json` so agents can consume structured output instead of parsing human text.
- `quality-gate` is a thin convenience wrapper around `validate-workspace` that maps a declared change scope to the appropriate validation profile and default checks:
  - `static` -> `ruff` + `mypy`
  - `pre-llm` -> quick validation profile
  - `full` -> default validation profile
- `validate-workspace` supports profiles:
  - `quick`: fast static + ingest verification
  - `default`: full local verification
  - `full`: currently equivalent to `default`, reserved as the strictest profile
- `smoke-ingest-netlogo` supports `--require-stage` so callers can assert that specific stage checks are present.
- `smoke-viz` provides stage-filtering and `--require-stage` for the generated simulation CSV and each ordered plot image.
- `smoke-doe` provides a structured pre-LLM view of the full DOE matrix and writes grouped shared artifacts plus compact case/request indexes that can be reviewed without opening thousands of files.
- `smoke-ingest-netlogo`, `smoke-viz`, and `smoke-doe` now follow the same audit contract as the later smokes: each invocation writes into `runs/run_<timestamp>/`, updates `latest_run.txt`, and emits a qualitative root `run.log.jsonl`.
- `smoke-local-qwen` is the legacy command name for the sampled real-inference smoke. It now runs through the configured API model and writes one self-contained folder per sampled case plus a review CSV with exact prompt text, evidence paths, hyperparameters, outputs, and a minimalist static `review.html` viewer.
- `smoke-local-qwen` supports `--resume` and reuses only successful case artifacts; failed or incomplete cases are rerun.
- `smoke-full-case-matrix` runs one ABM across evidence modes, prompt variants, and repetitions, with one context prompt plus all ordered trend prompts per case. It uses the same run separation, resume behavior, `run.log.jsonl`, and `review.html` reviewer surface as the sampled smoke.
- `render-run-viewer` builds the same minimalist static HTML viewer for any existing case-based run directory, including both one-trend sampled smokes and full-case multi-trend runs, or for the latest run when you point it at a root containing `latest_run.txt`.
- `monitor-local-qwen` and `monitor-run` render the same compact live dashboard for case-based smoke runs, including current case or trial, configured `num_ctx`, `max_tokens`, prompt lengths, and observed token usage.
- For all-ABM suite runs, `monitor-run` now reads the suite root directly and uses `suite_progress.json` plus the nested ABM run state so you can monitor the whole run from one place in the terminal.
- `health-check` performs a lightweight read-only validation of configured ABMs, model-registry resolution, and expected ingest/viz artifact roots.
- `ingest-netlogo` and `ingest-netlogo-suite` now write stable artifact manifests.
- Read-only inspection commands are available for agent loops:
  - `uv run distill-abm describe-abm --abm grazing --json`
  - `uv run distill-abm describe-ingest-artifacts --root results/ingest/grazing --json`
  - `uv run distill-abm describe-run --output-dir results/pipeline --json`

Run DOE analysis:

```bash
uv run distill-abm analyze-doe --input-csv results/sweep/combinations_report.csv
```

## Reproducibility Guarantees
Each run writes `pipeline_run_metadata.json` with:

1. Prompt signatures and lengths
2. Model/provider/request defaults
3. Summarizer configuration and enablement
4. Input artifact paths and hashes
5. Run signature for resumable execution
6. Score source provenance (`context_response` or human ground truth file)
7. Additional reference-score blocks when an ABM has optional secondary references

Each run also writes `debug_trace/` with:

1. Snapshotted input files used for the run
2. Request/response JSON for context and trend LLM calls
3. Summarization trace showing the selected text source and outputs
4. Artifact manifests with hashes, sizes, and previews
5. Validation warnings for placeholder-like inputs and missing metric columns

## Verification Commands

```bash
uv run pytest
uv run ruff check .
uv run mypy src tests
uv build
```
