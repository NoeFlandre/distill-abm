# distill-abm

Paper-aligned ABM distillation pipeline with reproducible experiment orchestration.

## What This Repository Does
`distill-abm` implements the workflow described in `data/paper/Main.txt`:

1. Build ABM context from parameters + documentation.
2. Generate trend narratives from simulation evidence.
3. Optionally summarize trend narratives.
4. Score outputs with BLEU, METEOR, ROUGE-1/2/L, and Flesch.
5. Run DOE/ANOVA analysis over experiment outputs.

## Benchmark Model Policy
Benchmark runs are restricted to:

1. `moonshotai/kimi-k2.5` via OpenRouter
2. `google/gemini-3.1-pro-preview` via OpenRouter
3. `qwen3.5:0.8b` via local Ollama

Debug-only model:

1. `qwen/qwen3-vl-235b-a22b-thinking` via OpenRouter

The CLI enforces this policy.

## Summarizers
First-class summarizers:

1. `bart`
2. `bert`
3. `t5`
4. `longformer_ext`

## Core Ablation Axes
1. Evidence mode: `plot`, `table`, `plot+table`
2. Text source mode: `summary_only`, `full_text_only`

## Repository Layout

```text
configs/
  abms/                      ABM presets
  models.yaml                canonical model registry
  runtime_defaults.yaml      runtime defaults (modes, summarizers, requests)
  experiment_settings.yaml   ABM ground-truth mapping
  prompts.yaml               prompt templates + style factors
  ground_truth/              human references for lexical scoring
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

DOE smoke notes:

- `smoke-doe` does not call any LLM.
- It materializes the full pre-LLM DOE matrix across ABMs, benchmark models, evidence modes, summarization conditions, prompt variants, and repetitions.
- It treats candidate models as design factors only, so local model availability does not affect DOE smoke success or failure.
- It groups shared DOE factors under `results/doe_smoke_latest/10_shared/global/`, shared ABM artifacts under `results/doe_smoke_latest/10_shared/<abm>/`, and compact case/request indexes under `results/doe_smoke_latest/20_case_index/`.
- It writes the exact context prompt, the exact trend prompt for each plot, the per-request model and hyperparameter settings, the exact image/table evidence paths, and the unresolved context placeholder that would still exist before the first LLM call.
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

Run the canonical non-LLM validation suite for coding-agent verification:

```bash
uv run distill-abm validate-workspace --json
```

This command is intended to be the default "verify my work" entrypoint for coding agents. It runs the repository's standard local checks, emits a structured JSON report, and nests the NetLogo ingest-smoke report under `results/agent_validation/latest/` so agents can validate outcomes without ad hoc artifact hunting.

Repo workflow notes:

- See `docs/AGENT_WORKFLOW.md` for the repository-specific agent workflow, verification order, and evidence expectations.
- See `docs/MANUAL_VALIDATION.md` for the current evidence-style manual verification record covering the pre-LLM workflow surfaces.

Agent-oriented CLI additions:

- Most verification-oriented commands now support `--json` so agents can consume structured output instead of parsing human text.
- `validate-workspace` supports profiles:
  - `quick`: fast static + ingest verification
  - `default`: full local verification
  - `full`: currently equivalent to `default`, reserved as the strictest profile
- `smoke-ingest-netlogo` supports `--require-stage` so callers can assert that specific stage checks are present.
- `smoke-viz` provides stage-filtering and `--require-stage` for the generated simulation CSV and each ordered plot image.
- `smoke-doe` provides a structured pre-LLM view of the full DOE matrix and writes grouped shared artifacts plus compact case/request indexes that can be reviewed without opening thousands of files.
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
