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
