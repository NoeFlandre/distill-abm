# Architecture

## Runtime Flow
1. `distill_abm.cli` parses commands and validates model policy.
2. `pipeline.run.run_pipeline` orchestrates context generation, trend generation, optional summarization, scoring, and metadata.
3. `pipeline.smoke.run_qwen_smoke_suite` executes matrix smoke validation over evidence/text-source modes.
4. `eval.metrics` computes lexical metrics; `eval.doe_full` computes ANOVA/factor contributions.

## Module Responsibilities

### `src/distill_abm/cli.py`
- Command entrypoint for run, smoke, qualitative scoring, DOE analysis.
- Exposes `validate-workspace` as the canonical non-LLM verification contract for coding agents.
- Benchmark/debug model gating.
- Model registry resolution via `configs/models.yaml`.

### `src/distill_abm/agent_validation.py`
- Canonical local validation orchestration for agents.
- Runs pytest, Ruff, mypy, build, and NetLogo ingest smoke checks behind one structured report.
- Writes stable machine-readable and markdown reports for post-run inspection.

### `src/distill_abm/pipeline/run.py`
- End-to-end run orchestration.
- Prompt composition and evidence handling (`plot`, `table`, `plot+table`).
- Text-source selection (`summary_only`, `full_text_only`).
- Reproducibility metadata and resumable run signatures.

### `src/distill_abm/pipeline/smoke.py`
- Smoke matrix execution and per-case artifacts.
- Prompt/response bundle exports.
- Optional qualitative checks and DOE/sweep integration.

### `src/distill_abm/llm/*`
- Provider-neutral adapter interface.
- Provider-specific adapters for OpenRouter, Ollama, OpenAI, Anthropic, Janus, Echo.

### `src/distill_abm/summarize/*`
- Summarizer runners: BART, BERT, T5, LongformerExt.
- Text normalization and postprocessing helpers.

### `src/distill_abm/eval/*`
- Lexical metric computation and batch scoring.
- DOE/ANOVA utilities.

### `src/distill_abm/ingest/*`
- CSV ingestion and NetLogo preprocessing artifacts.
- NetLogo ingestion is dynamic: documentation, parameters, narratives, and code are extracted from the supplied `.nlogo` files at runtime.
- The implementation targets the NetLogo structures used by the benchmark ABMs in this repository and similar models.
- It is not a universal parser for every possible `.nlogo` variant; models with materially different info blocks, interface declarations, or experiment layouts may require extractor updates.

### `src/distill_abm/viz/*`
- Plot generation and stats table construction.

## Configuration
- `configs/models.yaml`: canonical model aliases and provider routing.
- `configs/runtime_defaults.yaml`: default modes, summarizers, and request settings.
- `configs/experiment_settings.yaml`: ABM-to-ground-truth mapping.
- `configs/prompts.yaml`: prompt templates and style features.
- `configs/abms/*.yaml`: ABM presets.

## Artifact Contracts
Each run writes:
- `plot_*.png`
- `stats_table.csv`
- `report.csv`
- `pipeline_run_metadata.json`
