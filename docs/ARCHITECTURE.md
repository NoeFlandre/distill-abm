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
- Exposes read-only `describe-*` commands so agents can inspect ABMs, ingest outputs, and run artifacts without rerunning workflows.
- Exposes `--json` output on the main verification and inspection surfaces.
- Exposes `smoke-viz` for artifact-focused verification of the NetLogo-to-CSV-to-plot workflow that runs before any LLM inference.
  - `smoke-viz` is production fallback-first: it preserves repo-local reference CSVs and plot images under `data/*/legacy/` and records whether each ABM artifact bundle was produced from a live simulation or from fallback reference artifacts.
  - Milk model input CSVs and grazing `.nls` include files are now stored inside the repository so the NetLogo launch path no longer depends on the temporary notebook workspace.
- Exposes `smoke-doe` for pre-LLM inspection of the smoke matrix.
  - `smoke-doe` resolves the full benchmark DOE matrix over ABMs, candidate models, evidence modes, summarization settings, prompt variants, and repetitions.
  - It groups shared ABM artifacts under `results/doe_smoke_latest/shared/<abm>/` and writes case-specific manifests under `results/doe_smoke_latest/cases/...`.
  - It writes the exact context prompt, the exact trend prompt for each plot, per-request hyperparameters, image/table evidence paths, and unresolved context placeholders that define the pre-LLM boundary.
  - The command is intended to debug wrong input artifacts, wrong prompts, wrong model settings, and placeholder leakage before any model call is made.
- Benchmark/debug model gating.
- Model registry resolution via `configs/models.yaml`.

### `src/distill_abm/agent_validation.py`
- Canonical local validation orchestration for agents.
- Runs pytest, Ruff, mypy, build, and NetLogo ingest smoke checks behind one structured report.
- Supports validation profiles and explicit per-check status reporting.
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

### `src/distill_abm/pipeline/doe_smoke.py`
- Pre-LLM smoke matrix inspection.
- Materializes prompt/evidence/request bundles without executing any provider call.
- Writes a design matrix CSV plus per-case artifact manifests for debugging smoke inputs before `pipeline.smoke` runs.

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
- Plot generation utilities for repeated simulation runs.
- Visualization smoke checks for the pre-LLM NetLogo plotting workflow:
  - resolve ABM-specific simulation settings from config
  - run the NetLogo model
  - write the generated simulation CSV
  - emit the ordered plot PNGs later consumed by trend-description prompts

### `configs/abms/*.yaml`
- ABM presets.
- `netlogo_viz` is the source of truth for the pre-LLM plotting workflow:
  - BehaviorSpace experiment name
  - generated reporter list
  - run-count / tick / interval settings
  - ordered plot definitions and labels

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
