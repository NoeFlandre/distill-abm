# Architecture

## Core packages

- `distill_abm.ingest`:
  - CSV ingestion and NetLogo preprocessing orchestration
  - `netlogo.py`: extraction helpers (`extract_code`, `extract_parameters`, `extract_documentation`)
  - `netlogo_notebook_workflow.py`: deterministic NetLogo run loop and artifact generation
  - `netlogo_notebook_artifacts.py`: reusable artifact builders (parameters, docs, code, narratives)
  - `netlogo_notebook_steps.py`: reusable NetLogo execution primitives (parameter coercion, single-run capture)
- `distill_abm.viz`:
  - metric plotting for repeated simulations
  - stats table computation (`generate_stats_table`)
  - optional mean overlay rendering
  - stats table serialization for text evidence (`table-csv`)
- `distill_abm.llm`:
  - adapter interface + OpenAI, Anthropic, Ollama, Janus implementations
  - unified request schema (`LLMRequest`) and provider factory
- `distill_abm.summarize`:
  - prompt-safe text cleanup
  - summarization runners for BART, BERT, T5, and Longformer-like models
  - default summary stack: BART + BERT
  - optional add-on stack via CLI: `--additional-summarizer t5|longformer_ext` (repeatable)
  - report post-processing utilities (`clean_markdown_symbols`, etc.)
- `distill_abm.eval`:
  - token overlap metrics
  - BLEU/METEOR/ROUGE/Flesch scoring
  - qualitative score extraction (`extract_coverage_score`, `extract_faithfulness_score`)
  - DoE ANOVA contribution analysis
- `distill_abm.pipeline`:
  - end-to-end orchestration from CSV -> plot -> LLM -> (full text and/or summarized text) -> scoring
  - dual text handling via CLI:
    - `--summarization-mode` (`full`, `summary`, `both`) [default: `both`]
    - `--score-on` (`full`, `summary`, `both`) [default: `both`]
    - `--skip-summarization` maps to `summarization_mode=full`
    - `--additional-summarizer` appends optional summarizers (`t5`, `longformer_ext`) to the default BART+BERT summary
  - evidence ablation modes via CLI `--evidence-mode`: `plot`, `table-csv`, `plot+table`
  - `table-csv` is text-only evidence (no vision image attached)
  - reference-style multi-feature sweep API:
    - `run_pipeline_sweep` for role/example/insights combinations
    - `write_combinations_csv` for wide per-combination output schema
    - optional split context/trend adapters and resumable wide-CSV updates for Claude/Deepseek parity workflows
  - smoke orchestration:
    - `pipeline.smoke.run_qwen_smoke_suite` for all evidence/summarization/scoring mode checks in one run
    - case-by-case artifact capture (`prompts`, `responses`, `metadata`, qualitative checks, DoE, sweep)
- `distill_abm.compat`:
  - reference implementation loader with source provenance and priority ordering
  - compatibility wrappers preserving stable function names
  - reference-first dispatch for selected deterministic helpers with fallback to refactored code paths
- `distill_abm.configs`:
  - validated, typed models for prompts, ABMs, experiment settings, and CLI defaults

## Hyperparameter reference

All numeric defaults and runtime hyperparameters are documented in:

- `docs/HYPERPARAMETERS.md`

## Configs

- `configs/models.yaml`
- `configs/prompts.yaml`
- `configs/abms/*.yaml`
- `configs/evaluation.yaml`
- `configs/logging.yaml`

## Testing strategy

- Unit tests for deterministic utilities and contracts
- Integration tests for pipeline flows
- E2E tests for CLI behavior
- Regression tests for reference parity and function-coverage accounting
- Compatibility loader/dispatch tests to verify source priority, provenance, and fallback robustness

## Runtime Data Flow

1. `run_pipeline` / `run_pipeline_sweep` loads simulation CSV artifacts.
2. `viz` module renders trend plots and stats tables.
3. Prompt text is assembled from `PromptsConfig` plus optional style features.
4. `llm` adapters submit image/context requests.
5. Qualitative and lexical scorers evaluate outputs.
6. `distill_abm.compat` and `distill_abm.compat.reference_loader` provide safe fallbacks for parity-critical helpers.

## Deployment and reproducibility

- `uv sync --extra dev` installs the project and development tooling.
- `docker build -t distill-abm .` creates a reproducible runtime image.
- `uv run distill-abm run ...` executes deterministic CLI workflows with explicit modes and paths.
- `uv run distill-abm evaluate-qualitative ...` runs qualitative coverage/faithfulness scoring.
- `uv run distill-abm analyze-doe ...` runs ANOVA/DoE analysis.
- `uv run distill-abm smoke-qwen ...` runs the full local Qwen smoke matrix with debug artifacts.
- Each run writes `pipeline_run_metadata.json` and captures:
  - input artifact paths
  - resolved evidence/summarization modes
  - full prompt texts and prompt signatures (`context` and `trend`)
  - model/provider and request hyperparameters
  - trend/tracker score outputs and generated artifact paths
