# Architecture

## Core packages

- `distill_abm.ingest`:
  - CSV ingestion
  - NetLogo code/documentation extraction and cleaning
- `distill_abm.viz`:
  - metric plotting for repeated simulations
- `distill_abm.llm`:
  - adapter interface + OpenAI, Anthropic, Ollama, Janus implementations
- `distill_abm.summarize`:
  - notebook-compatible text cleanup
  - BART/BERT summarization runners
- `distill_abm.eval`:
  - token overlap metrics
  - BLEU/METEOR/ROUGE/Flesch legacy scoring
  - qualitative score parsing
  - DoE ANOVA contribution analysis
- `distill_abm.pipeline`:
  - end-to-end orchestration from CSV -> plot -> LLM -> (optional BART/BERT summarization) -> scoring
  - no-summarization baseline mode via CLI `--skip-summarization`
  - evidence ablation modes via CLI `--evidence-mode`: `plot`, `stats-markdown`, `stats-image`, `plot+stats`
- `distill_abm.legacy`:
  - notebook function loader with source provenance and priority ordering
  - compatibility wrappers preserving notebook-era function names
  - notebook-first dispatch for selected deterministic helpers with fallback to refactored code paths

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
- Regression tests for notebook parity and function-coverage accounting
- Legacy loader/dispatch tests to verify source priority, provenance, notebook-first behavior, and fallback robustness
