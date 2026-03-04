# Notebook Parity

## Goal

Provide executable parity coverage across ingestion, plotting, scoring, prompt composition, and qualitative evaluation while allowing notebooks to be retired progressively without losing behavior.

## Mechanisms

1. `distill_abm.legacy.compat` preserves notebook-era function names with typed, testable wrappers.
2. `distill_abm.legacy.notebook_loader` builds a runtime function registry from notebooks and records provenance (`get_notebook_source_path`) while preferring non-archive, non-checkpoint, non-copy sources.
3. `distill_abm.legacy.compat` consults `notebook_loader.should_dispatch_notebook` before any notebook-first call, so production does not depend on notebooks when required coverage is empty.
4. `tests/regression/test_notebook_equivalence.py` compares notebook behavior against production implementations and compatibility wrappers where needed.
5. `tests/regression/test_notebook_function_coverage.py` parses all notebook function definitions from `archive/legacy_repo/Code` and validates that each supported name is represented in production or explicitly exempt.
6. `tests/unit/pipeline/test_sweep.py` validates the notebook-style prompt-combination sweep (`role`, `example`, `insights`) and multi-image trend generation parity.
7. `docs/archive_full_manifest.json` provides file-level classification/action mapping for all archived artifacts.
8. `tests/regression/test_prompt_reference_equivalence.py` locks runtime prompts against `configs/notebook_prompt_reference.yaml`.
9. `scripts/archive_audit.py` and `scripts/refresh_parity_artifacts.py` keep migration/parity artifacts consistent after notebook deletions.
10. `distill_abm.ingest.netlogo_notebook_workflow` captures the notebook-style NetLogo preprocessing flow (parameter JSON, GUI JSON, documentation/code outputs).
11. `configs/notebook_experiment_settings.yaml` preserves Fauna, Grazing, and Milk Consumption ingestion settings and parameter dictionaries for provenance and replay.
12. `tests/unit/legacy` validates loader priority, notebook dispatch behavior, fallback safety, and source resolution order.

## Current parity status

- Runtime-required notebook functions: `0` (`docs/archive_full_manifest.json`).
- Notebook-first runtime dependency map: empty (`docs/runtime_notebook_dependencies.json`).
- Legacy notebook execution is not required for core runtime flows.

The following runtime-deprecated notebooks have been replaced by production implementations and removed from runtime dependency:

- `archive/legacy_repo/Code/Models/Fauna/3. (GPT) With combinations-Copy1 copy.ipynb`
- `archive/legacy_repo/Code/Models/Grazing/3. With combinations-Copy1.ipynb`
- `archive/legacy_repo/Code/Models/Milk Consumption/3. (GPT) With combinations.ipynb`

## Scope

- ABM ingestion from NetLogo outputs.
- Text extraction and cleanup utilities.
- Plot creation and stats table generation.
- Trend/context prompt composition and adapter dispatch.
- Qualitative scoring support (`coverage`, `faithfulness`) and metric scoring.
- DoE factor/anova helpers.

## Prompts and evidence modes

- Prompt assembly in production follows notebook ordering:
  - context phase: optional `style_features.role` + context template
  - trend phase: optional `style_features.role`, trend template, optional `style_features.example`, optional `style_features.insights`, plus the plot description.
- Evidence modes supported by pipeline:
  - `plot`
  - `stats-markdown`
  - `stats-image`
  - `plot+stats`

The default request temperature remains `0.5` (`mypy`-validated default in `LLMRequest`).

## Notebook-style combination API coverage

`distill_abm.pipeline.run` now implements:

- `run_pipeline_sweep`: all prompt-feature combinations across multiple plot images.
- `write_combinations_csv`: notebook-style wide CSV output with per-image trend prompt/response columns.
- `build_style_feature_combinations`: powerset generation helper used by sweep execution.

These functions provide parity for the retired combination notebooks and keep the notebook artifacts reproducible through audit outputs.

## Explicit limits

- Notebook execution is AST-restricted in the loader, so notebook side effects outside supported patterns are intentionally not replayed.
- If notebook-discovered callables are missing or fail, fallback implementations are used for resilience.
- Visual parity artifacts (`.png/.jpg/.jpeg/.svg`) and historical outputs are retained unless explicitly reclassified by archive policy.
