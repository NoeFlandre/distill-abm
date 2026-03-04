# Reference Parity

## Goal

Keep executable parity coverage across ingestion, plotting, scoring, prompt composition, and qualitative evaluation while retiring reference scripts safely.

## Mechanisms

1. `distill_abm.compat` preserves historical function names with typed wrappers.
2. `distill_abm.compat.reference_loader` builds a runtime function registry from archived scripts and records provenance (`get_notebook_source_path`) while selecting preferred sources.
3. `distill_abm.compat.compat_callables` consults `reference_loader.should_dispatch_notebook` before any reference-first call, so runtime can operate without archived scripts when required coverage is exhausted.
4. `tests/regression/test_reference_equivalence.py` compares reference-facing behavior against production implementations and wrappers where needed.
5. `tests/regression/test_reference_function_coverage.py` validates snapshot coverage from `docs/notebook_coverage_matrix.json` against implemented production callables.
6. `tests/unit/pipeline/test_sweep.py` validates role/example/insights sweep behavior and multi-image trend generation parity.
7. `docs/archive_full_manifest.json` records file-level migration/retention actions for all archived artifacts.
8. `tests/regression/test_prompt_reference_equivalence.py` locks runtime prompts against `configs/notebook_prompt_reference.yaml`.
9. `scripts/archive_audit.py` and `scripts/refresh_parity_artifacts.py` keep migration artifacts consistent when script files are retired.
10. `distill_abm.ingest.netlogo_notebook_workflow` captures reference-style NetLogo preprocessing flow (parameter JSON, GUI JSON, documentation/code outputs).
11. `configs/notebook_experiment_settings.yaml` preserves Fauna, Grazing, and Milk Consumption ingestion defaults and parameter dictionaries.
12. `tests/unit/compat` validates loader priority, reference dispatch behavior, fallback safety, and source resolution order.
13. `tests/fixtures/notebook_parity/model_assets/Models/` mirrors ABM `CSV`, `JSON`, `TXT`, `NetLogo`, and `Images` assets for deterministic replay.
14. `tests/regression/test_model_assets_migration.py` enforces full file coverage and byte-equivalence for mirrored model assets.
15. `tests/fixtures/notebook_parity/evaluation_assets/Evaluation/` mirrors reference evaluation artifacts.
16. `tests/regression/test_evaluation_assets_migration.py` enforces full tracked-file coverage and byte-equivalence for the evaluation mirror.
17. `tests/fixtures/notebook_parity/archive_assets/reference_repo/` mirrors remaining reference `Results/` data.

## Current parity status

- Runtime-required reference functions: `0` (`docs/archive_full_manifest.json`).
- Reference-first runtime dependency map: empty (`docs/runtime_notebook_dependencies.json`).
- Reference script execution is not required for core runtime flows.
- Runtime now uses only `tests/fixtures/notebook_parity/` and production modules in `src/distill_abm/`.

The following reference scripts have been replaced by production implementations and removed from runtime dependency:

- `archive/reference_repo/Code/Models/Fauna/3. (GPT) With combinations-Copy1 copy.ipynb`
- `archive/reference_repo/Code/Models/Fauna/3. (Deepseek) With combinations.ipynb`
- `archive/reference_repo/Code/Models/Fauna/3bis. (Claude) With combinations-Copy1.ipynb`
- `archive/reference_repo/Code/Models/Grazing/3. With combinations-Copy1.ipynb`
- `archive/reference_repo/Code/Models/Grazing/3. (Deepseek) With combinations.ipynb`
- `archive/reference_repo/Code/Models/Grazing/3bis. (Claude) With combinations.ipynb`
- `archive/reference_repo/Code/Models/Milk Consumption/3. (GPT) With combinations.ipynb`
- `archive/reference_repo/Code/Models/Milk Consumption/3. (Deepseek) With combinations.ipynb`
- `archive/reference_repo/Code/Models/Milk Consumption/3bis. (Claude) With combinations.ipynb`

## Scope

- ABM ingestion from NetLogo outputs.
- Text extraction and cleanup utilities.
- Plot creation and stats table generation.
- Trend/context prompt composition and adapter dispatch.
- Qualitative scoring support (`coverage`, `faithfulness`) and metric scoring.
- DoE factorial and ANOVA helpers.

## Prompt assembly and evidence modes

- Prompt assembly follows the same stage ordering used by the reference scripts:
  - context phase: optional `style_features.role` + context template
  - trend phase: optional `style_features.role`, trend template, optional `style_features.example`, optional `style_features.insights`, plus plot description
- Evidence modes supported by pipeline:
  - `plot`
  - `stats-markdown`
  - `stats-image`
  - `plot+stats`

Default request temperature is `0.5` (validated in `LLMRequest`).

## Combination API coverage

`distill_abm.pipeline.run` now implements:

- `run_pipeline_sweep` for every role/example/insights combination across multiple plots.
- `write_combinations_csv` for wide per-combination output with one trend prompt/response pair per image.
- `build_style_feature_combinations` to materialize the combination lattice.
- Context/trend split-provider support for multi-provider parity:
  - `context_adapter`, `trend_adapter`
  - `context_model`, `trend_model`
  - `csv_column_style="plot"`
  - `resume_existing=True`

## Explicit limits

- Reference execution is AST-restricted, so side effects outside supported patterns are not replayed.
- If reference-discovered callables are missing or fail, fallback implementations are used for resilience.
- Visual parity artifacts (`.png/.jpg/.jpeg/.svg`) and historical outputs are retained unless explicitly reclassified.
