# Notebook Parity

## Goal

Provide functional continuity between legacy notebook workflows and production package modules.

## Mechanisms

1. `distill_abm.legacy.compat` preserves notebook function names with typed, testable wrappers.
2. `distill_abm.legacy.notebook_loader` builds a runtime function registry from notebooks and records provenance (`get_notebook_source_path`), preferring non-archive, non-checkpoint, non-copy sources (case-insensitive).
3. `tests/regression/test_notebook_equivalence.py` compares notebook functions directly against package implementations, including deterministic DoE helper parity and mock-based regression checks for external-call wrappers.
4. `tests/regression/test_notebook_function_coverage.py` parses all notebook function definitions from `archive/legacy_repo/Code` and asserts every function is represented in the compatibility surface or explicitly exempt (`main`).
5. `docs/archive_full_manifest.json` provides file-level classification/action mapping for every archived artifact (runtime, prompt references, human ground truth, experiment settings, or historical-only).
6. `tests/regression/test_prompt_reference_equivalence.py` locks runtime prompts against `configs/notebook_prompt_reference.yaml`.
7. `docs/archive_full_manifest.json` marks notebook files still required by `distill_abm.legacy.notebook_loader` as `runtime_required`, preventing accidental deletion before migration.
8. Archive CSV artifacts and visualization files (`.png/.jpg/.jpeg/.svg`) are retained for reproducibility; they are not marked as discardable.

## Scope

- Core ABM ingestion, text cleaning, post-processing, scoring, DoE helpers, qualitative evaluators, and plotting are implemented in production modules.
- External LLM calls are abstracted through adapters with deterministic test doubles.
- Heavy model dependencies (BART/BERT/ANOVA backends) are wrapped to support both runtime use and test-mode mocking.
- Prompt composition follows notebook ordering:
  - context: `role` (optional) + context template
  - trend: `role` (optional) + trend template + `example` (optional) + `insights` (optional) + plot description
  - ABM selection injects the first notebook plot description by default
- LLM request defaults preserve notebook-style generation settings, including `temperature=0.5`.

## Explicit limits

- Notebook-first dispatch is used only where wrappers are deterministic and safe to call in production flows.
- If notebook callables are missing or raise at runtime, compat wrappers fall back to refactored implementations; this preserves availability but can change legacy output formatting in edge cases.
- Loader execution is AST-restricted (imports, literal-like assignments, defs/classes), so notebook side effects outside that subset are not executed.
