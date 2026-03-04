# Testing Evidence Report

This document is a compact but complete record of verification performed for reviewer-facing confidence and reproducibility.

## Scope

The suite validates:

- Reference parity against retained implementation artifacts.
- Compatibility loader behavior and fallback safety.
- Pipeline end-to-end behavior across evidence modes, summarization modes, scoring modes, and sweep behavior.
- Deterministic ingestion and CSV/metadata generation.
- Static quality checks used in CI.

## Verification Commands

Run from repository root with `uv`:

```bash
uv run pre-commit run --all-files
uv run ruff check .
uv run black --check .
uv run mypy src tests
uv run pytest --cov=distill_abm --cov-report=term-missing --cov-fail-under=85
```

## Expected CI-equivalent Gate Set

The project CI executes the same command set:

1. `pre-commit run --all-files`
2. `ruff check .`
3. `black --check .`
4. `mypy src tests`
5. `pytest --cov=distill_abm --cov-report=term-missing --cov-fail-under=85`

## Current Baseline Result

- Total tests: 175
- Passing: 175
- Coverage: 85.29% (threshold 85%)
- Failures: 0

## Targeted Test Areas

### Core regression and migration
- `tests/regression/test_reference_equivalence.py`
- `tests/regression/test_reference_function_coverage.py`
- `tests/regression/test_runtime_reference_dependencies.py`
- `tests/regression/test_archive_retired.py`
- `tests/regression/test_archive_manifest.py`
- `tests/regression/test_model_assets_migration.py`
- `tests/regression/test_evaluation_assets_migration.py`

### Pipeline behavior and sweep parity
- `tests/integration/test_pipeline_run.py`
- `tests/integration/test_pipeline_uses_abm_and_full_metrics.py`
- `tests/unit/pipeline/test_helpers.py`
- `tests/unit/pipeline/test_sweep.py`

### Summarization and LLM text/evaluation flow
- `tests/unit/summarize/test_*`
- `tests/unit/eval/test_*`
- `tests/unit/compat/test_compat_*`

### CLI and repo quality
- `tests/e2e/test_cli.py`
- `tests/unit/repo/*`

## Reproducibility artifacts

- Each pipeline run writes `pipeline_run_metadata.json` in the output directory.
- Use this artifact in combination with `report.csv` and generated plots/images for replay and audit.

## Interpretation

Passing this set is the acceptance gate for:

- Publishing-ready releases.
- Any future structural refactors.
- Any extension work in prompts, evidence modes, or scoring policy.
